"""Per-layer sensitivity screening (no retraining).

What this measures
------------------
For each selected linear layer, the FP32 model is left unchanged **except** for
that one layer, which is replaced by the current 1-bit quantizer. We then compare
the variant against the FP32 reference on a fixed dev subset:

- relative L2 logit change,
- mean per-frame KL(reference || variant),
- frame-level argmax agreement,
- greedy CTC WER/CER and WER delta.

This is a **screening** experiment, not QAT. It ranks layers by how much damage
their isolated quantization causes; it does not claim that keeping a layer FP
will improve QAT WER. The top candidates must be validated with an actual
mixed-precision QAT run.

Usage:
    python scripts/screen_layer_sensitivity.py \
        --config configs/qat_ffn.yaml \
        --n 32 \
        --output results/layer_sensitivity_ffn.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.data.text import normalize_pair
from onebit_asr.evaluation.metrics import compute_cer, compute_wer
from onebit_asr.models.conformer_utils import (
    get_parent_module,
    select_by_category,
)
from onebit_asr.quantization.bitlinear import BitLinear

ROOT = Path(__file__).resolve().parent.parent
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def decode_all(processor, logits_list: list[torch.Tensor]) -> list[str]:
    return [
        processor.batch_decode(torch.argmax(logits, dim=-1))[0]
        for logits in logits_list
    ]


@torch.no_grad()
def forward_all(model, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [model(x).logits.detach().cpu() for x in inputs]


def compare_logits(
    references: list[torch.Tensor], variants: list[torch.Tensor]
) -> dict:
    rel_l2, kls, agreements = [], [], []
    for ref, var in zip(references, variants):
        if ref.shape != var.shape:
            raise ValueError(f"shape mismatch {tuple(ref.shape)} vs {tuple(var.shape)}")
        denom = float(ref.norm())
        rel_l2.append(float((ref - var).norm()) / denom if denom > 0 else 0.0)
        logp_ref = F.log_softmax(ref, dim=-1)
        logp_var = F.log_softmax(var, dim=-1)
        p_ref = logp_ref.exp()
        kls.append(float((p_ref * (logp_ref - logp_var)).sum(-1).mean()))
        agreements.append(
            float((ref.argmax(-1) == var.argmax(-1)).float().mean())
        )
    n = len(references)
    return {
        "mean_relative_l2": sum(rel_l2) / n,
        "mean_kl_ref_variant": sum(kls) / n,
        "mean_argmax_agreement": sum(agreements) / n,
    }


def wer_cer(processor, samples, logits_list) -> tuple[float, float, list[str]]:
    refs, hyps = [], []
    for sample, logits in zip(samples, logits_list):
        hyp = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
        ref_n, hyp_n = normalize_pair(sample["text"], hyp)
        refs.append(ref_n)
        hyps.append(hyp_n)
    return compute_wer(refs, hyps), compute_cer(refs, hyps), hyps


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-layer 1-bit sensitivity screen.")
    parser.add_argument("--config", default="configs/qat_ffn.yaml")
    parser.add_argument("--n", type=int, default=32, help="dev utterances")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="screen only the first N layers (smoke use)")
    parser.add_argument("--scale-mode", default=None,
                        help="override the config scale mode")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    quant = cfg.get("quantization", {})
    scale_mode = args.scale_mode or quant.get("scale", "per_tensor_mean_abs")
    ste_clip = quant.get("ste_clip")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_id = cfg["qat"]["init_from"]
    print(f"Device: {device} | config: {args.config} | scale: {scale_mode}")

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id).to(device)
    model.eval()

    samples = load_dev_subset(
        subset_size=cfg["eval"]["subset_size"], seed=cfg["eval"]["seed"]
    )[: args.n]
    inputs = [
        processor(s["audio"], sampling_rate=16_000, return_tensors="pt", padding=True)
        .input_values.to(device)
        for s in samples
    ]
    print(f"Screening {len(samples)} dev utterances; reference FP32 forward ...")

    ref_logits = forward_all(model, inputs)
    ref_wer, ref_cer, _ref_hyps = wer_cer(processor, samples, ref_logits)

    # select_by_category is the same taxonomy used by replacement. It returns
    # the pristine FP32 layers because the model has not been modified yet.
    targets = select_by_category(model, cfg["layers"])
    if args.max_layers is not None:
        targets = targets[: args.max_layers]
    print(f"Selected layers to screen: {len(targets)}")

    rows = []
    for i, (name, module, category) in enumerate(targets, start=1):
        parent, attr = get_parent_module(model, name)
        original = getattr(parent, attr)
        quantized = BitLinear.from_linear(
            original, scale_mode=scale_mode, ste_clip=ste_clip
        )
        setattr(parent, attr, quantized)
        try:
            variant_logits = forward_all(model, inputs)
            logit_metrics = compare_logits(ref_logits, variant_logits)
            var_wer, var_cer, _hyp = wer_cer(processor, samples, variant_logits)
        finally:
            # Always restore the FP32 layer, even if a forward/decode fails.
            setattr(parent, attr, original)

        m = LAYER_RE.search(name)
        # e.g. ...layers.0.ffn1.intermediate_dense -> ffn1
        ffn_block = name.split(".")[-2]
        group = "ffn_in" if "intermediate_dense" in name else (
            "ffn_out" if "output_dense" in name else category
        )
        row = {
            "name": name,
            "category": category,
            "group": group,
            "ffn_block": ffn_block,
            "layer": int(m.group(1)) if m else None,
            "in_features": original.in_features,
            "out_features": original.out_features,
            "params": sum(p.numel() for p in original.parameters()),
            "relative_l2": logit_metrics["mean_relative_l2"],
            "kl_ref_variant": logit_metrics["mean_kl_ref_variant"],
            "argmax_agreement": logit_metrics["mean_argmax_agreement"],
            "wer": var_wer,
            "cer": var_cer,
            "delta_wer_vs_fp32": var_wer - ref_wer,
        }
        rows.append(row)
        print(
            f"  [{i:>3}/{len(targets)}] {name}: "
            f"relL2={row['relative_l2']:.4f} KL={row['kl_ref_variant']:.4f} "
            f"argmax={row['argmax_agreement']:.3f} "
            f"WER={row['wer']:.4f} (d={row['delta_wer_vs_fp32']:+.4f})",
            flush=True,
        )

    ranking = sorted(rows, key=lambda r: r["delta_wer_vs_fp32"], reverse=True)
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    group_summary = {}
    for group, group_rows in by_group.items():
        deltas = [r["delta_wer_vs_fp32"] for r in group_rows]
        group_summary[group] = {
            "n_layers": len(group_rows),
            "mean_delta_wer": sum(deltas) / len(deltas),
            "max_delta_wer": max(deltas),
            "mean_relative_l2": sum(r["relative_l2"] for r in group_rows) / len(group_rows),
        }

    result = {
        "phase": "Layer sensitivity screen",
        "method": (
            "one-layer PTQ screen: replace exactly one selected FP32 Linear with "
            "BitLinear, leave the rest FP32, no retraining"
        ),
        "screening_only": True,
        "not_qat": True,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "config_path": cfg.get("_config_path"),
        "model_id": model_id,
        "scale_mode": scale_mode,
        "ste_clip": ste_clip,
        "split": "validation",
        "seed": cfg["eval"]["seed"],
        "n_utterances": len(samples),
        "sample_ids": [s["id"] for s in samples],
        "fp32_reference": {"wer": ref_wer, "cer": ref_cer},
        "layers": rows,
        "ranking_by_delta_wer": [r["name"] for r in ranking],
        "top_10": ranking[:10],
        "group_summary": group_summary,
    }
    out = Path(args.output) if args.output else (
        ROOT / "results" / f"layer_sensitivity_{Path(args.config).stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nFP32 reference WER on {len(samples)} utts: {ref_wer:.4f}")
    print("Most sensitive layers (delta WER):")
    for row in ranking[:10]:
        print(f"  {row['name']}: {row['delta_wer_vs_fp32']:+.4f} "
              f"(WER {row['wer']:.4f}, relL2 {row['relative_l2']:.4f})")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
