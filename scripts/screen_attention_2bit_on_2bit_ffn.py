"""Screen isolated 2-bit attention perturbations on the 2-bit FFN checkpoint.

The base is the selected 2-bit FFN full checkpoint (step 1500): all 79 FFN
BitLinear layers are 2-bit, the 17 FFN exclusions and all attention layers are
FP32. For each attention projection, exactly that layer is replaced by a 2-bit
BitLinear, evaluated on the fixed dev subset, then restored. No retraining.

This is a screening experiment only; it ranks layers by measured dev WER
degradation, not a QAT guarantee. Test-clean is never loaded.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
from pathlib import Path

import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.data.text import normalize_pair
from onebit_asr.evaluation.metrics import compute_cer, compute_wer
from onebit_asr.evaluation.sensitivity import weight_reconstruction_error
from onebit_asr.models.conformer_utils import get_parent_module, select_by_category
from onebit_asr.quantization.bitlinear import BitLinear
from onebit_asr.training.qat import load_qat_checkpoint

ROOT = Path(__file__).resolve().parent.parent
LAYER_RE = re.compile(r"layers\.(\d+)\.")
ATTENTION = {
    "attention_q": True, "attention_k": True, "attention_v": True,
    "attention_output": True, "ffn_in": False, "ffn_out": False,
    "feature_projection": False, "ctc_head": False,
}


@torch.no_grad()
def forward_all(model, inputs):
    return [model(x).logits.detach().cpu() for x in inputs]


def compare_logits(references, variants):
    rel, kls, agrs = [], [], []
    for ref, var in zip(references, variants):
        denom = float(ref.norm())
        rel.append(float((ref - var).norm()) / denom if denom else 0.0)
        lp_ref = torch.log_softmax(ref, -1)
        lp_var = torch.log_softmax(var, -1)
        kls.append(float((lp_ref.exp() * (lp_ref - lp_var)).sum(-1).mean()))
        agrs.append(float((ref.argmax(-1) == var.argmax(-1)).float().mean()))
    n = len(references)
    return {"relative_l2": sum(rel) / n, "kl": sum(kls) / n,
            "argmax": sum(agrs) / n}


def wer_cer(processor, samples, logits_list):
    refs, hyps = [], []
    for s, logits in zip(samples, logits_list):
        hyp = processor.batch_decode(torch.argmax(logits, -1))[0]
        ref_n, hyp_n = normalize_pair(s["text"], hyp)
        refs.append(ref_n)
        hyps.append(hyp_n)
    return compute_wer(refs, hyps), compute_cer(refs, hyps)


def main() -> int:
    ap = argparse.ArgumentParser(description="2-bit attention screen on 2-bit FFN.")
    ap.add_argument("--base-config", default="configs/qat_ffn_mixed_2bit.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/qat_ffn_mixed_2bit_full")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="results/layer_sensitivity_attention_2bit_on_2bit_ffn.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.base_config)
    q = cfg.get("quantization", {})
    scale_mode = q.get("scale", "per_tensor_mean_abs")
    ste_clip = q.get("ste_clip")
    bits = int(q.get("bits", 2))
    if bits != 2:
        raise SystemExit(f"this screen requires the 2-bit base config, got bits={bits}")
    excl = list(q.get("exclude_layers") or [])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | base: {args.checkpoint} | bits={bits} | n={args.n}")

    processor = Wav2Vec2Processor.from_pretrained(cfg["qat"]["init_from"])
    base, _ = load_qat_checkpoint(
        ROOT / args.checkpoint, Wav2Vec2ConformerForCTC, cfg["layers"],
        scale_mode=scale_mode, ste_clip=ste_clip, exclude_layers=excl, bits=bits,
    )
    base.to(device).eval()
    fp32 = Wav2Vec2ConformerForCTC.from_pretrained(cfg["qat"]["init_from"]).to(device).eval()

    samples = load_dev_subset(subset_size=cfg["eval"]["subset_size"],
                              seed=cfg["eval"]["seed"])[: args.n]
    inputs = [processor(s["audio"], sampling_rate=16_000, return_tensors="pt",
                        padding=True).input_values.to(device) for s in samples]
    print(f"Fixed dev utterances: {len(samples)}")

    base_logits = forward_all(base, inputs)
    base_wer, base_cer = wer_cer(processor, samples, base_logits)
    fp32_logits = forward_all(fp32, inputs)
    fp32_wer, fp32_cer = wer_cer(processor, samples, fp32_logits)
    print(f"2-bit FFN base: WER={base_wer:.4f} CER={base_cer:.4f}; "
          f"FP32: WER={fp32_wer:.4f} CER={fp32_cer:.4f}")

    targets = select_by_category(base, ATTENTION)
    print(f"Attention layers to screen: {len(targets)}")
    rows = []
    for i, (name, module, category) in enumerate(targets, 1):
        parent, attr = get_parent_module(base, name)
        original = getattr(parent, attr)
        quantized = BitLinear.from_linear(
            original, scale_mode=scale_mode, ste_clip=ste_clip, bits=2
        )
        setattr(parent, attr, quantized)
        try:
            variant_logits = forward_all(base, inputs)
            logit = compare_logits(base_logits, variant_logits)
            w, c = wer_cer(processor, samples, variant_logits)
        finally:
            setattr(parent, attr, original)
        m = LAYER_RE.search(name)
        row = {
            "name": name, "category": category,
            "layer": int(m.group(1)) if m else None,
            "in_features": original.in_features, "out_features": original.out_features,
            "params": sum(p.numel() for p in original.parameters()),
            "weight_reconstruction_error": weight_reconstruction_error(
                original.weight, scale_mode=scale_mode, ste_clip=ste_clip, bits=2),
            "relative_l2": logit["relative_l2"], "kl_ref_variant": logit["kl"],
            "argmax_agreement": logit["argmax"],
            "wer": w, "cer": c,
            "delta_wer_vs_base": w - base_wer,
            "delta_wer_vs_fp32": w - fp32_wer,
        }
        rows.append(row)
        print(f"  [{i:>3}/{len(targets)}] {name}: recon={row['weight_reconstruction_error']:.4f} "
              f"relL2={row['relative_l2']:.4f} WER={w:.4f} "
              f"(d_base={row['delta_wer_vs_base']:+.4f})", flush=True)

    ranking = sorted(rows, key=lambda r: r["delta_wer_vs_base"], reverse=True)
    groups = {}
    for r in rows:
        groups.setdefault(r["category"], []).append(r)
    group_summary = {}
    for g, rs in groups.items():
        ds = [r["delta_wer_vs_base"] for r in rs]
        group_summary[g] = {
            "n_layers": len(rs), "mean_delta_wer_vs_base": sum(ds) / len(ds),
            "max_delta_wer_vs_base": max(ds),
            "mean_relative_l2": sum(r["relative_l2"] for r in rs) / len(rs),
            "mean_weight_reconstruction_error": sum(
                r["weight_reconstruction_error"] for r in rs) / len(rs),
        }
    result = {
        "phase": "2-bit attention sensitivity on 2-bit FFN",
        "method": ("one-layer 2-bit attention PTQ screen on the selected 2-bit FFN "
                   "full checkpoint; all FFN 2-bit, 17 fixed FP32 exclusions, "
                   "attention base FP32, no retraining"),
        "screening_only": True, "not_qat": True, "preliminary": True,
        "small_subset_warning": "Small dev subset; ranking only, never test selection.",
        "test_set_used": False,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "base_config": str(ROOT / args.base_config),
        "base_checkpoint": str(ROOT / args.checkpoint),
        "model_id": cfg["qat"]["init_from"],
        "bits": bits, "scale_mode": scale_mode, "ste_clip": ste_clip,
        "exclude_layers": excl, "n_excluded": len(excl),
        "split": "validation", "seed": cfg["eval"]["seed"],
        "n_utterances": len(samples), "sample_ids": [s["id"] for s in samples],
        "base_reference": {"wer": base_wer, "cer": base_cer},
        "fp32_reference": {"wer": fp32_wer, "cer": fp32_cer},
        "layers": rows,
        "ranking_by_delta_wer_vs_base": [r["name"] for r in ranking],
        "top_10": ranking[:10],
        "group_summary": group_summary,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
