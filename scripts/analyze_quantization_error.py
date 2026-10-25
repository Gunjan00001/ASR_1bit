"""Weight-level screening: per-tensor vs per-output-channel quantization error.

This is the cheap pre-training gate for the per-channel FFN experiment. It does
**not** train and does **not** claim a WER improvement. It answers one question:

    For the FFN weights, how much reconstruction error does one scale per row
    remove compared with one scale per matrix?

If per-channel does not materially reduce the error, a training run is unlikely
to help and the next experiment should be the layer-sensitivity screen instead.

Usage:
    python scripts/analyze_quantization_error.py \
        --checkpoint checkpoints/qat_ffn_a100 \
        --output results/quantization_error_ffn.json

    # no checkpoint (e.g. on Kaggle): use the FP32 HF weights
    python scripts/analyze_quantization_error.py \
        --output results/quantization_error_ffn.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
FFN_PATTERNS = {
    "ffn_in": re.compile(r"ffn[12]\.intermediate_dense\.weight$"),
    "ffn_out": re.compile(r"ffn[12]\.output_dense\.weight$"),
}
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def load_state_dict(checkpoint: str | None, model_id: str) -> tuple[dict, str]:
    """Load a saved QAT checkpoint or the FP32 HF model, on CPU."""
    if checkpoint:
        ckpt = Path(checkpoint)
        if ckpt.is_dir():
            for name in ("model.safetensors", "pytorch_model.bin"):
                path = ckpt / name
                if path.is_file():
                    if path.suffix == ".safetensors":
                        return load_file(str(path), device="cpu"), str(ckpt)
                    return torch.load(path, map_location="cpu", weights_only=True), str(ckpt)
            raise SystemExit(f"no model.safetensors/pytorch_model.bin in {ckpt}")
        if ckpt.suffix == ".safetensors":
            return load_file(str(ckpt), device="cpu"), str(ckpt)
        return torch.load(ckpt, map_location="cpu", weights_only=True), str(ckpt)

    from transformers import Wav2Vec2ConformerForCTC

    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
    return {k: v.detach() for k, v in model.state_dict().items()}, f"hf:{model_id}"


def analyze_weight(weight: torch.Tensor) -> dict:
    """Per-tensor vs per-output-channel reconstruction stats for one matrix."""
    w = weight.detach().to(torch.float32)
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(w.shape)}")
    norm = float(w.norm())
    if norm == 0:
        return {
            "err_per_tensor": 0.0,
            "err_per_channel": 0.0,
            "relative_improvement": 0.0,
            "row_cv": 0.0,
            "frac_rows_gt_2x_mean": 0.0,
            "frac_rows_lt_half_mean": 0.0,
            "zero_rows": int(w.shape[0]),
        }

    alpha_t = w.abs().mean()
    wq_t = alpha_t * torch.sign(w)
    err_t = float((w - wq_t).norm() / norm)

    alpha_c = w.abs().mean(dim=1, keepdim=True)
    wq_c = alpha_c * torch.sign(w)
    err_c = float((w - wq_c).norm() / norm)
    improvement = (err_t - err_c) / err_t if err_t > 0 else 0.0

    row = w.abs().mean(dim=1)
    row_mean = float(row.mean())
    row_cv = float(row.std(unbiased=False) / row_mean) if row_mean > 0 else 0.0
    return {
        "err_per_tensor": err_t,
        "err_per_channel": err_c,
        "relative_improvement": improvement,
        "row_cv": row_cv,
        "frac_rows_gt_2x_mean": float((row > 2 * row_mean).float().mean()),
        "frac_rows_lt_half_mean": float((row < 0.5 * row_mean).float().mean()),
        "zero_rows": int((row == 0).sum()),
    }


def summarize(entries: list[dict]) -> dict:
    if not entries:
        return {}
    n = len(entries)
    return {
        "n_layers": n,
        "mean_err_per_tensor": sum(e["err_per_tensor"] for e in entries) / n,
        "mean_err_per_channel": sum(e["err_per_channel"] for e in entries) / n,
        "mean_relative_improvement": sum(e["relative_improvement"] for e in entries) / n,
        "median_relative_improvement": sorted(
            e["relative_improvement"] for e in entries
        )[n // 2],
        "mean_row_cv": sum(e["row_cv"] for e in entries) / n,
        "max_row_cv": max(e["row_cv"] for e in entries),
        "mean_frac_rows_gt_2x_mean": sum(e["frac_rows_gt_2x_mean"] for e in entries) / n,
        "mean_frac_rows_lt_half_mean": sum(e["frac_rows_lt_half_mean"] for e in entries) / n,
        "zero_rows_total": sum(e["zero_rows"] for e in entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FFN quantization-error screen.")
    parser.add_argument("--checkpoint", default=None,
                        help="QAT checkpoint dir/file; omit to use the FP32 HF model")
    parser.add_argument("--model", default="facebook/wav2vec2-conformer-rope-large-960h-ft")
    parser.add_argument("--output", default=str(ROOT / "results" / "quantization_error_ffn.json"))
    parser.add_argument("--gate", type=float, default=0.10,
                        help="minimum mean relative improvement to proceed (default 0.10)")
    args = parser.parse_args()

    state, source = load_state_dict(args.checkpoint, args.model)
    print(f"Loaded {source}: {len(state)} tensors")

    layers: list[dict] = []
    by_group: dict[str, list[dict]] = {"ffn_in": [], "ffn_out": []}
    for name, tensor in state.items():
        group = next((g for g, pat in FFN_PATTERNS.items() if pat.search(name)), None)
        if group is None:
            continue
        stats = analyze_weight(tensor)
        m = LAYER_RE.search(name)
        entry = {
            "name": name,
            "group": group,
            "layer": int(m.group(1)) if m else None,
            "shape": list(tensor.shape),
            **stats,
        }
        layers.append(entry)
        by_group[group].append(stats)

    if not layers:
        raise SystemExit(
            "no FFN weights found; check the checkpoint key names. "
            f"First keys: {list(state)[:10]}"
        )

    summary = {"all": summarize([e for e in layers])}
    summary.update({g: summarize(entries) for g, entries in by_group.items() if entries})
    gate_passes = summary["all"]["mean_relative_improvement"] >= args.gate

    payload = {
        "purpose": "pre-training gate for per-output-channel FFN scaling",
        "source": source,
        "model_id": args.model,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "method": {
            "per_tensor": "alpha = mean(|W|); Wq = alpha*sign(W)",
            "per_channel": "alpha_i = mean_j |W[i,j]|; Wq[i,j] = alpha_i*sign(W[i,j])",
            "metric": "relative Frobenius reconstruction error ||W-Wq||_F/||W||_F",
        },
        "gate": args.gate,
        "gate_passes": bool(gate_passes),
        "summary": summary,
        "layers": layers,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\nFFN layers analyzed: {len(layers)}")
    for group, stats in summary.items():
        print(
            f"  {group:<8} n={stats['n_layers']:>3}  "
            f"err tensor={stats['mean_err_per_tensor']:.4f}  "
            f"channel={stats['mean_err_per_channel']:.4f}  "
            f"improvement={stats['mean_relative_improvement']:.2%}  "
            f"row_cv={stats['mean_row_cv']:.3f}"
        )
    print(f"\nGate (mean improvement >= {args.gate:.0%}): "
          f"{'PASS' if gate_passes else 'FAIL'}")
    print(f"Saved {out}")
    return 0 if gate_passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
