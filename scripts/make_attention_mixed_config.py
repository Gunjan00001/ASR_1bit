"""Generate a mixed-attention config from a measured sensitivity screen.

Two policy shapes are supported, deliberately limited to two:

- ``category``: quantize one attention category (the least sensitive category in
  the screen); keep the other attention categories and all 17 FFN exclusions
  FP32.
- ``sensitive``: quantize all attention categories except the top-``k`` most
  sensitive attention layers; keep the 17 FFN exclusions FP32.

The script reads the measured screen JSON, never the test set. It copies the
existing 17-layer FFN exclusion list from ``configs/qat_ffn_mixed.yaml`` without
changing it, and writes the selected/excluded layers into a new config plus a
metadata JSON. It computes only an *arithmetic size estimate*; it does not
predict WER. Selection from a 16/32-utterance screen is preliminary.

Usage:
    python scripts/make_attention_mixed_config.py \
        --screen results/layer_sensitivity_attention_v1_cpu16.json \
        --policy sensitive --keep-count 24 --id attention_v1_cpu16
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from onebit_asr.config import load_config

ROOT = Path(__file__).resolve().parent.parent
ATTENTION_CATEGORIES = ("attention_q", "attention_k", "attention_v", "attention_output")
METRIC_FIELDS = {
    "delta_wer": "delta_wer_vs_fp32",
    "relative_l2": "relative_l2",
    "reconstruction": "weight_reconstruction_error",
}
GROUP_FIELDS = {
    "delta_wer": "mean_delta_wer",
    "relative_l2": "mean_relative_l2",
    "reconstruction": "mean_weight_reconstruction_error",
}
BASE_ARTIFACT_BYTES_DEFAULT = 1_089_643_604  # measured mixed FFN full artifact


def _category_rows(screen: dict) -> list[dict]:
    return [r for r in screen["layers"] if r.get("category") in ATTENTION_CATEGORIES]


def select_category(screen: dict, metric: str) -> tuple[str, dict]:
    """Return the least-sensitive attention category and its group stats."""
    field = GROUP_FIELDS[metric]
    groups = {
        g: stats for g, stats in screen.get("group_summary", {}).items()
        if g in ATTENTION_CATEGORIES
    }
    if not groups:
        raise ValueError("screen has no attention category summaries")
    chosen = min(groups, key=lambda g: groups[g][field])
    return chosen, groups[chosen]


def select_sensitive_layers(screen: dict, metric: str, keep_count: int) -> list[str]:
    """Return the top-``keep_count`` attention layer names (kept FP32)."""
    rows = _category_rows(screen)
    if not rows:
        raise ValueError("screen has no attention layer rows")
    if not 0 <= keep_count <= len(rows):
        raise ValueError(f"keep_count must be 0..{len(rows)}, got {keep_count}")
    field = METRIC_FIELDS[metric]
    ranked = sorted(rows, key=lambda r: r[field], reverse=True)
    return [r["name"] for r in ranked[:keep_count]]


def estimate_artifact_bytes(
    base_artifact_bytes: int,
    quantized_weight_count: int,
    quantized_layer_count: int,
) -> dict:
    """Arithmetic estimate: replace FP32 weights with packed bits + scales.

    Biases and all non-attention FP parameters are unchanged. This is a size
    calculation, not a WER prediction.
    """
    fp32_bytes = int(quantized_weight_count) * 4
    packed_bytes = int(math.ceil(quantized_weight_count / 8))
    scale_bytes = int(quantized_layer_count) * 4  # one FP32 per-tensor scale/layer
    saved = fp32_bytes - packed_bytes - scale_bytes
    estimated = int(base_artifact_bytes) - saved
    return {
        "base_artifact_bytes": int(base_artifact_bytes),
        "quantized_attention_weight_count": int(quantized_weight_count),
        "quantized_attention_layer_count": int(quantized_layer_count),
        "replaced_fp32_bytes": fp32_bytes,
        "packed_payload_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "saved_bytes": int(saved),
        "estimated_artifact_bytes": int(estimated),
        "estimated_artifact_gb": estimated / 1e9,
        "note": "Arithmetic size estimate only; not a WER prediction.",
    }


def build_policy(
    screen: dict,
    base_layers: dict,
    base_excluded: list[str],
    base_quantization: dict,
    *,
    policy: str,
    metric: str = "relative_l2",
    category: str | None = None,
    keep_count: int = 24,
    base_artifact_bytes: int = BASE_ARTIFACT_BYTES_DEFAULT,
) -> tuple[dict, dict]:
    """Build the config payload and selection metadata for one policy."""
    if metric not in METRIC_FIELDS:
        raise ValueError(f"unknown metric {metric!r}")
    if policy not in ("category", "sensitive"):
        raise ValueError(f"unknown policy {policy!r}")

    layers = dict(base_layers)
    excluded = list(base_excluded)

    if policy == "category":
        if category is None:
            category, group_stats = select_category(screen, metric)
        else:
            if category not in ATTENTION_CATEGORIES:
                raise ValueError(f"unknown attention category {category!r}")
            group_stats = screen.get("group_summary", {}).get(category, {})
        for cat in ATTENTION_CATEGORIES:
            layers[cat] = cat == category
        attention_kept: list[str] = []
        quantized_layers = [
            r for r in _category_rows(screen) if r.get("category") == category
        ]
        selection = {
            "policy": "category",
            "selected_category": category,
            "selection_metric": metric,
            "group_stats": group_stats,
        }
    else:
        for cat in ATTENTION_CATEGORIES:
            layers[cat] = True
        attention_kept = select_sensitive_layers(screen, metric, keep_count)
        excluded.extend(attention_kept)
        kept_set = set(attention_kept)
        quantized_layers = [r for r in _category_rows(screen) if r["name"] not in kept_set]
        selection = {
            "policy": "sensitive",
            "keep_count": int(keep_count),
            "selection_metric": metric,
            "kept_attention_layers": attention_kept,
        }

    # Preserve the base 17 exactly; add attention exclusions only.
    excluded = list(dict.fromkeys(excluded))
    if any(name not in excluded for name in base_excluded):
        raise AssertionError("base FFN exclusions were lost")

    quantized_weight_count = sum(
        int(r["in_features"]) * int(r["out_features"]) for r in quantized_layers
    )
    size = estimate_artifact_bytes(
        base_artifact_bytes, quantized_weight_count, len(quantized_layers)
    )

    config = {
        "extends": "qat_ffn_mixed.yaml",  # overridden by main for non-configs paths
        "compute_profile": "cloud-gpu",
        "layers": layers,
        "quantization": {
            "scale": base_quantization.get("scale", "per_tensor_mean_abs"),
            "ste": base_quantization.get("ste", "pass_through"),
            "ste_clip": base_quantization.get("ste_clip"),
            "exclude_layers": excluded,
        },
    }
    metadata = {
        "policy": policy,
        "selected_category": selection.get("selected_category"),
        "keep_count": selection.get("keep_count"),
        "preliminary": True,
        "screen_metric": metric,
        "screen": selection,
        "base_excluded_layers": list(base_excluded),
        "attention_kept_fp32": attention_kept,
        "quantized_attention_layers": [r["name"] for r in quantized_layers],
        "quantized_attention_layer_count": len(quantized_layers),
        "estimated_size": size,
        "warning": (
            "Selection is based on a small preliminary screen. No WER is "
            "predicted; validate with the comparable pilot before promotion."
        ),
    }
    return config, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a mixed-attention config.")
    parser.add_argument("--screen", required=True)
    parser.add_argument("--base-config", default="configs/qat_ffn_mixed.yaml")
    parser.add_argument("--policy", choices=["category", "sensitive"], required=True)
    parser.add_argument("--metric", choices=sorted(METRIC_FIELDS), default="relative_l2")
    parser.add_argument("--category", default=None)
    parser.add_argument("--keep-count", type=int, default=24)
    parser.add_argument("--base-artifact-bytes", type=int, default=BASE_ARTIFACT_BYTES_DEFAULT)
    parser.add_argument("--id", default=None, help="experiment identifier")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metadata-output", default=None)
    args = parser.parse_args()

    screen = json.loads(Path(args.screen).read_text())
    base_path = ROOT / args.base_config
    base = load_config(base_path)
    base_layers = dict(base["layers"])
    base_excluded = list(base.get("quantization", {}).get("exclude_layers") or [])
    if not base_excluded:
        raise SystemExit("base config has no 17-layer FFN exclusion list")

    config, metadata = build_policy(
        screen, base_layers, base_excluded, base.get("quantization", {}),
        policy=args.policy, metric=args.metric, category=args.category,
        keep_count=args.keep_count, base_artifact_bytes=args.base_artifact_bytes,
    )
    exp_id = args.id or f"attention_{args.policy}_{args.metric}"
    output = Path(args.output) if args.output else ROOT / "configs" / f"{exp_id}.yaml"
    metadata_out = Path(args.metadata_output) if args.metadata_output else (
        ROOT / "results" / f"{exp_id}.json"
    )
    # `extends` must resolve from the generated file's directory.
    try:
        config["extends"] = str(base_path.relative_to(output.parent.resolve()))
    except ValueError:
        config["extends"] = str(base_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata.update({
        "id": exp_id,
        "screen_path": str(Path(args.screen)),
        "base_config": str(base_path),
        "config_output": str(output),
    })
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote config {output}")
    print(f"Wrote metadata {metadata_out}")
    print(f"policy={args.policy} metric={args.metric} "
          f"quantized_attention_layers={metadata['quantized_attention_layer_count']} "
          f"estimated_artifact_gb={metadata['estimated_size']['estimated_artifact_gb']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
