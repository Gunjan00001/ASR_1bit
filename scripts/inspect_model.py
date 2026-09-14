"""Phase 1 — model inspection (read-only; never modifies the model).

Loads the pretrained Wav2Vec2-Conformer CTC checkpoint and prints:
- architecture summary, total/trainable parameters, dtype, size
- every nn.Linear with input/output dims and parameter count
- parameter counts grouped by category
  (attention_q/k/v/output, ffn_in, ffn_out, feature_projection, ctc_head,
   attention_other, other)

Categorization uses the shared taxonomy in
``onebit_asr.models.conformer_utils`` (single source of truth, also used by
Phase 8 layer replacement).

Also saves a machine-readable map to results/model_inspection.json
for use by the Phase 8 layer-replacement config.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import Wav2Vec2ConformerForCTC

from onebit_asr.models.conformer_utils import categorize

RESULTS = Path(__file__).resolve().parent.parent / "results"


def encoder_depth(model: nn.Module) -> str:
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            first = type(module[0]).__name__
            if ("EncoderLayer" in first or "ConformerLayer" in first) and "Norm" not in first:
                return f"{len(module)} x {first}"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Inspect Wav2Vec2-Conformer checkpoint.")
    parser.add_argument("--model", default="facebook/wav2vec2-conformer-rope-large-960h-ft")
    parser.add_argument("--output", default=str(RESULTS / "model_inspection.json"))
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model = Wav2Vec2ConformerForCTC.from_pretrained(args.model)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtypes = {str(p.dtype) for p in model.parameters()}
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print("=" * 78)
    print(f"Model class : {type(model).__name__}")
    print(f"Total params: {total:,}  |  trainable: {trainable:,}")
    print(f"Dtypes      : {sorted(dtypes)}  |  in-memory size: {size_mb:,.1f} MB")
    print(f"Encoder     : {encoder_depth(model)}")
    print("=" * 78)

    linears = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params = sum(p.numel() for p in module.parameters())
            linears.append(
                {
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None,
                    "params": params,
                    "category": categorize(name),
                }
            )

    print(f"\nFound {len(linears)} nn.Linear modules:\n")
    print(f"{'category':<18} {'in':>6} {'out':>6} {'params':>12}  name")
    print("-" * 78)
    for entry in linears:
        print(
            f"{entry['category']:<18} {entry['in_features']:>6} "
            f"{entry['out_features']:>6} {entry['params']:>12,}  {entry['name']}"
        )

    by_category: dict = {}
    for entry in linears:
        cat = by_category.setdefault(entry["category"], {"count": 0, "params": 0})
        cat["count"] += 1
        cat["params"] += entry["params"]
    linear_params = sum(e["params"] for e in linears)

    print("\n" + "=" * 78)
    print(f"{'category':<18} {'layers':>7} {'params':>14} {'% of total':>11}")
    print("-" * 78)
    for cat, stats in sorted(by_category.items(), key=lambda kv: -kv[1]["params"]):
        print(f"{cat:<18} {stats['count']:>7} {stats['params']:>14,} {100 * stats['params'] / total:>10.2f}%")
    print("-" * 78)
    print(
        f"{'ALL LINEAR':<18} {len(linears):>7} {linear_params:>14,} "
        f"{100 * linear_params / total:>10.2f}%"
    )
    print("=" * 78)

    # Flag anything the heuristic could not classify.
    unclassified = [e["name"] for e in linears if e["category"] in ("other", "attention_other")]
    if unclassified:
        print(f"\nNOTE: {len(unclassified)} Linear(s) need manual category review:")
        for name in unclassified:
            print(f"  - {name}")

    out = {
        "model_id": args.model,
        "model_class": type(model).__name__,
        "total_params": total,
        "trainable_params": trainable,
        "dtypes": sorted(dtypes),
        "size_mb": size_mb,
        "encoder": encoder_depth(model),
        "linears": linears,
        "by_category": by_category,
        "linear_params_total": linear_params,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved machine-readable map to {out_path}")


if __name__ == "__main__":
    main()
