"""Verify the attention-only layer replacement produces exactly 96 BitLinear.

Runs on Kaggle (or anywhere) before any training. Loads the fine-tuned FP32
checkpoint and applies the configured taxonomy selection, then asserts the
expected BitLinear count. Writes a machine-readable report and exits non-zero
if the count is wrong, so a probe/full run can never start on a misconfigured
experiment.

Usage:
    python kaggle/verify_replacement.py --config configs/qat_attn.yaml \
        --output replacement_check.json
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import Wav2Vec2ConformerForCTC

from onebit_asr.config import load_config
from onebit_asr.models.replace_layers import replace_linears
from onebit_asr.quantization.bitlinear import BitLinear

EXPECTED_ATTN_REPLACEMENTS = 96


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify attention-only replacement count.")
    parser.add_argument("--config", default="configs/qat_attn.yaml")
    parser.add_argument("--output", default="replacement_check.json")
    parser.add_argument("--expected", type=int, default=EXPECTED_ATTN_REPLACEMENTS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    quant_cfg = cfg.get("quantization", {})
    scale_mode = quant_cfg.get("scale", "per_tensor_mean_abs")
    ste_clip = quant_cfg.get("ste_clip", None)
    model_id = cfg["qat"]["init_from"]

    print(f"Loading {model_id} to verify layer replacement ...")
    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
    report = replace_linears(model, cfg["layers"], scale_mode=scale_mode, ste_clip=ste_clip)

    n_bitlinear = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    passed = n_bitlinear == args.expected

    result = {
        "model_id": model_id,
        "expected_bitlinear": args.expected,
        "n_bitlinear": n_bitlinear,
        "passed": passed,
        "layers": cfg["layers"],
        "quantization": {"scale": scale_mode, "ste_clip": ste_clip},
        "replacement": {
            "n_replaced": report["n_replaced"],
            "n_retained": report["n_retained"],
            "replaced_params": report["replaced_params"],
            "retained_params": report["retained_params"],
            "replaced_categories": sorted({e["category"] for e in report["replaced"]}),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print(f"BitLinear layers: {n_bitlinear} (expected {args.expected}) -> "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"Report: {output}")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
