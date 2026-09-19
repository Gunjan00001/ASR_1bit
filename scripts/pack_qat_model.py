"""Write a real, deployable packed 1-bit artifact from a QAT checkpoint.

Loads a QAT checkpoint (FP32 master weights), reconstructs the binarized model
with the configured layer selection, and writes the packed safetensors artifact
(``numpy.packbits`` weights + per-tensor FP32 scales + all non-quantized FP
parameters). Reports the measured artifact size honestly.

This is Phase 16 tooling: the QAT checkpoint itself stores FP32 *masters* (not
packed). This script produces the actual 1-bit storage artifact.

Usage:
    python scripts/pack_qat_model.py \
        --config configs/qat_attn.yaml \
        --checkpoint checkpoints/qat_attn \
        --output checkpoints/qat_attn_packed/model_packed.safetensors \
        --report checkpoints/qat_attn_packed/size_report.json
"""

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

from transformers import Wav2Vec2ConformerForCTC

from onebit_asr.config import load_config
from onebit_asr.evaluation.model_size import write_packed_artifact
from onebit_asr.training.qat import load_qat_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a packed 1-bit deploy artifact.")
    parser.add_argument("--config", default="configs/qat_attn.yaml")
    parser.add_argument("--checkpoint", required=True, help="QAT checkpoint dir (FP32 masters)")
    parser.add_argument("--output", required=True, help="output .safetensors path")
    parser.add_argument("--report", default=None,
                        help="size report JSON path (default: alongside the artifact)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    quant_cfg = cfg.get("quantization", {})
    scale_mode = quant_cfg.get("scale", "per_tensor_mean_abs")
    ste_clip = quant_cfg.get("ste_clip", None)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint dir not found: {checkpoint}")

    print(f"Loading QAT checkpoint {checkpoint} ...")
    model, replacement = load_qat_checkpoint(
        checkpoint, Wav2Vec2ConformerForCTC, cfg["layers"],
        scale_mode=scale_mode, ste_clip=ste_clip,
    )
    model.eval()

    output = Path(args.output)
    print(f"Writing packed artifact -> {output} ...")
    report = write_packed_artifact(model, output, scale_mode=scale_mode)
    report.update({
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "checkpoint": str(checkpoint),
        "config_path": cfg.get("_config_path"),
        "layers": cfg["layers"],
        "replacement": {
            "n_replaced": replacement["n_replaced"],
            "n_retained": replacement["n_retained"],
        },
        "note": (
            "Deployable packed artifact = packed binary weights + FP32 scales + "
            "non-quantized FP params. FP32 master weights are training-time state "
            "and are excluded."
        ),
    })

    report_path = Path(args.report) if args.report else output.parent / "size_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Packed artifact bytes (measured): {report['packed_artifact_bytes_measured']:,}")
    print(f"Packed payload bytes (1 bit/weight): {report['packed_payload_bytes']:,}")
    print(f"BitLinear layers: {report['n_bitlinear_layers']}")
    print(f"Size report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
