"""Write a real, deployable packed 1-bit artifact from a QAT checkpoint.

Loads a QAT checkpoint (FP32 master weights), reconstructs the binarized model
with the configured layer selection, and writes the packed safetensors artifact
(``numpy.packbits`` weights + scales (per-tensor or per-output-channel) + all
non-quantized FP parameters). Reports the measured artifact size honestly and
writes a sidecar manifest so the artifact can be reloaded and verified.

This is Phase 16 tooling: the QAT checkpoint itself stores FP32 *masters* (not
packed). This script produces the actual 1-bit storage artifact.

Usage:
    python scripts/pack_qat_model.py \
        --config configs/qat_ffn_per_channel.yaml \
        --checkpoint checkpoints/qat_ffn_per_channel \
        --output checkpoints/qat_ffn_per_channel_packed/model_packed.safetensors \
        --report checkpoints/qat_ffn_per_channel_packed/size_report.json
"""

import argparse
import datetime
import json
import os
import platform
import sys
from pathlib import Path

from safetensors import safe_open
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
    exclude_layers = list(quant_cfg.get("exclude_layers") or [])

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint dir not found: {checkpoint}")

    print(f"Loading QAT checkpoint {checkpoint} ...")
    model, replacement = load_qat_checkpoint(
        checkpoint, Wav2Vec2ConformerForCTC, cfg["layers"],
        scale_mode=scale_mode, ste_clip=ste_clip,
        exclude_layers=exclude_layers,
    )
    model.eval()

    output = Path(args.output)
    print(f"Writing packed artifact -> {output} ...")
    report = write_packed_artifact(model, output, scale_mode=scale_mode)
    report.update({
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "checkpoint": str(checkpoint),
        "config_path": cfg.get("_config_path"),
        "layers": cfg["layers"],
        "exclude_layers": exclude_layers,
        "mixed_precision": bool(exclude_layers),
        "replacement": {
            "n_replaced": replacement["n_replaced"],
            "n_retained": replacement["n_retained"],
            "n_excluded": replacement["n_excluded"],
        },
        "note": (
            "Deployable packed artifact = packed binary weights + scales + "
            "non-quantized FP params. FP32 master weights are training-time state "
            "and are excluded."
        ),
    })

    # Sidecar manifest: the safetensors file alone does not record which layers
    # were quantized, the scale granularity, or the original shapes. Without
    # this, the artifact cannot be reloaded unambiguously.
    with safe_open(str(output), framework="pt", device="cpu") as fh:
        shapes = {key: list(fh.get_slice(key).get_shape()) for key in fh.keys()}
    non_quantized_keys = [
        key for key in shapes if not key.endswith(("_packed", "_scale"))
    ]
    # The packed tensor is flat and the scale shape does not identify both
    # dimensions (per-tensor scale is [1]); record the original [out,in] shapes
    # so a loader can rebuild PackedBitLinear unambiguously.
    weight_shapes = {
        f"{name}.weight": list(module.weight.shape)
        for name, module in model.named_modules()
        if getattr(module, "is_bitlinear", False)
    }
    artifact_manifest = {
        "artifact": output.name,
        "artifact_bytes_measured": report["packed_artifact_bytes_measured"],
        "checkpoint": str(checkpoint),
        "pbs_jobid": report.get("pbs_jobid"),
        "config_path": cfg.get("_config_path"),
        "layers": cfg["layers"],
        "exclude_layers": exclude_layers,
        "mixed_precision": bool(exclude_layers),
        "scale_mode": scale_mode,
        "ste_clip": ste_clip,
        "n_bitlinear_layers": report["n_bitlinear_layers"],
        "quantized_weight_count": report["quantized_weight_count"],
        "packed_payload_bytes": report["packed_payload_bytes"],
        "fp_scale_bytes": report["fp_scale_bytes"],
        "fp_scale_element_count": report.get("fp_scale_element_count"),
        "non_quantized_fp_param_bytes": report["non_quantized_fp_param_bytes"],
        "effective_bits_per_quantized_param": report.get(
            "effective_bits_per_quantized_param"
        ),
        "effective_bits_per_total_param": report.get("effective_bits_per_total_param"),
        "effective_bits_per_total_param_measured": report.get(
            "effective_bits_per_total_param_measured"
        ),
        "shapes": shapes,
        "weight_shapes": weight_shapes,
        "non_quantized_keys": non_quantized_keys,
    }
    manifest_path = Path(str(output) + ".manifest.json")
    manifest_path.write_text(json.dumps(artifact_manifest, indent=2))
    report["artifact_manifest_path"] = str(manifest_path)

    report_path = Path(args.report) if args.report else output.parent / "size_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Packed artifact bytes (measured): {report['packed_artifact_bytes_measured']:,}")
    print(f"Packed payload bytes (1 bit/weight): {report['packed_payload_bytes']:,}")
    print(f"Scale bytes: {report['fp_scale_bytes']:,} "
          f"({report.get('fp_scale_element_count')} values, {scale_mode})")
    print(f"Effective bits / quantized weight: "
          f"{report.get('effective_bits_per_quantized_param'):.4f}")
    print(f"Effective bits / model parameter: "
          f"{report.get('effective_bits_per_total_param'):.4f} "
          f"(measured {report.get('effective_bits_per_total_param_measured'):.4f})")
    print(f"BitLinear layers: {report['n_bitlinear_layers']}")
    print(f"Artifact manifest: {report['artifact_manifest_path']}")
    print(f"Size report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
