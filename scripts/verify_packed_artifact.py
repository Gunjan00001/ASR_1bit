"""Load a packed 1-bit artifact and verify it reproduces the QAT model.

Why this exists
---------------
The packed artifact used to be write-only: the safetensors file existed, but no
code could rebuild a model from it. This script loads the artifact + sidecar
manifest, reconstructs `PackedBitLinear` layers from the *artifact* tensors
(not from the FP32 masters), loads the non-quantized parameters, and compares
logits/decoded output with the QAT reference on a few dev utterances.

The artifact is accepted only if:
- every non-quantized artifact key exists in the rebuilt model;
- the packed sign bits and scale shapes match the manifest;
- argmax agreement with the QAT model is >= 0.99;
- max |logit difference| <= 1e-2.

Usage:
    python scripts/verify_packed_artifact.py \
        --config configs/qat_ffn_per_channel.yaml \
        --checkpoint checkpoints/qat_ffn_per_channel \
        --artifact checkpoints/qat_ffn_per_channel_packed/model_packed.safetensors \
        --output results/packed_verify_ffn_per_channel.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.models.conformer_utils import get_parent_module
from onebit_asr.models.replace_layers import replace_linears
from onebit_asr.quantization.packed import PackedBitLinear
from onebit_asr.training.qat import load_qat_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def build_packed_from_artifact(base_model, artifact: dict, manifest: dict):
    """Replace every artifact-packed layer with a PackedBitLinear.

    The base model supplies the architecture/config; all numbers come from the
    artifact. The FP32 masters in the base model are deliberately not used for
    the packed layers.
    """
    weight_shapes = manifest.get("weight_shapes", {})
    packed_keys = [k for k in artifact if k.endswith("_packed")]
    if not packed_keys:
        raise ValueError("artifact has no *_packed tensors")

    replaced = 0
    for packed_key in packed_keys:
        base = packed_key[: -len("_packed")]          # ...linear_q.weight
        module_name = base.rsplit(".", 1)[0]          # ...linear_q
        scale_key = f"{base}_scale"
        if scale_key not in artifact:
            raise ValueError(f"missing scale tensor {scale_key!r} for {packed_key!r}")
        if base not in weight_shapes:
            raise ValueError(f"manifest has no weight shape for {base!r}")

        packed = artifact[packed_key]
        alpha = artifact[scale_key]
        bias_key = f"{module_name}.bias"
        bias = artifact.get(bias_key)
        layer = PackedBitLinear.from_artifact(
            packed, tuple(weight_shapes[base]), alpha, bias
        )
        parent, attr = get_parent_module(base_model, module_name)
        setattr(parent, attr, layer)
        replaced += 1

    # Load every non-quantized artifact tensor. After replacement, the model's
    # keys include ...bias/...alpha for packed layers and the original keys for
    # the rest; the artifact's non_quantized_keys must be a subset.
    non_quantized = manifest.get("non_quantized_keys", [])
    model_keys = set(base_model.state_dict())
    missing = [k for k in non_quantized if k not in model_keys]
    if missing:
        raise ValueError(
            f"{len(missing)} non-quantized artifact keys are not in the model: "
            f"{missing[:8]}"
        )
    state = {k: artifact[k] for k in non_quantized}
    base_model.load_state_dict(state, strict=False)
    return base_model, replaced


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a packed 1-bit artifact.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", default=None,
                        help="sidecar manifest (default: <artifact>.manifest.json)")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-logit-diff", type=float, default=1e-2)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.99)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    quant = cfg.get("quantization", {})
    scale_mode = quant.get("scale", "per_tensor_mean_abs")
    ste_clip = quant.get("ste_clip")
    exclude_layers = list(quant.get("exclude_layers") or [])
    checkpoint = ROOT / args.checkpoint
    artifact_path = ROOT / args.artifact
    manifest_path = Path(args.manifest) if args.manifest else Path(str(artifact_path) + ".manifest.json")
    if not artifact_path.is_file():
        raise SystemExit(f"artifact not found: {artifact_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"artifact manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    artifact = load_file(str(artifact_path), device="cpu")
    if manifest.get("layers") is not None and dict(manifest["layers"]) != dict(cfg["layers"]):
        raise SystemExit("artifact layer selection does not match the config")
    if "exclude_layers" in manifest and sorted(manifest.get("exclude_layers") or []) != sorted(exclude_layers):
        raise SystemExit("artifact mixed-precision exclusion does not match the config")

    print("Loading QAT reference (FP32 masters) ...")
    reference, _report = load_qat_checkpoint(
        checkpoint, Wav2Vec2ConformerForCTC, cfg["layers"],
        scale_mode=scale_mode, ste_clip=ste_clip,
        exclude_layers=exclude_layers,
    )
    reference.eval()

    print("Rebuilding the model from the packed artifact ...")
    packed_base = Wav2Vec2ConformerForCTC.from_pretrained(checkpoint)
    packed, n_replaced = build_packed_from_artifact(packed_base, artifact, manifest)
    packed.eval()
    print(f"Rebuilt {n_replaced} packed layers")

    processor = Wav2Vec2Processor.from_pretrained(cfg["qat"]["init_from"])
    samples = load_dev_subset(subset_size=max(args.n, 4), seed=cfg["eval"]["seed"])[: args.n]

    rows = []
    all_argmax = []
    max_diff = 0.0
    decoded_matches = 0
    for sample in samples:
        inputs = processor(sample["audio"], sampling_rate=16_000,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            ref_logits = reference(inputs.input_values).logits
            packed_logits = packed(inputs.input_values).logits
        diff = float((ref_logits - packed_logits).abs().max())
        argmax = float((ref_logits.argmax(-1) == packed_logits.argmax(-1)).float().mean())
        ref_text = processor.batch_decode(ref_logits.argmax(-1))[0]
        packed_text = processor.batch_decode(packed_logits.argmax(-1))[0]
        max_diff = max(max_diff, diff)
        all_argmax.append(argmax)
        decoded_matches += int(ref_text == packed_text)
        rows.append({
            "id": sample["id"],
            "max_abs_logit_diff": diff,
            "argmax_agreement": argmax,
            "decoded_match": bool(ref_text == packed_text),
        })

    mean_argmax = sum(all_argmax) / len(all_argmax)
    passed = (
        max_diff <= args.max_logit_diff
        and mean_argmax >= args.min_argmax_agreement
        and decoded_matches == len(rows)
    )
    result = {
        "phase": "Phase 16/18 packed artifact reload verification",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "config_path": cfg.get("_config_path"),
        "checkpoint": str(checkpoint),
        "artifact": str(artifact_path),
        "artifact_manifest": str(manifest_path),
        "scale_mode": manifest.get("scale_mode"),
        "layers": manifest.get("layers"),
        "n_rebuilt_layers": n_replaced,
        "n_utterances": len(rows),
        "max_abs_logit_diff": max_diff,
        "mean_argmax_agreement": mean_argmax,
        "decoded_match_count": decoded_matches,
        "criteria": {
            "max_logit_diff": args.max_logit_diff,
            "min_argmax_agreement": args.min_argmax_agreement,
        },
        "passed": bool(passed),
        "utterances": rows,
    }
    out = Path(args.output) if args.output else (
        ROOT / "results" / f"packed_verify_{Path(args.checkpoint).name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"max |logit diff| = {max_diff:.3e}; mean argmax agreement = {mean_argmax:.6f}; "
          f"decoded matches = {decoded_matches}/{len(rows)}")
    print(f"PASSED: {passed}")
    print(f"Saved {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
