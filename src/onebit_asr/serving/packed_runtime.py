"""Standalone packed-artifact reconstruction for serving."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, Wav2Vec2ConformerForCTC

from onebit_asr.models.conformer_utils import get_parent_module
from onebit_asr.quantization.packed import PackedBitLinear


def load_packed_model(
    model_dir: str | Path,
    artifact_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict]:
    """Rebuild a model solely from a model directory and a packed artifact.

    ``model_dir`` must contain the HF-compatible ``config.json`` and the packed
    artifact sidecar manifest. The artifact must contain all non-quantized
    parameters and buffers.
    """
    model_dir = Path(model_dir)
    artifact_path = Path(artifact_path)
    manifest_path = Path(str(artifact_path) + ".manifest.json")
    if not manifest_path.is_file():
        manifest_path = model_dir / "model_packed.safetensors.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"packed manifest not found for {artifact_path}")
    manifest = json.loads(manifest_path.read_text())
    artifact = load_file(str(artifact_path), device="cpu")
    config = AutoConfig.from_pretrained(model_dir)
    model = Wav2Vec2ConformerForCTC(config)
    replaced = 0
    for packed_key in [key for key in artifact if key.endswith("_packed")]:
        base = packed_key[: -len("_packed")]
        module_name = base.rsplit(".", 1)[0]
        layer = PackedBitLinear.from_artifact(
            artifact[packed_key],
            tuple(manifest["weight_shapes"][base]),
            artifact[f"{base}_scale"],
            artifact.get(f"{module_name}.bias"),
            bits=int(manifest["bits"]),
        )
        parent, attr = get_parent_module(model, module_name)
        setattr(parent, attr, layer)
        replaced += 1
    if replaced != int(manifest["n_bitlinear_layers"]):
        raise ValueError(
            f"replaced {replaced} packed layers, manifest says "
            f"{manifest['n_bitlinear_layers']}"
        )
    state = {key: artifact[key] for key in manifest["non_quantized_keys"]}
    missing = [key for key in state if key not in model.state_dict()]
    if missing:
        raise ValueError(f"artifact keys missing from model: {missing[:5]}")
    load_result = model.load_state_dict(state, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"unexpected artifact keys: {load_result.unexpected_keys[:5]}")
    model.to(device)
    model.eval()
    return model, {
        "n_packed_layers": replaced,
        "bits": int(manifest["bits"]),
        "scale_mode": manifest.get("scale_mode"),
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact": str(artifact_path),
        "manifest": str(manifest_path),
    }
