"""Model size accounting and binary packing.

Phase 3 provides FP32 accounting. Phase 10/16 add honest binary accounting:
theoretical payload, naive (1-byte-per-weight) representation, real packed
representation (``numpy.packbits``), FP scales, non-quantized FP parameters,
and real container overhead.

Nothing here reports a "1-bit model size" that silently omits scales, FP
parameters, or container overhead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

WEIGHT_SUFFIXES = (".safetensors", ".bin")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fp32_weight_bytes(total_params: int) -> int:
    return total_params * 4


def checkpoint_size_bytes(repo_id: str, weight_format: str = "safetensors") -> tuple[int, list[str]]:
    """Size of downloaded weight files.

    Args:
        repo_id: Hugging Face repo id.
        weight_format: ``"safetensors"`` (canonical, single weight format) or
            ``"all"`` (every ``.safetensors``/``.bin`` file, which on many repos
            double-counts the same weights).

    Returns:
        ``(total_bytes, [file paths])``.
    """
    if weight_format not in ("safetensors", "all"):
        raise ValueError(f"weight_format must be 'safetensors' or 'all', got {weight_format!r}")

    snap = Path(snapshot_download(repo_id))
    files = [
        str(p) for p in sorted(snap.rglob("*"))
        if p.is_file() and p.suffix in WEIGHT_SUFFIXES
    ]
    if weight_format == "safetensors":
        files = [f for f in files if f.endswith(".safetensors")]
    return sum(Path(f).stat().st_size for f in files), files


def checkpoint_size_report(repo_id: str) -> dict:
    """Both checkpoint figures.

    ``safetensors`` is the canonical comparison basis; ``full_download`` sums
    every weight file and is reported only for transparency (it often
    double-counts when a repo ships both formats).
    """
    full_bytes, full_files = checkpoint_size_bytes(repo_id, "all")
    st_bytes, st_files = checkpoint_size_bytes(repo_id, "safetensors")
    return {
        "checkpoint_bytes_safetensors": st_bytes,
        "checkpoint_bytes_full_download": full_bytes,
        "safetensors_files": [Path(f).name for f in st_files],
        "all_weight_files": [Path(f).name for f in full_files],
    }


# --- Binary representation accounting (Phase 10/16) -------------------------

def naive_binary_bytes(num_weights: int) -> int:
    """One byte per weight (how ``bool``/``int8`` tensors actually serialize)."""
    return int(num_weights)


def theoretical_payload_bytes(num_weights: int) -> int:
    """Raw 1 bit/weight payload only — excludes scales and everything else."""
    return int(np.ceil(num_weights / 8))


def pack_binary_weights(weight: torch.Tensor) -> np.ndarray:
    """Bit-pack ``sign(weight)`` into ``uint8`` (1 bit per weight).

    ``numpy.packbits`` packs along the flattened array; we record the original
    shape so it can be unpacked and reshaped.
    """
    bits = (torch.sign(weight.detach()).flatten().cpu().numpy() > 0).astype(np.uint8)
    return np.packbits(bits)


def packed_binary_bytes(num_weights: int) -> int:
    """Byte length of the real ``numpy.packbits`` payload for ``num_weights``."""
    return int(np.packbits(np.zeros(int(num_weights), dtype=np.uint8)).nbytes)


def _is_bitlinear(module: nn.Module) -> bool:
    # Marker attribute set by BitLinear; avoids importing quantization here.
    return bool(getattr(module, "is_bitlinear", False))


def binary_model_size_report(model: nn.Module, repo_id: str | None = None, scale_mode: str = "per_tensor_mean_abs") -> dict:
    """Honest size accounting for a (partially) binarized model.

    Separates, with no double counting:

    - theoretical 1-bit weight payload (``ceil(n/8)``, payload only)
    - naive binary representation (1 byte/weight, how bool/int8 serializes)
    - real packed payload (``numpy.packbits``)
    - FP scales (one FP32 scalar per BitLinear, per-tensor)
    - non-quantized FP parameters (everything not a BitLinear weight)
    - a *measured* container overhead: a real safetensors file is written with
      the packed weights + scales + remaining FP params, and its size recorded.

    The FP32 master weights of BitLinear layers are intentionally excluded from
    the packed artifact (they are training-time state for Phase 11 QAT); they
    are accounted separately as ``fp_master_bytes``.
    """
    import tempfile

    from safetensors.torch import save_file

    bitlinear_weights: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if _is_bitlinear(module) and hasattr(module, "weight"):
            bitlinear_weights[f"{name}.weight"] = module

    quantized_weight_count = sum(m.weight.numel() for m in bitlinear_weights.values())
    component_bytes = 0  # tracked for cross-check against the measured file

    tensors: dict[str, torch.Tensor] = {}
    packed_payload_bytes = 0
    scale_bytes = 0
    for weight_name, module in bitlinear_weights.items():
        packed = pack_binary_weights(module.weight)
        packed_payload_bytes += packed.nbytes
        tensors[f"{weight_name}_packed"] = torch.from_numpy(packed)
        alpha = module.weight.detach().abs().mean().to(torch.float32).reshape(1)
        scale_bytes += alpha.numel() * alpha.element_size()
        tensors[f"{weight_name}_scale"] = alpha

    non_quantized_params = 0
    non_quantized_bytes = 0
    for pname, param in model.named_parameters():
        if pname in bitlinear_weights:
            continue
        tensors[pname] = param.detach().to(torch.float32).contiguous()
        non_quantized_params += param.numel()
        non_quantized_bytes += param.numel() * param.element_size()

    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "packed_model.safetensors"
        save_file(tensors, str(artifact))
        packed_artifact_bytes = artifact.stat().st_size

    fp_master_bytes = quantized_weight_count * 4
    result = {
        "scale_mode": scale_mode,
        "n_bitlinear_layers": len(bitlinear_weights),
        "quantized_weight_count": int(quantized_weight_count),
        "non_quantized_param_count": int(non_quantized_params),
        "theoretical_payload_bytes_1bit": theoretical_payload_bytes(quantized_weight_count),
        "naive_binary_bytes": naive_binary_bytes(quantized_weight_count),
        "packed_payload_bytes": int(packed_payload_bytes),
        "fp_scale_bytes": int(scale_bytes),
        "non_quantized_fp_param_bytes": int(non_quantized_bytes),
        "fp_master_bytes_excluded_from_artifact": int(fp_master_bytes),
        "packed_artifact_bytes_measured": int(packed_artifact_bytes),
        "container_overhead_bytes": int(packed_artifact_bytes - packed_payload_bytes - scale_bytes - non_quantized_bytes),
        "packed_artifact_components_bytes": int(packed_payload_bytes + scale_bytes + non_quantized_bytes),
    }

    if repo_id is not None:
        result.update(checkpoint_size_report(repo_id))
    return result

