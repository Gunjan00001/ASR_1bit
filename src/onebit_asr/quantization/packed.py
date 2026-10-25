"""Packed 1-bit weight inference (Phase 18 investigation).

What this module does
---------------------
It serves a ``BitLinear`` layer from **1-bit packed weights** (``numpy.packbits``)
plus a FP32 per-tensor scale, instead of keeping the FP32 master weight in
memory. This is a real *storage* win: the weight payload is 1/32 of FP32.

What this module deliberately does **not** claim
------------------------------------------------
It is **not** a compute accelerator. The model uses FP activations, so a
binary-weight layer still needs FP multiply-accumulates; the packed bits are
expanded to ±1 floats inside ``forward`` and passed to ``F.linear``. A true
bitwise kernel (XNOR + popcount) requires **1-bit activations too**, which is a
different model. ``xnor_popcount_packed`` is included as a kernel-level
reference so that boundary is measured rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from onebit_asr.quantization.bitlinear import BitLinear

#: popcount lookup table for one byte
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def pack_sign_bits(weight: torch.Tensor) -> np.ndarray:
    """Pack ``sign(weight) > 0`` into a flat ``uint8`` bit array."""
    bits = (torch.sign(weight.detach()).flatten().cpu().numpy() > 0).astype(np.uint8)
    return np.packbits(bits)


def unpack_sign_bits(packed: np.ndarray, numel: int, shape: tuple[int, ...]) -> np.ndarray:
    """Unpack to a float32 array of ±1 with the original shape.

    Bit convention: 1 → +1, 0 → −1. `pack_sign_bits` used ``sign(W) > 0``, so a
    zero weight becomes −1 here (the same measure-zero edge case as the size
    accounting; trained weights are not exactly zero).
    """
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[:numel]
    return (bits.astype(np.float32) * 2.0 - 1.0).reshape(shape)


class PackedBitLinear(nn.Module):
    """Inference-only BitLinear backed by packed sign bits + a FP32 scale."""

    def __init__(
        self,
        packed: np.ndarray,
        shape: tuple[int, int],
        alpha: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = int(shape[1])
        self.out_features = int(shape[0])
        self.numel = self.in_features * self.out_features
        self.register_buffer("packed", torch.from_numpy(np.asarray(packed, dtype=np.uint8)))
        self.alpha = nn.Parameter(alpha.detach().clone().float(), requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_bitlinear(cls, module: BitLinear) -> "PackedBitLinear":
        weight = module.weight.detach()
        packed = pack_sign_bits(weight)
        alpha = weight.abs().mean().float()
        return cls(packed, tuple(weight.shape), alpha, module.bias)

    def effective_weight(self) -> torch.Tensor:
        raw = unpack_sign_bits(self.packed.cpu().numpy(), self.numel,
                               (self.out_features, self.in_features))
        weight = torch.from_numpy(raw).to(device=self.alpha.device, dtype=self.alpha.dtype)
        return weight * self.alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Every call unpacks the bits and materializes an FP weight. That is the
        # honest cost of serving from packed storage with an FP-activation
        # runtime. If you cache `effective_weight()` for speed, you lose the
        # storage win in memory; do not do both and claim compression.
        return F.linear(input, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:  # noqa: D105
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, packed_bytes={self.packed.numel()}, "
            f"storage=1-bit+fp32_scale"
        )


def pack_model(model: nn.Module) -> tuple[nn.Module, dict]:
    """Replace every ``BitLinear`` in ``model`` with ``PackedBitLinear`` in place.

    This drops the FP32 masters (they are not copied into the packed module).
    It is intentionally destructive: keep the QAT checkpoint if you need to
    resume training. The returned model is for inference/benchmarking only.
    """
    replaced = 0
    packed_bytes = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, BitLinear):
            continue
        # Same parent-setattr trick as replace_linears: modules are attributes
        # of their parent, not of themselves.
        parent_path, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_path) if parent_path else model
        new = PackedBitLinear.from_bitlinear(module)
        setattr(parent, attr, new)
        packed_bytes += int(new.packed.numel())
        replaced += 1
    return model, {
        "n_packed_layers": replaced,
        "packed_weight_bytes": packed_bytes,
        "scale_bytes": replaced * 4,
        "note": "Forward expands packed bits to FP; storage-only compression.",
    }


def packed_model_size_report(model: nn.Module) -> dict:
    """Size accounting for a converted (PackedBitLinear) model."""
    packed_bytes = 0
    n_layers = 0
    packed_param_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            n_layers += 1
            packed_bytes += int(module.packed.numel())
            packed_param_names.update({f"{name}.alpha", f"{name}.bias"})

    fp_scale_bias_bytes = 0
    non_packed_bytes = 0
    total_params = 0
    for name, param in model.named_parameters():
        byte_count = param.numel() * param.element_size()
        total_params += param.numel()
        if name in packed_param_names:
            fp_scale_bias_bytes += byte_count
        else:
            non_packed_bytes += byte_count

    return {
        "n_packed_layers": n_layers,
        "packed_weight_bytes": packed_bytes,
        "fp_scale_and_bias_bytes": fp_scale_bias_bytes,
        "non_packed_fp_param_bytes": non_packed_bytes,
        "total_fp_param_bytes": fp_scale_bias_bytes + non_packed_bytes,
        "total_params": total_params,
        "note": "Total deploy bytes = packed weights + FP scales/biases + non-packed FP params. "
                "No FP32 master weights are retained.",
    }


def xnor_popcount_packed(x_packed: np.ndarray, w_packed: np.ndarray, n_bits: int) -> np.ndarray:
    """XNOR + popcount similarity for 1-bit×1-bit matrices.

    Args:
        x_packed: ``(M, ceil(N/8))`` uint8 packed rows.
        w_packed: ``(K, ceil(N/8))`` uint8 packed rows.
        n_bits: original inner dimension ``N``.

    Returns:
        ``(M, K)`` int32 matrix of matching-bit counts (higher = more similar).

    This is the only place in the repo that performs true 1-bit×1-bit compute.
    The intermediate `xor` tensor is `(M, K, N/8)` bytes, so this reference is
    memory-hungry for large shapes; it is a correctness/speed probe, not a
    production kernel.
    """
    x = np.asarray(x_packed, dtype=np.uint8)
    w = np.asarray(w_packed, dtype=np.uint8)
    xor = np.bitwise_xor(x[:, None, :], w[None, :, :])
    return (n_bits - POPCOUNT[xor].sum(axis=-1)).astype(np.int32)


def xnor_similarity_to_dot(similarity: np.ndarray, n_bits: int) -> np.ndarray:
    """Convert matching-bit counts to the ±1 dot product ``2*matches - N``."""
    return (similarity * 2 - n_bits).astype(np.int32)


def packed_artifact_paths(report: dict) -> dict:
    """Normalize a report's paths for JSON output."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in report.items()}
