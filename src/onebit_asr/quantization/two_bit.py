"""Genuine symmetric 2-bit weight quantization.

Representation
--------------
A weight matrix is quantized with a per-tensor (or per-output-channel) positive
scale ``alpha`` into one of four **signed** levels::

    code 0 -> -1.5 * alpha
    code 1 -> -0.5 * alpha
    code 2 -> +0.5 * alpha
    code 3 -> +1.5 * alpha

This is a *mid-rise* uniform quantizer: the decision boundaries are at
``w = -2a, -a, 0, +a, +2a``, and the outer levels saturate (clip) beyond
``±2a``.  The codebook is stored explicitly in artifacts; packing is four
2-bit codes per uint8 byte, so the payload is exactly 2 bits per weight
(plus FP32 scales).  It is not a 1-bit sign plus a hidden extra float.

``alpha = mean(|W|)`` (same scaling family as the 1-bit arm).  The four level
magnitudes have mean ``alpha``, so the dequantized matrix has a comparable
scale to the FP master.  Exact zero maps to code 2 (+0.5a); this quantizer has
no zero level by design.  A zero scale (e.g. an all-zero per-channel row) maps
to zero codes and zero dequantized weights, avoiding NaNs.

Training keeps FP32 masters.  The forward derives codes/levels from the live
master; gradients reach the master through a straight-through estimator.  The
packed artifact stores only codes + scales + non-quantized FP parameters; it
does not store FP32 masters.
"""

from __future__ import annotations

import numpy as np
import torch

from onebit_asr.quantization.scaling import compute_scale

BITS = 2
LEVELS = (-1.5, -0.5, 0.5, 1.5)
LEVEL_TABLE = np.asarray(LEVELS, dtype=np.float32)
VALID_BITS = (1, 2)


def validate_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got {bits}")
    return bits


def _safe_scale(alpha: torch.Tensor) -> torch.Tensor:
    return torch.where(alpha != 0, alpha, torch.ones_like(alpha))


def codes_for_weight(weight: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    """Return uint8 codes ``0..3`` for a 2-bit quantization of ``weight``."""
    with torch.no_grad():
        w = weight.detach().float()
        a = alpha.detach().float().to(w.device)
        safe = _safe_scale(a)
        code = torch.clamp(torch.floor(w / safe) + 2.0, 0.0, 3.0)
        code = torch.where(a != 0, code, torch.zeros_like(code))
        return code.to(torch.uint8).cpu().numpy()


class TwoBitQuantize(torch.autograd.Function):
    """Mid-rise 2-bit levels in forward; straight-through gradient in backward."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, alpha: torch.Tensor,
                clip_value: float | None) -> torch.Tensor:  # noqa: D102
        ctx.save_for_backward(weight)
        ctx.clip_value = clip_value
        safe = _safe_scale(alpha)
        code = torch.clamp(torch.floor(weight / safe) + 2.0, 0.0, 3.0)
        level = code - 1.5
        return torch.where(alpha != 0, level, torch.zeros_like(level))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        (weight,) = ctx.saved_tensors
        if ctx.clip_value is None:
            grad = grad_output
        else:
            grad = grad_output * (weight.abs() <= ctx.clip_value).to(grad_output.dtype)
        # No gradient for alpha or the Python clip value. Alpha still receives
        # a gradient through the outer ``alpha * level`` product, exactly as in
        # the 1-bit quantizer's ``alpha * sign(W)``.
        return grad, None, None


def two_bit_quantize(weight: torch.Tensor, alpha: torch.Tensor,
                     clip_value: float | None = None) -> torch.Tensor:
    """Differentiable 2-bit level assignment (values ``{-1.5..1.5}``)."""
    if clip_value is not None and clip_value <= 0:
        raise ValueError(f"clip_value must be positive or None, got {clip_value}")
    return TwoBitQuantize.apply(weight, alpha, clip_value)


def quantize_weight_2bit(
    weight: torch.Tensor,
    scale_mode: str = "per_tensor_mean_abs",
    ste_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(alpha * level, alpha)`` for the symmetric 2-bit codebook."""
    alpha = compute_scale(weight, scale_mode)
    levels = two_bit_quantize(weight, alpha.detach(), ste_clip)
    return alpha * levels, alpha


def pack_2bit_codes(codes: np.ndarray) -> np.ndarray:
    """Pack uint8 codes ``0..3`` four-per-byte; zero-pad the final group."""
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    if flat.size and int(flat.max()) > 3:
        raise ValueError("2-bit codes must be in 0..3")
    pad = (-flat.size) % 4
    if pad:
        flat = np.pad(flat, (0, pad), constant_values=0)
    groups = flat.reshape(-1, 4).astype(np.uint16)
    packed = (groups[:, 0] | (groups[:, 1] << 2) |
              (groups[:, 2] << 4) | (groups[:, 3] << 6)).astype(np.uint8)
    return packed


def unpack_2bit_codes(packed: np.ndarray, numel: int,
                      shape: tuple[int, ...]) -> np.ndarray:
    """Unpack four 2-bit codes per byte and preserve the original shape."""
    p = np.asarray(packed, dtype=np.uint8).ravel()
    codes = np.empty(p.size * 4, dtype=np.uint8)
    codes[0::4] = p & 0x03
    codes[1::4] = (p >> 2) & 0x03
    codes[2::4] = (p >> 4) & 0x03
    codes[3::4] = (p >> 6) & 0x03
    return codes[:numel].reshape(shape)


def unpack_2bit_levels(packed: np.ndarray, numel: int,
                       shape: tuple[int, ...]) -> np.ndarray:
    """Unpack codes to the float level table (before scaling by alpha)."""
    codes = unpack_2bit_codes(packed, numel, shape)
    return LEVEL_TABLE[codes]


def packed_2bit_bytes(numel: int) -> int:
    return int(np.ceil(int(numel) / 4))
