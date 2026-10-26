"""Binary weight quantization (Phase 5).

``Wb = sign(W)``
``Wq = alpha * Wb``  with ``alpha = mean(|W|)`` by default.

``sign`` is routed through the STE so gradients reach the FP master weights.
The master weights themselves are never modified here — this module only
*derives* a quantized view used in the forward pass.

Invariants (checked by tests)
-----------------------------
1. ``binarize(w)`` contains only ``{-1, 0, +1}``; zero is possible only when a
   master weight is exactly zero.
2. ``quantize_weight(w)[1] == w.abs().mean()``.
3. ``quantize_weight`` does **not** mutate ``w``.
4. The returned ``alpha`` is a 0-D tensor attached to the autograd graph.

Debugging note: if you need to inspect the quantized weights without affecting
training, use ``BitLinear.effective_weight()`` (it runs under the current graph)
or wrap the call in ``torch.no_grad()``. Do not assign the result back to
``self.weight``; that would destroy the FP master.
"""

from __future__ import annotations

import torch

from onebit_asr.quantization.scaling import compute_scale
from onebit_asr.quantization.ste import binary_quantize
from onebit_asr.quantization.two_bit import (
    quantize_weight_2bit,
    validate_bits,
)


def binarize(weight: torch.Tensor, clip_value: float | None = None) -> torch.Tensor:
    """``Wb = sign(W)`` with straight-through gradients."""
    return binary_quantize(weight, clip_value)


def quantize_weight(
    weight: torch.Tensor,
    scale_mode: str = "per_tensor_mean_abs",
    ste_clip: float | None = None,
    bits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Wq, alpha)`` for the configured bit width.

    ``bits=1`` is the original path: ``Wq = alpha * sign(W)``. ``bits=2`` uses
    the symmetric mid-rise codebook in :mod:`onebit_asr.quantization.two_bit`.
    The 1-bit default keeps every historical experiment and checkpoint path
    unchanged.
    """
    bits = validate_bits(bits)
    if bits == 2:
        return quantize_weight_2bit(weight, scale_mode, ste_clip)
    # Shape contract check: a scalar or one scale per output row. A wrong shape
    # here would silently broadcast in a surprising direction.
    alpha = compute_scale(weight, scale_mode)  # differentiable scale(s)
    if alpha.dim() == 2:
        if alpha.shape != (weight.shape[0], 1):
            raise ValueError(
                f"alpha shape {tuple(alpha.shape)} does not match weight "
                f"{tuple(weight.shape)} for mode {scale_mode!r}"
            )
    elif alpha.dim() != 0:
        raise ValueError(
            f"alpha must be 0-D or [out,1], got {tuple(alpha.shape)} for mode {scale_mode!r}"
        )
    wb = binarize(weight, ste_clip)          # ±1 pattern; STE-backed
    return alpha * wb, alpha
