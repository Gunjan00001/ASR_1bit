"""Straight-Through Estimator (STE) for binarization (Phase 6).

Forward:  ``sign(x)``
Backward: gradient passes straight through to the FP master weights.

Design decision (documented, not silent): the default STE is *pure*
pass-through. An optional clipped variant is available and must be requested
explicitly via ``clip_value``. With clipping, gradients are zeroed where
``|x| > clip_value`` (the "hard tanh" / BNN-style variant). The mode actually
in use is always reported by :class:`BitLinear`'s ``extra_repr``.
"""

from __future__ import annotations

import torch


class BinaryQuantize(torch.autograd.Function):
    """``sign`` in the forward pass, straight-through gradient in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, clip_value: float | None) -> torch.Tensor:  # noqa: D102
        ctx.save_for_backward(x)
        ctx.clip_value = clip_value
        # sign(0) == 0. Exact zeros are measure-zero for trained weights; the
        # behaviour is documented and covered by tests rather than nudged.
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        (x,) = ctx.saved_tensors
        clip_value = ctx.clip_value
        if clip_value is None:
            grad = grad_output
        else:
            # Clipped STE: no gradient for inputs outside [-clip, clip].
            grad = grad_output * (x.abs() <= clip_value).to(grad_output.dtype)
        # No gradient w.r.t. clip_value (it is a Python float, not a tensor).
        return grad, None


def binary_quantize(x: torch.Tensor, clip_value: float | None = None) -> torch.Tensor:
    """Binarize ``x`` to {-1, +1} with straight-through gradients.

    Args:
        x: input tensor (typically an FP master weight).
        clip_value: ``None`` (default) for pure pass-through, or a positive
            float to enable the clipped STE variant.
    """
    if clip_value is not None and clip_value <= 0:
        raise ValueError(f"clip_value must be positive or None, got {clip_value}")
    return BinaryQuantize.apply(x, clip_value)
