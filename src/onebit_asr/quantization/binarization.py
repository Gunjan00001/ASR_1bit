"""Binary weight quantization (Phase 5).

``Wb = sign(W)``
``Wq = alpha * Wb``  with ``alpha = mean(|W|)`` by default.

``sign`` is routed through the STE so gradients reach the FP master weights.
The master weights themselves are never modified here — this module only
*derives* a quantized view used in the forward pass.
"""

from __future__ import annotations

import torch

from onebit_asr.quantization.scaling import compute_scale
from onebit_asr.quantization.ste import binary_quantize


def binarize(weight: torch.Tensor, clip_value: float | None = None) -> torch.Tensor:
    """``Wb = sign(W)`` with straight-through gradients."""
    return binary_quantize(weight, clip_value)


def quantize_weight(
    weight: torch.Tensor,
    scale_mode: str = "per_tensor_mean_abs",
    ste_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Wq, alpha)`` where ``Wq = alpha * sign(W)``.

    The scale is computed from the same FP master ``weight`` that is being
    binarized, so gradients flow through both the scale and the STE path.
    """
    wb = binarize(weight, ste_clip)
    alpha = compute_scale(weight, scale_mode)
    return alpha * wb, alpha
