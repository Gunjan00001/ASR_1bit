"""Weight-level sensitivity helpers for the attention/FFN screening scripts.

The layer screen already measures output perturbation. The attention screen also
needs the quantizer's *weight reconstruction* error, which is independent of the
dev set and tells us whether a layer's weights are intrinsically hard to
represent with one sign + scale.

These are pure helpers; no model/dataset/GPU is required, so they are cheap to
unit-test.
"""

from __future__ import annotations

import torch

from onebit_asr.quantization.binarization import quantize_weight


def weight_reconstruction_error(
    weight: torch.Tensor,
    scale_mode: str = "per_tensor_mean_abs",
    ste_clip: float | None = None,
) -> float:
    """Relative Frobenius error ``||W - Wq||_F / ||W||_F`` for one weight.

    ``Wq`` is produced by the exact forward-time quantizer, so this is directly
    comparable with the model's effective weights. A zero matrix is defined to
    have zero error (there is no relative scale). The STE clip does not change
    the forward value, but is passed through to keep the call identical to the
    layer replacement path.
    """
    w = weight.detach().to(torch.float32)
    denom = float(w.norm())
    if denom == 0.0:
        return 0.0
    wq, _alpha = quantize_weight(w, scale_mode=scale_mode, ste_clip=ste_clip)
    return float((w - wq).norm() / denom)
