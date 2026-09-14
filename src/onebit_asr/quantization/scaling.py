"""Weight scaling for binarization (Phase 5).

Default strategy (plan §Phase 5, amended): per-tensor
``alpha = mean(|W|)`` — the XNOR-Net style L1 scaling factor.

Design decisions:
- **per-tensor** (one scalar per layer weight matrix), not per-channel;
- **recomputed on every forward** from the current FP master weights;
- **not a learned parameter** — but gradients do flow back through ``alpha``
  into the master weights because ``alpha`` is a differentiable function of W
  (``mean(abs(W))``). ``sign()`` gradients flow via the STE.
"""

from __future__ import annotations

import torch

SCALE_MODES = ("per_tensor_mean_abs",)


def per_tensor_mean_abs(weight: torch.Tensor) -> torch.Tensor:
    """``alpha = mean(|W|)`` over every element of the tensor."""
    return weight.abs().mean()


_SCALE_FNS = {
    "per_tensor_mean_abs": per_tensor_mean_abs,
}


def compute_scale(weight: torch.Tensor, mode: str = "per_tensor_mean_abs") -> torch.Tensor:
    """Compute the scalar scale for ``weight`` under the named strategy."""
    if mode not in _SCALE_FNS:
        raise ValueError(f"Unknown scale mode {mode!r}. Available: {list(SCALE_MODES)}")
    return _SCALE_FNS[mode](weight)
