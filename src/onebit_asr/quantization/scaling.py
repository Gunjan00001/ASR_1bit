"""Weight scaling for binarization (Phase 5).

Default strategy (plan §Phase 5, amended): per-tensor
``alpha = mean(|W|)`` — the XNOR-Net style L1 scaling factor.

Design decisions:
- **per-tensor** (one scalar per layer weight matrix), not per-channel;
- **recomputed on every forward** from the current FP master weights;
- **not a learned parameter** — but gradients do flow back through ``alpha``
  into the master weights because ``alpha`` is a differentiable function of W
  (``mean(abs(W))``). ``sign()`` gradients flow via the STE.

Why this scale
--------------
``sign(W)`` has no magnitude information. A layer full of ±1s would have a
completely different output scale than the FP32 layer it replaced. Multiplying
by ``mean(|W|)`` restores a magnitude comparable to the original weights while
keeping one scalar per layer (768 bytes for 192 layers). Per-channel or learned
scales would be more expressive but are not implemented.

Debugging notes
---------------
- ``alpha`` must be a finite, non-negative **scalar** (``alpha.dim() == 0``).
  Print it for the first/last layer; a zero or NaN means dead weights or a bad
  checkpoint load.
- Do not replace ``mean(abs(W))`` with ``detach()``: that would remove the
  (small) gradient path through the scale and change QAT behavior.
- ``compute_scale`` raises on an unknown mode. A silent fallback would make a
  config typo look like a successful experiment.
"""

from __future__ import annotations

import torch

SCALE_MODES = ("per_tensor_mean_abs",)


def per_tensor_mean_abs(weight: torch.Tensor) -> torch.Tensor:
    """``alpha = mean(|W|)`` over every element of the tensor.

    Returns a 0-D tensor (one scalar per weight matrix). Keeping it as a tensor
    (not ``float(...)``) is what lets autograd flow through the scale.
    """
    return weight.abs().mean()


_SCALE_FNS = {
    "per_tensor_mean_abs": per_tensor_mean_abs,
}


def compute_scale(weight: torch.Tensor, mode: str = "per_tensor_mean_abs") -> torch.Tensor:
    """Compute the scalar scale for ``weight`` under the named strategy.

    Raises ``ValueError`` for an unknown mode rather than silently falling back;
    this is intentional so a config typo cannot masquerade as a valid arm.
    """
    if mode not in _SCALE_FNS:
        raise ValueError(f"Unknown scale mode {mode!r}. Available: {list(SCALE_MODES)}")
    return _SCALE_FNS[mode](weight)
