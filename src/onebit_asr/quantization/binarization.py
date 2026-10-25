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

    Order matters: ``wb`` is derived first, but ``alpha`` is computed from
    ``weight`` (not from ``wb``). Computing it from ``wb`` would detach the
    scale from the master and change the gradient (see ``deepdive.md`` §3.4).
    """
    wb = binarize(weight, ste_clip)          # ±1 pattern; STE-backed
    alpha = compute_scale(weight, scale_mode)  # differentiable scalar
    return alpha * wb, alpha
