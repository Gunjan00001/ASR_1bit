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

Scale granularity
-----------------
``per_tensor_mean_abs`` returns a 0-D scalar per weight matrix. The optional
``per_output_channel_mean_abs`` returns a ``[out_features, 1]`` tensor, one
scale per output row. Both broadcast against a ``[out, in]`` sign matrix, so
``binarization.quantize_weight`` and ``BitLinear`` do not need separate code
paths. The default remains per-tensor; changing the default would silently
change every historical experiment.

Debugging notes
---------------
- ``alpha`` must be finite and non-negative. For per-tensor it is 0-D; for
  per-output-channel it is ``[out,1]``. A zero or NaN scale means dead weights or
  a bad checkpoint load.
- Do not replace ``mean(abs(W))`` with ``detach()``: that would remove the
  (small) gradient path through the scale and change QAT behavior.
- ``compute_scale`` raises on an unknown mode. A silent fallback would make a
  config typo look like a successful experiment.
- A zero row yields a zero per-channel scale and a zero quantized row; the code
  never divides by the scale, so there is no NaN path.
"""

from __future__ import annotations

import torch

SCALE_MODES = ("per_tensor_mean_abs", "per_output_channel_mean_abs")


def per_tensor_mean_abs(weight: torch.Tensor) -> torch.Tensor:
    """``alpha = mean(|W|)`` over every element of the tensor.

    Returns a 0-D tensor (one scalar per weight matrix). Keeping it as a tensor
    (not ``float(...)``) is what lets autograd flow through the scale.
    """
    return weight.abs().mean()


def per_output_channel_mean_abs(weight: torch.Tensor) -> torch.Tensor:
    """``alpha_i = mean_j |W_ij|``, one scale per output row.

    Args:
        weight: 2-D tensor ``[out_features, in_features]``.

    Returns:
        ``[out_features, 1]`` tensor, so it broadcasts against the ``[out, in]``
        sign matrix and against the bias of shape ``[out]`` (via the linear
        operation, not directly).

    A zero row produces a zero scale. The forward never divides by alpha, so a
    zero row is safe; it simply contributes a zero output for that unit.
    """
    if weight.dim() != 2:
        raise ValueError(
            f"per_output_channel_mean_abs expects a 2-D weight, got shape {tuple(weight.shape)}"
        )
    return weight.abs().mean(dim=1, keepdim=True)


_SCALE_FNS = {
    "per_tensor_mean_abs": per_tensor_mean_abs,
    "per_output_channel_mean_abs": per_output_channel_mean_abs,
}


def compute_scale(weight: torch.Tensor, mode: str = "per_tensor_mean_abs") -> torch.Tensor:
    """Compute the scale for ``weight`` under the named strategy.

    Shape contract:
    - ``per_tensor_mean_abs``: 0-D scalar;
    - ``per_output_channel_mean_abs``: ``[out_features, 1]``.

    Raises ``ValueError`` for an unknown mode rather than silently falling back;
    this is intentional so a config typo cannot masquerade as a valid arm.
    """
    if mode not in _SCALE_FNS:
        raise ValueError(f"Unknown scale mode {mode!r}. Available: {list(SCALE_MODES)}")
    return _SCALE_FNS[mode](weight)
