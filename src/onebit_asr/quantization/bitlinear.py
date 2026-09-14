"""BitLinear — a drop-in 1-bit replacement for ``nn.Linear`` (Phase 4).

Design decisions
----------------
1. **Subclass of** ``nn.Linear``. This preserves the ``weight``/``bias``
   parameter names and shapes, so it is a drop-in swap for a pretrained
   checkpoint, `state_dict` keys do not change, and the stock Hugging Face
   ``Trainer`` needs no modification.
2. **FP master weights are retained.** ``self.weight`` always holds the full
   precision weights. Quantization happens *inside* ``forward()`` and only
   produces a derived tensor — the master weight is never overwritten.
3. **Quantization inside forward.** This is required by the plan (§Phase 11)
   so the stock Trainer can run QAT unmodified: forward computes
   ``alpha * sign(W)`` from the current master weights each call.
4. **Bias stays FP** unless explicitly requested otherwise (a ``bias`` of
   ``None`` is preserved too).
5. **Explicit, inspectable configuration.** ``scale_mode`` / ``ste_clip`` are
   surfaced through ``extra_repr`` so the active mode is never ambiguous.

Forward:
    W_master (FP32/FP16, never mutated)
        -> sign(W)            [STE backward]
        -> alpha * sign(W)    [alpha = mean(|W|), recomputed each forward]
        -> F.linear(input, Wq, bias)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from onebit_asr.quantization.binarization import quantize_weight


class BitLinear(nn.Linear):
    """1-bit (binary-weight) linear layer with FP master weights."""

    #: Marker used for isinstance-style identification without importing this class.
    is_bitlinear = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale_mode: str = "per_tensor_mean_abs",
        ste_clip: float | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.scale_mode = scale_mode
        self.ste_clip = ste_clip

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        scale_mode: str = "per_tensor_mean_abs",
        ste_clip: float | None = None,
    ) -> "BitLinear":
        """Create a BitLinear that numerically copies ``linear``'s FP weights."""
        bit = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            scale_mode=scale_mode,
            ste_clip=ste_clip,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            bit.weight.copy_(linear.weight)
            if linear.bias is not None:
                bit.bias.copy_(linear.bias)
        return bit

    def effective_weight(self) -> torch.Tensor:
        """The quantized weight ``alpha * sign(W)`` actually used in forward."""
        wq, _alpha = quantize_weight(self.weight, self.scale_mode, self.ste_clip)
        return wq

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        wq, _alpha = quantize_weight(self.weight, self.scale_mode, self.ste_clip)
        return F.linear(input, wq, self.bias)

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, scale_mode={self.scale_mode}, "
            f"ste_clip={self.ste_clip}"
        )
