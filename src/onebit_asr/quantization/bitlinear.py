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

Debugging checklist
-------------------
1. ``type(layer) is BitLinear`` and the surrounding model still has the same
   ``state_dict`` keys as the original ``nn.Linear``.
2. ``layer.weight`` is FP (not ±1). If it is ±1, something overwrote the master.
3. ``layer.effective_weight().abs().mean()`` is close to
   ``layer.weight.abs().mean()`` (both should be ~alpha).
4. ``layer.weight.grad`` is not ``None`` after a backward pass. If it is, the
   layer is frozen, detached, or outside the loss graph.
5. Under fp16 autocast, ``layer.weight`` stays fp32 (the masters are fp32); the
   *derived* ``wq`` follows the autocast dtype. This is expected.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from onebit_asr.quantization.binarization import quantize_weight
from onebit_asr.quantization.two_bit import validate_bits


class BitLinear(nn.Linear):
    """Quantized linear layer (1-bit binary or 2-bit symmetric) with FP masters.

    ``bits=1`` is the original BitLinear. ``bits=2`` uses the symmetric
    mid-rise four-level codebook; both keep FP32 masters and quantize inside
    ``forward``.
    """

    #: Marker used for isinstance-style identification without importing this class.
    #: `model_size.build_packed_tensors` and `packed.pack_model` rely on it.
    is_bitlinear = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale_mode: str = "per_tensor_mean_abs",
        ste_clip: float | None = None,
        bits: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        # `super().__init__` creates `weight` (FP master) and `bias` with exactly
        # the same parameter names/shapes as nn.Linear. The two config fields are
        # plain Python attributes, not parameters, so they do not enter the
        # optimizer or the checkpoint state_dict.
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.scale_mode = scale_mode
        self.ste_clip = ste_clip
        self.bits = validate_bits(bits)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        scale_mode: str = "per_tensor_mean_abs",
        ste_clip: float | None = None,
        bits: int = 1,
    ) -> "BitLinear":
        """Create a BitLinear that numerically copies ``linear``'s FP weights.

        The copy is intentional and lossless: the new layer starts from the exact
        FP32 values of the layer it replaces. Quantization happens later, inside
        ``forward``. Do not copy binary values here.
        """
        bit = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            scale_mode=scale_mode,
            ste_clip=ste_clip,
            bits=bits,
            # Preserve device/dtype: a model loaded in fp16 or on CUDA must not
            # silently get a CPU/fp32 layer.
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            # `copy_` (not assignment) preserves the Parameter object and dtype.
            bit.weight.copy_(linear.weight)
            if linear.bias is not None:
                bit.bias.copy_(linear.bias)
        return bit

    def effective_weight(self) -> torch.Tensor:
        """The quantized weight ``alpha * sign(W)`` actually used in forward.

        This is a *derived* tensor for inspection/debugging. It is not stored,
        not registered as a parameter, and not saved in the checkpoint. Calling
        it in a loop under ``torch.enable_grad()`` builds a graph each time;
        wrap it in ``torch.no_grad()`` for pure inspection.
        """
        wq, _alpha = quantize_weight(
            self.weight, self.scale_mode, self.ste_clip, bits=self.bits
        )
        return wq

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # Recompute every call. This is deliberate: QAT needs the quantized view
        # to follow the live master weights. Caching `wq` would freeze the
        # quantization after the first step and silently turn QAT into PTQ.
        wq, _alpha = quantize_weight(
            self.weight, self.scale_mode, self.ste_clip, bits=self.bits
        )
        return F.linear(input, wq, self.bias)

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, scale_mode={self.scale_mode}, "
            f"ste_clip={self.ste_clip}, bits={self.bits}"
        )
