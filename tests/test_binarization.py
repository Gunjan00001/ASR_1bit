"""Binarization + scaling correctness tests (Phase 5 / Phase 7)."""

import pytest
import torch

from onebit_asr.quantization.binarization import binarize, quantize_weight
from onebit_asr.quantization.scaling import compute_scale, per_tensor_mean_abs


def test_sign_values_are_pm1(weight):
    wb = binarize(weight)
    unique = set(wb.flatten().tolist())
    assert unique <= {-1.0, 1.0}, f"binary weights must be ±1, got {unique}"


def test_binarize_preserves_shape_and_dtype(weight):
    wb = binarize(weight)
    assert wb.shape == weight.shape
    assert wb.dtype == weight.dtype


def test_scale_is_mean_abs(weight):
    alpha = compute_scale(weight)
    assert torch.allclose(alpha, weight.abs().mean())
    assert torch.allclose(alpha, per_tensor_mean_abs(weight))


def test_quantized_weight_is_alpha_times_sign(weight):
    wq, alpha = quantize_weight(weight)
    assert torch.allclose(wq, alpha * torch.sign(weight))
    # And every element magnitude equals alpha.
    assert torch.allclose(wq.abs(), alpha.expand_as(wq))


def test_scale_is_scalar_per_tensor(weight):
    _wq, alpha = quantize_weight(weight)
    assert alpha.dim() == 0


def test_unknown_scale_mode_raises(weight):
    with pytest.raises(ValueError, match="Unknown scale mode"):
        quantize_weight(weight, scale_mode="per_channel_max")


def test_sign_of_zero_is_zero():
    w = torch.tensor([[-1.0, 0.0, 2.0]])
    assert torch.equal(binarize(w), torch.tensor([[-1.0, 0.0, 1.0]]))


def test_quantize_weight_is_differentiable(weight):
    w = weight.clone().requires_grad_(True)
    wq, alpha = quantize_weight(w)
    wq.sum().backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert w.grad.abs().sum() > 0
