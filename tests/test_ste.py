"""Straight-Through Estimator tests (Phase 6 / Phase 7)."""

import pytest
import torch

from onebit_asr.quantization.ste import binary_quantize


def test_forward_is_sign():
    x = torch.tensor([[-3.0, -0.5, 0.0, 0.5, 3.0]])
    assert torch.equal(binary_quantize(x), torch.tensor([[-1.0, -1.0, 0.0, 1.0, 1.0]]))


def test_backward_is_pure_pass_through():
    x = torch.tensor([[-3.0, -0.1, 0.1, 3.0]], requires_grad=True)
    out = binary_quantize(x)
    out.sum().backward()
    # Upstream grad is 1 for every element -> pass-through gives all ones.
    assert torch.equal(x.grad, torch.ones_like(x))


def test_backward_preserves_upstream_values():
    x = torch.tensor([[-2.0, 2.0]], requires_grad=True)
    binary_quantize(x).backward(torch.tensor([[5.0, -7.0]]))
    assert torch.equal(x.grad, torch.tensor([[5.0, -7.0]]))


def test_clipped_ste_zeroes_gradients_outside_clip():
    x = torch.tensor([[-2.0, -0.5, 0.5, 2.0]], requires_grad=True)
    out = binary_quantize(x, clip_value=1.0)
    out.sum().backward()
    # |x| <= 1 keeps gradient; |x| > 1 is zeroed.
    assert torch.equal(x.grad, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))


def test_clip_value_must_be_positive():
    x = torch.zeros(2)
    with pytest.raises(ValueError, match="clip_value must be positive"):
        binary_quantize(x, clip_value=0.0)
    with pytest.raises(ValueError, match="clip_value must be positive"):
        binary_quantize(x, clip_value=-1.0)


def test_gradients_flow_to_input_scalar():
    x = torch.tensor([0.3], requires_grad=True)
    binary_quantize(x).backward()
    assert x.grad is not None and x.grad.item() == 1.0
