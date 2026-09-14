"""BitLinear tests — the Phase 7 integration gate.

Covers the plan's required cases:
1. binary forward weights are ±1 before scaling
2. scaling is correct (alpha = mean(|W|))
3. output shape matches nn.Linear
4. gradient reaches the FP master weight
5. optimizer updates the master weight
6. master weight is NOT overwritten by binary values
7. save/load round-trip works
8. tiny toy task reduces loss
(+ edge cases: bias handling, dtype/device preservation, drop-in state_dict,
   explicit config reporting)
"""

import io

import torch
import torch.nn as nn

from onebit_asr.quantization.bitlinear import BitLinear


def _rand_input(batch=5, features=4):
    return torch.randn(batch, features)


def test_output_shape_matches_nn_linear(linear):
    bit = BitLinear.from_linear(linear)
    x = _rand_input()
    assert bit(x).shape == linear(x).shape


def test_binary_forward_weights_are_pm1_before_scaling(linear):
    bit = BitLinear.from_linear(linear)
    alpha = bit.weight.abs().mean()
    unit = bit.effective_weight() / alpha
    unique = set(unit.flatten().tolist())
    assert unique <= {-1.0, 1.0}, f"pre-scale weights must be ±1, got {unique}"


def test_scaling_is_mean_abs_of_master_weight(linear):
    bit = BitLinear.from_linear(linear)
    alpha = bit.weight.abs().mean()
    assert torch.allclose(bit.effective_weight(), alpha * torch.sign(bit.weight))


def test_gradient_reaches_master_weight(linear):
    bit = BitLinear.from_linear(linear)
    x = _rand_input()
    bit(x).sum().backward()
    assert bit.weight.grad is not None
    assert torch.isfinite(bit.weight.grad).all()
    assert bit.weight.grad.abs().sum() > 0


def test_optimizer_updates_master_weight(linear):
    bit = BitLinear.from_linear(linear)
    before = bit.weight.detach().clone()
    opt = torch.optim.SGD(bit.parameters(), lr=0.1)
    bit(_rand_input()).sum().backward()
    opt.step()
    assert not torch.allclose(before, bit.weight.detach()), "master weight did not update"


def test_master_weight_is_not_overwritten_by_binary_values(linear):
    bit = BitLinear.from_linear(linear)
    original = bit.weight.detach().clone()
    tiny = (original.abs() < 0.5).any().item()
    assert tiny, "fixture should contain sub-unit magnitudes to make this meaningful"
    _ = bit(_rand_input())  # run several forwards/backwards
    _ = bit.effective_weight()
    assert torch.equal(bit.weight.detach(), original), "master weight was mutated"
    # The master weight is genuinely FP, not ±1.
    assert not set(bit.weight.detach().flatten().tolist()) <= {-1.0, 1.0}


def test_bias_is_preserved_as_fp(linear):
    bit = BitLinear.from_linear(linear)
    assert bit.bias is not None
    assert torch.allclose(bit.bias.detach(), linear.bias.detach())
    assert bit.bias.requires_grad


def test_from_linear_without_bias():
    lin = nn.Linear(4, 3, bias=False)
    bit = BitLinear.from_linear(lin)
    assert bit.bias is None
    assert torch.allclose(bit.weight.detach(), lin.weight.detach())


def test_from_linear_preserves_dtype_and_device(linear):
    lin = linear.to(torch.float64)
    bit = BitLinear.from_linear(lin)
    assert bit.weight.dtype == torch.float64
    assert bit.weight.device == lin.weight.device


def test_state_dict_keys_match_nn_linear(linear):
    bit = BitLinear.from_linear(linear)
    assert set(bit.state_dict().keys()) == set(linear.state_dict().keys())


def test_save_load_roundtrip(linear):
    bit = BitLinear.from_linear(linear)
    x = _rand_input()
    expected = bit(x).detach()

    buffer = io.BytesIO()
    torch.save(bit.state_dict(), buffer)
    buffer.seek(0)

    fresh = BitLinear(linear.in_features, linear.out_features, bias=True)
    fresh.load_state_dict(torch.load(buffer))
    assert torch.allclose(fresh(x).detach(), expected, atol=1e-6)


def test_extra_repr_exposes_configuration(linear):
    bit = BitLinear.from_linear(linear, scale_mode="per_tensor_mean_abs", ste_clip=1.5)
    text = repr(bit)
    assert "per_tensor_mean_abs" in text
    assert "ste_clip=1.5" in text

    default = BitLinear.from_linear(linear)
    assert "ste_clip=None" in repr(default)


def test_forward_can_run_under_no_grad_and_eval(linear):
    bit = BitLinear.from_linear(linear).eval()
    with torch.no_grad():
        out = bit(_rand_input())
    assert torch.isfinite(out).all()


def test_toy_task_reduces_loss():
    """Target weights are themselves binary, so BitLinear can fit nearly exactly."""
    torch.manual_seed(0)
    n, d = 128, 16
    x = torch.randn(n, d)
    w_true = torch.sign(torch.randn(d, 1)) * 0.5
    y = x @ w_true

    bit = BitLinear(d, 1, bias=False)
    opt = torch.optim.SGD(bit.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        initial = loss_fn(bit(x), y).item()

    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(bit(x), y)
        loss.backward()
        opt.step()

    final = loss_fn(bit(x), y).item()
    assert final < 0.1 * initial, f"loss did not reduce enough: {initial:.4f} -> {final:.4f}"


def test_repeated_forwards_are_deterministic(linear):
    bit = BitLinear.from_linear(linear).eval()
    x = _rand_input()
    with torch.no_grad():
        a = bit(x)
        b = bit(x)
    assert torch.equal(a, b)
