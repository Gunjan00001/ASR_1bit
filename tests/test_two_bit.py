"""Genuine 2-bit quantization, packing, and reload tests."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from torch import nn

from onebit_asr.evaluation.model_size import build_packed_tensors, write_packed_artifact
from onebit_asr.quantization.bitlinear import BitLinear
from onebit_asr.quantization.binarization import quantize_weight
from onebit_asr.quantization.packed import PackedBitLinear
from onebit_asr.quantization.two_bit import (
    LEVELS,
    codes_for_weight,
    pack_2bit_codes,
    packed_2bit_bytes,
    unpack_2bit_codes,
    validate_bits,
)


def _weight(shape=(3, 4), seed=0):
    torch.manual_seed(seed)
    w = torch.randn(*shape)
    w[w == 0] = 0.25  # avoid the exact-zero boundary for this fixture
    return w


def test_codebook_has_four_signed_levels():
    assert tuple(LEVELS) == (-1.5, -0.5, 0.5, 1.5)
    assert len(set(LEVELS)) == 4
    assert validate_bits(1) == 1 and validate_bits(2) == 2
    with pytest.raises(ValueError, match="bits must"):
        validate_bits(3)


def test_midrise_boundaries_and_codes():
    w = torch.tensor([[-2.0, -1.01, -1.0, -0.99, 0.0, 0.99, 1.0, 1.01, 2.0]])
    alpha = torch.tensor(1.0)
    codes = codes_for_weight(w, alpha)
    assert codes.tolist() == [[0, 0, 1, 1, 2, 2, 3, 3, 3]]
    q = ((torch.from_numpy(codes).float() - 1.5) * alpha)
    assert q.tolist() == [[-1.5, -1.5, -0.5, -0.5, 0.5, 0.5, 1.5, 1.5, 1.5]]


def test_2bit_quantize_reconstruction_and_shape():
    w = _weight()
    wq, alpha = quantize_weight(w, bits=2)
    assert wq.shape == w.shape
    assert alpha.dim() == 0
    assert torch.isfinite(wq).all()
    allowed = torch.tensor(LEVELS) * alpha
    assert set(wq.flatten().tolist()) <= set(allowed.tolist())


def test_2bit_per_channel_scale_and_zero_row_is_safe():
    w = torch.tensor([[1.0, -2.0], [0.0, 0.0], [3.0, 4.0]])
    wq, alpha = quantize_weight(w, "per_output_channel_mean_abs", bits=2)
    assert alpha.shape == (3, 1)
    assert alpha[1].item() == 0.0
    assert torch.equal(wq[1], torch.zeros(2))
    assert torch.isfinite(wq).all()
    codes = codes_for_weight(w, alpha)
    assert codes[1].tolist() == [0, 0]


def test_2bit_zero_scale_returns_zero_not_nan():
    w = torch.zeros(3, 4)
    wq, alpha = quantize_weight(w, bits=2)
    assert alpha.item() == 0.0
    assert torch.equal(wq, torch.zeros_like(w))
    assert torch.isfinite(wq).all()


def test_2bit_gradient_reaches_master():
    w = _weight().clone().requires_grad_(True)
    wq, alpha = quantize_weight(w, bits=2)
    wq.sum().backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert w.grad.abs().sum() > 0


def test_2bit_bitlinear_matches_manual_levels():
    torch.manual_seed(0)
    bit = BitLinear(4, 3, bias=True, bits=2)
    with torch.no_grad():
        bit.weight.copy_(_weight((3, 4)))
        bit.bias.copy_(torch.randn(3))
    alpha = bit.weight.abs().mean()
    levels = torch.tensor(LEVELS)
    q = levels[torch.clamp(torch.floor(bit.weight / alpha) + 2, 0, 3).long()]
    expected = torch.nn.functional.linear(torch.randn(5, 4), alpha * q, bit.bias)
    x = torch.randn(5, 4)
    with torch.no_grad():
        got = bit(x)
    # Compare against the same forward computed with the same x.
    with torch.no_grad():
        expected = torch.nn.functional.linear(x, alpha * q, bit.bias)
    assert torch.allclose(got, expected, atol=1e-6)
    assert bit.bits == 2


def test_2bit_pack_unpack_roundtrip():
    codes = np.arange(24, dtype=np.uint8) % 4
    packed = pack_2bit_codes(codes)
    assert packed.nbytes == packed_2bit_bytes(codes.size)
    assert np.array_equal(unpack_2bit_codes(packed, codes.size, (4, 6)),
                          codes.reshape(4, 6))
    with pytest.raises(ValueError, match="0..3"):
        pack_2bit_codes(np.array([4], dtype=np.uint8))


def test_2bit_packed_bitlinear_matches_bitlinear():
    torch.manual_seed(0)
    bit = BitLinear(4, 3, bias=True, bits=2)
    with torch.no_grad():
        bit.weight.copy_(_weight((3, 4)))
        bit.bias.copy_(torch.randn(3))
    packed = PackedBitLinear.from_bitlinear(bit)
    x = torch.randn(5, 4)
    with torch.no_grad():
        assert torch.allclose(bit(x), packed(x), atol=1e-6)
    assert packed.bits == 2
    assert packed.packed.numel() == int(np.ceil(bit.weight.numel() / 4))


def test_2bit_artifact_tensors_and_serialization(tmp_path):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = BitLinear(4, 3, bits=2)
            self.b = nn.Linear(3, 2)

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))

    torch.manual_seed(2)
    model = _Model()
    x = torch.randn(6, 4)
    with torch.no_grad():
        expected = model(x)
    tensors, stats = build_packed_tensors(model, bits=2)
    assert stats["bits"] == 2
    assert stats["packed_payload_bytes"] == int(np.ceil(12 / 4))
    assert stats["theoretical_payload_bytes"] == int(np.ceil(12 / 4))
    assert stats["quantization_levels"] == list(LEVELS)
    assert tensors["a.weight_packed"].numel() == 3

    out = tmp_path / "packed.safetensors"
    report = write_packed_artifact(model, out, bits=2)
    assert report["packed_artifact_bytes_measured"] == out.stat().st_size
    assert report["bits"] == 2

    from safetensors.torch import load_file
    artifact = load_file(str(out), device="cpu")
    rebuilt = PackedBitLinear.from_artifact(
        artifact["a.weight_packed"], (3, 4),
        artifact["a.weight_scale"], artifact["a.bias"], bits=2,
    )
    with torch.no_grad():
        got = model.b(torch.relu(rebuilt(x)))
    assert torch.allclose(expected, got, atol=1e-6)


def test_2bit_state_dict_roundtrip_preserves_forward():
    torch.manual_seed(0)
    bit = BitLinear(4, 3, bias=True, bits=2)
    with torch.no_grad():
        bit.weight.copy_(_weight((3, 4)))
        bit.bias.copy_(torch.randn(3))
    x = torch.randn(5, 4)
    with torch.no_grad():
        expected = bit(x)
    buffer = io.BytesIO()
    torch.save(bit.state_dict(), buffer)
    buffer.seek(0)
    fresh = BitLinear(4, 3, bias=True, bits=2)
    fresh.load_state_dict(torch.load(buffer))
    with torch.no_grad():
        assert torch.allclose(fresh(x), expected, atol=1e-6)
