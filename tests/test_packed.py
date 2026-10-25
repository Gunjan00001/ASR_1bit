"""Packed 1-bit inference tests (Phase 18). Tiny tensors only."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from onebit_asr.quantization.bitlinear import BitLinear
from onebit_asr.quantization.packed import (
    PackedBitLinear,
    pack_model,
    pack_sign_bits,
    packed_model_size_report,
    unpack_sign_bits,
    xnor_popcount_packed,
    xnor_similarity_to_dot,
)


def _nonzero_weight(shape=(3, 4), seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(*shape)
    weight[weight == 0] = 1.0
    return weight


def test_pack_unpack_roundtrip_is_sign():
    weight = _nonzero_weight()
    packed = pack_sign_bits(weight)
    unpacked = unpack_sign_bits(packed, weight.numel(), tuple(weight.shape))
    expected = torch.sign(weight).numpy()
    assert np.array_equal(unpacked, expected)
    assert packed.nbytes == int(np.ceil(weight.numel() / 8))


def test_xnor_popcount_matches_float_dot():
    rng = np.random.default_rng(0)
    n_bits, m, k = 37, 5, 7
    x_bits = rng.integers(0, 2, size=(m, n_bits), dtype=np.uint8)
    w_bits = rng.integers(0, 2, size=(k, n_bits), dtype=np.uint8)
    sim = xnor_popcount_packed(np.packbits(x_bits, axis=1),
                               np.packbits(w_bits, axis=1), n_bits)
    got = xnor_similarity_to_dot(sim, n_bits)
    expected = ((x_bits.astype(np.float32) * 2 - 1)
                @ (w_bits.astype(np.float32) * 2 - 1).T).astype(np.int32)
    assert np.array_equal(got, expected)


def test_packed_bitlinear_matches_bitlinear():
    torch.manual_seed(0)
    bit = BitLinear(4, 3, bias=True)
    with torch.no_grad():
        bit.weight.copy_(_nonzero_weight((3, 4)))
        bit.bias.copy_(torch.randn(3))
    packed = PackedBitLinear.from_bitlinear(bit)

    x = torch.randn(5, 4)
    with torch.no_grad():
        assert torch.allclose(bit(x), packed(x), atol=1e-6)
    assert packed.packed.numel() == int(np.ceil(bit.weight.numel() / 8))


def test_pack_model_swaps_only_bitlinear_and_preserves_output():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = BitLinear(4, 3)
            self.b = nn.Linear(3, 2)

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))

    torch.manual_seed(1)
    model = _Model()
    x = torch.randn(6, 4)
    with torch.no_grad():
        before = model(x)

    _, info = pack_model(model)
    assert isinstance(model.a, PackedBitLinear)
    assert type(model.b) is nn.Linear
    assert info["n_packed_layers"] == 1
    assert info["packed_weight_bytes"] == int(np.ceil(4 * 3 / 8))

    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_packed_model_size_report_counts_storage_not_masters():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = BitLinear(8, 4)
            self.b = nn.Linear(4, 2)

    model = _Model()
    pack_model(model)
    report = packed_model_size_report(model)
    assert report["n_packed_layers"] == 1
    assert report["packed_weight_bytes"] == int(np.ceil(8 * 4 / 8))
    assert report["fp_scale_and_bias_bytes"] == 4 + 4 * 4  # alpha fp32 + 4 fp32 biases
    assert report["non_packed_fp_param_bytes"] == 4 * 2 * 4 + 2 * 4  # b.weight + b.bias


# --- Per-output-channel packed / artifact path -----------------------------

PER_CHANNEL = "per_output_channel_mean_abs"


def test_packed_per_channel_matches_bitlinear():
    torch.manual_seed(0)
    bit = BitLinear(4, 3, bias=True, scale_mode=PER_CHANNEL)
    with torch.no_grad():
        bit.weight.copy_(_nonzero_weight((3, 4)))
        bit.bias.copy_(torch.randn(3))
    packed = PackedBitLinear.from_bitlinear(bit, scale_mode=PER_CHANNEL)
    x = torch.randn(5, 4)
    with torch.no_grad():
        assert torch.allclose(bit(x), packed(x), atol=1e-6)
    assert packed.alpha.shape == (3, 1)


def test_pack_model_per_channel_preserves_output():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = BitLinear(4, 3, scale_mode=PER_CHANNEL)
            self.b = nn.Linear(3, 2)

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))

    torch.manual_seed(1)
    model = _Model()
    x = torch.randn(6, 4)
    with torch.no_grad():
        before = model(x)
    _, info = pack_model(model, scale_mode=PER_CHANNEL)
    assert info["scale_mode"] == PER_CHANNEL
    assert model.a.alpha.shape == (3, 1)
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_artifact_tensors_roundtrip_per_channel():
    from onebit_asr.evaluation.model_size import build_packed_tensors

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = BitLinear(4, 3, scale_mode=PER_CHANNEL)
            self.b = nn.Linear(3, 2)

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))

    torch.manual_seed(2)
    model = _Model()
    x = torch.randn(6, 4)
    with torch.no_grad():
        expected = model(x)

    tensors, stats = build_packed_tensors(model, scale_mode=PER_CHANNEL)
    assert stats["scale_mode"] == PER_CHANNEL
    assert stats["fp_scale_element_count"] == 3          # one per output row
    assert stats["fp_scale_bytes"] == 3 * 4
    assert tensors["a.weight_scale"].shape == (3, 1)
    assert stats["effective_bits_per_quantized_param"] > 1.0

    # Rebuild just the BitLinear from artifact tensors (the non-quantized
    # `b` is loaded separately in the real script).
    rebuilt = PackedBitLinear.from_artifact(
        tensors["a.weight_packed"], (3, 4), tensors["a.weight_scale"], tensors["a.bias"]
    )
    with torch.no_grad():
        got = model.b(torch.relu(rebuilt(x)))
    assert torch.allclose(expected, got, atol=1e-6)
