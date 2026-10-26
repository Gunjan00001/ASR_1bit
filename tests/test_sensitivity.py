"""Tests for the weight reconstruction sensitivity helper."""

from __future__ import annotations

import torch

from onebit_asr.evaluation.sensitivity import weight_reconstruction_error


def test_binary_weight_has_zero_reconstruction_error():
    torch.manual_seed(0)
    w = torch.sign(torch.randn(5, 7))
    w[w == 0] = 1.0
    assert weight_reconstruction_error(w) < 1e-6
    assert weight_reconstruction_error(w, scale_mode="per_output_channel_mean_abs") < 1e-6


def test_zero_weight_is_defined_as_zero_error():
    assert weight_reconstruction_error(torch.zeros(4, 3)) == 0.0


def test_per_channel_improves_a_row_scaled_matrix():
    # Each row is a scalar multiple of a sign vector: one row scale is exact,
    # one tensor scale is not.
    torch.manual_seed(1)
    signs = torch.sign(torch.randn(6, 8))
    signs[signs == 0] = 1.0
    row_scales = torch.linspace(0.1, 2.0, 6).view(6, 1)
    w = signs * row_scales

    tensor_err = weight_reconstruction_error(w, scale_mode="per_tensor_mean_abs")
    channel_err = weight_reconstruction_error(w, scale_mode="per_output_channel_mean_abs")
    assert channel_err < 1e-6
    assert tensor_err > channel_err


def test_error_is_finite_and_bounded_for_random_weights():
    torch.manual_seed(2)
    w = torch.randn(16, 32)
    err = weight_reconstruction_error(w)
    assert 0.0 <= err < 1.0
    assert err == err  # not NaN
