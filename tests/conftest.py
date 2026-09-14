"""Shared fixtures for the quantization / replacement tests.

All fixtures use tiny synthetic tensors and models — the 593M-parameter
checkpoint is never loaded for unit tests (plan §Phase 7).
"""

import pytest
import torch

from onebit_asr.models.conformer_utils import categorize  # noqa: F401  (re-exported for tests)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture
def weight():
    """A small, non-symmetric FP weight matrix."""
    return torch.randn(8, 4, dtype=torch.float32)


@pytest.fixture
def linear():
    import torch.nn as nn

    lin = nn.Linear(4, 3)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(3, 4))
        lin.bias.copy_(torch.randn(3))
    return lin
