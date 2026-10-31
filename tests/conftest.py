"""Shared pytest configuration."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import tensortrail as tt

NATIVE_AVAILABLE = tt.is_native_available()
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

requires_native = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="compiled TensorTrail native extension is not available",
)
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")

BACKEND_PARAMS = [
    pytest.param("numpy", id="numpy"),
    pytest.param("rust", id="rust", marks=requires_native),
]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261030)
