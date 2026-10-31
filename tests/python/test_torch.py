"""Optional PyTorch CPU tensor support.

These tests are skipped when PyTorch is not installed. TensorTrail never imports
torch unless the caller supplies a tensor. Torch is imported inside each test
rather than at collection time so the rest of the suite is unaffected by
platform-specific OpenMP/MKL interactions between PyTorch and NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import tensortrail as tt
from tests.conftest import requires_torch

pytestmark = requires_torch


def _torch() -> Any:
    import torch

    return torch


def _config(**overrides: object) -> tt.ComparisonConfig:
    values: dict[str, object] = {"backend": "numpy"}
    values.update(overrides)
    return tt.ComparisonConfig(**values)  # type: ignore[arg-type]


def test_float32_cpu_tensor() -> None:
    torch = _torch()
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    candidate = torch.tensor([1.0, 4.0, 3.0], dtype=torch.float32)
    report = tt.compare(reference, candidate, config=_config())
    assert report.reference.source == "torch"
    assert report.reference.device == "cpu"
    assert report.reference.dtype == "float32"
    assert report.reference.working_dtype == "float32"
    assert report.metrics.mae == pytest.approx(2 / 3)


def test_float64_tensor() -> None:
    torch = _torch()
    values = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64)
    report = tt.compare(values, values.clone(), config=_config())
    assert report.reference.dtype == "float64"
    assert report.metrics.mae == 0.0


@pytest.mark.parametrize(
    "dtype_name,note",
    [
        ("int64", "torch_int64_to_float64"),
        ("int32", "torch_int32_to_float64"),
        ("uint8", "torch_uint8_to_float64"),
        ("bool", "torch_bool_to_float64"),
    ],
)
def test_integer_and_bool_tensors(dtype_name: str, note: str) -> None:
    torch = _torch()
    dtype = getattr(torch, dtype_name)
    values = torch.tensor([1, 0, 1], dtype=dtype)
    report = tt.compare(values, values, config=_config())
    assert note in report.reference.conversion_notes
    assert report.reference.working_dtype == "float64"


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_half_precision_tensors_are_widened(dtype_name: str) -> None:
    torch = _torch()
    values = torch.tensor([1.0, 2.0, 3.0], dtype=getattr(torch, dtype_name))
    report = tt.compare(values, values, config=_config())
    assert f"torch_{dtype_name}_to_float32" in report.reference.conversion_notes
    assert report.reference.working_dtype == "float32"


def test_autograd_tensor_is_detached_and_recorded() -> None:
    torch = _torch()
    reference = torch.tensor([1.0, 2.0], requires_grad=True)
    candidate = torch.tensor([1.0, 3.0])
    report = tt.compare(reference, candidate, config=_config())
    assert "torch_detached_from_autograd" in report.reference.conversion_notes
    assert report.metrics.mae == pytest.approx(0.5)


def test_non_contiguous_tensor_is_made_contiguous() -> None:
    torch = _torch()
    base = torch.arange(24, dtype=torch.float64).reshape(4, 6)
    transposed = base.t()
    assert not transposed.is_contiguous()
    report = tt.compare(transposed, transposed.clone(), config=_config())
    assert report.reference.shape == (6, 4)
    assert "torch_non_contiguous_to_contiguous" in report.reference.conversion_notes
    assert report.metrics.mae == 0.0


def test_mixed_torch_and_numpy_inputs() -> None:
    torch = _torch()
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    candidate = np.array([1.0, 3.0], dtype=np.float32)
    report = tt.compare(reference, candidate, config=_config())
    assert report.reference.source == "torch"
    assert report.candidate.source == "numpy"
    assert report.metadata.working_dtype == "float32"
    assert report.metrics.mae == pytest.approx(0.5)


def test_complex_tensor_is_rejected() -> None:
    torch = _torch()
    values = torch.tensor([1 + 2j], dtype=torch.complex64)
    with pytest.raises(tt.UnsupportedDtypeError, match="complex"):
        tt.compare(values, values.clone(), config=_config())


def test_quantized_tensor_is_rejected() -> None:
    torch = _torch()
    values = torch.quantize_per_tensor(torch.arange(4, dtype=torch.float32), 0.1, 0, torch.quint8)
    with pytest.raises(tt.UnsupportedDtypeError, match="quantized"):
        tt.compare(values, values.clone(), config=_config())


def test_sparse_tensor_is_rejected() -> None:
    torch = _torch()
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.sparse_coo_tensor(indices, torch.tensor([1.0, 2.0]), (2, 2))
    with pytest.raises(tt.UnsupportedDtypeError, match="dense strided"):
        tt.compare(values, values.clone(), config=_config())


def test_meta_tensor_is_rejected_without_silent_transfer() -> None:
    torch = _torch()
    values = torch.empty(3, device="meta")
    with pytest.raises(tt.BackendError, match="does not silently copy"):
        tt.compare(values, values, config=_config())


def test_cuda_tensor_is_rejected_without_silent_transfer() -> None:
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    values = torch.ones(4, device="cuda")
    with pytest.raises(tt.BackendError, match="does not silently copy"):
        tt.compare(values, values, config=_config())
