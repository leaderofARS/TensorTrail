"""PyTorch metadata, persistence policy, and in-memory validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tensortrail as tt
from tensortrail.execution.checkpoint import describe_tensor
from tests.conftest import requires_torch


@requires_torch
def test_describe_torch_tensor_without_conversion() -> None:
    import torch

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    description = describe_tensor(tensor, "tensor")
    assert description.source == "torch"
    assert description.shape == (2, 3)
    assert description.dtype == "float32"
    assert description.nbytes == 24
    assert description.device == "cpu"


@requires_torch
def test_quantized_torch_tensor_is_rejected() -> None:
    import torch

    tensor = torch.quantize_per_tensor(
        torch.arange(4, dtype=torch.float32),
        scale=0.1,
        zero_point=0,
        dtype=torch.qint8,
    )
    with pytest.raises(tt.UnsupportedDtypeError, match="quantized"):
        describe_tensor(tensor, "tensor")


@requires_torch
def test_torch_payload_persistence_is_rejected(tmp_path: Path) -> None:
    import torch

    session = tt.ExecutionSession.create(
        model_id="classifier",
        model_version="1.0",
        framework="pytorch",
        precision="float32",
    )
    session.add_checkpoint(name="a", tensor=torch.arange(3, dtype=torch.float32))
    session.finalize()
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    with pytest.raises(tt.TensorValidationError, match="PyTorch"):
        storage.save_execution(session)
    assert not storage.execution_path(session.execution_id).exists()
    if storage.executions_dir.exists():
        assert not list(storage.executions_dir.glob(".*"))


@requires_torch
def test_torch_payloads_validate_in_memory() -> None:
    import torch

    reference = tt.ExecutionSession.create(
        model_id="classifier",
        model_version="1.0",
        framework="pytorch",
        precision="float32",
    )
    candidate = tt.ExecutionSession.create(
        model_id="classifier",
        model_version="2.0",
        framework="pytorch",
        precision="float32",
    )
    reference.add_checkpoint(name="a", tensor=torch.arange(4, dtype=torch.float32))
    candidate.add_checkpoint(name="a", tensor=torch.arange(4, dtype=torch.float32))
    reference.finalize()
    candidate.finalize()
    result = tt.validate(
        reference,
        candidate,
        policy=tt.ValidationPolicy(backend="numpy"),
    )
    assert result.status == "passed"
    item = result.checkpoint_results[0]
    assert item.reference_dtype == "float32"
    assert item.report is not None
    assert np.array_equal(
        reference.get_tensor(reference.checkpoints[0].checkpoint_id).numpy(),
        np.arange(4, dtype=np.float32),
    )
