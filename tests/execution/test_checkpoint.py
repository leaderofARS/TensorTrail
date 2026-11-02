"""Checkpoint metadata and metadata-only tensor description."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pytest

import tensortrail as tt
from tensortrail.execution.checkpoint import TensorCheckpoint, describe_tensor


def _checkpoint(**overrides: object) -> TensorCheckpoint:
    values: dict[str, object] = {
        "execution_id": "exec_1",
        "checkpoint_name": "encoder.output",
        "tensor_shape": (2, 3),
        "tensor_dtype": "float32",
        "tensor_nbytes": 24,
        "checkpoint_index": 1,
    }
    values.update(overrides)
    return TensorCheckpoint(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected_source", "expected_shape", "expected_dtype", "expected_nbytes"),
    [
        (
            np.arange(6, dtype=np.float32).reshape(2, 3),
            "numpy",
            (2, 3),
            "float32",
            24,
        ),
        ([1, 2, 3], "array-like", (3,), "int64", 24),
        (3.5, "array-like", (), "float64", 8),
    ],
)
def test_describe_tensor_metadata(
    value: object,
    expected_source: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    expected_nbytes: int,
) -> None:
    description = describe_tensor(value, "tensor")
    assert description.source == expected_source
    assert description.shape == expected_shape
    assert description.dtype == expected_dtype
    assert description.nbytes == expected_nbytes
    assert description.device == "cpu"


def test_describe_tensor_does_not_copy_a_numpy_payload() -> None:
    array = np.arange(4, dtype=np.float64)
    description = describe_tensor(array, "tensor")
    assert description.shape == array.shape
    assert description.nbytes == array.nbytes


def test_masked_array_is_rejected() -> None:
    masked = np.ma.MaskedArray([1.0, 2.0], mask=[False, True])
    with pytest.raises(tt.TensorValidationError, match="masked"):
        describe_tensor(masked, "tensor")


def test_object_array_is_rejected() -> None:
    with pytest.raises(tt.UnsupportedDtypeError, match="object array"):
        describe_tensor(np.array([{"a": 1}], dtype=object), "tensor")


def test_ragged_sequence_error_is_actionable() -> None:
    with pytest.raises(tt.TensorValidationError, match="numeric array"):
        describe_tensor([[1, 2], [3]], "tensor")


def test_checkpoint_round_trip() -> None:
    checkpoint = _checkpoint(
        stage_name="encoder",
        operation_type="matmul",
        tensor_source="numpy",
        tensor_device="cpu",
        tensor_reference="memory",
        metadata={"note": "captured"},
    )
    document = checkpoint.to_dict()
    restored = TensorCheckpoint.from_dict(document)
    assert restored == checkpoint
    assert restored.to_dict() == document
    assert restored.elements == 6


def test_checkpoint_normalizes_dtype_source_and_reference() -> None:
    checkpoint = _checkpoint(
        tensor_dtype=" torch.Float32 ",
        tensor_source=" numpy ",
        tensor_reference=" ARTIFACT ",
        tensor_device=" CPU ",
    )
    assert checkpoint.tensor_dtype == "float32"
    assert checkpoint.tensor_source == "numpy"
    assert checkpoint.tensor_reference == "artifact"
    assert checkpoint.tensor_device == "cpu"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tensor_source", "pickle"),
        ("tensor_reference", "disk"),
        ("tensor_nbytes", -1),
        ("checkpoint_index", -1),
        ("tensor_shape", (2, -3)),
        ("tensor_shape", (True,)),
        ("tensor_dtype", " "),
        ("checkpoint_id", "../escape"),
    ],
)
def test_checkpoint_rejects_invalid_metadata(field_name: str, value: object) -> None:
    with pytest.raises(tt.TensorValidationError):
        _checkpoint(**{field_name: value})


def test_checkpoint_requires_aware_timestamp() -> None:
    with pytest.raises(tt.TensorValidationError, match="timezone"):
        _checkpoint(capture_timestamp=datetime(2026, 1, 1))


def test_checkpoint_from_dict_rejects_unknown_and_missing_fields() -> None:
    document = _checkpoint().to_dict()
    document["unknown"] = 1
    with pytest.raises(tt.TensorValidationError, match="unknown checkpoint"):
        TensorCheckpoint.from_dict(document)

    incomplete = _checkpoint().to_dict()
    del incomplete["checkpoint_name"]
    with pytest.raises(tt.TensorValidationError, match="missing checkpoint"):
        TensorCheckpoint.from_dict(incomplete)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tensor_shape", "not-a-list"),
        ("tensor_shape", [2, -1]),
        ("checkpoint_index", True),
        ("metadata", ["not", "a", "mapping"]),
        ("tensor_dtype", 5),
    ],
)
def test_checkpoint_from_dict_rejects_malformed_values(field_name: str, value: object) -> None:
    document = _checkpoint().to_dict()
    document[field_name] = value
    with pytest.raises(tt.TensorValidationError):
        TensorCheckpoint.from_dict(document)


def test_checkpoint_is_frozen() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(dataclasses.FrozenInstanceError):
        checkpoint.checkpoint_name = "other"  # type: ignore[misc]
