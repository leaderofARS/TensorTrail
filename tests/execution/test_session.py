"""Execution session lifecycle, payload ownership, and serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

import tensortrail as tt


def _session(**overrides: object) -> tt.ExecutionSession:
    values: dict[str, object] = {
        "model_id": "classifier",
        "model_version": "1.0",
        "framework": "numpy",
        "precision": "float64",
    }
    values.update(overrides)
    return tt.ExecutionSession.create(**values)  # type: ignore[arg-type]


def test_creation_starts_in_created_state() -> None:
    session = _session(session_metadata={"run": 1})
    assert session.status == "created"
    assert session.session_status == "created"
    assert session.checkpoint_count == 0
    assert session.checkpoints == ()
    assert session.completed_at is None
    assert session.failure_reason is None
    assert session.created_at.tzinfo is not None
    assert dict(session.session_metadata) == {"run": 1}
    assert session.execution_id == session.metadata.execution_id
    assert session.schema_version == tt.EXECUTION_SCHEMA_VERSION


def test_first_checkpoint_moves_session_to_recording() -> None:
    session = _session()
    session.add_checkpoint(name="a", tensor=np.array([1.0, 2.0]))
    assert session.status == "recording"


def test_checkpoint_indices_are_assigned_in_order() -> None:
    session = _session()
    first = session.add_checkpoint(name="a", tensor=np.array([1.0]))
    second = session.add_checkpoint(name="b", tensor=np.array([2.0]))
    third = session.add_checkpoint(name="c", tensor=np.array([3.0]), checkpoint_index=5)
    fourth = session.add_checkpoint(name="d", tensor=np.array([4.0]))
    assert first.checkpoint_index == 0
    assert second.checkpoint_index == 1
    assert third.checkpoint_index == 5
    assert fourth.checkpoint_index == 6
    assert [item.checkpoint_name for item in session.checkpoints] == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_checkpoint_metadata_is_separate_from_payload() -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    session = _session()
    checkpoint = session.add_checkpoint(
        name="encoder.output",
        tensor=array,
        stage_name="encoder",
        operation_type="matmul",
        metadata={"layer": 3},
    )
    assert checkpoint.tensor_shape == (2, 3)
    assert checkpoint.tensor_dtype == "float32"
    assert checkpoint.tensor_nbytes == 24
    assert checkpoint.tensor_source == "numpy"
    assert checkpoint.tensor_device == "cpu"
    assert checkpoint.tensor_reference == "memory"
    assert session.get_tensor(checkpoint.checkpoint_id) is array
    assert session.get_checkpoint(checkpoint.checkpoint_id) is checkpoint
    assert session.get_checkpoint_by_name("encoder.output") is checkpoint


def test_checkpoint_timestamps_are_timezone_aware() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    session = _session()
    checkpoint = session.add_checkpoint(
        name="a", tensor=np.array([1.0]), capture_timestamp=timestamp
    )
    assert checkpoint.capture_timestamp == timestamp


def test_duplicate_checkpoint_names_and_ids_are_rejected() -> None:
    session = _session()
    first = session.add_checkpoint(name="a", tensor=np.array([1.0]), checkpoint_id="cp_fixed")
    with pytest.raises(tt.CheckpointError, match="checkpoint_name"):
        session.add_checkpoint(name="a", tensor=np.array([2.0]))
    with pytest.raises(tt.CheckpointError, match="checkpoint_id"):
        session.add_checkpoint(name="b", tensor=np.array([2.0]), checkpoint_id=first.checkpoint_id)


def test_invalid_checkpoint_payload_is_rejected_without_state_change() -> None:
    session = _session()
    with pytest.raises(tt.UnsupportedDtypeError):
        session.add_checkpoint(name="bad", tensor=np.array([{"a": 1}], dtype=object))
    assert session.status == "created"
    assert session.checkpoint_count == 0


def test_finalize_transitions_to_completed() -> None:
    session = _session()
    session.add_checkpoint(name="a", tensor=np.array([1.0]))
    session.finalize()
    assert session.status == "completed"
    assert session.completed_at is not None
    assert session.failure_reason is None
    assert "status=completed" in session.summary()
    assert "checkpoints=1" in str(session)


def test_empty_session_can_be_finalized() -> None:
    session = _session()
    session.finalize()
    assert session.status == "completed"
    assert session.checkpoints == ()


def test_terminal_sessions_reject_further_transitions() -> None:
    completed = _session()
    completed.finalize()
    with pytest.raises(tt.ExecutionStateError, match="finalize"):
        completed.finalize()
    with pytest.raises(tt.ExecutionStateError, match="add checkpoint"):
        completed.add_checkpoint(name="late", tensor=np.array([1.0]))
    with pytest.raises(tt.ExecutionStateError, match="fail"):
        completed.fail("late failure")

    failed = _session()
    failed.fail("inference crashed")
    assert failed.status == "failed"
    assert failed.failure_reason == "inference crashed"
    with pytest.raises(tt.ExecutionStateError, match="fail"):
        failed.fail("again")
    with pytest.raises(tt.ExecutionStateError, match="finalize"):
        failed.finalize()
    with pytest.raises(tt.ExecutionStateError, match="add checkpoint"):
        failed.add_checkpoint(name="late", tensor=np.array([1.0]))


def test_failure_reason_must_be_non_empty() -> None:
    session = _session()
    with pytest.raises(tt.TensorValidationError, match="failure reason"):
        session.fail("   ")
    assert session.status == "created"


def test_lookup_errors_are_structured() -> None:
    session = _session()
    with pytest.raises(tt.CheckpointNotFoundError):
        session.get_checkpoint("cp_missing")
    with pytest.raises(tt.CheckpointNotFoundError):
        session.get_checkpoint_by_name("missing")


def test_as_dict_contains_metadata_but_no_tensor_payload() -> None:
    session = _session(input_identifier="sample-1")
    session.add_checkpoint(name="a", tensor=np.array([1.0, 2.0]), metadata={"key": "value"})
    session.finalize()
    document = session.as_dict()
    assert document["schema_version"] == "1.0"
    assert document["execution_id"] == session.execution_id
    assert document["session_status"] == "completed"
    assert document["metadata"]["input_identifier"] == "sample-1"
    assert document["checkpoints"][0]["checkpoint_name"] == "a"
    serialized = json.dumps(document)
    assert '"tensor":' not in serialized
    json.loads(serialized)


def test_session_save_requires_a_storage_backend() -> None:
    session = _session()
    session.finalize()
    with pytest.raises(TypeError, match="storage must implement"):
        session.save(object())  # type: ignore[arg-type]


def test_create_forwards_metadata_fields() -> None:
    session = _session(
        model_fingerprint="sha256:abc",
        framework_version="2.5",
        device_type="cpu",
        device_name="host",
        device_identifier="cpu-0",
        input_identifier="batch",
        dataset_identifier="eval",
        execution_configuration={"batch_size": 4},
        custom_metadata={"owner": "team"},
    )
    metadata = session.metadata
    assert metadata.model_fingerprint == "sha256:abc"
    assert metadata.framework_version == "2.5"
    assert metadata.device_type == "cpu"
    assert metadata.input_identifier == "batch"
    assert dict(metadata.execution_configuration) == {"batch_size": 4}
    assert dict(metadata.custom_metadata) == {"owner": "team"}
