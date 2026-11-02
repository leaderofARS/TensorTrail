"""Execution metadata validation, provenance, and JSON round-trips."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

import tensortrail as tt
from tensortrail.execution.metadata import (
    ExecutionMetadata,
    format_datetime,
    parse_datetime,
    validate_artifact_id,
    validate_identifier,
)


def _metadata(**overrides: object) -> ExecutionMetadata:
    values: dict[str, object] = {
        "model_id": "classifier",
        "model_version": "1.0",
        "framework": "pytorch",
        "precision": "float32",
    }
    values.update(overrides)
    return ExecutionMetadata(**values)  # type: ignore[arg-type]


def test_defaults_are_generated_and_timezone_aware() -> None:
    metadata = _metadata()
    assert metadata.execution_id.startswith("exec_")
    assert metadata.execution_timestamp.tzinfo is not None
    assert metadata.execution_timestamp.utcoffset() == timedelta(0)
    assert metadata.device_type is None
    assert dict(metadata.execution_configuration) == {}
    assert dict(metadata.custom_metadata) == {}


def test_normalizes_case_and_whitespace() -> None:
    metadata = _metadata(
        framework="  PyTorch ",
        precision=" FLOAT32 ",
        device_type=" CUDA ",
        model_id=" model ",
    )
    assert metadata.framework == "pytorch"
    assert metadata.precision == "float32"
    assert metadata.device_type == "cuda"
    assert metadata.model_id == "model"


def test_timestamp_is_converted_to_utc() -> None:
    source = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=2)))
    metadata = _metadata(execution_timestamp=source)
    assert metadata.execution_timestamp == datetime(2026, 1, 2, 1, 4, 5, tzinfo=UTC)
    assert format_datetime(metadata.execution_timestamp) == "2026-01-02T01:04:05.000000Z"


@pytest.mark.parametrize(
    "field_name",
    ["model_id", "model_version", "framework", "precision"],
)
def test_required_identifiers_reject_empty_values(field_name: str) -> None:
    with pytest.raises(tt.TensorValidationError, match=field_name):
        _metadata(**{field_name: "  "})


def test_optional_identifiers_are_validated() -> None:
    assert _metadata(device_name="  cpu  ").device_name == "cpu"
    with pytest.raises(tt.TensorValidationError, match="device_name"):
        _metadata(device_name="   ")


@pytest.mark.parametrize(
    "execution_id",
    ["", "bad id", "../escape", ".hidden", "a/b", "a" * 129],
)
def test_execution_id_must_be_a_safe_path_segment(execution_id: str) -> None:
    with pytest.raises(tt.TensorValidationError, match="execution_id"):
        _metadata(execution_id=execution_id)


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(tt.TensorValidationError, match="timezone"):
        _metadata(execution_timestamp=datetime(2026, 1, 1))


def test_non_mapping_configuration_is_rejected() -> None:
    with pytest.raises(tt.TensorValidationError, match="execution_configuration"):
        _metadata(execution_configuration=["not", "a", "mapping"])


@pytest.mark.parametrize(
    "value",
    [
        {"callback": object()},
        {"shape": {1, 2}},
        {"nested": {"value": {1, 2}}},
        {"nan": float("nan")},
        {"infinity": float("inf")},
    ],
)
def test_non_json_configuration_is_rejected(value: object) -> None:
    with pytest.raises(tt.TensorValidationError, match="execution_configuration"):
        _metadata(execution_configuration=value)


def test_metadata_mappings_are_copied_and_top_level_read_only() -> None:
    source = {"nested": {"value": 1}, "items": [1, 2]}
    metadata = _metadata(custom_metadata=source)
    source["nested"] = {"value": 2}
    source["items"].append(3)
    assert metadata.custom_metadata["nested"] == {"value": 1}
    assert metadata.custom_metadata["items"] == [1, 2]
    with pytest.raises(TypeError):
        metadata.custom_metadata["new"] = "value"  # type: ignore[index]


def test_json_round_trip_is_deterministic() -> None:
    metadata = _metadata(
        execution_id="exec_roundtrip",
        model_fingerprint="sha256:abc",
        framework_version="2.5",
        device_type="cpu",
        device_name="host",
        device_identifier="cpu-0",
        input_identifier="batch-7",
        dataset_identifier="eval",
        execution_configuration={"batch_size": 8, "flags": [True, None]},
        custom_metadata={"owner": "team"},
    )
    document = metadata.to_dict()
    restored = ExecutionMetadata.from_dict(document)
    assert restored == metadata
    assert restored.to_dict() == document
    assert json.loads(json.dumps(document)) == document


def test_from_dict_rejects_unknown_and_missing_fields() -> None:
    document = _metadata().to_dict()
    document["unknown"] = 1
    with pytest.raises(tt.TensorValidationError, match="unknown execution metadata"):
        ExecutionMetadata.from_dict(document)

    incomplete = _metadata().to_dict()
    del incomplete["model_id"]
    with pytest.raises(tt.TensorValidationError, match="missing execution metadata"):
        ExecutionMetadata.from_dict(incomplete)


def test_from_dict_rejects_malformed_optional_values() -> None:
    document = _metadata().to_dict()
    document["device_name"] = 5
    with pytest.raises(tt.TensorValidationError, match="device_name"):
        ExecutionMetadata.from_dict(document)


def test_parse_datetime_requires_timezone_and_valid_text() -> None:
    assert parse_datetime("2026-01-01T00:00:00Z", "created") == datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(tt.TensorValidationError, match="created"):
        parse_datetime("2026-01-01T00:00:00", "created")
    with pytest.raises(tt.TensorValidationError, match="created"):
        parse_datetime("not-a-date", "created")
    with pytest.raises(tt.TensorValidationError, match="created"):
        parse_datetime("", "created")


def test_identifier_helpers_strip_and_reject() -> None:
    assert validate_identifier("  abc  ", "name") == "abc"
    assert validate_artifact_id("exec_1", "id") == "exec_1"
    with pytest.raises(tt.TensorValidationError, match="id"):
        validate_artifact_id("_hidden", "id")
    with pytest.raises(tt.TensorValidationError, match="id"):
        validate_artifact_id("a b", "id")


def test_metadata_is_frozen_and_slots() -> None:
    metadata = _metadata()
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.model_id = "other"  # type: ignore[misc]
    assert not hasattr(metadata, "__dict__")
