"""Validation report serialization and reload."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ..exceptions import TensorValidationError
from ..execution.metadata import ExecutionMetadata, format_datetime, parse_datetime
from ..reporting.json_report import _atomic_write, _json_safe
from .alignment import CheckpointAlignment
from .engine import aggregate_status
from .policy import ValidationPolicy
from .results import (
    VALIDATION_SCHEMA_VERSION,
    CheckpointValidationResult,
    ValidationResult,
    ValidationStatus,
)

__all__ = [
    "load_validation_report",
    "save_validation_report",
    "validation_from_dict",
    "validation_to_dict",
    "validation_to_json",
]


def validation_to_dict(result: ValidationResult) -> dict[str, Any]:
    """Build the schema-versioned, JSON-safe validation document."""
    highest = result.highest_deviation
    document = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "artifact_type": "validation",
        "library": {
            "name": "tensortrail",
            "version": _library_version(),
        },
        "validation_id": result.validation_id,
        "created_at": format_datetime(result.created_at),
        "status": result.status,
        "reference_execution": result.reference_metadata.to_dict(),
        "candidate_execution": result.candidate_metadata.to_dict(),
        "policy": result.policy.as_dict(),
        "alignment": result.alignment.as_dict(),
        "counts": {
            "total_checkpoints": result.total_checkpoint_count,
            "matched_checkpoints": result.alignment.matched_count,
            "passed_checkpoints": result.passed_checkpoint_count,
            "failed_checkpoints": result.failed_checkpoint_count,
            "missing_checkpoints": result.missing_checkpoint_count,
            "shape_mismatch_checkpoints": result.shape_mismatch_count,
            "dtype_mismatch_checkpoints": result.dtype_mismatch_count,
            "incomplete_checkpoints": result.incomplete_checkpoint_count,
        },
        "summary": result.summary,
        "highest_observed_deviation": (
            {"checkpoint_name": highest[0], "max_abs_error": highest[1]}
            if highest is not None
            else None
        ),
        "checkpoint_results": [
            checkpoint_result.as_dict() for checkpoint_result in result.checkpoint_results
        ],
    }
    return cast(dict[str, Any], _json_safe(document))


def validation_to_json(result: ValidationResult, *, indent: int = 2) -> str:
    """Serialize a validation result to strict JSON."""
    return json.dumps(
        validation_to_dict(result),
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    )


def save_validation_report(result: ValidationResult, destination: Path) -> Path:
    """Write a validation report as strict JSON using an atomic replace."""
    return _atomic_write(destination, validation_to_json(result))


def load_validation_report(path: Path) -> ValidationResult:
    """Load a validation report from a JSON file."""
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise TensorValidationError(
            f"validation report {path} is not valid JSON: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise TensorValidationError("validation report root must be a mapping")
    return validation_from_dict(cast(Mapping[str, Any], document))


def validation_from_dict(document: Mapping[str, Any]) -> ValidationResult:
    """Rebuild a validation result from its documented JSON representation."""
    required = {
        "schema_version",
        "artifact_type",
        "library",
        "validation_id",
        "created_at",
        "status",
        "reference_execution",
        "candidate_execution",
        "policy",
        "alignment",
        "counts",
        "checkpoint_results",
    }
    allowed = required | {"summary", "highest_observed_deviation"}
    unknown = set(document) - allowed
    if unknown:
        raise TensorValidationError(
            "unknown validation report field(s): " + ", ".join(sorted(unknown))
        )
    missing = required - set(document)
    if missing:
        raise TensorValidationError(
            "missing validation report field(s): " + ", ".join(sorted(missing))
        )

    schema_version = _text(document, "schema_version")
    if schema_version != VALIDATION_SCHEMA_VERSION:
        raise TensorValidationError(
            f"unsupported validation schema version {schema_version!r}; "
            f"expected {VALIDATION_SCHEMA_VERSION!r}"
        )
    if _text(document, "artifact_type") != "validation":
        raise TensorValidationError("validation artifact_type must be 'validation'")
    library = _mapping(document, "library")
    if _text(library, "name") != "tensortrail":
        raise TensorValidationError("validation library name must be 'tensortrail'")

    reference_metadata = ExecutionMetadata.from_dict(_mapping(document, "reference_execution"))
    candidate_metadata = ExecutionMetadata.from_dict(_mapping(document, "candidate_execution"))
    policy = ValidationPolicy.from_dict(_mapping(document, "policy"))
    alignment = CheckpointAlignment.from_dict(_mapping(document, "alignment"))
    if alignment.selection != policy.checkpoint_selection:
        raise TensorValidationError(
            "validation alignment selection does not match the policy selection"
        )

    results_document = document["checkpoint_results"]
    if not isinstance(results_document, list):
        raise TensorValidationError("checkpoint_results must be a list")
    checkpoint_results = tuple(
        CheckpointValidationResult.from_dict(_checkpoint_document(item))
        for item in results_document
    )

    status = _text(document, "status")
    if status not in {"passed", "failed", "incomplete"}:
        raise TensorValidationError(f"unknown validation status {status!r}")

    result = ValidationResult(
        validation_id=_text(document, "validation_id"),
        created_at=parse_datetime(document["created_at"], "created_at"),
        reference_execution_id=reference_metadata.execution_id,
        candidate_execution_id=candidate_metadata.execution_id,
        reference_metadata=reference_metadata,
        candidate_metadata=candidate_metadata,
        status=cast(ValidationStatus, status),
        policy=policy,
        alignment=alignment,
        checkpoint_results=checkpoint_results,
    )
    expected_status = aggregate_status(policy, checkpoint_results)
    if expected_status != result.status:
        raise TensorValidationError(
            f"validation status {result.status!r} does not match aggregated "
            f"checkpoint outcomes {expected_status!r}"
        )
    _verify_counts(result, _mapping(document, "counts"))
    return result


def _verify_counts(result: ValidationResult, counts: Mapping[str, Any]) -> None:
    expected = {
        "total_checkpoints": result.total_checkpoint_count,
        "matched_checkpoints": result.alignment.matched_count,
        "passed_checkpoints": result.passed_checkpoint_count,
        "failed_checkpoints": result.failed_checkpoint_count,
        "missing_checkpoints": result.missing_checkpoint_count,
        "shape_mismatch_checkpoints": result.shape_mismatch_count,
        "dtype_mismatch_checkpoints": result.dtype_mismatch_count,
        "incomplete_checkpoints": result.incomplete_checkpoint_count,
    }
    unknown = set(counts) - set(expected)
    if unknown:
        raise TensorValidationError(
            "unknown validation count field(s): " + ", ".join(sorted(unknown))
        )
    for name, expected_value in expected.items():
        actual = counts.get(name)
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TensorValidationError(f"validation count {name!r} must be an integer")
        if actual != expected_value:
            raise TensorValidationError(
                f"validation count {name!r} is {actual}, expected {expected_value}"
            )


def _checkpoint_document(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TensorValidationError("checkpoint_results entries must be mappings")
    return cast(Mapping[str, Any], value)


def _mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document[key]
    if not isinstance(value, Mapping):
        raise TensorValidationError(f"{key} must be a mapping")
    return cast(Mapping[str, Any], value)


def _text(document: Mapping[str, Any], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise TensorValidationError(f"{key} must be a string")
    return value


def _library_version() -> str:
    from .._version import __version__

    return __version__
