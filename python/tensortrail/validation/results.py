"""Execution validation result contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from ..core.result import ComparisonReport
from ..exceptions import TensorValidationError
from ..execution.metadata import (
    ExecutionMetadata,
    ensure_utc,
    new_validation_id,
    utc_now,
    validate_artifact_id,
    validate_identifier,
)
from .alignment import CheckpointAlignment
from .policy import ValidationPolicy

__all__ = [
    "VALIDATION_SCHEMA_VERSION",
    "CheckpointOutcome",
    "CheckpointValidationResult",
    "ValidationResult",
    "ValidationStatus",
]

VALIDATION_SCHEMA_VERSION = "1.0"

CheckpointOutcome = Literal[
    "passed",
    "failed",
    "missing_reference",
    "missing_candidate",
    "shape_mismatch",
    "dtype_mismatch",
    "unsupported_dtype",
    "invalid_tensor",
    "backend_error",
]
ValidationStatus = Literal["passed", "failed", "incomplete"]

_CHECKPOINT_OUTCOMES = {
    "passed",
    "failed",
    "missing_reference",
    "missing_candidate",
    "shape_mismatch",
    "dtype_mismatch",
    "unsupported_dtype",
    "invalid_tensor",
    "backend_error",
}
_VALIDATION_STATUSES = {"passed", "failed", "incomplete"}
_COMPARED_OUTCOMES = {"passed", "failed"}


def _strict_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TensorValidationError(f"{field_name} must be a bool")
    return value


def _optional_shape(value: object, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TensorValidationError(f"{field_name} must be a tuple or list of integers")
    dimensions: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise TensorValidationError(f"{field_name} must contain non-negative integers")
        dimensions.append(item)
    return tuple(dimensions)


def _optional_dtype(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_identifier(value, field_name).lower().removeprefix("torch.")


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointValidationResult:
    """Outcome of one aligned checkpoint pair or one missing checkpoint."""

    checkpoint_name: str
    outcome: CheckpointOutcome
    required: bool
    reference_checkpoint_id: str | None = None
    candidate_checkpoint_id: str | None = None
    reference_shape: tuple[int, ...] | None = None
    candidate_shape: tuple[int, ...] | None = None
    reference_dtype: str | None = None
    candidate_dtype: str | None = None
    report: ComparisonReport | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_name",
            validate_identifier(self.checkpoint_name, "checkpoint_name"),
        )
        if self.outcome not in _CHECKPOINT_OUTCOMES:
            raise TensorValidationError(f"unknown checkpoint outcome {self.outcome!r}")
        object.__setattr__(self, "required", _strict_bool(self.required, "required"))
        if self.reference_checkpoint_id is not None:
            object.__setattr__(
                self,
                "reference_checkpoint_id",
                validate_artifact_id(self.reference_checkpoint_id, "reference_checkpoint_id"),
            )
        if self.candidate_checkpoint_id is not None:
            object.__setattr__(
                self,
                "candidate_checkpoint_id",
                validate_artifact_id(self.candidate_checkpoint_id, "candidate_checkpoint_id"),
            )
        object.__setattr__(
            self,
            "reference_shape",
            _optional_shape(self.reference_shape, "reference_shape"),
        )
        object.__setattr__(
            self,
            "candidate_shape",
            _optional_shape(self.candidate_shape, "candidate_shape"),
        )
        object.__setattr__(
            self,
            "reference_dtype",
            _optional_dtype(self.reference_dtype, "reference_dtype"),
        )
        object.__setattr__(
            self,
            "candidate_dtype",
            _optional_dtype(self.candidate_dtype, "candidate_dtype"),
        )
        if self.outcome in _COMPARED_OUTCOMES and self.report is None:
            raise TensorValidationError(
                f"checkpoint outcome {self.outcome!r} requires a comparison report"
            )
        if self.outcome not in _COMPARED_OUTCOMES and self.report is not None:
            raise TensorValidationError(
                f"checkpoint outcome {self.outcome!r} cannot carry a comparison report"
            )
        if self.report is not None:
            expected_report_status = "pass" if self.outcome == "passed" else "fail"
            if self.report.status != expected_report_status:
                raise TensorValidationError(
                    f"checkpoint outcome {self.outcome!r} disagrees with comparison "
                    f"report status {self.report.status!r}"
                )
        if self.message is not None:
            object.__setattr__(self, "message", validate_identifier(self.message, "message"))

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def numerical_failure(self) -> bool:
        return self.outcome == "failed"

    @property
    def skipped(self) -> bool:
        return self.outcome not in _COMPARED_OUTCOMES

    def as_dict(self) -> dict[str, Any]:
        from ..reporting.json_report import report_to_dict

        return {
            "checkpoint_name": self.checkpoint_name,
            "outcome": self.outcome,
            "required": self.required,
            "passed": self.passed,
            "skipped": self.skipped,
            "reference_checkpoint_id": self.reference_checkpoint_id,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "reference_shape": (
                list(self.reference_shape) if self.reference_shape is not None else None
            ),
            "candidate_shape": (
                list(self.candidate_shape) if self.candidate_shape is not None else None
            ),
            "reference_dtype": self.reference_dtype,
            "candidate_dtype": self.candidate_dtype,
            "message": self.message,
            "report": report_to_dict(self.report) if self.report is not None else None,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> CheckpointValidationResult:
        from ..reporting.json_report import report_from_dict

        required = {
            "checkpoint_name",
            "outcome",
            "required",
            "reference_checkpoint_id",
            "candidate_checkpoint_id",
            "reference_shape",
            "candidate_shape",
            "reference_dtype",
            "candidate_dtype",
            "message",
            "report",
        }
        unknown = set(document) - required - {"passed", "skipped"}
        if unknown:
            raise TensorValidationError(
                "unknown checkpoint result field(s): " + ", ".join(sorted(unknown))
            )
        missing = required - set(document)
        if missing:
            raise TensorValidationError(
                "missing checkpoint result field(s): " + ", ".join(sorted(missing))
            )
        outcome = document["outcome"]
        if not isinstance(outcome, str):
            raise TensorValidationError("checkpoint outcome must be a string")
        report_document = document["report"]
        if report_document is not None and not isinstance(report_document, Mapping):
            raise TensorValidationError("checkpoint report must be a mapping or null")
        return cls(
            checkpoint_name=_required_text(document, "checkpoint_name"),
            outcome=cast(CheckpointOutcome, outcome),
            required=document["required"],
            reference_checkpoint_id=_optional_text(document, "reference_checkpoint_id"),
            candidate_checkpoint_id=_optional_text(document, "candidate_checkpoint_id"),
            reference_shape=_optional_shape_from_document(document, "reference_shape"),
            candidate_shape=_optional_shape_from_document(document, "candidate_shape"),
            reference_dtype=_optional_text(document, "reference_dtype"),
            candidate_dtype=_optional_text(document, "candidate_dtype"),
            message=_optional_text(document, "message"),
            report=(
                report_from_dict(cast(Mapping[str, Any], report_document))
                if report_document is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationResult:
    """Execution-level validation outcome and evidence."""

    reference_execution_id: str
    candidate_execution_id: str
    reference_metadata: ExecutionMetadata
    candidate_metadata: ExecutionMetadata
    status: ValidationStatus
    policy: ValidationPolicy
    alignment: CheckpointAlignment
    checkpoint_results: tuple[CheckpointValidationResult, ...]
    validation_id: str = field(default_factory=new_validation_id)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_id",
            validate_artifact_id(self.validation_id, "validation_id"),
        )
        object.__setattr__(
            self,
            "reference_execution_id",
            validate_artifact_id(self.reference_execution_id, "reference_execution_id"),
        )
        object.__setattr__(
            self,
            "candidate_execution_id",
            validate_artifact_id(self.candidate_execution_id, "candidate_execution_id"),
        )
        if not isinstance(self.reference_metadata, ExecutionMetadata):
            raise TensorValidationError("reference_metadata must be ExecutionMetadata")
        if not isinstance(self.candidate_metadata, ExecutionMetadata):
            raise TensorValidationError("candidate_metadata must be ExecutionMetadata")
        if self.reference_metadata.execution_id != self.reference_execution_id:
            raise TensorValidationError(
                "reference_execution_id does not match reference_metadata.execution_id"
            )
        if self.candidate_metadata.execution_id != self.candidate_execution_id:
            raise TensorValidationError(
                "candidate_execution_id does not match candidate_metadata.execution_id"
            )
        if self.status not in _VALIDATION_STATUSES:
            raise TensorValidationError(f"unknown validation status {self.status!r}")
        if not isinstance(self.policy, ValidationPolicy):
            raise TensorValidationError("policy must be ValidationPolicy")
        if not isinstance(self.alignment, CheckpointAlignment):
            raise TensorValidationError("alignment must be CheckpointAlignment")
        if (
            self.alignment.reference_execution_id != self.reference_execution_id
            or self.alignment.candidate_execution_id != self.candidate_execution_id
        ):
            raise TensorValidationError(
                "alignment execution IDs do not match validation execution IDs"
            )
        object.__setattr__(self, "checkpoint_results", tuple(self.checkpoint_results))
        if not all(
            isinstance(result, CheckpointValidationResult) for result in self.checkpoint_results
        ):
            raise TensorValidationError(
                "checkpoint_results must contain CheckpointValidationResult objects"
            )
        object.__setattr__(self, "created_at", ensure_utc(self.created_at, "created_at"))

    @property
    def validation_status(self) -> ValidationStatus:
        """Alias for :attr:`status`."""
        return self.status

    @property
    def total_checkpoint_count(self) -> int:
        return len(self.checkpoint_results)

    @property
    def passed_checkpoint_count(self) -> int:
        return sum(1 for result in self.checkpoint_results if result.passed)

    @property
    def failed_checkpoint_count(self) -> int:
        return sum(1 for result in self.checkpoint_results if result.numerical_failure)

    @property
    def missing_checkpoint_count(self) -> int:
        return sum(
            1
            for result in self.checkpoint_results
            if result.outcome in {"missing_reference", "missing_candidate"}
        )

    @property
    def shape_mismatch_count(self) -> int:
        return sum(1 for result in self.checkpoint_results if result.outcome == "shape_mismatch")

    @property
    def dtype_mismatch_count(self) -> int:
        return sum(1 for result in self.checkpoint_results if result.outcome == "dtype_mismatch")

    @property
    def incomplete_checkpoint_count(self) -> int:
        return sum(1 for result in self.checkpoint_results if result.skipped)

    @property
    def highest_deviation(self) -> tuple[str, float] | None:
        """Checkpoint with the largest observed ``max_abs_error``, if any.

        This is measured evidence from completed comparisons. It is not a claim
        about which internal operation introduced a difference.
        """
        best: tuple[str, float] | None = None
        for result in self.checkpoint_results:
            report = result.report
            if report is None or report.max_abs_error is None:
                continue
            value = report.max_abs_error
            if math.isnan(value):
                continue
            if best is None or value > best[1]:
                best = (result.checkpoint_name, value)
        return best

    @property
    def summary(self) -> str:
        reference = self.reference_metadata
        candidate = self.candidate_metadata
        lines = [
            f"TensorTrail execution validation: {self.status.upper()}",
            f"  reference: {reference.model_id} {reference.model_version} "
            f"[{self.reference_execution_id}]",
            f"  candidate: {candidate.model_id} {candidate.model_version} "
            f"[{self.candidate_execution_id}]",
            f"  total checkpoints: {self.total_checkpoint_count}",
            f"  matched checkpoints: {self.alignment.matched_count}",
            f"  passed: {self.passed_checkpoint_count}",
            f"  failed: {self.failed_checkpoint_count}",
            f"  missing: {self.missing_checkpoint_count}",
            f"  shape mismatches: {self.shape_mismatch_count}",
            f"  dtype mismatches: {self.dtype_mismatch_count}",
            f"  incomplete: {self.incomplete_checkpoint_count}",
        ]
        if self.alignment.selection is not None:
            lines.append("  selection: " + ", ".join(self.alignment.selection))
        highest = self.highest_deviation
        if highest is not None:
            lines.append(
                f"  highest observed deviation: {highest[0]} (max_abs_error={highest[1]:.8g})"
            )
        lines.append(f"  overall status: {self.status.upper()}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        from .reporting import validation_to_dict

        return validation_to_dict(self)

    to_dict = as_dict

    def to_json(self, *, indent: int = 2) -> str:
        from .reporting import validation_to_json

        return validation_to_json(self, indent=indent)

    def save(self, path: str | Path) -> Path:
        """Write the validation result as strict JSON."""
        from .reporting import save_validation_report

        destination = Path(path)
        if destination.suffix and destination.suffix.lower() != ".json":
            raise ValueError(
                f"unsupported validation report extension {destination.suffix!r}; use .json"
            )
        return save_validation_report(self, destination)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ValidationResult:
        from .reporting import validation_from_dict

        return validation_from_dict(document)

    @classmethod
    def load(cls, path: str | Path) -> ValidationResult:
        from .reporting import load_validation_report

        return load_validation_report(Path(path))

    def __str__(self) -> str:
        return self.summary


def _required_text(document: Mapping[str, Any], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise TensorValidationError(f"{key} must be a string")
    return value


def _optional_text(document: Mapping[str, Any], key: str) -> str | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TensorValidationError(f"{key} must be a string or null")
    return value


def _optional_shape_from_document(
    document: Mapping[str, Any],
    key: str,
) -> tuple[int, ...] | None:
    return _optional_shape(document.get(key), key)
