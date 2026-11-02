"""Execution validation orchestration and status aggregation."""

from __future__ import annotations

from ..backends import rust_backend
from ..comparison import engine as comparison_engine
from ..core.result import ComparisonReport
from ..exceptions import (
    AlignmentError,
    BackendError,
    CheckpointNotFoundError,
    ExecutionStateError,
    TensorShapeError,
    TensorValidationError,
    UnsupportedDtypeError,
)
from ..execution.checkpoint import TensorCheckpoint
from ..execution.metadata import new_validation_id, validate_artifact_id
from ..execution.session import ExecutionSession
from .alignment import CheckpointAlignment, CheckpointAlignmentEntry, align_checkpoints
from .policy import ValidationPolicy
from .results import (
    CheckpointOutcome,
    CheckpointValidationResult,
    ValidationResult,
    ValidationStatus,
)

__all__ = ["aggregate_status", "validate"]

_INCOMPLETE_OUTCOMES = {
    "missing_reference",
    "missing_candidate",
    "shape_mismatch",
    "unsupported_dtype",
    "invalid_tensor",
    "backend_error",
}


def validate(
    reference: ExecutionSession,
    candidate: ExecutionSession,
    *,
    policy: ValidationPolicy | None = None,
    alignment: CheckpointAlignment | None = None,
    validation_id: str | None = None,
) -> ValidationResult:
    """Validate a candidate execution against a reference execution.

    Both sessions must be completed. Every matched checkpoint is compared with
    the existing numerical engine. Missing, shape-incompatible, dtype-rejected,
    and backend-failed comparisons are reported explicitly rather than treated
    as passes.
    """
    if not isinstance(reference, ExecutionSession):
        raise TypeError(f"reference must be ExecutionSession, received {type(reference).__name__}")
    if not isinstance(candidate, ExecutionSession):
        raise TypeError(f"candidate must be ExecutionSession, received {type(candidate).__name__}")
    resolved_policy = ValidationPolicy() if policy is None else policy
    if not isinstance(resolved_policy, ValidationPolicy):
        raise TypeError(
            f"policy must be ValidationPolicy, received {type(resolved_policy).__name__}"
        )
    _require_completed(reference, "reference")
    _require_completed(candidate, "candidate")

    if resolved_policy.backend == "rust" and not rust_backend.is_available():
        raise BackendError(
            "the rust backend is not available in this environment; install a matching "
            "native extension or set ValidationPolicy(backend='numpy') explicitly"
        )

    resolved_alignment = (
        alignment
        if alignment is not None
        else align_checkpoints(
            reference,
            candidate,
            selection=resolved_policy.checkpoint_selection,
        )
    )
    _validate_alignment(reference, candidate, resolved_alignment, resolved_policy)

    checkpoint_results = tuple(
        _evaluate_entry(reference, candidate, entry, resolved_policy)
        for entry in resolved_alignment.entries
    )
    status = aggregate_status(resolved_policy, checkpoint_results)

    identifier = (
        validate_artifact_id(validation_id, "validation_id")
        if validation_id is not None
        else new_validation_id()
    )
    return ValidationResult(
        validation_id=identifier,
        reference_execution_id=reference.execution_id,
        candidate_execution_id=candidate.execution_id,
        reference_metadata=reference.metadata,
        candidate_metadata=candidate.metadata,
        status=status,
        policy=resolved_policy,
        alignment=resolved_alignment,
        checkpoint_results=checkpoint_results,
    )


def aggregate_status(
    policy: ValidationPolicy,
    checkpoint_results: tuple[CheckpointValidationResult, ...],
) -> ValidationStatus:
    """Apply the documented status precedence to checkpoint outcomes."""
    structural_failure = False
    required_incomplete = False
    numerical_failure = False
    passed = False

    for result in checkpoint_results:
        if result.outcome == "dtype_mismatch":
            structural_failure = True
        if (
            result.outcome in {"missing_reference", "missing_candidate"}
            and result.required
            and policy.fail_on_missing_checkpoints
        ):
            structural_failure = True
        if result.outcome == "shape_mismatch" and result.required and policy.fail_on_shape_mismatch:
            structural_failure = True
        if result.required and result.outcome in _INCOMPLETE_OUTCOMES:
            required_incomplete = True
        if result.outcome == "failed":
            numerical_failure = True
        if result.outcome == "passed":
            passed = True

    if structural_failure:
        return "failed"
    if required_incomplete:
        return "incomplete"
    if policy.aggregation_strategy == "any" and passed:
        return "passed"
    if numerical_failure:
        return "failed"
    if passed:
        return "passed"
    return "incomplete"


def _evaluate_entry(
    reference: ExecutionSession,
    candidate: ExecutionSession,
    entry: CheckpointAlignmentEntry,
    policy: ValidationPolicy,
) -> CheckpointValidationResult:
    reference_checkpoint = (
        reference.get_checkpoint(entry.reference_checkpoint_id)
        if entry.reference_checkpoint_id is not None
        else None
    )
    candidate_checkpoint = (
        candidate.get_checkpoint(entry.candidate_checkpoint_id)
        if entry.candidate_checkpoint_id is not None
        else None
    )
    required = (
        True
        if reference_checkpoint is not None and candidate_checkpoint is not None
        else policy.require_all_checkpoints
    )

    if reference_checkpoint is None:
        assert candidate_checkpoint is not None
        return CheckpointValidationResult(
            checkpoint_name=candidate_checkpoint.checkpoint_name,
            outcome="missing_reference",
            required=required,
            candidate_checkpoint_id=candidate_checkpoint.checkpoint_id,
            candidate_shape=candidate_checkpoint.tensor_shape,
            candidate_dtype=candidate_checkpoint.tensor_dtype,
            message=(
                f"candidate checkpoint {candidate_checkpoint.checkpoint_name!r} "
                "has no corresponding reference checkpoint"
            ),
        )
    if candidate_checkpoint is None:
        return CheckpointValidationResult(
            checkpoint_name=reference_checkpoint.checkpoint_name,
            outcome="missing_candidate",
            required=required,
            reference_checkpoint_id=reference_checkpoint.checkpoint_id,
            reference_shape=reference_checkpoint.tensor_shape,
            reference_dtype=reference_checkpoint.tensor_dtype,
            message=(
                f"reference checkpoint {reference_checkpoint.checkpoint_name!r} "
                "has no corresponding candidate checkpoint"
            ),
        )

    if reference_checkpoint.tensor_shape != candidate_checkpoint.tensor_shape:
        return _matched_result(
            reference_checkpoint,
            candidate_checkpoint,
            required,
            "shape_mismatch",
            message=(
                f"checkpoint {reference_checkpoint.checkpoint_name!r} shape mismatch: "
                f"{reference_checkpoint.tensor_shape} vs "
                f"{candidate_checkpoint.tensor_shape}"
            ),
        )

    if (
        reference_checkpoint.tensor_dtype != candidate_checkpoint.tensor_dtype
        and policy.fail_on_dtype_change
    ):
        return _matched_result(
            reference_checkpoint,
            candidate_checkpoint,
            required,
            "dtype_mismatch",
            message=(
                f"checkpoint {reference_checkpoint.checkpoint_name!r} dtype changed from "
                f"{reference_checkpoint.tensor_dtype!r} to "
                f"{candidate_checkpoint.tensor_dtype!r} and the policy forbids it"
            ),
        )

    reference_payload = reference.get_tensor(reference_checkpoint.checkpoint_id)
    candidate_payload = candidate.get_tensor(candidate_checkpoint.checkpoint_id)
    config = policy.to_comparison_config()
    try:
        report = comparison_engine.compare(
            reference_payload,
            candidate_payload,
            config,
            reference_id=(f"{reference.execution_id}:{reference_checkpoint.checkpoint_name}"),
            candidate_id=(f"{candidate.execution_id}:{candidate_checkpoint.checkpoint_name}"),
        )
    except TensorShapeError as error:
        return _matched_result(
            reference_checkpoint,
            candidate_checkpoint,
            required,
            "shape_mismatch",
            message=str(error),
        )
    except UnsupportedDtypeError as error:
        return _matched_result(
            reference_checkpoint,
            candidate_checkpoint,
            required,
            "unsupported_dtype",
            message=str(error),
        )
    except BackendError as error:
        return _matched_result(
            reference_checkpoint,
            candidate_checkpoint,
            required,
            "backend_error",
            message=str(error),
        )
    except TensorValidationError as error:
        return _matched_result(
            reference_checkpoint,
            candidate_checkpoint,
            required,
            "invalid_tensor",
            message=str(error),
        )

    outcome: CheckpointOutcome = "passed" if report.status == "pass" else "failed"
    message = None
    if reference_checkpoint.tensor_dtype != candidate_checkpoint.tensor_dtype:
        message = (
            f"dtype changed from {reference_checkpoint.tensor_dtype!r} to "
            f"{candidate_checkpoint.tensor_dtype!r}; comparison was executed"
        )
    return _matched_result(
        reference_checkpoint,
        candidate_checkpoint,
        required,
        outcome,
        report=report,
        message=message,
    )


def _matched_result(
    reference_checkpoint: TensorCheckpoint,
    candidate_checkpoint: TensorCheckpoint,
    required: bool,
    outcome: CheckpointOutcome,
    *,
    report: ComparisonReport | None = None,
    message: str | None = None,
) -> CheckpointValidationResult:
    return CheckpointValidationResult(
        checkpoint_name=reference_checkpoint.checkpoint_name,
        outcome=outcome,
        required=required,
        reference_checkpoint_id=reference_checkpoint.checkpoint_id,
        candidate_checkpoint_id=candidate_checkpoint.checkpoint_id,
        reference_shape=reference_checkpoint.tensor_shape,
        candidate_shape=candidate_checkpoint.tensor_shape,
        reference_dtype=reference_checkpoint.tensor_dtype,
        candidate_dtype=candidate_checkpoint.tensor_dtype,
        report=report,
        message=message,
    )


def _require_completed(session: ExecutionSession, role: str) -> None:
    if session.status != "completed":
        raise ExecutionStateError(
            f"{role} execution session must be completed before validation; "
            f"session {session.execution_id!r} is {session.status}"
        )


def _validate_alignment(
    reference: ExecutionSession,
    candidate: ExecutionSession,
    alignment: CheckpointAlignment,
    policy: ValidationPolicy,
) -> None:
    if not isinstance(alignment, CheckpointAlignment):
        raise TypeError(
            f"alignment must be CheckpointAlignment, received {type(alignment).__name__}"
        )
    if alignment.reference_execution_id != reference.execution_id:
        raise AlignmentError(
            "alignment reference execution ID does not match the reference session"
        )
    if alignment.candidate_execution_id != candidate.execution_id:
        raise AlignmentError(
            "alignment candidate execution ID does not match the candidate session"
        )
    if alignment.selection != policy.checkpoint_selection:
        raise AlignmentError(
            "alignment checkpoint selection does not match the validation policy; "
            "align with the same selection or omit checkpoint_selection"
        )

    reference_ids: set[str] = set()
    candidate_ids: set[str] = set()
    all_reference_ids = {checkpoint.checkpoint_id for checkpoint in reference.checkpoints}
    all_candidate_ids = {checkpoint.checkpoint_id for checkpoint in candidate.checkpoints}
    selection = set(alignment.selection) if alignment.selection is not None else None
    if selection is not None:
        known_names = {
            checkpoint.checkpoint_name
            for checkpoint in (*reference.checkpoints, *candidate.checkpoints)
        }
        unknown_selection = selection - known_names
        if unknown_selection:
            raise AlignmentError(
                "alignment selection contains name(s) not present in either execution: "
                + ", ".join(repr(name) for name in sorted(unknown_selection))
            )

    for entry in alignment.entries:
        if selection is not None and not ({entry.reference_name, entry.candidate_name} & selection):
            raise AlignmentError(
                "alignment contains an entry outside the configured checkpoint selection: "
                f"{entry.checkpoint_name!r}"
            )
        if entry.reference_checkpoint_id is not None:
            if entry.reference_checkpoint_id in reference_ids:
                raise AlignmentError(
                    f"reference checkpoint {entry.reference_checkpoint_id!r} appears "
                    "more than once in the alignment"
                )
            reference_ids.add(entry.reference_checkpoint_id)
            try:
                checkpoint = reference.get_checkpoint(entry.reference_checkpoint_id)
            except CheckpointNotFoundError as error:
                raise AlignmentError(
                    f"alignment references checkpoint ID {entry.reference_checkpoint_id!r} "
                    "which does not exist in the reference session"
                ) from error
            if checkpoint.checkpoint_name != entry.reference_name:
                raise AlignmentError(
                    "alignment reference checkpoint name does not match the session"
                )
            if checkpoint.stage_name != entry.reference_stage:
                raise AlignmentError(
                    "alignment reference checkpoint stage does not match the session"
                )
        if entry.candidate_checkpoint_id is not None:
            if entry.candidate_checkpoint_id in candidate_ids:
                raise AlignmentError(
                    f"candidate checkpoint {entry.candidate_checkpoint_id!r} appears "
                    "more than once in the alignment"
                )
            candidate_ids.add(entry.candidate_checkpoint_id)
            try:
                checkpoint = candidate.get_checkpoint(entry.candidate_checkpoint_id)
            except CheckpointNotFoundError as error:
                raise AlignmentError(
                    f"alignment references checkpoint ID {entry.candidate_checkpoint_id!r} "
                    "which does not exist in the candidate session"
                ) from error
            if checkpoint.checkpoint_name != entry.candidate_name:
                raise AlignmentError(
                    "alignment candidate checkpoint name does not match the session"
                )
            if checkpoint.stage_name != entry.candidate_stage:
                raise AlignmentError(
                    "alignment candidate checkpoint stage does not match the session"
                )

    if selection is None:
        if reference_ids != all_reference_ids:
            raise AlignmentError(
                "alignment does not cover every reference checkpoint; provide an "
                "explicit checkpoint_selection to validate a subset"
            )
        if candidate_ids != all_candidate_ids:
            raise AlignmentError(
                "alignment does not cover every candidate checkpoint; provide an "
                "explicit checkpoint_selection to validate a subset"
            )
    else:
        for checkpoint in reference.checkpoints:
            if (
                checkpoint.checkpoint_name in selection
                and checkpoint.checkpoint_id not in reference_ids
            ):
                raise AlignmentError(
                    f"selected reference checkpoint {checkpoint.checkpoint_name!r} "
                    "is missing from the alignment"
                )
        for checkpoint in candidate.checkpoints:
            if (
                checkpoint.checkpoint_name in selection
                and checkpoint.checkpoint_id not in candidate_ids
            ):
                raise AlignmentError(
                    f"selected candidate checkpoint {checkpoint.checkpoint_name!r} "
                    "is missing from the alignment"
                )
