"""Execution validation engine behavior and status aggregation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

import tensortrail as tt
from tensortrail.validation import engine as validation_engine
from tests.conftest import requires_native


def _session(
    checkpoints: Mapping[str, object] | None = None,
    *,
    version: str = "1.0",
    finalize: bool = True,
    framework: str = "numpy",
) -> tt.ExecutionSession:
    session = tt.ExecutionSession.create(
        model_id="classifier",
        model_version=version,
        framework=framework,
        precision="float64",
    )
    for name, tensor in (checkpoints or {}).items():
        session.add_checkpoint(name=name, tensor=tensor)
    if finalize:
        session.finalize()
    return session


def _policy(**overrides: object) -> tt.ValidationPolicy:
    values: dict[str, object] = {"backend": "numpy"}
    values.update(overrides)
    return tt.ValidationPolicy(**values)  # type: ignore[arg-type]


def test_all_matching_checkpoints_pass() -> None:
    reference = _session(
        {
            "encoder.output": np.arange(6, dtype=np.float64).reshape(2, 3),
            "head.output": np.array([1.0, 2.0], dtype=np.float32),
        },
        version="1.0",
    )
    candidate = _session(
        {
            "head.output": np.array([1.0, 2.0], dtype=np.float32),
            "encoder.output": np.arange(6, dtype=np.float64).reshape(2, 3),
        },
        version="2.0",
    )
    result = tt.validate(reference, candidate, policy=_policy())
    assert isinstance(result, tt.ValidationResult)
    assert result.status == "passed"
    assert result.validation_status == "passed"
    assert result.validation_id.startswith("val_")
    assert result.total_checkpoint_count == 2
    assert result.passed_checkpoint_count == 2
    assert result.failed_checkpoint_count == 0
    assert result.missing_checkpoint_count == 0
    assert result.alignment.matched_count == 2
    assert result.highest_deviation == ("encoder.output", 0.0)
    assert "PASSED" in result.summary
    assert "matched checkpoints: 2" in result.summary


def test_numerical_difference_fails_all_aggregation() -> None:
    reference = _session({"a": np.array([0.0, 0.0]), "b": np.array([1.0, 1.0])})
    candidate = _session({"a": np.array([1.0, 1.0]), "b": np.array([1.0, 1.0])})
    result = tt.validate(reference, candidate, policy=_policy(atol=0.0, rtol=0.0))
    assert result.status == "failed"
    assert result.failed_checkpoint_count == 1
    assert result.passed_checkpoint_count == 1
    outcomes = {item.checkpoint_name: item.outcome for item in result.checkpoint_results}
    assert outcomes == {"a": "failed", "b": "passed"}


def test_any_aggregation_passes_when_one_checkpoint_passes() -> None:
    reference = _session({"a": np.array([0.0]), "b": np.array([1.0])})
    candidate = _session({"a": np.array([2.0]), "b": np.array([1.0])})
    result = tt.validate(
        reference,
        candidate,
        policy=_policy(atol=0.0, rtol=0.0, aggregation_strategy="any"),
    )
    assert result.status == "passed"
    assert result.failed_checkpoint_count == 1
    assert result.passed_checkpoint_count == 1


def test_missing_candidate_is_incomplete_by_default() -> None:
    reference = _session({"a": np.array([1.0]), "b": np.array([2.0])})
    candidate = _session({"a": np.array([1.0])})
    result = tt.validate(reference, candidate, policy=_policy())
    assert result.status == "incomplete"
    assert result.missing_checkpoint_count == 1
    missing = next(
        item for item in result.checkpoint_results if item.outcome == "missing_candidate"
    )
    assert missing.checkpoint_name == "b"
    assert missing.required is True
    assert missing.report is None
    assert "no corresponding candidate" in (missing.message or "")


def test_missing_candidate_can_be_optional() -> None:
    reference = _session({"a": np.array([1.0]), "b": np.array([2.0])})
    candidate = _session({"a": np.array([1.0])})
    result = tt.validate(
        reference,
        candidate,
        policy=_policy(require_all_checkpoints=False),
    )
    assert result.status == "passed"
    missing = next(
        item for item in result.checkpoint_results if item.outcome == "missing_candidate"
    )
    assert missing.required is False


def test_missing_candidate_can_be_a_policy_failure() -> None:
    reference = _session({"a": np.array([1.0]), "b": np.array([2.0])})
    candidate = _session({"a": np.array([1.0])})
    result = tt.validate(
        reference,
        candidate,
        policy=_policy(fail_on_missing_checkpoints=True),
    )
    assert result.status == "failed"


def test_missing_reference_is_incomplete_by_default() -> None:
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0]), "candidate.only": np.array([2.0])})
    result = tt.validate(reference, candidate, policy=_policy())
    assert result.status == "incomplete"
    missing = next(
        item for item in result.checkpoint_results if item.outcome == "missing_reference"
    )
    assert missing.checkpoint_name == "candidate.only"
    assert "no corresponding reference" in (missing.message or "")


def test_shape_mismatch_is_incomplete_by_default() -> None:
    reference = _session({"a": np.zeros((2, 2), dtype=np.float64)})
    candidate = _session({"a": np.zeros(4, dtype=np.float64)})
    result = tt.validate(reference, candidate, policy=_policy())
    assert result.status == "incomplete"
    assert result.shape_mismatch_count == 1
    item = result.checkpoint_results[0]
    assert item.outcome == "shape_mismatch"
    assert item.report is None
    assert item.reference_shape == (2, 2)
    assert item.candidate_shape == (4,)


def test_shape_mismatch_can_be_a_policy_failure() -> None:
    reference = _session({"a": np.zeros((2, 2), dtype=np.float64)})
    candidate = _session({"a": np.zeros(4, dtype=np.float64)})
    result = tt.validate(
        reference,
        candidate,
        policy=_policy(fail_on_shape_mismatch=True),
    )
    assert result.status == "failed"


def test_dtype_change_is_allowed_and_reported() -> None:
    reference = _session({"a": np.array([1.0, 2.0], dtype=np.float32)})
    candidate = _session({"a": np.array([1.0, 2.0], dtype=np.float64)})
    result = tt.validate(reference, candidate, policy=_policy())
    assert result.status == "passed"
    item = result.checkpoint_results[0]
    assert item.report is not None
    assert item.reference_dtype == "float32"
    assert item.candidate_dtype == "float64"
    assert "dtype changed" in (item.message or "")


def test_dtype_change_can_be_a_policy_failure() -> None:
    reference = _session({"a": np.array([1.0, 2.0], dtype=np.float32)})
    candidate = _session({"a": np.array([1.0, 2.0], dtype=np.float64)})
    result = tt.validate(
        reference,
        candidate,
        policy=_policy(fail_on_dtype_change=True),
    )
    assert result.status == "failed"
    assert result.dtype_mismatch_count == 1
    item = result.checkpoint_results[0]
    assert item.outcome == "dtype_mismatch"
    assert item.report is None


def test_unsupported_dtype_is_an_explicit_outcome() -> None:
    reference = _session({"a": np.array([1 + 2j], dtype=np.complex128)})
    candidate = _session({"a": np.array([1 + 2j], dtype=np.complex128)})
    result = tt.validate(reference, candidate, policy=_policy())
    assert result.status == "incomplete"
    item = result.checkpoint_results[0]
    assert item.outcome == "unsupported_dtype"
    assert item.report is None


def test_backend_failure_is_an_explicit_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0])})

    def _fail(*args: object, **kwargs: object) -> None:
        raise tt.BackendError("backend failed")

    monkeypatch.setattr(validation_engine.comparison_engine, "compare", _fail)
    result = tt.validate(reference, candidate, policy=_policy())
    assert result.status == "incomplete"
    item = result.checkpoint_results[0]
    assert item.outcome == "backend_error"
    assert item.report is None


def test_invalid_tensor_is_an_explicit_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0])})

    def _fail(*args: object, **kwargs: object) -> None:
        raise tt.TensorValidationError("cannot interpret tensor")

    monkeypatch.setattr(validation_engine.comparison_engine, "compare", _fail)
    result = tt.validate(reference, candidate, policy=_policy())
    assert result.status == "incomplete"
    item = result.checkpoint_results[0]
    assert item.outcome == "invalid_tensor"
    assert item.report is None


def test_empty_sessions_are_incomplete() -> None:
    reference = _session({})
    candidate = _session({})
    result = tt.validate(reference, candidate, policy=_policy())
    assert result.status == "incomplete"
    assert result.total_checkpoint_count == 0
    assert result.alignment.matched_count == 0


def test_validation_requires_completed_sessions() -> None:
    completed = _session({"a": np.array([1.0])})
    created = _session({}, finalize=False)
    recording = _session({"a": np.array([1.0])}, finalize=False)
    failed = _session({}, finalize=False)
    failed.fail("boom")

    for session, role in (
        (created, "reference"),
        (recording, "reference"),
        (failed, "reference"),
    ):
        with pytest.raises(tt.ExecutionStateError, match=role):
            tt.validate(session, completed, policy=_policy())
    for session, role in (
        (created, "candidate"),
        (recording, "candidate"),
        (failed, "candidate"),
    ):
        with pytest.raises(tt.ExecutionStateError, match=role):
            tt.validate(completed, session, policy=_policy())


def test_validation_rejects_wrong_argument_types() -> None:
    session = _session({"a": np.array([1.0])})
    with pytest.raises(TypeError, match="reference must be ExecutionSession"):
        tt.validate(object(), session)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="candidate must be ExecutionSession"):
        tt.validate(session, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy must be ValidationPolicy"):
        tt.validate(session, session, policy=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="alignment must be CheckpointAlignment"):
        tt.validate(
            session,
            session,
            policy=_policy(),
            alignment=object(),  # type: ignore[arg-type]
        )


def test_validation_id_can_be_supplied_and_is_validated() -> None:
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0])})
    result = tt.validate(
        reference,
        candidate,
        policy=_policy(),
        validation_id="val_fixed",
    )
    assert result.validation_id == "val_fixed"
    with pytest.raises(tt.TensorValidationError, match="validation_id"):
        tt.validate(
            reference,
            candidate,
            policy=_policy(),
            validation_id="../escape",
        )


def test_selection_validates_only_selected_checkpoints() -> None:
    reference = _session({"a": np.array([1.0]), "b": np.array([2.0])})
    candidate = _session({"a": np.array([1.0]), "b": np.array([3.0])})
    result = tt.validate(
        reference,
        candidate,
        policy=_policy(checkpoint_selection=["a"], atol=0.0, rtol=0.0),
    )
    assert result.status == "passed"
    assert result.total_checkpoint_count == 1
    assert result.alignment.selection == ("a",)
    assert result.checkpoint_results[0].checkpoint_name == "a"
    with pytest.raises(tt.AlignmentError, match="not present"):
        tt.validate(
            reference,
            candidate,
            policy=_policy(checkpoint_selection=["missing"]),
        )


def test_explicit_alignment_is_honored() -> None:
    reference = _session({"reference.name": np.array([1.0])})
    candidate = _session({"candidate.name": np.array([1.0])})
    alignment = tt.align_checkpoints(
        reference,
        candidate,
        mapping={"reference.name": "candidate.name"},
    )
    result = tt.validate(
        reference,
        candidate,
        policy=_policy(),
        alignment=alignment,
    )
    assert result.status == "passed"
    item = result.checkpoint_results[0]
    assert item.checkpoint_name == "reference.name"
    assert item.candidate_checkpoint_id == candidate.checkpoints[0].checkpoint_id
    assert result.highest_deviation == ("reference.name", 0.0)


def test_partial_explicit_alignment_is_rejected() -> None:
    reference = _session({"a": np.array([1.0]), "b": np.array([2.0])})
    candidate = _session({"a": np.array([1.0]), "b": np.array([2.0])})
    full = tt.align_checkpoints(reference, candidate)
    partial = tt.CheckpointAlignment(
        reference_execution_id=reference.execution_id,
        candidate_execution_id=candidate.execution_id,
        entries=(full.entries[0],),
    )
    with pytest.raises(tt.AlignmentError, match="does not cover"):
        tt.validate(reference, candidate, policy=_policy(), alignment=partial)


def test_alignment_selection_must_match_policy() -> None:
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0])})
    alignment = tt.align_checkpoints(reference, candidate, selection=["a"])
    with pytest.raises(tt.AlignmentError, match="selection"):
        tt.validate(reference, candidate, policy=_policy(), alignment=alignment)


def test_alignment_execution_ids_must_match_sessions() -> None:
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0])})
    full = tt.align_checkpoints(reference, candidate)
    wrong = tt.CheckpointAlignment(
        reference_execution_id=reference.execution_id,
        candidate_execution_id="exec_other",
        entries=full.entries,
        selection=full.selection,
    )
    with pytest.raises(tt.AlignmentError, match="candidate execution ID"):
        tt.validate(reference, candidate, policy=_policy(), alignment=wrong)


def test_alignment_entry_outside_selection_is_rejected() -> None:
    reference = _session({"a": np.array([1.0]), "b": np.array([2.0])})
    candidate = _session({"a": np.array([1.0]), "b": np.array([2.0])})
    full = tt.align_checkpoints(reference, candidate)
    extra = tt.CheckpointAlignment(
        reference_execution_id=reference.execution_id,
        candidate_execution_id=candidate.execution_id,
        entries=full.entries,
        selection=("a",),
    )
    with pytest.raises(tt.AlignmentError, match="outside"):
        tt.validate(
            reference,
            candidate,
            policy=_policy(checkpoint_selection=["a"]),
            alignment=extra,
        )


@requires_native
def test_rust_backend_validation_passes() -> None:
    reference = _session({"a": np.array([1.0, 2.0])})
    candidate = _session({"a": np.array([1.0, 2.0])})
    result = tt.validate(
        reference,
        candidate,
        policy=tt.ValidationPolicy(backend="rust"),
    )
    assert result.status == "passed"


def test_unavailable_rust_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0])})
    monkeypatch.setattr(validation_engine.rust_backend, "is_available", lambda: False)
    with pytest.raises(tt.BackendError, match="rust backend"):
        tt.validate(
            reference,
            candidate,
            policy=tt.ValidationPolicy(backend="rust"),
        )


def test_alignment_with_unknown_reference_checkpoint_id_raises_alignment_error() -> None:
    """An alignment entry referencing a non-existent checkpoint ID produces AlignmentError."""
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0])})
    good_entry = tt.align_checkpoints(reference, candidate).entries[0]

    # Craft an entry that points to an ID that doesn't exist in the reference session.
    bad_entry = tt.CheckpointAlignmentEntry(
        reference_checkpoint_id="chk_nonexistent000000000000000000",
        candidate_checkpoint_id=good_entry.candidate_checkpoint_id,
        reference_name=good_entry.reference_name,
        candidate_name=good_entry.candidate_name,
        match_rule="explicit",
        reference_stage=good_entry.reference_stage,
        candidate_stage=good_entry.candidate_stage,
    )
    bad_alignment = tt.CheckpointAlignment(
        reference_execution_id=reference.execution_id,
        candidate_execution_id=candidate.execution_id,
        entries=(bad_entry,),
    )
    with pytest.raises(tt.AlignmentError, match="does not exist in the reference session"):
        tt.validate(reference, candidate, policy=_policy(), alignment=bad_alignment)


def test_alignment_with_unknown_candidate_checkpoint_id_raises_alignment_error() -> None:
    """An alignment entry referencing a non-existent candidate ID produces AlignmentError."""
    reference = _session({"a": np.array([1.0])})
    candidate = _session({"a": np.array([1.0])})
    good_entry = tt.align_checkpoints(reference, candidate).entries[0]

    bad_entry = tt.CheckpointAlignmentEntry(
        reference_checkpoint_id=good_entry.reference_checkpoint_id,
        candidate_checkpoint_id="chk_nonexistent000000000000000000",
        reference_name=good_entry.reference_name,
        candidate_name=good_entry.candidate_name,
        match_rule="explicit",
        reference_stage=good_entry.reference_stage,
        candidate_stage=good_entry.candidate_stage,
    )
    bad_alignment = tt.CheckpointAlignment(
        reference_execution_id=reference.execution_id,
        candidate_execution_id=candidate.execution_id,
        entries=(bad_entry,),
    )
    with pytest.raises(tt.AlignmentError, match="does not exist in the candidate session"):
        tt.validate(reference, candidate, policy=_policy(), alignment=bad_alignment)
