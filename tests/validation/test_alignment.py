"""Deterministic checkpoint alignment and structured missing entries."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

import tensortrail as tt


def _session(label: str, names: list[str]) -> tt.ExecutionSession:
    session = tt.ExecutionSession.create(
        model_id=f"model-{label}",
        model_version="1.0",
        framework="numpy",
        precision="float64",
    )
    for index, name in enumerate(names):
        session.add_checkpoint(
            name=name,
            tensor=np.arange(4, dtype=np.float64) + index,
            stage_name=name.split(".")[0],
            operation_type="matmul",
        )
    session.finalize()
    return session


def test_name_alignment_ignores_insertion_order() -> None:
    reference = _session("ref", ["encoder.output", "head.output", "extra"])
    candidate = _session("cand", ["head.output", "encoder.output", "candidate.only"])
    alignment = tt.align_checkpoints(reference, candidate)

    assert alignment.reference_execution_id == reference.execution_id
    assert alignment.candidate_execution_id == candidate.execution_id
    assert alignment.matched_count == 2
    assert alignment.missing_candidate_count == 1
    assert alignment.missing_reference_count == 1
    assert alignment.metadata_mismatch_count == 0
    assert [entry.checkpoint_name for entry in alignment.entries] == [
        "encoder.output",
        "head.output",
        "extra",
        "candidate.only",
    ]
    assert [entry.match_rule for entry in alignment.entries] == [
        "name",
        "name",
        "unmatched",
        "unmatched",
    ]
    assert alignment.entries[2].missing_candidate is True
    assert alignment.entries[3].missing_reference is True
    assert alignment.matched_entries[0].reference_name == "encoder.output"
    assert alignment.matched_entries[0].candidate_name == "encoder.output"


def test_explicit_mapping_takes_precedence_over_names() -> None:
    reference = _session("ref", ["a"])
    candidate = _session("cand", ["a", "b"])
    alignment = tt.align_checkpoints(reference, candidate, mapping={"a": "b"})
    explicit = alignment.entries[0]
    assert explicit.match_rule == "explicit"
    assert explicit.reference_name == "a"
    assert explicit.candidate_name == "b"
    assert "checkpoint_name" in explicit.metadata_mismatches
    assert "stage_name" in explicit.metadata_mismatches
    assert alignment.entries[1].missing_reference is True
    assert alignment.entries[1].candidate_name == "a"


def test_explicit_mapping_accepts_checkpoint_ids() -> None:
    reference = _session("ref", ["a"])
    candidate = _session("cand", ["b"])
    alignment = tt.align_checkpoints(
        reference,
        candidate,
        mapping={reference.checkpoints[0].checkpoint_id: (candidate.checkpoints[0].checkpoint_id)},
    )
    assert alignment.matched_count == 1
    assert alignment.entries[0].match_rule == "explicit"


def test_alignment_records_metadata_mismatches() -> None:
    reference = _session("ref", ["a"])
    candidate = tt.ExecutionSession.create(
        model_id="cand",
        model_version="1.0",
        framework="numpy",
        precision="float64",
    )
    candidate.add_checkpoint(
        name="a",
        tensor=np.arange(6, dtype=np.float32),
        stage_name="different",
        operation_type="add",
    )
    candidate.finalize()
    alignment = tt.align_checkpoints(reference, candidate)
    mismatches = alignment.entries[0].metadata_mismatches
    assert "stage_name" in mismatches
    assert "operation_type" in mismatches
    assert "tensor_shape" in mismatches
    assert "tensor_dtype" in mismatches
    # Both payloads are CPU and array-backed; device and name still match.
    assert "tensor_device" not in mismatches
    assert "checkpoint_name" not in mismatches


class _DuplicateEndpointMapping(dict[str, str]):
    """Mapping that yields duplicate keys to exercise defensive validation."""

    def items(self) -> Iterator[tuple[str, str]]:
        return iter([("a", "x"), ("a", "y")])


def test_duplicate_explicit_mapping_endpoints_are_rejected() -> None:
    reference = _session("ref", ["a", "b"])
    candidate = _session("cand", ["x", "y"])
    with pytest.raises(tt.AlignmentError, match="more than one explicit"):
        tt.align_checkpoints(reference, candidate, mapping={"a": "x", "b": "x"})
    with pytest.raises(tt.AlignmentError, match="more than one explicit"):
        tt.align_checkpoints(
            reference,
            candidate,
            mapping=_DuplicateEndpointMapping({"a": "x"}),
        )


def test_mapping_that_consumes_a_name_match_is_ambiguous() -> None:
    reference = _session("ref", ["a", "b"])
    candidate = _session("cand", ["a", "b"])
    with pytest.raises(tt.AlignmentError, match="ambiguous"):
        tt.align_checkpoints(reference, candidate, mapping={"a": "b"})


def test_unknown_mapping_endpoints_are_rejected() -> None:
    reference = _session("ref", ["a"])
    candidate = _session("cand", ["a"])
    with pytest.raises(tt.AlignmentError, match="reference checkpoint"):
        tt.align_checkpoints(reference, candidate, mapping={"missing": "a"})
    with pytest.raises(tt.AlignmentError, match="candidate checkpoint"):
        tt.align_checkpoints(reference, candidate, mapping={"a": "missing"})


def test_selection_filters_alignment_entries() -> None:
    reference = _session("ref", ["encoder.output", "head.output"])
    candidate = _session("cand", ["encoder.output", "head.output"])
    alignment = tt.align_checkpoints(reference, candidate, selection=["encoder.output"])
    assert alignment.selection == ("encoder.output",)
    assert [entry.checkpoint_name for entry in alignment.entries] == ["encoder.output"]
    assert alignment.matched_count == 1


def test_selection_must_name_an_existing_checkpoint() -> None:
    reference = _session("ref", ["a"])
    candidate = _session("cand", ["a"])
    with pytest.raises(tt.AlignmentError, match="not present"):
        tt.align_checkpoints(reference, candidate, selection=["missing"])
    with pytest.raises(tt.TensorValidationError, match="duplicate"):
        tt.align_checkpoints(reference, candidate, selection=["a", "a"])


def test_alignment_round_trip() -> None:
    reference = _session("ref", ["a", "b"])
    candidate = _session("cand", ["b", "c"])
    alignment = tt.align_checkpoints(reference, candidate, selection=["a", "b"])
    document = alignment.as_dict()
    restored = tt.CheckpointAlignment.from_dict(document)
    assert restored == alignment
    assert restored.as_dict() == document


def test_alignment_from_dict_rejects_unknown_fields() -> None:
    reference = _session("ref", ["a"])
    candidate = _session("cand", ["a"])
    document = tt.align_checkpoints(reference, candidate).as_dict()
    document["unknown"] = 1
    with pytest.raises(tt.AlignmentError, match="unknown alignment"):
        tt.CheckpointAlignment.from_dict(document)


def test_alignment_entry_rejects_inconsistent_shapes() -> None:
    with pytest.raises(tt.AlignmentError, match="at least one checkpoint"):
        tt.CheckpointAlignmentEntry(
            reference_checkpoint_id=None,
            candidate_checkpoint_id=None,
            reference_name=None,
            candidate_name=None,
            match_rule="unmatched",
        )
    with pytest.raises(tt.AlignmentError, match="missing reference"):
        tt.CheckpointAlignmentEntry(
            reference_checkpoint_id=None,
            candidate_checkpoint_id="cp_1",
            reference_name="ghost",
            candidate_name="a",
            match_rule="unmatched",
        )
    with pytest.raises(tt.AlignmentError, match="both reference and candidate"):
        tt.CheckpointAlignmentEntry(
            reference_checkpoint_id="cp_1",
            candidate_checkpoint_id=None,
            reference_name="a",
            candidate_name=None,
            match_rule="name",
        )
    with pytest.raises(tt.AlignmentError, match="unknown alignment match rule"):
        tt.CheckpointAlignmentEntry(
            reference_checkpoint_id="cp_1",
            candidate_checkpoint_id="cp_2",
            reference_name="a",
            candidate_name="b",
            match_rule="guess",  # type: ignore[arg-type]
        )


def test_alignment_entry_requires_a_known_mismatch_name() -> None:
    with pytest.raises(tt.AlignmentError, match="unknown metadata mismatch"):
        tt.CheckpointAlignmentEntry(
            reference_checkpoint_id="cp_1",
            candidate_checkpoint_id="cp_2",
            reference_name="a",
            candidate_name="a",
            match_rule="name",
            metadata_mismatches=("tensor_values",),
        )


def test_alignment_type_errors_are_explicit() -> None:
    reference = _session("ref", ["a"])
    with pytest.raises(TypeError, match="reference must be ExecutionSession"):
        tt.align_checkpoints(object(), reference)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="candidate must be ExecutionSession"):
        tt.align_checkpoints(reference, object())  # type: ignore[arg-type]
