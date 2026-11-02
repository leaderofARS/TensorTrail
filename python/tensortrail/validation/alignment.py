"""Deterministic checkpoint alignment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..exceptions import AlignmentError, CheckpointNotFoundError
from ..execution.checkpoint import TensorCheckpoint
from ..execution.metadata import validate_artifact_id, validate_identifier
from ..execution.session import ExecutionSession
from .policy import normalize_checkpoint_selection

__all__ = [
    "CheckpointAlignment",
    "CheckpointAlignmentEntry",
    "align_checkpoints",
]

MatchRule = Literal["explicit", "name", "unmatched"]
_MATCH_RULES = ("explicit", "name", "unmatched")
_METADATA_FIELDS = (
    "checkpoint_name",
    "stage_name",
    "operation_type",
    "tensor_shape",
    "tensor_dtype",
    "tensor_device",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointAlignmentEntry:
    """One aligned or unmatched checkpoint pair."""

    reference_checkpoint_id: str | None
    candidate_checkpoint_id: str | None
    reference_name: str | None
    candidate_name: str | None
    match_rule: MatchRule
    reference_stage: str | None = None
    candidate_stage: str | None = None
    metadata_mismatches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.reference_checkpoint_id is None and self.candidate_checkpoint_id is None:
            raise AlignmentError("an alignment entry must reference at least one checkpoint")
        if self.reference_checkpoint_id is None and self.reference_name is not None:
            raise AlignmentError("a missing reference checkpoint cannot have a reference name")
        if self.candidate_checkpoint_id is None and self.candidate_name is not None:
            raise AlignmentError("a missing candidate checkpoint cannot have a candidate name")
        if self.match_rule not in _MATCH_RULES:
            raise AlignmentError(f"unknown alignment match rule {self.match_rule!r}")
        if self.match_rule != "unmatched" and (
            self.reference_checkpoint_id is None or self.candidate_checkpoint_id is None
        ):
            raise AlignmentError(
                "explicit and name matches require both reference and candidate checkpoints"
            )
        if self.match_rule == "unmatched" and (
            self.reference_checkpoint_id is not None and self.candidate_checkpoint_id is not None
        ):
            raise AlignmentError("an unmatched entry cannot contain both checkpoints")
        for field_name, value in (
            ("reference_name", self.reference_name),
            ("candidate_name", self.candidate_name),
            ("reference_stage", self.reference_stage),
            ("candidate_stage", self.candidate_stage),
        ):
            if value is not None:
                validate_identifier(value, field_name)
        for mismatch in self.metadata_mismatches:
            if mismatch not in _METADATA_FIELDS:
                raise AlignmentError(f"unknown metadata mismatch {mismatch!r}")

    @property
    def matched(self) -> bool:
        return self.reference_checkpoint_id is not None and self.candidate_checkpoint_id is not None

    @property
    def missing_reference(self) -> bool:
        return self.reference_checkpoint_id is None and self.candidate_checkpoint_id is not None

    @property
    def missing_candidate(self) -> bool:
        return self.reference_checkpoint_id is not None and self.candidate_checkpoint_id is None

    @property
    def checkpoint_name(self) -> str:
        """Reference name when present, otherwise the candidate name."""
        name = self.reference_name if self.reference_name is not None else self.candidate_name
        if name is None:  # pragma: no cover - guarded by __post_init__
            raise AlignmentError("alignment entry has no checkpoint name")
        return name

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_checkpoint_id": self.reference_checkpoint_id,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "reference_name": self.reference_name,
            "candidate_name": self.candidate_name,
            "checkpoint_name": self.checkpoint_name,
            "match_rule": self.match_rule,
            "matched": self.matched,
            "missing_reference": self.missing_reference,
            "missing_candidate": self.missing_candidate,
            "reference_stage": self.reference_stage,
            "candidate_stage": self.candidate_stage,
            "metadata_mismatches": list(self.metadata_mismatches),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> CheckpointAlignmentEntry:
        required = {
            "reference_checkpoint_id",
            "candidate_checkpoint_id",
            "reference_name",
            "candidate_name",
            "match_rule",
            "reference_stage",
            "candidate_stage",
            "metadata_mismatches",
        }
        unknown = (
            set(document)
            - required
            - {
                "checkpoint_name",
                "matched",
                "missing_reference",
                "missing_candidate",
            }
        )
        if unknown:
            raise AlignmentError("unknown alignment entry field(s): " + ", ".join(sorted(unknown)))
        missing = required - set(document)
        if missing:
            raise AlignmentError("missing alignment entry field(s): " + ", ".join(sorted(missing)))
        mismatches = document["metadata_mismatches"]
        if not isinstance(mismatches, (list, tuple)):
            raise AlignmentError("metadata_mismatches must be a list")
        match_rule = document["match_rule"]
        if not isinstance(match_rule, str):
            raise AlignmentError("match_rule must be a string")
        return cls(
            reference_checkpoint_id=_optional_text(document, "reference_checkpoint_id"),
            candidate_checkpoint_id=_optional_text(document, "candidate_checkpoint_id"),
            reference_name=_optional_text(document, "reference_name"),
            candidate_name=_optional_text(document, "candidate_name"),
            match_rule=cast(MatchRule, match_rule),
            reference_stage=_optional_text(document, "reference_stage"),
            candidate_stage=_optional_text(document, "candidate_stage"),
            metadata_mismatches=tuple(str(item) for item in mismatches),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointAlignment:
    """Structured result of aligning two executions."""

    reference_execution_id: str
    candidate_execution_id: str
    entries: tuple[CheckpointAlignmentEntry, ...]
    selection: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
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
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(
            self,
            "selection",
            normalize_checkpoint_selection(self.selection),
        )

    @property
    def matched_entries(self) -> tuple[CheckpointAlignmentEntry, ...]:
        return tuple(entry for entry in self.entries if entry.matched)

    @property
    def missing_reference_entries(self) -> tuple[CheckpointAlignmentEntry, ...]:
        return tuple(entry for entry in self.entries if entry.missing_reference)

    @property
    def missing_candidate_entries(self) -> tuple[CheckpointAlignmentEntry, ...]:
        return tuple(entry for entry in self.entries if entry.missing_candidate)

    @property
    def matched_count(self) -> int:
        return len(self.matched_entries)

    @property
    def missing_reference_count(self) -> int:
        return len(self.missing_reference_entries)

    @property
    def missing_candidate_count(self) -> int:
        return len(self.missing_candidate_entries)

    @property
    def metadata_mismatch_count(self) -> int:
        return sum(1 for entry in self.entries if entry.metadata_mismatches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_execution_id": self.reference_execution_id,
            "candidate_execution_id": self.candidate_execution_id,
            "selection": list(self.selection) if self.selection is not None else None,
            "matched_count": self.matched_count,
            "missing_reference_count": self.missing_reference_count,
            "missing_candidate_count": self.missing_candidate_count,
            "metadata_mismatch_count": self.metadata_mismatch_count,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> CheckpointAlignment:
        required = {
            "reference_execution_id",
            "candidate_execution_id",
            "selection",
            "entries",
        }
        unknown = (
            set(document)
            - required
            - {
                "matched_count",
                "missing_reference_count",
                "missing_candidate_count",
                "metadata_mismatch_count",
            }
        )
        if unknown:
            raise AlignmentError("unknown alignment field(s): " + ", ".join(sorted(unknown)))
        missing = required - set(document)
        if missing:
            raise AlignmentError("missing alignment field(s): " + ", ".join(sorted(missing)))
        entries = document["entries"]
        if not isinstance(entries, list):
            raise AlignmentError("alignment entries must be a list")
        selection = document["selection"]
        if selection is not None and not isinstance(selection, (list, tuple)):
            raise AlignmentError("alignment selection must be a list or null")
        return cls(
            reference_execution_id=_required_text(document, "reference_execution_id"),
            candidate_execution_id=_required_text(document, "candidate_execution_id"),
            entries=tuple(
                CheckpointAlignmentEntry.from_dict(_entry_document(item)) for item in entries
            ),
            selection=normalize_checkpoint_selection(selection),
        )


def align_checkpoints(
    reference: ExecutionSession,
    candidate: ExecutionSession,
    *,
    mapping: Mapping[str, str] | None = None,
    selection: Iterable[str] | None = None,
) -> CheckpointAlignment:
    """Align reference and candidate checkpoints deterministically.

    Explicit mapping entries take precedence, then unique checkpoint names.
    Position is never used as an implicit matching rule.
    """
    if not isinstance(reference, ExecutionSession):
        raise TypeError(f"reference must be ExecutionSession, received {type(reference).__name__}")
    if not isinstance(candidate, ExecutionSession):
        raise TypeError(f"candidate must be ExecutionSession, received {type(candidate).__name__}")

    reference_checkpoints = _unique_checkpoints(reference, "reference")
    candidate_checkpoints = _unique_checkpoints(candidate, "candidate")
    candidate_by_name = {item.checkpoint_name: item for item in candidate_checkpoints}

    explicit_by_reference: dict[str, tuple[TensorCheckpoint, TensorCheckpoint]] = {}
    used_candidates: set[str] = set()
    used_references: set[str] = set()

    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise AlignmentError("mapping must be a mapping of reference to candidate identifiers")
        for reference_token, candidate_token in mapping.items():
            if not isinstance(reference_token, str) or not isinstance(candidate_token, str):
                raise AlignmentError("mapping keys and values must be strings")
            reference_checkpoint = _resolve_checkpoint(reference, reference_token, "reference")
            candidate_checkpoint = _resolve_checkpoint(candidate, candidate_token, "candidate")
            if reference_checkpoint.checkpoint_id in used_references:
                raise AlignmentError(
                    f"reference checkpoint {reference_checkpoint.checkpoint_name!r} "
                    "appears in more than one explicit mapping entry"
                )
            if candidate_checkpoint.checkpoint_id in used_candidates:
                raise AlignmentError(
                    f"candidate checkpoint {candidate_checkpoint.checkpoint_name!r} "
                    "appears in more than one explicit mapping entry"
                )
            used_references.add(reference_checkpoint.checkpoint_id)
            used_candidates.add(candidate_checkpoint.checkpoint_id)
            explicit_by_reference[reference_checkpoint.checkpoint_id] = (
                reference_checkpoint,
                candidate_checkpoint,
            )

    entries: list[CheckpointAlignmentEntry] = []
    matched_candidates: set[str] = set(used_candidates)

    for reference_checkpoint in reference_checkpoints:
        explicit = explicit_by_reference.get(reference_checkpoint.checkpoint_id)
        if explicit is not None:
            _, candidate_checkpoint = explicit
            entries.append(
                _matched_entry(
                    reference_checkpoint,
                    candidate_checkpoint,
                    "explicit",
                )
            )
            continue

        named_candidate = candidate_by_name.get(reference_checkpoint.checkpoint_name)
        if named_candidate is None:
            entries.append(_unmatched_entry(reference_checkpoint, None))
            continue
        if named_candidate.checkpoint_id in matched_candidates:
            raise AlignmentError(
                f"candidate checkpoint {named_candidate.checkpoint_name!r} "
                "was already consumed by an explicit mapping; the mapping is ambiguous "
                f"for reference checkpoint {reference_checkpoint.checkpoint_name!r}"
            )
        matched_candidates.add(named_candidate.checkpoint_id)
        entries.append(_matched_entry(reference_checkpoint, named_candidate, "name"))

    for candidate_checkpoint in candidate_checkpoints:
        if candidate_checkpoint.checkpoint_id in matched_candidates:
            continue
        entries.append(_unmatched_entry(None, candidate_checkpoint))

    alignment = CheckpointAlignment(
        reference_execution_id=reference.execution_id,
        candidate_execution_id=candidate.execution_id,
        entries=tuple(entries),
        selection=None,
    )

    normalized_selection = normalize_checkpoint_selection(selection)
    if normalized_selection is None:
        return alignment

    known_names = {
        name
        for entry in alignment.entries
        for name in (entry.reference_name, entry.candidate_name)
        if name is not None
    }
    unknown = [name for name in normalized_selection if name not in known_names]
    if unknown:
        raise AlignmentError(
            "checkpoint_selection contains name(s) not present in either execution: "
            + ", ".join(repr(name) for name in unknown)
        )

    selected_entries = tuple(
        entry
        for entry in alignment.entries
        if entry.reference_name in normalized_selection
        or entry.candidate_name in normalized_selection
    )
    return CheckpointAlignment(
        reference_execution_id=alignment.reference_execution_id,
        candidate_execution_id=alignment.candidate_execution_id,
        entries=selected_entries,
        selection=normalized_selection,
    )


def _unique_checkpoints(
    session: ExecutionSession,
    role: str,
) -> tuple[TensorCheckpoint, ...]:
    checkpoints = session.checkpoints
    names: dict[str, str] = {}
    for checkpoint in checkpoints:
        previous = names.get(checkpoint.checkpoint_name)
        if previous is not None:
            raise AlignmentError(
                f"{role} execution has duplicate checkpoint name "
                f"{checkpoint.checkpoint_name!r} ({previous!r} and "
                f"{checkpoint.checkpoint_id!r})"
            )
        names[checkpoint.checkpoint_name] = checkpoint.checkpoint_id
    return checkpoints


def _resolve_checkpoint(
    session: ExecutionSession,
    token: str,
    role: str,
) -> TensorCheckpoint:
    identifier = validate_identifier(token, f"{role} mapping identifier")
    try:
        return session.get_checkpoint(identifier)
    except CheckpointNotFoundError:
        pass
    try:
        return session.get_checkpoint_by_name(identifier)
    except CheckpointNotFoundError as error:
        raise AlignmentError(
            f"{role} checkpoint {identifier!r} does not exist by ID or name"
        ) from error


def _matched_entry(
    reference: TensorCheckpoint,
    candidate: TensorCheckpoint,
    rule: Literal["explicit", "name"],
) -> CheckpointAlignmentEntry:
    mismatches: list[str] = []
    if reference.checkpoint_name != candidate.checkpoint_name:
        mismatches.append("checkpoint_name")
    if reference.stage_name != candidate.stage_name:
        mismatches.append("stage_name")
    if reference.operation_type != candidate.operation_type:
        mismatches.append("operation_type")
    if reference.tensor_shape != candidate.tensor_shape:
        mismatches.append("tensor_shape")
    if reference.tensor_dtype != candidate.tensor_dtype:
        mismatches.append("tensor_dtype")
    if reference.tensor_device != candidate.tensor_device:
        mismatches.append("tensor_device")

    return CheckpointAlignmentEntry(
        reference_checkpoint_id=reference.checkpoint_id,
        candidate_checkpoint_id=candidate.checkpoint_id,
        reference_name=reference.checkpoint_name,
        candidate_name=candidate.checkpoint_name,
        match_rule=rule,
        reference_stage=reference.stage_name,
        candidate_stage=candidate.stage_name,
        metadata_mismatches=tuple(mismatches),
    )


def _unmatched_entry(
    reference: TensorCheckpoint | None,
    candidate: TensorCheckpoint | None,
) -> CheckpointAlignmentEntry:
    return CheckpointAlignmentEntry(
        reference_checkpoint_id=(reference.checkpoint_id if reference is not None else None),
        candidate_checkpoint_id=(candidate.checkpoint_id if candidate is not None else None),
        reference_name=reference.checkpoint_name if reference is not None else None,
        candidate_name=candidate.checkpoint_name if candidate is not None else None,
        match_rule="unmatched",
        reference_stage=reference.stage_name if reference is not None else None,
        candidate_stage=candidate.stage_name if candidate is not None else None,
        metadata_mismatches=(),
    )


def _optional_text(document: Mapping[str, Any], key: str) -> str | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AlignmentError(f"{key} must be a string or null")
    return value


def _required_text(document: Mapping[str, Any], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise AlignmentError(f"{key} must be a string")
    return value


def _entry_document(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlignmentError("alignment entries must be mappings")
    return cast(Mapping[str, Any], value)
