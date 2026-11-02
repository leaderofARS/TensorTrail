"""Deterministic checkpoint registry."""

from __future__ import annotations

from collections.abc import Iterator

from ..exceptions import CheckpointError, CheckpointNotFoundError
from .checkpoint import TensorCheckpoint
from .metadata import validate_artifact_id

__all__ = ["CheckpointRegistry"]


class CheckpointRegistry:
    """Owns checkpoint metadata for one execution.

    IDs, names, and execution-order indices are unique. Iteration and
    ``all()`` use ascending ``checkpoint_index``; callers must not depend on
    dictionary insertion order.
    """

    __slots__ = ("_by_id", "_execution_id", "_indices", "_name_to_id")

    def __init__(self, execution_id: str) -> None:
        self._execution_id = validate_artifact_id(execution_id, "execution_id")
        self._by_id: dict[str, TensorCheckpoint] = {}
        self._name_to_id: dict[str, str] = {}
        self._indices: dict[int, str] = {}

    @property
    def execution_id(self) -> str:
        return self._execution_id

    def register(self, checkpoint: TensorCheckpoint) -> TensorCheckpoint:
        """Register a checkpoint or raise :class:`CheckpointError`."""
        if not isinstance(checkpoint, TensorCheckpoint):
            raise CheckpointError(
                f"checkpoint must be TensorCheckpoint, received {type(checkpoint).__name__}"
            )
        if checkpoint.execution_id != self._execution_id:
            raise CheckpointError(
                f"checkpoint {checkpoint.checkpoint_id!r} belongs to execution "
                f"{checkpoint.execution_id!r}, not {self._execution_id!r}"
            )
        if checkpoint.checkpoint_id in self._by_id:
            raise CheckpointError(
                f"checkpoint_id {checkpoint.checkpoint_id!r} is already registered"
            )
        if checkpoint.checkpoint_name in self._name_to_id:
            raise CheckpointError(
                f"checkpoint_name {checkpoint.checkpoint_name!r} is already registered; "
                "checkpoint names must be unique within an execution"
            )
        existing_index = self._indices.get(checkpoint.checkpoint_index)
        if existing_index is not None:
            raise CheckpointError(
                f"checkpoint_index {checkpoint.checkpoint_index} is already used by "
                f"checkpoint {existing_index!r}"
            )

        self._by_id[checkpoint.checkpoint_id] = checkpoint
        self._name_to_id[checkpoint.checkpoint_name] = checkpoint.checkpoint_id
        self._indices[checkpoint.checkpoint_index] = checkpoint.checkpoint_id
        return checkpoint

    def get(self, checkpoint_id: str) -> TensorCheckpoint:
        """Return a checkpoint by stable ID."""
        try:
            return self._by_id[checkpoint_id]
        except KeyError as error:
            raise CheckpointNotFoundError(
                f"checkpoint_id {checkpoint_id!r} is not registered"
            ) from error

    by_id = get

    def get_by_name(self, checkpoint_name: str) -> TensorCheckpoint:
        """Return a checkpoint by unique name."""
        checkpoint_id = self._name_to_id.get(checkpoint_name)
        if checkpoint_id is None:
            raise CheckpointNotFoundError(f"checkpoint_name {checkpoint_name!r} is not registered")
        return self._by_id[checkpoint_id]

    by_name = get_by_name

    def for_execution(self, execution_id: str) -> tuple[TensorCheckpoint, ...]:
        """Return this registry's checkpoints when the execution ID matches."""
        if execution_id != self._execution_id:
            return ()
        return self.all()

    def all(self) -> tuple[TensorCheckpoint, ...]:
        """Return every checkpoint in execution order."""
        ordered = sorted(
            self._by_id,
            key=lambda checkpoint_id: self._by_id[checkpoint_id].checkpoint_index,
        )
        return tuple(self._by_id[checkpoint_id] for checkpoint_id in ordered)

    def ids(self) -> tuple[str, ...]:
        """Return checkpoint IDs in execution order."""
        return tuple(checkpoint.checkpoint_id for checkpoint in self.all())

    def names(self) -> tuple[str, ...]:
        """Return checkpoint names in execution order."""
        return tuple(checkpoint.checkpoint_name for checkpoint in self.all())

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[TensorCheckpoint]:
        return iter(self.all())

    def __contains__(self, checkpoint_id: object) -> bool:
        return isinstance(checkpoint_id, str) and checkpoint_id in self._by_id

    def __repr__(self) -> str:
        return f"CheckpointRegistry(execution_id={self._execution_id!r}, checkpoints={len(self)})"
