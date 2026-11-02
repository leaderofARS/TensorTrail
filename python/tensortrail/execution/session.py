"""Execution session lifecycle and payload ownership."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..exceptions import ArtifactError, CheckpointError, ExecutionStateError
from .checkpoint import TensorCheckpoint, describe_tensor
from .metadata import (
    EXECUTION_SCHEMA_VERSION,
    ExecutionMetadata,
    JsonValue,
    ensure_json_mapping,
    ensure_utc,
    format_datetime,
    new_checkpoint_id,
    new_execution_id,
    plain_json,
    utc_now,
    validate_artifact_id,
    validate_identifier,
)
from .registry import CheckpointRegistry

if TYPE_CHECKING:
    from .artifacts import ExecutionStorage

__all__ = ["ExecutionSession", "ExecutionStatus"]

ExecutionStatus = Literal["created", "recording", "completed", "failed"]
_TERMINAL_STATUSES: tuple[ExecutionStatus, ...] = ("completed", "failed")


class ExecutionSession:
    """One inference execution with explicit lifecycle and separate payloads.

    A session is a single-writer object. It is not thread-safe. Checkpoint
    metadata lives in a :class:`CheckpointRegistry`; tensor payloads are held in
    a separate mapping and are never embedded in checkpoint metadata.
    """

    __slots__ = (
        "_checkpoints",
        "_completed_at",
        "_created_at",
        "_failure_reason",
        "_metadata",
        "_payloads",
        "_session_metadata",
        "_status",
    )

    def __init__(
        self,
        metadata: ExecutionMetadata,
        *,
        session_metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        if not isinstance(metadata, ExecutionMetadata):
            raise TypeError(
                f"metadata must be ExecutionMetadata, received {type(metadata).__name__}"
            )
        self._metadata = metadata
        self._session_metadata = ensure_json_mapping(session_metadata, "session_metadata")
        self._checkpoints = CheckpointRegistry(metadata.execution_id)
        self._payloads: dict[str, object] = {}
        self._status: ExecutionStatus = "created"
        self._created_at = utc_now()
        self._completed_at: datetime | None = None
        self._failure_reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        model_version: str,
        framework: str,
        precision: str,
        execution_id: str | None = None,
        model_fingerprint: str | None = None,
        framework_version: str | None = None,
        device_type: str | None = None,
        device_name: str | None = None,
        device_identifier: str | None = None,
        input_identifier: str | None = None,
        dataset_identifier: str | None = None,
        execution_timestamp: datetime | None = None,
        execution_configuration: Mapping[str, JsonValue] | None = None,
        custom_metadata: Mapping[str, JsonValue] | None = None,
        session_metadata: Mapping[str, JsonValue] | None = None,
    ) -> ExecutionSession:
        """Create a session in the ``created`` state."""
        metadata = ExecutionMetadata(
            execution_id=execution_id if execution_id is not None else new_execution_id(),
            model_id=model_id,
            model_version=model_version,
            framework=framework,
            precision=precision,
            model_fingerprint=model_fingerprint,
            framework_version=framework_version,
            device_type=device_type,
            device_name=device_name,
            device_identifier=device_identifier,
            input_identifier=input_identifier,
            dataset_identifier=dataset_identifier,
            execution_timestamp=(
                execution_timestamp if execution_timestamp is not None else utc_now()
            ),
            execution_configuration=(
                execution_configuration if execution_configuration is not None else {}
            ),
            custom_metadata=custom_metadata if custom_metadata is not None else {},
        )
        return cls(metadata, session_metadata=session_metadata)

    @property
    def execution_id(self) -> str:
        return self._metadata.execution_id

    @property
    def metadata(self) -> ExecutionMetadata:
        return self._metadata

    @property
    def session_metadata(self) -> Mapping[str, JsonValue]:
        return self._session_metadata

    @property
    def schema_version(self) -> str:
        return EXECUTION_SCHEMA_VERSION

    @property
    def status(self) -> ExecutionStatus:
        return self._status

    @property
    def session_status(self) -> ExecutionStatus:
        """Alias for :attr:`status`."""
        return self._status

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def completed_at(self) -> datetime | None:
        return self._completed_at

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def checkpoints(self) -> tuple[TensorCheckpoint, ...]:
        return self._checkpoints.all()

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    def add_checkpoint(
        self,
        *,
        name: str,
        tensor: object,
        checkpoint_id: str | None = None,
        checkpoint_index: int | None = None,
        stage_name: str | None = None,
        operation_type: str | None = None,
        capture_timestamp: datetime | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> TensorCheckpoint:
        """Register a tensor checkpoint and retain a reference to its payload.

        The payload is not copied, normalized, or transferred. CPU PyTorch
        tensors are inspected through the explicit CPU-only adapter. The first
        successful registration moves the session from ``created`` to
        ``recording``.
        """
        if self._status in _TERMINAL_STATUSES:
            raise ExecutionStateError(
                f"cannot add checkpoint to a {self._status} execution session"
            )

        checkpoint_name = validate_identifier(name, "checkpoint_name")
        if checkpoint_name in self._checkpoints.names():
            raise CheckpointError(f"checkpoint_name {checkpoint_name!r} is already registered")

        identifier = checkpoint_id if checkpoint_id is not None else new_checkpoint_id()
        if checkpoint_index is None:
            next_index = max(
                (item.checkpoint_index for item in self._checkpoints),
                default=-1,
            )
            next_index += 1
        else:
            next_index = checkpoint_index

        description = describe_tensor(
            tensor,
            f"tensor for checkpoint {checkpoint_name!r}",
        )
        checkpoint = TensorCheckpoint(
            checkpoint_id=identifier,
            execution_id=self.execution_id,
            checkpoint_name=checkpoint_name,
            checkpoint_index=next_index,
            stage_name=stage_name,
            operation_type=operation_type,
            tensor_shape=description.shape,
            tensor_dtype=description.dtype,
            tensor_nbytes=description.nbytes,
            tensor_source=description.source,
            tensor_device=description.device,
            tensor_reference="memory",
            capture_timestamp=(capture_timestamp if capture_timestamp is not None else utc_now()),
            metadata=metadata if metadata is not None else {},
        )
        self._checkpoints.register(checkpoint)
        self._payloads[checkpoint.checkpoint_id] = tensor
        if self._status == "created":
            self._status = "recording"
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> TensorCheckpoint:
        """Return checkpoint metadata by stable ID."""
        return self._checkpoints.get(checkpoint_id)

    def get_checkpoint_by_name(self, checkpoint_name: str) -> TensorCheckpoint:
        """Return checkpoint metadata by unique name."""
        return self._checkpoints.get_by_name(checkpoint_name)

    def get_tensor(self, checkpoint_id: str) -> object:
        """Return the registered payload reference without copying it."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        try:
            return self._payloads[checkpoint.checkpoint_id]
        except KeyError as error:
            raise ArtifactError(
                f"payload for checkpoint {checkpoint.checkpoint_id!r} "
                f"({checkpoint.checkpoint_name!r}) is unavailable in this session"
            ) from error

    def finalize(self) -> None:
        """Transition ``created``/``recording`` to ``completed``."""
        if self._status in _TERMINAL_STATUSES:
            raise ExecutionStateError(f"cannot finalize a {self._status} execution session")
        self._status = "completed"
        self._completed_at = utc_now()

    def fail(self, reason: str) -> None:
        """Transition ``created``/``recording`` to ``failed``."""
        if self._status in _TERMINAL_STATUSES:
            raise ExecutionStateError(f"cannot fail a {self._status} execution session")
        message = validate_identifier(reason, "failure reason")
        self._status = "failed"
        self._completed_at = utc_now()
        self._failure_reason = message

    def as_dict(self) -> dict[str, Any]:
        """Serialize session metadata; tensor payloads are not included."""
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "session_status": self._status,
            "created_at": format_datetime(self._created_at),
            "completed_at": (
                format_datetime(self._completed_at) if self._completed_at is not None else None
            ),
            "failure_reason": self._failure_reason,
            "metadata": self._metadata.to_dict(),
            "session_metadata": plain_json(self._session_metadata),
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
        }

    to_dict = as_dict

    def summary(self) -> str:
        """Human-readable one-line session description."""
        completed = ""
        if self._completed_at is not None:
            completed = f" completed_at={format_datetime(self._completed_at)}"
        return (
            f"ExecutionSession({self.execution_id}, model={self._metadata.model_id!r} "
            f"version={self._metadata.model_version!r}, status={self._status}, "
            f"checkpoints={self.checkpoint_count}{completed})"
        )

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return self.summary()

    def save(self, storage: ExecutionStorage) -> Path:
        """Persist this terminal session through a storage backend."""
        from .artifacts import ExecutionStorage as StorageProtocol

        if not isinstance(storage, StorageProtocol):
            raise TypeError(
                f"storage must implement ExecutionStorage, received {type(storage).__name__}"
            )
        if self._status not in _TERMINAL_STATUSES:
            raise ExecutionStateError(
                "only completed or failed execution sessions can be persisted; "
                f"session is {self._status}"
            )
        return storage.save_execution(self)

    @classmethod
    def load(
        cls,
        storage: ExecutionStorage,
        execution_id: str,
    ) -> ExecutionSession:
        """Load a terminal session from a storage backend."""
        from .artifacts import ExecutionStorage as StorageProtocol

        if not isinstance(storage, StorageProtocol):
            raise TypeError(
                f"storage must implement ExecutionStorage, received {type(storage).__name__}"
            )
        return storage.load_execution(validate_artifact_id(execution_id, "execution_id"))

    @classmethod
    def _restore(
        cls,
        *,
        metadata: ExecutionMetadata,
        checkpoints: tuple[TensorCheckpoint, ...],
        payloads: Mapping[str, object],
        status: ExecutionStatus,
        created_at: datetime,
        completed_at: datetime | None,
        failure_reason: str | None,
        session_metadata: Mapping[str, JsonValue],
    ) -> ExecutionSession:
        """Rebuild a terminal session from verified artifact data."""
        if status not in _TERMINAL_STATUSES:
            raise ArtifactError(
                f"persisted execution status must be 'completed' or 'failed', received {status!r}"
            )
        session = cls(metadata, session_metadata=session_metadata)
        if set(payloads) != {checkpoint.checkpoint_id for checkpoint in checkpoints}:
            raise ArtifactError(
                "persisted execution payload set does not match checkpoint metadata"
            )
        for checkpoint in checkpoints:
            session._checkpoints.register(checkpoint)
            session._payloads[checkpoint.checkpoint_id] = payloads[checkpoint.checkpoint_id]
        session._status = status
        session._created_at = ensure_utc(created_at, "created_at")
        session._completed_at = (
            ensure_utc(completed_at, "completed_at") if completed_at is not None else None
        )
        if session._completed_at is None:
            raise ArtifactError("persisted terminal execution is missing completed_at")
        session._failure_reason = failure_reason
        return session
