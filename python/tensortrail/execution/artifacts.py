"""Local filesystem persistence for execution and validation artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .._version import __version__
from ..backends.pytorch_backend import looks_like_torch_tensor
from ..exceptions import (
    ArtifactError,
    ArtifactIntegrityError,
    CheckpointError,
    ExecutionStateError,
    TensorValidationError,
)
from .checkpoint import TensorCheckpoint
from .metadata import (
    ExecutionMetadata,
    format_datetime,
    parse_datetime,
    validate_artifact_id,
)
from .session import ExecutionSession

if TYPE_CHECKING:
    from ..validation.results import ValidationResult

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ExecutionStorage",
    "FilesystemStorage",
]

ARTIFACT_SCHEMA_VERSION = "1.0"
_MANIFEST_NAME = "manifest.json"
_CHECKPOINTS_DIR = "checkpoints"
_CHECKPOINT_METADATA_NAME = "metadata.json"
_TENSOR_NAME = "tensor.npy"
_VALIDATION_DOCUMENT_NAME = "validation.json"


@runtime_checkable
class ExecutionStorage(Protocol):
    """Storage backend contract for execution artifacts."""

    def save_execution(self, session: ExecutionSession) -> Path:
        """Persist a terminal execution and return its artifact directory."""
        ...

    def load_execution(self, execution_id: str) -> ExecutionSession:
        """Load a terminal execution by ID."""
        ...

    def save_validation(self, result: ValidationResult) -> Path:
        """Persist a validation result and return its artifact directory."""
        ...

    def load_validation(self, validation_id: str) -> ValidationResult:
        """Load a validation result by ID."""
        ...


class FilesystemStorage:
    """Local filesystem artifact storage.

    Each artifact is written into a staging directory and published with a
    directory rename. Existing artifacts are never overwritten.
    """

    __slots__ = ("_root",)

    def __init__(self, root: str | Path = "tensortrail_artifacts") -> None:
        self._root = Path(root).expanduser()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def executions_dir(self) -> Path:
        return self._root / "executions"

    @property
    def validations_dir(self) -> Path:
        return self._root / "validations"

    def execution_path(self, execution_id: str) -> Path:
        """Return the artifact directory for a validated execution ID."""
        return self.executions_dir / validate_artifact_id(execution_id, "execution_id")

    def validation_path(self, validation_id: str) -> Path:
        """Return the artifact directory for a validated validation ID."""
        return self.validations_dir / validate_artifact_id(validation_id, "validation_id")

    def save_execution(self, session: ExecutionSession) -> Path:
        """Persist a completed or failed execution session."""
        if not isinstance(session, ExecutionSession):
            raise TypeError(f"session must be ExecutionSession, received {type(session).__name__}")
        if session.status not in {"completed", "failed"}:
            raise ExecutionStateError(
                "only completed or failed execution sessions can be persisted; "
                f"session is {session.status}"
            )
        if session.completed_at is None:
            raise ArtifactError("terminal execution session is missing completed_at")
        if session.status == "failed" and session.failure_reason is None:
            raise ArtifactError("failed execution session is missing a failure reason")
        if session.status == "completed" and session.failure_reason is not None:
            raise ArtifactError("completed execution session unexpectedly has a failure reason")

        destination = self.execution_path(session.execution_id)
        if destination.exists():
            raise ArtifactError(
                f"execution artifact {destination} already exists; artifacts are never overwritten"
            )
        self.executions_dir.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{session.execution_id}.", dir=self.executions_dir))
        published = False
        try:
            checkpoint_entries: list[dict[str, Any]] = []
            checkpoint_root = stage / _CHECKPOINTS_DIR
            checkpoint_root.mkdir()
            for checkpoint in session.checkpoints:
                payload = session.get_tensor(checkpoint.checkpoint_id)
                array = _payload_to_numpy(payload, checkpoint.checkpoint_name)
                _verify_payload_metadata(array, checkpoint, "live")
                stored_checkpoint = replace(checkpoint, tensor_reference="artifact")
                checkpoint_dir = checkpoint_root / checkpoint.checkpoint_id
                checkpoint_dir.mkdir()
                metadata_path = checkpoint_dir / _CHECKPOINT_METADATA_NAME
                _write_json(metadata_path, stored_checkpoint.to_dict())
                tensor_path = checkpoint_dir / _TENSOR_NAME
                try:
                    np.save(tensor_path, array, allow_pickle=False)
                except (OSError, ValueError) as error:
                    raise ArtifactError(
                        f"could not write tensor payload for checkpoint "
                        f"{checkpoint.checkpoint_name!r}: {error}"
                    ) from error
                checkpoint_entries.append(
                    {
                        "checkpoint": stored_checkpoint.to_dict(),
                        "metadata_path": (
                            f"{_CHECKPOINTS_DIR}/{checkpoint.checkpoint_id}/"
                            f"{_CHECKPOINT_METADATA_NAME}"
                        ),
                        "metadata_sha256": _sha256_file(metadata_path),
                        "tensor_path": (
                            f"{_CHECKPOINTS_DIR}/{checkpoint.checkpoint_id}/{_TENSOR_NAME}"
                        ),
                        "tensor_sha256": _sha256_file(tensor_path),
                        "tensor_nbytes": int(array.nbytes),
                    }
                )

            manifest = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "execution",
                "library": {"name": "tensortrail", "version": __version__},
                "execution_id": session.execution_id,
                "session_status": session.status,
                "created_at": format_datetime(session.created_at),
                "completed_at": format_datetime(session.completed_at),
                "failure_reason": session.failure_reason,
                "session_metadata": _plain_mapping(session.session_metadata),
                "metadata": session.metadata.to_dict(),
                "checkpoints": checkpoint_entries,
            }
            _write_json(stage / _MANIFEST_NAME, manifest)
            _publish(stage, destination)
            published = True
            return destination
        finally:
            if not published:
                shutil.rmtree(stage, ignore_errors=True)

    def load_execution(self, execution_id: str) -> ExecutionSession:
        """Load and verify a terminal execution artifact."""
        identifier = validate_artifact_id(execution_id, "execution_id")
        destination = self.execution_path(identifier)
        if not destination.is_dir():
            raise ArtifactError(f"execution artifact directory {destination} does not exist")
        manifest_path = destination / _MANIFEST_NAME
        manifest = _read_json(manifest_path)
        required = {
            "schema_version",
            "artifact_type",
            "library",
            "execution_id",
            "session_status",
            "created_at",
            "completed_at",
            "failure_reason",
            "session_metadata",
            "metadata",
            "checkpoints",
        }
        _require_keys(manifest, required, set(), "execution manifest")
        if _text(manifest, "schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactIntegrityError(
                f"execution manifest schema version {manifest['schema_version']!r} is "
                f"unsupported; expected {ARTIFACT_SCHEMA_VERSION!r}"
            )
        if _text(manifest, "artifact_type") != "execution":
            raise ArtifactIntegrityError("execution manifest artifact_type must be 'execution'")
        library = _mapping(manifest, "library")
        if _text(library, "name") != "tensortrail":
            raise ArtifactIntegrityError("execution library name must be 'tensortrail'")
        if _text(manifest, "execution_id") != identifier:
            raise ArtifactIntegrityError("execution manifest ID does not match its directory")
        status = _text(manifest, "session_status")
        if status not in {"completed", "failed"}:
            raise ArtifactIntegrityError(
                f"execution artifact status must be completed or failed, received {status!r}"
            )
        metadata_document = _mapping(manifest, "metadata")
        metadata = ExecutionMetadata.from_dict(metadata_document)
        if metadata.execution_id != identifier:
            raise ArtifactIntegrityError(
                "execution metadata ID does not match the manifest execution ID"
            )
        created_at = parse_datetime(manifest["created_at"], "created_at")
        completed_at = parse_datetime(manifest["completed_at"], "completed_at")
        failure_reason = _optional_text(manifest, "failure_reason")
        if status == "failed" and failure_reason is None:
            raise ArtifactIntegrityError("failed execution artifact is missing failure_reason")
        if status == "completed" and failure_reason is not None:
            raise ArtifactIntegrityError(
                "completed execution artifact unexpectedly has a failure_reason"
            )
        session_metadata = _mapping(manifest, "session_metadata")

        checkpoints_document = manifest["checkpoints"]
        if not isinstance(checkpoints_document, list):
            raise ArtifactIntegrityError("execution manifest checkpoints must be a list")
        checkpoints: list[TensorCheckpoint] = []
        payloads: dict[str, NDArray[Any]] = {}
        for value in checkpoints_document:
            entry = _as_mapping(value, "checkpoint entry")
            _require_keys(
                entry,
                {
                    "checkpoint",
                    "metadata_path",
                    "metadata_sha256",
                    "tensor_path",
                    "tensor_sha256",
                    "tensor_nbytes",
                },
                set(),
                "checkpoint entry",
            )
            checkpoint_document = _mapping(entry, "checkpoint")
            checkpoint = TensorCheckpoint.from_dict(checkpoint_document)
            if checkpoint.execution_id != identifier:
                raise ArtifactIntegrityError(
                    f"checkpoint {checkpoint.checkpoint_id!r} belongs to a different execution"
                )
            expected_metadata_path = (
                f"{_CHECKPOINTS_DIR}/{checkpoint.checkpoint_id}/{_CHECKPOINT_METADATA_NAME}"
            )
            expected_tensor_path = f"{_CHECKPOINTS_DIR}/{checkpoint.checkpoint_id}/{_TENSOR_NAME}"
            if _text(entry, "metadata_path") != expected_metadata_path:
                raise ArtifactIntegrityError(
                    f"checkpoint {checkpoint.checkpoint_id!r} has an unsafe metadata path"
                )
            if _text(entry, "tensor_path") != expected_tensor_path:
                raise ArtifactIntegrityError(
                    f"checkpoint {checkpoint.checkpoint_id!r} has an unsafe tensor path"
                )
            metadata_file = destination / expected_metadata_path
            if _sha256_file(metadata_file) != _text(entry, "metadata_sha256"):
                raise ArtifactIntegrityError(
                    f"checkpoint metadata checksum mismatch for {checkpoint.checkpoint_id!r}"
                )
            metadata_on_disk = _read_json(metadata_file)
            if metadata_on_disk != checkpoint_document:
                raise ArtifactIntegrityError(
                    f"checkpoint metadata file does not match the manifest for "
                    f"{checkpoint.checkpoint_id!r}"
                )

            tensor_file = destination / expected_tensor_path
            if _sha256_file(tensor_file) != _text(entry, "tensor_sha256"):
                raise ArtifactIntegrityError(
                    f"tensor checksum mismatch for checkpoint {checkpoint.checkpoint_id!r}"
                )
            declared_nbytes = _integer(entry, "tensor_nbytes")
            if tensor_file.stat().st_size == 0:
                raise ArtifactIntegrityError(
                    f"tensor payload is empty for checkpoint {checkpoint.checkpoint_id!r}"
                )
            try:
                array = np.load(tensor_file, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ArtifactIntegrityError(
                    f"could not load tensor payload for checkpoint "
                    f"{checkpoint.checkpoint_id!r}: {error}"
                ) from error
            if not isinstance(array, np.ndarray):
                raise ArtifactIntegrityError(
                    f"tensor payload for checkpoint {checkpoint.checkpoint_id!r} is not an array"
                )
            _verify_payload_metadata(array, checkpoint, "artifact")
            if int(array.nbytes) != declared_nbytes:
                raise ArtifactIntegrityError(
                    f"tensor byte size for checkpoint {checkpoint.checkpoint_id!r} does not "
                    "match the manifest"
                )
            checkpoints.append(checkpoint)
            payloads[checkpoint.checkpoint_id] = array

        try:
            return ExecutionSession._restore(
                metadata=metadata,
                checkpoints=tuple(checkpoints),
                payloads=payloads,
                status=cast(Any, status),
                created_at=created_at,
                completed_at=completed_at,
                failure_reason=failure_reason,
                session_metadata=session_metadata,
            )
        except (CheckpointError, ExecutionStateError) as error:
            raise ArtifactIntegrityError(
                f"execution artifact is internally inconsistent: {error}"
            ) from error

    def save_validation(self, result: ValidationResult) -> Path:
        """Persist a validation result as a checksummed JSON artifact."""
        from ..validation.reporting import validation_to_json
        from ..validation.results import ValidationResult

        if not isinstance(result, ValidationResult):
            raise TypeError(f"result must be ValidationResult, received {type(result).__name__}")
        destination = self.validation_path(result.validation_id)
        if destination.exists():
            raise ArtifactError(
                f"validation artifact {destination} already exists; artifacts are never overwritten"
            )
        self.validations_dir.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{result.validation_id}.", dir=self.validations_dir))
        published = False
        try:
            document_path = stage / _VALIDATION_DOCUMENT_NAME
            _write_text(document_path, validation_to_json(result) + "\n")
            manifest = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "validation",
                "library": {"name": "tensortrail", "version": __version__},
                "validation_id": result.validation_id,
                "created_at": format_datetime(result.created_at),
                "status": result.status,
                "validation_path": _VALIDATION_DOCUMENT_NAME,
                "validation_sha256": _sha256_file(document_path),
                "validation_nbytes": int(document_path.stat().st_size),
            }
            _write_json(stage / _MANIFEST_NAME, manifest)
            _publish(stage, destination)
            published = True
            return destination
        finally:
            if not published:
                shutil.rmtree(stage, ignore_errors=True)

    def load_validation(self, validation_id: str) -> ValidationResult:
        """Load and verify a validation artifact."""
        from ..validation.reporting import load_validation_report

        identifier = validate_artifact_id(validation_id, "validation_id")
        destination = self.validation_path(identifier)
        if not destination.is_dir():
            raise ArtifactError(f"validation artifact directory {destination} does not exist")
        manifest = _read_json(destination / _MANIFEST_NAME)
        _require_keys(
            manifest,
            {
                "schema_version",
                "artifact_type",
                "library",
                "validation_id",
                "created_at",
                "status",
                "validation_path",
                "validation_sha256",
                "validation_nbytes",
            },
            set(),
            "validation manifest",
        )
        if _text(manifest, "schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactIntegrityError(
                f"validation manifest schema version {manifest['schema_version']!r} is "
                f"unsupported; expected {ARTIFACT_SCHEMA_VERSION!r}"
            )
        if _text(manifest, "artifact_type") != "validation":
            raise ArtifactIntegrityError("validation manifest artifact_type must be 'validation'")
        library = _mapping(manifest, "library")
        if _text(library, "name") != "tensortrail":
            raise ArtifactIntegrityError("validation library name must be 'tensortrail'")
        if _text(manifest, "validation_id") != identifier:
            raise ArtifactIntegrityError("validation manifest ID does not match its directory")
        if _text(manifest, "validation_path") != _VALIDATION_DOCUMENT_NAME:
            raise ArtifactIntegrityError("validation manifest has an unsafe document path")
        document_path = destination / _VALIDATION_DOCUMENT_NAME
        if _sha256_file(document_path) != _text(manifest, "validation_sha256"):
            raise ArtifactIntegrityError("validation document checksum mismatch")
        if document_path.stat().st_size != _integer(manifest, "validation_nbytes"):
            raise ArtifactIntegrityError("validation document byte size mismatch")
        result = load_validation_report(document_path)
        if result.validation_id != identifier:
            raise ArtifactIntegrityError(
                "validation document ID does not match the manifest validation ID"
            )
        if result.status != _text(manifest, "status"):
            raise ArtifactIntegrityError(
                "validation document status does not match the manifest status"
            )
        if format_datetime(result.created_at) != _text(manifest, "created_at"):
            raise ArtifactIntegrityError(
                "validation document creation time does not match the manifest"
            )
        return result

    def __repr__(self) -> str:
        return f"FilesystemStorage(root={self._root!s})"


def _payload_to_numpy(payload: object, checkpoint_name: str) -> NDArray[Any]:
    if looks_like_torch_tensor(payload):
        raise TensorValidationError(
            f"checkpoint {checkpoint_name!r} holds a PyTorch payload. Artifact persistence "
            "requires an explicit NumPy conversion; TensorTrail does not move or convert "
            "framework tensors implicitly. Register a NumPy array instead."
        )
    try:
        array = np.asarray(payload)
    except (TypeError, ValueError) as error:
        raise TensorValidationError(
            f"checkpoint {checkpoint_name!r} payload cannot be represented as a NumPy array: "
            f"{error}"
        ) from error
    if array.dtype.kind == "O":
        raise TensorValidationError(
            f"checkpoint {checkpoint_name!r} payload is an object array; object payloads "
            "cannot be persisted safely"
        )
    return array


def _verify_payload_metadata(
    array: NDArray[Any],
    checkpoint: TensorCheckpoint,
    source: str,
) -> None:
    if tuple(int(dimension) for dimension in array.shape) != checkpoint.tensor_shape:
        raise ArtifactIntegrityError(
            f"{source} payload shape for checkpoint {checkpoint.checkpoint_name!r} is "
            f"{tuple(array.shape)}, expected {checkpoint.tensor_shape}"
        )
    if array.dtype.name != checkpoint.tensor_dtype:
        raise ArtifactIntegrityError(
            f"{source} payload dtype for checkpoint {checkpoint.checkpoint_name!r} is "
            f"{array.dtype.name!r}, expected {checkpoint.tensor_dtype!r}"
        )
    if int(array.nbytes) != checkpoint.tensor_nbytes:
        raise ArtifactIntegrityError(
            f"{source} payload byte size for checkpoint {checkpoint.checkpoint_name!r} is "
            f"{int(array.nbytes)}, expected {checkpoint.tensor_nbytes}"
        )


def _publish(stage: Path, destination: Path) -> None:
    if destination.exists():
        raise ArtifactError(
            f"artifact {destination} already exists; artifacts are never overwritten"
        )
    try:
        stage.replace(destination)
    except OSError as error:
        raise ArtifactError(f"could not publish artifact {destination}: {error}") from error


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _plain_json(value))


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    text = json.dumps(
        _plain_json(document),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    _write_text(path, text + "\n")


def _write_text(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as error:
        raise ArtifactError(f"could not write artifact file {path}: {error}") from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ArtifactError(f"required artifact file {path} is missing") from error
    except OSError as error:
        raise ArtifactError(f"could not read artifact file {path}: {error}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError(f"artifact file {path} is not valid JSON: {error}") from error
    if not isinstance(document, Mapping):
        raise ArtifactIntegrityError(f"artifact file {path} must contain a JSON object")
    return cast(dict[str, Any], document)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise ArtifactError(f"required artifact file {path} is missing") from error
    except OSError as error:
        raise ArtifactError(f"could not read artifact file {path}: {error}") from error
    return digest.hexdigest()


def _require_keys(
    document: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = required - set(document)
    if missing:
        raise ArtifactIntegrityError(
            f"{context} is missing field(s): " + ", ".join(sorted(missing))
        )
    unknown = set(document) - required - optional
    if unknown:
        raise ArtifactIntegrityError(
            f"{context} has unknown field(s): " + ", ".join(sorted(unknown))
        )


def _mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _as_mapping(document[key], key)


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _text(document: Mapping[str, Any], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise ArtifactIntegrityError(f"{key} must be a string")
    return value


def _optional_text(document: Mapping[str, Any], key: str) -> str | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArtifactIntegrityError(f"{key} must be a string or null")
    return value


def _integer(document: Mapping[str, Any], key: str) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactIntegrityError(f"{key} must be an integer")
    return int(value)
