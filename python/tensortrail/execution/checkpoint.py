"""Checkpoint metadata and metadata-only tensor description."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import numpy as np

from ..backends.pytorch_backend import describe_torch_tensor
from ..exceptions import TensorValidationError, UnsupportedDtypeError
from .metadata import (
    JsonValue,
    ensure_json_mapping,
    ensure_utc,
    format_datetime,
    new_checkpoint_id,
    parse_datetime,
    plain_json,
    utc_now,
    validate_artifact_id,
    validate_identifier,
    validate_optional_identifier,
)

__all__ = [
    "TensorCheckpoint",
    "TensorDescription",
    "describe_tensor",
]

_VALUE_SOURCES = ("numpy", "array-like", "torch", "artifact")
_REFERENCE_KINDS = ("memory", "artifact")


@dataclass(frozen=True, slots=True)
class TensorDescription:
    """Shape, dtype, and device metadata read without copying a payload."""

    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    source: str
    device: str


def describe_tensor(value: object, role: str) -> TensorDescription:
    """Read tensor metadata without normalizing or persisting the payload.

    NumPy arrays and numeric array-like objects are inspected with
    ``np.asarray`` but are not retained here. CPU PyTorch tensors are described
    without conversion. Accelerator tensors are rejected.
    """
    torch_description = describe_torch_tensor(value, role)
    if torch_description is not None:
        return TensorDescription(
            shape=torch_description.shape,
            dtype=torch_description.dtype,
            nbytes=torch_description.nbytes,
            source="torch",
            device=torch_description.device,
        )

    if isinstance(value, np.ma.MaskedArray):
        raise TensorValidationError(
            f"{role} is a masked array; TensorTrail does not silently ignore masks. "
            "Call `.filled(np.nan)` or remove the mask explicitly."
        )

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TensorValidationError(
            f"could not interpret {role} as a numeric array: {error}"
        ) from error

    if array.dtype.kind == "O":
        raise UnsupportedDtypeError(
            f"{role} produced an object array. Ragged sequences and mixed Python objects "
            "are not numeric tensors; pass a homogeneous numeric array."
        )

    return TensorDescription(
        shape=tuple(int(dimension) for dimension in array.shape),
        dtype=str(array.dtype.name),
        nbytes=int(array.nbytes),
        source="numpy" if isinstance(value, np.ndarray) else "array-like",
        device="cpu",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorCheckpoint:
    """Immutable metadata for one captured tensor.

    The tensor payload is deliberately not a field. An :class:`ExecutionSession`
    owns payloads separately and resolves them by ``checkpoint_id``.
    """

    execution_id: str
    checkpoint_name: str
    tensor_shape: tuple[int, ...]
    tensor_dtype: str
    tensor_nbytes: int
    checkpoint_id: str = field(default_factory=new_checkpoint_id)
    checkpoint_index: int = 0
    stage_name: str | None = None
    operation_type: str | None = None
    tensor_source: str = "numpy"
    tensor_device: str = "cpu"
    tensor_reference: str = "memory"
    capture_timestamp: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checkpoint_id", validate_artifact_id(self.checkpoint_id, "checkpoint_id")
        )
        object.__setattr__(
            self, "execution_id", validate_artifact_id(self.execution_id, "execution_id")
        )
        object.__setattr__(
            self,
            "checkpoint_name",
            validate_identifier(self.checkpoint_name, "checkpoint_name"),
        )
        object.__setattr__(
            self,
            "stage_name",
            validate_optional_identifier(self.stage_name, "stage_name"),
        )
        object.__setattr__(
            self,
            "operation_type",
            validate_optional_identifier(self.operation_type, "operation_type"),
        )
        object.__setattr__(
            self,
            "tensor_shape",
            _normalize_shape(self.tensor_shape),
        )
        dtype = validate_identifier(self.tensor_dtype, "tensor_dtype")
        object.__setattr__(
            self,
            "tensor_dtype",
            dtype.removeprefix("torch.").lower(),
        )
        object.__setattr__(
            self,
            "tensor_nbytes",
            _non_negative_int(self.tensor_nbytes, "tensor_nbytes"),
        )
        object.__setattr__(
            self,
            "checkpoint_index",
            _non_negative_int(self.checkpoint_index, "checkpoint_index"),
        )
        source = validate_identifier(self.tensor_source, "tensor_source").lower()
        if source not in _VALUE_SOURCES:
            raise TensorValidationError(
                f"tensor_source must be one of {', '.join(_VALUE_SOURCES)}; received {source!r}"
            )
        object.__setattr__(self, "tensor_source", source)
        object.__setattr__(
            self,
            "tensor_device",
            validate_identifier(self.tensor_device, "tensor_device").lower(),
        )
        reference = validate_identifier(self.tensor_reference, "tensor_reference").lower()
        if reference not in _REFERENCE_KINDS:
            raise TensorValidationError(
                f"tensor_reference must be 'memory' or 'artifact'; received {reference!r}"
            )
        object.__setattr__(self, "tensor_reference", reference)
        object.__setattr__(
            self,
            "capture_timestamp",
            ensure_utc(self.capture_timestamp, "capture_timestamp"),
        )
        object.__setattr__(
            self,
            "metadata",
            ensure_json_mapping(self.metadata, "checkpoint metadata"),
        )

    @property
    def elements(self) -> int:
        """Number of elements implied by the shape."""
        count = 1
        for dimension in self.tensor_shape:
            count *= dimension
        return count

    def to_dict(self) -> dict[str, Any]:
        """Return the documented, JSON-compatible checkpoint representation."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_index": self.checkpoint_index,
            "stage_name": self.stage_name,
            "operation_type": self.operation_type,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
            "tensor_nbytes": self.tensor_nbytes,
            "tensor_source": self.tensor_source,
            "tensor_device": self.tensor_device,
            "tensor_reference": self.tensor_reference,
            "capture_timestamp": format_datetime(self.capture_timestamp),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> TensorCheckpoint:
        """Parse checkpoint metadata, rejecting unknown fields."""
        required = {
            "checkpoint_id",
            "execution_id",
            "checkpoint_name",
            "checkpoint_index",
            "tensor_shape",
            "tensor_dtype",
            "tensor_nbytes",
            "tensor_source",
            "tensor_device",
            "tensor_reference",
            "capture_timestamp",
        }
        allowed = required | {"stage_name", "operation_type", "metadata"}
        unknown = set(document) - allowed
        if unknown:
            raise TensorValidationError(
                "unknown checkpoint field(s): " + ", ".join(sorted(unknown))
            )
        missing = required - set(document)
        if missing:
            raise TensorValidationError(
                "missing checkpoint field(s): " + ", ".join(sorted(missing))
            )

        shape = document["tensor_shape"]
        if not isinstance(shape, (list, tuple)):
            raise TensorValidationError("tensor_shape must be a list of non-negative integers")
        metadata = document.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TensorValidationError("checkpoint metadata must be a mapping")

        return cls(
            checkpoint_id=_required_text(document, "checkpoint_id"),
            execution_id=_required_text(document, "execution_id"),
            checkpoint_name=_required_text(document, "checkpoint_name"),
            checkpoint_index=_required_int(document, "checkpoint_index"),
            stage_name=_optional_text(document, "stage_name"),
            operation_type=_optional_text(document, "operation_type"),
            tensor_shape=tuple(_coerce_dimension(item, "tensor_shape") for item in shape),
            tensor_dtype=_required_text(document, "tensor_dtype"),
            tensor_nbytes=_required_int(document, "tensor_nbytes"),
            tensor_source=_required_text(document, "tensor_source"),
            tensor_device=_required_text(document, "tensor_device"),
            tensor_reference=_required_text(document, "tensor_reference"),
            capture_timestamp=parse_datetime(document["capture_timestamp"], "capture_timestamp"),
            metadata=cast(Mapping[str, JsonValue], metadata),
        )


def _normalize_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TensorValidationError("tensor_shape must be a tuple or list of dimensions")
    dimensions = tuple(_coerce_dimension(item, "tensor_shape") for item in value)
    return dimensions


def _coerce_dimension(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TensorValidationError(f"{field_name} dimensions must be integers")
    if value < 0:
        raise TensorValidationError(f"{field_name} dimensions must be >= 0")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TensorValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise TensorValidationError(f"{field_name} must be >= 0")
    return value


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


def _required_int(document: Mapping[str, Any], key: str) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TensorValidationError(f"{key} must be an integer")
    return int(value)
