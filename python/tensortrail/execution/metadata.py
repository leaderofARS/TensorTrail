"""Immutable execution metadata and shared validation helpers."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from ..exceptions import TensorValidationError

__all__ = [
    "EXECUTION_SCHEMA_VERSION",
    "ExecutionMetadata",
    "JsonScalar",
    "JsonValue",
    "ensure_json_mapping",
    "ensure_utc",
    "format_datetime",
    "new_checkpoint_id",
    "new_execution_id",
    "new_validation_id",
    "parse_datetime",
    "plain_json",
    "utc_now",
    "validate_artifact_id",
    "validate_identifier",
    "validate_optional_identifier",
]

EXECUTION_SCHEMA_VERSION = "1.0"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: object, field_name: str) -> datetime:
    """Validate an aware datetime and normalize it to UTC."""
    if not isinstance(value, datetime):
        raise TensorValidationError(
            f"{field_name} must be a datetime, received {type(value).__name__}"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise TensorValidationError(f"{field_name} must include timezone information")
    return value.astimezone(UTC)


def format_datetime(value: datetime) -> str:
    """Format a datetime as an ISO 8601 UTC string with a trailing ``Z``."""
    return ensure_utc(value, "datetime").isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_datetime(value: object, field_name: str) -> datetime:
    """Parse an ISO 8601 string, requiring explicit timezone information."""
    if not isinstance(value, str) or not value.strip():
        raise TensorValidationError(f"{field_name} must be a non-empty ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise TensorValidationError(f"{field_name} is not a valid ISO 8601 timestamp") from error
    return ensure_utc(parsed, field_name)


def validate_identifier(value: object, field_name: str) -> str:
    """Validate a non-empty identifier and return it without surrounding space."""
    if not isinstance(value, str) or not value.strip():
        raise TensorValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def validate_optional_identifier(value: object, field_name: str) -> str | None:
    """Validate an optional non-empty identifier."""
    if value is None:
        return None
    return validate_identifier(value, field_name)


def validate_artifact_id(value: object, field_name: str) -> str:
    """Validate an identifier that can safely be used as one path segment."""
    text = validate_identifier(value, field_name)
    if not _ARTIFACT_ID.fullmatch(text):
        raise TensorValidationError(
            f"{field_name} must start with an ASCII letter or digit and contain only "
            "letters, digits, '.', '_', or '-' (maximum 128 characters)"
        )
    return text


def new_execution_id() -> str:
    """Generate an artifact-safe execution identifier."""
    return f"exec_{uuid.uuid4().hex}"


def new_checkpoint_id() -> str:
    """Generate an artifact-safe checkpoint identifier."""
    return f"cp_{uuid.uuid4().hex}"


def new_validation_id() -> str:
    """Generate an artifact-safe validation identifier."""
    return f"val_{uuid.uuid4().hex}"


def _copy_json_value(value: object, field_name: str) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise TensorValidationError(
            f"{field_name} must use finite JSON numbers; non-finite values are not "
            "representable in persisted metadata"
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item, field_name) for item in value]
    if isinstance(value, Mapping):
        copied: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TensorValidationError(
                    f"{field_name} keys must be strings, received {type(key).__name__}"
                )
            copied[key] = _copy_json_value(item, field_name)
        return copied
    raise TensorValidationError(
        f"{field_name} must contain only JSON-compatible values; received {type(value).__name__}"
    )


def ensure_json_mapping(
    value: object,
    field_name: str,
) -> Mapping[str, JsonValue]:
    """Copy and validate a JSON-compatible mapping, exposing it read-only."""
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TensorValidationError(
            f"{field_name} must be a mapping, received {type(value).__name__}"
        )
    copied = _copy_json_value(value, field_name)
    if not isinstance(copied, dict):  # pragma: no cover - defensive
        raise TensorValidationError(f"{field_name} must be a mapping")
    return MappingProxyType(copied)


def plain_json(value: Mapping[str, JsonValue]) -> dict[str, Any]:
    """Convert a read-only metadata mapping to plain JSON-compatible objects."""
    return cast(dict[str, Any], _copy_json_value(value, "metadata"))


def _normalize_optional(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_identifier(value, field_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionMetadata:
    """Provenance and configuration associated with one inference execution.

    ``execution_id`` is generated when omitted. Required strings are validated
    and stripped. Framework, precision, and device type are normalized to lower
    case. Metadata mappings are copied and exposed read-only.
    """

    model_id: str
    model_version: str
    framework: str
    precision: str
    execution_id: str = field(default_factory=new_execution_id)
    model_fingerprint: str | None = None
    framework_version: str | None = None
    device_type: str | None = None
    device_name: str | None = None
    device_identifier: str | None = None
    input_identifier: str | None = None
    dataset_identifier: str | None = None
    execution_timestamp: datetime = field(default_factory=utc_now)
    execution_configuration: Mapping[str, JsonValue] = field(default_factory=dict)
    custom_metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_id",
            validate_artifact_id(self.execution_id, "execution_id"),
        )
        object.__setattr__(self, "model_id", validate_identifier(self.model_id, "model_id"))
        object.__setattr__(
            self, "model_version", validate_identifier(self.model_version, "model_version")
        )
        object.__setattr__(
            self,
            "framework",
            validate_identifier(self.framework, "framework").lower(),
        )
        object.__setattr__(
            self,
            "precision",
            validate_identifier(self.precision, "precision").lower(),
        )
        object.__setattr__(
            self,
            "model_fingerprint",
            _normalize_optional(self.model_fingerprint, "model_fingerprint"),
        )
        object.__setattr__(
            self,
            "framework_version",
            _normalize_optional(self.framework_version, "framework_version"),
        )
        object.__setattr__(
            self,
            "device_type",
            None
            if self.device_type is None
            else validate_identifier(self.device_type, "device_type").lower(),
        )
        object.__setattr__(
            self,
            "device_name",
            _normalize_optional(self.device_name, "device_name"),
        )
        object.__setattr__(
            self,
            "device_identifier",
            _normalize_optional(self.device_identifier, "device_identifier"),
        )
        object.__setattr__(
            self,
            "input_identifier",
            _normalize_optional(self.input_identifier, "input_identifier"),
        )
        object.__setattr__(
            self,
            "dataset_identifier",
            _normalize_optional(self.dataset_identifier, "dataset_identifier"),
        )
        object.__setattr__(
            self,
            "execution_timestamp",
            ensure_utc(self.execution_timestamp, "execution_timestamp"),
        )
        object.__setattr__(
            self,
            "execution_configuration",
            ensure_json_mapping(self.execution_configuration, "execution_configuration"),
        )
        object.__setattr__(
            self,
            "custom_metadata",
            ensure_json_mapping(self.custom_metadata, "custom_metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the documented, JSON-compatible metadata representation."""
        return {
            "execution_id": self.execution_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_fingerprint": self.model_fingerprint,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "device_identifier": self.device_identifier,
            "precision": self.precision,
            "input_identifier": self.input_identifier,
            "dataset_identifier": self.dataset_identifier,
            "execution_timestamp": format_datetime(self.execution_timestamp),
            "execution_configuration": plain_json(self.execution_configuration),
            "custom_metadata": plain_json(self.custom_metadata),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ExecutionMetadata:
        """Parse metadata, rejecting unknown fields and malformed values."""
        required = {
            "execution_id",
            "model_id",
            "model_version",
            "framework",
            "precision",
            "execution_timestamp",
        }
        allowed = required | {
            "model_fingerprint",
            "framework_version",
            "device_type",
            "device_name",
            "device_identifier",
            "input_identifier",
            "dataset_identifier",
            "execution_configuration",
            "custom_metadata",
        }
        unknown = set(document) - allowed
        if unknown:
            raise TensorValidationError(
                "unknown execution metadata field(s): " + ", ".join(sorted(unknown))
            )
        missing = required - set(document)
        if missing:
            raise TensorValidationError(
                "missing execution metadata field(s): " + ", ".join(sorted(missing))
            )

        def optional_text(name: str) -> str | None:
            value = document.get(name)
            if value is None:
                return None
            if not isinstance(value, str):
                raise TensorValidationError(f"{name} must be a string or null")
            return value

        configuration = document.get("execution_configuration", {})
        custom_metadata = document.get("custom_metadata", {})
        if not isinstance(configuration, Mapping):
            raise TensorValidationError("execution_configuration must be a mapping")
        if not isinstance(custom_metadata, Mapping):
            raise TensorValidationError("custom_metadata must be a mapping")

        return cls(
            execution_id=_required_text(document, "execution_id"),
            model_id=_required_text(document, "model_id"),
            model_version=_required_text(document, "model_version"),
            framework=_required_text(document, "framework"),
            precision=_required_text(document, "precision"),
            model_fingerprint=optional_text("model_fingerprint"),
            framework_version=optional_text("framework_version"),
            device_type=optional_text("device_type"),
            device_name=optional_text("device_name"),
            device_identifier=optional_text("device_identifier"),
            input_identifier=optional_text("input_identifier"),
            dataset_identifier=optional_text("dataset_identifier"),
            execution_timestamp=parse_datetime(
                document["execution_timestamp"], "execution_timestamp"
            ),
            execution_configuration=cast(Mapping[str, JsonValue], configuration),
            custom_metadata=cast(Mapping[str, JsonValue], custom_metadata),
        )


def _required_text(document: Mapping[str, Any], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise TensorValidationError(f"{key} must be a string")
    return value
