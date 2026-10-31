"""Public exception hierarchy for TensorTrail."""

from __future__ import annotations

__all__ = [
    "BackendError",
    "TensorShapeError",
    "TensorTrailError",
    "TensorValidationError",
    "UnsupportedDtypeError",
]


class TensorTrailError(Exception):
    """Base class for every error raised by TensorTrail."""


class TensorValidationError(TensorTrailError, ValueError):
    """A tensor or configuration value is invalid."""


class TensorShapeError(TensorValidationError):
    """Reference and candidate tensors have incompatible shapes."""


class UnsupportedDtypeError(TensorValidationError, TypeError):
    """The supplied dtype cannot be represented by the comparison engine."""


class BackendError(TensorTrailError, RuntimeError):
    """A requested execution backend is unavailable or failed."""
