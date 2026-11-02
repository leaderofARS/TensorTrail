"""Public exception hierarchy for TensorTrail."""

from __future__ import annotations

__all__ = [
    "AlignmentError",
    "ArtifactError",
    "ArtifactIntegrityError",
    "BackendError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "ExecutionStateError",
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


class ExecutionStateError(TensorTrailError, RuntimeError):
    """An execution session lifecycle transition is not legal."""


class CheckpointError(TensorValidationError):
    """A checkpoint is invalid or conflicts with an existing checkpoint."""


class CheckpointNotFoundError(CheckpointError, KeyError):
    """A checkpoint cannot be found by ID or name."""


class AlignmentError(TensorValidationError):
    """Checkpoint alignment is invalid, ambiguous, or incomplete."""


class ArtifactError(TensorTrailError, RuntimeError):
    """An execution or validation artifact cannot be read or written."""


class ArtifactIntegrityError(ArtifactError):
    """An artifact exists but fails an integrity check."""
