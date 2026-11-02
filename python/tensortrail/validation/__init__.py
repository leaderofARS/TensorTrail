"""Execution-level validation.

Alignment, policy, status aggregation, and JSON reporting for multi-checkpoint
inference validation. Every matched tensor pair is compared through the existing
numerical engine; this package does not reimplement metric logic.
"""

from __future__ import annotations

from .alignment import (
    CheckpointAlignment,
    CheckpointAlignmentEntry,
    align_checkpoints,
)
from .engine import aggregate_status, validate
from .policy import ValidationPolicy
from .reporting import (
    load_validation_report,
    save_validation_report,
    validation_from_dict,
    validation_to_dict,
    validation_to_json,
)
from .results import (
    VALIDATION_SCHEMA_VERSION,
    CheckpointOutcome,
    CheckpointValidationResult,
    ValidationResult,
    ValidationStatus,
)

__all__ = [
    "VALIDATION_SCHEMA_VERSION",
    "CheckpointAlignment",
    "CheckpointAlignmentEntry",
    "CheckpointOutcome",
    "CheckpointValidationResult",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationStatus",
    "aggregate_status",
    "align_checkpoints",
    "load_validation_report",
    "save_validation_report",
    "validate",
    "validation_from_dict",
    "validation_to_dict",
    "validation_to_json",
]
