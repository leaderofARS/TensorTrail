"""Execution-level data model for inference validation.

This package owns execution metadata, checkpoint metadata, session lifecycle,
and local artifact persistence. It deliberately does not import PyTorch or the
native extension at package import time.
"""

from __future__ import annotations

from .artifacts import ExecutionStorage, FilesystemStorage
from .checkpoint import TensorCheckpoint, TensorDescription, describe_tensor
from .metadata import (
    EXECUTION_SCHEMA_VERSION,
    ExecutionMetadata,
    format_datetime,
    parse_datetime,
    utc_now,
)
from .registry import CheckpointRegistry
from .session import ExecutionSession, ExecutionStatus

__all__ = [
    "EXECUTION_SCHEMA_VERSION",
    "CheckpointRegistry",
    "ExecutionMetadata",
    "ExecutionSession",
    "ExecutionStatus",
    "ExecutionStorage",
    "FilesystemStorage",
    "TensorCheckpoint",
    "TensorDescription",
    "describe_tensor",
    "format_datetime",
    "parse_datetime",
    "utc_now",
]
