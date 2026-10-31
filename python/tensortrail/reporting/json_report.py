"""Stable JSON serialization for diagnostic reports."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from ..core.result import ComparisonReport

__all__ = ["report_to_dict", "report_to_json", "save_json_report"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def report_to_dict(report: ComparisonReport) -> dict[str, Any]:
    """Build the schema-versioned, JSON-safe report dictionary."""
    metadata = report.metadata
    document = {
        "schema_version": metadata.schema_version,
        "library": {
            "name": "tensortrail",
            "version": metadata.library_version,
        },
        "created_at": metadata.created_at,
        "status": report.status,
        "reference": report.reference.as_dict(),
        "candidate": report.candidate.as_dict(),
        "requested_metrics": list(metadata.requested_metrics),
        "tolerances": report.tolerances.as_dict(),
        "metrics": report.metrics.as_dict(metadata.requested_metrics),
        "statistics": report.statistics.as_dict() if report.statistics else None,
        "comparison": {
            **report.counts.as_dict(),
            "max_abs_error": report.max_abs_error,
        },
        "execution": {
            "backend": metadata.backend,
            "working_dtype": metadata.working_dtype,
            "python_version": metadata.python_version,
            "elapsed_seconds": metadata.elapsed_seconds,
            "normalization_seconds": metadata.normalization_seconds,
            "native_seconds": metadata.native_seconds,
            "parallel": metadata.parallel,
            "threads": metadata.threads,
            "chunks": metadata.chunks,
            "chunk_size": metadata.chunk_size,
            "release_gil": metadata.release_gil,
        },
    }
    return cast(dict[str, Any], _json_safe(document))


def report_to_json(report: ComparisonReport, *, indent: int = 2) -> str:
    """Serialize a report to strict JSON.

    Non-finite floats are encoded as string sentinels so the output remains
    valid JSON; see the numerical methodology documentation.
    """
    return json.dumps(
        report_to_dict(report),
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    )


def _atomic_write(destination: Path, text: str) -> Path:
    destination = destination.expanduser()
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        temporary_path.replace(destination)
    except BaseException:
        with suppress(OSError):
            temporary_path.unlink()
        raise
    return destination


def save_json_report(report: ComparisonReport, destination: Path) -> Path:
    """Write a report as strict, indented JSON using an atomic replace."""
    return _atomic_write(destination, report_to_json(report))
