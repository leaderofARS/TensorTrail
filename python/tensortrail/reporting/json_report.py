"""Stable JSON serialization for diagnostic reports."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from ..core.result import (
    SCHEMA_VERSION,
    ComparisonCounts,
    ComparisonReport,
    MetricValues,
    NonFiniteCounts,
    RelativeErrorValue,
    ReportMetadata,
    StatisticsPair,
    TensorInfo,
    TensorStatistics,
    Tolerances,
)
from ..exceptions import TensorValidationError

__all__ = ["report_from_dict", "report_to_dict", "report_to_json", "save_json_report"]


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


def report_from_dict(document: Mapping[str, Any]) -> ComparisonReport:
    """Rebuild a comparison report from its strict JSON representation.

    Only the documented schema is accepted. Non-finite sentinels are decoded
    back to floats. This is a data parser; it never imports Python objects
    encoded in the document.
    """
    try:
        return _report_from_dict(document)
    except TensorValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise TensorValidationError(f"invalid comparison report document: {error}") from error


def _report_from_dict(document: Mapping[str, Any]) -> ComparisonReport:
    schema_version = _text(document, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise TensorValidationError(
            f"unsupported comparison report schema version {schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )

    library = _mapping(document, "library")
    if _text(library, "name") != "tensortrail":
        raise TensorValidationError("comparison report library name must be 'tensortrail'")

    requested_metrics = tuple(_text_list(document, "requested_metrics"))
    metrics_document = _mapping(document, "metrics")
    relative_document = metrics_document.get("relative_error")
    relative_error: RelativeErrorValue | None
    if relative_document is None:
        relative_error = None
    else:
        relative_mapping = _as_mapping(relative_document, "metrics.relative_error")
        relative_error = RelativeErrorValue(
            mean=_float(relative_mapping, "mean"),
            maximum=_float(relative_mapping, "max"),
        )

    metrics = MetricValues(
        mae=_optional_float(metrics_document.get("mae")),
        mse=_optional_float(metrics_document.get("mse")),
        rmse=_optional_float(metrics_document.get("rmse")),
        relative_error=relative_error,
        cosine_similarity=_optional_float(metrics_document.get("cosine_similarity")),
    )

    statistics_document = document.get("statistics")
    statistics: StatisticsPair | None
    if statistics_document is None:
        statistics = None
    else:
        statistics_mapping = _as_mapping(statistics_document, "statistics")
        statistics = StatisticsPair(
            reference=_statistics_from_dict(_mapping(statistics_mapping, "reference")),
            candidate=_statistics_from_dict(_mapping(statistics_mapping, "candidate")),
        )

    comparison = _mapping(document, "comparison")
    counts = ComparisonCounts(
        elements=_integer(comparison, "elements"),
        finite_elements=_integer(comparison, "finite_elements"),
        finite_mismatches=_integer(comparison, "finite_mismatch_count"),
        nonfinite_mismatches=_integer(comparison, "nonfinite_mismatch_count"),
        mismatches=_integer(comparison, "mismatch_count"),
        reference_nonfinite=_nonfinite_from_dict(_mapping(comparison, "reference_nonfinite")),
        candidate_nonfinite=_nonfinite_from_dict(_mapping(comparison, "candidate_nonfinite")),
    )

    tolerances = _mapping(document, "tolerances")
    execution = _mapping(document, "execution")
    metadata = ReportMetadata(
        schema_version=schema_version,
        library_version=_text(library, "version"),
        created_at=_text(document, "created_at"),
        backend=_text(execution, "backend"),
        requested_metrics=requested_metrics,
        working_dtype=_text(execution, "working_dtype"),
        elapsed_seconds=_float(execution, "elapsed_seconds"),
        normalization_seconds=_float(execution, "normalization_seconds"),
        native_seconds=_optional_float(execution.get("native_seconds")),
        parallel=_boolean(execution, "parallel"),
        threads=_integer(execution, "threads"),
        chunks=_integer(execution, "chunks"),
        chunk_size=_integer(execution, "chunk_size"),
        release_gil=_boolean(execution, "release_gil"),
        python_version=_text(execution, "python_version"),
    )

    report = ComparisonReport(
        reference=_tensor_info_from_dict(_mapping(document, "reference")),
        candidate=_tensor_info_from_dict(_mapping(document, "candidate")),
        metrics=metrics,
        statistics=statistics,
        counts=counts,
        max_abs_error=_optional_float(comparison.get("max_abs_error")),
        tolerances=Tolerances(
            atol=_float(tolerances, "atol"),
            rtol=_float(tolerances, "rtol"),
            equal_nan=_boolean(tolerances, "equal_nan"),
            relative_epsilon=_float(tolerances, "relative_epsilon"),
        ),
        metadata=metadata,
    )
    status = _text(document, "status")
    if status not in {"pass", "fail"}:
        raise TensorValidationError(f"unknown comparison status {status!r}")
    if report.status != status:
        raise TensorValidationError("comparison report status does not match its mismatch counts")
    return report


def _tensor_info_from_dict(document: Mapping[str, Any]) -> TensorInfo:
    shape = document["shape"]
    if not isinstance(shape, list):
        raise TensorValidationError("tensor shape must be a list")
    notes = document["conversion_notes"]
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        raise TensorValidationError("conversion_notes must be a list of strings")
    return TensorInfo(
        id=_text(document, "id"),
        shape=tuple(_dimension(item) for item in shape),
        dtype=_text(document, "dtype"),
        working_dtype=_text(document, "working_dtype"),
        device=_text(document, "device"),
        elements=_integer(document, "elements"),
        nbytes=_integer(document, "nbytes"),
        source=_text(document, "source"),
        conversion_notes=tuple(notes),
    )


def _statistics_from_dict(document: Mapping[str, Any]) -> TensorStatistics:
    return TensorStatistics(
        elements=_integer(document, "elements"),
        finite_elements=_integer(document, "finite_elements"),
        nonfinite=_nonfinite_from_dict(_mapping(document, "nonfinite")),
        min=_optional_float(document.get("min")),
        max=_optional_float(document.get("max")),
        mean=_optional_float(document.get("mean")),
        variance=_optional_float(document.get("variance")),
        std_dev=_optional_float(document.get("std_dev")),
    )


def _nonfinite_from_dict(document: Mapping[str, Any]) -> NonFiniteCounts:
    return NonFiniteCounts(
        nan=_integer(document, "nan"),
        positive_infinity=_integer(document, "positive_infinity"),
        negative_infinity=_integer(document, "negative_infinity"),
    )


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TensorValidationError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _as_mapping(document[key], key)


def _text(document: Mapping[str, Any], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise TensorValidationError(f"{key} must be a string")
    return value


def _text_list(document: Mapping[str, Any], key: str) -> list[str]:
    value = document[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TensorValidationError(f"{key} must be a list of strings")
    return value


def _integer(document: Mapping[str, Any], key: str) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TensorValidationError(f"{key} must be an integer")
    return int(value)


def _boolean(document: Mapping[str, Any], key: str) -> bool:
    value = document[key]
    if not isinstance(value, bool):
        raise TensorValidationError(f"{key} must be a bool")
    return value


def _float(document: Mapping[str, Any], key: str) -> float:
    return _decode_float(document[key], key)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _decode_float(value, "metric")


def _decode_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TensorValidationError(f"{name} must be a real number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        sentinels = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
        if value in sentinels:
            return sentinels[value]
    raise TensorValidationError(f"{name} must be a real number or a non-finite sentinel")


def _dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TensorValidationError("tensor shape must contain non-negative integers")
    return value


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
