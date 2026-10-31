"""Comparison orchestration: normalize inputs, dispatch, assemble a report."""

from __future__ import annotations

import platform
import time
from datetime import UTC, datetime
from typing import Any

from numpy.typing import NDArray

from ..backends import BackendResult, numpy_backend, rust_backend
from ..config import ComparisonConfig
from ..core.metadata import normalize_pair, normalize_tensor
from ..core.result import (
    ComparisonCounts,
    ComparisonReport,
    MetricValues,
    RelativeErrorValue,
    StatisticsPair,
    TensorStatistics,
    Tolerances,
    make_metadata,
)
from ..exceptions import BackendError, TensorValidationError

__all__ = ["compare", "statistics"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validate_id(value: str, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TensorValidationError(f"{role}_id must be a non-empty string")
    return value


def _dispatch_backend(
    reference_work: NDArray[Any],
    candidate_work: NDArray[Any],
    config: ComparisonConfig,
) -> BackendResult:
    if config.backend == "rust":
        return rust_backend.compare(reference_work, candidate_work, config)
    if config.backend == "numpy":
        return numpy_backend.compare(reference_work, candidate_work, config)
    raise BackendError(
        f"backend {config.backend!r} was not resolved; TensorTrail never silently "
        "falls back between backends"
    )


def compare(
    reference: object,
    candidate: object,
    config: ComparisonConfig,
    *,
    reference_id: str = "reference",
    candidate_id: str = "candidate",
) -> ComparisonReport:
    """Normalize, execute, and package a comparison. Used by the public API."""
    reference_id = _validate_id(reference_id, "reference")
    candidate_id = _validate_id(candidate_id, "candidate")

    started = time.perf_counter()
    reference_tensor, candidate_tensor, working_dtype = normalize_pair(
        reference,
        candidate,
        reference_id=reference_id,
        candidate_id=candidate_id,
    )
    normalization_seconds = time.perf_counter() - started

    backend_started = time.perf_counter()
    result = _dispatch_backend(reference_tensor.array, candidate_tensor.array, config)
    elapsed_seconds = time.perf_counter() - backend_started

    metrics = MetricValues(
        mae=result.mae if "mae" in config.metrics else None,
        mse=result.mse if "mse" in config.metrics else None,
        rmse=result.rmse if "rmse" in config.metrics else None,
        relative_error=(
            RelativeErrorValue(
                mean=result.relative_error_mean,
                maximum=result.relative_error_max,
            )
            if "relative_error" in config.metrics
            and result.relative_error_mean is not None
            and result.relative_error_max is not None
            else None
        ),
        cosine_similarity=(
            result.cosine_similarity if "cosine_similarity" in config.metrics else None
        ),
    )

    statistics: StatisticsPair | None = None
    if "statistics" in config.metrics:
        if result.statistics_reference is None or result.statistics_candidate is None:
            raise BackendError(
                "backend did not return requested statistics; this is a backend contract bug"
            )
        statistics = StatisticsPair(
            reference=result.statistics_reference,
            candidate=result.statistics_candidate,
        )

    counts = ComparisonCounts(
        elements=result.elements,
        finite_elements=result.finite_elements,
        finite_mismatches=result.finite_mismatch_count,
        nonfinite_mismatches=result.nonfinite_mismatch_count,
        mismatches=result.mismatch_count,
        reference_nonfinite=result.reference_nonfinite,
        candidate_nonfinite=result.candidate_nonfinite,
    )

    metadata = make_metadata(
        backend=config.backend,
        requested_metrics=config.metrics,
        working_dtype=working_dtype,
        elapsed_seconds=elapsed_seconds,
        normalization_seconds=normalization_seconds,
        native_seconds=result.native_seconds,
        parallel=result.parallel,
        threads=result.threads,
        chunks=result.chunks,
        chunk_size=result.chunk_size,
        release_gil=config.release_gil,
        created_at=_utc_now(),
        python_version=platform.python_version(),
    )

    return ComparisonReport(
        reference=reference_tensor.info,
        candidate=candidate_tensor.info,
        metrics=metrics,
        statistics=statistics,
        counts=counts,
        max_abs_error=result.max_abs_error,
        tolerances=Tolerances(
            atol=config.atol,
            rtol=config.rtol,
            equal_nan=config.equal_nan,
            relative_epsilon=config.relative_epsilon,
        ),
        metadata=metadata,
    )


def statistics(
    values: object,
    config: ComparisonConfig,
    *,
    tensor_id: str = "tensor",
) -> TensorStatistics:
    """Normalize one tensor and compute backend-specific statistics."""
    tensor_id = _validate_id(tensor_id, "tensor")
    tensor = normalize_tensor(values, "values", tensor_id)
    if config.backend == "rust":
        return rust_backend.statistics(tensor.array, config)
    if config.backend == "numpy":
        return numpy_backend.statistics(tensor.array)
    raise BackendError(f"unknown backend {config.backend!r}")
