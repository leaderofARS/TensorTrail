"""NumPy reference backend.

This backend exists for parity testing, teaching the numerical policy, and
benchmark baselines. It is not the performance path and is never selected
implicitly. It promotes working arrays to float64 and may allocate several
temporary arrays proportional to input size.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backends import BackendResult
from ..config import ComparisonConfig
from ..core.result import NonFiniteCounts, TensorStatistics

__all__ = ["compare", "statistics"]

Array = NDArray[Any]

def _ignore_float_warnings() -> np.errstate:
    """A fresh errstate context; NumPy contexts are not re-entrant objects."""
    return np.errstate(over="ignore", invalid="ignore", divide="ignore")


def _nonfinite_counts(values: Array) -> NonFiniteCounts:
    return NonFiniteCounts(
        nan=int(np.count_nonzero(np.isnan(values))),
        positive_infinity=int(np.count_nonzero(np.isposinf(values))),
        negative_infinity=int(np.count_nonzero(np.isneginf(values))),
    )


def statistics(values: Array) -> TensorStatistics:
    """Compute reference statistics with a stable two-pass variance."""
    working = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(working)
    finite_values = working[finite_mask]
    count = int(finite_values.size)
    nonfinite = _nonfinite_counts(working)

    if count == 0:
        return TensorStatistics(
            elements=int(working.size),
            finite_elements=0,
            nonfinite=nonfinite,
            min=None,
            max=None,
            mean=None,
            variance=None,
            std_dev=None,
        )

    with _ignore_float_warnings():
        mean = float(np.mean(finite_values))
        centered = finite_values - mean
        variance = float(np.mean(centered * centered))
    return TensorStatistics(
        elements=int(working.size),
        finite_elements=count,
        nonfinite=nonfinite,
        min=float(np.min(finite_values)),
        max=float(np.max(finite_values)),
        mean=mean,
        variance=variance,
        std_dev=float(np.sqrt(variance)),
    )


def _scaled_cosine(reference: Array, candidate: Array) -> float | None:
    if reference.size == 0:
        return None
    reference_max = float(np.max(np.abs(reference)))
    candidate_max = float(np.max(np.abs(candidate)))
    if reference_max == 0.0 and candidate_max == 0.0:
        return 1.0
    if reference_max == 0.0 or candidate_max == 0.0:
        return 0.0

    normalized_reference = reference / reference_max
    normalized_candidate = candidate / candidate_max
    dot = float(np.dot(normalized_reference, normalized_candidate))
    reference_norm = float(np.dot(normalized_reference, normalized_reference))
    candidate_norm = float(np.dot(normalized_candidate, normalized_candidate))
    if reference_norm == 0.0 or candidate_norm == 0.0:
        return 0.0
    return float(np.clip(dot / np.sqrt(reference_norm * candidate_norm), -1.0, 1.0))


def compare(
    reference: Array,
    candidate: Array,
    config: ComparisonConfig,
) -> BackendResult:
    """Compare two normalized working arrays with NumPy."""
    reference_work = np.asarray(reference, dtype=np.float64)
    candidate_work = np.asarray(candidate, dtype=np.float64)

    finite_mask = np.isfinite(reference_work) & np.isfinite(candidate_work)
    finite_count = int(np.count_nonzero(finite_mask))
    elements = int(reference_work.size)

    reference_finite = reference_work[finite_mask]
    candidate_finite = candidate_work[finite_mask]

    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    relative_mean: float | None = None
    relative_max: float | None = None
    max_abs_error: float | None = None
    cosine: float | None = None
    finite_mismatches = 0

    if finite_count:
        with _ignore_float_warnings():
            absolute_difference = np.abs(reference_finite - candidate_finite)
            mae = float(np.mean(absolute_difference))
            squared_difference = absolute_difference * absolute_difference
            mse = float(np.mean(squared_difference))
            rmse = float(np.sqrt(mse))
            relative = absolute_difference / np.maximum(
                np.abs(reference_finite), config.relative_epsilon
            )
            relative_mean = float(np.mean(relative))
            relative_max = float(np.max(relative))
            max_abs_error = float(np.max(absolute_difference))
            tolerance = config.atol + config.rtol * np.abs(reference_finite)
            finite_mismatches = int(np.count_nonzero(absolute_difference > tolerance))
        cosine = _scaled_cosine(reference_finite, candidate_finite)

    nan_match = np.isnan(reference_work) & np.isnan(candidate_work) & config.equal_nan
    infinity_match = (
        np.isinf(reference_work) & np.isinf(candidate_work) & (reference_work == candidate_work)
    )
    nonfinite_positions = ~finite_mask
    nonfinite_mismatches = int(
        np.count_nonzero(nonfinite_positions & ~(nan_match | infinity_match))
    )
    reference_nonfinite = _nonfinite_counts(reference_work)
    candidate_nonfinite = _nonfinite_counts(candidate_work)

    statistics_requested = "statistics" in config.metrics
    statistics_reference = statistics(reference_work) if statistics_requested else None
    statistics_candidate = statistics(candidate_work) if statistics_requested else None

    return BackendResult(
        elements=elements,
        finite_elements=finite_count,
        mae=mae,
        mse=mse,
        rmse=rmse,
        relative_error_mean=relative_mean,
        relative_error_max=relative_max,
        cosine_similarity=cosine,
        max_abs_error=max_abs_error,
        finite_mismatch_count=finite_mismatches,
        nonfinite_mismatch_count=nonfinite_mismatches,
        mismatch_count=finite_mismatches + nonfinite_mismatches,
        reference_nonfinite=reference_nonfinite,
        candidate_nonfinite=candidate_nonfinite,
        statistics_reference=statistics_reference,
        statistics_candidate=statistics_candidate,
        parallel=False,
        threads=1,
        chunks=1 if elements else 0,
        chunk_size=elements,
        native_seconds=None,
    )
