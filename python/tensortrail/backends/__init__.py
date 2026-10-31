"""Backend result container and backend implementations."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.result import NonFiniteCounts, TensorStatistics

__all__ = ["BackendResult"]


@dataclass(frozen=True, slots=True)
class BackendResult:
    """Uniform result returned by native and reference backends."""

    elements: int
    finite_elements: int
    mae: float | None
    mse: float | None
    rmse: float | None
    relative_error_mean: float | None
    relative_error_max: float | None
    cosine_similarity: float | None
    max_abs_error: float | None
    finite_mismatch_count: int
    nonfinite_mismatch_count: int
    mismatch_count: int
    reference_nonfinite: NonFiniteCounts
    candidate_nonfinite: NonFiniteCounts
    statistics_reference: TensorStatistics | None
    statistics_candidate: TensorStatistics | None
    parallel: bool
    threads: int
    chunks: int
    chunk_size: int
    native_seconds: float | None
