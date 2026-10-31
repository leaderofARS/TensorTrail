"""Deterministic parallel execution and chunk-boundary coverage."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensortrail as tt
from tests.conftest import requires_native

pytestmark = requires_native

METRICS = ("mae", "mse", "rmse", "relative_error", "cosine_similarity", "statistics")


def _config(**overrides: object) -> tt.ComparisonConfig:
    values: dict[str, object] = {
        "backend": "rust",
        "metrics": METRICS,
        "parallel_threshold": 1,
        "chunk_size": 1_024,
    }
    values.update(overrides)
    return tt.ComparisonConfig(**values)  # type: ignore[arg-type]


def _data(size: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(size, dtype=np.float64)
    reference = np.sin(index * 0.001) + 1.0
    candidate = reference + np.cos(index * 0.0007) * 1e-4
    return reference, candidate


def test_parallel_results_are_identical_across_worker_counts() -> None:
    reference, candidate = _data(50_003)
    reports = [
        tt.compare(reference, candidate, config=_config(threads=threads))
        for threads in (2, 3, 4, 8)
    ]

    baseline = reports[0]
    assert baseline.metadata.parallel is True
    assert baseline.metadata.chunks == math.ceil(reference.size / 1_024)
    assert baseline.metadata.chunk_size == 1_024

    for report in reports[1:]:
        assert report.metadata.parallel is True
        assert report.metrics == baseline.metrics
        assert report.max_abs_error == baseline.max_abs_error
        assert report.counts == baseline.counts
        assert report.statistics == baseline.statistics


def test_reported_thread_count_matches_request() -> None:
    reference, candidate = _data(10_000)
    for threads in (2, 3, 5, 16):
        report = tt.compare(reference, candidate, config=_config(threads=threads))
        assert report.metadata.threads == threads


def test_chunk_tail_is_covered_exactly() -> None:
    reference = np.arange(10_001, dtype=np.float64)
    candidate = reference + 0.25
    report = tt.compare(
        reference,
        candidate,
        config=_config(chunk_size=1_024, threads=3, metrics=("mae", "mse", "rmse")),
    )
    assert report.metadata.parallel is True
    assert report.metadata.chunks == math.ceil(10_001 / 1_024)
    assert report.metrics.mae == pytest.approx(0.25, abs=1e-15)
    assert report.metrics.mse == pytest.approx(0.0625, abs=1e-15)
    assert report.metrics.rmse == pytest.approx(0.25, abs=1e-15)
    assert report.max_abs_error == pytest.approx(0.25, abs=1e-15)
    assert report.counts.finite_elements == 10_001


def test_below_threshold_runs_sequentially() -> None:
    reference, candidate = _data(100)
    report = tt.compare(
        reference,
        candidate,
        config=_config(parallel_threshold=1_000, chunk_size=16, threads=4),
    )
    assert report.metadata.parallel is False
    assert report.metadata.threads == 1
    assert report.metadata.chunks == 1
    assert report.metadata.chunk_size == 100


def test_one_worker_forces_the_sequential_path() -> None:
    reference, candidate = _data(20_000)
    report = tt.compare(
        reference,
        candidate,
        config=_config(threads=1, chunk_size=128),
    )
    assert report.metadata.parallel is False
    assert report.metadata.threads == 1
    assert report.metadata.chunks == 1
    assert report.metadata.chunk_size == 20_000


def test_single_chunk_is_not_parallel() -> None:
    reference, candidate = _data(500)
    report = tt.compare(
        reference,
        candidate,
        config=_config(chunk_size=1_024, threads=4),
    )
    assert report.metadata.parallel is False
    assert report.metadata.chunks == 1


def test_release_gil_flag_does_not_change_results() -> None:
    reference, candidate = _data(20_003)
    with_gil = tt.compare(reference, candidate, config=_config(threads=2, release_gil=False))
    without_gil = tt.compare(reference, candidate, config=_config(threads=2, release_gil=True))
    assert with_gil.metadata.release_gil is False
    assert without_gil.metadata.release_gil is True
    assert with_gil.metrics == without_gil.metrics
    assert with_gil.counts == without_gil.counts


def test_repeated_runs_are_stable() -> None:
    reference, candidate = _data(7_777)
    config = _config(threads=4, chunk_size=333)
    first = tt.compare(reference, candidate, config=config)
    for _ in range(3):
        repeated = tt.compare(reference, candidate, config=config)
        assert repeated.metrics == first.metrics
        assert repeated.statistics == first.statistics
        assert repeated.counts == first.counts


def test_worker_count_does_not_affect_statistics() -> None:
    values = np.sin(np.arange(40_001, dtype=np.float64) * 0.01) * 1e6 + 1e9
    reports = [
        tt.compare(
            values,
            values,
            config=_config(
                threads=threads,
                metrics=("statistics",),
                chunk_size=777,
            ),
        )
        for threads in (2, 4, 7)
    ]
    for report in reports[1:]:
        assert report.statistics == reports[0].statistics


def test_invalid_native_controls_are_rejected_before_execution() -> None:
    values = np.ones(4)
    with pytest.raises(tt.TensorValidationError, match="threads"):
        tt.ComparisonConfig(threads=0)
    with pytest.raises(tt.TensorValidationError, match="chunk_size"):
        tt.ComparisonConfig(chunk_size=0)
    with pytest.raises(tt.TensorValidationError, match="parallel_threshold"):
        tt.ComparisonConfig(parallel_threshold=0)
    with pytest.raises(tt.TensorValidationError):
        tt.compare(values, values, config=_config(threads=0))


def test_large_single_pass_result_matches_manual_formula() -> None:
    # candidate - reference is exactly 2.5 for every element, so MAE and RMSE
    # are independent of summation order.
    reference = np.linspace(-1.0, 1.0, 200_003)
    candidate = reference + 2.5
    report = tt.compare(
        reference,
        candidate,
        config=_config(chunk_size=4_096, threads=4),
    )
    assert report.metadata.parallel is True
    assert report.metrics.mae == 2.5
    assert report.metrics.rmse == 2.5
    assert report.metrics.mse == 6.25
    assert report.max_abs_error == 2.5
