"""Cross-backend parity: the native engine against the NumPy reference.

The two backends use different summation orders (NumPy pairwise reductions vs.
deterministic chunk merges), so scalar metrics are compared with tolerances
derived from floating-point accumulation, not assumed to be bit-identical.
Element-wise quantities such as minimum, maximum, and ``max_abs_error`` are
exact because they do not accumulate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import tensortrail as tt
from tests.conftest import requires_native

pytestmark = requires_native

SIZES = [0, 1, 7, 4_099, 100_003]


def _config(backend: str) -> tt.ComparisonConfig:
    return tt.ComparisonConfig(
        backend=backend,
        metrics=tt.SUPPORTED_METRICS,
        parallel_threshold=1,
        chunk_size=1_024,
    )


def _dataset(
    dtype: np.dtype[Any], size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    reference = rng.normal(size=size).astype(dtype)
    candidate = (reference + rng.normal(scale=1e-3, size=size)).astype(dtype)
    return reference, candidate


def _assert_metric_parity(
    numpy_report: tt.ComparisonReport,
    rust_report: tt.ComparisonReport,
    *,
    rel: float,
    abs_: float,
) -> None:
    np_metrics = numpy_report.metrics
    rs_metrics = rust_report.metrics

    for name in ("mae", "mse", "rmse", "cosine_similarity"):
        expected = getattr(np_metrics, name)
        actual = getattr(rs_metrics, name)
        if expected is None:
            assert actual is None, name
            continue
        assert actual is not None, name
        if not math.isfinite(float(expected)):
            # Overflow is legitimate IEEE-754 behavior and must agree exactly.
            assert actual == expected, name
        else:
            assert actual == pytest.approx(expected, rel=rel, abs=abs_), name

    if np_metrics.relative_error is None or rs_metrics.relative_error is None:
        assert np_metrics.relative_error is None
        assert rs_metrics.relative_error is None
    else:
        for attribute in ("mean", "maximum"):
            expected_relative = getattr(np_metrics.relative_error, attribute)
            actual_relative = getattr(rs_metrics.relative_error, attribute)
            if not math.isfinite(expected_relative):
                assert actual_relative == expected_relative, attribute
            else:
                assert actual_relative == pytest.approx(
                    expected_relative, rel=rel, abs=abs_
                ), attribute

    assert rust_report.max_abs_error == numpy_report.max_abs_error
    assert rust_report.counts.elements == numpy_report.counts.elements
    assert rust_report.counts.finite_elements == numpy_report.counts.finite_elements
    assert rust_report.counts.finite_mismatches == numpy_report.counts.finite_mismatches
    assert rust_report.counts.nonfinite_mismatches == numpy_report.counts.nonfinite_mismatches
    assert rust_report.counts.mismatches == numpy_report.counts.mismatches
    assert rust_report.counts.reference_nonfinite == numpy_report.counts.reference_nonfinite
    assert rust_report.counts.candidate_nonfinite == numpy_report.counts.candidate_nonfinite
    assert rust_report.status == numpy_report.status


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("size", SIZES, ids=[f"n={size}" for size in SIZES])
def test_random_data_parity(dtype: np.dtype[Any], size: int, rng: np.random.Generator) -> None:
    reference, candidate = _dataset(dtype, size, rng)
    numpy_report = tt.compare(reference, candidate, config=_config("numpy"))
    rust_report = tt.compare(reference, candidate, config=_config("rust"))

    _assert_metric_parity(
        numpy_report,
        rust_report,
        rel=1e-9,
        abs_=1e-12,
    )


def _special_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [
        ("zeros", np.zeros(17), np.zeros(17)),
        ("identical", np.linspace(-1.0, 1.0, 33), np.linspace(-1.0, 1.0, 33)),
        ("tiny", np.array([1e-300, 2e-300, 3e-300]), np.array([2e-300, 4e-300, 6e-300])),
        (
            "huge",
            np.array([1e308, 5e307, 2e307]),
            np.array([1e308, 5e307, 2e307]),
        ),
        (
            "mixed_magnitudes",
            np.array([0.0, 1e-200, 1.0, 1e200, -1e308]),
            np.array([1e-12, 2e-200, -1.0, 1e199, 1e308]),
        ),
        (
            "opposite",
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, -2.0, -3.0]),
        ),
        (
            "zero_versus_nonzero",
            np.zeros(5),
            np.arange(1.0, 6.0),
        ),
    ]


@pytest.mark.parametrize(
    "name,reference,candidate",
    _special_cases(),
    ids=[case[0] for case in _special_cases()],
)
def test_special_magnitude_parity(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    del name
    numpy_report = tt.compare(reference, candidate, config=_config("numpy"))
    rust_report = tt.compare(reference, candidate, config=_config("rust"))
    _assert_metric_parity(numpy_report, rust_report, rel=1e-10, abs_=1e-14)


def test_nonfinite_pattern_parity() -> None:
    reference = np.array([1.0, np.nan, np.inf, -np.inf, 2.0, np.nan, -0.0, 4.0])
    candidate = np.array([1.0, np.nan, np.inf, np.inf, 3.0, 2.0, 0.0, 4.0])
    numpy_report = tt.compare(reference, candidate, config=_config("numpy"))
    rust_report = tt.compare(reference, candidate, config=_config("rust"))
    _assert_metric_parity(numpy_report, rust_report, rel=1e-12, abs_=1e-15)


def test_all_nonfinite_parity() -> None:
    reference = np.array([np.nan, np.inf, -np.inf])
    candidate = np.array([np.nan, np.inf, -np.inf])
    numpy_report = tt.compare(reference, candidate, config=_config("numpy"))
    rust_report = tt.compare(reference, candidate, config=_config("rust"))

    assert numpy_report.counts.finite_elements == 0
    assert rust_report.counts.finite_elements == 0
    assert numpy_report.metrics.mae is None
    assert rust_report.metrics.mae is None
    assert numpy_report.metrics.cosine_similarity is None
    assert rust_report.metrics.cosine_similarity is None
    assert numpy_report.status == rust_report.status == "pass"


def test_statistics_parity(rng: np.random.Generator) -> None:
    values = rng.normal(loc=1e6, scale=3.0, size=50_003)
    numpy_report = tt.compare(values, values, config=_config("numpy"))
    rust_report = tt.compare(values, values, config=_config("rust"))

    assert numpy_report.statistics is not None
    assert rust_report.statistics is not None
    for side in ("reference", "candidate"):
        expected = getattr(numpy_report.statistics, side)
        actual = getattr(rust_report.statistics, side)
        assert actual.elements == expected.elements
        assert actual.finite_elements == expected.finite_elements
        assert actual.nonfinite == expected.nonfinite
        assert actual.min == expected.min
        assert actual.max == expected.max
        assert actual.mean == pytest.approx(expected.mean, rel=1e-12, abs=1e-9)
        assert actual.variance == pytest.approx(expected.variance, rel=1e-9, abs=1e-9)
        assert actual.std_dev == pytest.approx(expected.std_dev, rel=1e-9, abs=1e-9)


def test_status_uses_identical_tolerance_boundary_on_both_backends() -> None:
    reference = np.array([1.0, 100.0, -3.0, 0.0])
    candidate = np.array([1.0 + 1e-6 + 1e-11, 100.0 * (1.0 + 1e-5) + 1e-9, -3.0, 0.0])
    config = tt.ComparisonConfig(
        backend="numpy",
        atol=1e-6,
        rtol=1e-5,
        metrics=("mae",),
    )
    numpy_report = tt.compare(reference, candidate, config=config)
    rust_report = tt.compare(
        reference,
        candidate,
        config=tt.ComparisonConfig(
            backend="rust",
            atol=1e-6,
            rtol=1e-5,
            metrics=("mae",),
        ),
    )
    assert numpy_report.counts.mismatches == rust_report.counts.mismatches
    assert numpy_report.status == rust_report.status


@pytest.mark.parametrize("equal_nan", [True, False])
def test_equal_nan_parity(equal_nan: bool) -> None:
    reference = np.array([np.nan, np.nan, 1.0])
    candidate = np.array([np.nan, 2.0, 1.0])
    numpy_report = tt.compare(
        reference,
        candidate,
        config=tt.ComparisonConfig(backend="numpy", equal_nan=equal_nan),
    )
    rust_report = tt.compare(
        reference,
        candidate,
        config=tt.ComparisonConfig(backend="rust", equal_nan=equal_nan),
    )
    assert numpy_report.counts.mismatches == rust_report.counts.mismatches
    assert numpy_report.status == rust_report.status


@pytest.mark.parametrize("epsilon", [1e-12, 1e-6, 1.0])
def test_relative_error_epsilon_parity(epsilon: float) -> None:
    reference = np.array([0.0, 1e-9, 1.0, -5.0])
    candidate = np.array([1e-6, 2e-9, 0.5, -5.0])
    numpy_report = tt.compare(
        reference,
        candidate,
        config=tt.ComparisonConfig(
            backend="numpy",
            metrics=("relative_error",),
            relative_epsilon=epsilon,
        ),
    )
    rust_report = tt.compare(
        reference,
        candidate,
        config=tt.ComparisonConfig(
            backend="rust",
            metrics=("relative_error",),
            relative_epsilon=epsilon,
        ),
    )
    assert numpy_report.metrics.relative_error is not None
    assert rust_report.metrics.relative_error is not None
    assert rust_report.metrics.relative_error.mean == pytest.approx(
        numpy_report.metrics.relative_error.mean, rel=1e-12
    )
    assert rust_report.metrics.relative_error.maximum == pytest.approx(
        numpy_report.metrics.relative_error.maximum, rel=1e-12
    )
