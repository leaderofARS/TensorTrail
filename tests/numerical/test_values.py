"""Hand-calculated numerical policy tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensortrail as tt

from ..conftest import BACKEND_PARAMS


def config(backend: str, **overrides) -> tt.ComparisonConfig:
    return tt.ComparisonConfig(backend=backend, **overrides)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_hand_calculated_binary_metrics(backend: str) -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = np.array([1.0, 4.0, 2.0, 8.0])
    report = tt.compare(
        reference,
        candidate,
        config=config(
            backend,
            metrics=("mae", "mse", "rmse", "relative_error", "cosine_similarity"),
        ),
    )

    assert report.metrics.mae == pytest.approx(1.75)
    assert report.metrics.mse == pytest.approx(5.25)
    assert report.metrics.rmse == pytest.approx(math.sqrt(5.25))
    assert report.metrics.relative_error is not None
    assert report.metrics.relative_error.mean == pytest.approx((0 + 1 + 1 / 3 + 1) / 4)
    assert report.metrics.relative_error.maximum == pytest.approx(1.0)
    assert report.max_abs_error == pytest.approx(4.0)
    assert report.counts.elements == 4
    assert report.counts.finite_elements == 4
    assert report.counts.mismatches == 3  # the equal pair matches; the other three differ


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_hand_calculated_cosine(backend: str) -> None:
    reference = np.array([1.0, 2.0, 3.0])
    candidate = np.array([4.0, 5.0, 6.0])
    report = tt.compare(
        reference,
        candidate,
        config=config(backend, metrics=("cosine_similarity",)),
    )
    expected = 32.0 / math.sqrt(14.0 * 77.0)
    assert report.metrics.cosine_similarity == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_identical_tensors_pass(backend: str) -> None:
    values = np.array([0.0, -0.0, 1e-20, -3.5, 9.0], dtype=np.float64)
    report = tt.compare(values, values.copy(), config=config(backend))
    assert report.status == "pass"
    assert report.diverged is False
    assert report.metrics.mae == 0.0
    assert report.metrics.mse == 0.0
    assert report.metrics.rmse == 0.0
    assert report.metrics.cosine_similarity == pytest.approx(1.0)
    assert report.max_abs_error == 0.0
    assert report.counts.mismatches == 0


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_negative_values(backend: str) -> None:
    reference = np.array([-5.0, -1.0, 3.0])
    candidate = np.array([-2.0, -1.0, -3.0])
    report = tt.compare(
        reference,
        candidate,
        config=config(backend, metrics=("mae", "mse", "rmse")),
    )
    assert report.metrics.mae == pytest.approx((3 + 0 + 6) / 3)
    assert report.metrics.mse == pytest.approx((9 + 0 + 36) / 3)
    assert report.metrics.rmse == pytest.approx(math.sqrt(15))


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_zero_vectors_have_documented_cosine_policy(backend: str) -> None:
    zeros = np.zeros(8, dtype=np.float32)
    values = np.arange(1, 9, dtype=np.float32)
    both_zero = tt.compare(
        zeros,
        zeros,
        config=config(backend, metrics=("cosine_similarity",)),
    )
    one_zero = tt.compare(
        zeros,
        values,
        config=config(backend, metrics=("cosine_similarity",)),
    )
    mirrored = tt.compare(
        values,
        zeros,
        config=config(backend, metrics=("cosine_similarity",)),
    )
    assert both_zero.metrics.cosine_similarity == 1.0
    assert one_zero.metrics.cosine_similarity == 0.0
    assert mirrored.metrics.cosine_similarity == 0.0


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_empty_tensors_return_none_not_fake_zero(backend: str) -> None:
    empty = np.array([], dtype=np.float64)
    report = tt.compare(
        empty,
        empty.copy(),
        config=config(backend, metrics=tt.SUPPORTED_METRICS),
    )
    assert report.metrics.mae is None
    assert report.metrics.mse is None
    assert report.metrics.rmse is None
    assert report.metrics.relative_error is None
    assert report.metrics.cosine_similarity is None
    assert report.max_abs_error is None
    assert report.counts.elements == 0
    assert report.counts.finite_elements == 0
    assert report.counts.mismatches == 0
    assert report.status == "pass"
    assert report.statistics is not None
    assert report.statistics.reference.elements == 0
    assert report.statistics.reference.mean is None


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_single_element(backend: str) -> None:
    report = tt.compare(
        np.array([2.0]),
        np.array([2.5]),
        config=config(backend, metrics=("mae", "mse", "rmse", "relative_error")),
    )
    assert report.metrics.mae == pytest.approx(0.5)
    assert report.metrics.mse == pytest.approx(0.25)
    assert report.metrics.rmse == pytest.approx(0.5)
    assert report.metrics.relative_error is not None
    assert report.metrics.relative_error.mean == pytest.approx(0.25)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_nonfinite_values_are_excluded_and_counted(backend: str) -> None:
    reference = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    candidate = np.array([1.0, np.nan, np.inf, -np.inf, 3.0])
    report = tt.compare(
        reference,
        candidate,
        config=config(backend, metrics=("mae", "mse", "relative_error")),
    )
    assert report.metrics.mae == pytest.approx(0.5)
    assert report.metrics.mse == pytest.approx(0.5)
    assert report.counts.elements == 5
    assert report.counts.finite_elements == 2
    assert report.counts.reference_nonfinite.nan == 1
    assert report.counts.reference_nonfinite.positive_infinity == 1
    assert report.counts.reference_nonfinite.negative_infinity == 1
    assert report.counts.nonfinite_mismatches == 0
    assert report.counts.finite_mismatches == 1
    assert report.status == "fail"


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_nonfinite_pattern_mismatch(backend: str) -> None:
    reference = np.array([np.nan, np.inf, -np.inf, 1.0])
    candidate = np.array([1.0, np.inf, np.inf, 1.0])
    report = tt.compare(reference, candidate, config=config(backend))
    assert report.counts.nonfinite_mismatches == 2  # NaN vs finite, -inf vs +inf
    assert report.counts.mismatches >= 2
    assert report.status == "fail"


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_equal_nan_policy(backend: str) -> None:
    reference = np.array([np.nan])
    candidate = np.array([np.nan])
    pass_report = tt.compare(reference, candidate, config=config(backend))
    fail_report = tt.compare(
        reference,
        candidate,
        config=config(backend, equal_nan=False),
    )
    assert pass_report.status == "pass"
    assert fail_report.status == "fail"
    assert pass_report.metrics.mae is None
    assert pass_report.counts.finite_elements == 0


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_absolute_tolerance_boundary(backend: str) -> None:
    reference = np.array([1.0])
    candidate = np.array([1.125])
    passing = tt.compare(
        reference,
        candidate,
        config=config(backend, atol=0.125, rtol=0.0),
    )
    failing = tt.compare(
        reference,
        candidate,
        config=config(backend, atol=0.124, rtol=0.0),
    )
    assert passing.status == "pass"
    assert failing.status == "fail"


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_relative_tolerance_uses_reference_magnitude(backend: str) -> None:
    reference = np.array([100.0])
    candidate = np.array([101.0])
    passing = tt.compare(
        reference,
        candidate,
        config=config(backend, atol=0.0, rtol=0.01),
    )
    failing = tt.compare(
        reference,
        candidate,
        config=config(backend, atol=0.0, rtol=0.009),
    )
    assert passing.status == "pass"
    assert failing.status == "fail"


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_relative_error_zero_denominator_floor(backend: str) -> None:
    report = tt.compare(
        np.array([0.0]),
        np.array([1e-6]),
        config=config(backend, metrics=("relative_error",), relative_epsilon=1e-12),
    )
    assert report.metrics.relative_error is not None
    assert report.metrics.relative_error.mean == pytest.approx(1e6)
    assert math.isfinite(report.metrics.relative_error.mean)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_tiny_values_cosine_is_stable(backend: str) -> None:
    reference = np.array([1e-300, 2e-300, 3e-300])
    candidate = reference * 2.0
    report = tt.compare(
        reference,
        candidate,
        config=config(backend, metrics=("cosine_similarity",)),
    )
    assert report.metrics.cosine_similarity == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_huge_values_cosine_is_stable(backend: str) -> None:
    reference = np.array([1e308, 5e307, 2e307])
    candidate = np.array([1e308, 5e307, 2e307])
    report = tt.compare(
        reference,
        candidate,
        config=config(backend, metrics=("cosine_similarity", "mae")),
    )
    assert report.metrics.cosine_similarity == pytest.approx(1.0, abs=1e-14)
    assert report.metrics.mae == 0.0


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_statistics_hand_values(backend: str) -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    stats = tt.statistics(values, config=config(backend))
    assert stats.elements == 4
    assert stats.finite_elements == 4
    assert stats.min == 1.0
    assert stats.max == 4.0
    assert stats.mean == pytest.approx(2.5)
    assert stats.variance == pytest.approx(1.25)
    assert stats.std_dev == pytest.approx(math.sqrt(1.25))


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_statistics_exclude_nonfinite(backend: str) -> None:
    stats = tt.statistics(
        np.array([1.0, np.nan, 3.0, np.inf, -np.inf]),
        config=config(backend),
    )
    assert stats.elements == 5
    assert stats.finite_elements == 2
    assert stats.nonfinite.nan == 1
    assert stats.nonfinite.positive_infinity == 1
    assert stats.nonfinite.negative_infinity == 1
    assert stats.mean == pytest.approx(2.0)
    assert stats.variance == pytest.approx(1.0)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_mixed_precision_uses_float64(backend: str) -> None:
    report = tt.compare(
        np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float64),
        config=config(backend),
    )
    assert report.metadata.working_dtype == "float64"
    assert "upcast_to_float64_for_mixed_dtypes" in report.reference.conversion_notes
    assert report.candidate.dtype == "float64"


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_integer_inputs_are_supported(backend: str) -> None:
    report = tt.compare(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 4, 3], dtype=np.int64),
        config=config(backend, metrics=("mae", "mse")),
    )
    assert report.reference.dtype == "int64"
    assert report.reference.working_dtype == "float64"
    assert "int64_to_float64" in report.reference.conversion_notes
    assert report.metrics.mae == pytest.approx(2 / 3)
    assert report.metrics.mse == pytest.approx(4 / 3)


def test_shape_mismatch_is_actionable() -> None:
    with pytest.raises(tt.TensorShapeError, match="shapes must match exactly"):
        tt.compare(np.zeros((2, 3)), np.zeros((3, 2)))


def test_scalar_inputs() -> None:
    report = tt.compare(np.array(2.0), np.array(2.5), config=config("numpy"))
    assert report.reference.shape == ()
    assert report.reference.elements == 1
    assert report.metrics.mae == pytest.approx(0.5)
