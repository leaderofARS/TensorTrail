"""Public API behavior, configuration validation, and report accessors."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import tensortrail as tt
from tests.conftest import BACKEND_PARAMS, requires_native


def _config(backend: str = "numpy", **overrides: object) -> tt.ComparisonConfig:
    values: dict[str, object] = {"backend": backend}
    values.update(overrides)
    return tt.ComparisonConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_config_defaults(backend: str) -> None:
    config = tt.ComparisonConfig(backend=backend)
    assert config.metrics == tt.DEFAULT_METRICS
    assert config.atol == 1e-6
    assert config.rtol == 1e-5
    assert config.equal_nan is True
    assert config.relative_epsilon == 1e-12
    assert config.threads is None
    assert config.parallel_threshold == 32_768
    assert config.chunk_size == 32_768
    assert config.release_gil is True


def test_config_is_frozen_and_slotted() -> None:
    config = tt.ComparisonConfig(backend="numpy")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.atol = 1.0  # type: ignore[misc]
    assert not hasattr(config, "__dict__")


def test_config_normalizes_metric_names() -> None:
    config = tt.ComparisonConfig(backend="numpy", metrics=(" MAE ", "mse", "mae"))
    assert config.metrics == ("mae", "mse")
    assert tt.ComparisonConfig(backend="numpy", metrics="cosine_similarity").metrics == (
        "cosine_similarity",
    )
    assert tt.ComparisonConfig(backend="numpy", metrics=()).metrics == ()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"metrics": ("mae", "nope")}, "unknown metric"),
        ({"atol": -1.0}, "atol"),
        ({"atol": math.nan}, "atol"),
        ({"atol": True}, "atol"),
        ({"rtol": -1e-3}, "rtol"),
        ({"relative_epsilon": 0.0}, "relative_epsilon"),
        ({"relative_epsilon": -1e-12}, "relative_epsilon"),
        ({"threads": 0}, "threads"),
        ({"threads": 2.5}, "threads"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"parallel_threshold": 0}, "parallel_threshold"),
        ({"backend": "cuda"}, "unknown backend"),
        ({"equal_nan": 1}, "equal_nan"),
        ({"release_gil": "yes"}, "release_gil"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {"backend": "numpy"}
    values.update(kwargs)
    with pytest.raises(tt.TensorValidationError, match=message):
        tt.ComparisonConfig(**values)  # type: ignore[arg-type]


def test_unknown_backend_names_available_options() -> None:
    with pytest.raises(tt.TensorValidationError) as raised:
        tt.ComparisonConfig(backend="gpu")
    assert "rust" in str(raised.value)
    assert "numpy" in str(raised.value)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_metrics_argument_overrides_config(backend: str) -> None:
    report = tt.compare(
        np.array([1.0, 2.0]),
        np.array([2.0, 4.0]),
        metrics=("mae",),
        config=_config(backend, metrics=("mse",)),
    )
    assert report.metadata.requested_metrics == ("mae",)
    assert report.metrics.mae is not None
    assert report.metrics.mse is None
    assert report.metrics.rmse is None


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_status_is_independent_of_requested_metrics(backend: str) -> None:
    report = tt.compare(
        np.array([1.0, 2.0]),
        np.array([1.0, 9.0]),
        config=_config(backend, metrics=(), atol=0.0, rtol=0.0),
    )
    assert report.metadata.requested_metrics == ()
    assert report.metrics.mae is None
    assert report.counts.mismatches == 1
    assert report.status == "fail"
    assert report.diverged is True


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_statistics_metric_populates_pair_without_binary_values(backend: str) -> None:
    report = tt.compare(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        config=_config(backend, metrics=("statistics",)),
    )
    assert report.metrics.mae is None
    assert report.statistics is not None
    assert report.statistics.reference.mean == pytest.approx(2.0)
    assert report.statistics.candidate.mean == pytest.approx(2.0)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_failing_comparison_can_still_request_only_statistics(backend: str) -> None:
    report = tt.compare(
        np.array([1.0]),
        np.array([2.0]),
        config=_config(backend, metrics=("statistics",), atol=0.0, rtol=0.0),
    )
    assert report.status == "fail"
    assert report.statistics is not None


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_tensor_identifiers_are_stored(backend: str) -> None:
    report = tt.compare(
        np.ones(3),
        np.ones(3),
        config=_config(backend),
        reference_id="layer_7/reference",
        candidate_id="layer_7/candidate",
    )
    assert report.reference.id == "layer_7/reference"
    assert report.candidate.id == "layer_7/candidate"
    assert "layer_7/reference" in report.summary()


@pytest.mark.parametrize("identifier", ["", "   "])
def test_empty_identifiers_are_rejected(identifier: str) -> None:
    with pytest.raises(tt.TensorValidationError, match="non-empty string"):
        tt.compare(
            np.ones(2),
            np.ones(2),
            config=_config("numpy"),
            reference_id=identifier,
        )


def test_non_config_argument_is_rejected() -> None:
    with pytest.raises(TypeError, match="ComparisonConfig"):
        tt.compare(np.ones(2), np.ones(2), config={"backend": "numpy"})  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_metric_accessors(backend: str) -> None:
    report = tt.compare(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 4.0, 2.0]),
        config=_config(backend),
    )
    assert report.metrics["mae"] == report.metrics.mae
    assert report.metrics["relative_error"] is report.metrics.relative_error
    assert dict(report.metrics)["mse"] == report.metrics.mse
    with pytest.raises(KeyError, match="statistics"):
        report.metrics["statistics"]
    with pytest.raises(KeyError):
        report.metrics["nope"]


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_summary_contains_requested_metrics_and_counts(backend: str) -> None:
    report = tt.compare(
        np.array([1.0, 2.0]),
        np.array([1.0, 4.0]),
        config=_config(backend, metrics=("mae", "rmse", "relative_error")),
    )
    summary = report.summary()
    assert summary.startswith("TensorTrail comparison: FAIL")
    assert "mae:" in summary
    assert "rmse:" in summary
    assert "relative_error: mean=" in summary
    assert "mismatches: 1" in summary
    assert "backend: numpy" in summary or "backend: rust" in summary
    assert str(report) == summary


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_empty_metric_summary_is_explicit(backend: str) -> None:
    report = tt.compare(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        config=_config(backend, metrics=("mae", "relative_error")),
    )
    summary = report.summary()
    assert "mae: undefined" in summary
    assert "relative_error: undefined" in summary
    assert report.status == "pass"


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_convenience_metric_functions(backend: str) -> None:
    config = _config(backend)
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = np.array([1.0, 4.0, 2.0, 8.0])
    assert tt.mae(reference, candidate, config=config) == pytest.approx(1.75)
    assert tt.mse(reference, candidate, config=config) == pytest.approx(5.25)
    assert tt.rmse(reference, candidate, config=config) == pytest.approx(math.sqrt(5.25))
    relative = tt.relative_error(reference, candidate, config=config)
    assert relative is not None
    assert relative.mean == pytest.approx((0 + 1 + 1 / 3 + 1) / 4)
    assert relative.maximum == pytest.approx(1.0)
    cosine = tt.cosine_similarity(reference, candidate, config=config)
    assert cosine == pytest.approx(47.0 / math.sqrt(30.0 * 85.0))


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_convenience_functions_return_none_for_empty(backend: str) -> None:
    empty = np.array([], dtype=np.float64)
    config = _config(backend)
    assert tt.mae(empty, empty, config=config) is None
    assert tt.mse(empty, empty, config=config) is None
    assert tt.rmse(empty, empty, config=config) is None
    assert tt.relative_error(empty, empty, config=config) is None
    assert tt.cosine_similarity(empty, empty, config=config) is None


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_statistics_api(backend: str) -> None:
    stats = tt.statistics(np.array([1.0, 2.0, 3.0, 4.0]), config=_config(backend))
    assert stats.elements == 4
    assert stats.finite_elements == 4
    assert stats.min == 1.0
    assert stats.max == 4.0
    assert stats.mean == pytest.approx(2.5)
    assert stats.variance == pytest.approx(1.25)
    assert stats.std_dev == pytest.approx(math.sqrt(1.25))
    assert stats.has_nonfinite is False


def test_statistics_empty_are_undefined() -> None:
    stats = tt.statistics(np.array([], dtype=np.float64), config=_config("numpy"))
    assert stats.elements == 0
    assert stats.min is None
    assert stats.max is None
    assert stats.mean is None
    assert stats.variance is None
    assert stats.std_dev is None


def test_available_backends_always_contains_numpy() -> None:
    backends = tt.available_backends()
    assert "numpy" in backends
    assert isinstance(backends, tuple)
    if tt.is_native_available():
        assert "rust" in backends


@requires_native
def test_native_info_reports_build_details() -> None:
    info = tt.native_info()
    assert info["version"] == tt.__version__
    assert info["build_profile"] in {"debug", "release"}
    assert isinstance(info["rayon_threads"], int)
    assert info["rayon_threads"] >= 1
    assert info["simd"] == "scalar"


def test_exception_hierarchy() -> None:
    assert issubclass(tt.TensorValidationError, tt.TensorTrailError)
    assert issubclass(tt.TensorValidationError, ValueError)
    assert issubclass(tt.TensorShapeError, tt.TensorValidationError)
    assert issubclass(tt.UnsupportedDtypeError, tt.TensorValidationError)
    assert issubclass(tt.UnsupportedDtypeError, TypeError)
    assert issubclass(tt.BackendError, tt.TensorTrailError)
    assert issubclass(tt.BackendError, RuntimeError)


def test_version_is_semver() -> None:
    parts = tt.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
