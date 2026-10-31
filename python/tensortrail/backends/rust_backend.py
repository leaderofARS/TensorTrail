"""Native Rust backend adapter.

The extension is loaded lazily. Importing :mod:`tensortrail` does not import the
native module, PyTorch, or any optional dependency. A missing extension raises
an actionable :class:`BackendError`; it never triggers a silent NumPy fallback.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._version import __version__
from ..backends import BackendResult
from ..config import ComparisonConfig
from ..core.result import NonFiniteCounts, TensorStatistics
from ..exceptions import BackendError, TensorValidationError

__all__ = ["compare", "is_available", "native_info", "statistics"]

Array = NDArray[Any]

_CORE: Any | None = None
_LOAD_ERROR: Exception | None = None


def _load_core() -> Any:
    global _CORE, _LOAD_ERROR
    if _CORE is not None:
        return _CORE
    if _LOAD_ERROR is not None:
        raise BackendError(str(_LOAD_ERROR)) from _LOAD_ERROR

    try:
        core = importlib.import_module("tensortrail._core")
    except Exception as error:  # ImportError, OSError, version mismatch handled below
        _LOAD_ERROR = BackendError(
            "TensorTrail's native extension is unavailable. Install a wheel built for "
            "this Python/platform, or build it from source with "
            "`maturin develop --release`. The NumPy reference backend remains available "
            "explicitly with backend='numpy'."
        )
        raise _LOAD_ERROR from error

    native_version = getattr(core, "__version__", None)
    if native_version != __version__:
        _LOAD_ERROR = BackendError(
            f"TensorTrail Python version {__version__} does not match native extension "
            f"version {native_version!r}. Reinstall the package to ensure matching artifacts."
        )
        raise _LOAD_ERROR

    _CORE = core
    return core


def is_available() -> bool:
    """Whether the native extension can be imported with a matching version."""
    try:
        _load_core()
    except BackendError:
        return False
    return True


def native_info() -> dict[str, Any]:
    """Return native build/runtime information (requires the extension)."""
    core = _load_core()
    info = core.native_info()
    return dict(info)


def _from_native_statistics(statistics: Any) -> TensorStatistics:
    return TensorStatistics(
        elements=int(statistics.elements),
        finite_elements=int(statistics.finite_elements),
        nonfinite=NonFiniteCounts(
            nan=int(statistics.nan_count),
            positive_infinity=int(statistics.positive_infinity_count),
            negative_infinity=int(statistics.negative_infinity_count),
        ),
        min=_optional_float(statistics.min),
        max=_optional_float(statistics.max),
        mean=_optional_float(statistics.mean),
        variance=_optional_float(statistics.variance),
        std_dev=_optional_float(statistics.std_dev),
    )


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _dispatch(core: Any, prefix: str, array: Array) -> Any:
    if array.dtype == np.float32:
        return getattr(core, f"{prefix}_f32")
    if array.dtype == np.float64:
        return getattr(core, f"{prefix}_f64")
    raise BackendError(
        f"native backend received unexpected working dtype {array.dtype}; this is an "
        "internal normalization bug"
    )


def compare(
    reference: Array,
    candidate: Array,
    config: ComparisonConfig,
) -> BackendResult:
    """Run the native fused comparison."""
    core = _load_core()
    function = _dispatch(core, "compare", reference)
    return _call_native(
        function,
        reference,
        candidate,
        config,
        compute_statistics="statistics" in config.metrics,
    )


def _call_native(
    function: Any,
    reference: Array,
    candidate: Array,
    config: ComparisonConfig,
    *,
    compute_statistics: bool,
) -> BackendResult:
    try:
        native = function(
            reference,
            candidate,
            parallel_threshold=config.parallel_threshold,
            chunk_size=config.chunk_size,
            num_threads=config.threads,
            epsilon=config.relative_epsilon,
            atol=config.atol,
            rtol=config.rtol,
            equal_nan=config.equal_nan,
            compute_statistics=compute_statistics,
            release_gil=config.release_gil,
        )
    except (ValueError, TypeError) as error:
        raise TensorValidationError(f"native comparison rejected the input: {error}") from error
    except Exception as error:  # pragma: no cover - defensive around extension boundary
        raise BackendError(f"native comparison failed: {error}") from error

    statistics_reference = (
        _from_native_statistics(native.statistics_reference)
        if native.statistics_reference is not None
        else None
    )
    statistics_candidate = (
        _from_native_statistics(native.statistics_candidate)
        if native.statistics_candidate is not None
        else None
    )

    return BackendResult(
        elements=int(native.elements),
        finite_elements=int(native.finite_elements),
        mae=_optional_float(native.mae),
        mse=_optional_float(native.mse),
        rmse=_optional_float(native.rmse),
        relative_error_mean=_optional_float(native.relative_error_mean),
        relative_error_max=_optional_float(native.relative_error_max),
        cosine_similarity=_optional_float(native.cosine_similarity),
        max_abs_error=_optional_float(native.max_abs_error),
        finite_mismatch_count=int(native.finite_mismatch_count),
        nonfinite_mismatch_count=int(native.nonfinite_mismatch_count),
        mismatch_count=int(native.mismatch_count),
        reference_nonfinite=NonFiniteCounts(
            nan=int(native.reference_nan_count),
            positive_infinity=int(native.reference_positive_infinity_count),
            negative_infinity=int(native.reference_negative_infinity_count),
        ),
        candidate_nonfinite=NonFiniteCounts(
            nan=int(native.candidate_nan_count),
            positive_infinity=int(native.candidate_positive_infinity_count),
            negative_infinity=int(native.candidate_negative_infinity_count),
        ),
        statistics_reference=statistics_reference,
        statistics_candidate=statistics_candidate,
        parallel=bool(native.parallel),
        threads=int(native.threads),
        chunks=int(native.chunks),
        chunk_size=int(native.chunk_size),
        native_seconds=float(native.native_seconds),
    )


def statistics(values: Array, config: ComparisonConfig) -> TensorStatistics:
    """Compute native tensor statistics.

    Statistics are independent of the comparison tolerance config; only thread
    and threshold controls are reused. The caller has already selected the
    native backend.
    """
    core = _load_core()
    function = _dispatch(core, "statistics", values)
    try:
        native = function(
            values,
            parallel_threshold=config.parallel_threshold,
            chunk_size=config.chunk_size,
            num_threads=config.threads,
            release_gil=config.release_gil,
        )
    except (ValueError, TypeError) as error:
        raise TensorValidationError(f"native statistics rejected the input: {error}") from error
    except Exception as error:  # pragma: no cover - defensive
        raise BackendError(f"native statistics failed: {error}") from error
    return _from_native_statistics(native)
