"""TensorTrail: high-performance numerical drift detection for ML inference.

Public import is lightweight: PyTorch and the native extension are loaded only
when required. The NumPy backend is always available; the Rust backend is the
default and raises an actionable error if its extension is missing.
"""

from __future__ import annotations

from ._version import __version__
from .api import (
    available_backends,
    compare,
    cosine_similarity,
    is_native_available,
    mae,
    mse,
    native_info,
    relative_error,
    rmse,
    statistics,
)
from .config import DEFAULT_METRICS, SUPPORTED_METRICS, ComparisonConfig
from .core.result import (
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
from .exceptions import (
    BackendError,
    TensorShapeError,
    TensorTrailError,
    TensorValidationError,
    UnsupportedDtypeError,
)

__all__ = [
    "DEFAULT_METRICS",
    "SCHEMA_VERSION",
    "SUPPORTED_METRICS",
    "BackendError",
    "ComparisonConfig",
    "ComparisonCounts",
    "ComparisonReport",
    "MetricValues",
    "NonFiniteCounts",
    "RelativeErrorValue",
    "ReportMetadata",
    "StatisticsPair",
    "TensorInfo",
    "TensorShapeError",
    "TensorStatistics",
    "TensorTrailError",
    "TensorValidationError",
    "Tolerances",
    "UnsupportedDtypeError",
    "__version__",
    "available_backends",
    "compare",
    "cosine_similarity",
    "is_native_available",
    "mae",
    "mse",
    "native_info",
    "relative_error",
    "rmse",
    "statistics",
]
