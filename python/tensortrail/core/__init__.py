"""Core result and provenance types."""

from .metadata import NormalizedTensor, normalize_pair, normalize_tensor
from .result import (
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

__all__ = [
    "SCHEMA_VERSION",
    "ComparisonCounts",
    "ComparisonReport",
    "MetricValues",
    "NonFiniteCounts",
    "NormalizedTensor",
    "RelativeErrorValue",
    "ReportMetadata",
    "StatisticsPair",
    "TensorInfo",
    "TensorStatistics",
    "Tolerances",
    "normalize_pair",
    "normalize_tensor",
]
