"""Stable, typed result objects returned by TensorTrail."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

from .._version import __version__

__all__ = [
    "ComparisonCounts",
    "ComparisonReport",
    "MetricValues",
    "NonFiniteCounts",
    "RelativeErrorValue",
    "ReportMetadata",
    "StatisticsPair",
    "TensorInfo",
    "TensorStatistics",
    "Tolerances",
]

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class NonFiniteCounts:
    """Counts of NaN and infinity values on one side of a comparison."""

    nan: int = 0
    positive_infinity: int = 0
    negative_infinity: int = 0

    @property
    def total(self) -> int:
        """Total number of non-finite values."""
        return self.nan + self.positive_infinity + self.negative_infinity

    def as_dict(self) -> dict[str, int]:
        return {
            "nan": self.nan,
            "positive_infinity": self.positive_infinity,
            "negative_infinity": self.negative_infinity,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class TensorStatistics:
    """Summary statistics for one tensor.

    Min, max, mean, variance, and standard deviation are computed over finite
    elements only. Variance is the population variance (denominator ``n``).
    """

    elements: int
    finite_elements: int
    nonfinite: NonFiniteCounts
    min: float | None
    max: float | None
    mean: float | None
    variance: float | None
    std_dev: float | None

    @property
    def has_nonfinite(self) -> bool:
        return self.nonfinite.total > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "elements": self.elements,
            "finite_elements": self.finite_elements,
            "nonfinite": self.nonfinite.as_dict(),
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "variance": self.variance,
            "std_dev": self.std_dev,
        }


@dataclass(frozen=True, slots=True)
class StatisticsPair:
    """Reference and candidate statistics."""

    reference: TensorStatistics
    candidate: TensorStatistics

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.as_dict(),
            "candidate": self.candidate.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RelativeErrorValue:
    """Mean and maximum relative error."""

    mean: float
    maximum: float

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "max": self.maximum}


@dataclass(frozen=True, slots=True)
class MetricValues:
    """Requested numerical metrics.

    A field is ``None`` when the metric was not requested or when it is
    mathematically undefined (for example, metrics over empty tensors).
    ``report.metadata.requested_metrics`` distinguishes the two cases.
    """

    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    relative_error: RelativeErrorValue | None = None
    cosine_similarity: float | None = None

    def __getitem__(self, name: str) -> float | RelativeErrorValue | None:
        if name == "statistics":
            raise KeyError("statistics are available as report.statistics, not report.metrics")
        if name not in {field.name for field in fields(self)}:
            raise KeyError(name)
        value = getattr(self, name)
        return cast(float | RelativeErrorValue | None, value)

    def __iter__(self) -> Iterator[tuple[str, float | RelativeErrorValue | None]]:
        for name in ("mae", "mse", "rmse", "relative_error", "cosine_similarity"):
            value = getattr(self, name)
            yield name, cast(float | RelativeErrorValue | None, value)

    def as_dict(self, requested: tuple[str, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in requested:
            if name == "statistics":
                continue
            value = getattr(self, name)
            result[name] = value.as_dict() if isinstance(value, RelativeErrorValue) else value
        return result


@dataclass(frozen=True, slots=True)
class ComparisonCounts:
    """Element counts and tolerance outcomes for a comparison."""

    elements: int
    finite_elements: int
    finite_mismatches: int
    nonfinite_mismatches: int
    mismatches: int
    reference_nonfinite: NonFiniteCounts
    candidate_nonfinite: NonFiniteCounts

    def as_dict(self) -> dict[str, Any]:
        return {
            "elements": self.elements,
            "finite_elements": self.finite_elements,
            "finite_mismatch_count": self.finite_mismatches,
            "nonfinite_mismatch_count": self.nonfinite_mismatches,
            "mismatch_count": self.mismatches,
            "reference_nonfinite": self.reference_nonfinite.as_dict(),
            "candidate_nonfinite": self.candidate_nonfinite.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Tolerances:
    """Tolerance policy attached to a report."""

    atol: float
    rtol: float
    equal_nan: bool
    relative_epsilon: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "atol": self.atol,
            "rtol": self.rtol,
            "equal_nan": self.equal_nan,
            "relative_epsilon": self.relative_epsilon,
            "rule": "abs(reference - candidate) <= atol + rtol * abs(reference)",
        }


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Provenance and layout information for one input tensor."""

    id: str
    shape: tuple[int, ...]
    dtype: str
    working_dtype: str
    device: str
    elements: int
    nbytes: int
    source: str
    conversion_notes: tuple[str, ...] = ()

    @property
    def converted(self) -> bool:
        """Whether TensorTrail changed dtype or memory layout."""
        return bool(self.conversion_notes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "working_dtype": self.working_dtype,
            "device": self.device,
            "elements": self.elements,
            "nbytes": self.nbytes,
            "source": self.source,
            "converted": self.converted,
            "conversion_notes": list(self.conversion_notes),
        }


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    """Execution and environment metadata for a report."""

    schema_version: str
    library_version: str
    created_at: str
    backend: str
    requested_metrics: tuple[str, ...]
    working_dtype: str
    elapsed_seconds: float
    normalization_seconds: float
    native_seconds: float | None
    parallel: bool
    threads: int
    chunks: int
    chunk_size: int
    release_gil: bool
    python_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "library_version": self.library_version,
            "created_at": self.created_at,
            "backend": self.backend,
            "requested_metrics": list(self.requested_metrics),
            "working_dtype": self.working_dtype,
            "elapsed_seconds": self.elapsed_seconds,
            "normalization_seconds": self.normalization_seconds,
            "native_seconds": self.native_seconds,
            "parallel": self.parallel,
            "threads": self.threads,
            "chunks": self.chunks,
            "chunk_size": self.chunk_size,
            "release_gil": self.release_gil,
            "python_version": self.python_version,
        }


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Complete result of a TensorTrail comparison.

    ``status`` is derived only from the tolerance policy and non-finite pattern
    matching. It does not depend on which metrics were requested.
    """

    reference: TensorInfo
    candidate: TensorInfo
    metrics: MetricValues
    statistics: StatisticsPair | None
    counts: ComparisonCounts
    max_abs_error: float | None
    tolerances: Tolerances
    metadata: ReportMetadata

    @property
    def status(self) -> Literal["pass", "fail"]:
        return "pass" if self.counts.mismatches == 0 else "fail"

    @property
    def diverged(self) -> bool:
        return self.status == "fail"

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"TensorTrail comparison: {self.status.upper()}",
            f"  reference: '{self.reference.id}' shape={self.reference.shape} "
            f"dtype={self.reference.dtype} device={self.reference.device}",
            f"  candidate: '{self.candidate.id}' shape={self.candidate.shape} "
            f"dtype={self.candidate.dtype} device={self.candidate.device}",
        ]

        for name in self.metadata.requested_metrics:
            if name == "statistics":
                continue
            value = getattr(self.metrics, name)
            if name == "relative_error":
                if value is None:
                    lines.append("  relative_error: undefined (no finite pairs)")
                else:
                    lines.append(
                        f"  relative_error: mean={_format(value.mean)} max={_format(value.maximum)}"
                    )
            else:
                lines.append(f"  {name}: {_format(value)}")

        lines.extend(
            [
                f"  max_abs_error: {_format(self.max_abs_error)}",
                f"  elements: {self.counts.elements} "
                f"(finite pairs: {self.counts.finite_elements})",
                f"  mismatches: {self.counts.mismatches} "
                f"({self.counts.finite_mismatches} finite, "
                f"{self.counts.nonfinite_mismatches} non-finite pattern)",
                f"  tolerance: atol={_format(self.tolerances.atol)} "
                f"rtol={_format(self.tolerances.rtol)} "
                f"equal_nan={self.tolerances.equal_nan}",
                f"  backend: {self.metadata.backend} "
                f"threads={self.metadata.threads} "
                f"elapsed={_format(self.metadata.elapsed_seconds)}s",
            ]
        )

        reference_nonfinite = self.counts.reference_nonfinite
        candidate_nonfinite = self.counts.candidate_nonfinite
        if reference_nonfinite.total or candidate_nonfinite.total:
            lines.append(
                "  non-finite reference: "
                f"nan={reference_nonfinite.nan} +inf={reference_nonfinite.positive_infinity} "
                f"-inf={reference_nonfinite.negative_infinity}"
            )
            lines.append(
                "  non-finite candidate: "
                f"nan={candidate_nonfinite.nan} +inf={candidate_nonfinite.positive_infinity} "
                f"-inf={candidate_nonfinite.negative_infinity}"
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary using the documented report schema."""
        from ..reporting.json_report import report_to_dict

        return report_to_dict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to strict JSON.

        Non-finite metric values (only possible through extreme overflow) are
        encoded as the strings ``"NaN"``, ``"Infinity"``, and ``"-Infinity"``.
        """
        from ..reporting.json_report import report_to_json

        return report_to_json(self, indent=indent)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        """Write a JSON (``.json``) or HTML (``.html``) report.

        The extension is used when ``format`` is not supplied. Reports are
        written atomically (temporary file + replace).
        """
        destination = Path(path)
        selected = format.lower().lstrip(".") if format else destination.suffix.lower().lstrip(".")
        if not selected:
            selected = "json"
        if selected == "json":
            from ..reporting.json_report import save_json_report

            return save_json_report(self, destination)
        if selected == "html":
            from ..reporting.html_report import save_html_report

            return save_html_report(self, destination)
        raise ValueError(
            f"unsupported report format {selected!r}; use 'json' or 'html' "
            "or give the path a .json/.html extension"
        )


def _format(value: float | None) -> str:
    if value is None:
        return "undefined"
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.8g}"


def make_metadata(
    *,
    backend: str,
    requested_metrics: tuple[str, ...],
    working_dtype: str,
    elapsed_seconds: float,
    normalization_seconds: float,
    native_seconds: float | None,
    parallel: bool,
    threads: int,
    chunks: int,
    chunk_size: int,
    release_gil: bool,
    created_at: str,
    python_version: str,
) -> ReportMetadata:
    """Internal factory keeping metadata construction consistent."""
    return ReportMetadata(
        schema_version=SCHEMA_VERSION,
        library_version=__version__,
        created_at=created_at,
        backend=backend,
        requested_metrics=requested_metrics,
        working_dtype=working_dtype,
        elapsed_seconds=elapsed_seconds,
        normalization_seconds=normalization_seconds,
        native_seconds=native_seconds,
        parallel=parallel,
        threads=threads,
        chunks=chunks,
        chunk_size=chunk_size,
        release_gil=release_gil,
        python_version=python_version,
    )
