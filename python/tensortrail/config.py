"""Comparison configuration and validation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from .exceptions import TensorValidationError

__all__ = [
    "BACKENDS",
    "DEFAULT_METRICS",
    "SUPPORTED_METRICS",
    "ComparisonConfig",
]

SUPPORTED_METRICS: tuple[str, ...] = (
    "mae",
    "mse",
    "rmse",
    "relative_error",
    "cosine_similarity",
    "statistics",
)
DEFAULT_METRICS: tuple[str, ...] = (
    "mae",
    "mse",
    "rmse",
    "relative_error",
    "cosine_similarity",
)
BACKENDS: tuple[str, ...] = ("rust", "numpy")


def _non_negative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        received = type(value).__name__
        raise TensorValidationError(f"{name} must be a real number, received {received}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise TensorValidationError(f"{name} must be finite and >= 0, received {number!r}")
    return number


def _positive_float(name: str, value: object) -> float:
    number = _non_negative_float(name, value)
    if number <= 0.0:
        raise TensorValidationError(f"{name} must be > 0, received {number!r}")
    return number


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TensorValidationError(f"{name} must be an integer, received {type(value).__name__}")
    if value < 1:
        raise TensorValidationError(f"{name} must be at least 1, received {value!r}")
    return value


def _normalize_metrics(metrics: Iterable[str]) -> tuple[str, ...]:
    if isinstance(metrics, str):
        metrics = (metrics,)
    try:
        raw_names = tuple(metrics)
    except TypeError as error:
        raise TensorValidationError("metrics must be an iterable of metric names") from error

    normalized: list[str] = []
    for name in raw_names:
        if not isinstance(name, str):
            received = type(name).__name__
            raise TensorValidationError(f"metric names must be strings, received {received}")
        key = name.strip().lower()
        if key not in SUPPORTED_METRICS:
            supported = ", ".join(SUPPORTED_METRICS)
            raise TensorValidationError(f"unknown metric {name!r}; supported metrics: {supported}")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Controls numerical policy and native execution.

    Parameters
    ----------
    metrics:
        Metrics to populate in the report. ``"statistics"`` additionally requests
        per-tensor summary statistics. The fused engine always computes the
        binary metrics in one pass, but only requested values are exposed.
    atol, rtol:
        Per-element tolerance:
        ``|reference - candidate| <= atol + rtol * |reference|``.
    equal_nan:
        Whether corresponding NaNs count as matching.
    relative_epsilon:
        Positive floor for the relative-error denominator.
    threads:
        Rayon worker count. ``None`` uses the global Rayon pool. ``1`` forces
        the sequential path.
    parallel_threshold:
        Minimum element count before parallel execution is considered.
    chunk_size:
        Fixed reduction chunk size. Chunk boundaries do not depend on the worker
        count, so results are reproducible across thread counts.
    backend:
        ``"rust"`` (default, native engine) or ``"numpy"`` (reference backend,
        intended for validation and benchmarks). TensorTrail never falls back
        from Rust to NumPy silently.
    release_gil:
        Release the Python GIL while the native engine runs. Inputs must not be
        mutated by another thread concurrently.
    """

    metrics: tuple[str, ...] = field(default_factory=lambda: DEFAULT_METRICS)
    atol: float = 1e-6
    rtol: float = 1e-5
    equal_nan: bool = True
    relative_epsilon: float = 1e-12
    threads: int | None = None
    parallel_threshold: int = 32_768
    chunk_size: int = 32_768
    backend: str = "rust"
    release_gil: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _normalize_metrics(self.metrics))
        object.__setattr__(self, "atol", _non_negative_float("atol", self.atol))
        object.__setattr__(self, "rtol", _non_negative_float("rtol", self.rtol))
        object.__setattr__(
            self,
            "relative_epsilon",
            _positive_float("relative_epsilon", self.relative_epsilon),
        )
        object.__setattr__(
            self,
            "parallel_threshold",
            _positive_int("parallel_threshold", self.parallel_threshold),
        )
        object.__setattr__(self, "chunk_size", _positive_int("chunk_size", self.chunk_size))
        if self.threads is not None:
            object.__setattr__(self, "threads", _positive_int("threads", self.threads))
        if not isinstance(self.equal_nan, bool):
            raise TensorValidationError("equal_nan must be a bool")
        if not isinstance(self.release_gil, bool):
            raise TensorValidationError("release_gil must be a bool")

        backend = self.backend.strip().lower() if isinstance(self.backend, str) else ""
        if backend not in BACKENDS:
            supported = ", ".join(BACKENDS)
            raise TensorValidationError(
                f"unknown backend {self.backend!r}; available backends: {supported}. "
                "TensorTrail does not silently fall back between backends."
            )
        object.__setattr__(self, "backend", backend)
