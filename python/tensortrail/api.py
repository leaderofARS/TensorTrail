"""Public TensorTrail API."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from .backends import rust_backend
from .comparison import engine
from .config import BACKENDS, ComparisonConfig
from .core.result import (
    ComparisonReport,
    RelativeErrorValue,
    TensorStatistics,
)

if TYPE_CHECKING:
    from .execution.session import ExecutionSession
    from .validation.alignment import CheckpointAlignment
    from .validation.policy import ValidationPolicy
    from .validation.results import ValidationResult

__all__ = [
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
    "validate",
]


def _resolve_config(
    config: ComparisonConfig | None,
    metrics: Iterable[str] | None = None,
) -> ComparisonConfig:
    resolved = ComparisonConfig() if config is None else config
    if not isinstance(resolved, ComparisonConfig):
        raise TypeError(f"config must be ComparisonConfig, received {type(resolved).__name__}")
    if metrics is not None:
        names = (metrics,) if isinstance(metrics, str) else tuple(metrics)
        resolved = replace(resolved, metrics=names)
    return resolved


def compare(
    reference: object,
    candidate: object,
    *,
    metrics: Iterable[str] | None = None,
    config: ComparisonConfig | None = None,
    reference_id: str = "reference",
    candidate_id: str = "candidate",
) -> ComparisonReport:
    """Compare two tensors and return a structured diagnostic report.

    Parameters
    ----------
    reference:
        Reference execution tensor. Accepts NumPy arrays, numeric array-like
        objects, and CPU PyTorch tensors.
    candidate:
        Candidate execution tensor with exactly the same shape.
    metrics:
        Subset of ``{"mae", "mse", "rmse", "relative_error",
        "cosine_similarity", "statistics"}``. Overrides ``config.metrics``.
    config:
        Numerical policy and native execution controls. Defaults to all binary
        metrics, ``atol=1e-6``, ``rtol=1e-5``, and the native Rust backend.
    reference_id, candidate_id:
        Identifiers stored in the report.

    Returns
    -------
    ComparisonReport
        Metrics, counts, tolerances, provenance, and execution metadata.

    Notes
    -----
    Non-finite elements are excluded from metric sums and counted separately.
    Their pattern differences still affect ``report.status``. Metrics over empty
    tensors are ``None`` rather than fabricated zeros.
    """
    resolved = _resolve_config(config, metrics)
    return engine.compare(
        reference,
        candidate,
        resolved,
        reference_id=reference_id,
        candidate_id=candidate_id,
    )


def _single_metric(
    name: str,
    reference: object,
    candidate: object,
    config: ComparisonConfig | None,
) -> ComparisonReport:
    return compare(
        reference,
        candidate,
        metrics=(name,),
        config=config,
    )


def mae(
    reference: object,
    candidate: object,
    *,
    config: ComparisonConfig | None = None,
) -> float | None:
    """Mean absolute error; ``None`` when no finite pairs exist."""
    return _single_metric("mae", reference, candidate, config).metrics.mae


def mse(
    reference: object,
    candidate: object,
    *,
    config: ComparisonConfig | None = None,
) -> float | None:
    """Mean squared error; ``None`` when no finite pairs exist."""
    return _single_metric("mse", reference, candidate, config).metrics.mse


def rmse(
    reference: object,
    candidate: object,
    *,
    config: ComparisonConfig | None = None,
) -> float | None:
    """Root mean squared error; ``None`` when no finite pairs exist."""
    return _single_metric("rmse", reference, candidate, config).metrics.rmse


def relative_error(
    reference: object,
    candidate: object,
    *,
    config: ComparisonConfig | None = None,
) -> RelativeErrorValue | None:
    """Mean and maximum relative error with an epsilon denominator floor."""
    return _single_metric("relative_error", reference, candidate, config).metrics.relative_error


def cosine_similarity(
    reference: object,
    candidate: object,
    *,
    config: ComparisonConfig | None = None,
) -> float | None:
    """Cosine similarity with documented zero-vector behavior."""
    return _single_metric(
        "cosine_similarity", reference, candidate, config
    ).metrics.cosine_similarity


def statistics(
    values: object,
    *,
    config: ComparisonConfig | None = None,
) -> TensorStatistics:
    """Compute min, max, mean, population variance, and standard deviation.

    Non-finite values are counted separately and excluded from the numeric
    statistics.
    """
    resolved = _resolve_config(config)
    return engine.statistics(values, resolved)


def validate(
    reference: ExecutionSession,
    candidate: ExecutionSession,
    *,
    policy: ValidationPolicy | None = None,
    alignment: CheckpointAlignment | None = None,
    validation_id: str | None = None,
) -> ValidationResult:
    """Validate a candidate execution against a reference execution.

    Every matched checkpoint is compared through :func:`compare`. Missing,
    shape-incompatible, dtype-rejected, and backend-failed checkpoints are
    reported explicitly; they are never treated as passes. Both sessions must
    be completed before validation.

    Parameters
    ----------
    reference:
        Completed reference execution.
    candidate:
        Completed candidate execution.
    policy:
        Execution-level and numerical policy. Defaults to all checkpoints, the
        standard comparison tolerances, and the native Rust backend.
    alignment:
        Explicit alignment produced by :func:`align_checkpoints`. When omitted,
        alignment is computed with unique checkpoint names.
    validation_id:
        Stable validation identifier. Generated when omitted.
    """
    from .validation.engine import validate as _validate

    return _validate(
        reference,
        candidate,
        policy=policy,
        alignment=alignment,
        validation_id=validation_id,
    )


def is_native_available() -> bool:
    """Whether the compiled native extension is importable."""
    return rust_backend.is_available()


def available_backends() -> tuple[str, ...]:
    """Backends usable in this installation.

    The NumPy backend is always available. The Rust backend is listed only when
    the native extension loads successfully with a matching version.
    """
    if is_native_available():
        return BACKENDS
    return ("numpy",)


def native_info() -> dict[str, object]:
    """Native build and runtime information.

    Raises
    ------
    BackendError
        If the native extension is unavailable.
    """
    return rust_backend.native_info()
