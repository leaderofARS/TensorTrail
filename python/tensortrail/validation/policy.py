"""Validation policy built on the existing comparison configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..config import DEFAULT_METRICS, ComparisonConfig
from ..exceptions import TensorValidationError
from ..execution.metadata import validate_identifier

__all__ = ["ValidationPolicy", "normalize_checkpoint_selection"]

_AGGREGATION_STRATEGIES = ("all", "any")


def _strict_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TensorValidationError(f"{field_name} must be a bool")
    return value


def normalize_checkpoint_selection(
    value: object,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_values: tuple[object, ...] = (value,)
    else:
        if not isinstance(value, Iterable):
            raise TensorValidationError(
                "checkpoint_selection must be None, a string, or an iterable of names"
            )
        try:
            raw_values = tuple(value)
        except TypeError as error:  # pragma: no cover - defensive
            raise TensorValidationError(
                "checkpoint_selection must be None, a string, or an iterable of names"
            ) from error

    selected: list[str] = []
    for item in raw_values:
        name = validate_identifier(item, "checkpoint_selection entry")
        if name in selected:
            raise TensorValidationError(f"checkpoint_selection contains duplicate name {name!r}")
        selected.append(name)
    return tuple(selected)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationPolicy:
    """Execution-level policy.

    Numerical fields are validated by constructing a :class:`ComparisonConfig`;
    the comparison engine receives that configuration for every checkpoint.
    """

    metrics: tuple[str, ...] = field(default_factory=lambda: DEFAULT_METRICS)
    atol: float = 1e-6
    rtol: float = 1e-5
    relative_epsilon: float = 1e-12
    equal_nan: bool = True
    backend: str = "rust"
    require_all_checkpoints: bool = True
    fail_on_missing_checkpoints: bool = False
    fail_on_shape_mismatch: bool = False
    fail_on_dtype_change: bool = False
    checkpoint_selection: tuple[str, ...] | None = None
    aggregation_strategy: str = "all"

    def __post_init__(self) -> None:
        config = ComparisonConfig(
            metrics=self.metrics,
            atol=self.atol,
            rtol=self.rtol,
            equal_nan=self.equal_nan,
            relative_epsilon=self.relative_epsilon,
            backend=self.backend,
        )
        object.__setattr__(self, "metrics", config.metrics)
        object.__setattr__(self, "atol", config.atol)
        object.__setattr__(self, "rtol", config.rtol)
        object.__setattr__(self, "relative_epsilon", config.relative_epsilon)
        object.__setattr__(self, "equal_nan", config.equal_nan)
        object.__setattr__(self, "backend", config.backend)
        object.__setattr__(
            self,
            "require_all_checkpoints",
            _strict_bool(self.require_all_checkpoints, "require_all_checkpoints"),
        )
        object.__setattr__(
            self,
            "fail_on_missing_checkpoints",
            _strict_bool(self.fail_on_missing_checkpoints, "fail_on_missing_checkpoints"),
        )
        object.__setattr__(
            self,
            "fail_on_shape_mismatch",
            _strict_bool(self.fail_on_shape_mismatch, "fail_on_shape_mismatch"),
        )
        object.__setattr__(
            self,
            "fail_on_dtype_change",
            _strict_bool(self.fail_on_dtype_change, "fail_on_dtype_change"),
        )
        if self.fail_on_missing_checkpoints and not self.require_all_checkpoints:
            raise TensorValidationError(
                "fail_on_missing_checkpoints requires require_all_checkpoints=True; "
                "optional missing checkpoints cannot fail validation"
            )
        object.__setattr__(
            self,
            "checkpoint_selection",
            normalize_checkpoint_selection(self.checkpoint_selection),
        )
        if not isinstance(self.aggregation_strategy, str):
            raise TensorValidationError("aggregation_strategy must be a string")
        strategy = self.aggregation_strategy.strip().lower()
        if strategy not in _AGGREGATION_STRATEGIES:
            raise TensorValidationError(
                "aggregation_strategy must be 'all' or 'any'; "
                f"received {self.aggregation_strategy!r}"
            )
        object.__setattr__(self, "aggregation_strategy", strategy)

    @property
    def epsilon(self) -> float:
        """Alias for ``relative_epsilon`` matching the existing numerical policy."""
        return self.relative_epsilon

    def to_comparison_config(self) -> ComparisonConfig:
        """Build the numerical configuration used for each checkpoint."""
        return ComparisonConfig(
            metrics=self.metrics,
            atol=self.atol,
            rtol=self.rtol,
            equal_nan=self.equal_nan,
            relative_epsilon=self.relative_epsilon,
            backend=self.backend,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy document."""
        return {
            "metrics": list(self.metrics),
            "atol": self.atol,
            "rtol": self.rtol,
            "relative_epsilon": self.relative_epsilon,
            "equal_nan": self.equal_nan,
            "backend": self.backend,
            "require_all_checkpoints": self.require_all_checkpoints,
            "fail_on_missing_checkpoints": self.fail_on_missing_checkpoints,
            "fail_on_shape_mismatch": self.fail_on_shape_mismatch,
            "fail_on_dtype_change": self.fail_on_dtype_change,
            "checkpoint_selection": (
                list(self.checkpoint_selection) if self.checkpoint_selection is not None else None
            ),
            "aggregation_strategy": self.aggregation_strategy,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ValidationPolicy:
        """Parse a policy document emitted by :meth:`as_dict`."""
        required = {
            "metrics",
            "atol",
            "rtol",
            "relative_epsilon",
            "equal_nan",
            "backend",
            "require_all_checkpoints",
            "fail_on_missing_checkpoints",
            "fail_on_shape_mismatch",
            "fail_on_dtype_change",
            "checkpoint_selection",
            "aggregation_strategy",
        }
        unknown = set(document) - required
        if unknown:
            raise TensorValidationError(
                "unknown validation policy field(s): " + ", ".join(sorted(unknown))
            )
        missing = required - set(document)
        if missing:
            raise TensorValidationError(
                "missing validation policy field(s): " + ", ".join(sorted(missing))
            )

        metrics = document["metrics"]
        selection = document["checkpoint_selection"]
        if not isinstance(metrics, (list, tuple)):
            raise TensorValidationError("policy metrics must be a list")
        if selection is not None and not isinstance(selection, (list, tuple)):
            raise TensorValidationError("checkpoint_selection must be a list or null")

        return cls(
            metrics=tuple(metrics),
            atol=document["atol"],
            rtol=document["rtol"],
            relative_epsilon=document["relative_epsilon"],
            equal_nan=document["equal_nan"],
            backend=document["backend"],
            require_all_checkpoints=document["require_all_checkpoints"],
            fail_on_missing_checkpoints=document["fail_on_missing_checkpoints"],
            fail_on_shape_mismatch=document["fail_on_shape_mismatch"],
            fail_on_dtype_change=document["fail_on_dtype_change"],
            checkpoint_selection=normalize_checkpoint_selection(selection),
            aggregation_strategy=document["aggregation_strategy"],
        )
