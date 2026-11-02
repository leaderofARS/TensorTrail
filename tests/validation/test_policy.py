"""Validation policy defaults, validation, and comparison-config reuse."""

from __future__ import annotations

import math

import pytest

import tensortrail as tt


def _policy(**overrides: object) -> tt.ValidationPolicy:
    values: dict[str, object] = {"backend": "numpy"}
    values.update(overrides)
    return tt.ValidationPolicy(**values)  # type: ignore[arg-type]


def test_policy_defaults() -> None:
    policy = _policy()
    assert policy.metrics == tt.DEFAULT_METRICS
    assert policy.atol == 1e-6
    assert policy.rtol == 1e-5
    assert policy.relative_epsilon == 1e-12
    assert policy.epsilon == 1e-12
    assert policy.equal_nan is True
    assert policy.backend == "numpy"
    assert policy.require_all_checkpoints is True
    assert policy.fail_on_missing_checkpoints is False
    assert policy.fail_on_shape_mismatch is False
    assert policy.fail_on_dtype_change is False
    assert policy.checkpoint_selection is None
    assert policy.aggregation_strategy == "all"


def test_policy_builds_the_existing_comparison_config() -> None:
    policy = _policy(
        metrics=("mae", "rmse"),
        atol=0.25,
        rtol=0.5,
        relative_epsilon=1e-9,
        equal_nan=False,
    )
    config = policy.to_comparison_config()
    assert isinstance(config, tt.ComparisonConfig)
    assert config.metrics == ("mae", "rmse")
    assert config.atol == 0.25
    assert config.rtol == 0.5
    assert config.relative_epsilon == 1e-9
    assert config.equal_nan is False
    assert config.backend == "numpy"


@pytest.mark.parametrize(
    "overrides",
    [
        {"metrics": ("unknown",)},
        {"atol": -1.0},
        {"rtol": -1.0},
        {"atol": math.nan},
        {"rtol": math.inf},
        {"atol": True},
        {"relative_epsilon": -1e-12},
        {"backend": "cuda"},
        {"backend": None},
    ],
)
def test_policy_reuses_comparison_validation(overrides: dict[str, object]) -> None:
    with pytest.raises(tt.TensorValidationError):
        _policy(**overrides)


@pytest.mark.parametrize(
    "field_name",
    [
        "require_all_checkpoints",
        "fail_on_missing_checkpoints",
        "fail_on_shape_mismatch",
        "fail_on_dtype_change",
    ],
)
def test_policy_requires_strict_booleans(field_name: str) -> None:
    with pytest.raises(tt.TensorValidationError, match=field_name):
        _policy(**{field_name: 1})


def test_fail_on_missing_requires_require_all() -> None:
    with pytest.raises(tt.TensorValidationError, match="require_all_checkpoints"):
        _policy(
            require_all_checkpoints=False,
            fail_on_missing_checkpoints=True,
        )


def test_checkpoint_selection_is_normalized() -> None:
    assert _policy(checkpoint_selection="a").checkpoint_selection == ("a",)
    assert _policy(checkpoint_selection=["a", "b"]).checkpoint_selection == (
        "a",
        "b",
    )
    assert _policy(checkpoint_selection=("a",)).checkpoint_selection == ("a",)
    assert _policy(checkpoint_selection=[]).checkpoint_selection == ()


@pytest.mark.parametrize(
    "selection",
    [
        ["a", "a"],
        ["a", "  "],
        [1],
        5,
        object(),
    ],
)
def test_invalid_checkpoint_selection_is_rejected(selection: object) -> None:
    with pytest.raises(tt.TensorValidationError):
        _policy(checkpoint_selection=selection)


def test_aggregation_strategy_is_case_insensitive() -> None:
    assert _policy(aggregation_strategy=" ALL ").aggregation_strategy == "all"
    assert _policy(aggregation_strategy="Any").aggregation_strategy == "any"
    with pytest.raises(tt.TensorValidationError, match="aggregation_strategy"):
        _policy(aggregation_strategy="most")
    with pytest.raises(tt.TensorValidationError, match="aggregation_strategy"):
        _policy(aggregation_strategy=1)


def test_policy_round_trip() -> None:
    policy = _policy(
        metrics=("mae", "statistics"),
        atol=1e-3,
        rtol=1e-2,
        equal_nan=False,
        require_all_checkpoints=False,
        fail_on_shape_mismatch=True,
        fail_on_dtype_change=True,
        checkpoint_selection=("a", "b"),
        aggregation_strategy="any",
    )
    document = policy.as_dict()
    restored = tt.ValidationPolicy.from_dict(document)
    assert restored == policy
    assert restored.as_dict() == document


def test_policy_from_dict_rejects_unknown_and_missing_fields() -> None:
    document = _policy().as_dict()
    document["unknown"] = 1
    with pytest.raises(tt.TensorValidationError, match="unknown validation policy"):
        tt.ValidationPolicy.from_dict(document)

    incomplete = _policy().as_dict()
    del incomplete["atol"]
    with pytest.raises(tt.TensorValidationError, match="missing validation policy"):
        tt.ValidationPolicy.from_dict(incomplete)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("metrics", "mae"),
        ("checkpoint_selection", "a"),
        ("atol", "loose"),
        ("aggregation_strategy", "most"),
    ],
)
def test_policy_from_dict_rejects_malformed_values(field_name: str, value: object) -> None:
    document = _policy().as_dict()
    document[field_name] = value
    with pytest.raises(tt.TensorValidationError):
        tt.ValidationPolicy.from_dict(document)
