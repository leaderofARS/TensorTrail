"""Deterministic checkpoint registry behavior."""

from __future__ import annotations

import pytest

import tensortrail as tt
from tensortrail.execution.checkpoint import TensorCheckpoint


def _checkpoint(
    name: str,
    index: int,
    *,
    execution_id: str = "exec_1",
    checkpoint_id: str | None = None,
    stage_name: str | None = None,
) -> TensorCheckpoint:
    values: dict[str, object] = {
        "execution_id": execution_id,
        "checkpoint_name": name,
        "checkpoint_index": index,
        "tensor_shape": (2,),
        "tensor_dtype": "float32",
        "tensor_nbytes": 8,
        "stage_name": stage_name,
    }
    if checkpoint_id is not None:
        values["checkpoint_id"] = checkpoint_id
    return TensorCheckpoint(**values)  # type: ignore[arg-type]


def test_registry_orders_by_checkpoint_index_not_insertion() -> None:
    registry = tt.CheckpointRegistry("exec_1")
    registry.register(_checkpoint("third", 2))
    registry.register(_checkpoint("first", 0))
    registry.register(_checkpoint("second", 1))
    assert registry.names() == ("first", "second", "third")
    assert registry.ids() == tuple(checkpoint.checkpoint_id for checkpoint in registry.all())
    assert [checkpoint.checkpoint_name for checkpoint in registry] == [
        "first",
        "second",
        "third",
    ]


def test_lookup_by_id_and_name() -> None:
    registry = tt.CheckpointRegistry("exec_1")
    checkpoint = registry.register(_checkpoint("encoder", 0))
    assert registry.get(checkpoint.checkpoint_id) is checkpoint
    assert registry.by_id(checkpoint.checkpoint_id) is checkpoint
    assert registry.get_by_name("encoder") is checkpoint
    assert registry.by_name("encoder") is checkpoint
    assert checkpoint.checkpoint_id in registry
    assert "missing" not in registry
    assert 5 not in registry
    assert len(registry) == 1


def test_duplicate_id_name_and_index_are_rejected() -> None:
    registry = tt.CheckpointRegistry("exec_1")
    first = registry.register(_checkpoint("first", 0, checkpoint_id="cp_fixed"))
    with pytest.raises(tt.CheckpointError, match="checkpoint_id"):
        registry.register(_checkpoint("other", 1, checkpoint_id=first.checkpoint_id))
    with pytest.raises(tt.CheckpointError, match="checkpoint_name"):
        registry.register(_checkpoint("first", 1))
    with pytest.raises(tt.CheckpointError, match="checkpoint_index"):
        registry.register(_checkpoint("other", 0))


def test_checkpoint_from_another_execution_is_rejected() -> None:
    registry = tt.CheckpointRegistry("exec_1")
    with pytest.raises(tt.CheckpointError, match="belongs to execution"):
        registry.register(_checkpoint("other", 0, execution_id="exec_2"))


def test_non_checkpoint_registration_is_rejected() -> None:
    registry = tt.CheckpointRegistry("exec_1")
    with pytest.raises(tt.CheckpointError, match="TensorCheckpoint"):
        registry.register(object())  # type: ignore[arg-type]


def test_missing_lookups_raise_structured_errors() -> None:
    registry = tt.CheckpointRegistry("exec_1")
    with pytest.raises(tt.CheckpointNotFoundError, match="checkpoint_id"):
        registry.get("cp_missing")
    with pytest.raises(tt.CheckpointNotFoundError, match="checkpoint_name"):
        registry.get_by_name("missing")
    assert issubclass(tt.CheckpointNotFoundError, KeyError)


def test_for_execution_filters_by_identifier() -> None:
    registry = tt.CheckpointRegistry("exec_1")
    checkpoint = registry.register(_checkpoint("first", 0))
    assert registry.for_execution("exec_1") == (checkpoint,)
    assert registry.for_execution("exec_2") == ()


def test_registry_repr_contains_identity_and_count() -> None:
    registry = tt.CheckpointRegistry("exec_1")
    registry.register(_checkpoint("first", 0))
    representation = repr(registry)
    assert "exec_1" in representation
    assert "checkpoints=1" in representation
