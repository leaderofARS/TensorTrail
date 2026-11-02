"""End-to-end execution capture, persistence, validation, and reload."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import tensortrail as tt


def _capture_execution(
    *,
    model_version: str,
    scale: float,
    include_head: bool = True,
) -> tt.ExecutionSession:
    session = tt.ExecutionSession.create(
        model_id="classifier",
        model_version=model_version,
        framework="pytorch",
        framework_version="2.5",
        precision="float32",
        device_type="cpu",
        input_identifier="sample-42",
        execution_configuration={"batch_size": 1},
        custom_metadata={"owner": "numerics"},
    )
    session.add_checkpoint(
        name="encoder.output",
        tensor=(np.arange(8, dtype=np.float32).reshape(2, 4) * scale),
        stage_name="encoder",
        operation_type="matmul",
    )
    if include_head:
        session.add_checkpoint(
            name="head.output",
            tensor=np.array([0.25, 0.5, 0.75], dtype=np.float32) * scale,
            stage_name="head",
            operation_type="softmax",
        )
    session.finalize()
    return session


def test_persisted_execution_validation_workflow(tmp_path: Path) -> None:
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    reference = _capture_execution(model_version="1.0", scale=1.0)
    candidate = _capture_execution(model_version="1.1", scale=1.0001)
    storage.save_execution(reference)
    storage.save_execution(candidate)

    loaded_reference = storage.load_execution(reference.execution_id)
    loaded_candidate = storage.load_execution(candidate.execution_id)
    assert loaded_reference.checkpoint_count == 2
    assert loaded_candidate.checkpoint_count == 2

    result = tt.validate(
        loaded_reference,
        loaded_candidate,
        policy=tt.ValidationPolicy(
            backend="numpy",
            metrics=("mae", "rmse", "statistics"),
            atol=1e-3,
            rtol=1e-3,
        ),
    )
    assert result.status == "passed"
    assert result.alignment.matched_count == 2
    assert result.passed_checkpoint_count == 2
    assert "encoder.output" in result.summary

    storage.save_validation(result)
    reloaded = storage.load_validation(result.validation_id)
    assert reloaded.status == "passed"
    assert reloaded.summary == result.summary
    assert reloaded.alignment == result.alignment
    assert reloaded.checkpoint_results[0].report is not None
    assert reloaded.checkpoint_results[0].report.statistics is not None


def test_missing_checkpoint_is_not_reported_as_a_pass(tmp_path: Path) -> None:
    storage = tt.FilesystemStorage(tmp_path / "artifacts")
    reference = _capture_execution(model_version="1.0", scale=1.0)
    candidate = _capture_execution(model_version="1.1", scale=1.0, include_head=False)
    storage.save_execution(reference)
    storage.save_execution(candidate)
    result = tt.validate(
        storage.load_execution(reference.execution_id),
        storage.load_execution(candidate.execution_id),
        policy=tt.ValidationPolicy(backend="numpy", atol=0.0, rtol=0.0),
    )
    assert result.status == "incomplete"
    assert result.missing_checkpoint_count == 1
    assert result.passed_checkpoint_count == 1
    outcomes = {item.checkpoint_name: item.outcome for item in result.checkpoint_results}
    assert outcomes == {"encoder.output": "passed", "head.output": "missing_candidate"}


def test_public_namespace_preserves_the_010_api() -> None:
    for name in (
        "compare",
        "mae",
        "mse",
        "rmse",
        "relative_error",
        "cosine_similarity",
        "statistics",
        "available_backends",
        "is_native_available",
        "native_info",
        "ComparisonConfig",
        "ComparisonReport",
        "TensorValidationError",
        "SCHEMA_VERSION",
        "__version__",
    ):
        assert name in tt.__all__
        assert hasattr(tt, name)
    for name in (
        "ExecutionSession",
        "ExecutionMetadata",
        "TensorCheckpoint",
        "ValidationPolicy",
        "ValidationResult",
        "align_checkpoints",
        "validate",
        "FilesystemStorage",
    ):
        assert name in tt.__all__
        assert hasattr(tt, name)
    assert len(tt.__all__) == len(set(tt.__all__))


def test_compare_still_works_after_execution_imports() -> None:
    report = tt.compare(
        np.array([1.0, 2.0]),
        np.array([1.0, 3.0]),
        config=tt.ComparisonConfig(backend="numpy"),
    )
    assert report.status == "fail"
    assert report.metrics.mae == 0.5
