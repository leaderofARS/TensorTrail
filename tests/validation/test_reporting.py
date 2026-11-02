"""Validation report serialization, strictness, and comparison reload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tensortrail as tt
from tensortrail.reporting.json_report import report_from_dict, report_to_dict


def _result(**policy_overrides: object) -> tt.ValidationResult:
    reference = tt.ExecutionSession.create(
        model_id="classifier",
        model_version="1.0",
        framework="numpy",
        precision="float64",
    )
    candidate = tt.ExecutionSession.create(
        model_id="classifier",
        model_version="2.0",
        framework="numpy",
        precision="float64",
    )
    reference.add_checkpoint(
        name="encoder.output",
        tensor=np.arange(6, dtype=np.float64).reshape(2, 3),
        stage_name="encoder",
    )
    candidate.add_checkpoint(
        name="encoder.output",
        tensor=np.arange(6, dtype=np.float64).reshape(2, 3) + 0.5,
        stage_name="encoder",
    )
    reference.finalize()
    candidate.finalize()
    policy_values: dict[str, object] = {
        "backend": "numpy",
        "atol": 0.0,
        "rtol": 0.0,
    }
    policy_values.update(policy_overrides)
    return tt.validate(
        reference,
        candidate,
        policy=tt.ValidationPolicy(**policy_values),  # type: ignore[arg-type]
    )


def test_validation_document_schema() -> None:
    result = _result()
    document = result.to_dict()
    assert set(document) == {
        "schema_version",
        "artifact_type",
        "library",
        "validation_id",
        "created_at",
        "status",
        "reference_execution",
        "candidate_execution",
        "policy",
        "alignment",
        "counts",
        "summary",
        "highest_observed_deviation",
        "checkpoint_results",
    }
    assert document["schema_version"] == tt.VALIDATION_SCHEMA_VERSION == "1.0"
    assert document["artifact_type"] == "validation"
    assert document["library"] == {
        "name": "tensortrail",
        "version": tt.__version__,
    }
    assert document["status"] == "failed"
    assert document["counts"]["total_checkpoints"] == 1
    assert document["counts"]["failed_checkpoints"] == 1
    assert document["highest_observed_deviation"]["checkpoint_name"] == ("encoder.output")
    assert json.loads(result.to_json()) == document


def test_validation_json_uses_nonfinite_sentinels() -> None:
    reference = tt.ExecutionSession.create(
        model_id="m",
        model_version="1",
        framework="numpy",
        precision="float64",
    )
    candidate = tt.ExecutionSession.create(
        model_id="m",
        model_version="2",
        framework="numpy",
        precision="float64",
    )
    reference.add_checkpoint(name="a", tensor=np.array([1e308]))
    candidate.add_checkpoint(name="a", tensor=np.array([-1e308]))
    reference.finalize()
    candidate.finalize()
    result = tt.validate(
        reference,
        candidate,
        policy=tt.ValidationPolicy(backend="numpy", atol=0.0, rtol=0.0),
    )
    raw = result.to_json()
    assert '"Infinity"' in raw
    document = json.loads(raw)
    checkpoint_report = document["checkpoint_results"][0]["report"]
    assert checkpoint_report["metrics"]["mae"] == "Infinity"
    assert document["highest_observed_deviation"]["max_abs_error"] == "Infinity"


def test_validation_report_round_trip(tmp_path: Path) -> None:
    result = _result()
    destination = tmp_path / "validation.json"
    assert result.save(destination) == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["status"] == "failed"
    loaded = tt.ValidationResult.load(destination)
    assert loaded.status == result.status
    assert loaded.validation_id == result.validation_id
    assert loaded.created_at == result.created_at
    assert loaded.summary == result.summary
    assert loaded.checkpoint_results == result.checkpoint_results
    assert loaded.alignment == result.alignment
    assert loaded.policy == result.policy


def test_validation_save_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported validation report extension"):
        _result().save(tmp_path / "validation.txt")


def test_validation_load_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(tt.TensorValidationError, match="not valid JSON"):
        tt.ValidationResult.load(path)


def test_validation_from_dict_rejects_unknown_and_missing_fields() -> None:
    document = _result().to_dict()
    document["unknown"] = 1
    with pytest.raises(tt.TensorValidationError, match="unknown validation report"):
        tt.ValidationResult.from_dict(document)

    incomplete = _result().to_dict()
    del incomplete["policy"]
    with pytest.raises(tt.TensorValidationError, match="missing validation report"):
        tt.ValidationResult.from_dict(incomplete)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("schema_version", "9.9"),
        ("artifact_type", "execution"),
        ("status", "maybe"),
    ],
)
def test_validation_from_dict_rejects_identity_mismatches(field_name: str, value: str) -> None:
    document = _result().to_dict()
    document[field_name] = value
    with pytest.raises(tt.TensorValidationError):
        tt.ValidationResult.from_dict(document)


def test_validation_from_dict_rejects_tampered_counts() -> None:
    document = _result().to_dict()
    document["counts"]["passed_checkpoints"] = 5
    with pytest.raises(tt.TensorValidationError, match="passed_checkpoints"):
        tt.ValidationResult.from_dict(document)


def test_validation_from_dict_rejects_tampered_status() -> None:
    document = _result().to_dict()
    document["status"] = "passed"
    with pytest.raises(tt.TensorValidationError, match="does not match aggregated"):
        tt.ValidationResult.from_dict(document)


def test_validation_from_dict_rejects_checkpoint_outcome_mismatch() -> None:
    document = _result().to_dict()
    document["checkpoint_results"][0]["outcome"] = "passed"
    with pytest.raises(tt.TensorValidationError, match="disagrees"):
        tt.ValidationResult.from_dict(document)


def test_validation_from_dict_rejects_alignment_selection_mismatch() -> None:
    document = _result().to_dict()
    document["policy"]["checkpoint_selection"] = ["encoder.output"]
    with pytest.raises(tt.TensorValidationError, match="selection"):
        tt.ValidationResult.from_dict(document)


def test_comparison_report_round_trip_with_all_metrics() -> None:
    reference = np.array([0.0, 1.0, np.nan, np.inf], dtype=np.float16)
    candidate = np.array([1.0, 1.0, np.nan, 2.0], dtype=np.float64)
    report = tt.compare(
        reference,
        candidate,
        metrics=tt.SUPPORTED_METRICS,
        config=tt.ComparisonConfig(backend="numpy"),
        reference_id="ref",
        candidate_id="cand",
    )
    document = report_to_dict(report)
    restored = report_from_dict(document)
    assert restored == report
    assert restored.to_dict() == document


def test_comparison_report_round_trip_with_nonfinite_metrics() -> None:
    report = tt.compare(
        np.array([1e308]),
        np.array([-1e308]),
        config=tt.ComparisonConfig(
            backend="numpy",
            metrics=("mae", "relative_error"),
            atol=0.0,
            rtol=0.0,
        ),
    )
    restored = report_from_dict(report_to_dict(report))
    assert restored == report


def test_comparison_report_from_dict_rejects_bad_schema_and_status() -> None:
    report = tt.compare(
        np.array([1.0]),
        np.array([2.0]),
        config=tt.ComparisonConfig(backend="numpy"),
    )
    document = report_to_dict(report)
    document["schema_version"] = "9.9"
    with pytest.raises(tt.TensorValidationError, match="schema version"):
        report_from_dict(document)

    document = report_to_dict(report)
    document["status"] = "pass"
    with pytest.raises(tt.TensorValidationError, match="status"):
        report_from_dict(document)


def test_comparison_report_from_dict_rejects_unknown_library() -> None:
    report = tt.compare(
        np.array([1.0]),
        np.array([1.0]),
        config=tt.ComparisonConfig(backend="numpy"),
    )
    document = report_to_dict(report)
    document["library"]["name"] = "other"
    with pytest.raises(tt.TensorValidationError, match="library name"):
        report_from_dict(document)
