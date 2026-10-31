"""JSON/HTML report contracts, escaping, and atomic file writes."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

import tensortrail as tt


def _report(**overrides: object) -> tt.ComparisonReport:
    config = tt.ComparisonConfig(
        backend="numpy",
        metrics=overrides.pop("metrics", ("mae", "mse", "rmse")),  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )
    return tt.compare(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.5, 2.0, 2.0]),
        config=config,
        reference_id="ref",
        candidate_id="cand",
    )


def test_json_document_has_stable_top_level_schema() -> None:
    report = _report(metrics=("mae", "statistics"))
    document = json.loads(report.to_json())

    assert set(document) == {
        "schema_version",
        "library",
        "created_at",
        "status",
        "reference",
        "candidate",
        "requested_metrics",
        "tolerances",
        "metrics",
        "statistics",
        "comparison",
        "execution",
    }
    assert document["schema_version"] == tt.SCHEMA_VERSION == "1.0"
    assert document["library"] == {"name": "tensortrail", "version": tt.__version__}
    assert document["status"] == "fail"
    assert document["requested_metrics"] == ["mae", "statistics"]
    assert list(document["metrics"]) == ["mae"]
    assert document["reference"]["shape"] == [3]
    assert document["reference"]["dtype"] == "float64"
    assert document["statistics"]["reference"]["mean"] == pytest.approx(2.0)
    assert document["comparison"]["elements"] == 3
    assert document["comparison"]["mismatch_count"] == 2
    assert document["execution"]["backend"] == "numpy"
    assert document["tolerances"]["rule"].startswith("abs(reference - candidate)")


def test_to_dict_matches_json_round_trip() -> None:
    report = _report()
    assert json.loads(report.to_json()) == report.to_dict()


def test_statistics_absent_when_not_requested() -> None:
    document = _report(metrics=("mae",)).to_dict()
    assert document["statistics"] is None


def test_nonfinite_metrics_use_json_sentinels() -> None:
    report = tt.compare(
        np.array([1e308]),
        np.array([-1e308]),
        config=tt.ComparisonConfig(
            backend="numpy",
            metrics=("mae", "mse", "rmse", "relative_error"),
            atol=0.0,
            rtol=0.0,
        ),
    )
    assert math.isinf(report.metrics.mae or 0.0)
    raw = report.to_json()
    assert '"Infinity"' in raw
    document = json.loads(raw)
    assert document["metrics"]["mae"] == "Infinity"
    assert document["metrics"]["relative_error"]["mean"] == "Infinity"
    assert document["comparison"]["max_abs_error"] == "Infinity"


def test_nan_metric_is_serialized_as_a_string() -> None:
    report = _report(metrics=("mae",))
    metrics = dataclasses.replace(report.metrics, mae=math.nan)
    modified = dataclasses.replace(report, metrics=metrics)
    document = json.loads(modified.to_json())
    assert document["metrics"]["mae"] == "NaN"


def test_json_serialization_is_strict_and_indented() -> None:
    raw = _report().to_json(indent=4)
    assert "\n    " in raw
    json.loads(raw)  # would raise if NaN/Infinity were emitted unquoted
    compact = _report().to_json(indent=None)  # type: ignore[arg-type]
    assert "\n" not in compact


def test_save_json_returns_path_and_writes_valid_document(tmp_path: Path) -> None:
    report = _report()
    destination = tmp_path / "report.json"
    returned = report.save(destination)
    assert returned == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["status"] == "fail"


def test_save_uses_extension_and_supports_overrides(tmp_path: Path) -> None:
    report = _report()
    html_path = tmp_path / "report.html"
    assert report.save(html_path) == html_path
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")

    uppercase = tmp_path / "REPORT.JSON"
    report.save(uppercase)
    assert json.loads(uppercase.read_text(encoding="utf-8"))["schema_version"] == "1.0"

    no_suffix = tmp_path / "report"
    report.save(no_suffix)
    assert json.loads(no_suffix.read_text(encoding="utf-8"))["schema_version"] == "1.0"

    forced = tmp_path / "report.data"
    report.save(forced, format=".html")
    assert forced.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported report format"):
        _report().save(tmp_path / "report.yaml")


def test_missing_parent_directory_is_not_created(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _report().save(tmp_path / "missing" / "report.json")


def test_html_report_is_self_contained_and_escaped() -> None:
    report = tt.compare(
        np.array([1.0, 2.0]),
        np.array([1.0, 3.0]),
        config=tt.ComparisonConfig(backend="numpy", metrics=("mae", "statistics")),
        reference_id="<script>alert(1)</script>",
        candidate_id="candidate & <tag>",
    )
    from tensortrail.reporting.html_report import report_to_html

    markup = report_to_html(report)
    assert markup.startswith("<!doctype html>")
    assert "TensorTrail numerical comparison" in markup
    assert "FAIL" in markup
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in markup
    assert "<script>alert(1)</script>" not in markup
    assert "candidate &amp; &lt;tag&gt;" in markup
    assert "abs(ref - cand)" in markup
    assert "http://" not in markup
    assert "https://" not in markup
    assert "<script src=" not in markup


def test_html_includes_conversion_notes_and_statistics() -> None:
    report = tt.compare(
        np.array([1.0, 2.0, 3.0], dtype=np.float16),
        np.array([1.0, 2.0, 4.0], dtype=np.float16),
        config=tt.ComparisonConfig(backend="numpy", metrics=("mae", "statistics")),
        reference_id="half",
        candidate_id="half_candidate",
    )
    from tensortrail.reporting.html_report import report_to_html

    markup = report_to_html(report)
    assert "float16_to_float32" in markup
    assert "Tensor statistics" in markup
    assert "std dev" in markup
    assert "half_candidate" in markup


def test_html_contains_embedded_json_that_round_trips() -> None:
    report = _report()
    from tensortrail.reporting.html_report import report_to_html

    markup = report_to_html(report)
    assert "JSON report" in markup
    assert "&quot;schema_version&quot;: &quot;1.0&quot;" in markup
    assert report.to_json() not in markup  # JSON must be HTML-escaped


def test_atomic_write_replaces_existing_content(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_text("stale", encoding="utf-8")
    _report().save(destination)
    content = destination.read_text(encoding="utf-8")
    assert "stale" not in content
    assert content.endswith("\n")
    assert not list(tmp_path.glob("*.tmp"))


def test_report_with_extreme_statistics_serializes() -> None:
    report = tt.compare(
        np.array([1e308, -1e308]),
        np.array([1e308, -1e308]),
        config=tt.ComparisonConfig(backend="numpy", metrics=("statistics",)),
    )
    raw = report.to_json()
    json.loads(raw)
    document = report.to_dict()
    assert document["statistics"] is not None
