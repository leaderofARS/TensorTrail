"""End-to-end workflow: compare, inspect, persist, and share a report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tensortrail as tt
from tests.conftest import requires_native


def _simulate_layer_outputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    activation = rng.normal(size=(32, 768)).astype(np.float32)
    # Simulate a mixed-precision inference path: small drift, a few saturated
    # values, and one NaN that the report must count separately.
    candidate = activation.astype(np.float64) + rng.normal(scale=1e-4, size=activation.shape)
    candidate[:, :4] = activation[:, :4] * 1.02
    candidate[0, 0] = np.nan
    return activation, candidate.astype(np.float32)


def test_compare_inspect_and_persist_report(tmp_path: Path) -> None:
    reference, candidate = _simulate_layer_outputs()
    report = tt.compare(
        reference,
        candidate,
        metrics=tt.SUPPORTED_METRICS,
        config=tt.ComparisonConfig(atol=1e-3, rtol=1e-3),
        reference_id="transformer.layer_12.output",
        candidate_id="transformer.layer_12.output.optimized",
    )

    assert report.status == "fail"
    assert report.metadata.schema_version == tt.SCHEMA_VERSION
    assert report.reference.shape == (32, 768)
    assert report.reference.elements == 32 * 768
    assert report.counts.reference_nonfinite.nan == 0
    assert report.counts.candidate_nonfinite.nan == 1
    assert report.counts.nonfinite_mismatches == 1
    assert report.statistics is not None
    assert report.statistics.reference.mean is not None

    summary = report.summary()
    assert "TensorTrail comparison: FAIL" in summary
    assert "transformer.layer_12.output" in summary
    assert "non-finite candidate: nan=1" in summary

    json_path = report.save(tmp_path / "layer12.json")
    html_path = report.save(tmp_path / "layer12.html")
    assert json_path.exists()
    assert html_path.exists()

    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["status"] == "fail"
    assert document["reference"]["id"] == "transformer.layer_12.output"
    assert document["comparison"]["candidate_nonfinite"]["nan"] == 1
    assert document["statistics"]["candidate"]["nonfinite"]["nan"] == 1
    assert document["metrics"]["mae"] > 0
    assert "transformer.layer_12.output.optimized" in html_path.read_text(encoding="utf-8")


@requires_native
def test_native_and_reference_backends_reach_the_same_verdict() -> None:
    reference, candidate = _simulate_layer_outputs()
    keyword = {"atol": 1e-3, "rtol": 1e-3, "metrics": ("mae", "rmse", "cosine_similarity")}
    native = tt.compare(
        reference,
        candidate,
        config=tt.ComparisonConfig(backend="rust", **keyword),
    )
    numpy = tt.compare(
        reference,
        candidate,
        config=tt.ComparisonConfig(backend="numpy", **keyword),
    )
    assert native.status == numpy.status
    assert native.counts == numpy.counts
    assert native.metrics.mae == pytest.approx(numpy.metrics.mae, rel=1e-6, abs=1e-9)
    assert native.metrics.rmse == pytest.approx(numpy.metrics.rmse, rel=1e-6, abs=1e-9)
    assert native.metadata.backend == "rust"
    assert numpy.metadata.backend == "numpy"


@requires_native
def test_default_backend_is_native() -> None:
    report = tt.compare(np.ones(8), np.ones(8))
    assert report.metadata.backend == "rust"
    assert report.metadata.native_seconds is not None
    assert report.metadata.parallel is False
