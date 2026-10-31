"""Self-contained HTML diagnostic reports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from ..core.result import ComparisonReport, RelativeErrorValue
from .json_report import _atomic_write, report_to_json

__all__ = ["report_to_html", "save_html_report"]


_CSS = """
:root { color-scheme: light; --ink:#101828; --muted:#667085; --line:#e4e7ec;
  --pass:#067647; --pass-bg:#ecfdf3; --fail:#b42318; --fail-bg:#fef3f2;
  --accent:#175cd3; --panel:#f9fafb; }
* { box-sizing: border-box; }
body { margin:0; background:#fff; color:var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height:1.5; }
main { max-width: 960px; margin: 0 auto; padding: 40px 24px 64px; }
header { display:flex; align-items:flex-start; justify-content:space-between;
  gap:24px; border-bottom:1px solid var(--line); padding-bottom:24px; }
h1 { font-size:26px; letter-spacing:-0.02em; margin:0 0 4px; }
h2 { font-size:16px; letter-spacing:0.04em; text-transform:uppercase;
  color:var(--muted); margin:32px 0 12px; }
p { margin:0; color:var(--muted); }
.badge { font-weight:700; font-size:13px; letter-spacing:0.08em;
  padding:8px 14px; border-radius:999px; white-space:nowrap; }
.badge.pass { color:var(--pass); background:var(--pass-bg); }
.badge.fail { color:var(--fail); background:var(--fail-bg); }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:16px; }
.card { border:1px solid var(--line); border-radius:12px; padding:16px 18px;
  background:var(--panel); }
.card h3 { margin:0 0 10px; font-size:13px; color:var(--muted);
  text-transform:uppercase; letter-spacing:0.05em; }
dl { display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; margin:0; }
dt { color:var(--muted); font-size:13px; } dd { margin:0; font-variant-numeric:tabular-nums; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0.05em; }
td.num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { overflow:auto; background:#0c111d; color:#d0d5dd; border-radius:10px;
  padding:16px; font-size:12px; }
footer { margin-top:48px; color:var(--muted); font-size:12px;
  border-top:1px solid var(--line); padding-top:16px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
"""


def _value(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.10g}"


def _metric_rows(report: ComparisonReport) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for name in report.metadata.requested_metrics:
        if name == "statistics":
            continue
        value: Any = getattr(report.metrics, name)
        if name == "relative_error":
            if isinstance(value, RelativeErrorValue):
                rows.append(("Relative error (mean)", _value(value.mean)))
                rows.append(("Relative error (max)", _value(value.maximum)))
            else:
                rows.append(("Relative error", "undefined"))
        else:
            rows.append((name.upper() if name in {"mae", "mse", "rmse"} else name, _value(value)))
    rows.append(("Maximum absolute error", _value(report.max_abs_error)))
    return rows


def _tensor_card(label: str, info: Any) -> str:
    notes = (
        "<br>".join(html.escape(note) for note in info.conversion_notes)
        if info.conversion_notes
        else "none"
    )
    return f"""
    <section class="card">
      <h3>{html.escape(label)} tensor</h3>
      <dl>
        <dt>id</dt><dd>{html.escape(info.id)}</dd>
        <dt>shape</dt><dd>{html.escape(str(info.shape))}</dd>
        <dt>dtype</dt><dd>{html.escape(info.dtype)}</dd>
        <dt>working dtype</dt><dd>{html.escape(info.working_dtype)}</dd>
        <dt>device</dt><dd>{html.escape(info.device)}</dd>
        <dt>elements</dt><dd>{info.elements:,}</dd>
        <dt>source</dt><dd>{html.escape(info.source)}</dd>
        <dt>conversions</dt><dd>{notes}</dd>
      </dl>
    </section>"""


def _statistics_table(report: ComparisonReport) -> str:
    if report.statistics is None:
        return "<p>Statistics were not requested.</p>"
    reference = report.statistics.reference
    candidate = report.statistics.candidate
    rows = [
        ("elements", reference.elements, candidate.elements),
        ("finite elements", reference.finite_elements, candidate.finite_elements),
        ("NaN", reference.nonfinite.nan, candidate.nonfinite.nan),
        ("+inf", reference.nonfinite.positive_infinity, candidate.nonfinite.positive_infinity),
        ("-inf", reference.nonfinite.negative_infinity, candidate.nonfinite.negative_infinity),
        ("min", _value(reference.min), _value(candidate.min)),
        ("max", _value(reference.max), _value(candidate.max)),
        ("mean", _value(reference.mean), _value(candidate.mean)),
        ("variance", _value(reference.variance), _value(candidate.variance)),
        ("std dev", _value(reference.std_dev), _value(candidate.std_dev)),
    ]
    body = "".join(
        f"<tr><th>{html.escape(str(name))}</th><td class='num'>{left}</td>"
        f"<td class='num'>{right}</td></tr>"
        for name, left, right in rows
    )
    return f"""
    <table>
      <thead><tr><th>Statistic</th><th>Reference</th><th>Candidate</th></tr></thead>
      <tbody>{body}</tbody>
    </table>"""


def report_to_html(report: ComparisonReport) -> str:
    """Render a standalone HTML document with no external assets."""
    status = report.status
    metric_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td class='num'>{html.escape(value)}</td></tr>"
        for name, value in _metric_rows(report)
    )
    counts = report.counts
    metadata = report.metadata
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorTrail report — {status.upper()}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>TensorTrail numerical comparison</h1>
      <p>Created {html.escape(metadata.created_at)} · schema {html.escape(metadata.schema_version)}
         · TensorTrail {html.escape(metadata.library_version)}</p>
    </div>
    <span class="badge {status}">{status.upper()}</span>
  </header>

  <h2>Inputs</h2>
  <div class="grid">
    {_tensor_card("Reference", report.reference)}
    {_tensor_card("Candidate", report.candidate)}
  </div>

  <h2>Metrics</h2>
  <table><tbody>{metric_rows}</tbody></table>

  <h2>Comparison</h2>
  <table><tbody>
    <tr><th>Elements</th><td class="num">{counts.elements:,}</td></tr>
    <tr><th>Finite pairs</th><td class="num">{counts.finite_elements:,}</td></tr>
    <tr><th>Finite mismatches</th><td class="num">{counts.finite_mismatches:,}</td></tr>
    <tr><th>Non-finite pattern mismatches</th>
      <td class="num">{counts.nonfinite_mismatches:,}</td></tr>
    <tr><th>Tolerance</th>
      <td class="mono">abs(ref - cand) &lt;= {metadata_tolerance(report)}</td></tr>
  </tbody></table>

  <h2>Tensor statistics</h2>
  {_statistics_table(report)}

  <h2>Execution</h2>
  <div class="card">
    <dl>
      <dt>backend</dt><dd>{html.escape(metadata.backend)}</dd>
      <dt>working dtype</dt><dd>{html.escape(metadata.working_dtype)}</dd>
      <dt>parallel</dt><dd>{metadata.parallel} ({metadata.threads} threads,
        {metadata.chunks} chunks × {metadata.chunk_size})</dd>
      <dt>elapsed</dt><dd>{metadata.elapsed_seconds:.6g}s total,
        {metadata.normalization_seconds:.6g}s normalization,
        {_value(metadata.native_seconds)}s native</dd>
      <dt>GIL released</dt><dd>{metadata.release_gil}</dd>
      <dt>Python</dt><dd>{html.escape(metadata.python_version)}</dd>
    </dl>
  </div>

  <h2>Raw data</h2>
  <details>
    <summary>JSON report</summary>
    <pre>{html.escape(report_to_json(report))}</pre>
  </details>

  <footer>Generated by TensorTrail. Non-finite values are excluded from metric sums and
  counted explicitly; see docs/numerical_methods.md for the full policy.</footer>
</main>
</body>
</html>"""
    return document


def metadata_tolerance(report: ComparisonReport) -> str:
    """Human-readable tolerance expression for the HTML report."""
    tolerance = report.tolerances
    return (
        f"{tolerance.atol:.6g} + {tolerance.rtol:.6g} × abs(reference)"
    )


def save_html_report(report: ComparisonReport, destination: Path) -> Path:
    """Write a self-contained HTML report atomically."""
    return _atomic_write(destination, report_to_html(report))
