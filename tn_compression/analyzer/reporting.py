"""Dependency-free analyzer artifacts for machines and human review."""

import html
import json
from pathlib import Path


def _format(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def readable_summary(report):
    summary = report.summary
    lines = [
        f"Analysis level: {report.level}",
        f"Task: {report.task}",
        f"Requested backend: {report.requested_backend}",
        f"Model fingerprint: {report.model_fingerprint}",
        "",
        f"Candidates: {summary.get('candidates', 0)}",
        f"Eligible: {summary.get('eligible', 0)}",
        f"Calibrated: {summary.get('calibrated', 0)}",
        f"Validated: {summary.get('validated', 0)}",
        f"Accepted: {summary.get('accepted', 0)}",
        f"Rejected: {summary.get('rejected', 0)}",
        f"Protected: {summary.get('protected', 0)}",
        f"Unsupported: {summary.get('unsupported', 0)}",
        "",
        "Accepted transformations:",
    ]
    accepted = [candidate for candidate in report.candidates if candidate.decision == "accepted"]
    lines.extend(
        f"- {candidate.layer_path}: {candidate.method} {candidate.configuration}"
        for candidate in accepted)
    if not accepted:
        lines.append("- none")
    lines.extend(["", "Rejected, protected and unsupported:"])
    rejected = [candidate for candidate in report.candidates if candidate.decision == "rejected"]
    lines.extend(
        f"- {candidate.layer_path}: {candidate.method} - "
        f"{candidate.decision_reason or candidate.rejection_reason}"
        for candidate in rejected)
    if not rejected:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def html_report(report):
    rows = []
    for candidate in report.candidates:
        rows.append("<tr>" + "".join(
            f"<td>{html.escape(_format(value))}</td>" for value in (
                candidate.layer_path,
                candidate.method,
                json.dumps(candidate.configuration, sort_keys=True),
                candidate.estimated_bytes_saved,
                candidate.normalized_local_error,
                candidate.quality_loss,
                candidate.backend_support.get("supported"),
                candidate.decision,
                candidate.decision_reason or candidate.rejection_reason,
            )) + "</tr>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Compression analysis</title>
  <style>
    body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d5d8dc; padding: .45rem; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f7; position: sticky; top: 0; }}
    code {{ background: #f4f6f7; padding: .1rem .25rem; }}
  </style>
</head>
<body>
  <h1>Damage-aware compression analysis</h1>
  <p>Level: <strong>{html.escape(report.level)}</strong> · Task: <strong>{html.escape(report.task)}</strong>
     · Backend: <strong>{html.escape(report.requested_backend)}</strong></p>
  <p>Model fingerprint: <code>{html.escape(report.model_fingerprint)}</code></p>
  <pre>{html.escape(readable_summary(report))}</pre>
  <table>
    <thead><tr><th>Layer</th><th>Method</th><th>Configuration</th><th>Estimated bytes saved</th>
      <th>Local error</th><th>Quality loss</th><th>Backend</th><th>Decision</th><th>Reason</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def write_analysis_artifacts(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "analysis": output / "analysis.json",
        "summary": output / "analysis-summary.txt",
        "plan": output / "compression_plan.json",
        "html": output / "analysis-report.html",
    }
    paths["analysis"].write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    paths["summary"].write_text(readable_summary(report))
    paths["plan"].write_text(json.dumps(report.compression_plan, indent=2) + "\n")
    paths["html"].write_text(html_report(report))
    return {name: str(path) for name, path in paths.items()}
