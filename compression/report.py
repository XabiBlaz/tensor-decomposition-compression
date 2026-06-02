from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from .metrics import format_bytes, json_safe
except ImportError:
    from metrics import format_bytes, json_safe


KNOWN_LIMITATIONS = [
    "Synthetic data does not prove real accuracy preservation.",
    "Output drift is a functional sanity check, not a benchmark metric.",
    "Compression ratio depends on rank and layer shapes.",
    "Unsupported or non-beneficial layers may be skipped.",
    "Task-specific evaluation belongs outside this compression-only repository.",
    "Latency and task quality must be validated in the deployment or evaluation stack.",
]


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(json_safe(payload), fh, indent=2)


def write_compression_report(path: str | Path, summary: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = summary.get("plan_summary", {})
    drift = summary.get("output_drift", {})
    params = summary.get("parameters", {})
    latency = summary.get("latency", {})
    artifacts = summary.get("artifacts", [])
    warnings = list(summary.get("warnings", [])) + list(drift.get("warnings", []) if isinstance(drift, Mapping) else [])

    lines = [
        "# Compression Demo Report",
        "",
        "## Run",
        "",
        f"- Model: `{summary.get('model')}`",
        f"- Weights: `{summary.get('weights', 'n/a')}`",
        f"- Model family/task assumption: {summary.get('model_family')} / {summary.get('task_assumption')}",
        f"- Input shape: `{summary.get('input_shape')}`",
        f"- Input shape source: `{summary.get('input_shape_source', 'n/a')}`",
        f"- Selected device: `{summary.get('device')}`",
        f"- Compression method: `{summary.get('method')}`",
        f"- Rank / rank policy: `{summary.get('rank')}`",
        f"- Automatic rank method: `{summary.get('rank_method', 'SVD')}`",
        f"- Energy / entropy threshold: `{summary.get('energy', 'n/a')}`",
        f"- Rank cap: `{summary.get('rank_cap', 'n/a')}`",
        f"- Structure: `{summary.get('structure')}`",
        f"- Dry run: `{summary.get('dry_run', False)}`",
        "",
        "## Plan",
        "",
        f"- Eligible layers: {plan.get('eligible_layers', 0)}",
        f"- Compressed layers: {summary.get('compressed_layers', 0)}",
        f"- Skipped layers: {summary.get('skipped_layers', plan.get('skipped_layers', 0))}",
    ]

    skip_reasons = plan.get("skip_reasons", [])
    if skip_reasons:
        lines.extend(["", "### Skip Reasons", ""])
        for item in skip_reasons[:40]:
            reason = item.get("reason") or "n/a"
            lines.append(f"- `{item.get('layer')}` ({item.get('type')}): {reason}")
        if len(skip_reasons) > 40:
            lines.append(f"- ... {len(skip_reasons) - 40} additional skipped layers omitted")

    if params:
        before = params.get("before")
        after = params.get("after")
        ratio = params.get("compression_ratio")
        lines.extend(
            [
                "",
                "## Parameters",
                "",
                f"- Parameters before: {before}",
                f"- Parameters after: {after}",
                f"- Estimated size before: {format_bytes(params.get('size_bytes_before'))}",
                f"- Estimated size after: {format_bytes(params.get('size_bytes_after'))}",
                f"- Compression ratio: {_fmt(ratio)}",
            ]
        )

    if latency:
        before_lat = latency.get("before", {})
        after_lat = latency.get("after", {})
        lines.extend(
            [
                "",
                "## Latency",
                "",
                f"- Before mean latency: {_fmt(before_lat.get('mean_ms'))} ms",
                f"- After mean latency: {_fmt(after_lat.get('mean_ms'))} ms",
                f"- Runs: before={before_lat.get('runs')}, after={after_lat.get('runs')}",
            ]
        )

    if drift:
        lines.extend(
            [
                "",
                "## Output Drift",
                "",
                f"- Numeric drift available: `{drift.get('numeric_drift_available')}`",
                f"- Max absolute difference: {_fmt(drift.get('max_abs_diff'))}",
                f"- Mean absolute difference: {_fmt(drift.get('mean_abs_diff'))}",
                f"- Relative L2 difference: {_fmt(drift.get('relative_l2_diff'))}",
            ]
        )

    lines.extend(["", "## Artifacts", ""])
    if artifacts:
        for artifact in artifacts:
            lines.append(f"- `{artifact.get('kind')}`: `{artifact.get('path')}`")
    else:
        lines.append("- No compressed model artifact saved. Use `--save-model` to write a state dict.")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Known Limitations", ""])
    for item in KNOWN_LIMITATIONS:
        lines.append(f"- {item}")

    path.write_text("\n".join(lines) + "\n")


def write_representation_report(path: str | Path, summary: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Representation Preservation Report",
        "",
        "Does post-training tensor compression preserve internal representation geometry?",
        "",
        "This is not a mechanistic interpretability claim by itself, but it is directly relevant to semantic compression and representation analysis.",
        "",
        "## Run",
        "",
        f"- Model: `{summary.get('model')}`",
        f"- Weights: `{summary.get('weights', 'n/a')}`",
        f"- Input shape: `{summary.get('input_shape')}`",
        f"- Input shape source: `{summary.get('input_shape_source', 'n/a')}`",
        f"- Device: `{summary.get('device')}`",
        f"- Method: `{summary.get('method')}`",
        f"- Rank: `{summary.get('rank')}`",
        f"- Automatic rank method: `{summary.get('rank_method', 'SVD')}`",
        f"- Energy / entropy threshold: `{summary.get('energy', 'n/a')}`",
        f"- Rank cap: `{summary.get('rank_cap', 'n/a')}`",
        f"- Layers analyzed: {len(summary.get('layers', []))}",
        "",
        "## Layer Metrics",
        "",
    ]
    for layer in summary.get("layers", []):
        lines.extend(
            [
                f"### `{layer.get('name')}`",
                "",
                f"- Activation shape: `{layer.get('activation_shape')}`",
                f"- Effective rank before/after: {_fmt(layer.get('before', {}).get('effective_rank'))} / {_fmt(layer.get('after', {}).get('effective_rank'))}",
                f"- Participation ratio before/after: {_fmt(layer.get('before', {}).get('participation_ratio'))} / {_fmt(layer.get('after', {}).get('participation_ratio'))}",
                f"- Cosine similarity: {_fmt(layer.get('cosine_similarity'))}",
                f"- Linear CKA: {_fmt(layer.get('linear_cka'))}",
                f"- Relative L2 difference: {_fmt(layer.get('relative_l2_diff'))}",
                "",
            ]
        )
        if layer.get("warnings"):
            for warning in layer["warnings"]:
                lines.append(f"- Warning: {warning}")
            lines.append("")

    warnings = summary.get("warnings", [])
    lines.extend(["## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Synthetic inputs probe geometry preservation under controlled random inputs, not real task accuracy.",
            "- CKA and rank summaries are lightweight diagnostics, not proof of semantic equivalence.",
            "- Layers may be skipped when activations are too large, unavailable, or shape-incompatible.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def maybe_write_compression_plot(path: str | Path, summary: Mapping[str, Any]) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None
    params = summary.get("parameters", {})
    if not params:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Before", "After"]
    values = [params.get("before", 0), params.get("after", 0)]
    try:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(labels, values, color=["#4c78a8", "#f58518"])
        ax.set_ylabel("Parameters")
        ax.set_title("Compression Summary")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def maybe_write_representation_plot(path: str | Path, summary: Mapping[str, Any]) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None
    layers = summary.get("layers", [])
    if not layers:
        return None
    path = Path(path)
    names = [str(layer.get("name")) for layer in layers]
    cka = [layer.get("linear_cka") or 0.0 for layer in layers]
    try:
        fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.6), 3))
        ax.bar(range(len(names)), cka, color="#54a24b")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Linear CKA")
        ax.set_title("Representation Similarity")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
