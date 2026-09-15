from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tn_compression.api import is_eligible_layer

try:
    from .compress_model import parse_input_shape, resolve_demo_device, resolve_input_shape
    from .metrics import (
        flatten_activation,
        format_bytes,
        linear_cka,
        set_reproducible_seed,
        singular_spectrum_summary,
    )
    from .models import available_model_names, build_model, get_model_info, synthetic_input
    from .report import write_json
except ImportError:
    from compress_model import parse_input_shape, resolve_demo_device, resolve_input_shape
    from metrics import (
        flatten_activation,
        format_bytes,
        linear_cka,
        set_reproducible_seed,
        singular_spectrum_summary,
    )
    from models import available_model_names, build_model, get_model_info, synthetic_input
    from report import write_json


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def analyze_model(args: argparse.Namespace) -> Dict[str, Any]:
    set_reproducible_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_demo_device(args.device)
    info = get_model_info(args.model)
    model = build_model(
        args.model,
        artifact_path=args.artifact_path,
        weights=args.weights,
        num_classes=args.num_classes,
    ).to(device)
    model.eval()
    input_shape, input_shape_source = resolve_input_shape(
        args.input_shape,
        None if args.model == "artifact" else info.default_input_shape,
        model,
        device,
        require_forward=False,
    )

    layers = inspect_layers(model)
    total_params = int(sum(item["params"] for item in layers))
    total_bytes = int(sum(item["param_bytes"] for item in layers))
    warnings: List[str] = []
    activation_summary: Dict[str, Any] = {}

    if input_shape is None:
        warnings.append(
            "Activation probes skipped because no input shape was provided. "
            "Parameter and layer-eligibility analysis does not require an input shape."
        )
    else:
        activation_summary = run_activation_probes(
            model,
            input_shape,
            device=device,
            max_layers=args.max_layers,
            max_activation_rows=args.max_activation_rows,
        )
        warnings.extend(activation_summary.get("warnings", []))
    segments, grouping_summary = build_cka_segments(
        layers,
        activation_summary,
        threshold=args.cka_threshold,
        fisher_classes=args.cka_fisher_classes,
    )

    summary = {
        "model": args.model,
        "model_family": info.family,
        "weights": None if args.model == "artifact" else args.weights,
        "input_shape": list(input_shape) if input_shape is not None else None,
        "input_shape_source": input_shape_source,
        "device": str(device),
        "total_parameters": total_params,
        "total_parameter_size_bytes": total_bytes,
        "total_parameter_size": format_bytes(total_bytes),
        "layers": layers,
        "activation_analysis": activation_summary,
        "cka_grouping": grouping_summary,
        "cka_segments": segments,
        "warnings": warnings,
        "notes": [
            "This is inspection only; no compression is applied.",
            "CKA requires representative inputs. With synthetic inputs it is a structural diagnostic, not proof of compression safety.",
            "Use the layer table, params-per-layer plot, CKA heatmap, and CKA groups to decide whether a global, individual, or grouped compression policy is appropriate.",
        ],
    }
    plots = maybe_write_precompression_plots(output_dir, summary)
    if plots:
        summary["plots"] = plots
    if segments:
        write_json(output_dir / "segments.json", segments)
    write_json(output_dir / "precompression_analysis.json", summary)
    write_precompression_report(output_dir / "precompression_report.md", summary)
    print(f"Layers inspected: {len(layers)}")
    print(f"Total parameters: {total_params} ({format_bytes(total_bytes)})")
    if input_shape is None:
        print("Activation probes skipped; pass --input-shape if this artifact uses a non-standard input.")
    else:
        print(f"Activation layers analyzed: {len(activation_summary.get('layers', []))}")
    print(f"Wrote summary: {output_dir / 'precompression_analysis.json'}")
    print(f"Wrote report: {output_dir / 'precompression_report.md'}")
    if segments:
        print(f"Wrote CKA groups: {output_dir / 'segments.json'}")
    return summary


def inspect_layers(model: nn.Module) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if not name or list(module.children()):
            continue
        params = int(sum(p.numel() for p in module.parameters(recurse=False)))
        param_bytes = int(sum(p.numel() * p.element_size() for p in module.parameters(recurse=False)))
        eligible, reason = is_eligible_layer(module)
        layer_info = {
            "name": name,
            "type": module.__class__.__name__,
            "params": params,
            "param_bytes": param_bytes,
            "param_size": format_bytes(param_bytes),
            "eligible_for_compression": eligible,
            "eligibility_reason": reason,
            "compatible_methods": compatible_methods(module),
            "shape": module_shape(module),
        }
        layers.append(layer_info)
    return layers


def compatible_methods(module: nn.Module) -> List[str]:
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            return []
        return ["tensor_train", "partial_tucker", "cp3", "cp4"]
    if isinstance(module, nn.Linear):
        return ["tensor_train", "svd", "cp2"]
    return []


def module_shape(module: nn.Module) -> Optional[List[int]]:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return list(weight.shape)
    return None


def run_activation_probes(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    *,
    device: torch.device,
    max_layers: int,
    max_activation_rows: int,
) -> Dict[str, Any]:
    layer_names = select_probe_layers(model, max_layers=max_layers)
    activations, warnings = collect_activations(model, layer_names, input_shape, device=device)
    activation_layers: List[Dict[str, Any]] = []
    matrices: Dict[str, torch.Tensor] = {}
    for name in layer_names:
        act = activations.get(name)
        if act is None:
            continue
        matrix = flatten_activation(act, max_rows=max_activation_rows)
        matrices[name] = matrix
        spectrum = singular_spectrum_summary(matrix)
        activation_layers.append(
            {
                "name": name,
                "activation_shape": list(act.shape),
                "matrix_shape": list(matrix.shape),
                "effective_rank": spectrum.get("effective_rank"),
                "participation_ratio": spectrum.get("participation_ratio"),
                "top5_energy_fraction": spectrum.get("top5_energy_fraction"),
            }
        )

    cka_matrix, cka_warnings = pairwise_cka(matrices)
    warnings.extend(cka_warnings)
    return {
        "layers": activation_layers,
        "layer_names": list(matrices.keys()),
        "cka_matrix": cka_matrix,
        "warnings": warnings,
    }


def select_probe_layers(model: nn.Module, *, max_layers: int) -> List[str]:
    names: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(name)
        if len(names) >= max_layers:
            break
    return names


def collect_activations(
    model: nn.Module,
    layer_names: List[str],
    input_shape: Tuple[int, ...],
    *,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    activations: Dict[str, torch.Tensor] = {}
    warnings: List[str] = []
    module_map = dict(model.named_modules())
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                warnings.append(f"{layer_name}: activation output is not tensor-compatible")
                return
            activations[layer_name] = tensor.detach().cpu()
        return hook

    for name in layer_names:
        module = module_map.get(name)
        if module is None:
            warnings.append(f"{name}: layer not found")
            continue
        handles.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        _ = model(synthetic_input(input_shape, device=device))
    for handle in handles:
        handle.remove()
    return activations, warnings


def pairwise_cka(matrices: Mapping[str, torch.Tensor]) -> Tuple[List[List[Optional[float]]], List[str]]:
    names = list(matrices.keys())
    warnings: List[str] = []
    result: List[List[Optional[float]]] = []
    for left in names:
        row: List[Optional[float]] = []
        for right in names:
            score, warning = linear_cka(matrices[left], matrices[right])
            row.append(score)
            if warning and left != right:
                warnings.append(f"{left} vs {right}: {warning}")
        result.append(row)
    return result, warnings


def build_cka_segments(
    layers: List[Dict[str, Any]],
    activation_summary: Mapping[str, Any],
    *,
    threshold: Optional[float],
    fisher_classes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Cluster eligible probed layers using Fisher-Jenks segmentation over CKA scores."""
    names = [str(name) for name in activation_summary.get("layer_names", [])]
    cka_matrix = activation_summary.get("cka_matrix", [])
    if not names or not cka_matrix:
        return [], {"method": "fisher_jenks_cka", "available": False, "reason": "missing_cka_matrix"}
    eligible = {str(layer["name"]) for layer in layers if layer.get("eligible_for_compression")}
    candidate_indices = [index for index, name in enumerate(names) if name in eligible]
    if not candidate_indices:
        return [], {"method": "fisher_jenks_cka", "available": False, "reason": "no_eligible_probed_layers"}

    scores = [
        score
        for left_pos, left in enumerate(candidate_indices)
        for right in candidate_indices[left_pos + 1 :]
        for score in [_cka_value(cka_matrix, left, right)]
        if score is not None
    ]
    if threshold is None:
        resolved_threshold, breaks = fisher_jenks_high_similarity_threshold(scores, classes=fisher_classes)
        threshold_source = "fisher_jenks"
    else:
        resolved_threshold = float(threshold)
        breaks = []
        threshold_source = "manual"

    remaining = set(candidate_indices)
    components: List[List[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for other in list(remaining):
                score = _cka_value(cka_matrix, current, other)
                if score is not None and score >= resolved_threshold:
                    remaining.remove(other)
                    stack.append(other)
                    component.append(other)
        components.append(sorted(component))

    segments: List[Dict[str, Any]] = []
    for index, component in enumerate(components):
        group_names = [names[item] for item in component]
        internal_scores = [
            score
            for left_pos, left in enumerate(component)
            for right in component[left_pos + 1 :]
            for score in [_cka_value(cka_matrix, left, right)]
            if score is not None
        ]
        mean_cka = (sum(internal_scores) / len(internal_scores)) if internal_scores else None
        segments.append(
            {
                "id": f"group_{index}",
                "name": f"cka_group_{index}",
                "criterion": "fisher_jenks_cka_connected_components",
                "threshold": resolved_threshold,
                "threshold_source": threshold_source,
                "fisher_breaks": breaks,
                "layers": group_names,
                "mean_internal_cka": mean_cka,
                "interpretation": _cka_segment_interpretation(len(group_names), mean_cka),
            }
        )
    return segments, {
        "method": "fisher_jenks_cka",
        "available": True,
        "threshold": resolved_threshold,
        "threshold_source": threshold_source,
        "fisher_classes": fisher_classes,
        "fisher_breaks": breaks,
        "pairwise_scores": len(scores),
        "eligible_probed_layers": len(candidate_indices),
        "groups": len(segments),
    }


def fisher_jenks_high_similarity_threshold(scores: List[float], *, classes: int) -> Tuple[float, List[float]]:
    values = sorted(float(score) for score in scores if score is not None)
    if not values:
        return 1.0, []
    unique_values = sorted(set(values))
    if len(unique_values) < 2:
        return unique_values[0], unique_values
    class_count = max(2, min(int(classes), len(unique_values), len(values)))
    breaks = fisher_jenks_breaks(values, class_count)
    threshold = breaks[-2] if len(breaks) >= 2 else unique_values[-1]
    return float(threshold), [float(item) for item in breaks]


def fisher_jenks_breaks(values: List[float], classes: int) -> List[float]:
    """Return Fisher-Jenks natural breaks for sorted numeric values."""
    data = sorted(values)
    n = len(data)
    k = max(1, min(classes, n))
    lower = [[0] * (k + 1) for _ in range(n + 1)]
    variance = [[float("inf")] * (k + 1) for _ in range(n + 1)]
    for class_index in range(1, k + 1):
        lower[1][class_index] = 1
        variance[1][class_index] = 0.0
    for length in range(2, n + 1):
        sum_values = 0.0
        sum_squares = 0.0
        count = 0
        for start in range(length, 0, -1):
            value = data[start - 1]
            count += 1
            sum_values += value
            sum_squares += value * value
            segment_variance = sum_squares - (sum_values * sum_values) / count
            previous = start - 1
            if previous:
                for class_index in range(2, k + 1):
                    candidate = segment_variance + variance[previous][class_index - 1]
                    if variance[length][class_index] >= candidate:
                        lower[length][class_index] = start
                        variance[length][class_index] = candidate
        lower[length][1] = 1
        variance[length][1] = segment_variance

    breaks = [0.0] * (k + 1)
    breaks[k] = data[-1]
    breaks[0] = data[0]
    count = n
    class_index = k
    while class_index > 1:
        idx = int(lower[count][class_index]) - 2
        breaks[class_index - 1] = data[max(0, idx)]
        count = int(lower[count][class_index] - 1)
        class_index -= 1
    return breaks


def _cka_value(matrix: Any, left: int, right: int) -> Optional[float]:
    try:
        value = matrix[left][right]
    except Exception:
        return None
    if value is None:
        return None
    return float(value)


def _cka_segment_interpretation(size: int, mean_cka: Optional[float]) -> str:
    if size <= 1:
        return "No other probed eligible layer exceeded the CKA threshold."
    if mean_cka is not None and mean_cka >= 0.95:
        return "Very high internal activation-geometry similarity; a shared conservative policy is reasonable to test."
    return "Layers exceeded the CKA threshold; test a shared policy and validate task metrics after compression."


def first_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def write_precompression_report(path: Path, summary: Mapping[str, Any]) -> None:
    layers = list(summary.get("layers", []))
    trainable_layers = [layer for layer in layers if layer.get("params", 0) > 0]
    top_layers = sorted(trainable_layers, key=lambda item: int(item.get("params", 0)), reverse=True)[:20]
    activation = summary.get("activation_analysis", {})
    lines = [
        "# Pre-Compression Model Inspection",
        "",
        f"- Model: `{summary.get('model')}`",
        f"- Weights: `{summary.get('weights', 'n/a')}`",
        f"- Input shape for activation probes: `{summary.get('input_shape')}`",
        f"- Input shape source: `{summary.get('input_shape_source')}`",
        f"- Total parameters: {summary.get('total_parameters')}",
        f"- Estimated parameter memory: {summary.get('total_parameter_size')}",
        "",
        "## Largest Parameter Layers",
        "",
        "| Layer | Type | Params | Compatible methods |",
        "|---|---|---:|---|",
    ]
    for layer in top_layers:
        lines.append(
            f"| `{layer.get('name')}` | {layer.get('type')} | {layer.get('params')} | "
            f"{', '.join(layer.get('compatible_methods') or []) or 'none'} |"
        )
    eligible = [layer for layer in layers if layer.get("eligible_for_compression")]
    skipped = [layer for layer in layers if not layer.get("eligible_for_compression")]
    lines.extend(
        [
            "",
            "## Compression Eligibility",
            "",
            f"- Eligible leaf layers: {len(eligible)}",
            f"- Skipped leaf layers: {len(skipped)}",
            "",
            "Common skip reasons include unsupported module types, grouped convolutions, and layers with no direct parameters.",
        ]
    )
    if activation.get("layers"):
        lines.extend(
            [
                "",
                "## Activation Probes",
                "",
                "Activation probes use a synthetic input with the requested shape. Use task data externally for stronger conclusions.",
                "",
                "| Layer | Effective rank | Participation ratio | Top-5 energy fraction |",
                "|---|---:|---:|---:|",
            ]
        )
        for layer in activation["layers"]:
            lines.append(
                f"| `{layer.get('name')}` | {_fmt(layer.get('effective_rank'))} | "
                f"{_fmt(layer.get('participation_ratio'))} | {_fmt(layer.get('top5_energy_fraction'))} |"
            )
    segments = summary.get("cka_segments", [])
    if segments:
        lines.extend(
            [
                "",
                "## CKA Groups",
                "",
                "`segments.json` was written for `compression.mode = cka_groups`.",
                "Groups are connected components of the layer CKA similarity graph. By default, the threshold is selected with Fisher-Jenks natural breaks over pairwise CKA scores.",
                "",
                "| Segment | Layers | Mean internal CKA | Interpretation |",
                "|---|---:|---:|---|",
            ]
        )
        for segment in segments:
            lines.append(
                f"| `{segment.get('name')}` | {len(segment.get('layers', []))} | "
                f"{_fmt(segment.get('mean_internal_cka'))} | {segment.get('interpretation')} |"
            )
    plots = summary.get("plots", [])
    if plots:
        lines.extend(["", "## Visualizations", ""])
        for plot in plots:
            lines.append(f"- `{plot}`")
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## How To Use This",
            "",
            "- Large parameter layers usually give the largest compression payoff.",
            "- High CKA similarity between layers suggests similar activation geometry and can define grouped compression candidates.",
            "- Low activation effective rank can suggest that a layer may tolerate a lower-rank decomposition.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def maybe_write_precompression_plots(output_dir: Path, summary: Mapping[str, Any]) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return []
    plots: List[str] = []
    layers = [layer for layer in summary.get("layers", []) if layer.get("params", 0) > 0]
    top_layers = sorted(layers, key=lambda item: int(item.get("params", 0)), reverse=True)[:25]
    if top_layers:
        path = output_dir / "params_per_layer.png"
        names = [str(layer.get("name")) for layer in top_layers]
        values = [int(layer.get("params") or 0) for layer in top_layers]
        fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.35), 4))
        ax.bar(range(len(names)), values, color="#4c78a8")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Parameters")
        ax.set_title("Largest Parameter Layers")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(str(path))

    activation = summary.get("activation_analysis", {})
    names = activation.get("layer_names", [])
    cka_matrix = activation.get("cka_matrix", [])
    if names and cka_matrix:
        path = output_dir / "cka_heatmap.png"
        matrix = torch.tensor([[0.0 if value is None else float(value) for value in row] for row in cka_matrix])
        fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.45), max(4, len(names) * 0.4)))
        image = ax.imshow(matrix.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title("Layer CKA Similarity")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(str(path))

    activation_layers = activation.get("layers", [])
    if activation_layers:
        path = output_dir / "activation_effective_rank.png"
        names = [str(item.get("name")) for item in activation_layers]
        values = [float(item.get("effective_rank") or 0.0) for item in activation_layers]
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.5), 3.5))
        ax.bar(range(len(names)), values, color="#54a24b")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Effective rank")
        ax.set_title("Activation Effective Rank")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(str(path))

    segments = summary.get("cka_segments", [])
    if segments:
        path = output_dir / "cka_groups.png"
        names = [str(item.get("name")) for item in segments]
        counts = [len(item.get("layers", [])) for item in segments]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(range(len(names)), counts, color="#b279a2")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Eligible layers")
        ax.set_title("CKA Groups")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(str(path))
    return plots


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a model before compression without mutating it.")
    parser.add_argument("--model", default="torchvision_resnet18", choices=available_model_names())
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"], help="Torchvision weights for built-in models. 'imagenet' may download weights if unavailable locally.")
    parser.add_argument("--input-shape", type=parse_input_shape, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-layers", type=int, default=16)
    parser.add_argument("--max-activation-rows", type=int, default=512)
    parser.add_argument("--cka-threshold", type=float, default=None, help="Manual pairwise CKA grouping threshold. If omitted, Fisher-Jenks selects the high-similarity threshold.")
    parser.add_argument("--cka-fisher-classes", type=int, default=3, help="Number of Fisher-Jenks classes used to segment pairwise CKA scores.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Classifier classes for --weights none. ImageNet weights keep the standard 1000-class head.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return analyze_model(args)


if __name__ == "__main__":
    main()
