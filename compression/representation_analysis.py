from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tn_compression.api import apply_compression_plan, generate_compression_plan

try:
    from .compress_model import build_compression_config, parse_input_shape, parse_rank, resolve_demo_device, resolve_input_shape
    from .metrics import (
        cosine_and_l2,
        flatten_activation,
        linear_cka,
        set_reproducible_seed,
        singular_spectrum_summary,
    )
    from .models import available_model_names, build_model, get_model_info, load_full_module_artifact, synthetic_input
    from .report import maybe_write_representation_plot, write_json, write_representation_report
except ImportError:
    from compress_model import build_compression_config, parse_input_shape, parse_rank, resolve_demo_device, resolve_input_shape
    from metrics import (
        cosine_and_l2,
        flatten_activation,
        linear_cka,
        set_reproducible_seed,
        singular_spectrum_summary,
    )
    from models import available_model_names, build_model, get_model_info, load_full_module_artifact, synthetic_input
    from report import maybe_write_representation_plot, write_json, write_representation_report


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def select_layers(model: nn.Module, selector: str, *, max_layers: int) -> List[str]:
    if selector != "auto":
        return [item.strip().replace(".", "/") for item in selector.split(",") if item.strip()]
    selected: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        is_block = bool(list(module.children())) and "block" in module.__class__.__name__.lower()
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)) or is_block:
            selected.append(name.replace(".", "/"))
        if len(selected) >= max_layers:
            break
    return selected


def collect_activations(
    model: nn.Module,
    layer_names: List[str],
    *,
    input_batches: List[torch.Tensor],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    model.eval().to(device)
    module_map = {name.replace(".", "/"): module for name, module in model.named_modules()}
    activations: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
    warnings: List[str] = []
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is None:
                warnings.append(f"{layer_name}: hook output is not tensor-compatible")
                return
            activations[layer_name].append(tensor.detach().cpu())
        return hook

    for name in layer_names:
        module = module_map.get(name)
        if module is None:
            warnings.append(f"{name}: layer not found")
            continue
        handles.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        for batch in input_batches:
            _ = model(batch.to(device))
    for handle in handles:
        handle.remove()

    merged: Dict[str, torch.Tensor] = {}
    for name, values in activations.items():
        if not values:
            continue
        try:
            merged[name] = torch.cat(values, dim=0)
        except RuntimeError:
            merged[name] = values[0]
            warnings.append(f"{name}: activation batches could not be concatenated; using first batch")
    return merged, warnings


def analyze_representations(args: argparse.Namespace) -> Dict[str, Any]:
    set_reproducible_seed(args.seed)
    if args.rank_cap is not None and args.rank_cap <= 0:
        args.rank_cap = None
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
    base_input_shape, input_shape_source = resolve_input_shape(
        args.input_shape,
        None if args.model == "artifact" else info.default_input_shape,
        model,
        device,
        require_forward=True,
    )
    assert base_input_shape is not None
    input_shape = (args.batch_size, *tuple(base_input_shape[1:]))
    layer_names = select_layers(model, args.layers, max_layers=args.max_layers)
    plan_path = None
    if args.compressed_artifact_path:
        compressed_model = load_full_module_artifact(args.compressed_artifact_path).to(device)
        compressed_layers = None
        skipped_layers = None
        compression_warnings = []
    else:
        config = build_compression_config(
            output_dir,
            method=args.method,
            rank=args.rank,
            structure=args.structure,
            device=device,
            rank_method=args.rank_method,
            energy=args.energy,
            rank_cap=args.rank_cap,
        )
        plan_result = generate_compression_plan(model, config, inplace=False, device=device)
        plan_path = plan_result.plan_path
        compressed_model = copy.deepcopy(model).to(device)
        compressed_layers, skipped_layers, compression_warnings = apply_compression_plan(compressed_model, plan_result.plan)
    input_batches = [synthetic_input(input_shape, device=device) for _ in range(args.batches)]

    before_acts, before_warnings = collect_activations(
        model,
        layer_names,
        input_batches=input_batches,
        device=device,
    )
    after_acts, after_warnings = collect_activations(
        compressed_model,
        layer_names,
        input_batches=input_batches,
        device=device,
    )

    layer_summaries: List[Dict[str, Any]] = []
    warnings = list(compression_warnings) + before_warnings + after_warnings
    for name in layer_names:
        layer_warnings: List[str] = []
        before = before_acts.get(name)
        after = after_acts.get(name)
        if before is None or after is None:
            warnings.append(f"{name}: missing activation before or after compression")
            continue
        if list(before.shape) != list(after.shape):
            layer_warnings.append(f"activation shape mismatch {list(before.shape)} vs {list(after.shape)}")
        before_matrix = flatten_activation(before, max_rows=args.max_activation_rows)
        after_matrix = flatten_activation(after, max_rows=args.max_activation_rows)
        cka_value, cka_warning = linear_cka(before_matrix, after_matrix, max_rows=args.max_cka_rows)
        if cka_warning:
            layer_warnings.append(cka_warning)
        similarity = cosine_and_l2(before, after)
        layer_summaries.append(
            {
                "name": name,
                "activation_shape": list(before.shape),
                "matrix_shape": list(before_matrix.shape),
                "before": singular_spectrum_summary(before_matrix),
                "after": singular_spectrum_summary(after_matrix),
                "cosine_similarity": similarity.get("cosine_similarity"),
                "relative_l2_diff": similarity.get("relative_l2_diff"),
                "linear_cka": cka_value,
                "warnings": layer_warnings,
            }
        )

    summary = {
        "model": args.model,
        "model_family": info.family,
        "weights": None if args.model == "artifact" else args.weights,
        "input_shape": list(input_shape),
        "input_shape_source": input_shape_source,
        "device": str(device),
        "method": args.method,
        "rank": args.rank,
        "rank_method": args.rank_method,
        "energy": args.energy,
        "rank_cap": args.rank_cap,
        "structure": args.structure,
        "comparison_mode": "provided_compressed_artifact" if args.compressed_artifact_path else "internal_compression",
        "compressed_artifact_path": args.compressed_artifact_path,
        "compressed_layers": compressed_layers,
        "skipped_layers": skipped_layers,
        "plan_path": plan_path,
        "layers": layer_summaries,
        "warnings": warnings,
    }
    plot_path = maybe_write_representation_plot(output_dir / "representation_similarity.png", summary)
    if plot_path:
        summary["plot_path"] = plot_path
    write_json(output_dir / "representation_summary.json", summary)
    write_representation_report(output_dir / "representation_report.md", summary)
    print(f"Analyzed layers: {len(layer_summaries)}")
    if layer_summaries:
        cka_values = [item["linear_cka"] for item in layer_summaries if item.get("linear_cka") is not None]
        if cka_values:
            print(f"Mean linear CKA: {sum(cka_values) / len(cka_values):.6g}")
    print(f"Wrote summary: {output_dir / 'representation_summary.json'}")
    print(f"Wrote report: {output_dir / 'representation_report.md'}")
    return summary


def _first_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze whether compression preserves internal representation geometry.")
    parser.add_argument("--model", default="torchvision_resnet18", choices=available_model_names())
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--compressed-artifact-path", default=None)
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"], help="Torchvision weights for built-in models. 'imagenet' may download weights if unavailable locally.")
    parser.add_argument("--input-shape", type=parse_input_shape, default=None)
    parser.add_argument("--method", default="tensor_train", choices=["tensor_train", "partial_tucker", "cp3", "cp2", "tt"])
    parser.add_argument("--rank", type=parse_rank, default=None)
    parser.add_argument("--rank-method", default="SVD")
    parser.add_argument("--energy", type=float, default=0.94)
    parser.add_argument("--rank-cap", type=int, default=8)
    parser.add_argument("--structure", default="TTPWT")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--max-activation-rows", type=int, default=512)
    parser.add_argument("--max-cka-rows", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=1000, help="Classifier classes for --weights none. ImageNet weights keep the standard 1000-class head.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return analyze_representations(args)


if __name__ == "__main__":
    main()
