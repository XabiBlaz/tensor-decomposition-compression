from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tn_compression.api import apply_compression_plan, generate_compression_plan
from tn_compression.config import load_config

try:
    from .metrics import (
        benchmark_latency_ms,
        compare_outputs,
        count_parameters,
        parameter_size_bytes,
        plan_layer_summary,
        set_reproducible_seed,
    )
    from .models import available_model_names, build_model, get_model_info, synthetic_input
    from .report import maybe_write_compression_plot, write_compression_report, write_json
except ImportError:
    from metrics import (
        benchmark_latency_ms,
        compare_outputs,
        count_parameters,
        parameter_size_bytes,
        plan_layer_summary,
        set_reproducible_seed,
    )
    from models import available_model_names, build_model, get_model_info, synthetic_input
    from report import maybe_write_compression_plot, write_compression_report, write_json


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
COMMON_INPUT_SHAPES: Tuple[Tuple[int, ...], ...] = (
    (1, 3, 224, 224),
    (1, 3, 128, 128),
    (1, 3, 64, 64),
    (1, 3, 32, 32),
    (1, 1, 28, 28),
)


def parse_input_shape(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if text is None:
        return None
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--input-shape must be comma-separated integers, e.g. 1,3,224,224") from exc
    if len(values) < 2 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--input-shape must contain at least batch and feature dimensions, all positive")
    return values


def parse_rank(text: str) -> Any:
    if "," in text:
        try:
            return [int(part.strip()) for part in text.split(",") if part.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--rank values must be integers") from exc
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--rank must be an integer or comma-separated integer list") from exc


def resolve_demo_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{requested}' was requested, but torch.cuda.is_available() is False."
            )
        return torch.device(requested)
    raise RuntimeError(f"Unsupported device '{requested}'. Use cpu, auto, cuda, or cuda:0.")


def infer_working_input_shape(model: torch.nn.Module, device: torch.device) -> Optional[Tuple[int, ...]]:
    model.eval().to(device)
    with torch.no_grad():
        for shape in COMMON_INPUT_SHAPES:
            try:
                _ = model(torch.randn(*shape, device=device))
                return shape
            except Exception:
                continue
    return None


def resolve_input_shape(
    requested: Optional[Tuple[int, ...]],
    default_shape: Optional[Tuple[int, ...]],
    model: torch.nn.Module,
    device: torch.device,
    *,
    require_forward: bool,
) -> Tuple[Optional[Tuple[int, ...]], str]:
    if requested is not None:
        return requested, "provided"
    if default_shape is not None:
        return default_shape, "model_default"
    inferred = infer_working_input_shape(model, device)
    if inferred is not None:
        return inferred, "auto_probe"
    if require_forward:
        raise RuntimeError(
            "Could not infer an input tensor shape for this artifact. "
            "Pass --input-shape as batch,channels,height,width, for example 1,3,224,224."
        )
    return None, "unavailable"


def build_compression_config(
    output_dir: Path,
    *,
    method: str,
    rank: Any,
    structure: str,
    device: torch.device,
    rank_method: str = "SVD",
    energy: float = 0.94,
    rank_cap: Optional[int] = None,
) -> Dict[str, Any]:
    method_cfg: Dict[str, Any] = {
        "type": method,
        "method": rank_method,
        "energy": energy,
        "rank": rank,
    }
    if rank_cap is not None:
        method_cfg["rank_cap"] = rank_cap
    if method in {"tensor_train", "tt"}:
        method_cfg["structure"] = structure
    return {
        "schema_version": 1,
        "artifact": {"kind": "live_module"},
        "runtime": {"device": str(device)},
        "compression": {
            "mode": "default_all",
            "default_method": method_cfg,
        },
        "output": {
            "dir": str(output_dir),
            "plan_dir": str(output_dir),
            "name": "compressed_model",
            "artifacts": [{"kind": "state_dict"}],
        },
    }


def load_cli_config(args: argparse.Namespace, output_dir: Path, device: torch.device) -> Dict[str, Any]:
    if not args.config:
        return build_compression_config(
            output_dir,
            method=args.method,
            rank=args.rank,
            structure=args.structure,
            device=device,
            rank_method=args.rank_method,
            energy=args.energy,
            rank_cap=args.rank_cap,
        )
    config = load_config(args.config)
    config.setdefault("schema_version", 1)
    config.setdefault("artifact", {"kind": "live_module"})
    config.setdefault("runtime", {})
    config["runtime"]["device"] = str(device)
    config.setdefault("output", {})
    if args.output_dir:
        config["output"]["dir"] = str(output_dir)
        config["output"]["plan_dir"] = str(output_dir)
    else:
        config["output"].setdefault("dir", str(output_dir))
        config["output"].setdefault("plan_dir", config["output"]["dir"])
    config["output"].setdefault("name", "compressed_model")
    config["output"].setdefault("artifacts", [{"kind": "state_dict"}])
    return config


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    set_reproducible_seed(args.seed)
    if args.rank_cap is not None and args.rank_cap <= 0:
        args.rank_cap = None
    requested_output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    device = resolve_demo_device(args.device)
    info = get_model_info(args.model)
    model = build_model(
        args.model,
        artifact_path=args.artifact_path,
        weights=args.weights,
        num_classes=args.num_classes,
    )
    model.eval().to(device)
    input_shape, input_shape_source = resolve_input_shape(
        args.input_shape,
        None if args.model == "artifact" else info.default_input_shape,
        model,
        device,
        require_forward=True,
    )
    assert input_shape is not None
    sample = synthetic_input(input_shape, device=device)
    config = load_cli_config(args, requested_output_dir, device)
    output_dir = Path(config.get("output", {}).get("dir", requested_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_method = config.get("compression", {}).get("default_method", {}).get("type", args.method)
    selected_structure = config.get("compression", {}).get("default_method", {}).get("structure", args.structure)
    selected_rank = config.get("compression", {}).get("default_method", {}).get("rank", args.rank)
    selected_rank_method = config.get("compression", {}).get("default_method", {}).get("method", args.rank_method)
    selected_energy = config.get("compression", {}).get("default_method", {}).get("energy", args.energy)
    selected_rank_cap = config.get("compression", {}).get("default_method", {}).get("rank_cap", args.rank_cap)
    plan_result = generate_compression_plan(model, config, inplace=False, device=device)
    plan_summary = plan_layer_summary(plan_result.plan)

    if args.dry_run_plan:
        summary = {
            "model": args.model,
            "model_family": info.family,
            "task_assumption": info.task,
            "weights": None if args.model == "artifact" else args.weights,
            "input_shape": list(input_shape),
            "input_shape_source": input_shape_source,
            "device": str(device),
            "config_path": args.config,
            "method": selected_method,
            "rank": selected_rank,
            "rank_method": selected_rank_method,
            "energy": selected_energy,
            "rank_cap": selected_rank_cap,
            "structure": selected_structure,
            "dry_run": True,
            "plan_path": plan_result.plan_path,
            "plan_summary": plan_summary,
            "compressed_layers": 0,
            "skipped_layers": plan_summary["skipped_layers"],
            "warnings": [],
            "artifacts": [],
        }
        write_json(output_dir / "summary.json", summary)
        write_compression_report(output_dir / "report.md", summary)
        _print_plan(plan_summary, dry_run=True)
        return summary

    before_params = count_parameters(model)
    before_size = parameter_size_bytes(model)
    before_latency = benchmark_latency_ms(model, sample, device=device, warmup=args.warmup_runs, runs=args.latency_runs)
    with torch.no_grad():
        before_output = model(sample)

    compressed_model = copy.deepcopy(model).to(device)
    compressed_layers, skipped_layers, compression_warnings = apply_compression_plan(compressed_model, plan_result.plan)
    compressed_model.eval().to(device)
    after_params = count_parameters(compressed_model)
    after_size = parameter_size_bytes(compressed_model)
    after_latency = benchmark_latency_ms(
        compressed_model,
        sample,
        device=device,
        warmup=args.warmup_runs,
        runs=args.latency_runs,
    )
    with torch.no_grad():
        after_output = compressed_model(sample)
    drift = compare_outputs(before_output, after_output)

    warnings = list(compression_warnings)

    artifacts = []
    if args.save_model:
        state_path = output_dir / "compressed_model_state_dict.pth"
        torch.save(compressed_model.state_dict(), state_path)
        artifacts.append({"kind": "state_dict", "path": str(state_path)})
        if args.save_full_module:
            module_path = output_dir / "compressed_model_full.pt"
            torch.save(compressed_model, module_path)
            artifacts.append({"kind": "full_module", "path": str(module_path)})

    compression_ratio = (after_params / before_params) if before_params else None
    summary = {
        "model": args.model,
        "model_family": info.family,
        "task_assumption": info.task,
        "weights": None if args.model == "artifact" else args.weights,
        "input_shape": list(input_shape),
        "input_shape_source": input_shape_source,
        "device": str(device),
        "config_path": args.config,
        "method": selected_method,
        "rank": selected_rank,
        "rank_method": selected_rank_method,
        "energy": selected_energy,
        "rank_cap": selected_rank_cap,
        "structure": selected_structure,
        "dry_run": False,
        "plan_path": plan_result.plan_path,
        "plan_summary": plan_summary,
        "compressed_layers": compressed_layers,
        "skipped_layers": skipped_layers,
        "parameters": {
            "before": before_params,
            "after": after_params,
            "size_bytes_before": before_size,
            "size_bytes_after": after_size,
            "compression_ratio": compression_ratio,
        },
        "latency": {
            "before": before_latency,
            "after": after_latency,
        },
        "output_drift": drift,
        "warnings": warnings,
        "artifacts": artifacts,
    }
    plot_path = maybe_write_compression_plot(output_dir / "compression_summary.png", summary)
    if plot_path:
        summary["artifacts"].append({"kind": "plot", "path": plot_path})
    write_json(output_dir / "summary.json", summary)
    write_compression_report(output_dir / "report.md", summary)
    _print_plan(plan_summary, dry_run=False)
    print(f"Compressed layers: {compressed_layers}")
    print(f"Skipped layers: {skipped_layers}")
    print(f"Parameters: {before_params} -> {after_params}")
    if drift.get("numeric_drift_available"):
        print(f"Output relative L2 drift: {drift.get('relative_l2_diff'):.6g}")
    print(f"Wrote summary: {output_dir / 'summary.json'}")
    print(f"Wrote report: {output_dir / 'report.md'}")
    return summary


def _print_plan(plan_summary: Dict[str, Any], *, dry_run: bool) -> None:
    print(f"Eligible layers: {plan_summary['eligible_layers']}")
    if dry_run:
        print("Compressed layers: 0 (dry run)")
    print(f"Skipped layers: {plan_summary['skipped_layers']}")
    for name in plan_summary.get("eligible_layer_names", [])[:20]:
        print(f"  compress: {name}")
    if len(plan_summary.get("eligible_layer_names", [])) > 20:
        print(f"  ... {len(plan_summary['eligible_layer_names']) - 20} additional eligible layers")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-first research demo for tensor-decomposition model compression.")
    parser.add_argument("--model", default="torchvision_resnet18", choices=available_model_names())
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"], help="Torchvision weights for built-in models. 'imagenet' may download weights if unavailable locally.")
    parser.add_argument("--input-shape", type=parse_input_shape, default=None)
    parser.add_argument("--config", default=None, help="Optional YAML compression config. CLI method/rank flags are ignored when this is set.")
    parser.add_argument("--method", default="tensor_train", choices=["tensor_train", "partial_tucker", "cp3", "cp2", "tt"])
    parser.add_argument("--rank", type=parse_rank, default=None)
    parser.add_argument("--rank-method", default="SVD", help="Automatic rank policy when --rank is omitted: SVD, ENTROPY, VBMF, or EVBMF where supported.")
    parser.add_argument("--energy", type=float, default=0.94, help="Energy/entropy threshold used by automatic rank selection.")
    parser.add_argument("--rank-cap", type=int, default=8, help="Upper bound for automatic ranks. Use 0 to disable the cap.")
    parser.add_argument("--structure", default="TTPWT")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--save-full-module", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latency-runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=1000, help="Classifier classes for --weights none. ImageNet weights keep the standard 1000-class head.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_demo(args)


if __name__ == "__main__":
    main()
