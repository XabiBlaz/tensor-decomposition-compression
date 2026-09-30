"""User-selected, measured compression methods with explicit capability checks."""

import math


def _tensor_bytes(model):
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def validate_recipe(recipe, task):
    if not isinstance(recipe, dict):
        raise ValueError("Select a compression recipe.")
    methods = recipe.get("methods")
    allowed = {"pruning", "quantization", "tensor_decomposition"}
    if not isinstance(methods, list) or not methods or len(methods) != len(set(methods)) or set(methods) - allowed:
        raise ValueError("Choose one or more of pruning, quantization and tensor decomposition.")
    if task not in {"classification", "segmentation", "causal_lm"}:
        raise ValueError("Direct recipes support classification, segmentation and causal language models.")
    if "pruning" in methods and task != "causal_lm":
        raise ValueError("Physical pruning currently supports Llama/Qwen gated MLPs only, not arbitrary vision networks.")
    if "tensor_decomposition" in methods:
        energy = recipe.get("svd_energy")
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy) or not 0 < energy <= 1:
            raise ValueError("SVD energy must be a number greater than 0 and at most 1.")
    if "pruning" in methods:
        retain = recipe.get("pruning_retention")
        if isinstance(retain, bool) or not isinstance(retain, (int, float)) or not math.isfinite(retain) or not 0 < retain < 1:
            raise ValueError("Pruning retention must be between 0 and 1.")
    if "quantization" in methods and recipe.get("quantization_bits", 8) != 8:
        raise ValueError("The portable weight-storage backend supports 8-bit quantization only.")
    for key in ("include", "exclude"):
        paths = recipe.get(key, [])
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError(f"recipe.{key} must be a list of layer paths.")
    return methods


def apply_direct_recipe(model, recipe, *, task):
    """Mutate a model and return a JSON-safe, honest report of actual changes."""
    methods = validate_recipe(recipe, task)
    original = _tensor_bytes(model)
    stages = []
    # Structural pruning changes Llama/Qwen MLP widths and must precede factorization.
    if "pruning" in methods:
        from .pruning import gated_mlps, prune_gated_mlps, select_channels
        groups = gated_mlps(model)
        selections = {path: select_channels(module, max(1, int(module.down_proj.in_features * recipe["pruning_retention"])))
                      for path, module in groups.items()}
        transformation = prune_gated_mlps(model, selections)
        stages.append({"method": "pruning", "transformation": transformation,
                       "tensor_bytes_after": _tensor_bytes(model)})
    if "tensor_decomposition" in methods:
        from .vision_upload import compress_svd_energy
        report = compress_svd_energy(model, recipe["svd_energy"],
                                     include=recipe.get("include"), exclude=recipe.get("exclude"))
        stages.append({"method": "tensor_decomposition", "report": report,
                       "tensor_bytes_after": _tensor_bytes(model)})
    if "quantization" in methods:
        from .packed_quantization import quantize_int8_weights
        report = quantize_int8_weights(model, include=recipe.get("include"), exclude=recipe.get("exclude"))
        stages.append({"method": "quantization", "report": report,
                       "tensor_bytes_after": _tensor_bytes(model)})
    final = _tensor_bytes(model)
    previous = original
    changed = []
    for stage in stages:
        if stage["tensor_bytes_after"] < previous:
            changed.append(stage)
        previous = stage["tensor_bytes_after"]
    if not changed or final >= original:
        raise ValueError("Selected recipe produced no net tensor-byte saving. Choose a lower energy or eligible layers.")
    return {"status": "compressed", "requested_methods": methods,
            "applied_methods": [stage["method"] for stage in changed], "stages": stages,
            "original_tensor_bytes": original, "compressed_tensor_bytes": final,
            "tensor_bytes_saved": original - final,
            "caveat": "Quality and runtime must be measured on a held-out split after reloading the saved bundle."}


def direct_plan(report, config):
    """Translate a direct recipe into the final artifact comparison contract."""
    analysis = config.get("analysis", {})
    metric = analysis.get("quality_metric", "nll" if config.get("task") == "causal_lm" else "top1")
    if metric == "accuracy":
        metric = "top1"
    target_mb = analysis.get("target_size_mb")
    target = round(target_mb * 1048576) if isinstance(target_mb, (int, float)) and target_mb > 0 else report["original_tensor_bytes"] - 1
    return {"status": "feasible", "mode": "direct_recipe",
            "transformations": [{"method": method} for method in report["applied_methods"]],
            "constraints": {"quality_metric": metric,
                            "quality_direction": "lower" if metric in {"loss", "nll"} else "higher",
                            "quality_units": "native metric units",
                            "max_quality_loss": analysis.get("max_quality_loss", 0.05),
                            "target_size_bytes": target}}
