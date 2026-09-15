from __future__ import annotations
import copy
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch

from .artifacts import LoadedArtifact, load_artifact, resolve_device, save_outputs
from .config import normalize_config
from .compression_methods._compression_methods_utils import compress_layer

ConfigInput = Union[str, os.PathLike, Mapping[str, Any]]
_MISSING = object()


class _DictResult:
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for item in fields(self):
            if item.metadata.get("to_dict") is False:
                continue
            value = getattr(self, item.name)
            result[item.name] = asdict(value) if hasattr(value, "__dataclass_fields__") else value
        return result


@dataclass
class CompressionPlanResult(_DictResult):
    output_dir: str
    plan_path: str
    metadata_path: str
    plan: Dict[str, Any]
    manifest_path: Optional[str] = None


@dataclass
class CompressionResult(_DictResult):
    output_dir: str
    plan_path: str
    manifest_path: str
    artifacts: List[Dict[str, Any]]
    compressed_layers: int
    skipped_layers: int
    warnings: List[str] = field(default_factory=list)
    compressed_model_path: Optional[str] = None
    compressed_model_kind: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    model: Optional[torch.nn.Module] = field(default=None, repr=False, compare=False, metadata={"to_dict": False})


def output_dir(config: Mapping[str, Any]) -> Path:
    output_cfg = config.get("output", {}) if isinstance(config.get("output", {}), Mapping) else {}
    return Path(output_cfg.get("dir") or Path.cwd() / "compression_output")


def output_name(config: Mapping[str, Any], default: str = "compressed_model") -> str:
    output_cfg = config.get("output", {}) if isinstance(config.get("output", {}), Mapping) else {}
    return str(output_cfg.get("name") or default)


def plan_dir(config: Mapping[str, Any]) -> Path:
    output_cfg = config.get("output", {}) if isinstance(config.get("output", {}), Mapping) else {}
    return Path(output_cfg.get("plan_dir") or output_cfg.get("dir") or Path.cwd() / "compression_output")


def normalize_layer_name(name: str) -> str:
    return str(name).replace(".", "/")


def layer_name_variants(name: str) -> List[str]:
    normalized = normalize_layer_name(name)
    variants = [normalized, normalized.replace("/", ".")]
    if normalized.startswith("model/"):
        variants.extend([normalized[len("model/"):], normalized[len("model/"):].replace("/", ".")])
    else:
        variants.extend(["model/" + normalized, "model." + normalized.replace("/", ".")])
    seen = []
    for item in variants:
        if item not in seen:
            seen.append(item)
    return seen


def iter_named_leaf_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if name and not list(module.children()):
            yield normalize_layer_name(name), module


def get_submodule_by_path(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    current: Any = model
    for part in normalize_layer_name(layer_name).split("/"):
        if part.isnumeric():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def set_submodule_by_path(model: torch.nn.Module, layer_name: str, new_layer: torch.nn.Module) -> None:
    parts = normalize_layer_name(layer_name).split("/")
    parent: Any = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isnumeric() else getattr(parent, part)
    last = parts[-1]
    if last.isnumeric():
        parent[int(last)] = new_layer
    else:
        setattr(parent, last, new_layer)


def is_eligible_layer(module: torch.nn.Module) -> tuple[bool, str]:
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and type(module) not in {torch.nn.Conv2d, torch.nn.Linear}:
        return False, "custom_forward_requires_adapter"
    if isinstance(module, torch.nn.Conv2d):
        if module.groups != 1:
            return False, "grouped_conv_not_supported"
        if isinstance(module.padding, str):
            return False, "string_padding_requires_adapter"
        return True, "eligible"
    if isinstance(module, torch.nn.Linear):
        return True, "eligible"
    return False, "unsupported_layer_type"


def is_method_compatible(module: torch.nn.Module, policy: Optional[Mapping[str, Any]]) -> tuple[bool, str]:
    if not isinstance(policy, Mapping):
        return False, "configured_skip_or_no_policy"
    method_type = str(policy.get("type", "")).lower()
    if method_type == "tucker":
        return False, "unsupported_method_use_partial_tucker"
    if method_type in {"tensor_train", "tt"}:
        return isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)), "eligible"
    if method_type in {"cp2", "svd"}:
        return isinstance(module, torch.nn.Linear), "cp2_only_supports_linear"
    if method_type in {"partial_tucker", "cp3", "cp4"}:
        return isinstance(module, torch.nn.Conv2d), f"{method_type}_only_supports_conv2d"
    return False, f"unsupported_compression_method_{method_type or 'missing'}"


def protected_module_reasons(model):
    """Conservatively protect aliases and parents that use child weights directly."""
    owners = {}
    reasons = {}
    for name, module in model.named_modules(remove_duplicate=False):
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), []).append(name)
        if not name:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        try:
            source = inspect.getsource(type(parent).forward)
        except (OSError, TypeError):
            source = ""
        pattern = rf"\bself\s*\.\s*{re.escape(child_name)}\s*\.\s*(weight|bias)\b"
        if isinstance(parent, torch.nn.MultiheadAttention) or re.search(pattern, source):
            reasons[normalize_layer_name(name)] = "parent_accesses_weight_requires_adapter"
    for names in owners.values():
        if len(names) > 1:
            reasons.update({normalize_layer_name(name): "shared_parameter_requires_adapter" for name in names})
    return reasons


def default_method(config: Mapping[str, Any]) -> Dict[str, Any]:
    compression = config.get("compression", {}) if isinstance(config.get("compression", {}), Mapping) else {}
    method = compression.get("default_method") or compression.get("default")
    if isinstance(method, Mapping):
        return copy.deepcopy(dict(method))
    return {"type": "tensor_train", "method": "SVD", "structure": "TTPWT", "rank": 8}


def _lookup_layer_policy(mapping: Optional[Mapping[str, Any]], layer_name: str) -> Any:
    if not isinstance(mapping, Mapping):
        return _MISSING
    for variant in layer_name_variants(layer_name):
        if variant in mapping:
            return mapping[variant]
        nested = _nested_get(mapping, normalize_layer_name(variant).split("/"))
        if nested is not None:
            return nested
    return _MISSING


def _nested_get(mapping: Optional[Mapping[str, Any]], parts: Sequence[str]) -> Optional[Dict[str, Any]]:
    current: Any = mapping
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, dict) else None


def _is_skip_policy(policy: Any) -> bool:
    if policy is None:
        return True
    if isinstance(policy, str):
        return policy.lower() in {"skip", "none", "null", "false"}
    if isinstance(policy, Mapping):
        return bool(policy.get("skip", False)) or str(policy.get("type", "")).lower() in {"skip", "none"}
    return False


def _policy_for_layer(
    config: Mapping[str, Any],
    layer_name: str,
    group_policy_by_layer: Optional[Mapping[str, Optional[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    compression = config.get("compression", {}) if isinstance(config.get("compression", {}), Mapping) else {}
    skip_layers = compression.get("skip", [])
    if isinstance(skip_layers, str):
        skip_layers = [skip_layers]
    normalized_skip = {normalize_layer_name(item) for item in skip_layers if item is not None}
    if any(normalize_layer_name(variant) in normalized_skip for variant in layer_name_variants(layer_name)):
        return None

    direct_policy = _lookup_layer_policy(compression.get("layers", {}), layer_name)
    if direct_policy is not _MISSING:
        return None if _is_skip_policy(direct_policy) else copy.deepcopy(dict(direct_policy))

    mode = compression.get("mode", "default_all")
    if mode in {"cka_groups", "fisher_groups", "fisher_segments"}:
        if group_policy_by_layer is None:
            return None
        for variant in layer_name_variants(layer_name):
            normalized = normalize_layer_name(variant)
            if normalized in group_policy_by_layer:
                policy = group_policy_by_layer[normalized]
                return None if policy is None else copy.deepcopy(dict(policy))
        return None

    if mode == "individual":
        return None
    return default_method(config)


def _estimate_tensor_train_params(module: torch.nn.Module, policy: Mapping[str, Any]) -> Dict[str, Any]:
    original_params = int(sum(p.numel() for p in module.parameters(recurse=False)))
    result = {
        "original_params": original_params,
        "estimated_params": None,
        "estimated_ratio": None,
        "resolved_rank": None,
    }
    rank = policy.get("rank")
    if rank is None and policy.get("rank_cap") is not None:
        rank = policy.get("rank_cap")
    if rank is None or not isinstance(module, torch.nn.Conv2d) or policy.get("type") not in {"tt", "tensor_train"} or policy.get("structure", "TTPWT") != "TTPWT":
        return result
    if isinstance(rank, int):
        ranks = [max(1, int(rank))] * 3
    elif isinstance(rank, (list, tuple)) and len(rank) == 3:
        ranks = [max(1, int(item)) for item in rank]
    else:
        return result
    out_channels, in_channels, kernel_height, kernel_width = module.weight.shape
    rank_1 = min(ranks[0], in_channels)
    rank_2 = min(ranks[1], kernel_height * rank_1)
    rank_3 = min(ranks[2], kernel_width * rank_2)
    estimated_params = (
        in_channels * rank_1
        + rank_1 * kernel_height * rank_2
        + rank_2 * kernel_width * rank_3
        + rank_3 * out_channels
    )
    if module.bias is not None:
        estimated_params += int(module.bias.numel())
    result.update(
        {
            "estimated_params": int(estimated_params),
            "estimated_ratio": float(estimated_params) / float(original_params) if original_params else None,
            "resolved_rank": [rank_1, rank_2, rank_3],
        }
    )
    return result


def load_segments(analysis_dir: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    path = Path(analysis_dir)
    if path.is_dir():
        path = path / "segments.json"
    if not path.exists():
        raise FileNotFoundError(f"CKA group segments not found: {path}")
    with open(path, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Config error: segments.json must contain a list.")
    return data


def _group_policy(segment: Mapping[str, Any], index: int, config: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    compression = config.get("compression", {}) if isinstance(config.get("compression", {}), Mapping) else {}
    groups = compression.get("groups", compression.get("segments", {}))
    candidates = [f"group_{index}", f"segment_{index}", str(index), str(segment.get("id", "")), str(segment.get("name", ""))]
    if isinstance(groups, Mapping):
        for candidate in candidates:
            if candidate and candidate in groups:
                policy = groups[candidate]
                return None if _is_skip_policy(policy) else copy.deepcopy(dict(policy))
    return default_method(config)


def build_plan(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    analysis_dir: Optional[Union[str, os.PathLike]] = None,
    write: bool = True,
) -> CompressionPlanResult:
    compression = config.get("compression", {}) if isinstance(config.get("compression", {}), Mapping) else {}
    mode = compression.get("mode", "default_all")
    if mode not in {"default_all", "individual", "cka_groups", "fisher_groups", "fisher_segments"}:
        raise ValueError(f"Unsupported compression mode: {mode}")

    segments: List[Dict[str, Any]] = []
    group_policy_by_layer: Dict[str, Optional[Dict[str, Any]]] = {}
    if mode in {"cka_groups", "fisher_groups", "fisher_segments"}:
        resolved_analysis_dir = analysis_dir or compression.get("analysis_dir") or config.get("analysis", {}).get("output_dir")
        if not resolved_analysis_dir:
            raise ValueError("Config error: cka_groups planning requires analysis_dir or compression.analysis_dir.")
        segments = load_segments(resolved_analysis_dir)
        for index, segment in enumerate(segments):
            policy = _group_policy(segment, index, config)
            for layer in segment.get("layers", []):
                for variant in layer_name_variants(str(layer)):
                    group_policy_by_layer[normalize_layer_name(variant)] = policy

    layer_entries = []
    protected = protected_module_reasons(model)
    for layer_name, module in iter_named_leaf_modules(model):
        eligible, reason = is_eligible_layer(module)
        if layer_name in protected:
            eligible, reason = False, protected[layer_name]
        policy = _policy_for_layer(config, layer_name, group_policy_by_layer)
        forced = bool(isinstance(policy, Mapping) and policy.get("force", False))
        entry = {
            "name": layer_name,
            "display_name": layer_name.replace("/", "."),
            "type": module.__class__.__name__,
            "eligible": eligible,
            "policy": copy.deepcopy(policy) if policy is not None else None,
            "skip_reason": None,
        }
        entry.update(_estimate_tensor_train_params(module, policy if isinstance(policy, Mapping) else {}))
        if policy is None:
            entry["eligible"] = False
            entry["skip_reason"] = "configured_skip_or_no_policy"
        elif not eligible:
            entry["policy"] = None
            entry["skip_reason"] = reason
        else:
            compatible, compatibility_reason = is_method_compatible(module, policy)
            if not compatible:
                entry["eligible"] = False
                entry["policy"] = None
                entry["skip_reason"] = compatibility_reason
            elif not forced and entry.get("estimated_ratio") is not None and entry["estimated_ratio"] >= 1.0:
                entry["eligible"] = False
                entry["policy"] = None
                entry["skip_reason"] = "non_beneficial_compression"
        layer_entries.append(entry)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    covered_params = sum(entry["original_params"] for entry in layer_entries if entry["eligible"])
    plan = {"mode": mode, "layers": layer_entries, "segments": segments,
            "total_parameters": total_params, "eligible_parameters": covered_params,
            "eligible_fraction": covered_params / total_params if total_params else 0.0,
            "validation": "module_contract_only"}
    out_dir = plan_dir(config)
    plan_path = out_dir / "compression_plan.json"
    metadata_path = out_dir / "compression_metadata.json"
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(plan_path, plan)
        _write_json(
            metadata_path,
            {
                "mode": mode,
                "total_layers": len(layer_entries),
                "compressible_layers": sum(1 for item in layer_entries if item.get("eligible") and item.get("policy")),
                "skipped_layers": sum(1 for item in layer_entries if not (item.get("eligible") and item.get("policy"))),
                "default_method": default_method(config),
            },
        )
    return CompressionPlanResult(str(out_dir), str(plan_path), str(metadata_path), plan)


def apply_plan(model: torch.nn.Module, plan: Mapping[str, Any]) -> tuple[int, int, List[str]]:
    compressed_layers = 0
    skipped_layers = 0
    warnings: List[str] = []
    protected = protected_module_reasons(model)
    for entry in plan.get("layers", []):
        policy = entry.get("policy")
        if not entry.get("eligible", False) or not isinstance(policy, Mapping):
            skipped_layers += 1
            continue
        layer_name = entry["name"]
        if normalize_layer_name(layer_name) in protected:
            skipped_layers += 1
            warnings.append(f"Skipped {layer_name}: {protected[normalize_layer_name(layer_name)]}")
            continue
        try:
            layer = get_submodule_by_path(model, layer_name)
        except Exception as exc:
            skipped_layers += 1
            warnings.append(f"Skipped {layer_name}: could not resolve layer ({exc})")
            continue
        if isinstance(layer, torch.nn.Conv2d) and layer.groups != 1:
            skipped_layers += 1
            warnings.append(f"Skipped {layer_name}: grouped Conv2d is unsupported")
            continue
        try:
            new_layer = compress_layer(layer, copy.deepcopy(dict(policy)))
        except Exception as exc:
            skipped_layers += 1
            warnings.append(f"Skipped {layer_name}: compression failed ({exc})")
            continue
        if new_layer is None or new_layer is layer:
            skipped_layers += 1
            warnings.append(f"Skipped {layer_name}: compression returned no replacement")
            continue
        if not policy.get("force", False) and sum(p.numel() for p in new_layer.parameters()) >= sum(p.numel() for p in layer.parameters()):
            skipped_layers += 1
            warnings.append(f"Skipped {layer_name}: replacement does not reduce parameter count")
            continue
        if not all(torch.isfinite(parameter).all() for parameter in new_layer.parameters()):
            skipped_layers += 1
            warnings.append(f"Skipped {layer_name}: non-finite decomposition factors")
            continue
        set_submodule_by_path(model, layer_name, new_layer)
        compressed_layers += 1
    return compressed_layers, skipped_layers, warnings


def write_manifest(
    config: Mapping[str, Any],
    *,
    operation: str,
    loaded: Optional[LoadedArtifact],
    backend: str,
    device: torch.device,
    plan: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    out_dir = output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{operation}_manifest.json"
    try:
        package_version = version("tn-compression")
    except PackageNotFoundError:
        package_version = "0+local"
    manifest = {
        "operation": operation,
        "backend": backend,
        "device": str(device),
        "package_version": package_version,
        "config_hash": config.get("_config_hash"),
        "input_artifact": {
            "kind": loaded.input_kind if loaded else None,
            "source_path": loaded.source_path if loaded else None,
        },
        "plan_summary": _plan_summary(plan),
        "artifacts": artifacts or [],
        "warnings": warnings or [],
    }
    if extra:
        manifest.update(dict(extra))
    _write_json(manifest_path, manifest)
    return str(manifest_path)


def _plan_summary(plan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not plan:
        return {}
    layers = list(plan.get("layers", []))
    return {
        "mode": plan.get("mode"),
        "total_layers": len(layers),
        "compressible_layers": sum(1 for item in layers if item.get("eligible") and item.get("policy")),
        "skipped_layers": sum(1 for item in layers if not (item.get("eligible") and item.get("policy"))),
    }


def _write_json(path: Union[str, os.PathLike], data: Any) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=repr)


def _compressed_artifact(artifacts: Sequence[Mapping[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    for artifact in artifacts:
        if artifact.get("status") == "written" and artifact.get("kind") in {"full_module", "state_dict"}:
            return str(artifact.get("path")), str(artifact.get("kind"))
    return None, None


def _compress_loaded_artifact(
    loaded: LoadedArtifact,
    cfg: Mapping[str, Any],
    *,
    device=None,
    return_model: bool = False,
) -> CompressionResult:
    device_obj = resolve_device(device, cfg)
    loaded.model.to(device_obj)
    plan_result = build_plan(loaded.model, cfg)
    compressed_layers, skipped_layers, compression_warnings = apply_plan(loaded.model, plan_result.plan)
    artifacts = save_outputs(
        loaded.model,
        cfg,
        output_name=output_name(cfg),
        device=device_obj,
    )
    warnings = list(compression_warnings)
    manifest_path = write_manifest(
        cfg,
        operation="compress",
        loaded=loaded,
        backend="torch",
        device=device_obj,
        plan=plan_result.plan,
        artifacts=artifacts,
        warnings=warnings,
        extra={"plan_path": plan_result.plan_path},
    )
    compressed_model_path, compressed_model_kind = _compressed_artifact(artifacts)
    summary = {
        "status": "compressed" if compressed_layers else "no_op",
        "compressed_layers": compressed_layers,
        "skipped_layers": skipped_layers,
        "warnings": warnings,
    }
    return CompressionResult(
        output_dir=str(plan_result.output_dir),
        plan_path=plan_result.plan_path,
        manifest_path=manifest_path,
        artifacts=artifacts,
        compressed_layers=compressed_layers,
        skipped_layers=skipped_layers,
        warnings=warnings,
        compressed_model_path=compressed_model_path,
        compressed_model_kind=compressed_model_kind,
        summary=summary,
        model=loaded.model if return_model else None,
    )


def compress_model(
    model_or_artifact: Union[str, torch.nn.Module],
    config: ConfigInput,
    *,
    inplace: bool = False,
    device=None,
    return_model: bool = False,
) -> CompressionResult:
    cfg = normalize_config(config)
    loaded = load_artifact(model_or_artifact, inplace=inplace)
    return _compress_loaded_artifact(loaded, cfg, device=device, return_model=return_model)


def generate_compression_plan(
    model_or_artifact: Union[str, torch.nn.Module],
    config: ConfigInput,
    *,
    inplace: bool = False,
    device=None,
) -> CompressionPlanResult:
    cfg = normalize_config(config)
    loaded = load_artifact(model_or_artifact, inplace=inplace)
    device_obj = resolve_device(device, cfg)
    loaded.model.to(device_obj)
    plan_result = build_plan(loaded.model, cfg)
    manifest_path = write_manifest(
        cfg,
        operation="plan",
        loaded=loaded,
        backend="torch",
        device=device_obj,
        plan=plan_result.plan,
        warnings=[],
        extra={"plan_path": plan_result.plan_path},
    )
    plan_result.manifest_path = manifest_path
    return plan_result


def apply_compression_plan(model: torch.nn.Module, plan: Mapping[str, Any]) -> tuple[int, int, List[str]]:
    return apply_plan(model, plan)
