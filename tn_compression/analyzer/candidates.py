"""Structural discovery and bounded method-specific candidate generation."""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from ..api import is_eligible_layer, protected_module_reasons
from .schema import CandidateResult, capability


DEFAULT_GRID = {
    "linear": {
        "methods": ["svd", "weighted_svd"],
        "rank_ratios": [0.25, 0.5],
    },
    "conv2d": {
        "methods": ["partial_tucker", "tensor_train", "cp3", "cp4"],
        "rank_ratios": [0.25, 0.5],
    },
    "gated_mlp": {
        "methods": ["gated_mlp_pruning"],
        "width_ratios": [0.5, 0.75],
        "selection": "activation",
    },
    "quantization": {
        "enabled": True,
        "bits": [4, 8],
        "group_sizes": [128],
    },
}


def model_fingerprint(model: nn.Module, identity: Optional[Mapping[str, Any]] = None,
                      *, chunk_elements: int = 1_048_576) -> str:
    """Hash model identity and tensor values with bounded temporary storage."""
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive.")
    digest = hashlib.sha256()
    digest.update(json.dumps(identity or {}, sort_keys=True, default=str).encode())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        flat = tensor.detach().contiguous().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            values = flat[start:start + chunk_elements].cpu().contiguous()
            digest.update(values.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_bytes(module: nn.Module, *, recurse: bool = True) -> int:
    tensors = list(module.parameters(recurse=recurse)) + list(module.buffers(recurse=recurse))
    return sum(tensor.numel() * tensor.element_size()
               for tensor in {id(tensor): tensor for tensor in tensors}.values())


def serialized_state_bytes(module: nn.Module) -> int:
    """Measure a standalone PyTorch state artifact, including container overhead."""
    buffer = io.BytesIO()
    torch.save({name: value.detach().cpu() for name, value in module.state_dict().items()}, buffer)
    return buffer.tell()


def _matches(path: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    included = not includes or any(fnmatch.fnmatchcase(path, pattern) for pattern in includes)
    return included and not any(fnmatch.fnmatchcase(path, pattern) for pattern in excludes)


def _method_options(section: Mapping[str, Any], method: str) -> Dict[str, Any]:
    options = dict(section)
    options.pop("methods", None)
    method_value = section.get(method)
    if isinstance(method_value, Mapping):
        options.update(method_value)
    return options


def _bounded_unique(values: Iterable[Any], limit: int) -> list[Any]:
    result = []
    for value in values:
        normalized = tuple(value) if isinstance(value, list) else value
        if normalized not in result:
            result.append(normalized)
        if len(result) >= limit:
            break
    return [list(value) if isinstance(value, tuple) else value for value in result]


def _linear_ranks(layer: nn.Linear, options: Mapping[str, Any], limit: int) -> list[int]:
    maximum = min(layer.in_features, layer.out_features)
    explicit = options.get("ranks", [])
    ratios = options.get("rank_ratios", []) if not explicit else []
    values = [int(rank) for rank in explicit]
    values.extend(max(1, math.floor(maximum * float(ratio))) for ratio in ratios)
    return _bounded_unique((rank for rank in values if 0 < rank <= maximum), limit)


def _conv_ranks(layer: nn.Conv2d, method: str, options: Mapping[str, Any], limit: int) -> list[Any]:
    explicit = options.get("ranks", [])
    ratios = options.get("rank_ratios", []) if not explicit else []
    if method == "partial_tucker":
        values = list(explicit)
        values.extend([max(1, math.floor(layer.out_channels * float(ratio))),
                       max(1, math.floor(layer.in_channels * float(ratio)))] for ratio in ratios)
        legal = (value for value in values if isinstance(value, (list, tuple)) and len(value) == 2
                 and all(isinstance(rank, int) and rank > 0 for rank in value))
    else:
        values = [int(rank) for rank in explicit]
        values.extend(max(1, math.floor(min(layer.in_channels, layer.out_channels) * float(ratio)))
                      for ratio in ratios)
        legal = (rank for rank in values if isinstance(rank, int) and rank > 0)
    return _bounded_unique(legal, limit)


def _linear_parameters(layer: nn.Linear, rank: int) -> int:
    return rank * (layer.in_features + layer.out_features) + (layer.out_features if layer.bias is not None else 0)


def _conv_parameters(layer: nn.Conv2d, method: str, rank: Any) -> int:
    output, inputs, height, width = layer.weight.shape
    bias = output if layer.bias is not None else 0
    if method == "partial_tucker":
        output_rank, input_rank = min(rank[0], output), min(rank[1], inputs)
        return inputs * input_rank + input_rank * output_rank * height * width + output_rank * output + bias
    if method == "cp3":
        return inputs * rank + rank * height * width + rank * output + bias
    if method == "cp4":
        return inputs * rank + rank * height + rank * width + rank * output + bias
    if method == "tensor_train":
        rank_1 = min(rank, inputs)
        rank_2 = min(rank, height * rank_1)
        rank_3 = min(rank, width * rank_2)
        return inputs * rank_1 + rank_1 * height * rank_2 + rank_2 * width * rank_3 + rank_3 * output + bias
    raise ValueError(f"Cannot estimate {method} parameters.")


def _checkpoint_capability(method: str) -> Dict[str, Any]:
    if method in {"svd", "weighted_svd", "partial_tucker", "cp3", "cp4", "tensor_train"}:
        return capability(True, "tensor-only bundle has a reconstruction recipe", verified=True)
    if method == "gated_mlp_pruning":
        return capability(True, "bundle records sliced projections and retained indices", verified=True)
    if method == "round_to_nearest":
        return capability(True, "reference remains an ordinary dense Linear", verified=True)
    return capability(False, "no reconstruction recipe")


def backend_capability(method: str, backend: str) -> Dict[str, Any]:
    backend = backend.lower()
    if backend == "pytorch":
        return capability(True, "implemented with PyTorch modules", verified=True)
    if backend == "onnx" and method in {"svd", "weighted_svd", "partial_tucker", "cp3", "cp4"}:
        return capability(True, "uses standard exportable operators; artifact parity still required")
    if backend == "transformers" and method in {"svd", "weighted_svd", "gated_mlp_pruning"}:
        return capability(True, "runs through the Transformers PyTorch model; native export depends on shape support")
    if backend == "vllm":
        return capability(False, "the selected representation has no verified vLLM loader")
    return capability(False, f"{method} is not supported on requested backend {backend}")


def _candidate(model_hash: str, path: str, module: nn.Module, method: str,
               configuration: Dict[str, Any], candidate_parameters: Optional[int],
               requested_backend: str, *, eligible: bool = True, protected: bool = False,
               reason: Optional[str] = None, original_parameters: Optional[int] = None,
               original_bytes: Optional[int] = None) -> CandidateResult:
    original_parameters = original_parameters if original_parameters is not None else sum(
        parameter.numel() for parameter in module.parameters(recurse=False))
    original_bytes = original_bytes if original_bytes is not None else tensor_bytes(module, recurse=False)
    element_size = module.weight.element_size() if hasattr(module, "weight") else 4
    estimated_bytes = candidate_parameters * element_size if candidate_parameters is not None else None
    backend = backend_capability(method, requested_backend)
    result = CandidateResult(
        model_fingerprint=model_hash,
        layer_path=path,
        module_type=type(module).__name__,
        method=method,
        configuration=configuration,
        structurally_eligible=eligible,
        rejection_reason=reason,
        protected=protected,
        original_parameters=original_parameters,
        candidate_parameters=candidate_parameters,
        original_artifact_bytes=original_bytes,
        estimated_artifact_bytes=estimated_bytes,
        checkpoint_reconstruction=_checkpoint_capability(method),
        backend_support=backend,
        status="structural",
    )
    if not eligible:
        result.decision, result.decision_reason = "rejected", reason
    elif not backend["supported"]:
        result.decision, result.decision_reason = "rejected", backend["reason"]
    elif result.estimated_bytes_saved is not None and result.estimated_bytes_saved <= 0:
        result.decision, result.decision_reason = "rejected", "non_beneficial_parameter_count"
    return result


def _unsupported(model_hash: str, path: str, module: nn.Module, reason: str,
                 requested_backend: str, *, protected: bool = False) -> CandidateResult:
    return _candidate(model_hash, path, module, "unsupported", {}, None, requested_backend,
                      eligible=False, protected=protected, reason=reason)


def _gated_mlp_groups(model: nn.Module) -> Dict[str, nn.Module]:
    try:
        from ..pruning import gated_mlps
        return gated_mlps(model)
    except (ImportError, ValueError):
        return {}


def generate_candidates(model: nn.Module, analysis: Optional[Mapping[str, Any]] = None,
                        *, requested_backend: str = "pytorch",
                        fingerprint: Optional[str] = None) -> list[CandidateResult]:
    """Generate bounded legal candidates without changing or executing the model."""
    analysis = dict(analysis or {})
    grid = {key: dict(value) for key, value in DEFAULT_GRID.items()}
    for key, value in analysis.get("candidate_grid", {}).items():
        if isinstance(value, Mapping):
            grid.setdefault(key, {}).update(value)
    limit = int(analysis.get("max_candidates_per_layer", 12))
    method_limit = int(analysis.get("max_candidates_per_method", limit))
    if limit < 1 or method_limit < 1:
        raise ValueError("Candidate limits must be positive.")
    includes = list(analysis.get("include", []))
    excludes = list(analysis.get("exclude", []))
    model_hash = fingerprint or model_fingerprint(model, analysis.get("model_identity"))
    protected = protected_module_reasons(model)
    candidates: list[CandidateResult] = []

    for path, module in model.named_modules():
        if not path or list(module.children()) or not _matches(path, includes, excludes):
            continue
        normalized = path.replace(".", "/")
        layer_eligible, layer_reason = is_eligible_layer(module)
        protected_reason = protected.get(normalized)
        if protected_reason or not layer_eligible:
            candidates.append(_unsupported(model_hash, path, module, protected_reason or layer_reason,
                                           requested_backend, protected=bool(protected_reason)))
            continue
        layer_candidates = []
        if type(module) is nn.Linear:
            section = grid["linear"]
            for method in section.get("methods", []):
                options = _method_options(section, method)
                for rank in _linear_ranks(module, options, method_limit):
                    layer_candidates.append(_candidate(
                        model_hash, path, module, method, {"rank": rank},
                        _linear_parameters(module, rank), requested_backend))
            quantization = grid.get("quantization", {})
            if quantization.get("enabled", True):
                quantized = 0
                for bits in quantization.get("bits", []):
                    for group_size in quantization.get("group_sizes", []):
                        if quantized >= method_limit:
                            break
                        value = _candidate(model_hash, path, module, "round_to_nearest",
                                           {"bits": int(bits), "group_size": int(group_size)},
                                           sum(parameter.numel() for parameter in module.parameters(recurse=False)),
                                           requested_backend)
                        value.decision = "rejected"
                        value.decision_reason = "reference_quantization_is_dense_and_has_no_storage_saving"
                        layer_candidates.append(value)
                        quantized += 1
                    if quantized >= method_limit:
                        break
        elif type(module) is nn.Conv2d:
            section = grid["conv2d"]
            for method in section.get("methods", []):
                options = _method_options(section, method)
                for rank in _conv_ranks(module, method, options, method_limit):
                    config = {"rank": rank}
                    if method == "tensor_train":
                        config["structure"] = "TTPWT"
                    layer_candidates.append(_candidate(
                        model_hash, path, module, method, config,
                        _conv_parameters(module, method, rank), requested_backend))
        candidates.extend(layer_candidates[:limit])

    section = grid.get("gated_mlp", {})
    if section.get("methods") and (not includes or any("mlp" in pattern for pattern in includes)):
        for path, module in _gated_mlp_groups(model).items():
            if not _matches(path, includes, excludes):
                continue
            gate, up, down = module.gate_proj, module.up_proj, module.down_proj
            original_parameters = sum(parameter.numel() for layer in (gate, up, down)
                                      for parameter in layer.parameters(recurse=False))
            original_bytes = sum(tensor_bytes(layer, recurse=False) for layer in (gate, up, down))
            widths = list(section.get("widths", []))
            if not widths:
                widths = [max(1, math.floor(down.in_features * float(ratio)))
                          for ratio in section.get("width_ratios", [])]
            layer_candidates = []
            for width in _bounded_unique((int(value) for value in widths
                                          if 0 < int(value) <= down.in_features), method_limit):
                parameters = (width * gate.in_features + (width if gate.bias is not None else 0)
                              + width * up.in_features + (width if up.bias is not None else 0)
                              + down.out_features * width + (down.out_features if down.bias is not None else 0))
                layer_candidates.append(_candidate(
                    model_hash, path, module, "gated_mlp_pruning",
                    {"width": width, "selection": section.get("selection", "activation")},
                    parameters, requested_backend, original_parameters=original_parameters,
                    original_bytes=original_bytes))
            candidates.extend(layer_candidates[:limit])
    return candidates
