"""Safe vision state-dict loading and direct spectral-energy compression.

An uploaded .pt file supplies weights, not executable model architecture. The
architecture comes from a declarative TorchVision or SMP model specification.
"""

from __future__ import annotations

import copy
import fnmatch
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from .api import is_eligible_layer, protected_module_reasons, set_submodule_by_path
from .compression_methods.rank_selection import energy_rank_from_svals
from .decompositions.common import conv_sequence, copy_weights, decomposition_weight, finish_replacement
from .models import load_model


def load_vision_checkpoint(checkpoint_path, model_spec, *, device="cpu") -> nn.Module:
    """Construct a registered vision model and strictly load tensor-only weights.

    Accepted files are a bare state_dict or a mapping with ``state_dict`` or
    ``model_state_dict``. A uniform ``module.`` prefix from DataParallel is
    removed. Serialized nn.Module objects are deliberately unsupported.
    """
    path = Path(checkpoint_path)
    if not path.is_file() or path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Provide an existing .pt or .pth state_dict file.")
    if not isinstance(model_spec, Mapping) or model_spec.get("source") not in {"torchvision", "smp"}:
        raise ValueError("Uploaded vision weights require a TorchVision or SMP architecture specification.")
    if not isinstance(model_spec.get("name"), str) or not model_spec["name"]:
        raise ValueError("The architecture specification needs a model name.")
    spec = copy.deepcopy(dict(model_spec))
    # The upload supplies all weights. The constructor must never download a
    # pretrained head, backbone, or encoder while building the architecture.
    spec["weights"] = None
    spec.pop("state_dict_path", None)
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "Could not read tensor-only weights. Upload a state_dict, not a serialized model object."
        ) from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("The checkpoint must contain a state_dict mapping, not a serialized model object.")
    wrappers = [key for key in ("state_dict", "model_state_dict") if key in loaded]
    if len(wrappers) > 1:
        raise ValueError("Checkpoint contains multiple state_dict wrappers; choose an unambiguous file.")
    state = loaded[wrappers[0]] if wrappers else loaded
    if not isinstance(state, Mapping) or not state:
        raise ValueError("The checkpoint does not contain a nonempty state_dict.")
    if any(not isinstance(name, str) or not name or not isinstance(value, torch.Tensor)
           for name, value in state.items()):
        raise ValueError("Every state_dict entry must have a nonempty string key and a tensor value.")
    if all(name.startswith("module.") for name in state):
        state = {name[len("module."):]: value for name, value in state.items()}
    model = load_model(spec, load_weights=False)
    # load_state_dict otherwise silently casts an uploaded half/bfloat16 model
    # into the constructor's float32 parameters.
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if name in state:
            tensor.data = tensor.data.to(dtype=state[name].dtype)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Uploaded weights do not match the selected architecture and class count: " + str(exc)
        ) from exc
    model._tn_model_spec = spec
    return model.to(device)


def _matches(path: str, include, exclude) -> bool:
    return (not include or any(fnmatch.fnmatchcase(path, pattern) for pattern in include)) and not any(
        fnmatch.fnmatchcase(path, pattern) for pattern in exclude)


def _factorize_linear(layer: nn.Linear, left, singular, right, rank: int) -> nn.Module:
    scale = singular[:rank].sqrt()
    first = nn.Linear(layer.in_features, rank, bias=False)
    last = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
    copy_weights(first, scale[:, None] * right[:rank])
    copy_weights(last, left[:, :rank] * scale, layer.bias)
    return finish_replacement(layer, nn.Sequential(first, last))


def _factorize_conv(layer: nn.Conv2d, left, singular, right, rank: int) -> nn.Module:
    scale = singular[:rank].sqrt()
    padding = layer.padding if layer.padding_mode == "zeros" else (0, 0)
    first = nn.Conv2d(layer.in_channels, rank, layer.kernel_size, stride=layer.stride,
                      padding=padding, dilation=layer.dilation, bias=False)
    last = nn.Conv2d(rank, layer.out_channels, 1, bias=layer.bias is not None)
    copy_weights(first, (scale[:, None] * right[:rank]).reshape_as(first.weight))
    copy_weights(last, (left[:, :rank] * scale).reshape_as(last.weight), layer.bias)
    return conv_sequence(layer, [first, last])


@torch.no_grad()
def compress_svd_energy(model: nn.Module, energy: float, *, include=None, exclude=None) -> dict:
    """Replace beneficial Conv2d/Linear layers at the requested SVD energy.

    A convolution is unfolded into an ``[out_channels, in_channels * kh * kw]``
    matrix. The smallest rank retaining at least ``energy`` of that matrix's
    squared singular values is used. This is a weight reconstruction guarantee,
    not an accuracy or latency guarantee; held-out evaluation is still needed.
    """
    if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy):
        raise ValueError("SVD energy must be a finite number in (0, 1].")
    energy = float(energy)
    if not 0 < energy <= 1:
        raise ValueError("SVD energy must be in (0, 1].")
    if not isinstance(model, nn.Module):
        raise ValueError("Expected an nn.Module.")
    include = list(include or [])
    exclude = list(exclude or [])
    if any(not isinstance(pattern, str) for pattern in [*include, *exclude]):
        raise ValueError("Layer include/exclude patterns must be strings.")

    original_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    protected = protected_module_reasons(model)
    layers = []
    transformations = list(getattr(model, "_tn_transformations", []))
    # Snapshot named modules because replacements add new modules below each path.
    for path, layer in list(model.named_modules()):
        if not path or type(layer) not in {nn.Conv2d, nn.Linear} or not _matches(path, include, exclude):
            continue
        original_params = sum(parameter.numel() for parameter in layer.parameters(recurse=False))
        entry = {"path": path, "type": type(layer).__name__, "status": "skipped",
                 "original_parameters": original_params,
                 "original_tensor_bytes": sum(parameter.numel() * parameter.element_size()
                                              for parameter in layer.parameters(recurse=False))}
        parts = path.split(".")
        if any(getattr(model.get_submodule(".".join(parts[:index])), "_tn_replacement", False)
               for index in range(1, len(parts))):
            entry["reason"] = "already_factorized_parent"
            layers.append(entry)
            continue
        normalized = path.replace(".", "/")
        eligible, reason = is_eligible_layer(layer)
        if normalized in protected:
            reason = protected[normalized]
            eligible = False
        if not eligible:
            entry["reason"] = reason
            layers.append(entry)
            continue
        weight = decomposition_weight(layer)
        if not torch.isfinite(weight).all():
            entry["reason"] = "nonfinite_weight"
            layers.append(entry)
            continue
        matrix = weight.reshape(weight.shape[0], -1)
        try:
            left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError as exc:
            entry["reason"] = f"svd_failed: {exc}"
            layers.append(entry)
            continue
        rank = energy_rank_from_svals(singular, energy)
        powers = singular.double().square()
        total_power = powers.sum()
        retained_energy = float(powers[:rank].sum() / total_power) if total_power > 0 else 1.0
        if type(layer) is nn.Linear:
            candidate_params = rank * (layer.in_features + layer.out_features)
        else:
            candidate_params = rank * (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                       + layer.out_channels)
        candidate_params += layer.bias.numel() if layer.bias is not None else 0
        entry.update(rank=rank, maximum_rank=int(singular.numel()),
                     retained_energy=retained_energy, candidate_parameters=candidate_params,
                     candidate_tensor_bytes=candidate_params * layer.weight.element_size())
        if candidate_params >= original_params:
            entry["reason"] = "non_beneficial_parameter_count"
            layers.append(entry)
            continue
        replacement = (_factorize_linear(layer, left, singular, right, rank)
                       if type(layer) is nn.Linear else _factorize_conv(layer, left, singular, right, rank))
        replacement._tn_replacement = True
        set_submodule_by_path(model, path, replacement)
        entry["status"] = "compressed"
        entry["tensor_bytes_saved"] = entry["original_tensor_bytes"] - entry["candidate_tensor_bytes"]
        transformations.append({"layer_path": path, "method": "svd_energy",
                                "configuration": {"requested_energy": energy, "rank": rank,
                                                  "retained_energy": retained_energy}})
        layers.append(entry)
    model._tn_transformations = transformations
    compressed_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    compressed_layers = sum(entry["status"] == "compressed" for entry in layers)
    return {"status": "compressed" if compressed_layers else "unchanged",
            "method": "svd_energy", "requested_energy": energy,
            "energy_basis": "squared singular values of each weight matrix; Conv2d output-channel unfolding",
            "compressed_layers": compressed_layers, "skipped_layers": len(layers) - compressed_layers,
            "original_tensor_bytes": original_bytes, "compressed_tensor_bytes": compressed_bytes,
            "tensor_bytes_saved": original_bytes - compressed_bytes, "layers": layers}


# An explicit vision name keeps the UI-facing intent discoverable; the
# underlying transform also supports ordinary Linear layers in causal LMs.
compress_vision_svd_energy = compress_svd_energy
