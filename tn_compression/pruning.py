"""Physical channel removal for verified Transformers gated-MLP families."""

import copy

import torch
from torch import nn

from .api import protected_module_reasons


def inspect_gated_mlps(model):
    """Return verified groups and structured failures for supported architectures."""
    try:
        from transformers.models.llama.modeling_llama import LlamaMLP
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    except ImportError:
        return {}, []
    protected = protected_module_reasons(model)
    groups, failures = {}, []
    for path, module in model.named_modules():
        if type(module) not in {LlamaMLP, Qwen2MLP}:
            continue
        gate = getattr(module, "gate_proj", None)
        up = getattr(module, "up_proj", None)
        down = getattr(module, "down_proj", None)
        reason = None
        if any(type(layer) is not nn.Linear for layer in (gate, up, down)):
            reason = "surgery_requires_three_ordinary_linear_projections"
        elif gate.out_features != up.out_features or up.out_features != down.in_features:
            reason = "incompatible_intermediate_dimensions"
        elif gate.in_features != up.in_features or up.in_features != down.out_features:
            reason = "incompatible_residual_dimensions"
        else:
            projection_reasons = {
                protected.get(f"{path}.{name}".replace(".", "/"))
                for name in ("gate_proj", "up_proj", "down_proj")
            } - {None}
            if projection_reasons:
                reason = "projection_requires_adapter:" + ",".join(sorted(projection_reasons))
        if reason:
            failures.append({
                "path": path,
                "module_type": type(module).__name__,
                "reason": reason,
                "protected": reason.startswith("projection_requires_adapter:"),
            })
        else:
            groups[path] = module
    return groups, failures


def gated_mlps(model):
    """Scope surgery to known forward contracts, not arbitrary matching names."""
    groups, failures = inspect_gated_mlps(model)
    if failures:
        failure = failures[0]
        raise ValueError(f"{failure['path']}: {failure['reason']}")
    if not groups:
        raise ValueError("No supported Llama/Qwen2 gated MLPs; supply a verified dependency handler.")
    return groups


@torch.no_grad()
def select_channels(mlp, width, *, method="weight", activations=None, seed=0):
    size = mlp.down_proj.in_features
    if isinstance(width, bool) or not isinstance(width, int) or not 0 < width <= size:
        raise ValueError("Retained width must be a positive integer within the intermediate size.")
    if method == "random":
        indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed))[:width]
    else:
        down_norm = mlp.down_proj.weight.detach().double().cpu().norm(dim=0)
        if method == "activation":
            if activations is None or activations.ndim != 2 or activations.shape[1] != size or not len(activations):
                raise ValueError("Activation ranking needs nonempty inputs to the down projection.")
            # Squared score is the single-channel mean output-removal error.
            # Group removal includes cross terms and needs full-model validation.
            scores = activations.double().cpu().square().mean(0).sqrt() * down_norm
        elif method == "weight":
            scores = mlp.up_proj.weight.detach().double().cpu().norm(dim=1) * down_norm
        else:
            raise ValueError(f"Unknown channel selection method: {method}")
        indices = torch.argsort(scores, descending=True, stable=True)[:width]
    return indices.sort().values.tolist()


@torch.no_grad()
def slice_linear(layer, indices, axis):
    indices = torch.tensor(indices, device=layer.weight.device, dtype=torch.long)
    weight = layer.weight.index_select(axis, indices)
    result = nn.Linear(weight.shape[1], weight.shape[0], bias=layer.bias is not None,
                       device=weight.device, dtype=weight.dtype)
    result.weight.copy_(weight)
    result.weight.requires_grad_(layer.weight.requires_grad)
    if layer.bias is not None:
        result.bias.copy_(layer.bias.index_select(0, indices) if axis == 0 else layer.bias)
        result.bias.requires_grad_(layer.bias.requires_grad)
    result.train(layer.training)
    result._tn_replacement = True
    return result


def prune_gated_mlps(model, selections):
    """Apply one uniform width across every supported MLP, validating first."""
    groups = gated_mlps(model)
    if set(groups) != set(selections):
        raise ValueError("Uniform-width export requires a selection for every supported MLP.")
    widths = {len(indices) for indices in selections.values()}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("All MLPs must retain the same nonzero width.")
    width = next(iter(widths))
    for path, indices in selections.items():
        if len(set(indices)) != width or any(isinstance(i, bool) or not isinstance(i, int)
                or not 0 <= i < groups[path].down_proj.in_features for i in indices):
            raise ValueError(f"{path}: invalid or repeated retained indices.")
    # Build replacements before changing the model so validation/construction
    # failures cannot leave half of the network pruned.
    replacements = {path: {name: slice_linear(getattr(groups[path], name), indices, axis)
                    for name, axis in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1))}
                    for path, indices in selections.items()}
    for path, projections in replacements.items():
        for name, layer in projections.items():
            setattr(groups[path], name, layer)
        if hasattr(groups[path], "intermediate_size"):
            groups[path].intermediate_size = width
    model.config.intermediate_size = width
    if hasattr(model, "_tn_model_spec"):
        model._tn_model_spec = copy.deepcopy(model._tn_model_spec)
        model._tn_model_spec["config"] = model.config.to_dict()
    transformation = {"method": "gated_mlp_pruning", "width": width,
                      "retained_indices": copy.deepcopy(selections)}
    model._tn_transformations = [*getattr(model, "_tn_transformations", []), transformation]
    return transformation
