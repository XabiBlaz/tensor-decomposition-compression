"""Portable architecture recipes and tensor-only state for compressed models."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .decompositions.common import CircularPad2d
from .decompositions.tt import _TTConvPlain, _TTLinearCore
from .models import load_model

SCHEMA_VERSION = 1


def describe_module(module):
    """Describe replacement structure without any factorization or weight values."""
    if type(module) is nn.Sequential:
        return {"type": "Sequential", "children": {name: describe_module(child)
                                                   for name, child in module.named_children()}}
    if type(module) is nn.Conv2d:
        args = {name: getattr(module, name) for name in
                ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode")}
        args["bias"] = module.bias is not None
    elif type(module) is nn.Linear:
        args = {"in_features": module.in_features, "out_features": module.out_features,
                "bias": module.bias is not None}
    elif isinstance(module, (_TTLinearCore, _TTConvPlain)):
        args = {"core_shapes": [list(core.shape) for core in module.cores],
                "bias_shape": list(module.bias.shape) if module.bias is not None else None}
        if isinstance(module, _TTLinearCore):
            args.update(input_dims=module.input_dims, output_dims=module.output_dims)
        else:
            args.update(stride=module.stride, padding=module.padding, dilation=module.dilation)
    elif type(module) in {nn.ReflectionPad2d, nn.ReplicationPad2d, CircularPad2d}:
        args = {"padding": module.padding}
    else:
        raise ValueError(f"No reconstruction recipe for {type(module).__name__}.")
    return {"type": type(module).__name__, "args": args}


def construct_module(description):
    kind = description["type"]
    args = dict(description.get("args", {}))
    if kind == "Sequential":
        from collections import OrderedDict
        return nn.Sequential(OrderedDict((name, construct_module(child))
                                         for name, child in description["children"].items()))
    constructors = {cls.__name__: cls for cls in
                    (nn.Conv2d, nn.Linear, nn.ReflectionPad2d, nn.ReplicationPad2d,
                     CircularPad2d, _TTLinearCore, _TTConvPlain)}
    if kind not in constructors:
        raise ValueError(f"Unsupported checkpoint module type: {kind}")
    if kind in {"_TTLinearCore", "_TTConvPlain"}:
        args["cores"] = [torch.zeros(shape) for shape in args.pop("core_shapes")]
        bias_shape = args.pop("bias_shape")
        args["bias"] = torch.zeros(bias_shape) if bias_shape is not None else None
    return constructors[kind](**args)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_bundle(model, path, *, model_spec=None, metadata=None):
    """Save a reconstructible model; require a factory on reload for custom models."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Checkpoint directory is not empty: {path}")
    spec = model_spec or getattr(model, "_tn_model_spec", None)
    if spec is None:
        raise ValueError("Supply model_spec, including a custom source when using a reconstruction factory.")
    replacements = []
    for name, module in model.named_modules():
        if getattr(module, "_tn_replacement", False):
            if any(name.startswith(entry["path"] + ".") for entry in replacements):
                continue
            replacements.append({"path": name, "structure": describe_module(module)})
    manifest = {
        "schema_version": SCHEMA_VERSION, "model": spec, "replacements": replacements,
        "metadata": metadata or {}, "torch_version": str(torch.__version__),
        "training": {name: module.training for name, module in model.named_modules()},
        "requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
    }
    # Validate JSON before creating an incomplete checkpoint.
    json.dumps(manifest)
    path.mkdir(parents=True, exist_ok=True)
    weights = path / "weights.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, weights)
    manifest["weights_sha256"] = file_digest(weights)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_bundle(path, *, factory=None, device="cpu"):
    """Rebuild saved shapes and load strictly; never recompute decompositions."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version.")
    weights = path / "weights.pt"
    if file_digest(weights) != manifest["weights_sha256"]:
        raise ValueError("Checkpoint weight checksum mismatch.")
    model = load_model(manifest["model"], load_weights=False, factory=factory)
    for entry in manifest["replacements"]:
        parent, _, name = entry["path"].rpartition(".")
        target = model.get_submodule(parent) if parent else model
        if name not in target._modules:
            raise ValueError(f"Reconstruction factory has no module {entry['path']}.")
        replacement = construct_module(entry["structure"])
        replacement._tn_replacement = True
        setattr(target, name, replacement)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    # Preserve saved precision rather than silently casting every tensor into
    # the constructor's float32 parameters. Shapes are checked by strict load.
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if name in state:
            tensor.data = tensor.data.to(dtype=state[name].dtype)
    model.load_state_dict(state, strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(manifest["requires_grad"][name])
    for name, module in model.named_modules():
        module.training = manifest["training"][name]
    model.to(device)
    return model, manifest
