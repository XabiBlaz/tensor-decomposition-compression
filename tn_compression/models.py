"""Configurable model construction, independent of compression and task metrics."""

import copy
import re
from pathlib import Path

import torch
from torch import nn


def _replace_classifier(model, classes, path=None):
    if path is None:
        candidates = [(name, child) for name, child in model.named_modules()
                      if isinstance(child, nn.Linear) and name.split('.')[0] in {"fc", "classifier", "heads"}]
        if not candidates:
            raise ValueError("Specify classifier_path to adapt this classification output contract.")
        path, old = candidates[-1]
    else:
        old = model.get_submodule(path)
    if not isinstance(old, nn.Linear):
        raise ValueError("classifier_path must identify a Linear layer.")
    parent, _, leaf = path.rpartition('.')
    target = model.get_submodule(parent) if parent else model
    setattr(target, leaf, nn.Linear(old.in_features, classes, bias=old.bias is not None))


def load_model(spec, *, load_weights=True, factory=None):
    """Construct any supported registered model, recording its reconstruction recipe.

    Benchmark names are not an allowlist. Custom outputs/parent weight access
    still need task/module adapters; a successful load is not validation.
    """
    spec = copy.deepcopy(dict(spec))
    source = spec.get("source")
    kwargs = dict(spec.get("kwargs", {}))
    weights = spec.get("weights") if load_weights else None
    if source == "custom":
        if factory is None:
            raise ValueError("A custom model needs an explicit reconstruction factory.")
        model = factory(**kwargs)
    elif source == "torchvision":
        from torchvision.models import get_model, get_model_weights
        name = spec["name"]
        classes = kwargs.pop("num_classes", None) if spec.get("task") == "classification" else None
        if weights is not None:
            if weights == "DEFAULT":
                raise ValueError("Pin a named TorchVision weight version instead of DEFAULT.")
            weights = get_model_weights(name)[weights]
        # Detection constructors otherwise download a backbone even with weights=None.
        if not load_weights and spec.get("task") == "detection":
            kwargs["weights_backbone"] = None
        model = get_model(name, weights=weights, **kwargs)
        if classes is not None:
            _replace_classifier(model, classes, spec.get("classifier_path"))
    elif source == "smp":
        import segmentation_models_pytorch as smp
        kwargs["encoder_weights"] = weights
        model = smp.create_model(spec["name"], **kwargs)
    elif source == "transformers":
        from transformers import AutoConfig, AutoModelForCausalLM
        if "config" in spec:
            config = dict(spec["config"])
            model_type = config.pop("model_type")
            config = AutoConfig.for_model(model_type, **config)
        else:
            revision = spec.get("revision")
            if not Path(spec["name"]).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
                raise ValueError("Pin a full Hugging Face model revision for reproducibility.")
            config = AutoConfig.from_pretrained(spec["name"], revision=revision, trust_remote_code=False)
            spec["config"] = config.to_dict()
        if load_weights and weights is not None:
            model = AutoModelForCausalLM.from_pretrained(spec["name"], revision=spec.get("revision"),
                                                       config=config, trust_remote_code=False, **kwargs)
        else:
            model = AutoModelForCausalLM.from_config(config, **kwargs)
    elif source == "torch_hub":
        local = spec.get("local", False)
        repository = spec["repository"]
        if not local:
            revision = spec.get("revision", "")
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Remote Hub loading requires a pinned 40-character commit.")
            repository = f"{repository}:{revision}"
        if not load_weights:
            kwargs.update(spec.get("reconstruction_kwargs", {"pretrained": False}))
        model = torch.hub.load(repository, spec["name"], source="local" if local else "github",
                               trust_repo=True, **kwargs)
    else:
        raise ValueError(f"Unknown model source: {source!r}")
    if not isinstance(model, nn.Module):
        raise ValueError("The loader returned a wrapper, not an nn.Module; supply an adapter/factory.")
    model._tn_model_spec = spec
    return model
