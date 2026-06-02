from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn


InputShape = Tuple[int, ...]


@dataclass(frozen=True)
class ModelInfo:
    name: str
    family: str
    task: str
    default_input_shape: InputShape
    builder: Callable[..., nn.Module]


def _normalize_weights(weights: str) -> str:
    value = weights.lower().strip()
    aliases = {
        "none": "none",
        "random": "none",
        "scratch": "none",
        "imagenet": "imagenet",
        "default": "imagenet",
        "pretrained": "imagenet",
    }
    if value not in aliases:
        raise RuntimeError("Unsupported --weights value. Use 'none' or 'imagenet'.")
    return aliases[value]


def build_torchvision_resnet18(*, weights: str = "none", num_classes: int = 1000) -> nn.Module:
    try:
        from torchvision import models
    except Exception as exc:
        raise RuntimeError("Model 'torchvision_resnet18' requires torchvision to be installed.") from exc

    selected = _normalize_weights(weights)
    if selected == "imagenet":
        try:
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            return models.resnet18(pretrained=True)
    try:
        return models.resnet18(weights=None, num_classes=num_classes)
    except TypeError:
        return models.resnet18(pretrained=False, num_classes=num_classes)


def build_torchvision_vgg16(*, weights: str = "none", num_classes: int = 1000) -> nn.Module:
    try:
        from torchvision import models
    except Exception as exc:
        raise RuntimeError("Model 'torchvision_vgg16' requires torchvision to be installed.") from exc

    selected = _normalize_weights(weights)
    if selected == "imagenet":
        try:
            return models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        except AttributeError:
            return models.vgg16(pretrained=True)
    try:
        return models.vgg16(weights=None, num_classes=num_classes)
    except TypeError:
        return models.vgg16(pretrained=False, num_classes=num_classes)


def load_full_module_artifact(path: str | Path) -> nn.Module:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {artifact_path}")
    try:
        try:
            obj = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(artifact_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load PyTorch artifact {artifact_path}: {exc}") from exc
    if not isinstance(obj, nn.Module):
        raise RuntimeError(
            "Artifact mode requires a full serialized torch.nn.Module, not a state dict or checkpoint mapping."
        )
    return obj


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "torchvision_resnet18": ModelInfo(
        name="torchvision_resnet18",
        family="Torchvision ResNet-18",
        task="image classification logits; weights can be random or ImageNet pretrained",
        default_input_shape=(1, 3, 224, 224),
        builder=build_torchvision_resnet18,
    ),
    "torchvision_vgg16": ModelInfo(
        name="torchvision_vgg16",
        family="Torchvision VGG-16",
        task="image classification logits; weights can be random or ImageNet pretrained",
        default_input_shape=(1, 3, 224, 224),
        builder=build_torchvision_vgg16,
    ),
}


ARTIFACT_INFO = ModelInfo(
    name="artifact",
    family="User-provided full torch.nn.Module artifact",
    task="task depends on the serialized model",
    default_input_shape=(1, 3, 224, 224),
    builder=lambda **_: nn.Identity(),
)


def available_model_names() -> Tuple[str, ...]:
    return tuple(sorted([*MODEL_REGISTRY.keys(), "artifact"]))


def get_model_info(name: str) -> ModelInfo:
    if name == "artifact":
        return ARTIFACT_INFO
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(available_model_names())}")
    return MODEL_REGISTRY[name]


def build_model(
    name: str,
    *,
    artifact_path: Optional[str] = None,
    weights: str = "none",
    num_classes: int = 1000,
) -> nn.Module:
    if name == "artifact":
        if not artifact_path:
            raise RuntimeError("--model artifact requires --artifact-path.")
        return load_full_module_artifact(artifact_path)
    info = get_model_info(name)
    return info.builder(weights=weights, num_classes=num_classes)


def synthetic_input(input_shape: InputShape, *, device: torch.device) -> torch.Tensor:
    return torch.randn(*input_shape, device=device)
