"""Artifact loading and saving for compression APIs."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch


@dataclass
class LoadedArtifact:
    model: torch.nn.Module
    input_kind: str
    source_path: Optional[str] = None


def resolve_device(device: Optional[Union[str, torch.device]], config: Mapping[str, Any]) -> torch.device:
    if device is None:
        runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
        device = config.get("device") or runtime.get("device")
    if device:
        text = str(device).lower()
        if text == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if text.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{text}' was requested, but torch.cuda.is_available() is False.")
        if text not in {"cpu"} and not text.startswith("cuda"):
            raise RuntimeError(f"Unsupported device '{text}'. Use cpu, auto, cuda, or cuda:0.")
        return torch.device(text)
    return torch.device("cpu")


def load_artifact(
    model_or_artifact: Union[str, os.PathLike, torch.nn.Module],
    *,
    inplace: bool = False,
) -> LoadedArtifact:
    """Resolve a model input into a PyTorch module."""
    if isinstance(model_or_artifact, torch.nn.Module):
        model = model_or_artifact if inplace else copy.deepcopy(model_or_artifact)
        return LoadedArtifact(model=model, input_kind="live_module")

    path = Path(model_or_artifact)
    suffix = path.suffix.lower()
    if suffix not in {".pt", ".pth", ".ckpt"}:
        raise RuntimeError(f"Artifact error: unsupported artifact extension {suffix!r}.")
    return _load_full_module(path, input_kind="full_module")


def _load_full_module(path: Path, *, input_kind: str) -> LoadedArtifact:
    obj = _torch_load(path)
    if isinstance(obj, torch.nn.Module):
        return LoadedArtifact(model=obj, input_kind=input_kind, source_path=str(path))
    raise RuntimeError("Artifact error: PyTorch artifact must contain a full torch.nn.Module.")


def _torch_load(path: Path):
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Artifact error: failed to load PyTorch artifact {path}: {exc}") from exc


def save_outputs(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    output_name: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Save configured output artifacts and return artifact metadata."""
    output_cfg = config.get("output", {}) if isinstance(config.get("output", {}), Mapping) else {}
    output_dir = Path(output_cfg.get("dir") or Path.cwd() / "compression_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_cfg.get("artifacts") or [{"kind": "full_module"}]
    results: List[Dict[str, Any]] = []

    model_to_save = model.to(device)
    for item in artifacts:
        kind = item.get("kind")
        suffix = item.get("suffix")
        if kind == "full_module":
            path = output_dir / f"{output_name}{suffix or '.pt'}"
            torch.save(model_to_save, path)
            results.append({"kind": kind, "path": str(path), "status": "written"})
        elif kind == "state_dict":
            path = output_dir / f"{output_name}{suffix or '.pth'}"
            torch.save(model_to_save.state_dict(), path)
            results.append({"kind": kind, "path": str(path), "status": "written"})
        else:
            raise ValueError(f"Config error: unsupported output artifact kind {kind!r}.")
    return results
