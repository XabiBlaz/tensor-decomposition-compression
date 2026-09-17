"""Deterministic fingerprints for calibration-dependent analysis."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Optional, Sequence

import torch

from .schema import SCHEMA_VERSION


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _hash_value(digest, value):
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        flat = value.detach().contiguous().reshape(-1)
        for start in range(0, flat.numel(), 1_048_576):
            chunk = flat[start:start + 1_048_576].cpu().contiguous()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list" if isinstance(value, list) else b"tuple")
        digest.update(str(len(value)).encode())
        for item in value:
            _hash_value(digest, item)
        return
    digest.update(type(value).__name__.encode())
    digest.update(_canonical_json(value).encode())


def calibration_content_fingerprint(batches: Sequence[Any]) -> str:
    """Hash the exact in-memory calibration batch values and structure."""
    digest = hashlib.sha256()
    _hash_value(digest, batches)
    return digest.hexdigest()


def analysis_context(model_fingerprint: str, config: Mapping[str, Any], *,
                     calibration_batches: Optional[Sequence[Any]] = None,
                     calibration_ids: Optional[Sequence[str]] = None) -> Mapping[str, Any]:
    try:
        package_version = version("tn-compression")
    except PackageNotFoundError:
        package_version = "0+local"
    analysis = dict(config.get("analysis", {}))
    seed = analysis.get("seed", config.get("seed", 0))
    context = {
        "model_fingerprint": model_fingerprint,
        "task": config.get("task", "classification"),
        "analysis": analysis,
        "calibration_identifiers": list(calibration_ids or []),
        "calibration_content_sha256": (
            calibration_content_fingerprint(calibration_batches)
            if calibration_batches is not None else None),
        "preprocessing": config.get("preprocessing", config.get("data", {})),
        "tokenizer": config.get("tokenizer"),
        "seed": seed,
        "candidate_grid": analysis.get("candidate_grid", {}),
        "analyzer_schema_version": SCHEMA_VERSION,
        "package_version": package_version,
    }
    fingerprint = hashlib.sha256(_canonical_json(context).encode()).hexdigest()
    return {"fingerprint": fingerprint, "evidence": context}
