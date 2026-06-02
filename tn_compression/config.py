"""Schema handling for the post-training compression API."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml

ConfigInput = Union[str, os.PathLike, Mapping[str, Any]]
SCHEMA_VERSION = 1


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Config error: duplicate YAML key {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_config(config: ConfigInput) -> Dict[str, Any]:
    """Load a config path or mapping, rejecting duplicate YAML keys for paths."""
    if isinstance(config, (str, os.PathLike)):
        with open(config, "r") as fh:
            data = yaml.load(fh, Loader=_UniqueKeyLoader)
    else:
        data = copy.deepcopy(dict(config))
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError("Config error: config must be a mapping.")
    return copy.deepcopy(dict(data))


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable hash for manifest provenance."""

    def _json_default(value):
        return repr(value)

    payload = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_config(config: ConfigInput) -> Dict[str, Any]:
    """Load and validate a schema-v1 compression config."""
    raw = load_config(config)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Config error: unsupported schema_version={raw.get('schema_version')!r}.")

    normalized = copy.deepcopy(raw)
    _validate_schema_v1(normalized)
    normalized["_config_hash"] = config_hash(normalized)
    return normalized


def build_compression_config(
    compression_config: Mapping[str, Any],
    *,
    output_dir: Union[str, os.PathLike],
    output_name: str = "compressed_model",
) -> Dict[str, Any]:
    """Build a schema-v1 config from a compact compression mapping."""
    compression = (
        compression_config.get("compression", {})
        if isinstance(compression_config.get("compression", {}), Mapping)
        else compression_config
    )
    method = copy.deepcopy(dict(compression.get("method", {}) or {}))
    method.setdefault("type", "tensor_train")
    method.setdefault("method", "SVD")
    method.setdefault("structure", "TTPWT")
    method.setdefault("rank", compression.get("rank"))
    method.setdefault("rank_cap", compression.get("rank_cap", 8))

    output_path = Path(output_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": {"kind": "live_module"},
        "runtime": copy.deepcopy(dict(compression_config.get("runtime", {}) or {})),
        "compression": {
            "mode": compression.get("mode", "default_all"),
            "default_method": method,
        },
        "output": {
            "dir": str(output_path),
            "plan_dir": str(output_path / "plans"),
            "name": output_name,
            "artifacts": [{"kind": "state_dict"}],
        },
    }


def _validate_schema_v1(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Config error: normalized config must have schema_version: 1.")

    artifact = config.get("artifact", {})
    if not isinstance(artifact, Mapping):
        raise ValueError("Config error: artifact must be a mapping.")
    artifact_kind = artifact.get("kind", "live_module")
    allowed_artifacts = {
        "live_module",
        "full_module",
    }
    if artifact_kind not in allowed_artifacts:
        raise ValueError(f"Config error: unsupported artifact.kind={artifact_kind!r}.")

    compression = config.get("compression", {})
    if not isinstance(compression, Mapping):
        raise ValueError("Config error: compression must be a mapping.")
    mode = compression.get("mode", "default_all")
    if mode not in {"default_all", "individual", "cka_groups"}:
        raise ValueError("Config error: compression.mode must be default_all, individual, or cka_groups.")

    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("Config error: output must be a mapping.")
    artifacts = output.get("artifacts", [{"kind": "full_module"}])
    if not isinstance(artifacts, list):
        raise ValueError("Config error: output.artifacts must be a list.")
    allowed_outputs = {"full_module", "state_dict"}
    for item in artifacts:
        if not isinstance(item, Mapping) or item.get("kind") not in allowed_outputs:
            raise ValueError("Config error: each output artifact must have kind full_module or state_dict.")
