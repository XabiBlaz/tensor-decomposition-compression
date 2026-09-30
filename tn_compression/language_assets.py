"""Preflight pinned causal-LM assets before starting an expensive workflow."""

import os
import re
from pathlib import Path


class LanguageAssetError(ValueError):
    """An input model, tokenizer, or dataset cannot be used as requested."""


_PINNED_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MODEL_FILES = (
    "*.json", "*.model", "*.txt", "tokenizer*", "vocab*", "merges.txt",
    "special_tokens_map.json",
)


def _local_or_pinned(spec, label):
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not spec["name"]:
        raise LanguageAssetError(f"{label} needs a repository ID or local directory.")
    name = spec["name"]
    local = Path(name).is_dir()
    if not local and not _PINNED_REVISION.fullmatch(spec.get("revision") or ""):
        raise LanguageAssetError(f"Pin {label} {name!r} to a full 40-character commit revision.")
    return local


def _weight_files_present(root):
    """Verify a cached model has its full safetensors or PyTorch shard set."""
    root = Path(root)
    import json

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / index_name
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            except (OSError, KeyError, ValueError, TypeError) as error:
                raise LanguageAssetError(f"Invalid model weight index: {index_name}") from error
            if shards and all((root / shard).is_file() for shard in shards):
                return True
    return (root / "model.safetensors").is_file() or (root / "pytorch_model.bin").is_file()


def _offline_enabled(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def preflight_language_assets(config, *, allow_download=False, check_data=True, check_model=True):
    """Resolve model, tokenizer and data while stating any missing cache clearly.

    The caller must choose online/offline mode *before starting the Python
    process*: huggingface_hub and datasets read their offline flags at import.
    This function never turns an offline process online or silently substitutes
    random model weights for a missing checkpoint.
    """
    if not isinstance(config, dict) or config.get("task") != "causal_lm":
        raise LanguageAssetError("Language preflight requires task: causal_lm.")
    model = config.get("model") or {}
    if not isinstance(model, dict):
        raise LanguageAssetError("Language model specification must be an object.")
    if model.get("source") != "transformers":
        raise LanguageAssetError("Language preflight requires a Transformers causal LM.")
    local_model = _local_or_pinned(model, "model")
    if check_model and model.get("weights") is None and not model.get("config"):
        raise LanguageAssetError("Select pretrained model weights; an uninitialized causal LM is not a compression input.")
    tokenizer = config.get("tokenizer", model)
    local_tokenizer = _local_or_pinned(tokenizer, "tokenizer")
    if allow_download and (_offline_enabled("HF_HUB_OFFLINE") or _offline_enabled("HF_DATASETS_OFFLINE")):
        raise LanguageAssetError("Downloads were requested but Hugging Face offline mode is enabled. "
                                 "Start the run with HF_HUB_OFFLINE=0 and HF_DATASETS_OFFLINE=0.")

    try:
        from transformers import AutoConfig, AutoTokenizer
        if not check_model:
            pass  # A verified bundle supplies its own model config and weights.
        elif model.get("config"):
            # Inline configs are supported for tests and local model generation.
            if "model_type" not in model["config"]:
                raise LanguageAssetError("Inline model config needs model_type.")
        else:
            AutoConfig.from_pretrained(model["name"], revision=model.get("revision"),
                                       trust_remote_code=False, local_files_only=not allow_download)
        AutoTokenizer.from_pretrained(tokenizer["name"], revision=tokenizer.get("revision"),
                                      trust_remote_code=False, local_files_only=not allow_download)
    except LanguageAssetError:
        raise
    except Exception as error:
        reason = str(error).splitlines()[0] or type(error).__name__
        raise LanguageAssetError(
            f"Model or tokenizer assets for {model['name']!r} are unavailable or incompatible: {reason}. "
            "Use a supported Transformers architecture and a pinned revision; "
            "enable downloads for the first run or mount a complete local model directory."
        ) from None

    if check_model and model.get("weights") is not None:
        if local_model:
            snapshot = Path(model["name"])
        else:
            try:
                from huggingface_hub import snapshot_download
                snapshot = Path(snapshot_download(
                    repo_id=model["name"], revision=model["revision"],
                    allow_patterns=(*_MODEL_FILES, "*.safetensors"),
                    local_files_only=not allow_download,
                ))
                if not _weight_files_present(snapshot):
                    # Prefer safetensors and avoid downloading two full weight
                    # formats from repositories that publish both.
                    snapshot = Path(snapshot_download(
                        repo_id=model["name"], revision=model["revision"],
                        allow_patterns=(*_MODEL_FILES, "*.bin"),
                        local_files_only=not allow_download,
                    ))
            except Exception as error:
                reason = str(error).splitlines()[0] or type(error).__name__
                raise LanguageAssetError(
                    f"Pretrained weights for {model['name']}@{model['revision']} are unavailable: {reason}. "
                    "Enable downloads for the first run or mount a complete local model directory."
                ) from None
        if not _weight_files_present(snapshot):
            raise LanguageAssetError(
                f"Pretrained weights are incomplete in {snapshot}. Expected model.safetensors, "
                "pytorch_model.bin, or every shard named by a weight index. "
                "Enable downloads for the first run or provide a complete local model directory."
            )

    data = config.get("data") or {}
    if not isinstance(data, dict):
        raise LanguageAssetError("Language data specification must be an object.")
    data_status = "not_checked"
    if check_data:
        if data.get("kind") == "huggingface" and not allow_download and not _offline_enabled("HF_DATASETS_OFFLINE"):
            raise LanguageAssetError("Offline dataset preflight requires HF_DATASETS_OFFLINE=1 before Python starts.")
        try:
            from .tasks.language import text_records
            records = text_records(config)
            if (config.get("recovery") or config.get("lora")) and "train" not in records:
                raise LanguageAssetError("Language recovery needs a separate train split in the text dataset.")
            data_status = {role: len(items) for role, items in records.items()}
        except Exception as error:
            reason = str(error).splitlines()[0] or type(error).__name__
            raise LanguageAssetError(
                f"Language dataset is unavailable or invalid: {reason}. "
                "Provide disjoint text_json splits, or a pinned Hugging Face dataset cached locally "
                "or downloaded with explicit opt-in."
            ) from None
    return {
        "model": model["name"], "model_revision": model.get("revision"),
        "tokenizer": tokenizer["name"], "tokenizer_revision": tokenizer.get("revision"),
        "data": data_status, "download_allowed": bool(allow_download),
        "model_checked": bool(check_model),
        "model_location": "local" if local_model else "huggingface",
        "tokenizer_location": "local" if local_tokenizer else "huggingface",
    }
