import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tn_compression.language_assets import LanguageAssetError, preflight_language_assets
from tn_compression.models import load_model


def _local_language_config(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model_dir = tmp_path / "local-model"
    model = load_model({"source": "transformers", "config": {
        "model_type": "llama", "vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 2,
    }})
    model.save_pretrained(model_dir)
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "one": 2, "two": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.save_pretrained(model_dir)
    records = {role: [{"id": role, "text": f"{role} one two"}]
               for role in ("calibration", "validation", "test")}
    data_path = tmp_path / "texts.json"
    data_path.write_text(json.dumps(records), encoding="utf-8")
    return {"task": "causal_lm", "model": {"source": "transformers", "name": str(model_dir),
                                          "weights": "pretrained"},
            "data": {"kind": "text_json", "path": str(data_path)}}


def test_local_preflight_checks_all_assets_without_network(tmp_path):
    config = _local_language_config(tmp_path)
    report = preflight_language_assets(config)
    assert report["model_location"] == "local"
    assert report["data"] == {"calibration": 1, "validation": 1, "test": 1}
    assert report["download_allowed"] is False


def test_local_preflight_rejects_incomplete_weights(tmp_path):
    config = _local_language_config(tmp_path)
    (tmp_path / "local-model" / "model.safetensors").unlink()
    with pytest.raises(LanguageAssetError, match="weights are incomplete"):
        preflight_language_assets(config)
    # A saved bundle already has its model weights; tokenization and data still
    # need to work when the original Hugging Face cache is incomplete.
    report = preflight_language_assets(config, check_model=False)
    assert report["model_checked"] is False
    assert report["data"]["test"] == 1


def test_remote_source_requires_pinned_revision():
    with pytest.raises(LanguageAssetError, match="40-character"):
        preflight_language_assets({"task": "causal_lm", "model": {
            "source": "transformers", "name": "Qwen/Qwen2.5-0.5B", "weights": "pretrained"},
            "data": {"kind": "text_json", "path": "unused"}})


def test_recovery_preflight_requires_separate_training_partition(tmp_path):
    config = _local_language_config(tmp_path)
    config["recovery"] = {"policy": "trainable"}
    with pytest.raises(LanguageAssetError, match="separate train split"):
        preflight_language_assets(config)
