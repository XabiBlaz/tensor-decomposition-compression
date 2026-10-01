import pytest
import torch

pytest.importorskip("transformers")

from tn_compression.decompositions.cp import cp_linear
from tn_compression.language_export import export_fixed_logits_onnx, export_language
from tn_compression.models import load_model
from tn_compression.pruning import gated_mlps, prune_gated_mlps


def test_standard_export_reloads_pruned_width_and_rejects_factors(tmp_path):
    model = load_model({"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 16}}).eval()
    prune_gated_mlps(model, {path: list(range(8)) for path in gated_mlps(model)})
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    result = export_language(model, tmp_path / "hf", batch)
    assert result["status"] == "verified_transformers"
    mlp = model.get_submodule("model.layers.0.mlp")
    mlp.up_proj = cp_linear(mlp.up_proj, 2)
    mlp.up_proj._tn_replacement = True
    with pytest.raises(ValueError, match="factorized"):
        export_language(model, tmp_path / "unsupported", batch)


def test_fixed_logits_onnx_is_verified_but_not_serving_claim(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = load_model({"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "vocab_size": 16}}).eval()
    path = "model.layers.0.mlp.up_proj"
    factor = cp_linear(model.get_submodule(path), 2)
    factor._tn_replacement = True
    model.get_submodule("model.layers.0.mlp").up_proj = factor
    batch = {"input_ids": torch.tensor([[1, 2, 3]]),
             "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    result = export_fixed_logits_onnx(model, tmp_path / "fixed.onnx", batch)
    assert result["status"] == "verified_fixed_shape_logits_only"
    assert result["serving_ready"] is False
    assert any("cache" in limitation for limitation in result["limitations"])
