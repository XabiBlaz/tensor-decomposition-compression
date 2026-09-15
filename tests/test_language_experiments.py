import pytest
import torch

pytest.importorskip("transformers")

from tn_compression.language_experiments import evaluate_candidates
from tn_compression.models import load_model


def test_candidate_trials_leave_model_unchanged():
    model = load_model({"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 32}})
    path = "model.layers.0.mlp.up_proj"
    original = model.get_submodule(path)
    batches = [{"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}]
    result = evaluate_candidates(model, batches, batches, {path: [2, 4]}, max_rows=3)
    assert model.get_submodule(path) is original and model.training
    assert len(result["candidates"]) == 3
    assert all(row["bytes_saved"] > 0 for row in result["candidates"][1:])
    assert all(row["tokens"] == 3 for row in result["candidates"][1:])
    with pytest.raises(ValueError, match="integer ranks"):
        evaluate_candidates(model, batches, batches, {path: [0]})
