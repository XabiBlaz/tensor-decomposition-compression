import pytest
import torch

pytest.importorskip("transformers")

from tn_compression.models import load_model
from tn_compression.tasks.recovery import recover_language


def test_recovery_changes_only_selected_parameters_and_obeys_token_budget():
    model = load_model({"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 16}}).eval()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batches = [{"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.ones(1, 5)}]
    selected = "model.layers.0.mlp.up_proj"
    result = recover_language(model, batches, token_budget=6, max_updates=3, layers=[selected], learning_rate=0.01)
    assert result["tokens"] == 6 and result["updates"] == 2 and not model.training
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])]
    assert changed == [selected + ".weight"]


def test_recovery_rejects_missing_targets():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="no valid targets"):
        recover_language(model, [{"input_ids": torch.ones(1, 1, dtype=torch.long)}],
                         token_budget=2, max_updates=1, layers=["0"])
