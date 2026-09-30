import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tn_compression.relevance import analyze_relevance


def test_classification_saliency_matches_weight_gradient_and_preserves_model():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = torch.ones_like(parameter)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    x = torch.tensor([[1.0, 0.5, -1.0], [0.0, 2.0, 1.0]])
    y = torch.tensor([0, 1])

    result = analyze_relevance(model, [(x, y)], task="classification")
    assert result["units"] == 2
    assert [row["layer_path"] for row in result["layers"]] == ["0", "2"]
    assert all(0 <= row["relative_relevance"] <= 1 for row in result["layers"])
    assert max(row["relative_relevance"] for row in result["layers"]) == 1
    json.dumps(result)

    # Compare against a direct first-order calculation on an independent copy.
    expected_model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    expected_model.load_state_dict(before)
    gradient = torch.autograd.grad(F.cross_entropy(expected_model(x), y), expected_model[0].weight)[0]
    expected = (expected_model[0].weight.detach() * gradient).abs().mean().item()
    assert result["layers"][0]["taylor_saliency"] == pytest.approx(expected)

    assert model.training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    for parameter in model.parameters():
        assert not parameter.requires_grad
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter.grad), rtol=0, atol=0)


def test_segmentation_masks_ignored_pixels_and_limits_batches():
    model = nn.Sequential(nn.Conv2d(3, 2, 1))
    images = torch.randn(1, 3, 2, 2)
    targets = torch.tensor([[[0, 1], [255, 0]]])
    result = analyze_relevance(model, [(images, targets), (images, targets)],
                               task="segmentation", max_batches=1)
    assert result["batches"] == 1
    assert result["units"] == 3
    assert result["layers"][0]["layer_path"] == "0"
    with pytest.raises(ValueError, match="no valid"):
        analyze_relevance(model, [(images, torch.full_like(targets, 255))], task="segmentation")


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 8)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


def test_causal_lm_uses_only_valid_next_token_targets():
    model = TinyCausalLM()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4], [0, 0, 5, 6]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])}
    result = analyze_relevance(model, [batch], task="causal_lm")
    assert result["units"] == 4
    assert result["unit_label"] == "valid next-token targets"
    assert [row["layer_path"] for row in result["layers"]] == ["proj"]
    empty = {key: value.clone() for key, value in batch.items()}
    empty["attention_mask"].zero_()
    with pytest.raises(ValueError, match="no valid"):
        analyze_relevance(model, [empty], task="causal_lm")


def test_relevance_spreads_bounded_layer_selection():
    model = nn.Sequential(*(nn.Linear(4, 4) for _ in range(9)))
    result = analyze_relevance(model, [(torch.randn(2, 4), torch.tensor([0, 1]))],
                               task="classification", max_layers=3)
    assert result["layers_available"] == 9
    assert result["layers_analyzed"] == 3
    assert [row["layer_path"] for row in result["layers"]] == ["0", "4", "8"]


def test_relevance_runs_on_transformers_causal_lm():
    pytest.importorskip("transformers")
    from tn_compression.models import load_model

    model = load_model({"source": "transformers", "config": {
        "model_type": "llama", "vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "max_position_embeddings": 32,
    }})
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
             "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    result = analyze_relevance(model, [batch], task="causal_lm", max_layers=3)
    assert result["units"] == 3
    assert result["layers"]
    assert all(row["layer_path"] for row in result["layers"])
