import copy

import pytest
import torch

pytest.importorskip("transformers")

from tn_compression.checkpoints import load_bundle, save_bundle
from tn_compression.models import load_model
from tn_compression.pruning import gated_mlps, prune_gated_mlps, select_channels


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_slicing_matches_masked_mlp_and_survives_reload(tmp_path, family):
    model = load_model({"source": "transformers", "config": {
        "model_type": family, "hidden_size": 16, "intermediate_size": 32, "mlp_bias": True,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 32}}).double().eval()
    masked = copy.deepcopy(model)
    groups = gated_mlps(model)
    selections = {path: select_channels(mlp, 12, method="random", seed=2) for path, mlp in groups.items()}
    count = sum(p.numel() for p in model.parameters())
    for path, indices in selections.items():
        keep = torch.zeros(32, dtype=torch.bool)
        keep[indices] = True
        with torch.no_grad():
            masked.get_submodule(path).down_proj.weight[:, ~keep] = 0
    prune_gated_mlps(model, selections)
    assert sum(p.numel() for p in model.parameters()) < count
    assert model.config.intermediate_size == 12
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(tokens, use_cache=True).logits
        torch.testing.assert_close(expected, masked(tokens).logits, rtol=1e-10, atol=1e-10)
    save_bundle(model, tmp_path / "bundle")
    restored, metadata = load_bundle(tmp_path / "bundle")
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)
        assert restored.generate(tokens, max_new_tokens=1, pad_token_id=0).shape == (1, 5)
    assert metadata["transformations"][0]["retained_indices"] == selections


def test_invalid_selection_does_not_partially_prune():
    model = load_model({"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 16}})
    groups = gated_mlps(model)
    selections = {path: [0, 1] for path in groups}
    selections[list(groups)[-1]] = [1, 1]
    with pytest.raises(ValueError, match="repeated"):
        prune_gated_mlps(model, selections)
    assert all(mlp.up_proj.out_features == 16 for mlp in groups.values())


def test_activation_score_matches_single_channel_removal():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    mlp = Qwen2MLP(Qwen2Config(hidden_size=4, intermediate_size=8)).double()
    activations = torch.randn(9, 8, dtype=torch.float64)
    errors = [(activations[:, index:index + 1] @ mlp.down_proj.weight[:, index:index + 1].T).square().sum(1).mean()
              for index in range(8)]
    expected = torch.argsort(torch.stack(errors), descending=True)[:3].sort().values.tolist()
    assert select_channels(mlp, 3, method="activation", activations=activations) == expected


def test_rejected_pruning_restores_original_architecture(monkeypatch):
    import tn_compression.language_experiments as experiments
    model = load_model({"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 16}})
    mlp = next(iter(gated_mlps(model).values()))
    original = mlp.up_proj
    measurements = iter([{"nll": 1.0}, {"nll": 2.0}])
    monkeypatch.setattr(experiments, "evaluate_language", lambda *args, **kwargs: next(measurements))
    result = experiments.prune_language(model, [], [], width=8, max_delta_nll=0.1)
    assert result["status"] == "rejected"
    assert mlp.up_proj is original and model.config.intermediate_size == 16
