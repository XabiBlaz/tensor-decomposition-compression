import pytest
import torch
from torch import nn

from tn_compression.api import apply_compression_plan, build_plan
from tn_compression.analyzer.service import analyze_model, apply_compression_plan as apply_analyzer_plan
from tn_compression.checkpoints import load_bundle, save_bundle
from tn_compression.models import load_model


def custom_model():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(32, 4))


@pytest.mark.parametrize("method", ["partial_tucker", "tensor_train", "cp3", "cp4", "svd"])
def test_bundle_rebuilds_without_decomposition(tmp_path, monkeypatch, method):
    model = custom_model().double().eval()
    inputs = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    config = {"compression": {"default_method": {"type": method, "rank": 2}}}
    plan = build_plan(model, config, write=False).plan
    modified, _, _ = apply_compression_plan(model, plan)
    assert modified > 0
    expected = model(inputs)
    save_bundle(model, tmp_path, model_spec={"source": "custom"}, metadata={"plan": plan})
    monkeypatch.setattr(torch.linalg, "svd", lambda *args, **kwargs: pytest.fail("decomposition during load"))
    restored, manifest = load_bundle(tmp_path, factory=custom_model)
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
    assert manifest["metadata"]["plan"] == plan
    assert not restored.training


def test_registered_nonbenchmark_model_can_roundtrip(tmp_path):
    spec = {"source": "torchvision", "name": "resnet34", "weights": None,
            "task": "classification", "kwargs": {"num_classes": 7}}
    model = load_model(spec).eval()
    save_bundle(model, tmp_path)
    restored, _ = load_bundle(tmp_path)
    assert restored.fc.out_features == 7
    assert model.state_dict().keys() == restored.state_dict().keys()


def test_bundle_rejects_missing_factory_and_corrupted_weights(tmp_path):
    save_bundle(custom_model(), tmp_path, model_spec={"source": "custom"})
    with pytest.raises(ValueError, match="factory"):
        load_bundle(tmp_path)
    with open(tmp_path / "weights.pt", "ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path, factory=custom_model)


def test_local_hub_loader_uses_entrypoint_contract(tmp_path):
    (tmp_path / "hubconf.py").write_text(
        "from torch import nn\ndef fixture(pretrained=False):\n    return nn.Linear(3, 2)\n"
        "def wrapped():\n    return {'model': nn.Linear(3, 2)}\n")
    spec = {"source": "torch_hub", "repository": str(tmp_path), "local": True, "name": "fixture"}
    assert isinstance(load_model(spec), nn.Linear)
    with pytest.raises(ValueError, match="wrapper"):
        load_model({**spec, "name": "wrapped"})
def test_transformer_bundle_preserves_attention_backend_and_rotary_buffers(tmp_path):
    pytest.importorskip("transformers")
    model = load_model({"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 32}}).double().eval()
    model._tn_model_spec["config"] = model.config.to_dict()
    save_bundle(model, tmp_path / "bundle")
    restored, _ = load_bundle(tmp_path / "bundle")
    assert restored.config._attn_implementation == model.config._attn_implementation
    assert restored.model.rotary_emb.inv_freq.dtype == torch.float64
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(model(tokens).logits, restored(tokens).logits, rtol=0, atol=0)


def test_analyzer_pruning_plan_round_trips_resolved_structure(tmp_path):
    pytest.importorskip("transformers")
    spec = {"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "vocab_size": 16}}
    workflow = {
        "seed": 4, "task": "causal_lm", "model": spec,
        "analysis": {
            "include": ["model.layers.0.mlp"], "max_samples": 8,
            "candidate_grid": {
                "linear": {"methods": []}, "conv2d": {"methods": []},
                "quantization": {"enabled": False},
                "gated_mlp": {
                    "methods": ["gated_mlp_pruning"], "widths": [8],
                    "selection": "activation",
                },
            },
        },
    }
    model = load_model(spec).eval()
    batches = [{"input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long)}]
    report = analyze_model(
        model, workflow, level="validated", calibration_batches=batches,
        calibration_ids=["tiny-language-0"], validation_batches=batches,
        target_size_bytes=1, max_quality_loss=100)
    transformation = report.compression_plan["transformations"][0]
    retained = transformation["configuration"]["retained_indices"]
    assert len(retained) == 8
    apply_analyzer_plan(
        model, report.compression_plan, workflow,
        calibration_batches=batches, calibration_ids=["tiny-language-0"])
    mlp = model.model.layers[0].mlp
    shapes = tuple(tuple(layer.weight.shape) for layer in
                   (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
    tokens = batches[0]["input_ids"]
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits
    save_bundle(model, tmp_path / "pruned-bundle")
    restored, manifest = load_bundle(tmp_path / "pruned-bundle")
    restored_mlp = restored.model.layers[0].mlp
    restored_shapes = tuple(tuple(layer.weight.shape) for layer in
                            (restored_mlp.gate_proj, restored_mlp.up_proj, restored_mlp.down_proj))
    assert restored_shapes == shapes
    assert manifest["transformations"] == model._tn_transformations
    assert manifest["transformations"][-1]["configuration"]["retained_indices"] == retained
    with torch.no_grad():
        torch.testing.assert_close(
            restored(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
