import pytest
import torch
from torch import nn

from tn_compression.api import apply_compression_plan, build_plan
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

