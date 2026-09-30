import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from tn_compression.checkpoints import load_bundle, save_bundle
from tn_compression.models import load_model
from tn_compression.tasks.language import evaluate_language
from tn_compression.tasks.lora_recovery import load_lora_adapter, recover_language_lora


def test_lora_adapter_trains_and_reloads_on_exact_bundle(tmp_path):
    spec = {"source": "transformers", "config": {
        "model_type": "llama", "vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 2,
    }}
    base = load_model(spec).eval()
    bundle = tmp_path / "base"
    save_bundle(base, bundle)
    model, _ = load_bundle(bundle)
    batches = [{"input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.ones(1, 5, dtype=torch.long)}]
    trained, report = recover_language_lora(
        model, batches, output_dir=tmp_path / "recovered", base_bundle_path=bundle,
        token_budget=4, max_updates=1, rank=2, alpha=4, target_modules=["q_proj"],
        learning_rate=1e-3,
    )
    restored = load_lora_adapter(tmp_path / "recovered" / "adapter", base_bundle_path=bundle)
    assert report["training"]["tokens"] == 4
    assert report["training"]["trainable_parameters"] > 0
    assert report["base_weights_sha256"]
    assert evaluate_language(trained, batches)["nll"] == pytest.approx(
        evaluate_language(restored, batches)["nll"], abs=1e-6)

    other = tmp_path / "other"
    save_bundle(load_model(spec), other)
    with pytest.raises(ValueError, match="different base bundle"):
        load_lora_adapter(tmp_path / "recovered" / "adapter", base_bundle_path=other)
