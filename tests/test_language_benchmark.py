import pytest

pytest.importorskip("transformers")
pytest.importorskip("psutil")

from tn_compression.benchmark import benchmark_bundle
from tn_compression.checkpoints import save_bundle
from tn_compression.models import load_model


def test_isolated_generation_records_actual_steps(tmp_path):
    model = load_model({"source": "transformers", "config": {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 16}})
    save_bundle(model, tmp_path / "bundle")
    result = benchmark_bundle(tmp_path / "bundle", task="causal_lm", input_shape=[1, 8],
                              output_tokens=3, iterations=2, warmup=0)
    assert result["status"] == "ok", result
    assert len(result["requests"]) == 2
    assert all(len(row["inter_token_ms"]) == 2 and row["output_tokens"] == 3 for row in result["requests"])
    assert result["output_tokens_per_second"] > 0
