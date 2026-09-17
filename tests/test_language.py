import json

import pytest
import torch

pytest.importorskip("transformers")

from tn_compression.api import apply_compression_plan, build_plan
from tn_compression.checkpoints import load_bundle, save_bundle
from tn_compression.models import load_model
from tn_compression.tasks.language import evaluate_language, shifted_targets, teacher_kl, text_records


def tiny_model():
    return load_model({"source": "transformers", "config": {"model_type": "llama", "vocab_size": 32,
                       "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "num_key_value_heads": 2, "max_position_embeddings": 64}}).eval()


def batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4], [0, 0, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])}


def test_causal_loss_excludes_padding_and_weights_tokens():
    model = tiny_model()
    targets, mask = shifted_targets(batch())
    assert mask.tolist() == [[True, True, True], [False, False, True]]
    result = evaluate_language(model, [batch()])
    with torch.no_grad():
        logits = model(**batch()).logits[:, :-1]
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1), ignore_index=-100)
    assert result["tokens"] == 4
    assert result["nll"] == pytest.approx(expected.item())
    assert teacher_kl(model, model, [batch()])["teacher_to_candidate_kl"] == pytest.approx(0, abs=1e-7)


def test_linear_compression_preserves_cache_generation_and_bundle(tmp_path):
    model = tiny_model()
    path = "model.layers.0.mlp.up_proj"
    plan = build_plan(model, {"compression": {"mode": "individual", "layers": {path: {"type": "svd", "rank": 4}}}}, write=False).plan
    assert apply_compression_plan(model, plan)[0] == 1
    with torch.no_grad():
        expected = model(**batch(), use_cache=True)
        generated = model.generate(torch.tensor([[1, 2]]), max_new_tokens=2, do_sample=False,
                                   pad_token_id=0, eos_token_id=None)
    assert expected.past_key_values is not None
    assert generated.shape == (1, 4)
    save_bundle(model, tmp_path)
    restored, _ = load_bundle(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch()).logits, expected.logits, rtol=0, atol=0)


def test_loss_aggregation_uses_tokens_not_batch_means():
    model = tiny_model()
    data = batch()
    separate = [{key: value[i:i + 1] for key, value in data.items()} for i in range(2)]
    assert evaluate_language(model, separate)["nll"] == pytest.approx(evaluate_language(model, [data])["nll"])
    data["attention_mask"].zero_()
    with pytest.raises(ValueError, match="no valid"):
        evaluate_language(model, [data])


@pytest.mark.parametrize("duplicate", ["id", "text"])
def test_text_roles_reject_overlap(tmp_path, duplicate):
    records = {role: [{"id": role, "text": role}] for role in ("calibration", "validation", "test")}
    records["test"][0][duplicate] = records["calibration"][0][duplicate]
    path = tmp_path / "text.json"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="disjoint"):
        text_records({"data": {"kind": "text_json", "path": str(path)}})
