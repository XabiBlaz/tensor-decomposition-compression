import copy

import pytest
import torch

from tn_compression.analyzer.candidates import generate_candidates, model_fingerprint
from tn_compression.pruning import inspect_gated_mlps


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 6, 3, padding=1)
        self.depthwise = torch.nn.Conv2d(4, 4, 3, groups=4)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(6, 3)

    def forward(self, inputs):
        return self.head(self.pool(self.conv(inputs)).flatten(1))


def test_model_fingerprint_changes_with_weights():
    model = TinyModel()
    first = model_fingerprint(model)
    with torch.no_grad():
        model.head.weight[0, 0].add_(1)
    assert model_fingerprint(model) != first


def test_candidate_generation_is_bounded_legal_and_non_mutating():
    model = TinyModel()
    state = copy.deepcopy(model.state_dict())
    candidates = generate_candidates(model, {
        "include": ["conv", "depthwise", "head"],
        "max_candidates_per_layer": 5,
        "max_candidates_per_method": 2,
        "candidate_grid": {
            "linear": {"methods": ["svd", "weighted_svd"], "ranks": [1, 2, 99]},
            "conv2d": {"methods": ["partial_tucker", "cp3"], "rank_ratios": [0.25, 0.5, 0.75]},
            "quantization": {"enabled": True, "bits": [4], "group_sizes": [4]},
            "gated_mlp": {"methods": []},
        },
    })
    assert {item.method for item in candidates if item.layer_path == "conv"} == {"partial_tucker", "cp3"}
    assert all(item.configuration["rank"] <= 3 for item in candidates
               if item.layer_path == "head" and item.method in {"svd", "weighted_svd"})
    assert len([item for item in candidates if item.layer_path == "conv"
                and item.decision != "rejected"]) <= 5
    assert len([item for item in candidates if item.layer_path == "head"
                and item.decision != "rejected"]) <= 5
    unsupported = next(item for item in candidates if item.layer_path == "depthwise")
    assert unsupported.rejection_reason == "grouped_conv_not_supported"
    quantized = next(item for item in candidates if item.method == "round_to_nearest")
    assert quantized.structurally_eligible
    assert quantized.estimated_bytes_saved == 0
    assert quantized.decision == "rejected"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])


def test_backend_support_is_separate_from_method_eligibility():
    candidates = generate_candidates(TinyModel(), {
        "include": ["head"],
        "candidate_grid": {
            "linear": {"methods": ["svd"], "ranks": [1]},
            "quantization": {"enabled": False},
            "gated_mlp": {"methods": []},
        },
    }, requested_backend="vllm")
    candidate = candidates[0]
    assert candidate.structurally_eligible
    assert not candidate.backend_support["supported"]
    assert candidate.decision == "rejected"


def test_candidate_limit_is_total_per_layer_across_methods():
    options = {
        "include": ["conv"],
        "max_candidates_per_layer": 5,
        "max_candidates_per_method": 2,
        "candidate_grid": {
            "conv2d": {"methods": ["partial_tucker", "cp3", "cp4"],
                       "rank_ratios": [0.25, 0.5, 0.75]},
            "quantization": {"enabled": False},
            "gated_mlp": {"methods": []},
        },
    }
    candidates = generate_candidates(TinyModel(), options)
    repeated = generate_candidates(TinyModel(), options, fingerprint=candidates[0].model_fingerprint)
    assert len(candidates) == 5
    assert [item.method for item in candidates] == [
        "partial_tucker", "cp3", "cp4", "partial_tucker", "cp3"]
    assert {item.method for item in candidates[:3]} == {
        "partial_tucker", "cp3", "cp4"}
    assert all(sum(item.method == method for item in candidates) <= 2
               for method in {item.method for item in candidates})
    assert [(item.method, item.configuration, item.candidate_id) for item in repeated] == [
        (item.method, item.configuration, item.candidate_id) for item in candidates]


def test_no_supported_gated_mlp_is_an_absence_not_a_discovery_failure():
    groups, failures = inspect_gated_mlps(TinyModel())
    assert groups == {} and failures == []
    candidates = generate_candidates(TinyModel(), {
        "include": ["*"],
        "candidate_grid": {
            "linear": {"methods": []}, "conv2d": {"methods": []},
            "quantization": {"enabled": False},
            "gated_mlp": {"methods": ["gated_mlp_pruning"], "widths": [2]},
        },
    })
    assert not any(item.method == "gated_mlp_pruning" for item in candidates)


def _qwen_mlp_model():
    pytest.importorskip("transformers")
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = Qwen2Config(hidden_size=8, intermediate_size=12,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=2, vocab_size=16)
            self.mlp = Qwen2MLP(config)

    return Holder()


def test_supported_qwen_block_produces_pruning_candidates():
    model = _qwen_mlp_model()
    groups, failures = inspect_gated_mlps(model)
    assert list(groups) == ["mlp"] and failures == []
    candidates = generate_candidates(model, {
        "include": ["mlp"],
        "candidate_grid": {
            "linear": {"methods": []}, "conv2d": {"methods": []},
            "quantization": {"enabled": False},
            "gated_mlp": {"methods": ["gated_mlp_pruning"], "widths": [6]},
        },
    })
    assert len(candidates) == 1 and candidates[0].structurally_eligible


def test_malformed_qwen_block_is_reported_with_reason():
    model = _qwen_mlp_model()
    model.mlp.down_proj = torch.nn.Linear(11, 8, bias=False)
    groups, failures = inspect_gated_mlps(model)
    assert groups == {}
    assert failures[0]["reason"] == "incompatible_intermediate_dimensions"
    candidates = generate_candidates(model, {
        "include": ["mlp"],
        "candidate_grid": {
            "linear": {"methods": []}, "conv2d": {"methods": []},
            "quantization": {"enabled": False},
            "gated_mlp": {"methods": ["gated_mlp_pruning"], "widths": [6]},
        },
    })
    assert candidates[0].rejection_reason == "incompatible_intermediate_dimensions"


def test_shared_qwen_projection_is_reported_as_protected():
    model = _qwen_mlp_model()
    model.mlp.up_proj.weight = model.mlp.gate_proj.weight
    groups, failures = inspect_gated_mlps(model)
    assert groups == {}
    assert "shared_parameter_requires_adapter" in failures[0]["reason"]
    assert failures[0]["protected"]
