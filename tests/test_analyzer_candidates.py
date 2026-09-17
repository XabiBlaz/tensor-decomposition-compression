import copy

import torch

from tn_compression.analyzer.candidates import generate_candidates, model_fingerprint


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
        "max_candidates_per_layer": 2,
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
    assert sum(item.method == "cp3" for item in candidates if item.layer_path == "conv") <= 2
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
