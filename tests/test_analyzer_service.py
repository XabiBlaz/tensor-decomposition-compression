import copy

import torch

from tn_compression.analyzer.schema import CandidateResult
from tn_compression.analyzer.service import analyze_model, apply_compression_plan, pareto_candidates


class TinyClassifier(torch.nn.Module):
    def __init__(self, rank_one=False):
        super().__init__()
        self.head = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            if rank_one:
                self.head.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0])[:, None]
                                       @ torch.tensor([0.5, -0.5, 1.0, 0.25])[None])
            else:
                self.head.weight.copy_(torch.eye(4))

    def forward(self, inputs):
        return self.head(inputs)


def config():
    return {
        "seed": 3,
        "task": "classification",
        "num_classes": 4,
        "model": {"source": "custom", "name": "tiny"},
        "data": {"kind": "synthetic", "normalization": "none"},
        "analysis": {
            "include": ["head"],
            "max_samples": 8,
            "max_validated_per_layer": 2,
            "candidate_grid": {
                "linear": {"methods": ["svd"], "ranks": [1, 2]},
                "quantization": {"enabled": False},
                "gated_mlp": {"methods": []},
            },
        },
    }


def batches():
    inputs = torch.eye(4).repeat(2, 1)
    targets = torch.arange(4).repeat(2)
    return [(inputs, targets)]


def test_calibrated_analysis_is_non_mutating_and_records_evidence():
    model = TinyClassifier()
    original = model.head
    state = copy.deepcopy(model.state_dict())
    report = analyze_model(model, config(), level="calibrated", calibration_batches=batches(),
                           calibration_ids=["cal-0"])
    scored = [candidate for candidate in report.candidates if candidate.status == "calibrated"]
    assert scored and all(candidate.normalized_local_error is not None for candidate in scored)
    assert all(candidate.calibration_evidence["identifiers"] == ["cal-0"] for candidate in scored)
    assert model.head is original
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value)


def test_validated_analysis_rolls_back_rejected_cumulative_change():
    model = TinyClassifier()
    original = model.head
    original_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    report = analyze_model(
        model, config(), level="validated", calibration_batches=batches(), validation_batches=batches(),
        target_size_bytes=1, max_quality_loss=0.0)
    assert report.compression_plan["status"] == "infeasible"
    assert not report.compression_plan["transformations"]
    assert any(candidate.decision_reason == "cumulative_quality_constraint"
               for candidate in report.candidates)
    assert report.summary["estimated_final_tensor_bytes"] == original_bytes
    assert model.head is original


def test_validated_plan_is_consumable_and_analysis_restores_model():
    model = TinyClassifier(rank_one=True)
    original = model.head
    before = model(torch.eye(4))
    original_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    report = analyze_model(
        model, config(), level="validated", calibration_batches=batches(), validation_batches=batches(),
        target_size_bytes=original_bytes - 1, max_quality_loss=1e-6)
    assert report.compression_plan["status"] == "feasible"
    assert len(report.compression_plan["transformations"]) == 1
    assert model.head is original
    result = apply_compression_plan(model, report.compression_plan, config())
    assert result["applied"] == 1
    assert isinstance(model.head, torch.nn.Sequential)
    torch.testing.assert_close(model(torch.eye(4)), before)


def test_pareto_filter_marks_dominated_candidates_within_layer_only():
    def row(path, identifier, saved, error):
        item = CandidateResult(
            model_fingerprint="m", layer_path=path, module_type="Linear", method="svd",
            configuration={"rank": identifier}, structurally_eligible=True,
            original_parameters=20, candidate_parameters=10,
            original_artifact_bytes=100, estimated_artifact_bytes=100 - saved,
            normalized_local_error=error,
            backend_support={"supported": True, "verified": True, "reason": "test"},
        )
        return item
    dominated = row("a", 1, 10, 0.2)
    winner = row("a", 2, 20, 0.1)
    other_layer = row("b", 3, 5, 0.5)
    assert pareto_candidates([dominated, winner, other_layer]) == [winner, other_layer]
    assert dominated.decision_reason == "dominated_within_layer"
