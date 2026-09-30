"""Quality bounds must never silently accept unavailable or invalid evidence."""

import json

import pytest
import torch

import tn_compression.analyzer.service as service
from tn_compression.analyzer.schema import AnalysisReport, CandidateResult


@pytest.mark.parametrize("task,metric,key,direction", [
    ("classification", None, "loss", "lower"),
    ("classification", "accuracy", "top1", "higher"),
    ("segmentation", "mean_iou", "mean_iou", "higher"),
    ("segmentation", "mean_dice", "mean_dice", "higher"),
    ("causal_lm", None, "nll", "lower"),
    ("causal_lm", "perplexity", "perplexity", "lower"),
    ("detection", None, "ap", "higher"),
    ("detection", "ap50", "ap50", "higher"),
])
def test_metric_direction_and_absolute_units(task, metric, key, direction):
    config = {"analysis": {"quality_metric": metric}} if metric else {}
    adapter = service.TaskAdapter(task, config)
    measured = 0.69 if direction == "higher" else 0.71
    assert adapter.quality_loss({key: 0.70}, {key: measured}) == pytest.approx(0.01)
    assert adapter.quality_contract()["quality_metric"] == key
    assert adapter.quality_contract()["quality_direction"] == direction


@pytest.mark.parametrize("metrics", [{}, {"top1": None}, {"top1": float("nan")},
                                     {"top1": float("inf")}, {"top1": True},
                                     {"top1": 0.9, "loss": float("nan")}])
def test_required_quality_evidence_is_finite(metrics):
    adapter = service.TaskAdapter("classification", {"analysis": {"quality_metric": "accuracy"}})
    with pytest.raises(ValueError, match="finite number"):
        adapter.quality_loss({"top1": 0.9}, metrics)
    with pytest.raises(ValueError, match="Baseline metric"):
        adapter.quality_loss(metrics, {"top1": 0.9})


def test_task_incompatible_metric_is_rejected():
    with pytest.raises(ValueError, match="Unsupported quality_metric"):
        service.TaskAdapter("segmentation", {"analysis": {"quality_metric": "accuracy"}})


@pytest.mark.parametrize("target,bound", [(float("nan"), 0), (float("inf"), 0),
                                         (1, float("nan")), (1, float("inf")),
                                         (True, 0), (1, True), (1, -0.01), (0, 0)])
def test_invalid_constraints_fail_before_calibration(target, bound):
    with pytest.raises(ValueError):
        service.analyze_model(torch.nn.Linear(4, 4), {}, level="validated",
                              target_size_bytes=target, max_quality_loss=bound)


def workflow():
    return {
        "task": "classification", "num_classes": 4,
        "analysis": {
            "quality_metric": "accuracy", "include": ["0"], "max_samples": 4,
            "candidate_grid": {
                "linear": {"methods": ["svd"], "ranks": [1]},
                "quantization": {"enabled": False}, "gated_mlp": {"methods": []},
            },
        },
    }


def test_selection_uses_accuracy_and_preserves_contract_in_plan(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    data = [(torch.eye(4), torch.arange(4))]
    monkeypatch.setattr(service.TaskAdapter, "evaluate", lambda *args: {"top1": 0.90, "loss": 1.0})
    monkeypatch.setattr(service.TaskAdapter, "measure_intervention",
                        lambda *args: {"top1": 0.89, "loss": 100.0})
    report = service.analyze_model(model, workflow(), level="validated",
                                   calibration_batches=data, validation_batches=data,
                                   target_size_bytes=32, max_quality_loss=0.02)
    assert report.compression_plan["status"] == "feasible"
    assert report.summary["accepted"] == 1
    constraints = report.compression_plan["constraints"]
    assert constraints == report.constraints
    assert constraints["quality_metric"] == "top1"
    assert constraints["quality_direction"] == "higher"
    assert constraints["quality_units"] == "absolute_fraction"
    assert constraints["max_quality_loss"] == 0.02
    # No NaN/Infinity may leak into this evidence artifact.
    restored = AnalysisReport.from_dict(json.loads(json.dumps(report.to_dict(), allow_nan=False)))
    assert restored.constraints == constraints


@pytest.mark.parametrize("invalid", [{}, {"top1": float("nan")}, {"top1": None}])
def test_invalid_baseline_stops_analysis(monkeypatch, invalid):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    original = model[0]
    data = [(torch.eye(4), torch.arange(4))]
    monkeypatch.setattr(service.TaskAdapter, "evaluate", lambda *args: invalid)
    with pytest.raises(ValueError, match="Baseline metric"):
        service.analyze_model(model, workflow(), level="validated", calibration_batches=data,
                              validation_batches=data, target_size_bytes=1, max_quality_loss=0.01)
    assert model[0] is original


def test_invalid_intervention_rejected_without_invalid_report_values(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    data = [(torch.eye(4), torch.arange(4))]
    monkeypatch.setattr(service.TaskAdapter, "evaluate", lambda *args: {"top1": 0.9})
    monkeypatch.setattr(service.TaskAdapter, "measure_intervention",
                        lambda *args: {"top1": float("nan")})
    report = service.analyze_model(model, workflow(), level="validated", calibration_batches=data,
                                   validation_batches=data, target_size_bytes=1, max_quality_loss=0.01)
    assert report.summary["accepted"] == 0
    assert any(row.status == "validation_failed" for row in report.candidates)
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("invalid", [{}, {"loss": float("nan")}, {"loss": float("inf")}])
def test_invalid_cumulative_measurement_restores_all_model_layers(monkeypatch, invalid):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False))
    originals = list(model)
    candidates = [CandidateResult(
        model_fingerprint="test", layer_path=str(index), module_type="Linear", method="svd",
        configuration={"rank": 1}, structurally_eligible=True, original_parameters=16,
        candidate_parameters=4, original_artifact_bytes=64, estimated_artifact_bytes=16,
        status="validated", quality_loss=0.0,
    ) for index in range(2)]
    adapter = service.TaskAdapter("classification", {})
    results = iter([{"loss": 1.0}, invalid])

    def measure(current_model, path, replacement, batches):
        with service.candidate_intervention(current_model, path, replacement):
            return next(results)

    monkeypatch.setattr(adapter, "measure_intervention", measure)
    monkeypatch.setattr(service, "materialize_candidate",
                        lambda *args: torch.nn.Linear(4, 1, bias=False))
    accepted, cumulative, final_bytes = service._select_candidates(
        model, candidates, [], adapter, {"loss": 1.0},
        {(str(index), "module_inputs"): None for index in range(2)}, 1, 0.01)
    assert accepted == [candidates[0]]
    assert cumulative == {"loss": 1.0}
    assert final_bytes == 80
    assert candidates[1].decision_reason.startswith("cumulative_validation_failed:")
    assert all(model[index] is original for index, original in enumerate(originals))
