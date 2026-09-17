import copy
from types import SimpleNamespace

import pytest
import torch

import tn_compression.analyzer.service as analyzer_service
from tn_compression.analyzer.candidates import model_fingerprint
from tn_compression.analyzer.schema import CandidateResult, stable_candidate_id
from tn_compression.analyzer.service import (
    _pruned_mlp_candidate,
    analyze_model,
    apply_compression_plan,
    pareto_candidates,
)


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
    assert all(candidate.measured_artifact_bytes > 0 for candidate in scored)
    assert all(candidate.calibration_evidence["identifiers"] == ["cal-0"] for candidate in scored)
    assert report.analysis_context_fingerprint
    assert report.analysis_context["calibration_content_sha256"]
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


def test_dense_reference_quantization_is_validated_but_never_allocated():
    workflow = config()
    workflow["analysis"]["candidate_grid"]["linear"]["methods"] = []
    workflow["analysis"]["candidate_grid"]["quantization"] = {
        "enabled": True, "bits": [4], "group_sizes": [4]}
    model = TinyClassifier()
    report = analyze_model(
        model, workflow, level="validated", calibration_batches=batches(),
        validation_batches=batches(), target_size_bytes=1, max_quality_loss=100.0)
    diagnostic = next(candidate for candidate in report.candidates
                      if candidate.method == "round_to_nearest")
    assert diagnostic.status == "validated"
    assert diagnostic.decision == "validated"
    assert not diagnostic.allocation_eligible
    assert diagnostic.estimated_bytes_saved == 0
    assert diagnostic.normalized_local_error is not None
    assert diagnostic.full_model_metrics is not None
    assert report.compression_plan["status"] == "infeasible"
    assert report.compression_plan["transformations"] == []


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


def test_weighted_plan_rejects_changed_calibration_context_before_mutation():
    workflow = config()
    workflow["analysis"]["candidate_grid"]["linear"]["methods"] = ["weighted_svd"]
    workflow["analysis"]["candidate_grid"]["linear"]["ranks"] = [1]
    model = TinyClassifier(rank_one=True)
    original = model.head
    calibration = batches()
    original_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    report = analyze_model(
        model, workflow, level="validated", calibration_batches=calibration,
        calibration_ids=["cal-0"], validation_batches=batches(),
        target_size_bytes=original_bytes - 1, max_quality_loss=1e-6)
    assert report.compression_plan["transformations"][0]["method"] == "weighted_svd"
    changed = [(calibration[0][0] + 0.25, calibration[0][1])]
    with pytest.raises(ValueError, match="Analysis-context fingerprint mismatch"):
        apply_compression_plan(
            model, report.compression_plan, workflow,
            calibration_batches=changed, calibration_ids=["cal-0"])
    assert model.head is original


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


class TinyConvClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 3, padding=1, bias=False)

    def forward(self, inputs):
        return self.conv(inputs).mean(dim=(2, 3))


def test_calibrated_vision_candidate_uses_bounded_feature_maps():
    model = TinyConvClassifier()
    workflow = {
        "task": "classification", "num_classes": 2, "model": {"name": "tiny-conv"},
        "analysis": {
            "include": ["conv"], "max_samples": 2, "max_spatial_size": 4,
            "candidate_grid": {
                "conv2d": {"methods": ["partial_tucker"], "ranks": [[1, 1]]},
                "linear": {"methods": []}, "quantization": {"enabled": False},
                "gated_mlp": {"methods": []},
            },
        },
    }
    data = [(torch.randn(3, 2, 8, 8), torch.tensor([0, 1, 0]))]
    report = analyze_model(model, workflow, level="calibrated", calibration_batches=data)
    candidate = next(item for item in report.candidates if item.method == "partial_tucker")
    assert candidate.status == "calibrated"
    assert candidate.calibration_evidence["sample_shape"] == [2, 2, 4, 4]


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        inputs = torch.nn.functional.one_hot(input_ids, 4).float()
        return SimpleNamespace(logits=self.projection(inputs))


def test_validated_language_candidate_reports_nll_perplexity_and_kl():
    model = TinyLanguageModel()
    workflow = {
        "task": "causal_lm", "model": {"name": "tiny-language"},
        "analysis": {
            "include": ["projection"], "max_samples": 8,
            "candidate_grid": {
                "linear": {"methods": ["svd"], "ranks": [1]},
                "quantization": {"enabled": False}, "gated_mlp": {"methods": []},
            },
        },
    }
    data = [{"input_ids": torch.tensor([[0, 1, 2, 3]]),
             "attention_mask": torch.ones(1, 4, dtype=torch.long)}]
    report = analyze_model(model, workflow, level="validated",
                           calibration_batches=data, validation_batches=data,
                           target_size_bytes=1, max_quality_loss=10)
    measured = next(item.full_model_metrics for item in report.candidates
                    if item.status == "validated")
    assert {"nll", "perplexity", "teacher_to_candidate_kl"} <= measured.keys()
    assert measured["teacher_to_candidate_kl"] >= 0


def test_pruned_mlp_marks_reconstructible_projections_not_custom_parent():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 6, bias=False)
            self.up_proj = torch.nn.Linear(4, 6, bias=False)
            self.down_proj = torch.nn.Linear(6, 4, bias=False)

    candidate, indices = _pruned_mlp_candidate(
        MLP(), {"width": 3, "selection": "weight", "seed": 0})
    assert len(indices) == 3
    assert not getattr(candidate, "_tn_replacement", False)
    assert all(getattr(getattr(candidate, name), "_tn_replacement", False)
               for name in ("gate_proj", "up_proj", "down_proj"))


def test_plan_application_rolls_back_after_late_installation_failure(monkeypatch):
    class TwoLayers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(4, 4)
            self.second = torch.nn.Linear(4, 4)
            self.config = SimpleNamespace(label="original")

        def forward(self, inputs):
            return self.second(self.first(inputs))

    workflow = {"task": "classification", "model": {"name": "two-layers"}, "analysis": {}}
    model = TwoLayers().train()
    model._tn_transformations = [{"method": "existing"}]
    model._tn_model_spec = {"source": "custom", "marker": [1, 2]}
    originals = (model.first, model.second)
    state = copy.deepcopy(model.state_dict())
    inputs = torch.randn(3, 4)
    expected = model(inputs).detach().clone()
    fingerprint = model_fingerprint(model, workflow["model"])

    def transformation(path, rank):
        configuration = {"rank": rank}
        return {
            "candidate_id": stable_candidate_id(fingerprint, path, "svd", configuration),
            "layer_path": path, "method": "svd", "configuration": configuration,
        }

    plan = {
        "schema_version": 1, "kind": "damage_aware_compression_plan",
        "model_fingerprint": fingerprint,
        "transformations": [transformation("first", 1), transformation("second", 1)],
    }

    real_set_submodule = analyzer_service.set_submodule_by_path

    def fail_on_second_install(target, path, replacement):
        if path == "second" and replacement is not originals[1]:
            target.eval()
            target.config.label = "changed"
            target.config.injected = True
            target._tn_transformations.append({"method": "injected"})
            target._tn_model_spec["marker"].append(3)
            raise RuntimeError("injected installation failure")
        real_set_submodule(target, path, replacement)

    monkeypatch.setattr(analyzer_service, "set_submodule_by_path", fail_on_second_install)
    with pytest.raises(RuntimeError, match="injected installation failure"):
        apply_compression_plan(model, plan, workflow)
    assert (model.first, model.second) == originals
    assert model.training and all(module.training for module in model.modules())
    assert model.config.label == "original" and not hasattr(model.config, "injected")
    assert model._tn_transformations == [{"method": "existing"}]
    assert model._tn_model_spec == {"source": "custom", "marker": [1, 2]}
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value)
    torch.testing.assert_close(model(inputs), expected)
