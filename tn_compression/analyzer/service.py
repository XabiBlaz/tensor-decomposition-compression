"""Shared service layer for structural, calibrated and validated analysis."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from ..api import set_submodule_by_path
from ..calibration import candidate_intervention, collect_module_inputs, reconstruction_error
from ..compression_methods._compression_methods_utils import compress_layer
from ..decompositions.weighted_svd import weighted_svd
from ..pruning import select_channels, slice_linear
from ..quantization import round_to_nearest
from ..tasks.vision import evaluate_vision, evaluation_mode
from .candidates import generate_candidates, model_fingerprint, serialized_state_bytes, tensor_bytes
from .context import analysis_context
from .schema import AnalysisReport, CandidateResult, capability


LEVELS = {"structural", "calibrated", "validated"}


class TaskAdapter:
    """Small task contract used by analysis rather than CLI-specific logic."""

    def __init__(self, task: str, config: Mapping[str, Any], *, device="cpu", detection_coco=None):
        self.task = task
        self.config = config
        self.device = device
        self.detection_coco = detection_coco

    def forward(self, model: nn.Module, batch: Any):
        if self.task == "causal_lm":
            from ..tasks.language import model_inputs
            return model(**model_inputs(batch, self.device), use_cache=False)
        if self.task == "detection":
            images = batch[0]
            return model([image.to(self.device) for image in images])
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        return model(images.to(self.device))

    def input_mask(self, batch: Any):
        return batch.get("attention_mask") if self.task == "causal_lm" else None

    def evaluate(self, model: nn.Module, batches: Sequence[Any]) -> Dict[str, Any]:
        if self.task == "causal_lm":
            from ..tasks.language import evaluate_language
            return evaluate_language(model, batches, device=self.device)
        if self.task == "detection":
            if self.detection_coco is None:
                raise ValueError("Validated detection analysis requires the dataset COCO API.")
            from ..tasks.detection import evaluate_detection
            return evaluate_detection(model, batches, self.detection_coco, device=self.device,
                                      category_mapping=self.config.get("category_mapping"))
        return evaluate_vision(
            model, batches, task=self.task,
            num_classes=self.config.get("num_classes", 2),
            binary=self.config.get("binary", False), device=self.device)

    def quality_loss(self, baseline: Mapping[str, Any], measured: Mapping[str, Any]) -> float:
        if self.task == "causal_lm":
            return float(measured["nll"] - baseline["nll"])
        if self.task == "detection":
            return float(baseline["ap"] - measured["ap"])
        return float(measured["loss"] - baseline["loss"])

    def metric_delta(self, baseline: Mapping[str, Any], measured: Mapping[str, Any]) -> Dict[str, float]:
        delta = {}
        for name, value in measured.items():
            if name in baseline and isinstance(value, (int, float)) and not isinstance(value, bool):
                delta[name] = float(value - baseline[name])
        return delta

    @torch.no_grad()
    def measure_intervention(self, model: nn.Module, path: str, candidate: nn.Module,
                             batches: Sequence[Any]) -> Dict[str, Any]:
        if self.task != "causal_lm":
            with candidate_intervention(model, path, candidate):
                return self.evaluate(model, batches)
        return self._measure_language_intervention(model, path, candidate, batches)

    @torch.no_grad()
    def _measure_language_intervention(self, model, path, candidate, batches):
        """Stream teacher/student logits one batch at a time to bound memory."""
        from torch.nn import functional as functional
        from ..tasks.language import model_inputs, shifted_targets

        loss_sum = kl_sum = 0.0
        tokens = sequences = 0
        with evaluation_mode(model), evaluation_mode(candidate):
            for batch in batches:
                targets, valid = shifted_targets(batch)
                if not valid.any():
                    continue
                inputs = model_inputs(batch, self.device)
                teacher = model(**inputs, use_cache=False).logits[:, :-1].float()
                with candidate_intervention(model, path, candidate):
                    student = model(**inputs, use_cache=False).logits[:, :-1].float()
                selected_teacher = teacher[valid]
                selected_student = student[valid]
                loss_sum += functional.cross_entropy(
                    student.reshape(-1, student.shape[-1]), targets.to(self.device).reshape(-1),
                    ignore_index=-100, reduction="sum").item()
                for start in range(0, len(selected_teacher), 32):
                    teacher_log = selected_teacher[start:start + 32].log_softmax(-1)
                    student_log = selected_student[start:start + 32].log_softmax(-1)
                    kl_sum += (teacher_log.exp() * (teacher_log - student_log)).sum().item()
                tokens += valid.sum().item()
                sequences += len(batch["input_ids"])
        if not tokens:
            raise ValueError("Language validation has no valid next-token targets.")
        nll = loss_sum / tokens
        return {
            "task": "causal_lm", "nll": nll,
            "perplexity": math.exp(nll) if nll < 700 else None,
            "teacher_to_candidate_kl": kl_sum / tokens,
            "tokens": tokens, "sequences": sequences,
        }


def _pruned_mlp_candidate(mlp: nn.Module, configuration: Mapping[str, Any], activations=None) -> nn.Module:
    indices = configuration.get("retained_indices")
    if indices is None:
        indices = select_channels(
            mlp, configuration["width"], method=configuration.get("selection", "activation"),
            activations=activations, seed=configuration.get("seed", 0))
    result = copy.deepcopy(mlp)
    result.gate_proj = slice_linear(result.gate_proj, indices, 0)
    result.up_proj = slice_linear(result.up_proj, indices, 0)
    result.down_proj = slice_linear(result.down_proj, indices, 1)
    if hasattr(result, "intermediate_size"):
        result.intermediate_size = len(indices)
    return result, list(indices)


def materialize_candidate(model: nn.Module, candidate: CandidateResult, samples=None,
                          pruning_activations=None) -> nn.Module:
    layer = model.get_submodule(candidate.layer_path)
    configuration = candidate.configuration
    devices = sorted({parameter.device.index for parameter in layer.parameters() if parameter.is_cuda})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(configuration.get("seed", 0)))
        if devices:
            torch.cuda.manual_seed_all(int(configuration.get("seed", 0)))
        if candidate.method == "weighted_svd":
            if samples is None:
                raise ValueError("Weighted SVD requires calibration inputs.")
            result = weighted_svd(layer, samples, configuration["rank"])
        elif candidate.method == "round_to_nearest":
            result = round_to_nearest(layer, bits=configuration["bits"],
                                      group_size=configuration["group_size"])
        elif candidate.method == "gated_mlp_pruning":
            result, indices = _pruned_mlp_candidate(layer, configuration, pruning_activations)
            candidate.configuration["retained_indices"] = indices
            candidate.refresh_candidate_id()
        else:
            policy = {"type": candidate.method, **configuration}
            result = compress_layer(layer, policy)
    if result is None:
        raise ValueError("Candidate materialization returned no replacement.")
    if candidate.method != "gated_mlp_pruning":
        result._tn_replacement = True
    return result


def pareto_candidates(candidates: Iterable[CandidateResult]) -> list[CandidateResult]:
    """Filter by local error within the same layer, never across layers."""
    grouped = defaultdict(list)
    for candidate in candidates:
        if (candidate.structurally_eligible and candidate.normalized_local_error is not None
                and candidate.estimated_bytes_saved is not None and candidate.estimated_bytes_saved > 0
                and candidate.backend_support.get("supported")):
            grouped[candidate.layer_path].append(candidate)
    frontier = []
    for rows in grouped.values():
        for row in rows:
            dominated = any(
                other.estimated_bytes_saved >= row.estimated_bytes_saved
                and other.normalized_local_error <= row.normalized_local_error
                and (other.estimated_bytes_saved > row.estimated_bytes_saved
                     or other.normalized_local_error < row.normalized_local_error)
                for other in rows if other is not row)
            if dominated:
                row.decision = "rejected"
                row.decision_reason = "dominated_within_layer"
            else:
                frontier.append(row)
    return frontier


def _calibration_samples(model, candidate, batches, adapter, options, cache):
    key = (candidate.layer_path, "module_inputs")
    if key not in cache:
        module = model.get_submodule(candidate.layer_path)
        mode = "feature_maps" if type(module) is nn.Conv2d else "rows"
        cache[key] = collect_module_inputs(
            model, candidate.layer_path, batches, adapter.forward,
            max_samples=options.get("max_samples", options.get("max_rows", 256)),
            max_spatial_size=options.get("max_spatial_size", 32),
            seed=options.get("seed", 0), input_mask=adapter.input_mask,
            sample_mode=mode)
    pruning = None
    if candidate.method == "gated_mlp_pruning" and candidate.configuration.get("selection") == "activation":
        key = (candidate.layer_path, "down_inputs")
        if key not in cache:
            cache[key] = collect_module_inputs(
                model, candidate.layer_path + ".down_proj", batches, adapter.forward,
                max_samples=options.get("max_samples", options.get("max_rows", 256)),
                seed=options.get("seed", 0), input_mask=adapter.input_mask, sample_mode="rows")
        pruning = cache[key]
    return cache[(candidate.layer_path, "module_inputs")], pruning


def _evidence(samples, options, identifiers, preprocessing):
    return {
        "identifiers": list(identifiers or []),
        "sample_count": len(samples),
        "sample_shape": list(samples.shape),
        "seed": int(options.get("seed", 0)),
        "max_samples": int(options.get("max_samples", options.get("max_rows", 256))),
        "preprocessing": copy.deepcopy(preprocessing or {}),
    }


def _score_candidates(model, candidates, batches, adapter, options, identifiers, preprocessing):
    cache = {}
    for candidate in candidates:
        if candidate.decision == "rejected" or not candidate.structurally_eligible:
            continue
        try:
            samples, pruning = _calibration_samples(model, candidate, batches, adapter, options, cache)
            replacement = materialize_candidate(model, candidate, samples, pruning)
            measured = reconstruction_error(model.get_submodule(candidate.layer_path), replacement, samples)
            candidate.local_error = measured
            candidate.normalized_local_error = measured["relative_squared_error"]
            candidate.measured_artifact_bytes = serialized_state_bytes(replacement)
            candidate.calibration_evidence = _evidence(samples, options, identifiers, preprocessing)
            candidate.status = "calibrated"
            candidate.decision = "candidate"
            candidate.decision_reason = None
        except (RuntimeError, ValueError) as error:
            candidate.status = "calibration_failed"
            candidate.decision = "rejected"
            candidate.decision_reason = f"candidate_materialization_failed: {error}"
    return cache


def _validate_frontier(model, frontier, validation_batches, adapter, baseline, cache, options):
    per_layer = defaultdict(list)
    for candidate in frontier:
        per_layer[candidate.layer_path].append(candidate)
    selected = []
    limit = int(options.get("max_validated_per_layer", 3))
    for rows in per_layer.values():
        rows.sort(key=lambda row: (row.normalized_local_error, -row.estimated_bytes_saved,
                                  row.method, row.candidate_id))
        selected.extend(rows[:limit])
        for row in rows[limit:]:
            row.decision, row.decision_reason = "rejected", "validation_shortlist_limit"
    for candidate in selected:
        samples = cache[(candidate.layer_path, "module_inputs")]
        pruning = cache.get((candidate.layer_path, "down_inputs"))
        try:
            replacement = materialize_candidate(model, candidate, samples, pruning)
            measured = adapter.measure_intervention(model, candidate.layer_path, replacement, validation_batches)
            candidate.full_model_metrics = measured
            candidate.full_model_metric_delta = adapter.metric_delta(baseline, measured)
            candidate.quality_loss = adapter.quality_loss(baseline, measured)
            candidate.status = "validated"
            candidate.decision = "validated"
            candidate.decision_reason = None
        except (RuntimeError, ValueError) as error:
            candidate.status = "validation_failed"
            candidate.decision = "rejected"
            candidate.decision_reason = f"candidate_validation_failed: {error}"
    return selected


def _select_candidates(model, candidates, validation_batches, adapter, baseline, cache,
                       target_bytes, max_quality_loss):
    current_bytes = tensor_bytes(model)
    if target_bytes is None or max_quality_loss is None:
        raise ValueError("Validated selection requires target_size_bytes and max_quality_loss.")
    if target_bytes <= 0 or max_quality_loss < 0:
        raise ValueError("Use a positive target size and nonnegative maximum quality loss.")
    viable = [candidate for candidate in candidates
              if candidate.status == "validated" and candidate.quality_loss is not None
              and math.isfinite(candidate.quality_loss) and candidate.estimated_bytes_saved > 0]
    viable.sort(key=lambda row: (
        row.quality_loss / row.estimated_bytes_saved,
        row.quality_loss, -row.estimated_bytes_saved, row.layer_path, row.candidate_id))
    originals = {}
    accepted = []
    cumulative = baseline
    transformations = copy.deepcopy(getattr(model, "_tn_transformations", []))
    try:
        for candidate in viable:
            if current_bytes <= target_bytes:
                break
            if any(candidate.layer_path == path or candidate.layer_path.startswith(path + ".")
                   or path.startswith(candidate.layer_path + ".") for path in originals):
                candidate.decision = "rejected"
                candidate.decision_reason = "overlaps_selected_transformation"
                continue
            samples = cache[(candidate.layer_path, "module_inputs")]
            pruning = cache.get((candidate.layer_path, "down_inputs"))
            try:
                replacement = materialize_candidate(model, candidate, samples, pruning)
                measured = adapter.measure_intervention(
                    model, candidate.layer_path, replacement, validation_batches)
            except (RuntimeError, ValueError) as error:
                candidate.decision = "rejected"
                candidate.decision_reason = f"cumulative_materialization_failed: {error}"
                continue
            quality_loss = adapter.quality_loss(baseline, measured)
            if quality_loss > max_quality_loss:
                candidate.decision = "rejected"
                candidate.decision_reason = "cumulative_quality_constraint"
                candidate.full_model_metrics = measured
                candidate.full_model_metric_delta = adapter.metric_delta(baseline, measured)
                candidate.quality_loss = quality_loss
                continue
            original = model.get_submodule(candidate.layer_path)
            originals[candidate.layer_path] = original
            set_submodule_by_path(model, candidate.layer_path, replacement)
            current_bytes = tensor_bytes(model)
            cumulative = measured
            candidate.decision = "accepted"
            candidate.decision_reason = "greedy_measured_damage_per_byte"
            candidate.full_model_metrics = measured
            candidate.full_model_metric_delta = adapter.metric_delta(baseline, measured)
            candidate.quality_loss = quality_loss
            accepted.append(candidate)
        return accepted, cumulative, current_bytes
    finally:
        for path, original in originals.items():
            set_submodule_by_path(model, path, original)
        model._tn_transformations = transformations


def _plan(report, original_bytes, final_bytes):
    accepted = [candidate for candidate in report.candidates if candidate.decision == "accepted"]
    target_bytes = report.constraints.get("target_size_bytes")
    return {
        "schema_version": 1,
        "kind": "damage_aware_compression_plan",
        "model_fingerprint": report.model_fingerprint,
        "task": report.task,
        "requested_backend": report.requested_backend,
        "analysis_context_fingerprint": report.analysis_context_fingerprint,
        "analysis_context": copy.deepcopy(report.analysis_context),
        "original_tensor_bytes": original_bytes,
        "estimated_final_tensor_bytes": final_bytes,
        "status": ("not_selected" if target_bytes is None else
                   "feasible" if final_bytes <= target_bytes else "infeasible"),
        "policy": "greedy measured damage per byte with cumulative validation and rollback",
        "transformations": [
            {
                "candidate_id": candidate.candidate_id,
                "layer_path": candidate.layer_path,
                "method": candidate.method,
                "configuration": copy.deepcopy(candidate.configuration),
                "checkpoint_reconstruction": copy.deepcopy(candidate.checkpoint_reconstruction),
                "backend_support": copy.deepcopy(candidate.backend_support),
            }
            for candidate in accepted
        ],
    }


def _summary(candidates):
    return {
        "candidates": len(candidates),
        "eligible": sum(candidate.structurally_eligible for candidate in candidates),
        "protected": sum(candidate.protected for candidate in candidates),
        "unsupported": sum(candidate.method == "unsupported" for candidate in candidates),
        "calibrated": sum(candidate.status in {"calibrated", "validated"} for candidate in candidates),
        "validated": sum(candidate.status == "validated" for candidate in candidates),
        "accepted": sum(candidate.decision == "accepted" for candidate in candidates),
        "rejected": sum(candidate.decision == "rejected" for candidate in candidates),
    }


def analyze_model(model: nn.Module, config: Mapping[str, Any], *, level="structural",
                  calibration_batches=None, validation_batches=None, calibration_ids=None,
                  target_size_bytes=None, max_quality_loss=None, requested_backend="pytorch",
                  device="cpu", detection_coco=None) -> AnalysisReport:
    """Analyze transformation-specific evidence without changing the supplied model."""
    if level not in LEVELS:
        raise ValueError(f"Analysis level must be one of {sorted(LEVELS)}.")
    analysis = dict(config.get("analysis", {}))
    analysis.setdefault("seed", config.get("seed", 0))
    fingerprint = model_fingerprint(model, analysis.get("model_identity", config.get("model", {})))
    candidates = generate_candidates(model, analysis, requested_backend=requested_backend,
                                     fingerprint=fingerprint)
    report = AnalysisReport(
        level=level, model_fingerprint=fingerprint,
        task=config.get("task", "classification"), requested_backend=requested_backend,
        candidates=candidates,
        capabilities={
            "model_loading": capability(True, "caller supplied a loaded torch.nn.Module", verified=True),
            "compression_method": capability(True, "candidate-specific capability recorded per row", verified=True),
            "checkpoint_reconstruction": capability(True, "candidate-specific bundle support recorded per row"),
            "inference_backend": capability(True, "candidate-specific backend support recorded per row"),
        },
        constraints={"target_size_bytes": target_size_bytes, "max_quality_loss": max_quality_loss},
        limitations=[
            "Tensor-byte estimates exclude checkpoint container headers.",
            "Greedy selection does not guarantee a globally optimal allocation.",
        ],
    )
    original_bytes = tensor_bytes(model)
    if level == "structural":
        context = analysis_context(fingerprint, config)
        report.analysis_context_fingerprint = context["fingerprint"]
        report.analysis_context = dict(context["evidence"])
        report.summary = _summary(candidates)
        report.compression_plan = _plan(report, original_bytes, original_bytes)
        return report
    if calibration_batches is None:
        raise ValueError(f"{level} analysis requires calibration_batches.")
    calibration_batches = list(calibration_batches)
    if not calibration_batches:
        raise ValueError("Calibration data is empty.")
    context = analysis_context(
        fingerprint, config, calibration_batches=calibration_batches,
        calibration_ids=calibration_ids)
    report.analysis_context_fingerprint = context["fingerprint"]
    report.analysis_context = dict(context["evidence"])
    adapter = TaskAdapter(report.task, config, device=device, detection_coco=detection_coco)
    cache = _score_candidates(
        model, candidates, calibration_batches, adapter, analysis,
        calibration_ids, config.get("preprocessing", config.get("data", {})))
    report.calibration = {
        "identifiers": list(calibration_ids or []), "batches": len(calibration_batches),
        "seed": analysis.get("seed", 0),
        "preprocessing": copy.deepcopy(config.get("preprocessing", config.get("data", {}))),
    }
    frontier = pareto_candidates(candidates)
    if level == "calibrated":
        report.summary = _summary(candidates)
        report.compression_plan = _plan(report, original_bytes, original_bytes)
        return report
    if validation_batches is None:
        raise ValueError("Validated analysis requires validation_batches.")
    validation_batches = list(validation_batches)
    if not validation_batches:
        raise ValueError("Validation data is empty.")
    baseline = adapter.evaluate(model, validation_batches)
    report.baseline_metrics = baseline
    shortlisted = _validate_frontier(model, frontier, validation_batches, adapter, baseline, cache, analysis)
    accepted, cumulative, final_bytes = _select_candidates(
        model, shortlisted, validation_batches, adapter, baseline, cache,
        target_size_bytes, max_quality_loss)
    report.cumulative_metrics = cumulative
    report.summary = _summary(candidates)
    report.summary.update({
        "original_tensor_bytes": original_bytes,
        "estimated_final_tensor_bytes": final_bytes,
        "target_reached": final_bytes <= target_size_bytes,
    })
    report.compression_plan = _plan(report, original_bytes, final_bytes)
    return report


def apply_compression_plan(model: nn.Module, plan: Mapping[str, Any], config: Mapping[str, Any],
                           *, calibration_batches=None, calibration_ids=None, device="cpu") -> Dict[str, Any]:
    """Apply a resolved analyzer plan; weighted SVD recollects bounded inputs."""
    if plan.get("kind") != "damage_aware_compression_plan" or plan.get("schema_version") != 1:
        raise ValueError("Unsupported analyzer compression plan.")
    fingerprint = model_fingerprint(model, config.get("analysis", {}).get("model_identity", config.get("model", {})))
    if fingerprint != plan.get("model_fingerprint"):
        raise ValueError("Compression plan model fingerprint does not match the loaded model.")
    transformations = list(plan.get("transformations", []))
    if not transformations:
        raise ValueError("Compression plan contains no accepted transformations.")
    batches = list(calibration_batches or [])
    requires_calibration = any(item.get("method") == "weighted_svd" for item in transformations)
    if requires_calibration:
        if not batches:
            raise ValueError("Applying this plan requires its original calibration data.")
        actual_context = analysis_context(
            fingerprint, config, calibration_batches=batches, calibration_ids=calibration_ids)
        expected_context = plan.get("analysis_context_fingerprint")
        if not expected_context:
            raise ValueError("Calibration-dependent plan has no analysis-context fingerprint.")
        if actual_context["fingerprint"] != expected_context:
            raise ValueError(
                "Analysis-context fingerprint mismatch: model, calibration data, preprocessing, "
                "tokenizer, seed or analyzer configuration changed.")
    adapter = TaskAdapter(config.get("task", "classification"), config, device=device)
    options = {"seed": config.get("seed", 0), **config.get("analysis", {})}
    applied = []
    for transformation in transformations:
        candidate = CandidateResult(
            model_fingerprint=fingerprint, layer_path=transformation["layer_path"],
            module_type=type(model.get_submodule(transformation["layer_path"])).__name__,
            method=transformation["method"], configuration=copy.deepcopy(transformation["configuration"]),
            structurally_eligible=True,
            original_parameters=sum(parameter.numel() for parameter in model.get_submodule(
                transformation["layer_path"]).parameters()),
            candidate_parameters=None, estimated_artifact_bytes=None,
        )
        if candidate.candidate_id != transformation.get("candidate_id"):
            raise ValueError(
                f"Plan candidate ID does not match resolved configuration for {candidate.layer_path}.")
        samples = pruning = None
        if candidate.method == "weighted_svd":
            samples = collect_module_inputs(
                model, candidate.layer_path, batches, adapter.forward,
                max_samples=options.get("max_samples", options.get("max_rows", 256)),
                seed=options.get("seed", 0), input_mask=adapter.input_mask, sample_mode="rows")
        replacement = materialize_candidate(model, candidate, samples, pruning)
        set_submodule_by_path(model, candidate.layer_path, replacement)
        applied.append(transformation)
    model._tn_transformations = [*getattr(model, "_tn_transformations", []), *copy.deepcopy(applied)]
    return {"applied": len(applied), "transformations": applied,
            "model_tensor_bytes": tensor_bytes(model), "plan_model_fingerprint": fingerprint}
