"""Small, reproducible language-model interventions over shared primitives."""

import time
import copy

import torch

from .api import protected_module_reasons
from .calibration import candidate_intervention, collect_linear_inputs, reconstruction_error
from .decompositions.cp import cp_linear
from .decompositions.weighted_svd import weighted_svd
from .tasks.language import evaluate_language, model_inputs


def tensor_bytes(module):
    """Resident tensor payload, counting tied parameters once; excludes file headers."""
    tensors = list(module.parameters()) + list(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in {id(t): t for t in tensors}.values())


def linear_candidates(model, layers):
    reasons = protected_module_reasons(model)
    for path, ranks in layers.items():
        layer = model.get_submodule(path)
        reason = reasons.get(path.replace(".", "/"))
        if type(layer) is not torch.nn.Linear or reason:
            raise ValueError(f"{path} cannot use linear factorization: {reason or 'requires nn.Linear'}.")
        if not ranks or any(isinstance(rank, bool) or not isinstance(rank, int)
                            or not 0 < rank <= min(layer.weight.shape) for rank in ranks):
            raise ValueError(f"{path} needs integer ranks within its matrix dimensions.")
        yield path, layer, sorted(set(ranks), reverse=True)


def evaluate_candidates(model, calibration_batches, validation_batches, layers, *, device="cpu",
                        max_rows=256, seed=0, method="svd"):
    """Measure isolated SVD candidates; no final-test data enters selection."""
    candidates = list(linear_candidates(model, layers))
    if method not in {"svd", "weighted_svd"}:
        raise ValueError("Candidate method must be svd or weighted_svd.")
    started = time.perf_counter()
    baseline = evaluate_language(model, validation_batches, device=device)
    original_bytes = tensor_bytes(model)
    rows = []
    for path, layer, ranks in candidates:
        samples = collect_linear_inputs(model, path, calibration_batches,
                    lambda net, batch: net(**model_inputs(batch, device), use_cache=False),
                    max_rows=max_rows, seed=seed, input_mask=lambda batch: batch.get("attention_mask"))
        layer_bytes = tensor_bytes(layer)
        rows.append({"path": path, "method": "unchanged", "rank": None, "bytes_saved": 0,
                     "model_tensor_bytes": original_bytes, "delta_nll": 0.0,
                     "relative_squared_error": 0.0, "nll": baseline["nll"]})
        for rank in ranks:
            trial_started = time.perf_counter()
            candidate = cp_linear(layer, rank) if method == "svd" else weighted_svd(layer, samples, rank)
            error = reconstruction_error(layer, candidate, samples)
            with candidate_intervention(model, path, candidate):
                measured = evaluate_language(model, validation_batches, device=device)
            saved = layer_bytes - tensor_bytes(candidate)
            rows.append({"path": path, "method": method, "rank": rank, **error,
                         "bytes_saved": saved, "model_tensor_bytes": original_bytes - saved,
                         "delta_nll": measured["nll"] - baseline["nll"], **measured,
                         "calibration_rows": len(samples), "seconds": time.perf_counter() - trial_started,
                         "backend": "pytorch", "vllm": "unsupported_factorized_representation"})
    return {"schema_version": 1, "baseline": baseline, "model_tensor_bytes": original_bytes,
            "candidates": rows, "search_seconds": time.perf_counter() - started,
            "scope": "isolated interventions; joint damage and inference speed are not inferred"}


def prune_language(model, calibration_batches, validation_batches, *, width, method="weight",
                   max_delta_nll=0.1, device="cpu", max_rows=256, seed=0):
    """Validate a uniform-width intervention and roll back rejected surgery."""
    from .pruning import gated_mlps, prune_gated_mlps, select_channels
    if max_delta_nll < 0:
        raise ValueError("max_delta_nll must be nonnegative.")
    groups = gated_mlps(model)
    original_bytes = tensor_bytes(model)
    baseline = evaluate_language(model, validation_batches, device=device)
    started = time.perf_counter()
    selections = {}
    for path, mlp in groups.items():
        activations = None
        if method == "activation":
            activations = collect_linear_inputs(model, path + ".down_proj", calibration_batches,
                lambda net, batch: net(**model_inputs(batch, device), use_cache=False),
                max_rows=max_rows, seed=seed, input_mask=lambda batch: batch.get("attention_mask"))
        selections[path] = select_channels(mlp, width, method=method, activations=activations, seed=seed)
    original_projections = {path: (mlp.gate_proj, mlp.up_proj, mlp.down_proj) for path, mlp in groups.items()}
    old_width = model.config.intermediate_size
    old_spec = copy.deepcopy(getattr(model, "_tn_model_spec", None))
    old_transformations = copy.deepcopy(getattr(model, "_tn_transformations", []))
    accepted = False
    try:
        transformation = prune_gated_mlps(model, selections)
        measured = evaluate_language(model, validation_batches, device=device)
        size = tensor_bytes(model)
        accepted = size < original_bytes and measured["nll"] - baseline["nll"] <= max_delta_nll
        return {"status": "accepted" if accepted else "rejected", "baseline": baseline,
                "measured": measured, "delta_nll": measured["nll"] - baseline["nll"],
                "model_tensor_bytes": size, "bytes_saved": original_bytes - size,
                "selection": method, "transformation": transformation,
                "search_seconds": time.perf_counter() - started}
    finally:
        if not accepted:
            for path, layers in original_projections.items():
                groups[path].gate_proj, groups[path].up_proj, groups[path].down_proj = layers
                if hasattr(groups[path], "intermediate_size"):
                    groups[path].intermediate_size = old_width
            model.config.intermediate_size = old_width
            if old_spec is not None:
                model._tn_model_spec = old_spec
            model._tn_transformations = old_transformations
