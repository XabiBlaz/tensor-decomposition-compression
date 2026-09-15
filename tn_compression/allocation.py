"""Greedy byte allocation with measured cumulative language-model damage."""

import math
import time

from .calibration import collect_linear_inputs
from .decompositions.cp import cp_linear
from .decompositions.weighted_svd import weighted_svd
from .language_experiments import evaluate_candidates, tensor_bytes
from .tasks.language import evaluate_language, model_inputs


def candidate_frontier(rows):
    """Remove non-saving, nonfinite and dominated alternatives within each layer."""
    rows = [row for row in rows if row["bytes_saved"] > 0 and math.isfinite(row["delta_nll"])]
    return [row for row in rows if not any(
        other["path"] == row["path"] and other["bytes_saved"] >= row["bytes_saved"]
        and other["delta_nll"] <= row["delta_nll"]
        and (other["bytes_saved"] > row["bytes_saved"] or other["delta_nll"] < row["delta_nll"])
        for other in rows)]


def allocate_ranks(model, calibration_batches, validation_batches, layers, *, target_bytes,
                   max_delta_nll, method="svd", device="cpu", max_rows=256, seed=0):
    """Accept one layer at a time, refreshing all remaining candidate measurements.

    Each layer is replaced at most once. This bounded greedy search is not an
    optimizer over all rank combinations. An infeasible result retains the best
    validated partial allocation and explicitly reports the missed byte target.
    """
    if not isinstance(target_bytes, int) or target_bytes <= 0 or not math.isfinite(max_delta_nll) or max_delta_nll < 0:
        raise ValueError("Use positive integer target_bytes and finite nonnegative max_delta_nll.")
    started = time.perf_counter()
    baseline = evaluate_language(model, validation_batches, device=device)
    original_bytes = tensor_bytes(model)
    remaining = dict(layers)
    rounds = []
    current = baseline
    while tensor_bytes(model) > target_bytes and remaining:
        measured = evaluate_candidates(model, calibration_batches, validation_batches, remaining,
                    method=method, device=device, max_rows=max_rows, seed=seed)
        frontier = candidate_frontier(measured["candidates"])
        feasible = [row for row in frontier if row["nll"] - baseline["nll"] <= max_delta_nll]
        if not feasible:
            rounds.append({"status": "quality_constraint", "search": measured})
            break
        choice = min(feasible, key=lambda row: (row["delta_nll"] / row["bytes_saved"], row["path"], -row["rank"]))
        path = choice["path"]
        original = model.get_submodule(path)
        if method == "svd":
            candidate = cp_linear(original, choice["rank"])
        else:
            samples = collect_linear_inputs(model, path, calibration_batches,
                lambda net, batch: net(**model_inputs(batch, device), use_cache=False),
                max_rows=max_rows, seed=seed, input_mask=lambda batch: batch.get("attention_mask"))
            candidate = weighted_svd(original, samples, choice["rank"])
        parent, _, name = path.rpartition(".")
        parent = model.get_submodule(parent) if parent else model
        parent._modules[name] = candidate
        accepted = False
        try:
            cumulative = evaluate_language(model, validation_batches, device=device)
            accepted = cumulative["nll"] - baseline["nll"] <= max_delta_nll
        finally:
            if not accepted:
                parent._modules[name] = original
        rounds.append({"status": "accepted" if accepted else "rolled_back", "choice": choice,
                       "observed": cumulative, "previous_nll": current["nll"], "search": measured})
        if accepted:
            candidate._tn_replacement = True
            model._tn_transformations = [*getattr(model, "_tn_transformations", []),
                                         {"method": method, "path": path, "rank": choice["rank"]}]
            current = cumulative
        remaining.pop(path)
    size = tensor_bytes(model)
    return {"status": "feasible" if size <= target_bytes else "infeasible", "baseline": baseline,
            "final": current, "original_tensor_bytes": original_bytes, "model_tensor_bytes": size,
            "target_bytes": target_bytes, "max_delta_nll": max_delta_nll, "rounds": rounds,
            "search_seconds": time.perf_counter() - started,
            "policy": "greedy measured marginal NLL per byte; remeasure after every accepted layer"}
