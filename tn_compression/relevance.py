"""Bounded, diagnostic layer relevance from labeled calibration batches.

These scores describe local loss sensitivity. They do not select compression
methods or predict the quality of a compressed model.
"""

from __future__ import annotations

from itertools import islice
import math

import torch
from torch import nn
from torch.nn import functional as F

from .tasks.language import model_inputs, shifted_targets
from .tasks.vision import evaluation_mode, task_loss


def _eligible_layers(model):
    layers = []
    seen_weights = set()
    for path, module in model.named_modules():
        weight = module._parameters.get("weight")
        supported = isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        # Hugging Face GPT-2 uses a transposed dense layer named Conv1D.
        supported |= type(module).__name__ == "Conv1D" and weight is not None
        if not path or not supported or weight is None or weight.ndim < 2 or not weight.is_floating_point():
            continue
        if id(weight) in seen_weights:
            continue
        seen_weights.add(id(weight))
        layers.append((path, module, weight))
    return layers


def _spread(layers, limit):
    """Include the depth range when a network has more layers than the budget."""
    if len(layers) <= limit:
        return layers
    if limit == 1:
        return [layers[0]]
    return [layers[index * (len(layers) - 1) // (limit - 1)] for index in range(limit)]


def _loss_for_batch(model, batch, task, device, *, binary, ignore_index):
    if task == "causal_lm":
        if not isinstance(batch, dict):
            raise ValueError("Causal LM relevance expects tokenized mapping batches.")
        targets, valid = shifted_targets(batch)
        if not valid.any():
            return None, 0
        outputs = model(**model_inputs(batch, device), use_cache=False)
        logits = outputs.logits[:, :-1].float()
        value = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.to(device).reshape(-1),
            ignore_index=-100,
        )
        return value, int(valid.sum().item())

    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Vision relevance expects (images, targets) batches.")
    images, targets = batch
    logits = model(images.to(device))
    if not isinstance(logits, torch.Tensor):
        raise ValueError("Vision relevance requires tensor logits.")
    targets = targets.to(device)
    if task == "segmentation":
        count = int((targets != ignore_index).sum().item())
        if not count:
            return None, 0
        value = task_loss(logits, targets, task=task, binary=binary, ignore_index=ignore_index)
    else:
        count = targets.numel()
        value = task_loss(logits, targets, task=task)
    return value, count


def analyze_relevance(model: nn.Module, batches, *, task: str, device="cpu",
                      max_batches: int = 4, max_layers: int = 64,
                      binary: bool = False, ignore_index: int = 255) -> dict:
    """Measure layer-local loss sensitivity without altering model state.

    Each score is averaged across at most ``max_batches`` valid batches.
    ``taylor_saliency`` is mean(abs(weight * d(loss)/d(weight))) over a layer's
    weight elements; ``gradient_rms`` is RMS(d(loss)/d(weight)). The loss is
    cross-entropy for classification/segmentation and masked next-token NLL
    for causal LMs. These are *diagnostics*, not quality guarantees.
    """
    if task not in {"classification", "segmentation", "causal_lm"}:
        raise ValueError("Relevance supports classification, segmentation and causal_lm.")
    if max_batches < 1 or max_layers < 1:
        raise ValueError("max_batches and max_layers must be positive.")

    all_layers = _eligible_layers(model)
    if not all_layers:
        raise ValueError("Model has no supported Linear or convolution layers.")
    layers = _spread(all_layers, max_layers)
    weights = [weight for _, _, weight in layers]
    original_requires_grad = [(weight, weight.requires_grad) for weight in weights]
    sums = [{"gradient_rms": 0.0, "taylor_saliency": 0.0, "observations": 0,
             "observed_units": 0}
            for _ in layers]
    valid_batches = units = 0

    try:
        # Enabling gradients for the selected weights also supports inference
        # models whose parameters were frozen by the caller. autograd.grad does
        # not write to parameter .grad fields.
        for weight in weights:
            weight.requires_grad_(True)
        with evaluation_mode(model), torch.enable_grad():
            for batch in islice(batches, max_batches):
                value, batch_units = _loss_for_batch(
                    model, batch, task, device, binary=binary,
                    ignore_index=ignore_index)
                if value is None:
                    continue
                if not torch.isfinite(value):
                    raise ValueError("Relevance loss is nonfinite.")
                gradients = torch.autograd.grad(value, weights, allow_unused=True)
                for (path, _, weight), gradient, row in zip(layers, gradients, sums):
                    if gradient is None:
                        continue
                    gradient = gradient.detach().float()
                    weight_value = weight.detach().float()
                    gradient_rms = gradient.square().mean().sqrt().item()
                    saliency = (weight_value * gradient).abs().mean().item()
                    if not math.isfinite(gradient_rms) or not math.isfinite(saliency):
                        raise ValueError(f"Nonfinite relevance score for {path}.")
                    row["gradient_rms"] += gradient_rms * batch_units
                    row["taylor_saliency"] += saliency * batch_units
                    row["observations"] += 1
                    row["observed_units"] += batch_units
                valid_batches += 1
                units += batch_units
    finally:
        for weight, requires_grad in original_requires_grad:
            weight.requires_grad_(requires_grad)

    if not valid_batches:
        raise ValueError("Relevance data has no valid labeled targets.")
    result_layers = []
    for (path, module, weight), sums_row in zip(layers, sums):
        observations = sums_row["observations"]
        if not observations:
            continue
        result_layers.append({
            "layer_path": path,
            "module_type": type(module).__name__,
            "parameters": weight.numel(),
            "gradient_rms": sums_row["gradient_rms"] / sums_row["observed_units"],
            "taylor_saliency": sums_row["taylor_saliency"] / sums_row["observed_units"],
            "observed_batches": observations,
        })
    if not result_layers:
        raise ValueError("No selected layers contributed to the loss.")
    maximum = max(row["taylor_saliency"] for row in result_layers)
    for row in result_layers:
        row["relative_relevance"] = row["taylor_saliency"] / maximum if maximum else 0.0
    return {
        "task": task,
        "metric": "mean_abs_weight_grad",
        "metric_label": "Mean |weight × loss gradient| per parameter",
        "gradient_metric": "RMS loss gradient per parameter",
        "batches": valid_batches,
        "units": units,
        "unit_label": "valid next-token targets" if task == "causal_lm" else (
            "valid labeled pixels" if task == "segmentation" else "labeled images"),
        "layers_available": len(all_layers),
        "layers_analyzed": len(layers),
        "layers": result_layers,
        "limitations": [
            "Local gradient scores are diagnostic; validate actual compressed candidates before recommending them.",
            "Scores depend on the supplied labeled batches and cannot be compared across tasks or models.",
            "Relative relevance is normalized only among the layers shown.",
        ],
    }
