"""Shared vision training, recovery and metrics with task-specific contracts."""

from contextlib import contextmanager
import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@contextmanager
def evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def task_loss(logits, targets, *, task, binary=False, ignore_index=255,
              loss="cross_entropy", alpha=0.3, beta=0.7, gamma=4 / 3):
    if task == "classification":
        return F.cross_entropy(logits, targets)
    if task != "segmentation":
        raise ValueError("Training supports classification and segmentation; detector training is deferred.")
    valid = targets != ignore_index
    if not valid.any():
        raise ValueError("The batch contains no valid target elements.")
    if binary:
        if logits.shape[1] != 1:
            raise ValueError("Binary segmentation requires one output logit channel.")
        targets_float = targets.masked_fill(~valid, 0).float()
        probabilities = torch.sigmoid(logits[:, 0])
        if loss == "cross_entropy":
            return F.binary_cross_entropy_with_logits(logits[:, 0][valid], targets_float[valid])
        probabilities, truth = probabilities[:, None], targets_float[:, None]
    else:
        if loss == "cross_entropy":
            return F.cross_entropy(logits, targets, ignore_index=ignore_index)
        probabilities = logits.softmax(1)
        truth = F.one_hot(targets.masked_fill(~valid, 0), logits.shape[1]).permute(0, 3, 1, 2)
    if loss not in {"dice", "focal_tversky"}:
        raise ValueError(f"Unknown segmentation loss: {loss}")
    probabilities, truth = probabilities * valid[:, None], truth * valid[:, None]
    axes = (0, 2, 3)
    true_positive = (probabilities * truth).sum(axes)
    false_positive = (probabilities * (1 - truth) * valid[:, None]).sum(axes)
    false_negative = ((1 - probabilities) * truth).sum(axes)
    # alpha weights false positives; beta weights false negatives. State this
    # convention explicitly because the source variants used different defaults.
    if loss == "dice":
        alpha = beta = 0.5
        gamma = 1.0
    tversky = (true_positive + 1e-7) / (true_positive + alpha * false_positive + beta * false_negative + 1e-7)
    return ((1 - tversky) ** gamma).mean()


def segmentation_metrics(confusion):
    confusion = confusion.double()
    intersection = confusion.diag()
    target, predicted = confusion.sum(1), confusion.sum(0)
    union = target + predicted - intersection
    present = union > 0
    iou = intersection / union.clamp_min(1)
    dice = 2 * intersection / (target + predicted).clamp_min(1)
    return {
        "mean_iou": iou[present].mean().item() if present.any() else None,
        "mean_dice": dice[present].mean().item() if present.any() else None,
        "per_class_iou": [value.item() if seen else None for value, seen in zip(iou, present)],
        "per_class_dice": [value.item() if seen else None for value, seen in zip(dice, present)],
        "confusion": confusion.long().tolist(),
    }


@torch.no_grad()
def evaluate_vision(model, batches, *, task, num_classes, device="cpu", binary=False, ignore_index=255):
    count = correct = valid_count = 0
    loss_sum = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with evaluation_mode(model):
        for images, targets in batches:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            if not isinstance(logits, torch.Tensor):
                raise ValueError("This evaluator expects tensor logits; provide an adapter for the output contract.")
            if task == "classification":
                loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
                correct += (logits.argmax(1) == targets).sum().item()
                valid_count += targets.numel()
            else:
                valid = targets != ignore_index
                if ((targets[valid] < 0) | (targets[valid] >= num_classes)).any():
                    raise ValueError("Segmentation labels are outside the configured class mapping.")
                if valid.any():
                    loss_sum += task_loss(logits, targets, task=task, binary=binary,
                                          ignore_index=ignore_index).item() * valid.sum().item()
                predictions = (logits[:, 0] >= 0).long() if binary else logits.argmax(1)
                cells = num_classes * targets[valid] + predictions[valid]
                confusion += torch.bincount(cells.cpu(), minlength=num_classes ** 2).reshape(num_classes, num_classes)
                valid_count += valid.sum().item()
            count += len(images)
    if not valid_count:
        raise ValueError("Evaluation has no valid targets.")
    result = {"task": task, "examples": count, "loss": loss_sum / valid_count}
    result.update({"top1": correct / valid_count} if task == "classification" else segmentation_metrics(confusion))
    return result


def configure_trainability(model, policy="all", layers=None):
    if policy not in {"all", "decoder", "layers"}:
        raise ValueError("Recovery policy must be all, decoder or layers.")
    if policy == "decoder":
        layers = [name for name in ("decoder", "segmentation_head", "classification_head") if hasattr(model, name)]
        if not layers or not hasattr(model, "encoder"):
            raise ValueError("Decoder recovery requires an encoder/decoder task adapter.")
    if policy == "layers":
        if not layers:
            raise ValueError("Explicit-layer recovery needs at least one layer path.")
        for name in layers:
            model.get_submodule(name)
    for name, parameter in model.named_parameters():
        enabled = policy == "all" or any(name.startswith(path + ".") for path in layers or [])
        parameter.requires_grad_(enabled)
        parameter.grad = None
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Recovery selected no trainable parameters.")
    return trainable


def train_vision(model, train_batches, validation_batches, *, task, num_classes,
                 epochs=1, learning_rate=1e-3, weight_decay=1e-4, device="cpu",
                 policy="all", layers=None, binary=False, loss="cross_entropy"):
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    model.to(device)
    parameters = configure_trainability(model, policy, layers)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    history, best_state, best_score = [], None, -float("inf")
    for epoch in range(epochs):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):
                module.eval()
        steps = 0
        for images, targets in train_batches:
            optimizer.zero_grad(set_to_none=True)
            value = task_loss(model(images.to(device)), targets.to(device), task=task, binary=binary, loss=loss)
            if not torch.isfinite(value):
                raise RuntimeError("Non-finite training loss.")
            value.backward()
            optimizer.step()
            steps += 1
        if not steps:
            raise ValueError("Training data is empty.")
        metrics = evaluate_vision(model, validation_batches, task=task, num_classes=num_classes,
                                   device=device, binary=binary)
        history.append({"epoch": epoch + 1, **metrics})
        logger.info("Epoch %s/%s validation: %s", epoch + 1, epochs, metrics)
        score = metrics["top1"] if task == "classification" else metrics["mean_iou"]
        if score > best_score:
            best_score = score
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return history
