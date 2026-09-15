"""Explicit, token-budgeted recovery for causal language models."""

import time

import torch
from torch.nn import functional as F

from .language import model_inputs, shifted_targets


def recover_language(model, batches, *, token_budget, max_updates, policy="explicit", layers=(),
                     learning_rate=1e-5, device="cpu"):
    """Optimize selected parameters; count actual supervised next-token targets."""
    if token_budget < 1 or max_updates < 1 or learning_rate <= 0:
        raise ValueError("Recovery requires positive token/update budgets and learning rate.")
    if policy not in {"full", "explicit"} or (policy == "explicit" and not layers):
        raise ValueError("Choose full recovery or explicit nonempty layer paths.")
    for path in layers:
        if not list(model.get_submodule(path).parameters()):
            raise ValueError(f"Recovery layer has no parameters: {path}")
    selected = []
    for name, parameter in model.named_parameters():
        trainable = policy == "full" or any(name == path or name.startswith(path + ".") for path in layers)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append((name, parameter))
    if not selected:
        raise ValueError("No parameters selected for recovery.")
    # The optimizer must see the final architecture and freezing policy.
    optimizer = torch.optim.AdamW([parameter for _, parameter in selected], lr=learning_rate)
    modes = [(module, module.training) for module in model.modules()]
    model.train() if policy == "full" else model.eval()
    if policy == "explicit":
        for path in layers:
            model.get_submodule(path).train()
    started = time.perf_counter()
    tokens = updates = 0
    loss_sum = 0.0
    try:
        while tokens < token_budget and updates < max_updates:
            previous_tokens = tokens
            for batch in batches:
                targets, valid = shifted_targets(batch)
                positions = valid.reshape(-1).nonzero().flatten()
                count = min(len(positions), token_budget - tokens)
                if not count:
                    continue
                # A partially consumed final batch contributes exactly the
                # remaining token budget, even though its full forward is run.
                targets.reshape(-1)[positions[count:]] = -100
                optimizer.zero_grad(set_to_none=True)
                logits = model(**model_inputs(batch, device), use_cache=False).logits[:, :-1].float()
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(device).reshape(-1),
                                       ignore_index=-100, reduction="sum")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite recovery loss.")
                (loss / count).backward()
                optimizer.step()
                loss_sum += loss.detach().item()
                tokens += count
                updates += 1
                if tokens >= token_budget or updates >= max_updates:
                    break
            if tokens == previous_tokens:
                raise ValueError("Recovery has no valid targets; batches must also be re-iterable.")
    finally:
        for module, training in modes:
            module.training = training
    return {"tokens": tokens, "updates": updates, "token_budget": token_budget, "max_updates": max_updates,
            "budget_exhausted": tokens == token_budget, "training_nll": loss_sum / tokens,
            "seconds": time.perf_counter() - started, "policy": policy,
            "trainable_parameters": sum(parameter.numel() for _, parameter in selected),
            "trainable_names": [name for name, _ in selected]}
