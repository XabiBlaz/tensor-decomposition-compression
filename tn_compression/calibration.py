"""Stream bounded per-layer samples and measure actual candidate reconstruction."""

from contextlib import contextmanager

import torch

from .tasks.vision import evaluation_mode


def first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return next(value for value in output if isinstance(value, torch.Tensor))
    raise ValueError("The selected module needs an output adapter for calibration.")


@torch.no_grad()
def collect_linear_inputs(model, layer_path, batches, forward, *, max_rows=256, seed=0, input_mask=None):
    """Bound storage while collecting one layer at a time, not all activations."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    layer = model.get_submodule(layer_path)
    if type(layer) is not torch.nn.Linear:
        raise ValueError("Linear calibration requires an ordinary nn.Linear module.")
    samples = torch.empty(0, layer.in_features, dtype=layer.weight.dtype)
    priorities = torch.empty(0)
    generator = torch.Generator().manual_seed(seed)
    active_mask = None

    def capture(module, inputs):
        nonlocal samples, priorities
        values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        if active_mask is not None:
            if active_mask.numel() != len(values):
                raise ValueError("Calibration mask does not match the layer's input positions.")
            values = values[active_mask.reshape(-1).to(values.device).bool()]
        # Uniform priority sampling visits every batch, with bounded CPU storage.
        # Taking only the first N rows would systematically favor early examples.
        for chunk in values.split(1024):
            keys = torch.rand(len(chunk), generator=generator)
            priorities = torch.cat((priorities, keys))
            samples = torch.cat((samples, chunk.cpu()))
            keep = priorities.topk(min(max_rows, len(priorities))).indices
            priorities, samples = priorities[keep], samples[keep]

    handle = layer.register_forward_pre_hook(capture)
    try:
        with evaluation_mode(model):
            for batch in batches:
                active_mask = input_mask(batch) if input_mask else None
                forward(model, batch)
    finally:
        handle.remove()
    if not len(samples):
        raise ValueError(f"No inputs collected for {layer_path}.")
    return samples


@torch.no_grad()
def reconstruction_error(original, candidate, inputs, *, epsilon=1e-12):
    """Direct output error also works when factorized candidates have no .weight."""
    if inputs.numel() == 0 or epsilon <= 0:
        raise ValueError("Reconstruction needs nonempty inputs and positive epsilon.")
    device = next(original.parameters()).device
    inputs = inputs.to(device=device, dtype=next(original.parameters()).dtype)
    with evaluation_mode(original), evaluation_mode(candidate):
        expected, actual = first_tensor(original(inputs)), first_tensor(candidate(inputs))
    if actual.shape != expected.shape:
        raise ValueError("Candidate changed the module output contract.")
    error = (expected.double() - actual.double()).square().sum().item()
    scale = expected.double().square().sum().item()
    return {"squared_error": error, "relative_squared_error": error / (scale + epsilon),
            "mean_squared_error": error / expected.numel(), "output_elements": expected.numel()}


@contextmanager
def candidate_intervention(model, path, candidate):
    """Restore module identity, flags, buffers and RNG even after failed trials."""
    parent_path, _, name = path.rpartition('.')
    parent = model.get_submodule(parent_path) if parent_path else model
    original = parent._modules[name]
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    devices = sorted({parameter.device.index for parameter in model.parameters() if parameter.is_cuda})
    with torch.random.fork_rng(devices=devices):
        try:
            parent._modules[name] = candidate
            model.eval()
            yield
        finally:
            parent._modules[name] = original
            for module, mode in modes:
                module.training = mode
            with torch.no_grad():
                for buffer, value in buffers:
                    buffer.copy_(value)
