"""Stream bounded per-layer samples and measure actual candidate reconstruction."""

from contextlib import contextmanager
from typing import Callable, Optional

import torch

from .tasks.vision import evaluation_mode


def first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return next(value for value in output if isinstance(value, torch.Tensor))
    raise ValueError("The selected module needs an output adapter for calibration.")


@torch.no_grad()
def collect_module_inputs(model, layer_path, batches, forward: Callable, *, max_samples=256,
                          seed=0, input_mask: Optional[Callable] = None,
                          sample_mode="auto", max_spatial_size=32):
    """Collect bounded row or feature-map inputs from one module at a time.

    Linear-like modules use priority-sampled rows. Convolutions use complete,
    bounded feature-map crops so factorized modules can be evaluated directly
    without materializing every unfolded patch.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")
    layer = model.get_submodule(layer_path)
    if sample_mode == "auto":
        sample_mode = "feature_maps" if type(layer) is torch.nn.Conv2d else "rows"
    if sample_mode not in {"rows", "feature_maps"}:
        raise ValueError("sample_mode must be rows, feature_maps or auto.")
    if sample_mode == "feature_maps" and (type(layer) is not torch.nn.Conv2d or max_spatial_size < 1):
        raise ValueError("Feature-map calibration requires Conv2d and a positive max_spatial_size.")
    samples = None
    priorities = torch.empty(0)
    generator = torch.Generator().manual_seed(seed)
    active_mask = None

    def capture(module, inputs):
        nonlocal samples, priorities
        values = inputs[0].detach()
        if sample_mode == "rows":
            values = values.reshape(-1, values.shape[-1])
            if active_mask is not None:
                if active_mask.numel() != len(values):
                    raise ValueError("Calibration mask does not match the layer's input positions.")
                values = values[active_mask.reshape(-1).to(values.device).bool()]
        else:
            if values.ndim != 4:
                raise ValueError("Conv2d calibration inputs must be NCHW tensors.")
            crop_height = min(values.shape[-2], max_spatial_size)
            crop_width = min(values.shape[-1], max_spatial_size)
            maximum_top = values.shape[-2] - crop_height
            maximum_left = values.shape[-1] - crop_width
            top = int(torch.randint(maximum_top + 1, (), generator=generator)) if maximum_top else 0
            left = int(torch.randint(maximum_left + 1, (), generator=generator)) if maximum_left else 0
            values = values[:, :, top:top + crop_height, left:left + crop_width]
        # Uniform priority sampling visits every batch, with bounded CPU storage.
        # Taking only the first N rows would systematically favor early examples.
        for chunk in values.split(1024):
            keys = torch.rand(len(chunk), generator=generator)
            priorities = torch.cat((priorities, keys))
            chunk = chunk.cpu()
            samples = chunk if samples is None else torch.cat((samples, chunk))
            keep = priorities.topk(min(max_samples, len(priorities))).indices
            priorities, samples = priorities[keep], samples[keep]

    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    devices = sorted({parameter.device.index for parameter in model.parameters() if parameter.is_cuda})
    handle = layer.register_forward_pre_hook(capture)
    try:
        with torch.random.fork_rng(devices=devices), evaluation_mode(model):
            for batch in batches:
                active_mask = input_mask(batch) if input_mask else None
                forward(model, batch)
    finally:
        handle.remove()
        for module, mode in modes:
            module.training = mode
        for buffer, value in buffers:
            buffer.copy_(value)
    if samples is None or not len(samples):
        raise ValueError(f"No inputs collected for {layer_path}.")
    return samples


@torch.no_grad()
def collect_linear_inputs(model, layer_path, batches, forward, *, max_rows=256, seed=0, input_mask=None):
    """Compatibility wrapper for bounded ordinary-Linear calibration."""
    layer = model.get_submodule(layer_path)
    if type(layer) is not torch.nn.Linear:
        raise ValueError("Linear calibration requires an ordinary nn.Linear module.")
    return collect_module_inputs(model, layer_path, batches, forward, max_samples=max_rows,
                                 seed=seed, input_mask=input_mask, sample_mode="rows")


@torch.no_grad()
def reconstruction_error(original, candidate, inputs, *, epsilon=1e-12):
    """Direct output error also works when factorized candidates have no .weight."""
    values = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
    if not values or any(value.numel() == 0 for value in values) or epsilon <= 0:
        raise ValueError("Reconstruction needs nonempty inputs and positive epsilon.")
    device = next(original.parameters()).device
    dtype = next(original.parameters()).dtype
    error = scale = 0.0
    elements = 0
    with evaluation_mode(original), evaluation_mode(candidate):
        for value in values:
            value = value.to(device=device, dtype=dtype)
            expected, actual = first_tensor(original(value)), first_tensor(candidate(value))
            if actual.shape != expected.shape:
                raise ValueError("Candidate changed the module output contract.")
            error += (expected.double() - actual.double()).square().sum().item()
            scale += expected.double().square().sum().item()
            elements += expected.numel()
    return {"squared_error": error, "relative_squared_error": error / (scale + epsilon),
            "mean_squared_error": error / elements, "output_elements": elements}


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
