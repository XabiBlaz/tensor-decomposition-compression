"""A round-to-nearest reference, separate from deployable GPTQ integration."""

import copy

import torch


@torch.no_grad()
def round_to_nearest(layer, *, bits=4, group_size=128):
    """Symmetric grouped fake quantization; returns dense floating-point weights.

    This baseline measures rounding damage only. It neither packs integers nor
    provides a quantized kernel, and must not be used to claim storage savings.
    """
    if type(layer) is not torch.nn.Linear:
        raise ValueError("The rounding reference supports ordinary Linear layers.")
    if bits not in {2, 3, 4, 8} or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("Choose 2, 3, 4 or 8 bits and a positive integer group_size.")
    result = copy.deepcopy(layer)
    maximum = 2 ** (bits - 1) - 1
    for start in range(0, layer.in_features, group_size):
        block = layer.weight[:, start:start + group_size].float()
        if not torch.isfinite(block).all():
            raise ValueError("Cannot quantize nonfinite weights.")
        scale = block.abs().amax(dim=1, keepdim=True) / maximum
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        rounded = (block / scale).round().clamp(-maximum, maximum) * scale
        result.weight[:, start:start + group_size].copy_(rounded)
    return result
