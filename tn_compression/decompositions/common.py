"""Shared numerical and convolution contracts for factorized layers."""

import torch
from torch import nn


def decomposition_weight(layer):
    """SVD/ALS need at least float32 even when inference uses half precision."""
    weight = layer.weight.detach()
    return weight if weight.dtype == torch.float64 else weight.float()


def validate_conv(layer):
    if type(layer) is not nn.Conv2d or layer.groups != 1:
        raise ValueError("Only ordinary groups=1 Conv2d is supported; custom forwards need an adapter.")
    if isinstance(layer.padding, str):
        raise ValueError("String padding requires an explicit padding adapter.")


def finish_replacement(layer, replacement):
    replacement.to(device=layer.weight.device, dtype=layer.weight.dtype)
    replacement.train(layer.training)
    for name, parameter in replacement.named_parameters():
        source = layer.bias if name.endswith("bias") else layer.weight
        parameter.requires_grad_(source.requires_grad if source is not None else False)
    return replacement


def conv_sequence(layer, modules):
    """Apply nonzero padding once, before spatial factorization."""
    if layer.padding_mode != "zeros":
        padding = layer._reversed_padding_repeated_twice
        pad_type = {"reflect": nn.ReflectionPad2d, "replicate": nn.ReplicationPad2d,
                    "circular": CircularPad2d}[layer.padding_mode]
        modules = [pad_type(padding)] + list(modules)
    return finish_replacement(layer, nn.Sequential(*modules))


class CircularPad2d(nn.Module):
    """Circular padding compatible with older supported PyTorch versions."""

    def __init__(self, padding):
        super().__init__()
        self.padding = tuple(padding)

    def forward(self, inputs):
        return nn.functional.pad(inputs, self.padding, mode="circular")


def copy_weights(module, weight, bias=None):
    module.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        module.weight.copy_(weight)
        if bias is not None:
            module.bias.copy_(bias)
    return module
