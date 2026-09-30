"""Portable int8 weight storage for ordinary Linear and Conv2d layers.

The forward path dequantizes weights with PyTorch. This reduces saved weight
bytes, but deliberately makes no latency or peak-memory speedup claim.
"""

import fnmatch

import torch
from torch import nn
from torch.nn import functional as F


def _pack(weight):
    if not torch.isfinite(weight).all():
        raise ValueError("Cannot quantize nonfinite weights.")
    flat = weight.detach().float().reshape(weight.shape[0], -1)
    scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127
    quantized = (flat / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.reshape(weight.shape), scale


class PackedInt8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, layer):
        result = cls(layer.in_features, layer.out_features, layer.bias is not None).to(layer.weight.device)
        result.qweight, result.scale = _pack(layer.weight)
        if layer.bias is not None:
            result.bias.copy_(layer.bias)
            result.bias.requires_grad_(layer.bias.requires_grad)
        result.train(layer.training)
        result._tn_replacement = True
        return result.to(layer.weight.device)

    def forward(self, value):
        weight = self.qweight.to(dtype=value.dtype) * self.scale.to(dtype=value.dtype)[:, None]
        bias = self.bias.to(dtype=value.dtype) if self.bias is not None else None
        return F.linear(value, weight, bias)


class PackedInt8Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__()
        if padding_mode != "zeros":
            raise ValueError("Packed Conv2d supports zero padding only.")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.dilation, self.groups, self.padding_mode = dilation, groups, padding_mode
        self.register_buffer("qweight", torch.zeros(out_channels, in_channels // groups,
                                                    *nn.modules.utils._pair(kernel_size), dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    @classmethod
    @torch.no_grad()
    def from_conv(cls, layer):
        result = cls(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride,
                     layer.padding, layer.dilation, layer.groups, layer.bias is not None,
                     layer.padding_mode).to(layer.weight.device)
        result.qweight, result.scale = _pack(layer.weight)
        if layer.bias is not None:
            result.bias.copy_(layer.bias)
            result.bias.requires_grad_(layer.bias.requires_grad)
        result.train(layer.training)
        result._tn_replacement = True
        return result.to(layer.weight.device)

    def forward(self, value):
        weight = self.qweight.to(dtype=value.dtype) * self.scale.to(dtype=value.dtype).view(-1, 1, 1, 1)
        bias = self.bias.to(dtype=value.dtype) if self.bias is not None else None
        return F.conv2d(value, weight, bias, self.stride, self.padding, self.dilation, self.groups)


@torch.no_grad()
def quantize_int8_weights(model, *, include=None, exclude=None):
    """Pack selected eligible layers and report actual tensor-byte changes."""
    from .api import protected_module_reasons

    protected = protected_module_reasons(model)
    include, exclude = list(include or []), list(exclude or [])
    changes, skipped = [], []
    paths = [(path, module) for path, module in model.named_modules()
             if type(module) in {nn.Linear, nn.Conv2d}]
    for path, layer in paths:
        if not path or (include and not any(fnmatch.fnmatchcase(path, pattern) for pattern in include)) \
                or any(fnmatch.fnmatchcase(path, pattern) for pattern in exclude):
            continue
        reason = protected.get(path.replace(".", "/"))
        if reason or (type(layer) is nn.Conv2d and layer.padding_mode != "zeros"):
            skipped.append({"layer_path": path, "reason": reason or "nonzero_padding"})
            continue
        before = sum(t.numel() * t.element_size() for t in layer.state_dict().values())
        replacement = PackedInt8Linear.from_linear(layer) if type(layer) is nn.Linear else PackedInt8Conv2d.from_conv(layer)
        after = sum(t.numel() * t.element_size() for t in replacement.state_dict().values())
        if after >= before:
            skipped.append({"layer_path": path, "reason": "no_tensor_byte_saving"})
            continue
        parent, _, name = path.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, name, replacement)
        changes.append({"layer_path": path, "method": "int8_weight_storage", "original_tensor_bytes": before,
                        "compressed_tensor_bytes": after, "bytes_saved": before - after})
    return {"method": "int8_weight_storage", "layers": changes, "skipped": skipped,
            "caveat": "Weights are dequantized in the PyTorch forward path; latency and peak-memory gains are not implied."}
