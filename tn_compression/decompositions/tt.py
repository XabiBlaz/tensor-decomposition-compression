"""TT-SVD with explicit factorized execution for convolutions and matrices.

The sequential SVD construction follows Oseledets' tensor-train decomposition.
TTPWT contracts spatial cores as pointwise/vertical/horizontal/pointwise convs.
PLAIN_TT is an explicitly dense-reconstruction compatibility representation.
"""

import math
import string

import torch
from torch import nn
from torch.nn import functional as F

from .common import conv_sequence, copy_weights, decomposition_weight, finish_replacement, validate_conv
from ..compression_methods.rank_selection import energy_rank_from_svals, select_rank


def _factorize_balanced(size, count):
    """Distribute prime factors without losing dimensions for prime/odd sizes."""
    factors = [1] * count
    divisor = 2
    while divisor * divisor <= size:
        while size % divisor == 0:
            index = min(range(count), key=factors.__getitem__)
            factors[index] *= divisor
            size //= divisor
        divisor += 1
    if size > 1:
        factors[min(range(count), key=factors.__getitem__)] *= size
    return sorted(factors)


@torch.no_grad()
def tt_svd(tensor, *, ranks=None, method="SVD", energy=0.94, rank_cap=None):
    """Factor a tensor into (left rank, mode size, right rank) cores."""
    dimensions = tensor.shape
    if len(dimensions) < 2:
        raise ValueError("TT-SVD needs at least two modes.")
    if isinstance(ranks, int):
        ranks = [ranks] * (len(dimensions) - 1)
    if ranks is not None and (len(ranks) != len(dimensions) - 1 or
                             any(not isinstance(rank, int) or rank < 1 for rank in ranks)):
        raise ValueError("Specify one positive TT rank per internal bond.")
    cores = []
    remainder = tensor
    previous_rank = 1
    for index, size in enumerate(dimensions[:-1]):
        matrix = remainder.reshape(previous_rank * size, -1)
        left, values, right = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(ranks[index], len(values)) if ranks is not None else select_rank(
            matrix, method, energy, rank_cap)
        cores.append(left[:, :rank].reshape(previous_rank, size, rank))
        remainder = values[:rank, None] * right[:rank]
        previous_rank = rank
    cores.append(remainder.reshape(previous_rank, dimensions[-1], 1))
    return cores


def _rank_from_energy(singular_values, energy_threshold):
    return energy_rank_from_svals(singular_values, energy_threshold)


def _tt_svd_energy(tensor, energy_threshold=0.94, rank_cap=None):
    return tt_svd(tensor, energy=energy_threshold, rank_cap=rank_cap)


def _tt_svd_entropy(tensor, entropy_threshold=0.94, rank_cap=None):
    return tt_svd(tensor, method="ENTROPY", energy=entropy_threshold, rank_cap=rank_cap)


class _TTConvPlain(nn.Module):
    """Compatibility path that reconstructs dense weights on every forward.

    Avoid a persistent cache: optimizer/state-dict updates and no-grad/eval
    transitions otherwise risk stale weights or accidentally detached gradients.
    """

    def __init__(self, cores, stride, padding, dilation, bias=None):
        super().__init__()
        self.cores = nn.ParameterList([nn.Parameter(core.detach().clone()) for core in cores])
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def _reconstruct_weight(self):
        weight = self.cores[0][0]
        for core in self.cores[1:]:
            weight = torch.tensordot(weight, core, dims=([-1], [0]))
        return weight[..., 0]

    def forward(self, inputs):
        return F.conv2d(inputs, self._reconstruct_weight(), self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation)


class _TTLinearCore(nn.Module):
    """Contract TT-matrix cores directly, preserving arbitrary batch dimensions."""

    def __init__(self, input_dims, output_dims, cores, bias=None):
        super().__init__()
        self.input_dims, self.output_dims = list(input_dims), list(output_dims)
        self.in_features, self.out_features = math.prod(input_dims), math.prod(output_dims)
        self.cores = nn.ParameterList([nn.Parameter(core.detach().clone()) for core in cores])
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {inputs.shape[-1]}.")
        leading_shape = inputs.shape[:-1]
        # Carry produced output axes ahead of the remaining input axes. Each
        # contraction removes just one input axis; no dense weight is formed.
        count = len(self.cores)
        letters = string.ascii_lowercase
        input_axes = letters[:count]
        output_axes = letters[count:2 * count]
        state = inputs.reshape(-1, 1, *self.input_dims)
        for index, core in enumerate(self.cores):
            produced = output_axes[:index]
            pending = input_axes[index:]
            equation = f"BR{produced}{pending},R{output_axes[index]}{input_axes[index]}S->BS{produced}{output_axes[index]}{pending[1:]}"
            state = torch.einsum(equation, state, core)
        output = state.reshape(*leading_shape, self.out_features)
        return output + self.bias if self.bias is not None else output


def _statistics(layer, replacement):
    original = sum(parameter.numel() for parameter in layer.parameters())
    compressed = sum(parameter.numel() for parameter in replacement.parameters())
    return original, compressed, compressed / original, replacement


def tensor_train_decomposition_conv_layer_plain_tt(layer, method="SVD", energy=0.94,
                                                    ranks=None, rank_cap=None, **kwargs):
    validate_conv(layer)
    if kwargs.get("round_to_8"):
        raise ValueError("Choose explicit hardware-aligned ranks.")
    cores = tt_svd(decomposition_weight(layer), ranks=ranks, method=method,
                   energy=energy, rank_cap=rank_cap)
    padding = layer.padding if layer.padding_mode == "zeros" else (0, 0)
    replacement = _TTConvPlain(cores, layer.stride, padding, layer.dilation, layer.bias)
    if layer.padding_mode != "zeros":
        replacement = conv_sequence(layer, [replacement])
    else:
        finish_replacement(layer, replacement)
    return _statistics(layer, replacement)


def tensor_train_decomposition_conv_layer(layer, method="SVD", energy=0.94, ranks=None,
                                          rank_cap=None, round_to_8=False, structure="TTPWT"):
    validate_conv(layer)
    if round_to_8:
        raise ValueError("Choose explicit hardware-aligned ranks.")
    if structure.upper() == "PLAIN_TT":
        return tensor_train_decomposition_conv_layer_plain_tt(layer, method, energy, ranks, rank_cap)
    if structure.upper() != "TTPWT":
        raise ValueError(f"Unknown TT convolution structure: {structure}")
    # W[output,input,height,width] becomes T[input,height,width,output].
    weight = decomposition_weight(layer).permute(1, 2, 3, 0).contiguous()
    first, vertical, horizontal, last = tt_svd(weight, ranks=ranks, method=method,
                                              energy=energy, rank_cap=rank_cap)
    r1, r2, r3 = first.shape[-1], vertical.shape[-1], horizontal.shape[-1]
    pad_h, pad_w = layer.padding if layer.padding_mode == "zeros" else (0, 0)
    stride_h, stride_w = layer.stride
    dilation_h, dilation_w = layer.dilation
    modules = [
        nn.Conv2d(layer.in_channels, r1, 1, bias=False),
        nn.Conv2d(r1, r2, (layer.kernel_size[0], 1), stride=(stride_h, 1),
                  padding=(pad_h, 0), dilation=(dilation_h, 1), bias=False),
        nn.Conv2d(r2, r3, (1, layer.kernel_size[1]), stride=(1, stride_w),
                  padding=(0, pad_w), dilation=(1, dilation_w), bias=False),
        nn.Conv2d(r3, layer.out_channels, 1, bias=layer.bias is not None),
    ]
    copy_weights(modules[0], first[0].T[:, :, None, None])
    copy_weights(modules[1], vertical.permute(2, 0, 1)[:, :, :, None])
    copy_weights(modules[2], horizontal.permute(2, 0, 1)[:, :, None, :])
    copy_weights(modules[3], last[:, :, 0].T[:, :, None, None], layer.bias)
    return _statistics(layer, conv_sequence(layer, modules))


def tensor_train_decomposition_linear_layer_tt(layer, energy=0.95, num_cores=3,
                                               rank_cap=None, round_to_8=False, method="SVD"):
    if not 2 <= num_cores <= 8:
        raise ValueError("num_cores must be between 2 and 8.")
    if round_to_8:
        raise ValueError("Choose an explicit hardware-aligned rank cap.")
    weight = decomposition_weight(layer)
    inputs = _factorize_balanced(layer.in_features, num_cores)
    outputs = _factorize_balanced(layer.out_features, num_cores)
    # Reshape alone cannot interleave matrix indices: output axes precede all
    # input axes in W. Permute before fusing each (output,input) pair.
    axes = [axis for pair in zip(range(num_cores), range(num_cores, 2 * num_cores)) for axis in pair]
    tensor = weight.reshape(*outputs, *inputs).permute(axes).contiguous()
    tensor = tensor.reshape(*(out_size * in_size for out_size, in_size in zip(outputs, inputs)))
    cores = tt_svd(tensor, method=method, energy=energy, rank_cap=rank_cap)
    matrix_cores = [core.reshape(core.shape[0], out_size, in_size, core.shape[-1])
                    for core, out_size, in_size in zip(cores, outputs, inputs)]
    replacement = _TTLinearCore(inputs, outputs, matrix_cores, layer.bias)
    return _statistics(layer, finish_replacement(layer, replacement))


def tensor_train_decomposition_linear_layer(layer, method="SVD", energy=0.95, rank_cap=None,
                                            round_to_8=False, target_ratio_=None, num_cores=3):
    """True TT-matrix execution; use type=svd (or legacy cp2) for matrix SVD."""
    if target_ratio_ is not None:
        raise ValueError("target_ratio_ is unsupported; use an explicit rank cap or a budget planner.")
    return tensor_train_decomposition_linear_layer_tt(
        layer, energy, num_cores, rank_cap, round_to_8, "SVD" if method.upper() == "TT" else method)
