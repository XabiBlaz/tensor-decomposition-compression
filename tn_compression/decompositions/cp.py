"""CP convolutions and the historical cp2 alias for ordinary matrix SVD.

Adapted from the sources credited in docs/provenance.md.
"""

import tensorly as tl
import torch
from torch import nn

from .common import conv_sequence, copy_weights, decomposition_weight, finish_replacement, validate_conv
from ..compression_methods.rank_selection import channel_ranks, select_rank


def cp_convolution(layer, rank, *, split_spatial=False):
    validate_conv(layer)
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("CP rank must be a positive integer.")
    with tl.backend_context("pytorch"):
        scales, factors = tl.decomposition.parafac(
            decomposition_weight(layer), rank=rank, init="svd", n_iter_max=100, tol=1e-7)
    output_factor, input_factor, vertical, horizontal = factors
    # CP coefficients belong to each rank-one term; dropping them changes W.
    if scales is not None:
        output_factor = output_factor * scales
    first = nn.Conv2d(layer.in_channels, rank, 1, bias=False)
    last = nn.Conv2d(rank, layer.out_channels, 1, bias=layer.bias is not None)
    copy_weights(first, input_factor.T[:, :, None, None])
    copy_weights(last, output_factor[:, :, None, None], layer.bias)
    padding_h, padding_w = layer.padding if layer.padding_mode == "zeros" else (0, 0)
    stride_h, stride_w = layer.stride
    dilation_h, dilation_w = layer.dilation
    if split_spatial:
        height = nn.Conv2d(rank, rank, (layer.kernel_size[0], 1), stride=(stride_h, 1),
                           padding=(padding_h, 0), dilation=(dilation_h, 1), groups=rank, bias=False)
        width = nn.Conv2d(rank, rank, (1, layer.kernel_size[1]), stride=(1, stride_w),
                          padding=(0, padding_w), dilation=(1, dilation_w), groups=rank, bias=False)
        copy_weights(height, vertical.T[:, None, :, None])
        copy_weights(width, horizontal.T[:, None, None, :])
        middle = [height, width]
    else:
        spatial = nn.Conv2d(rank, rank, layer.kernel_size, stride=layer.stride,
                            padding=(padding_h, padding_w), dilation=layer.dilation, groups=rank, bias=False)
        copy_weights(spatial, torch.einsum("hr,wr->rhw", vertical, horizontal)[:, None])
        middle = [spatial]
    return conv_sequence(layer, [first, *middle, last])


def cp_conv_layer_pdp(layer, rank):
    return cp_convolution(layer, rank)


def cp_conv_layer_pddp(layer, rank):
    return cp_convolution(layer, rank, split_spatial=True)


def cp_linear(layer, rank):
    """Ordinary truncated SVD, W ≈ (U sqrt(S)) (sqrt(S) Vh)."""
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("SVD rank must be a positive integer.")
    weight = decomposition_weight(layer)
    rank = min(rank, *weight.shape)
    with torch.no_grad():
        left, values, right = torch.linalg.svd(weight, full_matrices=False)
        scale = values[:rank].sqrt()
        first = nn.Linear(layer.in_features, rank, bias=False)
        last = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
        copy_weights(first, scale[:, None] * right[:rank])
        copy_weights(last, left[:, :rank] * scale, layer.bias)
    return finish_replacement(layer, nn.Sequential(first, last))


def automatic_cp_rank(layer, method="SVD", energy=0.94, rank_cap=None, combine="max"):
    ranks = channel_ranks(decomposition_weight(layer), method, energy, rank_cap)
    if combine not in {"min", "max"}:
        raise ValueError("combine must be min or max.")
    return min(ranks) if combine == "min" else max(ranks)


def cp_conv_layer_pdp_auto(layer, method="EVBMF", energy=0.94, rank_cap=None, combine="max"):
    return cp_conv_layer_pdp(layer, automatic_cp_rank(layer, method, energy, rank_cap, combine))


def cp_linear_auto(layer, method="SVD", energy=0.95, rank_cap=None, round_to_8=False):
    rank = select_rank(decomposition_weight(layer), method, energy, rank_cap)
    if round_to_8:
        raise ValueError("Choose an explicit hardware-aligned rank; automatic rounding is not supported.")
    return cp_linear(layer, rank)
