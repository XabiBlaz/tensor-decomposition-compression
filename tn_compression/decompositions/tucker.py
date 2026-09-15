"""Channel-mode Tucker convolution, adapted from the attributed source versions.

See docs/provenance.md for upstream implementations and reconciled fixes.
"""

import tensorly as tl
from tensorly.decomposition import partial_tucker
from torch import nn

from .common import conv_sequence, copy_weights, decomposition_weight, validate_conv
from ..compression_methods.rank_selection import channel_ranks


def estimate_ranks(layer):
    return channel_ranks(decomposition_weight(layer), "EVBMF")


def partial_tucker_decomposition_conv_layer(layer, ranks=None, method="EVBMF", energy=None):
    validate_conv(layer)
    weight = decomposition_weight(layer)
    if ranks is None:
        ranks = channel_ranks(weight, method, 0.94 if energy is None else energy)
    if isinstance(ranks, int):
        ranks = [ranks, ranks]
    if len(ranks) != 2 or any(not isinstance(rank, int) or rank < 1 for rank in ranks):
        raise ValueError("Tucker requires two positive channel ranks, ordered output then input.")
    ranks = [min(rank, size) for rank, size in zip(ranks, weight.shape[:2])]
    with tl.backend_context("pytorch"):
        (core, factors), _ = partial_tucker(weight, modes=[0, 1], rank=ranks, init="svd")
    output_factor, input_factor = factors
    padding = layer.padding if layer.padding_mode == "zeros" else (0, 0)
    first = nn.Conv2d(layer.in_channels, ranks[1], 1, bias=False)
    middle = nn.Conv2d(ranks[1], ranks[0], layer.kernel_size, stride=layer.stride,
                       padding=padding, dilation=layer.dilation, bias=False)
    last = nn.Conv2d(ranks[0], layer.out_channels, 1, bias=layer.bias is not None)
    copy_weights(first, input_factor.T[:, :, None, None])
    copy_weights(middle, core)
    copy_weights(last, output_factor[:, :, None, None], layer.bias)
    return conv_sequence(layer, [first, middle, last])
