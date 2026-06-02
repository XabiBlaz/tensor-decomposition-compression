#-----------------------------------------------------------------------------------------------------------------------
# modified from decompositions.py from github.com/yuncheng97/Tensor-decomposition-pytorch/tree/main
# added alternative from github.com/JeanKossaifi/tensorly-notebooks/blob/master/05_pytorch_backend/cnn_acceleration_tensorly_and_pytorch.ypynb
#-----------------------------------------------------------------------------------------------------------------------

import logging

import tensorly as tl
from tensorly.decomposition import partial_tucker
import numpy as np
import torch
import torch.nn as nn
from . import VBMF


logger = logging.getLogger(__name__)


def estimate_ranks(layer):
    """
    Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0) 
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def partial_tucker_decomposition_conv_layer(layer, ranks=None, method='EVBMF', energy=None):
    """
    Gets a conv layer,
    returns a nn.Sequential object with the Tucker decomposition.
    The ranks are estimated with a Python implementation of VBMF (Variational Bayes Matrix Factorization)
    https://github.com/CasvandenBogaard/VBMF

    The ranks are estimated with VBMF by default. Supports:
      - method='EVBMF' (default)
      - method='SVD' with energy threshold
      - method='ENTROPY' with entropy threshold (tau)
    """
    weight = layer.weight.data

    if energy is None:
        energy = 0.94
    
    if ranks is None:
        if method.upper() == 'SVD':
            from ..compression_methods.rank_selection import energy_ranks_conv
            r_out, r_in = energy_ranks_conv(weight, energy=energy)
            ranks = (r_out, r_in)
            logger.debug("%s SVD energy-based estimated ranks %s", layer, ranks)
        elif method.upper() == 'ENTROPY':
            from ..compression_methods.rank_selection import entropy_ranks_conv
            r_out, r_in = entropy_ranks_conv(weight, tau=energy)
            ranks = (r_out, r_in)
            logger.debug("%s entropy-based estimated ranks %s", layer, ranks)
        else:  # Default EVBMF
            ranks = estimate_ranks(layer)
            logger.debug("%s VBMF estimated ranks %s", layer, ranks)
    # Normalize ranks type
    if isinstance(ranks, int):
        ranks = [ranks, ranks]
    elif isinstance(ranks, tuple):
        ranks = list(ranks)
    elif not isinstance(ranks, list):
        ranks = [ranks, ranks]

    (core, factors), _ = partial_tucker(layer.weight.data, modes=[0, 1], rank=ranks, init='svd')
    last = factors[0]
    first = factors[1]

    # TensorLy (>=0.8) returns factors already shaped (dim, rank) but can collapse to 1-D tensors
    out_channels = layer.weight.shape[0]
    in_channels = layer.weight.shape[1]
    if last.ndim < 2:
        last = last.reshape(out_channels, -1)
    if first.ndim < 2:
        first = first.reshape(in_channels, -1)

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                  out_channels=first.shape[1], 
                                  kernel_size=1,
                                  stride=1, 
                                  padding=0, 
                                  dilation=layer.dilation, 
                                  bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                 out_channels=core.shape[0], 
                                 kernel_size=layer.kernel_size,
                                 stride=layer.stride, 
                                 padding=layer.padding, 
                                 dilation=layer.dilation,
                                 bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                 out_channels=last.shape[0], 
                                 kernel_size=1, 
                                 stride=1,
                                 padding=0, 
                                 dilation=layer.dilation, 
                                 bias=True)

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

