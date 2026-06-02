#-----------------------------------------------------------------------------------------------------------------------
# modified from decompositions.py from github.com/yuncheng97/Tensor-decomposition-pytorch/tree/main
# added alternative from github.com/JeanKossaifi/tensorly-notebooks/blob/master/05_pytorch_backend/cnn_acceleration_tensorly_and_pytorch.ypynb
#-----------------------------------------------------------------------------------------------------------------------

import tensorly as tl
from torch.autograd import Variable
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from ..compression_methods.rank_selection import energy_rank_linear, entropy_rank_linear
from ..compression_methods.rank_selection import evbmf_ranks_conv, energy_ranks_conv, round_to_multiple

def cp_conv_layer_pdp(layer, rank):
    """
    Transforms into three layers:
    - pointwise
    - depthwise
    - pointwise
    """
    W = layer.weight.data

    a, b = tl.decomposition.parafac(layer.weight.data, rank=rank)
    last = b[0]
    first = b[1]
    vertical = b[2]
    horizontal = b[3]
    
    pointwise_s_to_r_layer = nn.Conv2d(in_channels=first.shape[0],
                                       out_channels=first.shape[1],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)

    depthwise_r_to_r_layer = nn.Conv2d(in_channels=rank,
                                       out_channels=rank,
                                       kernel_size=vertical.shape[0],
                                       stride=layer.stride,
                                       padding=layer.padding,
                                       dilation=layer.dilation,
                                       groups=rank,
                                       bias=False)
                                       
    pointwise_r_to_t_layer = nn.Conv2d(in_channels=last.shape[1],
                                       out_channels=last.shape[0],
                                       kernel_size=1,
                                       padding=0,
                                       bias=True)
    
    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    sr = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    rt = last.unsqueeze_(-1).unsqueeze_(-1)
    rr = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1) for i in range(rank)]).unsqueeze_(1)

    pointwise_s_to_r_layer.weight.data = sr 
    pointwise_r_to_t_layer.weight.data = rt
    depthwise_r_to_r_layer.weight.data = rr

    new_layers = [pointwise_s_to_r_layer,
                  depthwise_r_to_r_layer, pointwise_r_to_t_layer]
    return nn.Sequential(*new_layers)

def cp_conv_layer_pddp(layer, rank):
    """
    Compresses into four layers:
    - pointwise
    - depthwise
    - depthwise
    - pointwise
    """
    # Perform CP decomposition on the layer weight tensorly.
    '''last, first, vertical, horizontal = \
        parafac(layer.weight.data, rank=rank, init='svd')'''
    a, b = tl.decomposition.parafac(layer.weight.data, rank=rank)
    last = b[0]
    first = b[1]
    vertical = b[2]
    horizontal = b[3]

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0,
            dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
            groups=vertical.shape[1], bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1],
            out_channels=horizontal.shape[1],
            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
            padding=(0, layer.padding[0]),
            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
            out_channels=last.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=True)

    pointwise_r_to_t_layer.bias.data = layer.bias.data

    depthwise_horizontal_layer.weight.data = \
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]

    return nn.Sequential(*new_layers)


def cp_linear(layer, rank):
    """
    cp2: apply SVD to get two linear layers
    """
    weight = layer.weight.detach()
    rank = max(1, min(int(rank), weight.shape[0], weight.shape[1]))
    u, s, v = np.linalg.svd(weight.cpu().numpy(), full_matrices=False)
    u = u[:, :rank] * np.sqrt(s[:rank])
    v = np.transpose(np.transpose(v)[:, :rank] * np.sqrt(s[:rank]))

    linear1 = torch.nn.Linear(np.shape(layer.weight.data)[1], rank, bias=False)
    linear2 = torch.nn.Linear(rank, np.shape(layer.weight.data)[0], bias=layer.bias is not None)

    linear1.weight.data = torch.as_tensor(v, dtype=layer.weight.dtype, device=layer.weight.device)
    linear2.weight.data = torch.as_tensor(u, dtype=layer.weight.dtype, device=layer.weight.device)
    if layer.bias is not None:
        linear2.bias.data = layer.bias.data.detach().clone()

    new_layers = [linear1, linear2]

    return nn.Sequential(*new_layers)

def cp_conv_layer_pdp_auto(layer, method='EVBMF', energy=0.94, rank_cap=None, combine='max'):
    """Auto rank selection for CP decomposition"""
    
    if method.upper() == 'SVD':
        from ..compression_methods.rank_selection import energy_ranks_conv
        r_out, r_in = energy_ranks_conv(layer.weight.data, energy=energy)
        rank = max(r_out, r_in) if combine == 'max' else min(r_out, r_in)
    elif method.upper() == 'ENTROPY':
        from ..compression_methods.rank_selection import entropy_ranks_conv
        r_out, r_in = entropy_ranks_conv(layer.weight.data, tau=energy)
        rank = max(r_out, r_in) if combine == 'max' else min(r_out, r_in)
    else:  # EVBMF
        from ..compression_methods.rank_selection import evbmf_ranks_conv
        r_out, r_in = evbmf_ranks_conv(layer.weight.data)
        rank = max(r_out, r_in) if combine == 'max' else min(r_out, r_in)
    
    if rank_cap is not None:
        rank = min(rank, rank_cap)
    
    return cp_conv_layer_pdp(layer, rank)

def cp_linear_auto(layer, method='SVD', energy=0.95, rank_cap=None, round_to_8=False):
    """Auto-rank CP2 for Linear via SVD energy, ENTROPY, or EVBMF."""
    W = layer.weight.data
    if method.upper() == 'ENTROPY':
        rank = entropy_rank_linear(W, tau=energy)
    elif method.upper() == 'EVBMF':
        try:
            from . import VBMF
            _, diag, _, _ = VBMF.EVBMF(W.detach().cpu().numpy())
            rank = int(diag.shape[0])  # diag is square
        except Exception:
            rank = energy_rank_linear(W, energy=energy)
    else:  # SVD energy (default)
        rank = energy_rank_linear(W, energy=energy)
    if rank_cap is not None:
        rank = min(rank, int(rank_cap))
    if round_to_8 and rank >= 8:
        rank = round_to_multiple(rank, 8)
    return cp_linear(layer, rank)
