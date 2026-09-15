"""Ridge-regularized input-weighted SVD; not a full SVD-LLM reproduction."""

import torch
from torch import nn

from .common import copy_weights, finish_replacement


@torch.no_grad()
def weighted_svd(layer, inputs, rank, *, ridge=1e-4, max_features=2048):
    if type(layer) is not nn.Linear or isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError("Weighted SVD requires an ordinary Linear layer and integer rank.")
    if not 0 < rank <= min(layer.weight.shape) or ridge <= 0:
        raise ValueError("Rank must fit the matrix dimensions and ridge must be positive.")
    if inputs.ndim != 2 or inputs.shape[1] != layer.in_features or not len(inputs):
        raise ValueError("Calibration inputs must be a nonempty [samples, in_features] matrix.")
    if layer.in_features > max_features:
        raise ValueError("Full second moment exceeds max_features; use a smaller layer or a declared approximation.")
    values = inputs.to(device=layer.weight.device, dtype=torch.float64)
    weight = layer.weight.detach().double()
    if not torch.isfinite(values).all() or not torch.isfinite(weight).all():
        raise ValueError("Weighted SVD needs finite inputs and weights.")
    moment = values.T @ values / len(values)
    regularizer = ridge * moment.diagonal().mean().clamp_min(torch.finfo(moment.dtype).eps)
    moment.diagonal().add_(regularizer)
    lower = torch.linalg.cholesky(moment)
    # C = L L^T, so ||(W-Wc)L||_F^2 is the regularized output objective.
    # Truncate W L, then solve B L = Vh instead of explicitly inverting L.
    left, singular, right = torch.linalg.svd(weight @ lower, full_matrices=False)
    right = torch.linalg.solve_triangular(lower.T, right[:rank].T, upper=True).T
    scale = singular[:rank].sqrt()
    first = nn.Linear(layer.in_features, rank, bias=False)
    last = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
    copy_weights(first, scale[:, None] * right)
    copy_weights(last, left[:, :rank] * scale, layer.bias)
    result = finish_replacement(layer, nn.Sequential(first, last))
    result._tn_replacement = True
    return result
