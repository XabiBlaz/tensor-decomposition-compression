"""Spectral rank estimates; zero tensors have the minimum legal rank."""

import torch

from ..decompositions import VBMF


def validate_threshold(value):
    if not 0 < value <= 1:
        raise ValueError("Energy/entropy threshold must be in (0, 1].")


def energy_rank_from_svals(values, energy=0.95):
    validate_threshold(energy)
    powers = values.double().square()
    if not values.numel() or powers.sum() == 0:
        return 1
    cumulative = powers.cumsum(0)
    index = torch.searchsorted(cumulative, energy * cumulative[-1]).item()
    return min(values.numel(), int(index) + 1)


def entropy_rank_from_svals(values, tau=0.94, eps=1e-12):
    """Smallest prefix carrying tau of the singular-value distribution entropy."""
    validate_threshold(tau)
    values = values.double().clamp_min(0)
    if not values.numel() or values.sum() <= eps:
        return 1
    probabilities = values / values.sum()
    entropy = -probabilities * probabilities.clamp_min(eps).log()
    if entropy.sum() <= eps:
        return 1
    index = torch.searchsorted(entropy.cumsum(0), tau * entropy.sum()).item()
    return min(values.numel(), int(index) + 1)


def select_rank(matrix, method="SVD", energy=0.95, rank_cap=None):
    if rank_cap is not None and (not isinstance(rank_cap, int) or rank_cap < 1):
        raise ValueError("rank_cap must be a positive integer or None.")
    matrix = matrix.detach()
    if matrix.dtype not in (torch.float32, torch.float64):
        matrix = matrix.float()
    method = method.upper()
    if method in {"EVBMF", "VBMF"}:
        array = matrix.cpu().numpy()
        # EVBMF expects the shorter dimension to be the row dimension.
        if array.shape[0] > array.shape[1]:
            array = array.T
        if not array.any():
            rank = 1
        else:
            _, diagonal, _, _ = VBMF.EVBMF(array)
            rank = max(1, diagonal.shape[0])
    elif method in {"SVD", "ENTROPY"}:
        values = torch.linalg.svdvals(matrix)
        score = energy_rank_from_svals if method == "SVD" else entropy_rank_from_svals
        rank = score(values, energy)
    else:
        raise ValueError(f"Unknown rank method: {method}")
    return min(rank, rank_cap or rank, min(matrix.shape))


def channel_ranks(weight, method="SVD", energy=0.94, rank_cap=None):
    return tuple(select_rank(weight.movedim(mode, 0).reshape(weight.shape[mode], -1),
                             method, energy, rank_cap) for mode in (0, 1))


def energy_ranks_conv(weight, energy=0.94):
    return channel_ranks(weight, "SVD", energy)


def evbmf_ranks_conv(weight):
    return channel_ranks(weight, "EVBMF")


def entropy_ranks_conv(weight, tau=0.94):
    return channel_ranks(weight, "ENTROPY", tau)


def energy_rank_linear(weight, energy=0.95):
    return select_rank(weight, "SVD", energy)


def entropy_rank_matrix(matrix, tau=0.94):
    return select_rank(matrix, "ENTROPY", tau)


entropy_rank_linear = entropy_rank_matrix


def round_to_multiple(value, multiple=8):
    return int(max(multiple, multiple * round(value / multiple)))
