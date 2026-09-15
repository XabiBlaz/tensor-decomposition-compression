"""Dispatch a compression policy to the authoritative numerical implementation."""

from torch import nn

from ..decompositions import cp, tt, tucker


def check_compress_layer_input(compression):
    if compression is None or "type" not in compression:
        return None
    return {"rank": None, **compression}


def compress_layer(layer, compression):
    policy = check_compress_layer_input(compression)
    if policy is None:
        return None
    kind = policy["type"].lower()
    rank = policy["rank"]
    method, energy = policy.get("method", "SVD"), policy.get("energy", 0.94)
    cap = policy.get("rank_cap")
    if kind in {"svd", "cp2"} and type(layer) is nn.Linear:
        return cp.cp_linear(layer, rank) if rank is not None else cp.cp_linear_auto(layer, method, energy, cap)
    if kind == "partial_tucker" and type(layer) is nn.Conv2d:
        if rank is None and cap is not None:
            from .rank_selection import channel_ranks
            rank = channel_ranks(layer.weight, method, energy, cap)
        return tucker.partial_tucker_decomposition_conv_layer(layer, rank, method, energy)
    if kind in {"cp3", "cp4"} and type(layer) is nn.Conv2d:
        rank = rank if rank is not None else cp.automatic_cp_rank(layer, method, energy, cap)
        return cp.cp_convolution(layer, rank, split_spatial=kind == "cp4")
    if kind in {"tt", "tensor_train"}:
        if type(layer) is nn.Conv2d:
            return tt.tensor_train_decomposition_conv_layer(
                layer, method, energy, rank, cap, structure=policy.get("structure", "TTPWT"))[-1]
        if type(layer) is nn.Linear:
            return tt.tensor_train_decomposition_linear_layer(
                layer, method, energy, rank if rank is not None else cap,
                num_cores=policy.get("num_cores", 3))[-1]
    raise ValueError(f"{kind} does not support {type(layer).__name__}.")
