import logging

from ..decompositions import cp, tt, tucker


logger = logging.getLogger(__name__)


def check_compress_layer_input(compression):
    """
    sanity check of input data
    """
    if compression is None or 'type' not in compression:
        return None
    compression.setdefault('rank', None)
    return compression


def apply_partial_tucker(layer, compression):
    """
    apply partial tucker decomposition
    """
    method = compression.get('method', 'EVBMF')
    energy = compression.get('energy', 0.94)
    return tucker.partial_tucker_decomposition_conv_layer(layer, compression['rank'], method=method, energy=energy)


def apply_cp_linear(layer, compression):
    """
    apply cp decomposition to linear layers
    """
    if compression['rank'] is None:
        method = compression.get('method', 'SVD')
        energy = compression.get('energy', 0.95)
        rank_cap = compression.get('rank_cap')
        return cp.cp_linear_auto(layer, method=method, energy=energy, rank_cap=rank_cap)
    return cp.cp_linear(layer, compression['rank'])


def apply_cp_conv_pdp_auto(layer, compression):
    """
    apply cp decomposition to conv layers with unknown rank (automatic search or the rank)
    """
    method = compression.get('method', 'EVBMF')
    energy = compression.get('energy', 0.94)
    rank_cap = compression.get('rank_cap')
    return cp.cp_conv_layer_pdp_auto(layer, method=method, energy=energy, rank_cap=rank_cap)


def apply_cp_conv_pdp(layer, compression):
    """
    apply cp decomposition to conv layers with known rank
    """
    return cp.cp_conv_layer_pddp(layer, compression['rank'])


def apply_cp_conv(layer, compression):
    """
    apply cp decomposition method to conv layers
    """
    if compression['rank'] is None:
        return apply_cp_conv_pdp_auto(layer, compression)
    return apply_cp_conv_pdp(layer, compression)


def apply_tt_decomposition(layer, compression):
    """
    apply tt decomposition to conv2d or linear layers
    """
    layer_type = str(type(layer)).lower()
    if 'conv2d' in layer_type:
        method = compression.get('method', 'VBMF')
        rank = compression.get('rank', None)
        energy = compression.get('energy', 0.94)
        rank_cap = compression.get('rank_cap')
        structure = compression.get('structure', 'TTPWT')

        # Use fixed rank if provided, otherwise use automatic rank selection
        if rank is not None:
            # Fixed rank - pass it to the function
            num_param, num_param_decomp, ratio, new_layer = tt.tensor_train_decomposition_conv_layer(
                layer,
                ranks=rank,
                method=method,
                energy=energy,
                rank_cap=rank_cap,
                structure=structure)
        else:
            # Automatic rank selection
            num_param, num_param_decomp, ratio, new_layer = tt.tensor_train_decomposition_conv_layer(
                layer,
                method=method,
                energy=energy,
                rank_cap=rank_cap,
                structure=structure)  
        logger.debug(
            "TT (%s) Conv decomposition: %s -> %s params, ratio %.4f",
            structure,
            num_param,
            num_param_decomp,
            ratio,
        )
    elif 'linear' in layer_type:
        method = compression.get('method', 'VBMF')
        rank = compression.get('rank', None)
        energy = compression.get('energy', 0.95)
        rank_cap = compression.get('rank_cap', rank)

        # For linear layers, rank doesn't directly apply to TT, but we can pass it as rank_cap
        num_param, num_param_decomp, ratio, new_layer = \
            tt.tensor_train_decomposition_linear_layer(
                layer,
                method=method,
                rank_cap=rank_cap,
                energy=energy
            )
        logger.debug(
            "TT Linear decomposition: %s -> %s params, ratio %.4f",
            num_param,
            num_param_decomp,
            ratio,
        )
    else:
        logger.debug("TT decomposition not supported for layer type: %s", layer_type)
        new_layer = layer  # Keep original layer

    return new_layer


def compress_layer(layer, compression):
    """
    apply compression method to specified layer
    """
    compression = check_compress_layer_input(compression)
    if compression is None:
        return None
    methods = {
        'partial_tucker': apply_partial_tucker,
        'cp2': apply_cp_linear,
        'cp3': apply_cp_conv,
        'cp4': apply_cp_conv_pdp,
        'tt': apply_tt_decomposition,
        'tensor_train': apply_tt_decomposition,
    }
    apply_method = methods.get(compression['type'])
    return apply_method(layer, compression) if apply_method else None
