import logging

import torch
import torch.nn as nn
import numpy as np
import tensorly as tl
from tensorly.decomposition import tensor_train
from typing import Tuple, List, Optional, Union
import torch.nn.functional as F

# Set TensorLy backend to PyTorch
tl.set_backend('pytorch')
logger = logging.getLogger(__name__)


# ========================== Utility Functions =====================

def _round_to_multiple_of_8(x: int):
    """Round to nearest multiple of 8 for hardware efficiency."""
    return int(max(8, 8 * round(x / 8))) if x >= 8 else int(x)


def _factorize_balanced(n: int, num_factors: int):
    """
    Create balanced factorization of integer n into d factors.
    
    Args:
        n: Integer to factorize
        num_factors: Number of factors to create
        
    Returns:
        List of factors whose product equals n
    """
    factors = []
    remainder = int(n)
    
    for k in range(num_factors - 1, 0, -1):
        factor = max(2, int(round(remainder ** (1.0 / (k + 1)))))
        while remainder % factor != 0 and factor > 2:
            factor -= 1
        factors.append(factor)
        remainder //= factor
    
    factors.append(int(remainder))
    return factors


@torch.no_grad()
def _rank_from_energy(singular_values: torch.Tensor, energy_threshold: float):
    """
    Compute minimal rank such that cumulative energy >= threshold.
    
    Args:
        singular_values: 1D tensor of singular values
        energy_threshold: Energy threshold in [0, 1]
        
    Returns:
        Optimal rank based on energy criterion
    """
    if singular_values.numel() == 0:
        return 1
    
    energy_cumsum = torch.cumsum(singular_values**2, dim=0) / torch.sum(singular_values**2)
    indices = (energy_cumsum >= energy_threshold).nonzero(as_tuple=True)[0]
    
    if indices.numel() == 0:
        return singular_values.numel()
    
    rank = int(indices[0].item() + 1)
    return max(1, rank)


@torch.no_grad()
def _evbmf_ranks_conv(weight: torch.Tensor):
    """
    EVBMF-based rank estimation for Conv2d weight tensor.
    
    Args:
        weight: Conv2d weight tensor with shape (O, I, H, W)
        
    Returns:
        Tuple of (output_rank, input_rank) or (None, None) if EVBMF unavailable
    """
    try:
        from . import VBMF
        
        weight_np = weight.detach().cpu().numpy()
        # Unfold along output and input channels
        unfolded_output = tl.base.unfold(torch.from_numpy(weight_np), 0).numpy()
        unfolded_input = tl.base.unfold(torch.from_numpy(weight_np), 1).numpy()
        
        _, diag_output, _, _ = VBMF.EVBMF(unfolded_output)
        _, diag_input, _, _ = VBMF.EVBMF(unfolded_input)
        
        rank_output = max(1, int(diag_output.shape[0]))
        rank_input = max(1, int(diag_input.shape[0]))
        
        return rank_output, rank_input
    except ImportError:
        return None, None


# ========================== TT-SVD Algorithms =====================

@torch.no_grad()
def _tt_svd_energy(tensor: torch.Tensor, energy_threshold: float = 0.94, 
                   rank_cap: Optional[int] = None):
    """
    TT-SVD with energy-based rank selection (Oseledets algorithm).
    
    Args:
        tensor: Input tensor to decompose
        energy_threshold: Energy threshold for each decomposition step
        rank_cap: Optional maximum rank constraint
        
    Returns:
        List of TT cores as 3-way tensors
    """
    device, dtype = tensor.device, tensor.dtype
    tensor_dims = list(tensor.shape)
    num_modes = len(tensor_dims)
    
    cores = []
    current_matrix = tensor
    previous_rank = 1
    
    # Perform SVD for each mode except the last
    for mode_idx in range(num_modes - 1):
        current_mode_size = tensor_dims[mode_idx]
        remaining_size = int(np.prod(tensor_dims[mode_idx + 1:])) if mode_idx + 1 < num_modes else 1
        
        # Reshape for SVD
        matrix_for_svd = current_matrix.reshape(previous_rank * current_mode_size, remaining_size)
        
        # Compute SVD
        u_matrix, singular_values, vh_matrix = torch.linalg.svd(matrix_for_svd, full_matrices=False)
        
        # Determine rank using energy criterion
        current_rank = _rank_from_energy(singular_values, energy_threshold)
        
        if rank_cap is not None:
            current_rank = min(current_rank, rank_cap)
        
        # Truncate to selected rank
        u_truncated = u_matrix[:, :current_rank]
        singular_values_truncated = singular_values[:current_rank]
        vh_truncated = vh_matrix[:current_rank, :]
        
        # Store core with shape (previous_rank, mode_size, current_rank)
        core = u_truncated.reshape(previous_rank, current_mode_size, current_rank)
        cores.append(core.to(device=device, dtype=dtype))
        
        # Update matrix for next iteration
        current_matrix = (singular_values_truncated.unsqueeze(1) * vh_truncated)
        current_matrix = current_matrix.reshape(current_rank, *tensor_dims[mode_idx + 1:])
        previous_rank = current_rank
    
    # Last core with trailing rank 1
    final_core = current_matrix.reshape(previous_rank, tensor_dims[-1], 1)
    cores.append(final_core.to(device=device, dtype=dtype))
    
    return cores


@torch.no_grad()
def _tt_svd_entropy(tensor: torch.Tensor, entropy_threshold: float = 0.94, 
                    rank_cap: Optional[int] = None):
    """
    TT-SVD with entropy-based rank selection at each step.
    
    Args:
        tensor: Input tensor to decompose
        entropy_threshold: Entropy threshold for rank selection
        rank_cap: Optional maximum rank constraint
        
    Returns:
        List of TT cores as 3-way tensors
    """
    device, dtype = tensor.device, tensor.dtype
    tensor_dims = list(tensor.shape)
    num_modes = len(tensor_dims)
    
    cores = []
    current_matrix = tensor
    previous_rank = 1
    
    for mode_idx in range(num_modes - 1):
        current_mode_size = tensor_dims[mode_idx]
        remaining_size = int(np.prod(tensor_dims[mode_idx + 1:])) if mode_idx + 1 < num_modes else 1
        
        # Reshape for SVD
        matrix_for_svd = current_matrix.reshape(previous_rank * current_mode_size, remaining_size)
        
        # Compute SVD
        u_matrix, singular_values, vh_matrix = torch.linalg.svd(matrix_for_svd, full_matrices=False)
        
        # Entropy-based rank selection
        from ..compression_methods.rank_selection import entropy_rank_from_svals
        current_rank = entropy_rank_from_svals(singular_values, tau=entropy_threshold)
        
        if rank_cap is not None:
            current_rank = min(current_rank, rank_cap)
        
        # Truncate and store core
        u_truncated = u_matrix[:, :current_rank]
        core = u_truncated.reshape(previous_rank, current_mode_size, current_rank)
        cores.append(core.to(device=device, dtype=dtype))
        
        # Update for next iteration
        current_matrix = (singular_values[:current_rank].unsqueeze(1) * vh_matrix[:current_rank, :])
        current_matrix = current_matrix.reshape(current_rank, *tensor_dims[mode_idx + 1:])
        previous_rank = current_rank
    
    # Final core
    final_core = current_matrix.reshape(previous_rank, tensor_dims[-1], 1)
    cores.append(final_core.to(device=device, dtype=dtype))
    
    return cores


# ========================== Conv2d TT Decomposition =====================

class _TTConvPlain(nn.Module):
    """
    Plain TT-SVD Conv2d wrapper.
    Stores TT cores of weight tensor (O,I,H,W) decomposed as (n1=O, n2=I, n3=H, n4=W).
    Reconstructs weight; caches it in eval mode. Cache invalidated on grad updates
    or device/dtype mismatch.
    """
    def __init__(self, cores: List[torch.Tensor], stride, padding, dilation, bias: Optional[torch.Tensor]):
        super().__init__()
        self.cores = nn.ParameterList([nn.Parameter(c) for c in cores])
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None
        self._cached_weight: Optional[torch.Tensor] = None
        for p in self.cores:
            p.register_hook(lambda *_: self._invalidate_cache())

    def _invalidate_cache(self):
        self._cached_weight = None

    def _reconstruct_weight(self):
        # Cores: [(1,O,r1),(r1,I,r2),(r2,H,r3),(r3,W,1)]
        G = self.cores[0][0]  # (O, r1)
        for k in range(1, len(self.cores)-1):
            G = torch.tensordot(G, self.cores[k], dims=([-1],[0]))  # (..., n_k, r_k)
        G = torch.tensordot(G, self.cores[-1], dims=([-1],[0]))     # (..., W, 1)
        O = self.cores[0].shape[1]
        I = self.cores[1].shape[1]
        H = self.cores[2].shape[1]
        W = self.cores[3].shape[1]
        return G[..., 0].view(O, I, H, W).contiguous()

    def forward(self, x):
        if (self.training or
            self._cached_weight is None or
            self._cached_weight.device != x.device or
            self._cached_weight.dtype != x.dtype):
            self._cached_weight = self._reconstruct_weight().to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, self._cached_weight, bias=self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation)


@torch.no_grad()
def _build_plain_tt_from_weight(weight: torch.Tensor,
                                method: str,
                                energy: float,
                                rank_cap: Optional[int],
                                manual_ranks: Optional[Union[int,List[int]]],
                                device,
                                bias: Optional[torch.Tensor]):
    """
    Decompose weight (O,I,H,W) into plain TT cores (1,O,r1),(r1,I,r2),(r2,H,r3),(r3,W,1)
    using existing rank methods.
    """
    O, I, H, W = weight.shape
    dims = [O, I, H, W]

    # Manual ranks?
    if manual_ranks is not None:
        if isinstance(manual_ranks, int):
            r1 = r2 = r3 = max(1, int(manual_ranks))
        else:
            assert len(manual_ranks) == 3, "Manual ranks list must have length 3"
            r1, r2, r3 = [max(1,int(r)) for r in manual_ranks]
        # Basic feasibility clamps
        r1 = min(r1, min(O, I*H*W))
        r2 = min(r2, min(I*r1, H*W))
        r3 = min(r3, min(H*r2, W))
        tt_ranks = [1, r1, r2, r3, 1]
        # Use tensorly to factor with fixed ranks
        decomp = tensor_train(weight, rank=tt_ranks)
        cores = list(getattr(decomp, 'cores', getattr(decomp,'factors', decomp)))
        cores_t = [torch.as_tensor(c, device=device, dtype=weight.dtype) for c in cores]
        return cores_t, tt_ranks[1:4]

    # Automatic ranks: use existing methods on sequential TT-SVD
    if method.upper() == 'SVD':
        cores = _tt_svd_energy(weight, energy_threshold=energy, rank_cap=rank_cap)
    elif method.upper() == 'ENTROPY':
        cores = _tt_svd_entropy(weight, entropy_threshold=energy, rank_cap=rank_cap)
    elif method.upper() == 'VBMF':
        # VBMF fallback -> single uniform rank
        matrix_flat = weight.view(O*H, W*I)
        svals = torch.linalg.svdvals(matrix_flat)
        r_uni = _rank_from_energy(svals, energy)
        if rank_cap is not None:
            r_uni = min(r_uni, rank_cap)
        tt_ranks = [1, r_uni, r_uni, r_uni, 1]
        decomp = tensor_train(weight, rank=tt_ranks)
        factors = getattr(decomp, 'cores', getattr(decomp,'factors', decomp))
        cores = [torch.as_tensor(f, device=device, dtype=weight.dtype) for f in factors]
        return cores, tt_ranks[1:4]
    else:
        raise ValueError(f"Unknown method {method} for plain TT")

    # cores already 3-way with trailing rank 1 final
    r1, r2, r3 = cores[0].shape[-1], cores[1].shape[-1], cores[2].shape[-1]
    return cores, [r1, r2, r3]


@torch.no_grad()
def tensor_train_decomposition_conv_layer_plain_tt(
    layer: nn.Conv2d,
    ranks: Optional[Union[int, List[int]]] = None,
    method: str = 'SVD',
    energy: float = 0.94,
    rank_cap: Optional[int] = None,
    round_to_8: bool = False,
    target_ratio_: Optional[float] = None
):
    """
    Plain TT-SVD structure for Conv2d.
    Decomposes Conv2d weight (O,I,H,W) directly into TT cores and applies convolution
    by reconstructing the weight from cores at forward time.

    Returns:
        Tuple of (original_params, tt_params, compression_ratio, module)
    """
    assert isinstance(layer, nn.Conv2d), "Input must be nn.Conv2d"
    assert layer.groups == 1, "Plain TT decomposition supports groups=1"

    weight = layer.weight.data
    device, dtype = weight.device, weight.dtype
    O, I, H, W = weight.shape

    # Build TT cores using the same rank selection methods as TTPWT
    cores, auto_ranks = _build_plain_tt_from_weight(
        weight, method, energy, rank_cap, ranks, device, layer.bias
    )

    # Manual ranks rounding to 8 (mirror behavior: rounding only for manual ranks)
    if ranks is not None and round_to_8:
        r1, r2, r3 = auto_ranks
        r1 = _round_to_multiple_of_8(r1)
        r2 = _round_to_multiple_of_8(r2)
        r3 = _round_to_multiple_of_8(r3)
        # Clamp feasibility (avoid overshoot after rounding)
        O, I, H, W = weight.shape
        r1 = min(r1, O)
        r2 = min(r2, I * r1)
        r3 = min(r3, H * r2, W)
        # Rebuild cores if changed
        if [r1, r2, r3] != auto_ranks:
            tt_ranks = [1, r1, r2, r3, 1]
            decomp = tensor_train(weight, rank=tt_ranks)
            factors = getattr(decomp, 'cores', None) or getattr(decomp, 'factors', None) or decomp
            cores = [torch.as_tensor(f, device=device, dtype=dtype).contiguous() for f in factors]
            auto_ranks = [r1, r2, r3]

    # Wrap into a module that applies conv2d
    module = _TTConvPlain(
        cores,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=layer.bias
    ).to(device=device, dtype=dtype)

    # Parameter counts
    original_params = int(weight.numel()) + (layer.bias.numel() if layer.bias is not None else 0)
    tt_params = sum(int(c.numel()) for c in cores) + (layer.bias.numel() if layer.bias is not None else 0)
    compression_ratio = float(tt_params) / float(original_params)

    r1, r2, r3 = auto_ranks
    logger.debug(
        "%s PLAIN_TT estimated TT ranks [%s, %s, %s], params %s / %s, ratio %.4f",
        layer,
        r1,
        r2,
        r3,
        tt_params,
        original_params,
        compression_ratio,
    )

    return original_params, tt_params, compression_ratio, module


@torch.no_grad()
def tensor_train_decomposition_conv_layer(
    layer: nn.Conv2d,
    ranks: Optional[Union[int, List[int]]] = None,
    method: str = 'SVD',
    energy: float = 0.94,
    rank_cap: Optional[int] = None,
    round_to_8: bool = False,
    target_ratio_: Optional[float] = None,
    structure: str = 'TTPWT'   # 'TTPWT' (existing) or 'PLAIN_TT'
):
    """
    TT decomposition for Conv2d layer using TTPWT method.
    
    Decomposes Conv2d(O, I, H, W) into 4 sequential convolutions:
      1) Conv 1×1: I → r1
      2) Conv H×1: r1 → r2 (with height stride/padding)
      3) Conv 1×W: r2 → r3 (with width stride/padding)  
      4) Conv 1×1: r3 → O (with bias if present)
    
    Args:
        layer: Input Conv2d layer to decompose
        ranks: Manual rank specification (None for auto, int for uniform, list for per-mode)
        method: Rank selection method ('SVD', 'ENTROPY', or 'VBMF')
        energy: Energy/entropy threshold for automatic rank selection
        rank_cap: Maximum rank constraint
        round_to_8: Whether to round ranks to multiples of 8
        target_ratio_: Unused, kept for API compatibility
        structure: Decomposition structure ('TTPWT' or 'PLAIN_TT')
        
    Returns:
        Tuple of (original_params, tt_params, compression_ratio, sequential_model)
    """
    assert isinstance(layer, nn.Conv2d), "Input must be nn.Conv2d"
    assert layer.groups == 1, "TT decomposition only supports groups=1"

    if structure.upper() == 'PLAIN_TT':
        return tensor_train_decomposition_conv_layer_plain_tt(
            layer=layer,
            ranks=ranks,
            method=method,
            energy=energy,
            rank_cap=rank_cap,
            round_to_8=round_to_8,
            target_ratio_=target_ratio_)
    
    # Extract layer parameters
    weight = layer.weight.data
    device, dtype = weight.device, weight.dtype
    output_channels, input_channels, kernel_height, kernel_width = weight.shape
    stride_h, stride_w = layer.stride
    padding_h, padding_w = layer.padding
    dilation_h, dilation_w = layer.dilation
    
    # Apply permutation: (O,I,H,W) → (I,H,W,O) as per TTPWT paper
    weight_permuted = weight.permute(1, 2, 3, 0).contiguous()
    
    # ========== Rank Selection ==========
    if ranks is None:
        rank_1, rank_2, rank_3, cores = _select_automatic_ranks(
            weight, weight_permuted, method, energy, rank_cap, layer
        )
    else:
        rank_1, rank_2, rank_3, cores = _apply_manual_ranks(
            weight, weight_permuted, ranks, input_channels, kernel_height, 
            kernel_width, round_to_8, layer
        )
    
    # ========== Create TT Cores ==========
    if cores is None:
        # Use TensorLy with determined ranks
        tt_ranks = [1, rank_1, rank_2, rank_3, 1]
        tt_decomposition = tensor_train(weight_permuted, rank=tt_ranks)
        
        factors = getattr(tt_decomposition, 'cores', None) or \
                 getattr(tt_decomposition, 'factors', None) or tt_decomposition
        
        core_1 = torch.as_tensor(factors[0], device=device, dtype=dtype)
        core_2 = torch.as_tensor(factors[1], device=device, dtype=dtype)
        core_3 = torch.as_tensor(factors[2], device=device, dtype=dtype)
        core_4 = torch.as_tensor(factors[3], device=device, dtype=dtype)
    else:
        core_1, core_2, core_3, core_4 = cores
    
    # ========== Build Sequential Conv2d Layers ==========
    conv_layers = _build_tt_conv_layers(
        core_1, core_2, core_3, core_4,
        input_channels, output_channels, kernel_height, kernel_width,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        layer.bias, device, dtype
    )
    
    # ========== Calculate Compression Statistics ==========
    original_params = int(weight.numel())
    if layer.bias is not None:
        original_params += int(layer.bias.numel())
    
    tt_params = int(input_channels * rank_1 + rank_1 * kernel_height * rank_2 + 
                   rank_2 * kernel_width * rank_3 + rank_3 * output_channels)
    if layer.bias is not None:
        tt_params += int(layer.bias.numel())
    
    compression_ratio = float(tt_params) / float(original_params)
    
    return original_params, tt_params, compression_ratio, conv_layers


def _select_automatic_ranks(weight: torch.Tensor, weight_permuted: torch.Tensor, 
                          method: str, energy: float, rank_cap: Optional[int], 
                          layer: nn.Conv2d):
    """Select ranks automatically based on the specified method."""
    output_channels, input_channels, kernel_height, kernel_width = weight.shape
    
    if method.upper() == 'SVD':
        cores = _tt_svd_energy(weight_permuted, energy_threshold=energy, rank_cap=rank_cap)
        rank_1 = cores[0].shape[-1]
        rank_2 = cores[1].shape[-1]
        rank_3 = cores[2].shape[-1]
        logger.debug("%s SVD energy-based estimated TT ranks [%s, %s, %s]", layer, rank_1, rank_2, rank_3)
        return rank_1, rank_2, rank_3, cores
        
    elif method.upper() == 'ENTROPY':
        cores = _tt_svd_entropy(weight_permuted, entropy_threshold=energy, rank_cap=rank_cap)
        rank_1 = cores[0].shape[-1]
        rank_2 = cores[1].shape[-1]
        rank_3 = cores[2].shape[-1]
        logger.debug("%s entropy-based estimated TT ranks [%s, %s, %s]", layer, rank_1, rank_2, rank_3)
        return rank_1, rank_2, rank_3, cores
        
    elif method.upper() == 'VBMF':
        rank_output, rank_input = _evbmf_ranks_conv(weight)
        
        if rank_output is None:
            # Fallback to energy-based method
            matrix_flat = weight.view(output_channels * kernel_height, kernel_width * input_channels)
            singular_values = torch.linalg.svdvals(matrix_flat)
            rank = _rank_from_energy(singular_values, energy)
            logger.debug("%s energy-based fallback estimated rank %s", layer, rank)
        else:
            rank = min(rank_output, rank_input)
            logger.debug("%s VBMF estimated ranks (%s, %s), using %s", layer, rank_output, rank_input, rank)
        
        if rank_cap is not None:
            rank = min(rank, rank_cap)
        
        return rank, rank, rank, None
        
    else:
        raise ValueError(f"Unknown method: {method}")


def _apply_manual_ranks(weight: torch.Tensor, weight_permuted: torch.Tensor, 
                       ranks: Union[int, List[int]], input_channels: int, 
                       kernel_height: int, kernel_width: int, round_to_8: bool,
                       layer: nn.Conv2d):
    """Apply manually specified ranks with constraints."""
    if isinstance(ranks, int):
        # Uniform ranks with automatic constraints
        base_rank = max(1, int(ranks))
        
        # Apply layer-specific constraints to avoid impossible ranks
        max_rank_1 = min(input_channels, base_rank)
        max_rank_2 = min(kernel_height * max_rank_1, base_rank)
        max_rank_3 = min(kernel_width * max_rank_2, base_rank)
        
        rank_1 = max_rank_1
        rank_2 = min(max_rank_2, base_rank)
        rank_3 = min(max_rank_3, base_rank)
        
        logger.debug("%s fixed rank %s adjusted to TT ranks [%s, %s, %s]", layer, base_rank, rank_1, rank_2, rank_3)
    else:
        # Custom ranks per mode with constraints
        assert len(ranks) == 3, "Need exactly 3 ranks for TT decomposition"
        base_rank_1, base_rank_2, base_rank_3 = [max(1, int(r)) for r in ranks]
        
        rank_1 = min(base_rank_1, input_channels)
        rank_2 = min(base_rank_2, kernel_height * rank_1)
        rank_3 = min(base_rank_3, kernel_width * rank_2)
        
        if (rank_1, rank_2, rank_3) != (base_rank_1, base_rank_2, base_rank_3):
            logger.debug("%s ranks %s adjusted to TT ranks [%s, %s, %s]", layer, ranks, rank_1, rank_2, rank_3)
        else:
            logger.debug("%s using custom TT ranks [%s, %s, %s]", layer, rank_1, rank_2, rank_3)
    
    # Apply rounding if requested
    if round_to_8:
        rank_1 = _round_to_multiple_of_8(rank_1)
        rank_2 = _round_to_multiple_of_8(rank_2)
        rank_3 = _round_to_multiple_of_8(rank_3)
    
    return rank_1, rank_2, rank_3, None


def _build_tt_conv_layers(core_1: torch.Tensor, core_2: torch.Tensor, 
                         core_3: torch.Tensor, core_4: torch.Tensor,
                         input_channels: int, output_channels: int,
                         kernel_height: int, kernel_width: int,
                         stride_h: int, stride_w: int,
                         padding_h: int, padding_w: int,
                         dilation_h: int, dilation_w: int,
                         bias: Optional[torch.Tensor],
                         device: torch.device, dtype: torch.dtype):
    """Build the four sequential Conv2d layers from TT cores."""
    
    rank_1 = core_1.shape[-1]
    rank_2 = core_2.shape[-1]
    rank_3 = core_3.shape[-1]
    
    # Layer 1: 1×1 convolution I → r1
    conv_1 = nn.Conv2d(input_channels, rank_1, kernel_size=1, stride=1, padding=0, bias=False)
    conv_1.weight.data = core_1[0, :, :].T.contiguous().view(rank_1, input_channels, 1, 1)
    
    # Layer 2: H×1 convolution r1 → r2 (vertical, applies height stride/padding)
    conv_2 = nn.Conv2d(rank_1, rank_2, kernel_size=(kernel_height, 1),
                      stride=(stride_h, 1), padding=(padding_h, 0), 
                      dilation=(dilation_h, 1), bias=False)
    conv_2.weight.data = core_2.permute(2, 0, 1).contiguous().view(rank_2, rank_1, kernel_height, 1)
    
    # Layer 3: 1×W convolution r2 → r3 (horizontal, applies width stride/padding)
    conv_3 = nn.Conv2d(rank_2, rank_3, kernel_size=(1, kernel_width),
                      stride=(1, stride_w), padding=(0, padding_w), 
                      dilation=(1, dilation_w), bias=False)
    conv_3.weight.data = core_3.permute(2, 0, 1).contiguous().view(rank_3, rank_2, 1, kernel_width)
    
    # Layer 4: 1×1 convolution r3 → O (includes bias if present)
    conv_4 = nn.Conv2d(rank_3, output_channels, kernel_size=1, stride=1, padding=0,
                      bias=(bias is not None))
    conv_4.weight.data = core_4[:, :, 0].T.contiguous().view(output_channels, rank_3, 1, 1)
    
    if bias is not None:
        conv_4.bias.data = bias.data.clone()
    
    # Move all layers to correct device and dtype
    conv_layers = [conv_1, conv_2, conv_3, conv_4]
    for conv in conv_layers:
        conv.to(device=device, dtype=dtype)
    
    return nn.Sequential(*conv_layers)


# ========================== Linear TT Decomposition =====================

class _TTLinearCore(nn.Module):
    """TT-matrix linear layer with cores G_k of shape (r_{k-1}, o_k, i_k, r_k)."""
    
    def __init__(self, input_dims: List[int], output_dims: List[int], 
                 cores: List[torch.Tensor], bias: Optional[torch.Tensor] = None):
        super().__init__()
        assert len(input_dims) == len(output_dims) == len(cores)
        
        self.input_dims = [int(x) for x in input_dims]
        self.output_dims = [int(x) for x in output_dims]
        self.in_features = int(np.prod(self.input_dims))
        self.out_features = int(np.prod(self.output_dims))
        
        self.cores = nn.ParameterList([nn.Parameter(core) for core in cores])
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x: torch.Tensor):
        """Forward pass through TT-matrix multiplication."""
        batch_size = x.size(0)
        # Initialize with rank dimension
        y = x.view(batch_size, 1, self.in_features)

        tail_size = self.in_features
        for core, (output_k, input_k) in zip(self.cores, zip(self.output_dims, self.input_dims)):
            rank_prev, o_k, i_k, rank_next = core.shape
            assert i_k == input_k and o_k == output_k

            # Reshape to (batch_size, rank_prev * i_k, tail_size / i_k)
            assert tail_size % i_k == 0, "Input dimensions factorization mismatch"
            y = y.view(batch_size, rank_prev * i_k, tail_size // i_k)

            # Core as matrix: (rank_prev * i_k, rank_next * o_k)
            core_matrix = core.permute(0, 2, 3, 1).reshape(rank_prev * i_k, rank_next * o_k)

            # Matrix multiplication and reshape
            y = torch.matmul(y.transpose(1, 2), core_matrix).transpose(1, 2)

            # Update tail size and reshape
            tail_size = (tail_size // i_k) * o_k
            y = y.view(batch_size, rank_next, tail_size)

        y = y.squeeze(1)  # Remove rank dimension
        
        if self.bias is not None:
            y = y + self.bias
            
        return y


@torch.no_grad()
def tensor_train_decomposition_linear_layer_tt(
    layer: nn.Linear,
    energy: float = 0.95,
    num_cores: int = 3,
    rank_cap: Optional[int] = None,
    round_to_8: bool = False
):
    """
    True TT-matrix decomposition for Linear layer via TT-SVD.
    
    Args:
        layer: Input Linear layer to decompose
        energy: Energy threshold for per-step rank selection
        num_cores: Number of TT cores to create
        rank_cap: Optional hard cap on per-step TT ranks
        round_to_8: Whether to round ranks to multiples of 8
        
    Returns:
        Tuple of (original_params, tt_params, compression_ratio, tt_module)
    """
    weight = layer.weight.data
    device = weight.device
    output_features, input_features = weight.shape
    num_factors = max(2, int(num_cores))

    # Factorize dimensions into balanced factors
    output_factors = _factorize_balanced(output_features, num_factors)
    input_factors = _factorize_balanced(input_features, num_factors)
    
    assert int(np.prod(output_factors)) == output_features
    assert int(np.prod(input_factors)) == input_features

    # Build interleaved tensor: (o1, i1, o2, i2, ..., od, id)
    interleaved_shape = [x for pair in zip(output_factors, input_factors) for x in pair]
    interleaved_tensor = weight.view(*interleaved_shape)
    total_modes = 2 * num_factors

    # TT-SVD across interleaved modes with per-step energy threshold
    cores_3way = []
    current_matrix = interleaved_tensor
    previous_rank = 1
    remaining_dims = list(interleaved_shape)

    # Perform TT-SVD for all modes except the last
    for mode_idx in range(total_modes - 1):
        current_mode_size = remaining_dims[0]
        rest_size = int(np.prod(remaining_dims[1:])) if len(remaining_dims) > 1 else 1
        
        matrix_for_svd = current_matrix.reshape(previous_rank * current_mode_size, rest_size)
        u_matrix, singular_values, vh_matrix = torch.linalg.svd(matrix_for_svd, full_matrices=False)
        
        current_rank = _rank_from_energy(singular_values, energy)
        if rank_cap is not None:
            current_rank = min(current_rank, rank_cap)
        if round_to_8 and current_rank >= 8:
            current_rank = _round_to_multiple_of_8(current_rank)

        u_truncated = u_matrix[:, :current_rank]
        cores_3way.append(u_truncated.reshape(previous_rank, current_mode_size, current_rank))
        
        current_matrix = (singular_values[:current_rank].unsqueeze(1) * vh_matrix[:current_rank, :])
        current_matrix = current_matrix.reshape(current_rank, *remaining_dims[1:])
        
        previous_rank = current_rank
        remaining_dims.pop(0)

    # Final core with trailing rank 1
    final_mode_size = remaining_dims[0]
    cores_3way.append(current_matrix.reshape(previous_rank, final_mode_size, 1))

    # Regroup 3-way cores into TT-matrix cores G_k: (r_{k-1}, o_k, i_k, r_k)
    tt_cores = []
    for k in range(num_factors):
        core_a = cores_3way[2 * k]      # (r_{2k}, o_k, r_mid)
        core_b = cores_3way[2 * k + 1]  # (r_mid, i_k, r_next)
        
        # Combine cores using Einstein summation
        combined_core = torch.einsum('a o r, r i b -> a o i b', core_a, core_b).contiguous()
        tt_cores.append(combined_core.to(device))

    # Build TT module and calculate parameters
    bias = layer.bias.data if layer.bias is not None else None
    tt_module = _TTLinearCore(input_factors, output_factors, tt_cores, bias=bias).to(device)

    original_params = int(weight.numel() + (layer.bias.numel() if layer.bias is not None else 0))
    tt_params = sum(int(core.numel()) for core in tt_cores)
    if layer.bias is not None:
        tt_params += layer.bias.numel()
    
    compression_ratio = float(tt_params) / float(original_params)
    
    return original_params, tt_params, compression_ratio, tt_module


@torch.no_grad()
def tensor_train_decomposition_linear_layer(
    layer: nn.Linear,
    method: str = 'SVD',
    energy: float = 0.95,
    rank_cap: Optional[int] = None,
    round_to_8: bool = False,
    target_ratio_: Optional[float] = None,
    num_cores: int = 3
):
    """
    TT decomposition for Linear layer with multiple method options.
    
    Args:
        layer: Input Linear layer to decompose
        method: Decomposition method ('SVD', 'ENTROPY', or 'TT')
        energy: Energy/entropy threshold for rank selection
        rank_cap: Optional rank constraint
        round_to_8: Whether to round ranks to multiples of 8
        target_ratio_: Unused, kept for API compatibility
        num_cores: Number of cores for TT-matrix decomposition
        
    Returns:
        Tuple of (original_params, decomposed_params, compression_ratio, decomposed_module)
    """
    if method.upper() == 'TT':
        logger.debug("%s using TT-matrix decomposition with %s cores", layer, num_cores)
        return tensor_train_decomposition_linear_layer_tt(
            layer, energy=energy, num_cores=num_cores,
            rank_cap=rank_cap, round_to_8=round_to_8
        )
    
    weight = layer.weight.data
    device = weight.device
    
    # Rank selection based on method
    singular_values = torch.linalg.svdvals(weight)
    
    if method.upper() == 'ENTROPY':
        from ..compression_methods.rank_selection import entropy_rank_from_svals
        rank = entropy_rank_from_svals(singular_values, tau=energy)
        logger.debug("%s entropy-based estimated rank %s", layer, rank)
    else:  # Default SVD (energy-based)
        energy_cumsum = torch.cumsum(singular_values ** 2, 0) / torch.sum(singular_values ** 2)
        rank = int((energy_cumsum >= energy).nonzero(as_tuple=True)[0][0].item() + 1)
        logger.debug("%s SVD energy-based estimated rank %s", layer, rank)
    
    # Apply constraints
    if rank_cap is not None:
        rank = min(rank, rank_cap)
    if round_to_8:
        rank = _round_to_multiple_of_8(rank)

    # Perform low-rank SVD decomposition
    u_matrix, singular_values_full, vh_matrix = torch.linalg.svd(weight, full_matrices=False)
    u_truncated = u_matrix[:, :rank]
    singular_values_truncated = singular_values_full[:rank]
    vh_truncated = vh_matrix[:rank, :]
    
    # Create balanced factorization: A = sqrt(S) * V^T, B = U * sqrt(S)
    sqrt_singular = torch.diag(torch.sqrt(singular_values_truncated))
    matrix_a = (sqrt_singular @ vh_truncated).to(device)      # (rank, input_features)
    matrix_b = (u_truncated @ sqrt_singular).to(device)       # (output_features, rank)

    # Build two-layer sequential model
    layer_1 = nn.Linear(layer.in_features, rank, bias=False).to(device)
    layer_2 = nn.Linear(rank, layer.out_features, bias=(layer.bias is not None)).to(device)
    
    layer_1.weight.data = matrix_a
    layer_2.weight.data = matrix_b
    
    if layer.bias is not None:
        layer_2.bias.data = layer.bias.data

    # Calculate compression statistics
    original_params = int(layer.in_features * layer.out_features)
    decomposed_params = int(layer.in_features * rank + rank * layer.out_features)
    
    if layer.bias is not None:
        bias_params = int(layer.bias.numel())
        original_params += bias_params
        decomposed_params += bias_params
    
    compression_ratio = float(decomposed_params) / float(original_params)
    
    return original_params, decomposed_params, compression_ratio, nn.Sequential(layer_1, layer_2)
