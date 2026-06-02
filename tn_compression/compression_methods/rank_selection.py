import torch, tensorly as tl, numpy as np

from ..decompositions import VBMF
tl.set_backend('pytorch')

@torch.no_grad()
def evbmf_ranks_conv(weight):  # weight: (O, I, H, W)
    W = weight.detach().cpu().numpy()
    # unfold along modes 0 and 1
    U0 = tl.base.unfold(torch.from_numpy(W), 0).numpy()
    U1 = tl.base.unfold(torch.from_numpy(W), 1).numpy()
    _, d0, _, _ = VBMF.EVBMF(U0)
    _, d1, _, _ = VBMF.EVBMF(U1)
    r_out = int(d0.shape[0]); r_in = int(d1.shape[1])
    return max(1, r_out), max(1, r_in)

@torch.no_grad()
def energy_ranks_conv(weight, energy=0.94):
    def r_from_energy(svals, rho):
        e = torch.cumsum(svals**2, 0) / torch.sum(svals**2)
        k = int((e >= rho).nonzero(as_tuple=True)[0][0].item() + 1)
        return max(1, k)
    U0 = tl.base.unfold(weight, 0)
    U1 = tl.base.unfold(weight, 1)
    s0 = torch.linalg.svdvals(U0)
    s1 = torch.linalg.svdvals(U1)
    return r_from_energy(s0, energy), r_from_energy(s1, energy)

@torch.no_grad()
def energy_rank_linear(weight, energy=0.95):
    s = torch.linalg.svdvals(weight)
    e = torch.cumsum(s**2, 0) / torch.sum(s**2)
    k = int((e >= energy).nonzero(as_tuple=True)[0][0].item() + 1)
    return max(1, k)

@torch.no_grad()
def entropy_rank_from_svals(s, tau=0.94, eps=1e-12):
    """
    ARSVD entropy-based rank selection.
    Choose smallest k where partial entropy H(k) >= tau * H_total.
    """
    if s.numel() == 0:
        return 1
    
    s = torch.clamp(s, min=0)
    s_sum = torch.sum(s)
    if s_sum <= eps:
        return 1
    
    # Probabilities: p_i = s_i / sum(s_j) 
    p = s / s_sum
    
    # Avoid log(0) 
    plogp = torch.where(p > 0, p * torch.log(p + eps), torch.zeros_like(p))
    H_total = -torch.sum(plogp)
    
    if H_total <= eps:
        return 1
    
    # Cumulative partial entropy
    H_partial = -torch.cumsum(plogp, dim=0)
    
    # Find smallest k reaching tau * H_total
    idx = (H_partial >= tau * H_total).nonzero(as_tuple=True)[0]
    k = int(idx[0].item() + 1) if idx.numel() > 0 else s.numel()
    return max(1, k)

@torch.no_grad()
def entropy_rank_matrix(M, tau=0.94):
    """Entropy-based rank for matrix."""
    s = torch.linalg.svdvals(M)
    return entropy_rank_from_svals(s, tau)

@torch.no_grad()
def entropy_ranks_conv(weight, tau=0.94):
    """Entropy-based ranks for Conv2d using unfoldings."""
    U0 = tl.base.unfold(weight, 0)
    U1 = tl.base.unfold(weight, 1)
    s0 = torch.linalg.svdvals(U0)
    s1 = torch.linalg.svdvals(U1)
    return entropy_rank_from_svals(s0, tau), entropy_rank_from_svals(s1, tau)

@torch.no_grad()
def entropy_rank_linear(weight, tau=0.95):
    """Entropy-based rank for Linear layer."""
    s = torch.linalg.svdvals(weight)
    return entropy_rank_from_svals(s, tau)

def round_to_multiple(x, m=8):
    return int(max(m, m * round(x / m)))
