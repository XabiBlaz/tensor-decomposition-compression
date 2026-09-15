import pytest
import tensorly as tl
import torch
from torch import nn

from tn_compression.compression_methods._compression_methods_utils import compress_layer
from tn_compression.compression_methods.rank_selection import energy_rank_linear
from tn_compression.decompositions import cp, tt


@pytest.mark.parametrize("kind", ["partial_tucker", "tensor_train"])
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate", "circular"])
def test_full_rank_convolution_reconstructs_geometry_without_bias(kind, padding_mode):
    torch.manual_seed(3)
    layer = nn.Conv2d(3, 4, (3, 5), stride=(2, 3), padding=(2, 1),
                      dilation=(2, 1), bias=False, padding_mode=padding_mode).double().eval()
    inputs = torch.randn(2, 3, 17, 19, dtype=torch.float64)
    rank = [4, 3] if kind == "partial_tucker" else 100
    replacement = compress_layer(layer, {"type": kind, "rank": rank})
    torch.testing.assert_close(replacement(inputs), layer(inputs), rtol=1e-9, atol=1e-9)
    assert not any(name.endswith("bias") for name, _ in replacement.named_parameters())
    assert not replacement.training


@pytest.mark.parametrize("kind", ["cp3", "cp4"])
def test_cp_rank_one_reconstructs_real_decomposition(kind):
    torch.manual_seed(2)
    layer = nn.Conv2d(3, 4, (3, 5), stride=(2, 3), padding=(2, 1), dilation=(2, 1), bias=False).double()
    factors = [torch.randn(size, 1, dtype=torch.float64) for size in layer.weight.shape]
    with torch.no_grad():
        layer.weight.copy_(torch.einsum("or,ir,hr,wr->oihw", *factors))
    replacement = compress_layer(layer, {"type": kind, "rank": 1})
    inputs = torch.randn(2, 3, 17, 19, dtype=torch.float64)
    torch.testing.assert_close(replacement(inputs), layer(inputs), rtol=1e-8, atol=1e-8)
    assert len(replacement) == (3 if kind == "cp3" else 4)


def test_cp_preserves_nonunit_coefficients_and_bias(monkeypatch):
    torch.manual_seed(5)
    layer = nn.Conv2d(2, 3, 3, padding=1).double()
    factors = [torch.randn(size, 2, dtype=torch.float64) for size in layer.weight.shape]
    scales = torch.tensor([2.0, -0.5], dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.einsum("r,or,ir,hr,wr->oihw", scales, *factors))
    monkeypatch.setattr(tl.decomposition, "parafac", lambda *args, **kwargs: (scales, factors))
    inputs = torch.randn(2, 2, 9, 9, dtype=torch.float64)
    for split in (False, True):
        result = cp.cp_convolution(layer, 2, split_spatial=split)
        torch.testing.assert_close(result(inputs), layer(inputs), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("shape", [(6, 12), (7, 11), (1, 5)])
def test_true_tt_linear_preserves_axis_order_and_leading_dimensions(shape):
    torch.manual_seed(4)
    layer = nn.Linear(*shape).double()
    result = compress_layer(layer, {"type": "tensor_train", "energy": 1.0})
    assert isinstance(result, tt._TTLinearCore)
    for inputs in (torch.randn(2, 3, shape[0], dtype=torch.float64),
                   torch.randn(shape[0], dtype=torch.float64)):
        torch.testing.assert_close(result(inputs), layer(inputs), rtol=1e-9, atol=1e-9)
    result(torch.randn(2, shape[0], dtype=torch.float64)).sum().backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all() for core in result.cores)


def test_svd_alias_is_two_linear_layers_and_preserves_frozen_weights():
    layer = nn.Linear(5, 4).double().eval()
    layer.weight.requires_grad_(False)
    inputs = torch.randn(2, 3, 5, dtype=torch.float64)
    for name in ("svd", "cp2"):
        result = compress_layer(layer, {"type": name, "rank": 4})
        assert isinstance(result, nn.Sequential) and len(result) == 2
        assert not result[0].weight.requires_grad and result[1].bias.requires_grad
        torch.testing.assert_close(result(inputs), layer(inputs))


def test_plain_tt_eval_recomputes_after_weights_change_and_keeps_gradients():
    layer = nn.Conv2d(2, 3, 3, padding=1).double().eval()
    result = compress_layer(layer, {"type": "tensor_train", "structure": "PLAIN_TT", "rank": 100})
    inputs = torch.randn(2, 2, 8, 8, dtype=torch.float64)
    with torch.no_grad():
        before = result(inputs).clone()
        result.cores[0].add_(0.2)
    after = result(inputs)
    assert not torch.allclose(before, after)
    after.sum().backward()
    assert result.cores[0].grad is not None


def test_zero_weight_and_full_energy_rank():
    assert energy_rank_linear(torch.zeros(4, 3), energy=1.0) == 1
    assert energy_rank_linear(torch.eye(4), energy=1.0) == 4
    for invalid in (0, -1, 1.1):
        with pytest.raises(ValueError, match="threshold"):
            energy_rank_linear(torch.eye(4), invalid)


def test_decomposition_does_not_change_tensorly_backend():
    with tl.backend_context("numpy"):
        compress_layer(nn.Conv2d(2, 3, 3), {"type": "partial_tucker", "rank": 2})
        assert tl.get_backend() == "numpy"
