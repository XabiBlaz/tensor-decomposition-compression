import pytest
import torch

from tn_compression.quantization import round_to_nearest


def test_rounding_preserves_bias_zero_groups_and_partial_group():
    layer = torch.nn.Linear(7, 3)
    with torch.no_grad():
        layer.weight[0].zero_()
    before = layer.weight.detach().clone()
    rounded = round_to_nearest(layer, group_size=4)
    torch.testing.assert_close(layer.weight, before)
    torch.testing.assert_close(layer.bias, rounded.bias)
    assert torch.isfinite(rounded.weight).all() and torch.count_nonzero(rounded.weight[0]) == 0
    for start in (0, 4):
        block = before[:, start:start + 4]
        bound = block.abs().amax(1, keepdim=True) / 14
        assert ((rounded.weight[:, start:start + 4] - block).abs() <= bound + 1e-7).all()
    assert rounded.weight.numel() == layer.weight.numel()


def test_rounding_rejects_nonfinite_weights():
    layer = torch.nn.Linear(2, 2)
    with torch.no_grad():
        layer.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        round_to_nearest(layer)
