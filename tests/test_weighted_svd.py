import pytest
import torch

from tn_compression.calibration import reconstruction_error
from tn_compression.decompositions.cp import cp_linear
from tn_compression.decompositions.weighted_svd import weighted_svd


def test_weighted_svd_optimizes_observed_outputs():
    layer = torch.nn.Linear(3, 3, bias=True, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.diag(torch.tensor([5., 2., 1.])))
    inputs = torch.diag(torch.tensor([0.01, 1., 10.], dtype=torch.float64))
    ordinary = cp_linear(layer, 1)
    weighted = weighted_svd(layer, inputs, 1)
    assert reconstruction_error(layer, weighted, inputs)["squared_error"] < reconstruction_error(layer, ordinary, inputs)["squared_error"]
    full = weighted_svd(layer, inputs, 3)
    torch.testing.assert_close(full(inputs), layer(inputs), rtol=1e-10, atol=1e-10)


def test_singular_calibration_is_regularized_and_size_guard_is_explicit():
    layer = torch.nn.Linear(4, 5).double()
    inputs = torch.zeros(2, 4, dtype=torch.float64)
    result = weighted_svd(layer, inputs, 2)
    assert torch.isfinite(result(torch.ones_like(inputs))).all()
    with pytest.raises(ValueError, match="max_features"):
        weighted_svd(layer, inputs, 2, max_features=3)
