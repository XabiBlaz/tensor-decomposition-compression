import copy

import pytest
import torch

from tn_compression.calibration import candidate_intervention, collect_linear_inputs, reconstruction_error


def test_failed_intervention_restores_module_modes_buffers_and_rng():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).train()
    original, before = model[0], copy.deepcopy(model.state_dict())
    candidate = torch.nn.Linear(4, 4)
    rng = torch.random.get_rng_state()
    with pytest.raises(RuntimeError):
        with candidate_intervention(model, "0", candidate):
            model[1].running_mean.add_(7)
            torch.rand(3)
            raise RuntimeError("trial failed")
    assert model[0] is original and model.training and model[1].training
    for name, tensor in before.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_reconstruction_matches_quadratic_identity():
    layer = torch.nn.Linear(4, 3, dtype=torch.float64)
    candidate = copy.deepcopy(layer)
    samples = torch.randn(11, 4, dtype=torch.float64)
    with torch.no_grad():
        candidate.weight.add_(0.1)
    direct = reconstruction_error(layer, candidate, samples)
    delta = layer.weight - candidate.weight
    quadratic = torch.trace(delta @ (samples.T @ samples) @ delta.T).item()
    assert direct["squared_error"] == pytest.approx(quadratic)


def test_sampling_excludes_padding_and_visits_all_batches():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    data = [{"x": torch.full((1, 8, 2), float(i)), "mask": torch.tensor([[1] * 4 + [0] * 4])}
            for i in range(4)]
    for batch in data:
        batch["x"][:, 4:] = -999
    kwargs = dict(max_rows=8, seed=42, input_mask=lambda batch: batch["mask"])
    samples = collect_linear_inputs(model, "0", data, lambda net, batch: net(batch["x"]), **kwargs)
    assert samples.shape == (8, 2)
    assert samples.min() >= 0 and samples.max() == 3
    torch.testing.assert_close(samples, collect_linear_inputs(
        model, "0", data, lambda net, batch: net(batch["x"]), **kwargs))
    assert not model[0]._forward_pre_hooks


def test_hook_removed_when_forward_fails():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    with pytest.raises(RuntimeError):
        collect_linear_inputs(model, "0", [torch.ones(2, 5)], lambda net, batch: net(batch))
    assert not model[0]._forward_pre_hooks
