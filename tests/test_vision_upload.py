import pytest
import torch
from torch import nn

from tn_compression.checkpoints import load_bundle, save_bundle
from tn_compression.models import load_model
from tn_compression.vision_upload import compress_svd_energy, load_vision_checkpoint


def tiny_model():
    return nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.Flatten(), nn.Linear(4 * 4 * 4, 3))


def test_uploaded_state_dict_uses_declarative_architecture_and_strict_load(tmp_path, monkeypatch):
    from tn_compression import vision_upload

    monkeypatch.setattr(vision_upload, "load_model", lambda spec, **kwargs: tiny_model())
    spec = {"source": "torchvision", "name": "resnet18", "task": "classification",
            "kwargs": {"num_classes": 3}, "weights": "IMAGENET1K_V1",
            "state_dict_path": str(tmp_path / "uploaded.pt")}
    reference = tiny_model()
    wrapped = {"state_dict": {"module." + name: value for name, value in reference.state_dict().items()},
               "epoch": 12}
    path = tmp_path / "uploaded.pt"
    torch.save(wrapped, path)
    model = load_vision_checkpoint(path, spec)
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    assert model._tn_model_spec["weights"] is None
    assert "state_dict_path" not in model._tn_model_spec

    del wrapped["state_dict"]["module.0.bias"]
    torch.save(wrapped, path)
    with pytest.raises(ValueError, match="do not match"):
        load_vision_checkpoint(path, spec)


def test_uploaded_model_objects_and_non_tensor_state_are_rejected(tmp_path, monkeypatch):
    from tn_compression import vision_upload

    monkeypatch.setattr(vision_upload, "load_model", lambda spec, **kwargs: tiny_model())
    spec = {"source": "torchvision", "name": "resnet18"}
    path = tmp_path / "object.pt"
    torch.save(nn.Linear(2, 2), path)
    with pytest.raises(ValueError, match="state_dict"):
        load_vision_checkpoint(path, spec)
    torch.save({"state_dict": {"weight": torch.ones(2, 2), "metadata": 1}}, path)
    with pytest.raises(ValueError, match="tensor value"):
        load_vision_checkpoint(path, spec)
    with pytest.raises(ValueError, match="TorchVision or SMP"):
        load_vision_checkpoint(path, {"source": "custom", "name": "foo"})


def test_real_torchvision_upload_round_trips_without_original_file(tmp_path):
    pytest.importorskip("torchvision")
    spec = {"source": "torchvision", "name": "resnet18", "task": "classification",
            "kwargs": {"num_classes": 3}, "weights": None}
    original = load_model(spec, load_weights=False).eval()
    uploaded = tmp_path / "weights.pt"
    torch.save(original.state_dict(), uploaded)
    model = load_vision_checkpoint(uploaded, {**spec, "state_dict_path": str(uploaded)}).eval()
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), original(inputs), rtol=0, atol=0)
    save_bundle(model, tmp_path / "bundle")
    uploaded.unlink()
    restored, _ = load_bundle(tmp_path / "bundle")
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), original(inputs), rtol=0, atol=0)


def test_energy_rank_is_reported_and_saved_factorization_reloads(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.diag(torch.tensor([5.0, 4.0, 0.0, 0.0])))
    inputs = torch.randn(3, 4)
    report = compress_svd_energy(model, 0.60)
    layer = report["layers"][0]
    assert report["compressed_layers"] == 1
    assert layer["rank"] == 1
    assert layer["retained_energy"] == pytest.approx(25 / 41)
    assert layer["tensor_bytes_saved"] == 32
    assert report["tensor_bytes_saved"] == 32
    expected = model(inputs)
    save_bundle(model, tmp_path / "bundle", model_spec={"source": "custom"})
    restored, _ = load_bundle(tmp_path / "bundle", factory=lambda: nn.Sequential(nn.Linear(4, 4, bias=False)))
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


def test_conv_energy_factorization_preserves_low_rank_output_and_skips_expansion(tmp_path):
    model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1, bias=False), nn.ReLU())
    with torch.no_grad():
        basis = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        model[0].weight.copy_(torch.stack([basis, 2 * basis, 3 * basis, 4 * basis]))
    inputs = torch.randn(2, 2, 8, 8)
    expected = model(inputs)
    report = compress_svd_energy(model, 0.999)
    assert report["compressed_layers"] == 1
    assert report["layers"][0]["rank"] == 1
    torch.testing.assert_close(model(inputs), expected, atol=1e-4, rtol=1e-5)
    save_bundle(model, tmp_path / "conv_bundle", model_spec={"source": "custom"})
    restored, _ = load_bundle(
        tmp_path / "conv_bundle",
        factory=lambda: nn.Sequential(nn.Conv2d(2, 4, 3, padding=1, bias=False), nn.ReLU()),
    )
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)

    dense = nn.Sequential(nn.Linear(2, 2))
    skipped = compress_svd_energy(dense, 1.0)
    assert skipped["status"] == "unchanged"
    assert skipped["layers"][0]["reason"] == "non_beneficial_parameter_count"


@pytest.mark.parametrize("energy", [0, -0.1, 1.1, float("nan"), True])
def test_energy_must_be_valid(energy):
    with pytest.raises(ValueError, match="SVD energy"):
        compress_svd_energy(nn.Sequential(nn.Linear(4, 4)), energy)
