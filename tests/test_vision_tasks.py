import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from tn_compression.api import apply_compression_plan, build_plan
from tn_compression.checkpoints import load_bundle, save_bundle
from tn_compression.tasks.data import SyntheticVisionDataset, map_pet_mask, split_indices
from tn_compression.tasks.vision import evaluate_vision, task_loss, train_vision


class TinySegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.decoder = nn.Conv2d(8, 3, 1)

    def forward(self, images):
        return self.decoder(self.encoder(images))


def test_offline_segmentation_lifecycle_and_frozen_buffers(tmp_path):
    torch.manual_seed(0)
    model = TinySegmentation()
    batches = DataLoader(SyntheticVisionDataset(count=4, size=8), batch_size=2)
    train_vision(model, batches, batches, task="segmentation", num_classes=3)
    plan = build_plan(model, {"compression": {"mode": "individual", "layers": {
        "encoder.0": {"type": "partial_tucker", "rank": 2}}}}, write=False).plan
    assert apply_compression_plan(model, plan)[0] == 1
    frozen = {name: value.clone() for name, value in model.encoder.state_dict().items()}
    decoder_before = model.decoder.weight.detach().clone()
    train_vision(model, batches, batches, task="segmentation", num_classes=3, policy="decoder")
    assert not torch.equal(decoder_before, model.decoder.weight)
    for name, tensor in frozen.items():
        torch.testing.assert_close(tensor, model.encoder.state_dict()[name], rtol=0, atol=0)
    save_bundle(model, tmp_path, model_spec={"source": "custom"})
    restored, _ = load_bundle(tmp_path, factory=TinySegmentation)
    assert evaluate_vision(restored, batches, task="segmentation", num_classes=3) == evaluate_vision(
        model, batches, task="segmentation", num_classes=3)


def test_masks_keep_class_ids_and_binary_ignores_border():
    mask = torch.tensor([[1, 2, 3]])
    assert map_pet_mask(mask).tolist() == [[0, 1, 2]]
    assert map_pet_mask(mask, binary=True).tolist() == [[1, 0, 255]]
    for binary, channels in [(False, 3), (True, 1)]:
        targets = map_pet_mask(mask, binary=binary).unsqueeze(0)
        logits = torch.randn(1, channels, 1, 3, requires_grad=True)
        for loss in ("cross_entropy", "dice", "focal_tversky"):
            value = task_loss(logits, targets, task="segmentation", binary=binary, loss=loss)
            assert torch.isfinite(value)
        with pytest.raises(ValueError, match="no valid"):
            task_loss(logits, torch.full_like(targets, 255), task="segmentation", binary=binary)


def test_roles_are_disjoint_reproducible_and_exhaustive():
    identifiers = [str(index) for index in range(100)]
    split = split_indices(identifiers)
    assert split == split_indices(identifiers)
    assert len(set(sum(split.values(), []))) == 100
    with pytest.raises(ValueError, match="unique"):
        split_indices(["same", "same"])


def test_classification_metrics_are_weighted_by_examples():
    model = nn.Identity()
    batches = [(torch.tensor([[5.0, 0.0], [0.0, 5.0]]), torch.tensor([0, 1])),
               (torch.tensor([[5.0, 0.0]]), torch.tensor([1]))]
    metrics = evaluate_vision(model, batches, task="classification", num_classes=2)
    assert metrics["top1"] == 2 / 3
    expected = nn.functional.cross_entropy(torch.cat([item[0] for item in batches]),
                                            torch.cat([item[1] for item in batches])).item()
    assert metrics["loss"] == pytest.approx(expected)
