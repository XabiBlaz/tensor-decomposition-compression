import pytest
import torch
from torch import nn

from tn_compression.api import apply_compression_plan, build_plan
from tn_compression.checkpoints import load_bundle, save_bundle
from tn_compression.models import load_model
from tn_compression.tasks.detection import detection_collate, validate_predictions


def test_detector_contract_after_compression_and_reload(tmp_path):
    pytest.importorskip("torchvision")
    model = load_model({"source": "torchvision", "name": "fasterrcnn_resnet50_fpn", "task": "detection",
                        "weights": None, "kwargs": {"weights_backbone": None, "min_size": 64, "max_size": 96,
                        "rpn_pre_nms_top_n_test": 20, "rpn_post_nms_top_n_test": 10, "box_detections_per_img": 5}}).eval()
    images, targets = detection_collate([(torch.rand(3, 64, 72), {"boxes": torch.empty(0, 4)}),
                                         (torch.rand(3, 72, 80), {"boxes": torch.empty(0, 4)})])
    plan = build_plan(model, {"compression": {"mode": "individual", "layers": {
        "backbone.body.layer1.0.conv2": {"type": "partial_tucker", "rank": 8}}}}, write=False).plan
    assert apply_compression_plan(model, plan)[0] == 1
    with torch.no_grad():
        predictions = model(images)
    validate_predictions(predictions, 2)
    save_bundle(model, tmp_path)
    restored, _ = load_bundle(tmp_path)
    with torch.no_grad():
        reloaded = restored(images)
    for expected, actual in zip(predictions, reloaded):
        for name in ("boxes", "labels", "scores"):
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


def test_smp_model_has_compressible_layers_and_valid_output(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    model = load_model({"source": "smp", "name": "Unet", "weights": None,
                        "kwargs": {"encoder_name": "resnet18", "encoder_depth": 3,
                                   "decoder_channels": [32, 16, 8], "classes": 3}}).eval()
    plan = build_plan(model, {"compression": {"mode": "individual", "layers": {
        "encoder.layer1.0.conv1": {"type": "partial_tucker", "rank": 8}}}}, write=False).plan
    assert apply_compression_plan(model, plan)[0] == 1
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = model(inputs)
    assert expected.shape == (1, 3, 32, 32)
    save_bundle(model, tmp_path)
    restored, _ = load_bundle(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


def test_onnx_parity_of_factorized_layer(tmp_path):
    pytest.importorskip("onnxruntime")
    from tn_compression.export import export_onnx
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()).eval()
    plan = build_plan(model, {"compression": {"default_method": {"type": "tensor_train", "rank": 2}}}, write=False).plan
    assert apply_compression_plan(model, plan)[0] == 1
    result = export_onnx(model, torch.randn(1, 3, 8, 8), tmp_path / "model.onnx")
    assert result["status"] == "verified"
