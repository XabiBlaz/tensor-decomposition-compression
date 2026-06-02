from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tn_compression.api import compress_model, generate_compression_plan

from compression.analyze_compression_plan import main as analyze_plan_main
from compression.compress_model import build_compression_config, main as compression_main
from compression.models import available_model_names, build_model
from compression.representation_analysis import main as representation_main
from compression.sae_postcompression import main as sae_main


class TinyArtifactModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = torch.nn.Linear(4 * 4 * 4, 2)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def _config(tmp_path: Path, method: str = "tensor_train", rank: int = 4):
    return build_compression_config(
        tmp_path,
        method=method,
        rank=rank,
        structure="TTPWT",
        device=torch.device("cpu"),
    )


def test_public_model_choices_are_real_models_or_artifact():
    assert set(available_model_names()) == {"artifact", "torchvision_resnet18", "torchvision_vgg16"}


def test_real_resnet_forward_pass():
    model = build_model("torchvision_resnet18", weights="none", num_classes=10)
    out = model(torch.randn(2, 3, 64, 64))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 10)


def test_real_vgg16_builds_with_random_weights():
    model = build_model("torchvision_vgg16", weights="none", num_classes=10)
    assert model.classifier[-1].out_features == 10


def test_public_compress_model_preserves_tensor_output_shape(tmp_path):
    model = build_model("torchvision_resnet18", weights="none", num_classes=10)
    x = torch.randn(2, 3, 64, 64)
    before = model(x)
    result = compress_model(model, _config(tmp_path, rank=4), inplace=False, device="cpu", return_model=True)
    assert result.compressed_layers >= 1
    assert result.model is not None
    after = result.model(x)
    assert after.shape == before.shape


def test_compression_demo_writes_summary_and_report(tmp_path):
    summary = compression_main(
        [
            "--model", "torchvision_resnet18",
            "--weights", "none",
            "--num-classes", "10",
            "--input-shape", "1,3,64,64",
            "--method", "tensor_train",
            "--rank", "2",
            "--latency-runs", "1",
            "--warmup-runs", "0",
            "--output-dir", str(tmp_path),
        ]
    )
    assert summary["weights"] == "none"
    assert summary["compressed_layers"] >= 1
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "compression_plan.json").exists()


def test_compression_demo_accepts_yaml_config(tmp_path):
    summary = compression_main(
        [
            "--model", "torchvision_resnet18",
            "--weights", "none",
            "--num-classes", "10",
            "--input-shape", "1,3,64,64",
            "--config", "compression/configs/individual_layers.yaml",
            "--latency-runs", "1",
            "--warmup-runs", "0",
            "--output-dir", str(tmp_path / "config_run"),
        ]
    )
    assert summary["config_path"]
    assert summary["compressed_layers"] >= 1
    assert (tmp_path / "config_run" / "summary.json").exists()
    assert (tmp_path / "config_run" / "compression_plan.json").exists()


def test_dry_run_plan_generated_without_mutating_model(tmp_path):
    model = build_model("torchvision_resnet18", weights="none", num_classes=10)
    before_types = [(name, type(module).__name__) for name, module in model.named_modules()]
    plan = generate_compression_plan(model, _config(tmp_path, rank=2), inplace=False, device="cpu")
    after_types = [(name, type(module).__name__) for name, module in model.named_modules()]
    assert (tmp_path / "compression_plan.json").exists()
    assert plan.plan["layers"]
    assert before_types == after_types

    dry_summary = compression_main(
        [
            "--model", "torchvision_resnet18",
            "--weights", "none",
            "--num-classes", "10",
            "--input-shape", "1,3,64,64",
            "--method", "tensor_train",
            "--rank", "2",
            "--dry-run-plan",
            "--output-dir", str(tmp_path / "dry"),
        ]
    )
    assert dry_summary["dry_run"] is True
    assert dry_summary["compressed_layers"] == 0
    assert (tmp_path / "dry" / "compression_plan.json").exists()


def test_precompression_analysis_generates_inspection_outputs(tmp_path):
    summary = analyze_plan_main(
        [
            "--model", "torchvision_resnet18",
            "--weights", "none",
            "--num-classes", "10",
            "--input-shape", "1,3,64,64",
            "--max-layers", "6",
            "--output-dir", str(tmp_path / "analysis"),
        ]
    )
    assert summary["layers"]
    assert summary["activation_analysis"]["layers"]
    assert summary["activation_analysis"]["cka_matrix"]
    assert summary["activation_analysis"]["cka_matrix"]
    assert summary["cka_segments"]
    assert (tmp_path / "analysis" / "precompression_analysis.json").exists()
    assert (tmp_path / "analysis" / "precompression_report.md").exists()
    assert (tmp_path / "analysis" / "segments.json").exists()


def test_artifact_mode_rejects_state_dict_and_accepts_full_module(tmp_path):
    state_path = tmp_path / "state_dict.pt"
    torch.save(TinyArtifactModule().state_dict(), state_path)
    with pytest.raises(RuntimeError, match="full serialized torch.nn.Module"):
        compression_main(
            [
                "--model", "artifact",
                "--artifact-path", str(state_path),
                "--input-shape", "1,3,32,32",
                "--output-dir", str(tmp_path / "reject"),
            ]
        )

    module_path = tmp_path / "module.pt"
    torch.save(TinyArtifactModule(), module_path)
    summary = compression_main(
        [
            "--model", "artifact",
            "--artifact-path", str(module_path),
            "--input-shape", "1,3,32,32",
            "--method", "tensor_train",
            "--rank", "2",
            "--latency-runs", "1",
            "--warmup-runs", "0",
            "--output-dir", str(tmp_path / "accept"),
        ]
    )
    assert summary["compressed_layers"] >= 1


def test_explicit_cuda_request_fails_clearly_when_unavailable(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")
    with pytest.raises(RuntimeError, match="CUDA device 'cuda' was requested"):
        compression_main(
            [
                "--model", "torchvision_resnet18",
                "--device", "cuda",
                "--output-dir", str(tmp_path),
            ]
        )


def test_cp2_only_marks_linear_layers_eligible(tmp_path):
    model = build_model("torchvision_resnet18", weights="none", num_classes=10)
    plan = generate_compression_plan(model, _config(tmp_path, method="cp2", rank=2), inplace=False, device="cpu")
    conv_entries = [item for item in plan.plan["layers"] if item["type"] == "Conv2d"]
    linear_entries = [item for item in plan.plan["layers"] if item["type"] == "Linear"]
    assert conv_entries
    assert linear_entries
    assert all(item["skip_reason"] == "cp2_only_supports_linear" for item in conv_entries)
    assert any(item["eligible"] and item["policy"] for item in linear_entries)


def test_representation_analysis_outputs_summary(tmp_path):
    summary = representation_main(
        [
            "--model", "torchvision_resnet18",
            "--weights", "none",
            "--num-classes", "10",
            "--input-shape", "2,3,64,64",
            "--method", "tensor_train",
            "--rank", "2",
            "--batch-size", "2",
            "--batches", "1",
            "--max-layers", "3",
            "--output-dir", str(tmp_path),
        ]
    )
    assert summary["layers"]
    assert (tmp_path / "representation_summary.json").exists()


def test_representation_analysis_accepts_compressed_artifact_path(tmp_path):
    original_path = tmp_path / "original.pt"
    compressed_path = tmp_path / "compressed.pt"
    torch.save(TinyArtifactModule(), original_path)
    torch.save(TinyArtifactModule(), compressed_path)
    summary = representation_main(
        [
            "--model", "artifact",
            "--artifact-path", str(original_path),
            "--compressed-artifact-path", str(compressed_path),
            "--input-shape", "2,3,32,32",
            "--batch-size", "2",
            "--batches", "1",
            "--max-layers", "3",
            "--output-dir", str(tmp_path / "repr_artifact"),
        ]
    )
    assert summary["comparison_mode"] == "provided_compressed_artifact"
    assert summary["layers"]


def test_save_model_writes_compressed_artifact(tmp_path):
    summary = compression_main(
        [
            "--model", "torchvision_resnet18",
            "--weights", "none",
            "--num-classes", "10",
            "--input-shape", "1,3,64,64",
            "--method", "tensor_train",
            "--rank", "2",
            "--latency-runs", "1",
            "--warmup-runs", "0",
            "--save-model",
            "--output-dir", str(tmp_path / "save_model"),
        ]
    )
    artifact_paths = [Path(item["path"]) for item in summary["artifacts"] if item["kind"] == "state_dict"]
    assert artifact_paths
    assert artifact_paths[0].exists()


def test_sae_postcompression_outputs_sparse_metrics(tmp_path):
    summary = sae_main(
        [
            "--model", "torchvision_resnet18",
            "--weights", "none",
            "--num-classes", "10",
            "--input-shape", "2,3,64,64",
            "--method", "tensor_train",
            "--rank", "2",
            "--batch-size", "2",
            "--batches", "1",
            "--max-layers", "3",
            "--max-rows", "128",
            "--steps", "5",
            "--sae-batch-size", "32",
            "--output-dir", str(tmp_path / "sae"),
        ]
    )
    assert summary["sae"]["reconstruction_mse"] >= 0
    assert "mean_l0" in summary["sae"]
    assert "dead_latent_fraction" in summary["sae"]
    assert (tmp_path / "sae" / "sae_summary.json").exists()
    assert (tmp_path / "sae" / "sae_report.md").exists()
