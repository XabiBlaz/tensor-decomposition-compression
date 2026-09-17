import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import tn_compression.api as tnc
from tn_compression.config import normalize_config


class TinyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(self.flatten(x))


def _config(tmp_path, *, artifacts=None, mode="default_all"):
    return {
        "schema_version": 1,
        "artifact": {"kind": "live_module"},
        "compression": {
            "mode": mode,
            "default_method": {"type": "tensor_train", "method": "SVD", "structure": "TTPWT", "rank": 2},
        },
        "output": {
            "dir": str(tmp_path),
            "plan_dir": str(tmp_path / "plans"),
            "name": "tiny",
            "artifacts": artifacts or [{"kind": "state_dict"}],
        },
    }


def test_strict_config_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("compression:\n  mode: default_all\ncompression:\n  mode: individual\n")
    with pytest.raises(ValueError, match="Config error: duplicate YAML key"):
        normalize_config(path)


def test_live_module_is_not_mutated_by_default(tmp_path):
    model = TinyConv()
    original_conv = model.conv
    result = tnc.compress_model(model, _config(tmp_path), device="cpu")

    assert model.conv is original_conv
    assert isinstance(model.conv, torch.nn.Conv2d)
    assert result.compressed_layers >= 1
    assert Path(result.manifest_path).exists()
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["input_artifact"]["kind"] == "live_module"


def test_inplace_true_mutates_live_module(tmp_path):
    model = TinyConv()
    result = tnc.compress_model(model, _config(tmp_path), inplace=True, device="cpu")

    assert result.compressed_layers >= 1
    assert not isinstance(model.conv, torch.nn.Conv2d)


def test_state_dict_artifact_is_rejected(tmp_path):
    model = TinyLinear()
    artifact = tmp_path / "weights.pth"
    torch.save(model.state_dict(), artifact)
    cfg = _config(tmp_path)

    with pytest.raises(RuntimeError, match="full torch.nn.Module"):
        tnc.compress_model(artifact, cfg)


def test_return_model_is_not_serialized_for_file_result(tmp_path):
    artifact = tmp_path / "model.pt"
    torch.save(TinyLinear(), artifact)

    result = tnc.compress_model(
        artifact,
        _config(tmp_path),
        device="cpu",
        return_model=True,
    )

    assert result.model is not None
    assert "model" not in result.to_dict()
    assert result.to_dict()["compressed_model_path"] == str(tmp_path / "tiny.pth")


def test_fisher_group_plan_uses_existing_segments(tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    segments = [
        {
            "start_index": 0,
            "end_index": 0,
            "start_layer": "conv",
            "end_layer": "conv",
            "num_layers": 1,
            "layers": ["conv"],
            "mean_similarity": 1.0,
            "mean_gradnorm": 0.1,
        }
    ]
    (analysis_dir / "segments.json").write_text(json.dumps(segments))
    cfg = _config(tmp_path, mode="fisher_groups")
    cfg["compression"]["analysis_dir"] = str(analysis_dir)

    result = tnc.generate_compression_plan(TinyConv(), cfg, device="cpu")

    assert Path(result.plan_path).exists()
    assert result.plan["mode"] == "fisher_groups"
    assert result.plan["segments"][0]["layers"] == ["conv"]


def test_dlf_compact_config_retains_layer_policies(tmp_path):
    from tn_compression.config import build_dlf_compression_config
    source = {"compression": {"rank": 4, "layers": {"head": "skip"}}}
    config = build_dlf_compression_config(source, output_dir=tmp_path)
    assert config["compression"]["layers"] == {"head": "skip"}
    assert config["compression"]["default_method"]["rank"] == 4
    assert config["output"]["dir"] == str(tmp_path / "tn_compression")
    assert normalize_config(normalize_config(config))["_config_hash"] == normalize_config(config)["_config_hash"]


def test_shared_parameters_are_protected_and_no_op_is_reported(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    model[1].weight = model[0].weight
    config = _config(tmp_path)
    config["compression"]["default_method"] = {"type": "svd", "rank": 1}
    result = tnc.compress_model(model, config, return_model=True)
    assert result.summary["status"] == "no_op"
    assert result.model[0].weight is result.model[1].weight
    plan = json.loads(Path(result.plan_path).read_text())
    assert {row["skip_reason"] for row in plan["layers"]} == {"shared_parameter_requires_adapter"}
