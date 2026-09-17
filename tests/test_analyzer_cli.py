import json

import pytest
import yaml

pytest.importorskip("torchvision")

from tn_compression.cli import main


def test_structural_analyze_cli_writes_public_artifacts(tmp_path):
    config = {
        "seed": 0,
        "task": "classification",
        "num_classes": 3,
        "model": {
            "source": "torchvision", "name": "resnet18", "weights": None,
            "task": "classification", "kwargs": {"num_classes": 3},
        },
        "analysis": {
            "include": ["fc"],
            "candidate_grid": {
                "linear": {"methods": ["svd"], "ranks": [1]},
                "quantization": {"enabled": False},
                "gated_mlp": {"methods": []},
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "analysis"
    main(["analyze", "--config", str(config_path), "--output-dir", str(output),
          "--level", "structural", "--backend", "pytorch"])
    analysis = json.loads((output / "analysis.json").read_text())
    plan = json.loads((output / "compression_plan.json").read_text())
    assert analysis["level"] == "structural"
    assert analysis["candidates"][0]["layer_path"] == "fc"
    assert plan["kind"] == "damage_aware_compression_plan"
    assert (output / "analysis-summary.txt").exists()
    assert (output / "analysis-report.html").exists()
