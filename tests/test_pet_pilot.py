import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tn_compression.pet_pilot import prepare, report, select


TEMPLATE = Path("examples/configs/pet-pilot.yaml")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_preparation_freezes_seed_paths_and_simple_trials(tmp_path):
    prepare(TEMPLATE, tmp_path / "pets", 7, tmp_path / "run")
    resolved = json.loads((tmp_path / "run/resolved-config.json").read_text())
    assert resolved["seed"] == 7
    assert resolved["data"]["seed"] == 42
    assert resolved["model"]["weights"] is None
    assert resolved["data"]["root"] == str((tmp_path / "pets").resolve())
    assert len(list((tmp_path / "run/configs").glob("energy-*.json"))) == 3
    prepare(TEMPLATE, tmp_path / "pets", 7, tmp_path / "run")
    with pytest.raises(ValueError, match="changed configuration"):
        prepare(TEMPLATE, tmp_path / "pets", 17, tmp_path / "run")


def test_validation_selection_records_failure_latency_and_storage_match(tmp_path):
    root = tmp_path / "study"
    prepare(TEMPLATE, tmp_path / "pets", 7, root)
    ids = ["pet-1", "pet-2"]
    put(root / "dense-validation/evaluate.json",
        {"role": "validation", "example_ids": ids, "mean_iou": 0.80})
    benchmark = {"status": "ok", "runtime": "pytorch", "device": "cpu", "task": "segmentation",
                 "input_shape": [1, 3, 160, 160], "iterations": 8, "warmup": 2, "threads": 2,
                 "seed": 0, "output_tokens": 32, "timing_scope": "complete_model_forward",
                 "load_seconds": 1, "latency_mean_ms": 10, "latency_p50_ms": 10,
                 "latency_p95_ms": 10, "rss_load_peak_bytes": 1000,
                 "rss_inference_peak_bytes": 1000, "tensor_bytes": 100, "parameters": 25}
    put(root / "dense-benchmark/benchmark.json", benchmark)
    put(root / "train/bundle/manifest.json", {})
    (root / "train/bundle/weights.pt").write_bytes(b"dense")
    plan = {"status": "feasible", "transformations": [{"method": "test"}],
            "constraints": {"quality_metric": "mean_iou", "quality_direction": "higher",
                            "max_quality_loss": .05, "target_size_bytes": 43 * 1048576}}
    put(root / "analysis/compression_plan.json", plan)
    for name, size, quality, latency in (("analyzer", 80, .79, 12),
                                         ("energy-55", 65, .76, 8),
                                         ("energy-70", 82, .78, 9)):
        folder = root / "candidates" / name
        put(folder / "validation/evaluate.json",
            {"role": "validation", "example_ids": ids, "mean_iou": quality})
        put(folder / "benchmark/benchmark.json",
            dict(benchmark, tensor_bytes=size, latency_mean_ms=latency,
                 latency_p50_ms=latency, latency_p95_ms=latency))
        put(folder / "compress/compression_plan.json", plan)
        put(folder / "compress/bundle/manifest.json", {})
        (folder / "compress/bundle/weights.pt").write_bytes(b"weights")
    # No held-out test artifact exists when selection runs.
    chosen = select(root)
    assert chosen["selected"] == ["analyzer", "energy-70"]
    assert chosen["storage_matched"] is True
    assert chosen["trials"][0]["slower_than_dense"] is True
    assert chosen["trials"][-1]["status"] == "failed"
    assert not (root / "dense-test").exists()

    put(root / "dense-test/evaluate.json",
        {"role": "test", "example_ids": ["heldout-1"], "mean_iou": .77})
    for name, quality in (("analyzer", .76), ("energy-70", .75)):
        put(root / "candidates" / name / "test/evaluate.json",
            {"role": "test", "example_ids": ["heldout-1"], "mean_iou": quality})
    final = report(root)
    assert set(final["comparisons"]) == {"analyzer", "energy-70"}
    assert final["analyzer_outperforms_simple_on_test"] is True

    training = {"train_ids": ["train-1"], "validation_ids": ids,
                "history": [{"epoch": 1}, {"epoch": 2}]}
    put(root / "dense-control/finetune.json", training)
    put(root / "dense-control/bundle/manifest.json", {})
    (root / "dense-control/bundle/weights.pt").write_bytes(b"control")
    put(root / "dense-control-test/evaluate.json",
        {"role": "test", "example_ids": ["heldout-1"], "mean_iou": .81})
    put(root / "dense-control-benchmark/benchmark.json", benchmark)
    for name, size, quality in (("analyzer", 80, .79), ("energy-70", 82, .78)):
        folder = root / "candidates" / name
        put(folder / "recovery/finetune.json", training)
        put(folder / "recovery/bundle/manifest.json", {})
        (folder / "recovery/bundle/weights.pt").write_bytes(b"recovered")
        put(folder / "recovery-test/evaluate.json",
            {"role": "test", "example_ids": ["heldout-1"], "mean_iou": quality})
        put(folder / "recovery-benchmark/benchmark.json", dict(benchmark, tensor_bytes=size))
    recovered = report(root)
    assert set(recovered["recovery_vs_equal_budget_dense_control"]) == {"analyzer", "energy-70"}


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("git") is None,
                    reason="Bash and Git are required")
def test_shell_dry_run_is_nonwriting_and_expands_three_seeds(tmp_path):
    output = tmp_path / "runs"
    arguments = ["--data-root", str(tmp_path / "missing-pets"), "--output-dir", str(output),
                 "--run-id", "dry", "--device", "cpu", "--seeds", "7,17,27",
                 "--with-recovery", "--dry-run"]
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    command = ([str(git_bash), "-lc", 'bash scripts/run_pet_pilot.sh "$@"', "tn-tests", *arguments]
               if os.name == "nt" and git_bash.is_file()
               else ["bash", "scripts/run_pet_pilot.sh", *arguments])
    env = dict(os.environ, PYTHON=".venv/Scripts/python.exe" if os.name == "nt" else sys.executable)
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("tn_compression train") == 3
    assert "dense-control" in result.stdout and "--role test" in result.stdout
    assert "pet_pilot select" in result.stdout
    assert not output.exists()


def test_split_inventory_command_needs_no_model(tmp_path):
    config = tmp_path / "config.json"
    put(config, {"task": "segmentation", "num_classes": 3,
                 "model": {"source": "no-such-loader"},
                 "data": {"kind": "synthetic", "count": 2, "size": 8}})
    output = tmp_path / "output"
    subprocess.run([sys.executable, "-m", "tn_compression", "split-ids", "--config", str(config),
                    "--output-dir", str(output), "--device", "cpu"], check=True, capture_output=True)
    roles = json.loads((output / "split-ids.json").read_text())["roles"]
    assert {role: len(ids) for role, ids in roles.items()} == dict.fromkeys(roles, 2)
