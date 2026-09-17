import subprocess


def test_experiment_script_dry_run_prints_pipeline_without_writes(tmp_path):
    output = tmp_path / "runs"
    result = subprocess.run(
        ["bash", "scripts/run_analyzer_experiments.sh", "--dry-run", "--suite", "smoke",
         "--output-dir", str(output), "--run-id", "test-run"],
        text=True, capture_output=True, check=True)
    assert "analyze" in result.stdout
    assert "--level calibrated" in result.stdout
    assert "--level validated" in result.stdout
    assert "compression_plan.json" in result.stdout
    assert "evaluate" in result.stdout and "benchmark" in result.stdout
    assert not output.exists()


def test_experiment_script_help_lists_offline_inputs():
    result = subprocess.run(
        ["bash", "scripts/run_analyzer_experiments.sh", "--help"],
        text=True, capture_output=True, check=True)
    assert "--model-cache" in result.stdout
    assert "--data-cache" in result.stdout
    assert "never enables downloads" in result.stdout
