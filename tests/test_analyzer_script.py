import subprocess
import os
from pathlib import Path


def command(*args):
    if os.name == "nt":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.is_file():
            return [str(git_bash), "-lc", 'bash scripts/run_analyzer_experiments.sh "$@"', "tn-tests", *args]
    return ["bash", "scripts/run_analyzer_experiments.sh", *args]


def test_experiment_script_dry_run_prints_pipeline_without_writes(tmp_path):
    output = tmp_path / "runs"
    result = subprocess.run(
        command("--dry-run", "--suite", "smoke", "--output-dir", str(output),
                "--run-id", "test-run"),
        text=True, capture_output=True, check=True)
    assert "analyze" in result.stdout
    assert "--level calibrated" in result.stdout
    assert "--level validated" in result.stdout
    assert "compression_plan.json" in result.stdout
    assert "evaluate" in result.stdout and "benchmark" in result.stdout
    assert "snapshot" in result.stdout
    assert "original-evaluation" in result.stdout and "original-benchmark" in result.stdout
    assert "tn_compression.comparison" in result.stdout
    assert not output.exists()


def test_experiment_script_help_lists_offline_inputs():
    result = subprocess.run(
        command("--help"),
        text=True, capture_output=True, check=True)
    assert "--model-cache" in result.stdout
    assert "--data-cache" in result.stdout
    assert "--allow-code-change" in result.stdout
    assert "never enables downloads" in result.stdout


def test_resume_rejects_a_different_git_revision_without_override(tmp_path):
    environment = tmp_path / "runs" / "resume" / "environment"
    environment.mkdir(parents=True)
    (environment / "git-commit.txt").write_text("0" * 40 + "\n")
    result = subprocess.run(
        command("--dry-run", "--suite", "smoke", "--output-dir", str(tmp_path / "runs"),
                "--run-id", "resume"), text=True, capture_output=True)
    assert result.returncode != 0
    assert "Refusing to resume" in result.stderr

    allowed = subprocess.run(
        command("--dry-run", "--suite", "smoke", "--output-dir", str(tmp_path / "runs"),
                "--run-id", "resume", "--allow-code-change"),
        text=True, capture_output=True, check=True)
    assert "record mixed-revision resume" in allowed.stdout


def test_resume_rejects_stale_stage_marker_even_in_dry_run(tmp_path):
    root = tmp_path / "runs" / "resume"
    environment = root / "environment"
    stages = root / "smoke" / ".stages"
    environment.mkdir(parents=True)
    stages.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (environment / "git-commit.txt").write_text(commit + "\n")
    (stages / "calibrated-screening.done").write_text(
        f"signature=stale\ngit_commit={commit}\n")
    result = subprocess.run(
        command("--dry-run", "--suite", "smoke", "--output-dir", str(tmp_path / "runs"),
                "--run-id", "resume"), text=True, capture_output=True)
    assert result.returncode != 0
    assert "completion marker does not match" in result.stderr


def test_vision_requires_trained_checkpoint_unless_explicitly_synthetic():
    result = subprocess.run(command("--dry-run", "--suite", "vision"), text=True, capture_output=True)
    assert result.returncode != 0
    assert "requires --vision-checkpoint" in result.stderr
    allowed = subprocess.run(command("--dry-run", "--suite", "vision", "--allow-synthetic-baseline"),
                             text=True, capture_output=True, check=True)
    assert "--synthetic-baseline" in allowed.stdout
