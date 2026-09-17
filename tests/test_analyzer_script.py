import subprocess


def command(*args):
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
