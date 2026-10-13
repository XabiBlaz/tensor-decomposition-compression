import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace, ModuleType

import pytest

from tn_compression.llm_pilot import (execute, read, run, score_probes, select_trials,
                                      sha256, validate_probes, validate_spec)
from tn_compression.llm_backend import benchmark_definitions, prompt_nll, recipe, tokenize_probes


CONFIG = Path("examples/configs/llm-pilot.json")
PROBES = Path("examples/fixtures/llm-behavior.json")


def measurement(method="dense", *, size=100, nll=2., latency=10):
    return {"status": "ok", "method": method, "role": "validation", "workload_hash": "same",
            "serialized_bytes": size, "nll": nll, "latency_mean_ms": latency,
            "device_used_peak_bytes_sampled": 10000,
            "behavior": {"success": .9, "by_slice": {"instruction": {"success": 1., "count": 4}}}}


def test_spec_and_behavior_roles_reject_unsafe_or_invalid_options():
    config = read(CONFIG)
    validate_spec(config)
    for key, value in (("methods", ["gptq", "shell"]), ("iterations", True),
                       ("max_probe_loss", float("nan")), ("max_length", 20)):
        changed = copy.deepcopy(config)
        changed["pilot"][key] = value
        with pytest.raises(ValueError):
            validate_spec(changed)
    probes = read(PROBES)
    validate_probes(probes)
    probes["test"][0]["prompt"] = probes["validation"][0]["prompt"]
    with pytest.raises(ValueError, match="overlapping"):
        validate_probes(probes)
    with pytest.raises(ValueError, match="Unsupported"):
        recipe("arbitrary")


def test_probe_scoring_rejects_extra_text_wrong_types_and_missing_responses():
    rows = read(PROBES)["validation"][:2]
    result = score_probes(rows, ['{"city":"Bilbao"}', '{"total":true}'])
    assert result["success"] == .5
    assert result["by_slice"]["extraction"]["count"] == 2
    assert not score_probes(rows[:1], ['Here is your JSON: {"city":"Bilbao"}'])["examples"][0]["passed"]
    numeric = {**rows[1], "expected": {"total": 1}}
    assert score_probes([numeric], ['{"total":true}'])["success"] == 0
    assert score_probes([numeric], ['{"total":1.0}'])["success"] == 1
    assert score_probes([numeric], ['{"total":NaN}'])["success"] == 0
    with pytest.raises(ValueError, match="Missing"):
        score_probes(rows, [])


def test_transformers5_chat_template_is_explicitly_token_ids():
    class Tokenizer:
        def apply_chat_template(self, messages, *, return_dict=True, **kwargs):
            return {"input_ids": [1, 2, 3]} if return_dict else [1, 2, 3]
    assert tokenize_probes(Tokenizer(), [{"prompt": "hello"}]) == [[1, 2, 3]]


def test_selection_retains_slower_failed_and_slice_regressions_without_test_data():
    config = read(CONFIG)
    dense = measurement()
    gptq = measurement("gptq", size=50, latency=12, nll=2.01)
    awq = measurement("awq", size=60, latency=8, nll=2.02)
    result = select_trials(dense, [gptq, awq, {"method": "failed", "status": "failed"}], config)
    assert result["pareto_frontier"] == ["gptq", "awq"]
    assert result["trials"][0]["slower_than_dense"]
    assert not result["trials"][2]["feasible"]
    awq["behavior"]["by_slice"]["instruction"]["success"] = .5
    result = select_trials(dense, [awq], config)
    assert "probe_guardrail:instruction" in result["trials"][0]["reasons"]
    gptq["role"] = "test"
    assert not select_trials(dense, [gptq], config)["trials"][0]["feasible"]
    gptq["nll"] = float("nan")
    assert select_trials(dense, [gptq], config)["trials"][0]["status"] == "failed"
    dense["behavior"]["success"] = float("nan")
    with pytest.raises(ValueError):
        select_trials(dense, [], config)


def test_prompt_likelihood_uses_observed_tokens_and_weights_by_token_count():
    def output(values):
        return SimpleNamespace(prompt_logprobs=[None, *[{token: SimpleNamespace(logprob=lp)} for token, lp in values]])
    result = prompt_nll([output([(2, -1), (3, -3)]), output([(5, -2)])], [[1, 2, 3], [4, 5]])
    assert result == {"nll": 2., "tokens": 3, "sequences": 2}
    with pytest.raises(ValueError, match="Invalid"):
        prompt_nll([output([(2, float("nan"))])], [[1, 2]])
    with pytest.raises(ValueError, match="Missing"):
        prompt_nll([], [[1, 2]])


def test_cpu_dry_run_has_no_side_effects_or_gpu_imports(tmp_path):
    output = tmp_path / "not-created"
    completed = subprocess.run([sys.executable, "-m", "tn_compression.llm_pilot", "--output-dir", str(output),
                                "--dry-run"], capture_output=True, text=True, check=True)
    report = json.loads(completed.stdout)
    assert report["status"] == "dry_run"
    assert len(report["commands"]) == 10
    assert not report["final_test_enabled"]
    assert not output.exists()


def test_resume_checks_completed_results_and_artifacts(tmp_path):
    artifact = tmp_path / "weights"
    artifact.write_bytes(b"weights")
    target = tmp_path / "stage.json"
    value = {"status": "ok", "artifact_hashes": {str(artifact): sha256(artifact)}}
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])"
    command = [sys.executable, "-c", code, str(target), json.dumps(value)]
    assert execute(command, target, timeout=5, allow_download=False) == value
    assert execute(["nonexistent"], target, timeout=5, allow_download=False) == value
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed artifact"):
        execute(command, target, timeout=5, allow_download=False)


def test_preflight_only_never_prepares_or_loads_model_assets(tmp_path, monkeypatch):
    calls = []
    def worker(command, target, **kwargs):
        calls.append(command[3])
        target.parent.mkdir(parents=True, exist_ok=True)
        value = {"status": "ok", "gpu": "test device"}
        target.write_text(json.dumps(value))
        return value
    monkeypatch.setattr("tn_compression.llm_pilot.execute", worker)
    monkeypatch.setattr("tn_compression.llm_pilot.subprocess.check_output", lambda *a, **kw: "a" * 40)
    args = argparse.Namespace(config=CONFIG, output_dir=tmp_path / "study", compression_python=sys.executable,
                              runtime_python=sys.executable, dry_run=False, allow_download=False,
                              finalize=False, select=None, preflight_only=True)
    assert run(args)["status"] == "preflight_complete"
    assert calls == ["preflight", "preflight"]
    assert not (args.output_dir / "inputs.json").exists()
    assert not (args.output_dir / "validation-report.json").exists()
    args.finalize, args.select = True, "gptq"
    with pytest.raises(ValueError, match="cannot select"):
        run(args)


@pytest.mark.skipif(os.name == "nt", reason="Native vLLM workers run on Linux")
def test_timeout_stops_worker_children_and_leaves_no_completion_marker(tmp_path):
    ready, unwanted = tmp_path / "ready", tmp_path / "unwanted"
    child = "import time,sys; from pathlib import Path; Path(sys.argv[1]).touch(); time.sleep(.8); Path(sys.argv[2]).touch()"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]]); time.sleep(10)"
    result = execute([sys.executable, "-c", parent, child, str(ready), str(unwanted)],
                     tmp_path / "stage.json", timeout=.3, allow_download=False)
    assert ready.exists(), "The child must have started for this test to prove cleanup."
    assert result["status"] == "failed"
    time.sleep(.9)
    assert not unwanted.exists()
    assert not (tmp_path / "stage.complete.json").exists()


def test_controller_lifecycle_freezes_selection_and_excludes_unselected_test(tmp_path, monkeypatch):
    calls = []
    fail_awq = True
    def worker(command, target, **kwargs):
        if kwargs.get("resume", True) and target.with_suffix(".complete.json").exists():
            return read(target)
        method = command[command.index("--method") + 1]
        role = command[command.index("--role") + 1]
        action = command[3]
        calls.append((action, method, role))
        result = measurement(method, size=100 if method == "dense" else 50)
        result["role"] = role
        if fail_awq and method == "awq":
            result = {"status": "failed", "error": "native loader failed"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result))
        if result["status"] == "ok" and action != "preflight":
            target.with_suffix(".complete.json").write_text(json.dumps({"result_sha256": sha256(target)}))
        return result
    monkeypatch.setattr("tn_compression.llm_pilot.execute", worker)
    monkeypatch.setattr("tn_compression.llm_pilot.subprocess.check_output", lambda *a, **kw: "a" * 40)
    args = argparse.Namespace(config=CONFIG, output_dir=tmp_path / "study", compression_python=sys.executable,
                              runtime_python=sys.executable, dry_run=False, allow_download=False, finalize=False, select=None)
    report = run(args)
    assert report["pareto_frontier"] == ["rtn", "gptq"]
    assert not any(role == "test" for _, _, role in calls)
    calls.clear()
    args.finalize, args.select = True, "gptq"
    report = run(args)
    assert [(method, role) for action, method, role in calls if role == "test"] == [("dense", "test"), ("gptq", "test")]
    assert report["selection"]["method"] == "gptq"
    fail_awq = False
    calls.clear()
    run(args)
    assert not any(method == "awq" for _, method, _ in calls), "Frozen failures must never be retried."
    assert read(args.output_dir / "stages/awq-validation.json")["status"] == "failed"
    args.select = "awq"
    with pytest.raises(ValueError, match="Cannot change"):
        run(args)


def test_public_benchmarks_are_final_only_and_dataset_revisions_are_required(tmp_path):
    config = read(CONFIG)
    config["benchmarks"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="pinned dataset"):
        validate_spec(config)
    args = argparse.Namespace(config=CONFIG, output_dir=tmp_path, compression_python=sys.executable,
                              runtime_python=sys.executable, dry_run=True, allow_download=False,
                              finalize=False, select=None, benchmarks=True, benchmark_limit="64")
    with pytest.raises(ValueError, match="require --finalize"):
        run(args)
    args.finalize, args.select = True, "awq"
    commands = run(args)["commands"]
    public = [command for command in commands if command[3] == "benchmarks"]
    assert len(public) == 2
    assert all(command[command.index("--role") + 1] == "test" for command in public)
    assert {command[command.index("--method") + 1] for command in public} == {"dense", "awq"}


def test_official_task_adapter_preserves_task_and_pins_dataset(monkeypatch, tmp_path):
    utils = ModuleType("lm_eval.tasks._yaml_loader")
    loaded = []
    def load(path):
        loaded.append(path)
        return {"task": "arc_easy" if "arc" in str(path) else "hellaswag", "output_type": "multiple_choice"}
    utils.load_yaml = load
    monkeypatch.setitem(sys.modules, "lm_eval.tasks._yaml_loader", utils)
    specifications = read(CONFIG)["benchmarks"]
    definitions = benchmark_definitions(tmp_path, specifications)
    assert len(loaded) == 2
    assert [value["dataset_kwargs"]["revision"] for value in definitions] == [item["revision"] for item in specifications]
    assert all(value["output_type"] == "multiple_choice" for value in definitions)
