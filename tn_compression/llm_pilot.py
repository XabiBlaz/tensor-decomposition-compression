"""Bounded native quantization study; GPU operations run in separate processes."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from .run_artifacts import read, write_frozen


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("rtn", "gptq", "awq")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")


def validate_spec(config):
    model = config["model"]
    if model.get("source") != "transformers" or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
        raise ValueError("Use a Transformers model with a full pinned revision.")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    data = config["data"]
    if data["kind"] == "huggingface" and not re.fullmatch(r"[0-9a-f]{40}", data.get("revision", "")):
        raise ValueError("Pin the dataset revision.")
    if data["kind"] not in {"huggingface", "text_json"}:
        raise ValueError("Use pinned Hugging Face text or text_json data.")
    if isinstance(data["max_length"], bool) or not isinstance(data["max_length"], int) or not 2 <= data["max_length"] <= config["pilot"]["max_length"]:
        raise ValueError("Data max_length must fit the inference context.")
    methods = config["pilot"]["methods"]
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError("Use distinct supported pilot methods: rtn, gptq, awq.")
    for key in ("max_nll_increase", "max_probe_loss"):
        positive(config["pilot"][key], key, zero=True)
    if config["pilot"]["max_probe_loss"] > 1:
        raise ValueError("max_probe_loss is an absolute fraction in [0, 1].")
    for key in ("timeout_seconds", "max_length", "output_tokens", "iterations", "kv_cache_memory_bytes"):
        value = config["pilot"][key]
        positive(value, key)
        if not isinstance(value, int):
            raise ValueError(f"{key} must be an integer.")
    positive(config["pilot"]["warmup"], "warmup", zero=True)
    if not isinstance(config["pilot"]["warmup"], int):
        raise ValueError("warmup must be an integer.")
    positive(config["pilot"]["gpu_memory_utilization"], "gpu_memory_utilization")
    if config["pilot"]["gpu_memory_utilization"] >= 1:
        raise ValueError("Reserve GPU memory: gpu_memory_utilization must be below 1.")
    lengths = config["pilot"]["prompt_lengths"]
    if not lengths or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in lengths):
        raise ValueError("prompt_lengths must contain positive integer lengths >= 2.")
    if len(set(lengths)) != len(lengths):
        raise ValueError("Benchmark prompt lengths must be distinct.")
    if max(lengths) + config["pilot"]["output_tokens"] > config["pilot"]["max_length"]:
        raise ValueError("Benchmark prompt plus output exceeds max_length.")
    benchmarks = config.get("benchmarks", [])
    if len({item["task"] for item in benchmarks}) != len(benchmarks):
        raise ValueError("Duplicate public benchmark tasks.")
    for item in benchmarks:
        if item["task"] not in {"arc_easy", "hellaswag"} or not re.fullmatch(r"[0-9a-f]{40}", item.get("revision", "")):
            raise ValueError("Use supported public benchmarks with pinned dataset revisions.")


def validate_probes(records):
    seen_ids, seen_prompts = set(), set()
    if set(records) != {"validation", "test"}:
        raise ValueError("Behavior probes require separate validation and test roles.")
    slices = {}
    for role, rows in records.items():
        if not rows:
            raise ValueError(f"Empty {role} behavior probes.")
        slices[role] = set()
        for row in rows:
            if any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ("id", "prompt", "slice")):
                raise ValueError("Each probe needs a nonempty id, prompt and slice.")
            normalized = " ".join(row["prompt"].split())
            if row["id"] in seen_ids or normalized in seen_prompts:
                raise ValueError("Duplicate or overlapping behavior probes.")
            seen_ids.add(row["id"])
            seen_prompts.add(normalized)
            if row.get("check") not in {"json_equals", "exact"}:
                raise ValueError("Probe check must be json_equals or exact.")
            if "expected" not in row:
                raise ValueError("Probe needs an expected result.")
            slices[role].add(row["slice"])
    if slices["validation"] != slices["test"]:
        raise ValueError("Validation and test must cover the same behavior slices.")


def json_equal(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(actual) and math.isfinite(expected) and actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(json_equal(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(json_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def score_probes(rows, responses):
    if len(rows) != len(responses):
        raise ValueError("Missing probe responses.")
    scored, groups = [], {}
    for row, response in zip(rows, responses):
        try:
            actual = json.loads(response) if row["check"] == "json_equals" else response.strip()
            passed = json_equal(actual, row["expected"])
        except (ValueError, TypeError):
            passed = False
        groups.setdefault(row["slice"], []).append(int(passed))
        scored.append({"id": row["id"], "slice": row["slice"], "response": response, "passed": passed})
    if not scored:
        raise ValueError("No behavior probes.")
    return {"success": sum(row["passed"] for row in scored) / len(scored),
            "by_slice": {name: {"success": sum(values) / len(values), "count": len(values)}
                         for name, values in groups.items()}, "examples": scored,
            "scope": "authored deterministic diagnostic probes, not a public benchmark"}


def quality_reasons(dense, candidate, pilot):
    reasons = []
    if candidate["nll"] - dense["nll"] > pilot["max_nll_increase"]:
        reasons.append("nll_guardrail")
    if dense["behavior"]["success"] - candidate["behavior"]["success"] > pilot["max_probe_loss"]:
        reasons.append("probe_guardrail:overall")
    before, after = dense["behavior"]["by_slice"], candidate["behavior"]["by_slice"]
    if set(before) != set(after):
        reasons.append("missing_behavior_slice")
    else:
        for name, baseline in before.items():
            if baseline["count"] != after[name]["count"]:
                reasons.append(f"mismatched_probe_count:{name}")
            if baseline["success"] - after[name]["success"] > pilot["max_probe_loss"]:
                reasons.append(f"probe_guardrail:{name}")
    return reasons


def select_trials(dense, trials, config):
    """Report feasibility and a Pareto frontier; never read test results here."""
    if dense.get("status") != "ok" or dense.get("role") != "validation":
        raise ValueError("A successful validation baseline is required.")
    rows = []
    metrics = ("serialized_bytes", "latency_mean_ms", "nll", "device_used_peak_bytes_sampled")
    def check_measurements(row):
        for key in metrics:
            positive(row[key], key)
        positive(row["behavior"]["success"], "behavior.success", zero=True)
        if row["behavior"]["success"] > 1 or not row.get("workload_hash"):
            raise ValueError("Invalid behavior score or missing workload identity.")
        if not row["behavior"]["by_slice"]:
            raise ValueError("No behavior slices were measured.")
        for group in row["behavior"]["by_slice"].values():
            positive(group["success"], "slice.success", zero=True)
            if group["success"] > 1 or isinstance(group["count"], bool) or not isinstance(group["count"], int) or group["count"] < 1:
                raise ValueError("Invalid behavior slice evidence.")
    check_measurements(dense)
    for trial in trials:
        row = dict(trial)
        reasons = []
        if row.get("status") != "ok":
            row["feasible"] = False
            rows.append(row)
            continue
        if row.get("role") != "validation" or row.get("workload_hash") != dense.get("workload_hash"):
            reasons.append("mismatched_validation_workload")
        try:
            check_measurements(row)
        except (KeyError, TypeError, ValueError) as error:
            row.update(status="failed", feasible=False, error=str(error))
            rows.append(row)
            continue
        row["nll_increase"] = row["nll"] - dense["nll"]
        row["probe_loss"] = dense["behavior"]["success"] - row["behavior"]["success"]
        reasons.extend(quality_reasons(dense, row, config["pilot"]))
        if row["serialized_bytes"] >= dense["serialized_bytes"]:
            reasons.append("no_storage_saving")
        row.update(feasible=not reasons, reasons=reasons,
                   slower_than_dense=row["latency_mean_ms"] > dense["latency_mean_ms"])
        rows.append(row)
    valid = [row for row in rows if row["feasible"]]
    frontier = [row["method"] for row in valid if not any(
        all(other[key] <= row[key] for key in metrics)
        and any(other[key] < row[key] for key in metrics) for other in valid)]
    return {"status": "validation_complete" if valid else "infeasible",
            "selection_role": "validation", "dense": dense, "trials": rows,
            "pareto_frontier": frontier,
            "note": "Frontier uses bytes, latency, NLL and sampled device memory after per-slice quality guardrails; use an exclusive GPU and choose a trade-off explicitly."}


def execute(command, result_path, *, timeout, allow_download, resume=True):
    """Only commands constructed by the controller reach this subprocess boundary."""
    result_path = Path(result_path)
    marker = result_path.with_suffix(".complete.json")
    if resume and marker.exists():
        recorded = read(marker)
        if sha256(result_path) != recorded["result_sha256"]:
            raise ValueError(f"Changed completed stage: {result_path}")
        result = read(result_path)
        for path, digest in result.get("artifact_hashes", {}).items():
            if sha256(path) != digest:
                raise ValueError(f"Changed artifact: {path}")
        return result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "HF_HUB_OFFLINE": "0" if allow_download else "1",
                   "HF_DATASETS_OFFLINE": "0" if allow_download else "1", "TOKENIZERS_PARALLELISM": "false",
                   "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    try:
        with result_path.with_suffix(".log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
            try:
                returncode = process.wait(timeout=timeout)
            finally:
                # vLLM spawns children: stop the whole worker group on timeout/interruption.
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                process.wait()
        if returncode:
            raise RuntimeError(f"Worker exited {returncode}; see {result_path.with_suffix('.log')}")
        result = read(result_path)
        if result.get("status") != "ok":
            raise ValueError("Worker did not report success.")
    except (subprocess.TimeoutExpired, OSError, ValueError, RuntimeError) as error:
        result = {"status": "failed", "error": str(error)}
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if resume:
        write_frozen(marker, {"command": command, "result_sha256": sha256(result_path)})
    return result


def run(args):
    config = read(args.config)
    validate_spec(config)
    probe_path = (ROOT / config["behavior_path"]).resolve()
    probes = read(probe_path)
    validate_probes(probes)
    output = Path(args.output_dir).resolve()
    benchmark_limit = getattr(args, "benchmark_limit", "64")
    if benchmark_limit != "full" and (not benchmark_limit.isdigit() or int(benchmark_limit) < 1):
        raise ValueError("benchmark-limit must be a positive sample count or full.")
    public_benchmarks = getattr(args, "benchmarks", False)
    preflight_only = getattr(args, "preflight_only", False)
    if preflight_only and (args.finalize or public_benchmarks or args.select):
        raise ValueError("--preflight-only cannot select or evaluate a final candidate.")
    if public_benchmarks and (not args.finalize or not config.get("benchmarks")):
        raise ValueError("Public benchmarks require --finalize and pinned task definitions.")
    python = {"compression": args.compression_python, "runtime": args.runtime_python}
    phases = [("preflight-compression", "compression", "preflight", "dense", "validation"),
              ("preflight-runtime", "runtime", "preflight", "dense", "validation"),
              ("prepare", "compression", "prepare", "dense", "validation"),
              ("dense-validation", "runtime", "evaluate", "dense", "validation")]
    for method in config["pilot"]["methods"]:
        phases.extend([(f"{method}-pack", "compression", "compress", method, "validation"),
                       (f"{method}-validation", "runtime", "evaluate", method, "validation")])
    if args.finalize:
        if args.select not in config["pilot"]["methods"]:
            raise ValueError("Finalization requires --select with an explicit pilot method.")
        phases.extend([(f"{method}-test", "runtime", "evaluate", method, "test")
                       for method in ("dense", args.select)])
        if public_benchmarks:
            phases.extend([(f"{method}-benchmarks-{benchmark_limit}", "runtime", "benchmarks", method, "test")
                           for method in ("dense", args.select)])
    if preflight_only:
        phases = phases[:2]
    def command(stage, env, action, method, role):
        argv = [python[env], "-m", "tn_compression.llm_backend", action, "--root", str(output),
                "--method", method, "--role", role, "--environment", env,
                "--result", str(output / "stages" / f"{stage}.json")]
        if action == "benchmarks":
            argv.extend(["--benchmark-limit", benchmark_limit])
        return argv
    if args.dry_run:
        return {"status": "dry_run", "downloads_enabled": args.allow_download,
                "commands": [command(*phase) for phase in phases],
                "final_test_enabled": args.finalize}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = [Path(__file__), ROOT / "tn_compression/llm_backend.py",
               ROOT / "tn_compression/run_artifacts.py", ROOT / "tn_compression/tasks/language.py",
               ROOT / "tn_compression/models.py", ROOT / "requirements/llm-compression.txt",
               ROOT / "requirements/llm-runtime.txt"]
    write_frozen(output / "study.json", {"config": config, "probes": probes, "git_revision": head,
                 "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
                 "python": python})
    def freeze_selection(report):
        if not any(row["method"] == args.select and row["feasible"] for row in report["trials"]):
            raise ValueError("Selected method did not meet validation guardrails.")
        value = {"method": args.select, "validation_report_sha256": sha256(output / "validation-report.json"),
                 "stage_hashes": {stage: sha256(output / "stages" / f"{stage}.json")
                                  for stage, _, action, _, role in phases if role == "validation" and action != "preflight"}}
        write_frozen(output / "selection.json", value)
        return value
    frozen = read(output / "selection.json") if (output / "selection.json").exists() else None
    if frozen:
        if args.finalize and args.select != frozen["method"]:
            raise ValueError("Cannot change a frozen selection after final evaluation.")
        if sha256(output / "validation-report.json") != frozen["validation_report_sha256"]:
            raise ValueError("Changed frozen validation report.")
    elif args.finalize and (output / "validation-report.json").exists():
        frozen = freeze_selection(read(output / "validation-report.json"))
    for phase in phases:
        stage, env, action, method, role = phase
        if role == "test":
            selection = read(output / "selection.json")
            if selection["method"] != args.select:
                raise ValueError("Cannot change a frozen selection after final evaluation.")
        path = output / "stages" / f"{stage}.json"
        if frozen and stage in frozen["stage_hashes"]:
            if sha256(path) != frozen["stage_hashes"][stage]:
                raise ValueError(f"Changed frozen stage: {stage}")
            result = read(path)
            if result["status"] == "ok":
                if not path.with_suffix(".complete.json").exists():
                    raise ValueError(f"Missing completion marker for frozen stage: {stage}")
                result = execute(command(*phase), path, timeout=config["pilot"]["timeout_seconds"],
                                 allow_download=args.allow_download)
        else:
            result = execute(command(*phase), path, timeout=config["pilot"]["timeout_seconds"],
                             allow_download=args.allow_download, resume=action != "preflight")
        if stage.startswith("preflight"):
            if result["status"] != "ok":
                raise RuntimeError(result["error"])
            write_frozen(output / f"{stage}-environment.json", result)
        if stage in {"prepare", "dense-validation"} and result["status"] != "ok":
            raise RuntimeError(result["error"])
        if action == "evaluate" and role == "validation" and method != "dense":
            result["method"] = method
        if stage == f"{config['pilot']['methods'][-1]}-validation":
            if frozen:
                report = read(output / "validation-report.json")
            else:
                dense = read(output / "stages/dense-validation.json")
                trials = [dict(read(output / "stages" / f"{m}-validation.json"), method=m)
                          for m in config["pilot"]["methods"]]
                report = select_trials(dense, trials, config)
                (output / "validation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                if args.finalize:
                    frozen = freeze_selection(report)
    if preflight_only:
        return {"status": "preflight_complete", "output_dir": str(output),
                "note": "Both GPU environments checked; no model or dataset was loaded."}
    if args.finalize:
        report = {"selection": read(output / "selection.json"),
                  "dense": read(output / "stages/dense-test.json"),
                  "compressed": read(output / "stages" / f"{args.select}-test.json"),
                  "note": "Held-out results are reported without changing the selected recipe."}
        if report["dense"]["status"] != "ok" or report["compressed"]["status"] != "ok":
            report.update(status="measurement_failed", quality_guardrails_passed=False)
        else:
            reasons = quality_reasons(report["dense"], report["compressed"], config["pilot"])
            report.update(status="test_quality_failed" if reasons else "test_complete",
                          quality_guardrails_passed=not reasons, quality_reasons=reasons)
        if public_benchmarks:
            report["public_benchmarks"] = {method: read(output / "stages" / f"{method}-benchmarks-{benchmark_limit}.json")
                                           for method in ("dense", args.select)}
            if any(value["status"] != "ok" for value in report["public_benchmarks"].values()):
                report["status"] = "measurement_failed"
        (output / "test-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "examples/configs/llm-pilot.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compression-python", default=sys.executable)
    parser.add_argument("--runtime-python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Check GPU environments without loading models or data.")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--select", choices=METHODS)
    parser.add_argument("--benchmarks", action="store_true", help="Run pinned public tasks only after selecting a recipe.")
    parser.add_argument("--benchmark-limit", default="64", help="Samples per public task; use full for the complete final suite.")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] in {"infeasible", "measurement_failed", "test_quality_failed"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
