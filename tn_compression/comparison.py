"""Final artifact comparisons. Uses only the standard library for offline reporting."""

import argparse
import json
import math
from pathlib import Path


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pair(before, after):
    before = before if _number(before) else None
    after = after if _number(after) else None
    change = after - before if before is not None and after is not None else None
    return {"original": before, "compressed": after,
            "change": change if _number(change) else None}


def _benchmark_evidence(benchmark):
    """Require the measurements produced by the benchmark worker, not a status label."""
    positive = ["latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "rss_load_peak_bytes",
                "rss_inference_peak_bytes", "tensor_bytes", "parameters"]
    nonnegative = ["load_seconds"]
    if str(benchmark.get("device", "")).startswith("cuda"):
        nonnegative += ["cuda_allocated_peak_bytes", "cuda_reserved_peak_bytes"]
    if benchmark.get("task") == "causal_lm":
        positive += ["output_tokens_per_second"]
    return (benchmark.get("status") == "ok"
            and all(_number(benchmark.get(key)) and benchmark[key] > 0 for key in positive)
            and all(_number(benchmark.get(key)) and benchmark[key] >= 0 for key in nonnegative))


def bundle_bytes(path):
    """Actual on-disk bundle file lengths, including weights and manifest."""
    path = Path(path)
    if not (path / "manifest.json").is_file() or not (path / "weights.pt").is_file():
        raise ValueError(f"Not a complete bundle: {path}")
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def build_comparison(original_evaluation, compressed_evaluation, original_benchmark,
                     compressed_benchmark, plan, *, original_bundle=None,
                     compressed_bundle=None, synthetic_baseline=False):
    """Compare final reloaded artifacts; a feasible analysis alone is not success.

    Quality metrics retain their native units. Resource changes are compressed
    minus original; negative latency/byte changes mean an improvement. A failed
    benchmark, missing provenance, or unverified final contract prevents success.
    """
    reasons = []
    role = original_evaluation.get("role")
    ids = original_evaluation.get("example_ids")
    same_data = (role is not None and role == compressed_evaluation.get("role")
                 and isinstance(ids, list) and bool(ids)
                 and ids == compressed_evaluation.get("example_ids"))
    if not same_data:
        reasons.append("Evaluation split or example identities are missing or differ.")
    metrics = {key: _pair(value, compressed_evaluation.get(key))
               for key, value in original_evaluation.items()
               if _number(value) and _number(compressed_evaluation.get(key))}
    constraints = plan.get("constraints", {})
    metric = constraints.get("quality_metric")
    direction = constraints.get("quality_direction")
    limit = constraints.get("max_quality_loss")
    before, after = original_evaluation.get(metric), compressed_evaluation.get(metric)
    loss = None
    if _number(before) and _number(after) and direction in {"higher", "lower"}:
        loss = before - after if direction == "higher" else after - before
        loss = loss if _number(loss) else None
    quality_passed = (same_data and loss is not None and _number(limit) and limit >= 0 and loss <= limit)
    if not quality_passed:
        reasons.append("Final artifact quality constraint failed or could not be verified.")
    benchmarks_ok = all(_benchmark_evidence(item) for item in (original_benchmark, compressed_benchmark))
    workload_keys = ("runtime", "device", "task", "input_shape", "iterations", "warmup", "threads", "seed", "output_tokens", "timing_scope")
    if original_benchmark.get("task") == "causal_lm" or compressed_benchmark.get("task") == "causal_lm":
        workload_keys += ("workload_id",)
    same_workload = all(key in original_benchmark and key in compressed_benchmark
                        and original_benchmark[key] == compressed_benchmark[key] for key in workload_keys)
    if not benchmarks_ok or not same_workload:
        reasons.append("Benchmarks failed, are unavailable, or used different workloads.")
    target = constraints.get("target_size_bytes")
    actual_bytes = compressed_benchmark.get("tensor_bytes")
    target_verified = _number(target) and target > 0 and _number(actual_bytes) and actual_bytes > 0
    target_passed = target_verified and actual_bytes <= target
    if not target_passed:
        reasons.append("Final artifact tensor-byte target failed or could not be verified.")
    resources = {key: _pair(original_benchmark.get(key), compressed_benchmark.get(key))
                 for key in ("parameters", "tensor_bytes", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                             "rss_load_peak_bytes", "rss_inference_peak_bytes", "cuda_allocated_peak_bytes",
                             "cuda_reserved_peak_bytes", "output_tokens_per_second")
                 if key in original_benchmark or key in compressed_benchmark}
    if original_bundle is not None and compressed_bundle is not None:
        resources["bundle_file_bytes"] = _pair(bundle_bytes(original_bundle), bundle_bytes(compressed_bundle))
    transformations = plan.get("transformations", [])
    if plan.get("status") != "feasible":
        status = "infeasible"
        reasons.append("The analyzer did not find a feasible plan.")
    elif not transformations:
        status = "no_op"
        reasons.append("The plan applies no compression transformations.")
    elif not quality_passed:
        status = "quality_failed"
    elif target_verified and not target_passed:
        status = "size_target_failed"
    elif not benchmarks_ok or not same_workload or not target_verified:
        status = "measurement_failed"
    else:
        status = "verified"
    if synthetic_baseline:
        reasons.append("Synthetic or untrained baseline: integration evidence only, not real-world quality evidence.")
        if status == "verified":
            status = "synthetic_only"
    return {"schema_version": 1, "status": status, "reasons": reasons,
            "plan_status": plan.get("status"), "transformations": len(transformations),
            "evaluation": {"role": role, "same_examples": same_data, "example_count": len(ids) if isinstance(ids, list) else 0, "metrics": metrics},
            "quality": {"metric": metric, "direction": direction, "units": constraints.get("quality_units"),
                        "loss": loss, "maximum_loss": limit if _number(limit) else None, "passed": quality_passed},
            "resources": resources,
            "size_target": {"maximum_tensor_bytes": target if _number(target) else None,
                            "actual_tensor_bytes": actual_bytes if _number(actual_bytes) else None,
                            "passed": target_passed},
            "benchmark_status": {"original": original_benchmark.get("status"),
                                 "compressed": compressed_benchmark.get("status"), "same_workload": same_workload,
                                 "measurements_valid": benchmarks_ok},
            "size_target_scope": "Analyzer target is tensor bytes; bundle_file_bytes is actual serialized file size.",
            "timing_scope": original_benchmark.get("timing_scope"),
            "synthetic_baseline": synthetic_baseline}


def write_comparison(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Original versus compressed", "", f"Outcome: **{report['status']}**", ""]
    lines.extend(f"- {reason}" for reason in report["reasons"])
    lines += ["", "| Measurement | Original | Compressed | Change |", "|---|---:|---:|---:|"]
    for name, row in {**report["evaluation"]["metrics"], **report["resources"]}.items():
        lines.append(f"| {name} | {row['original']} | {row['compressed']} | {row['change']} |")
    lines += ["", report["size_target_scope"], "", "Quality and resource results apply only to the recorded data and benchmark workload."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="One suite directory produced by the experiment runner.")
    parser.add_argument("--compressed-bundle")
    parser.add_argument("--check-plan", action="store_true", help="Write an infeasible/no-op outcome and exit 3 when no artifact should be produced.")
    parser.add_argument("--synthetic-baseline", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.run_dir)
    def read(relative):
        return json.loads((root / relative).read_text(encoding="utf-8"))
    plan = read("analysis/compression_plan.json")
    if args.check_plan:
        if plan.get("status") == "feasible" and plan.get("transformations"):
            return 0
        report = build_comparison(read("original-evaluation/evaluate.json"), {},
                                  read("original-benchmark/benchmark.json"), {}, plan,
                                  synthetic_baseline=args.synthetic_baseline)
        report["reasons"].append("No compressed artifact was produced.")
        write_comparison(report, root)
        print(f"{report['status']}: no compressed artifact produced; see {root / 'comparison.json'}")
        return 3
    if not args.compressed_bundle:
        parser.error("--compressed-bundle is required unless --check-plan is used")
    report = build_comparison(read("original-evaluation/evaluate.json"), read("evaluation/evaluate.json"),
                              read("original-benchmark/benchmark.json"), read("benchmark/benchmark.json"),
                              plan, original_bundle=root / "original/bundle",
                              compressed_bundle=args.compressed_bundle, synthetic_baseline=args.synthetic_baseline)
    write_comparison(report, root)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
