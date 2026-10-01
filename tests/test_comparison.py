import copy
import json
import tempfile
import unittest
from pathlib import Path

from tn_compression.comparison import build_comparison, main, write_comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.original = {"role": "test", "example_ids": ["a", "b"], "top1": 0.9}
        self.compressed = {**self.original, "top1": 0.89}
        self.benchmark = {"status": "ok", "device": "cpu", "input_shape": [1, 3, 32, 32],
                          "task": "classification", "runtime": "pytorch", "iterations": 3,
                          "warmup": 1, "threads": 1, "seed": 0, "output_tokens": 32,
                          "timing_scope": "complete_model_forward", "load_seconds": 0.1,
                          "latency_mean_ms": 10, "latency_p50_ms": 9, "latency_p95_ms": 12,
                          "rss_load_peak_bytes": 1000, "rss_inference_peak_bytes": 1000,
                          "tensor_bytes": 100, "parameters": 25}
        self.plan = {"status": "feasible", "transformations": [{"method": "svd"}],
                     "constraints": {"quality_metric": "top1", "quality_direction": "higher",
                                     "quality_units": "absolute_fraction", "max_quality_loss": 0.02,
                                     "target_size_bytes": 100}}

    def report(self, **kwargs):
        return build_comparison(self.original, self.compressed, self.benchmark,
                                self.benchmark, self.plan, **kwargs)

    def test_final_quality_not_plan_status_determines_success(self):
        self.assertEqual(self.report()["status"], "verified")
        self.compressed["top1"] = 0.8
        report = self.report()
        self.assertEqual(report["status"], "quality_failed")
        self.assertFalse(report["quality"]["passed"])

    def test_rejects_missing_nonfinite_or_different_split_evidence(self):
        for value in (None, float("nan"), float("inf")):
            self.compressed["top1"] = value
            self.assertEqual(self.report()["status"], "quality_failed")
            json.dumps(self.report(), allow_nan=False)
        self.compressed["top1"] = 0.9
        self.compressed["example_ids"] = ["c", "d"]
        self.assertEqual(self.report()["status"], "quality_failed")

    def test_infeasible_and_no_op_never_report_success(self):
        self.plan["status"] = "infeasible"
        self.assertEqual(self.report()["status"], "infeasible")
        self.plan["status"] = "feasible"
        self.plan["transformations"] = []
        self.assertEqual(self.report()["status"], "no_op")

    def test_failed_or_mismatched_benchmark_is_not_verified(self):
        other = {**self.benchmark, "status": "timeout"}
        report = build_comparison(self.original, self.compressed, self.benchmark, other, self.plan)
        self.assertEqual(report["status"], "measurement_failed")
        other = {**self.benchmark, "device": "cuda"}
        report = build_comparison(self.original, self.compressed, self.benchmark, other, self.plan)
        self.assertEqual(report["status"], "measurement_failed")

    def test_actual_bundle_sizes_and_synthetic_label_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, count in (("original", 10), ("compressed", 5)):
                path = root / name
                path.mkdir()
                (path / "manifest.json").write_text("{}")
                (path / "weights.pt").write_bytes(b"x" * count)
            report = self.report(original_bundle=root / "original", compressed_bundle=root / "compressed",
                                 synthetic_baseline=True)
            self.assertEqual(report["status"], "synthetic_only")
            self.assertEqual(report["resources"]["bundle_file_bytes"]["change"], -5)
            write_comparison(report, root)
            self.assertEqual(json.loads((root / "comparison.json").read_text())["status"], "synthetic_only")

    def test_lower_is_better_contract(self):
        self.original["nll"] = 1.0
        self.compressed["nll"] = 1.03
        self.plan["constraints"].update(quality_metric="nll", quality_direction="lower", max_quality_loss=0.02)
        self.assertEqual(self.report()["status"], "quality_failed")

    def test_actual_final_tensor_size_must_meet_target(self):
        self.benchmark["tensor_bytes"] = 101
        self.assertEqual(self.report()["status"], "size_target_failed")
        del self.benchmark["tensor_bytes"]
        self.assertEqual(self.report()["status"], "measurement_failed")

    def test_status_ok_does_not_replace_finite_measurements_or_workload(self):
        for key in ("latency_mean_ms", "latency_p50_ms", "rss_inference_peak_bytes", "tensor_bytes"):
            with self.subTest(key=key):
                original = self.benchmark[key]
                self.benchmark[key] = float("nan")
                report = self.report()
                self.assertEqual(report["status"], "measurement_failed")
                json.dumps(report, allow_nan=False)
                self.benchmark[key] = original
        del self.benchmark["threads"]
        self.assertEqual(self.report()["status"], "measurement_failed")

    def test_language_and_cuda_require_corresponding_measurements(self):
        self.benchmark["task"] = "causal_lm"
        self.benchmark["workload_id"] = "fixed-prompt"
        self.assertEqual(self.report()["status"], "measurement_failed")
        self.benchmark["output_tokens_per_second"] = 10.0
        self.assertEqual(self.report()["status"], "verified")
        self.benchmark["device"] = "cuda:0"
        self.assertEqual(self.report()["status"], "measurement_failed")
        self.benchmark.update(cuda_allocated_peak_bytes=100, cuda_reserved_peak_bytes=200)
        self.assertEqual(self.report()["status"], "verified")

    def test_infeasible_search_writes_report_without_compressed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = copy.deepcopy(self.plan)
            plan["status"] = "infeasible"
            for filename, value in (("analysis/compression_plan.json", plan),
                                    ("original-evaluation/evaluate.json", self.original),
                                    ("original-benchmark/benchmark.json", self.benchmark)):
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            self.assertEqual(main(["--run-dir", str(root), "--check-plan"]), 3)
            self.assertEqual(json.loads((root / "comparison.json").read_text())["status"], "infeasible")
