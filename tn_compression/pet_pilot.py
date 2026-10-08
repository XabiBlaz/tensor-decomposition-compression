"""Small file-based protocol for the bounded Oxford Pet comparison runner."""

import argparse
import copy
import json
import math
from pathlib import Path

from .comparison import build_comparison, bundle_bytes
from .config import load_config


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_frozen(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to resume with changed configuration: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def energies(config):
    values = config.get("pilot", {}).get("simple_energies", [])
    if (not isinstance(values, list) or not values
            or any(isinstance(value, bool) or not isinstance(value, (float, int))
                   or not math.isfinite(value) or not 0 < value < 1 for value in values)):
        raise ValueError("pilot.simple_energies needs a nonempty list of values in (0, 1).")
    names = [f"energy-{round(value * 100):02d}" for value in values]
    if len(set(names)) != len(names):
        raise ValueError("Simple energy values must have distinct rounded percentages.")
    return list(zip(names, values))


def prepare(template, data_root, seed, output):
    config = load_config(template)
    if config.get("task") != "segmentation" or config.get("data", {}).get("kind") != "oxford_pet":
        raise ValueError("This pilot requires Oxford Pet segmentation.")
    if config.get("model", {}).get("weights") is not None:
        raise ValueError("The pilot pins initialization to seeded random weights; set model.weights: null.")
    simple = energies(config)
    config["seed"] = seed
    config["data"]["root"] = str(Path(data_root).resolve())
    config["data"]["download"] = False
    output = Path(output)
    write_frozen(output / "resolved-config.json", config)
    recovery = copy.deepcopy(config)
    recovery["training"] = config["pilot"]["recovery_training"]
    write_frozen(output / "recovery-config.json", recovery)
    for name, energy in simple:
        trial = copy.deepcopy(config)
        trial["recipe"] = {"methods": ["tensor_decomposition"], "svd_energy": energy,
                           "include": config["analysis"]["include"]}
        write_frozen(output / "configs" / f"{name}.json", trial)


def trial_row(root, name, dense, dense_benchmark):
    folder = root / "candidates" / name
    try:
        measured = read(folder / "validation" / "evaluate.json")
        benchmark = read(folder / "benchmark" / "benchmark.json")
        bundle = folder / "compress" / "bundle"
        plan_path = (root / "analysis" / "compression_plan.json" if name == "analyzer"
                     else folder / "compress" / "compression_plan.json")
        comparison = build_comparison(dense, measured, dense_benchmark, benchmark, read(plan_path),
                                      original_bundle=root / "train" / "bundle", compressed_bundle=bundle)
        quality = measured["mean_iou"]
        tensor = benchmark["tensor_bytes"]
        if not all(isinstance(value, (float, int)) and math.isfinite(value) for value in (quality, tensor)):
            raise ValueError("Nonfinite quality or tensor-byte measurement.")
        loss = dense["mean_iou"] - quality
        status = ("valid" if comparison["status"] == "verified" else
                  "constraint_failed" if comparison["status"] in {"quality_failed", "size_target_failed"} else "failed")
        return {"id": name, "status": status,
                "validation_comparison_status": comparison["status"], "reasons": comparison["reasons"],
                "validation_mean_iou": quality, "validation_quality_loss": loss,
                "tensor_bytes": tensor, "serialized_bundle_bytes": bundle_bytes(bundle),
                "latency_mean_ms": benchmark["latency_mean_ms"],
                "rss_inference_peak_bytes": benchmark["rss_inference_peak_bytes"],
                "cuda_allocated_peak_bytes": benchmark.get("cuda_allocated_peak_bytes"),
                "benchmark_status": benchmark.get("status")}
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"id": name, "status": "failed", "error": str(error)}


def select(root):
    root = Path(root)
    config = read(root / "resolved-config.json")
    dense = read(root / "dense-validation" / "evaluate.json")
    benchmark = read(root / "dense-benchmark" / "benchmark.json")
    target = round(config["analysis"]["target_size_mb"] * 1048576)
    limit = config["analysis"]["max_quality_loss"]
    names = ["analyzer", *(name for name, _ in energies(config))]
    trials = [trial_row(root, name, dense, benchmark) for name in names]
    for row in trials:
        if row["status"] != "failed":
            row["slower_than_dense"] = row["latency_mean_ms"] > benchmark["latency_mean_ms"]
            if row["benchmark_status"] != "ok":
                row["status"] = "failed"
                row["error"] = "Benchmark did not produce valid measurements."
    analyzer = trials[0] if trials[0]["status"] == "valid" else None
    simple = [row for row in trials[1:] if row["status"] == "valid"]
    if analyzer:
        simple.sort(key=lambda row: (abs(row["tensor_bytes"] - analyzer["tensor_bytes"]),
                                     -row["validation_mean_iou"], row["id"]))
    else:
        simple.sort(key=lambda row: (-row["validation_mean_iou"], row["id"]))
    comparator = simple[0] if simple else None
    selected = [row["id"] for row in (analyzer, comparator) if row]
    gap = (abs(comparator["tensor_bytes"] - analyzer["tensor_bytes"]) / analyzer["tensor_bytes"]
           if analyzer and comparator else None)
    result = {"schema_version": 1, "selection_role": "validation", "dense_validation": dense,
              "dense_benchmark": benchmark, "target_tensor_bytes": target,
              "maximum_mean_iou_loss": limit, "trials": trials, "selected": selected,
              "storage_gap_fraction": gap,
              "storage_matched": gap is not None and gap <= config["pilot"]["storage_match_tolerance"],
              "storage_match_tolerance": config["pilot"]["storage_match_tolerance"],
              "note": "Different ranks and layer choices are compared by achieved bytes, never rank alone. Test was not read for selection."}
    write_frozen(root / "selection.json", result)
    return result


def report(root):
    root = Path(root)
    selected = read(root / "selection.json")
    dense_test = read(root / "dense-test" / "evaluate.json")
    dense_benchmark = read(root / "dense-benchmark" / "benchmark.json")
    comparisons = {}
    recovery = {}
    control_path = root / "dense-control" / "bundle"
    if control_path.exists():
        control_train = read(root / "dense-control" / "finetune.json")
        control_test = read(root / "dense-control-test" / "evaluate.json")
        control_benchmark = read(root / "dense-control-benchmark" / "benchmark.json")
    for name in selected["selected"]:
        folder = root / "candidates" / name
        plan = (root / "analysis" / "compression_plan.json" if name == "analyzer"
                else folder / "compress" / "compression_plan.json")
        comparisons[name] = build_comparison(
            dense_test, read(folder / "test" / "evaluate.json"),
            dense_benchmark, read(folder / "benchmark" / "benchmark.json"), read(plan),
            original_bundle=root / "train" / "bundle", compressed_bundle=folder / "compress" / "bundle")
        if control_path.exists():
            trained = read(folder / "recovery" / "finetune.json")
            if (trained["train_ids"] != control_train["train_ids"]
                    or trained["validation_ids"] != control_train["validation_ids"]
                    or len(trained["history"]) != len(control_train["history"])):
                raise ValueError("Recovery and dense control used different data or training epochs.")
            recovery[name] = build_comparison(
                control_test, read(folder / "recovery-test" / "evaluate.json"),
                control_benchmark, read(folder / "recovery-benchmark" / "benchmark.json"), read(plan),
                original_bundle=control_path, compressed_bundle=folder / "recovery" / "bundle")
    matched = len(selected["selected"]) == 2 and selected["storage_matched"]
    result = {"schema_version": 1, "selection": selected, "dense_test": dense_test,
              "comparisons": comparisons, "recovery_vs_equal_budget_dense_control": recovery,
              "analyzer_outperforms_simple_on_test": (
                  comparisons["analyzer"]["evaluation"]["metrics"]["mean_iou"]["compressed"] >
                  comparisons[selected["selected"][1]]["evaluation"]["metrics"]["mean_iou"]["compressed"]
                  if matched and all(comparisons[name]["status"] == "verified" for name in selected["selected"])
                  else None),
              "note": "Any test result is descriptive; selection was frozen using validation only."}
    (root / "final.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["energies", "prepare", "has-plan", "select", "selected", "report"])
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root")
    args = parser.parse_args(argv)
    if args.command == "energies":
        print("\n".join(name for name, _ in energies(load_config(args.config))))
    elif args.command == "prepare":
        prepare(args.config, args.data_root, args.seed, args.root)
    elif args.command == "has-plan":
        plan = read(Path(args.root) / "analysis" / "compression_plan.json")
        return 0 if plan.get("status") == "feasible" and plan.get("transformations") else 1
    elif args.command == "select":
        print(json.dumps(select(args.root), indent=2))
    elif args.command == "selected":
        print("\n".join(read(Path(args.root) / "selection.json")["selected"]))
    else:
        print(json.dumps(report(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
