"""Run budgeted PyTorch experiments with durable per-attempt evidence."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import platform
import socket
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from torch import nn

from .analyzer.candidates import model_fingerprint, tensor_bytes
from .analyzer.context import analysis_context, calibration_content_fingerprint
from .analyzer.schema import AnalysisReport
from .analyzer.service import (
    TaskAdapter,
    analyze_model,
    apply_compression_plan,
    candidate_transformation,
)
from .benchmark import benchmark_bundle
from .checkpoints import load_bundle, save_bundle
from .comparison import _benchmark_evidence, bundle_bytes
from .experiment_plan import (
    STRATEGIES,
    ExperimentRequest,
    current_trial,
    frontier,
    make_plan,
    validate_configuration,
)
from .run_artifacts import read, write_frozen


def benchmark_settings(
    options: Mapping[str, Any], request: ExperimentRequest, task: str
) -> dict[str, Any]:
    if "_pass_fds" in options:
        raise ValueError("Unexpected benchmark option _pass_fds.")
    bound = inspect.signature(benchmark_bundle).bind(
        ".", device=request.device, task=task, **options
    )
    bound.apply_defaults()
    settings = dict(bound.arguments)
    del settings["path"]
    del settings["_pass_fds"]
    for key in ("warmup", "iterations", "threads", "seed", "output_tokens"):
        value = settings[key]
        if type(value) is not int or value < (0 if key in {"warmup", "seed"} else 1):
            raise ValueError(f"Invalid benchmark {key}.")
    timeout = settings["timeout"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Benchmark timeout must be positive and finite.")
    shape = settings["input_shape"]
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != (2 if task == "causal_lm" else 4)
        or any(type(value) is not int or value < 1 for value in shape)
    ):
        raise ValueError(
            "Benchmark input_shape must contain positive dimensions for the task."
        )
    settings["input_shape"] = list(shape)
    tokens = settings["input_ids"]
    if task == "causal_lm":
        if shape[0] != 1 or (
            tokens is not None
            and (
                not isinstance(tokens, list)
                or len(tokens) != shape[1]
                or any(type(token) is not int or token < 0 for token in tokens)
            )
        ):
            raise ValueError(
                "Language timing requires batch one and matching prompt tokens."
            )
        if tokens is None and not settings["workload_id"]:
            settings["workload_id"] = "seeded_synthetic_token_ids"
    elif tokens is not None:
        raise ValueError("Prompt tokens require a causal language model.")
    if settings["seed"] >= 2**64:
        raise ValueError("Benchmark seed must fit in 64 unsigned bits.")
    return settings


def _hardware(device: str) -> dict[str, Any]:
    cpu = next(
        (
            line.partition(":")[2].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor() or platform.machine(),
    )
    details: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cpu": cpu,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "platform": platform.platform(),
    }
    if device.startswith("cuda"):
        gpu = torch.cuda.get_device_properties(device)
        driver = Path("/proc/driver/nvidia/version")
        details.update(
            gpu=gpu.name,
            gpu_memory_bytes=gpu.total_memory,
            gpu_capability=[gpu.major, gpu.minor],
            cuda_version=torch.version.cuda,
            cuda_driver=driver.read_text().strip() if driver.is_file() else None,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            cuda_current_device=torch.cuda.current_device(),
        )
    return details


def _software() -> dict[str, str]:
    packages = (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "tensorly",
        "transformers",
        "datasets",
        "segmentation-models-pytorch",
        "pycocotools",
        "psutil",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    versions["tn_compression_source_sha256"] = digest.hexdigest()
    return versions


def _measure(
    path: Path,
    adapter: TaskAdapter,
    validation: Sequence[Any],
    settings: Mapping[str, Any],
    result: dict[str, Any],
    lock_fd: int,
    baseline: Mapping[str, Any] | None = None,
) -> None:
    model, manifest = load_bundle(path, device=adapter.device)
    evaluation = adapter.evaluate(model, validation)
    result.update(quality=adapter.validate_metrics(evaluation), evaluation=evaluation)
    result["serialized_bytes"] = bundle_bytes(path)
    result["weights_sha256"] = manifest["weights_sha256"]
    if baseline is not None:
        result["quality_loss"] = adapter.quality_loss(baseline, evaluation)
    del model
    if adapter.device.startswith("cuda"):
        torch.cuda.empty_cache()
    # The worker retains the run lock if the parent is killed while timing.
    measured = benchmark_bundle(path, **settings, _pass_fds=(lock_fd,))
    result["benchmark"] = measured
    if (
        not _benchmark_evidence(measured)
        or measured.get("runtime") != "pytorch"
        or any(
            measured.get(key) != value
            for key, value in settings.items()
            if key != "timeout"
        )
    ):
        raise ValueError(
            "Benchmark failed or returned mismatched evidence. See benchmark details."
        )
    memory = (
        "cuda_allocated_peak_bytes"
        if adapter.device.startswith("cuda")
        else "rss_inference_peak_bytes"
    )
    result.update(
        status="measured",
        memory_metric=memory,
        memory_bytes=measured[memory],
        latency_ms=measured["latency_p50_ms"],
    )


def decision(
    result: Mapping[str, Any], loss: float, request: ExperimentRequest
) -> list[str]:
    return [
        name
        for name, value, limit in (
            ("quality", loss, request.max_quality_loss),
            (
                "serialized_bytes",
                result["serialized_bytes"],
                request.max_serialized_bytes,
            ),
            ("memory_bytes", result["memory_bytes"], request.max_memory_bytes),
            ("latency_ms", result["latency_ms"], request.max_latency_ms),
        )
        if value > limit
    ]


def run_experiment(
    model: nn.Module,
    config: Mapping[str, Any],
    request: ExperimentRequest,
    output: Path,
    *,
    calibration: Sequence[Any],
    calibration_ids: Sequence[str],
    validation: Sequence[Any],
    validation_ids: Sequence[str],
    benchmark: Mapping[str, Any],
    detection_coco: Any = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another experiment owns this output directory."
            ) from error
        return _run(
            model,
            config,
            request,
            output,
            list(calibration),
            list(calibration_ids),
            list(validation),
            list(validation_ids),
            benchmark,
            detection_coco,
            plan_only,
            lock.fileno(),
        )


def _save_trial(
    original: Path,
    output: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    request: ExperimentRequest,
    calibration: Sequence[Any],
    calibration_ids: Sequence[str],
) -> None:
    torch.manual_seed(request.seed)
    model, _ = load_bundle(original, device=request.device)
    apply_compression_plan(
        model,
        plan,
        config,
        calibration_batches=calibration,
        calibration_ids=calibration_ids,
        device=request.device,
    )
    save_bundle(model.cpu(), output, metadata={"workflow": config})


def _run(
    model: nn.Module,
    config: Mapping[str, Any],
    request: ExperimentRequest,
    output: Path,
    calibration: list[Any],
    calibration_ids: list[str],
    validation: list[Any],
    validation_ids: list[str],
    benchmark: Mapping[str, Any],
    detection_coco: Any,
    plan_only: bool,
    lock_fd: int,
) -> dict[str, Any]:
    task = config.get("task", "classification")
    adapter = TaskAdapter(
        task, config, device=request.device, detection_coco=detection_coco
    )
    if not calibration or not validation or not calibration_ids or not validation_ids:
        raise ValueError(
            "Experiments require nonempty, identified calibration and validation splits."
        )
    if set(calibration_ids) & set(validation_ids):
        raise ValueError("Calibration and validation example IDs overlap.")
    if getattr(model, "_tn_model_spec", {}).get("source") not in {
        "torchvision",
        "smp",
        "transformers",
    }:
        raise ValueError(
            "Experiments require a model reconstructible by the isolated bundle worker."
        )
    settings = benchmark_settings(benchmark, request, task)
    if (
        task == "causal_lm"
        and settings["input_ids"] is not None
        and any(
            token >= cast(Any, model).config.vocab_size
            for token in settings["input_ids"]
        )
    ):
        raise ValueError("Benchmark prompt token is outside the model vocabulary.")
    torch.set_num_threads(settings["threads"])

    fingerprint = model_fingerprint(
        model, config.get("analysis", {}).get("model_identity", config.get("model", {}))
    )
    context = analysis_context(
        fingerprint,
        config,
        calibration_batches=calibration,
        calibration_ids=calibration_ids,
    )
    provenance = {
        "request": asdict(request),
        "workflow": config,
        "benchmark": settings,
        "hardware": _hardware(request.device),
        "software": _software(),
        "model": model._tn_model_spec,
        "model_fingerprint": fingerprint,
        "analysis_context_fingerprint": context["fingerprint"],
        "validation": {
            "ids": validation_ids,
            "content_sha256": calibration_content_fingerprint(validation),
        },
        "quality": adapter.quality_contract(),
    }
    write_frozen(output / "request.json", provenance)
    evidence_path = output / "evidence.json"
    if evidence_path.exists():
        report = AnalysisReport.from_dict(read(evidence_path))
        if (
            report.analysis_context_fingerprint != context["fingerprint"]
            or report.model_fingerprint != fingerprint
        ):
            raise ValueError("Evidence fingerprint differs from the experiment inputs.")
    else:
        # Targeting the original tensor footprint runs the existing singleton screen
        # without letting cumulative selection spend any strategy's trial budget.
        report = analyze_model(
            model.to(request.device),
            config,
            level="validated",
            calibration_batches=calibration,
            calibration_ids=calibration_ids,
            validation_batches=validation,
            target_size_bytes=tensor_bytes(model),
            max_quality_loss=request.max_quality_loss,
            requested_backend=request.runtime,
            device=request.device,
            detection_coco=detection_coco,
        )
        write_frozen(evidence_path, report.to_dict())
    model.cpu()
    if request.device.startswith("cuda"):
        torch.cuda.empty_cache()

    original = output / "original"
    if not original.exists():
        with TemporaryDirectory(dir=output) as temporary:
            path = Path(temporary) / "bundle"
            save_bundle(model, path, metadata={"workflow": config})
            path.replace(original)
    original_bytes = bundle_bytes(original)
    plan, pool = make_plan(request, report, original_bytes)
    write_frozen(output / "plan.json", asdict(plan))
    if plan_only:
        return {"status": "planned", "plan": str(output / "plan.json")}

    baseline_path = output / "baseline.json"
    if baseline_path.exists():
        baseline = read(baseline_path)
    else:
        baseline = {}
        _measure(original, adapter, validation, settings, baseline, lock_fd)
        write_frozen(baseline_path, baseline)
    history: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGIES}
    results: list[dict[str, Any]] = []
    for strategy in plan.schedule:
        previous = history[strategy]
        index = len(previous)
        if strategy == "current":
            keys = current_trial(plan, pool, previous)
        else:
            keys = (plan.evidence if strategy == "evidence" else plan.random)[index]
        validate_configuration(keys, pool)
        trial_id = f"{strategy}-{index + 1:04d}"
        folder = output / "trials" / trial_id
        started = folder / "attempt.json"
        finished = folder / "result.json"
        attempt = {
            "trial_id": trial_id,
            "strategy": strategy,
            "candidate_ids": list(keys),
        }
        interrupted = started.exists()
        write_frozen(started, attempt)
        if finished.exists():
            result = read(finished)
            if any(result.get(key) != value for key, value in attempt.items()):
                raise ValueError(
                    f"Saved result does not match the trial plan: {trial_id}."
                )
        else:
            result = {**attempt, "status": "failed", "accepted": False}
            if interrupted:
                result["error"] = (
                    "Interrupted attempt counted against the trial budget."
                )
            else:
                try:
                    compression_plan = {
                        **report.compression_plan,
                        "transformations": [
                            candidate_transformation(pool[key]) for key in keys
                        ],
                    }
                    _save_trial(
                        original,
                        folder / "bundle",
                        compression_plan,
                        config,
                        request,
                        calibration,
                        calibration_ids,
                    )
                    _measure(
                        folder / "bundle",
                        adapter,
                        validation,
                        settings,
                        result,
                        lock_fd,
                        baseline["evaluation"],
                    )
                    result["violations"] = decision(
                        result, result["quality_loss"], request
                    )
                    result["accepted"] = not result["violations"]
                except Exception as error:  # noqa: BLE001 - record failed attempts
                    result.update(
                        status="failed", error=f"{type(error).__name__}: {error}"
                    )
            write_frozen(finished, result)
        previous.append(result)
        results.append(result)

    feasible = [row for row in results if row["accepted"]]
    direction = adapter.quality_direction
    summary = {
        "status": "complete",
        "baseline": baseline,
        "trials": results,
        "shared_screen": report.summary,
        "frontier": frontier(results, direction),
        "feasible_frontier": frontier(feasible, direction),
        "strategies": {
            name: {
                "attempted": len(rows),
                "accepted": sum(row["accepted"] for row in rows),
                "failed": sum(row["status"] == "failed" for row in rows),
                "feasible_frontier": frontier(
                    [row for row in rows if row["accepted"]], direction
                ),
            }
            for name, rows in history.items()
        },
    }
    write_frozen(output / "experiment.json", summary)
    return summary
