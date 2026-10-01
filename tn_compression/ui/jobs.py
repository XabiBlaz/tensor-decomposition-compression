"""Persistent, single-worker jobs over the existing CLI; no shell execution."""

import copy
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contained_path(root, value, *, must_exist=True):
    root = Path(root).resolve()
    candidate = Path(value)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Asset paths must stay inside the configured data directory.")
    if must_exist and not candidate.exists():
        raise ValueError(f"Asset does not exist: {candidate}")
    return candidate


def _validate_local_or_remote_name(spec, data_root):
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
        raise ValueError("Model and tokenizer specifications need a name.")
    name = spec["name"]
    if not name or "\\" in name or any(part in {".", ".."} for part in name.split("/")):
        raise ValueError("Invalid model or tokenizer name.")
    candidate = Path(name)
    local = candidate.is_absolute() or (Path(data_root) / candidate).exists()
    if local:
        spec["name"] = str(contained_path(data_root, name))
    elif spec.get("source") == "transformers" or "revision" in spec:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
            raise ValueError("Use a cached Hugging Face repository ID or a path inside the data directory.")
        if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
            raise ValueError("Pin the model and tokenizer to a full 40-character revision.")
    elif not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("Invalid registry model name.")


def _validate_model_spec(spec, task, data_root):
    if not isinstance(spec, dict) or spec.get("source") not in {"torchvision", "smp", "transformers"}:
        raise ValueError("Only TorchVision, SMP and Transformers model bundles are supported by the UI.")
    if (task == "causal_lm") != (spec["source"] == "transformers"):
        raise ValueError("The checkpoint model source does not match the task.")
    kwargs = spec.get("kwargs", {})
    if not isinstance(kwargs, dict) or kwargs.get("trust_remote_code") is True:
        raise ValueError("Remote model code is not supported by the UI.")
    _validate_local_or_remote_name(spec, data_root)
    if spec["source"] == "transformers":
        config = spec.get("config", {})
        if not isinstance(config, dict) or config.get("auto_map"):
            raise ValueError("Custom Transformers code is not supported by the UI.")


def _validate_bundle(bundle, model, task, data_root):
    bundle = contained_path(data_root, bundle)
    if not bundle.is_dir():
        raise ValueError("Checkpoint must be a bundle directory.")
    manifest_path = contained_path(data_root, bundle / "manifest.json")
    weights_path = contained_path(data_root, bundle / "weights.pt")
    if not manifest_path.is_file() or not weights_path.is_file():
        raise ValueError("Checkpoint needs manifest.json and weights.pt.")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema version.")
    saved = manifest.get("model")
    _validate_model_spec(saved, task, data_root)
    if saved["source"] != model["source"] or saved["name"] != model["name"]:
        raise ValueError("Checkpoint model source/name does not match the configured model.")
    if task != "causal_lm" and saved.get("kwargs", {}) != model.get("kwargs", {}):
        raise ValueError("Checkpoint model architecture differs from the configured model.")
    if task == "causal_lm" and saved.get("revision") != model.get("revision"):
        raise ValueError("Checkpoint model revision differs from the configured model.")
    return bundle


def validate_request(payload, data_root, upload_root=None):
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Provide a workflow configuration object.")
    config = copy.deepcopy(payload["config"])
    task = config.get("task")
    if task not in {"classification", "segmentation", "causal_lm"}:
        raise ValueError("The UI supports classification, segmentation and causal language models.")
    model, data, analysis = (config.get(key) for key in ("model", "data", "analysis"))
    if not all(isinstance(value, dict) for value in (model, data, analysis)):
        raise ValueError("Model, data and analysis must be configuration objects.")
    _validate_model_spec(model, task, data_root)
    if model.get("state_dict_path"):
        if task == "causal_lm" or model["source"] not in {"torchvision", "smp"}:
            raise ValueError("Uploaded .pt weights require a TorchVision or SMP vision architecture.")
        source = model["state_dict_path"]
        if not isinstance(source, str):
            raise ValueError("Uploaded weights path must be a string.")
        resolved = None
        for root in (upload_root, data_root):
            if root is None:
                continue
            try:
                resolved = contained_path(root, source)
                break
            except ValueError:
                pass
        if resolved is None or not resolved.is_file() or resolved.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("Select a .pt or .pth weights file in the upload or data directory.")
        model["state_dict_path"] = str(resolved)
        model["weights"] = None
    mode = payload.get("mode", "analyze")
    if mode not in {"analyze", "compress"}:
        raise ValueError("Choose analyze or compress mode.")
    device = payload.get("device", "cpu")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
        raise ValueError("Choose cpu or a CUDA device such as cuda:0.")
    recipe = config.get("recipe")
    if recipe is not None and mode == "compress":
        from ..direct_recipe import validate_recipe
        validate_recipe(recipe, task)
    study = config.get("language_study")
    if study is not None:
        if task != "causal_lm" or not isinstance(study, dict):
            raise ValueError("A language study requires a causal_lm configuration.")
    if study is not None and mode == "compress":
        trials = study.get("trials")
        if not isinstance(trials, list) or not 2 <= len(trials) <= 6:
            raise ValueError("A language study needs between two and six bounded trials.")
        identifiers = [trial.get("id") for trial in trials if isinstance(trial, dict)]
        if (len(identifiers) != len(trials) or len(set(identifiers)) != len(identifiers)
                or any(not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", value)
                       for value in identifiers)):
            raise ValueError("Language study trial IDs must be unique lowercase file-safe names.")
        from ..direct_recipe import validate_recipe
        for trial in trials:
            methods = validate_recipe(trial.get("recipe"), task)
            if len(methods) != 1 or methods[0] not in {"quantization", "tensor_decomposition"}:
                raise ValueError("Language study trials must test int8 or SVD individually.")
    for key in (() if mode == "analyze" else ("target_size_mb", "max_quality_loss")):
        value = analysis.get(key)
        if key == "target_size_mb" and value is None and (recipe is not None or study is not None):
            continue
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"analysis.{key} must be a finite number.")
        if value < 0 or (key == "target_size_mb" and value == 0):
            raise ValueError(f"Invalid analysis.{key}.")
    if analysis.get("backend", "pytorch") != "pytorch":
        raise ValueError("The UI currently verifies the PyTorch backend. Use the CLI for other backends.")
    analysis["level"] = "validated"
    kind = data.get("kind")
    allowed_data = {"huggingface", "text_json"} if task == "causal_lm" else {"synthetic", "oxford_pet", "image_folder"}
    if kind not in allowed_data:
        raise ValueError(f"Unsupported dataset for {task}: {kind}")
    # Asset reads are limited to the mounted directory, including local HF paths.
    if kind in {"oxford_pet", "image_folder"}:
        if not isinstance(data.get("root"), str):
            raise ValueError("Choose a dataset directory inside the data mount.")
        data["root"] = str(contained_path(data_root, data["root"]))
        data["download"] = False
        if kind == "image_folder":
            data.pop("download", None)
            if task != "classification":
                raise ValueError("Labeled image folders currently support classification only.")
    if kind == "text_json":
        if not isinstance(data.get("path"), str):
            raise ValueError("Choose a text dataset JSON file inside the data directory.")
        data["path"] = str(contained_path(data_root, data["path"]))
        if not Path(data["path"]).is_file():
            raise ValueError("The text dataset path must be a JSON file.")
    if kind == "huggingface":
        data_name = data.get("name")
        if not isinstance(data_name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", data_name):
            raise ValueError("Use a pinned Hugging Face dataset repository ID.")
        if not re.fullmatch(r"[0-9a-f]{40}", data.get("revision", "")):
            raise ValueError("Pin the dataset to a full 40-character revision.")
    if task == "causal_lm" and "tokenizer" in config:
        tokenizer = config["tokenizer"]
        if not isinstance(tokenizer, dict) or tokenizer.get("trust_remote_code"):
            raise ValueError("Remote tokenizer code is not supported by the UI.")
        _validate_local_or_remote_name(tokenizer, data_root)
    checkpoint = payload.get("checkpoint") or None
    if payload.get("preset") == "trained-synthetic" and not checkpoint and not model.get("state_dict_path"):
        raise ValueError("The trained synthetic preset requires its trained checkpoint bundle. Run the demo training step first.")
    if checkpoint and model.get("state_dict_path"):
        raise ValueError("Choose either an uploaded state_dict or an existing checkpoint bundle, not both.")
    if checkpoint:
        if not isinstance(checkpoint, str):
            raise ValueError("Checkpoint must be a bundle directory path.")
        checkpoint = str(_validate_bundle(checkpoint, model, task, data_root))
    if task != "causal_lm" and kind != "synthetic" and not checkpoint and not model.get("state_dict_path"):
        raise ValueError("Real vision workflows require a trained checkpoint bundle or uploaded state_dict.")
    allow_download = payload.get("allow_download", False)
    if not isinstance(allow_download, bool):
        raise ValueError("allow_download must be true or false.")
    recovery = payload.get("recovery", "none")
    if recovery not in {"none", "finetune", "lora"}:
        raise ValueError("Choose no recovery, vision fine-tuning or language LoRA.")
    if recovery == "finetune" and task == "causal_lm":
        raise ValueError("Choose LoRA recovery for a language model.")
    if recovery == "lora" and task != "causal_lm":
        raise ValueError("LoRA recovery is available for language models only.")
    if recovery != "none" and mode != "compress":
        raise ValueError("Recovery runs only after compression.")
    if recovery == "lora":
        lora = config.setdefault("lora", {})
        if not isinstance(lora, dict):
            raise ValueError("LoRA options must be a configuration object.")
        defaults = {"token_budget": 1024, "max_updates": 32, "rank": 8,
                    "alpha": 16, "learning_rate": 0.0001}
        for key, value in defaults.items():
            lora.setdefault(key, value)
        if any(not isinstance(lora[key], int) or isinstance(lora[key], bool) or lora[key] < 1
               for key in ("token_budget", "max_updates", "rank")):
            raise ValueError("LoRA token budget, update count and rank must be positive integers.")
        splits = data.get("splits", {})
        if kind == "huggingface" and (not isinstance(splits, dict) or not splits.get("train")):
            raise ValueError("LoRA needs a separate train split in the pinned text dataset.")
        if kind == "text_json" and "train" not in read_json(data["path"]):
            raise ValueError("LoRA needs a separate train split in the text JSON dataset.")
        config["recovery"] = {"method": "lora"}
    elif recovery == "finetune":
        training = config.setdefault("training", {})
        if not isinstance(training, dict):
            raise ValueError("Vision fine-tuning options must be a configuration object.")
        training.setdefault("epochs", 1)
        training.setdefault("learning_rate", 0.0001)
    return {"config": config, "mode": mode, "device": device, "checkpoint": checkpoint,
            "preset": str(payload.get("preset", "custom"))[:80],
            "allow_download": allow_download, "recovery": recovery}


class JobManager:
    def __init__(self, root, data_root, *, upload_root=None, timeout=21600, command_runner=None):
        self.root = Path(root).resolve()
        self.data_root = Path(data_root).resolve()
        self.upload_root = Path(upload_root).resolve() if upload_root is not None else self.data_root / "uploads"
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.command_runner = command_runner or subprocess.run
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compression")
        # A terminated process must never leave a job claiming to run forever.
        for path in self.root.glob("*/job.json"):
            job = read_json(path)
            if job.get("status") in {"queued", "running"}:
                job.update(status="interrupted", error="The server stopped during this run. Start a new run.")
                self._save(job)

    def _directory(self, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("Invalid run identifier.")
        return contained_path(self.root, identifier)

    def _save(self, job):
        with self.lock:
            path = self.root / job["id"] / "job.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(job, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(path)

    def list(self):
        with self.lock:
            jobs = []
            for path in self.root.glob("*/job.json"):
                job = read_json(path)
                if not job.get("model_name"):
                    try:
                        job["model_name"] = read_json(path.parent / "config.json")["model"]["name"]
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                jobs.append(job)
        return sorted(jobs, key=lambda item: item["created_at"], reverse=True)

    def get(self, identifier):
        directory = self._directory(identifier)
        with self.lock:
            job = read_json(directory / "job.json")
        job["config"] = read_json(directory / "config.json")
        log_path = directory / "run.log"
        if log_path.exists():
            with log_path.open("rb") as handle:
                handle.seek(max(0, log_path.stat().st_size - 64000))
                job["logs"] = handle.read().decode("utf-8", errors="replace")
        else:
            job["logs"] = ""
        job["results"] = {}
        for key, path in {"analysis": "analysis/analyze.json", "relevance": "relevance/relevance.json",
                          "inspection": "inspection/layers.json", "study": "study/study.json",
                          "direct": "compressed/compress.json", "comparison": "comparison/comparison.json",
                          "generation": "generation/generate.json",
                          "recovered": "recovered-evaluation/evaluate.json",
                          "lora": "lora/lora_recovery.json",
                          "recovered_comparison": "recovered-comparison/comparison.json",
                          "recovery_quality": "recovery-quality.json"}.items():
            source = directory / path
            if source.exists():
                try:
                    job["results"][key] = read_json(source)
                except (ValueError, OSError):
                    pass  # A stage may still be writing its result.
        job["artifacts"] = [
            {"name": str(path.relative_to(directory)).replace("\\", "/"),
             "url": f"/api/jobs/{identifier}/artifacts/{path.relative_to(directory).as_posix()}"}
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.resolve().is_relative_to(directory)
            and path.suffix in {".json", ".html", ".md", ".txt", ".pt", ".safetensors", ".onnx", ".tar"}
            and not (path.suffix == ".onnx" and "language-onnx" in path.parts)
            and path.name != "job.json"
        ]
        return job

    def submit(self, payload):
        request = validate_request(payload, self.data_root, self.upload_root)
        identifier = uuid.uuid4().hex
        directory = self.root / identifier
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(request.pop("config"), indent=2), encoding="utf-8")
        job = {"id": identifier, "status": "queued", "stage": "Waiting", "created_at": utc_now(), **request}
        self._save(job)
        self.pool.submit(self._execute, job)
        return job

    def _stage(self, job, label, command, output_name, *, checkpoint=None, extra=(), config_path=None,
               timeout=None):
        directory = self._directory(job["id"])
        job.update(status="running", stage=label)
        self._save(job)
        argv = [sys.executable, "-m", "tn_compression", command,
                "--config", str(config_path or directory / "config.json"), "--output-dir", str(directory / output_name),
                "--device", job["device"]]
        if checkpoint:
            argv += ["--checkpoint", str(checkpoint)]
        argv += list(extra)
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "0" if job.get("allow_download") else "1"
        env["HF_DATASETS_OFFLINE"] = "0" if job.get("allow_download") else "1"
        env["TRANSFORMERS_OFFLINE"] = "0" if job.get("allow_download") else "1"
        env["TN_ALLOW_HF_DOWNLOAD"] = "1" if job.get("allow_download") else "0"
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("OMP_NUM_THREADS", "2")
        with (directory / "run.log").open("ab", buffering=0) as log:
            log.write(f"\n--- {label} ---\n".encode())
            result = self.command_runner(argv, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, timeout=timeout or self.timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"{label} failed (exit {result.returncode}). See the run log for details.")

    def _language_study(self, job, config, original):
        """Run validation-only trials, then touch held-out test once for the winner."""
        directory = self._directory(job["id"])
        self._stage(job, "Inspecting supported language layers", "layers", "inspection", checkpoint=original)
        self._stage(job, "Recording the disjoint calibration split", "evaluate", "original-calibration",
                    checkpoint=original, extra=("--role", "calibration"))
        self._stage(job, "Evaluating uncompressed validation baseline", "evaluate", "original-validation",
                    checkpoint=original, extra=("--role", "validation"))
        baseline = read_json(directory / "original-validation" / "evaluate.json")
        trials = []
        trial_root = directory / "trial-configs"
        trial_root.mkdir()
        for trial in config["language_study"]["trials"]:
            identifier = trial["id"]
            trial_config = copy.deepcopy(config)
            trial_config["recipe"] = trial["recipe"]
            trial_config.pop("language_study", None)
            config_path = trial_root / f"{identifier}.json"
            config_path.write_text(json.dumps(trial_config, indent=2) + "\n", encoding="utf-8")
            root = f"trials/{identifier}"
            try:
                self._stage(job, f"Applying {trial.get('label', identifier)}", "compress",
                            f"{root}/compressed", checkpoint=original, config_path=config_path)
                bundle = directory / root / "compressed" / "bundle"
                self._stage(job, f"Validating reloaded {trial.get('label', identifier)}", "evaluate",
                            f"{root}/validation", checkpoint=bundle,
                            extra=("--role", "validation"), config_path=config_path)
                direct = read_json(directory / root / "compressed" / "compress.json")
                measured = read_json(directory / root / "validation" / "evaluate.json")
                if measured.get("example_ids") != baseline.get("example_ids"):
                    raise ValueError("Validation example IDs changed between the baseline and trial.")
                if (not isinstance(measured.get("nll"), (int, float))
                        or not math.isfinite(measured["nll"])):
                    raise ValueError("Validation NLL is missing or nonfinite.")
                from ..comparison import bundle_bytes
                trials.append({"id": identifier, "label": trial.get("label", identifier), "status": "ok",
                               "recipe": trial["recipe"], "nll": measured["nll"],
                               "perplexity": measured["perplexity"], "tokens": measured["tokens"],
                               "example_ids": measured["example_ids"],
                               "example_valid_tokens": measured.get("example_valid_tokens", []),
                               "applied_methods": direct["applied_methods"],
                               "compressed_tensor_bytes": direct["compressed_tensor_bytes"],
                               "serialized_bundle_bytes": bundle_bytes(bundle),
                               "bundle": str(bundle.relative_to(directory))})
            except Exception as error:
                trials.append({"id": identifier, "label": trial.get("label", identifier),
                               "status": "failed", "recipe": trial["recipe"], "error": str(error)})
        successful = [trial for trial in trials if trial["status"] == "ok"]
        if not successful:
            raise RuntimeError("Every language compression trial failed. Review each trial and the run log.")
        selected = min(successful, key=lambda trial: (trial["nll"], trial["id"]))
        selected_bundle = directory / selected["bundle"]
        study = {"schema_version": 1, "selection_metric": "validation_nll",
                 "selection_rule": "lowest finite validation NLL; test metrics were not evaluated during trial selection",
                 "calibration_provenance": read_json(directory / "original-calibration" / "evaluate.json"),
                 "baseline_validation": baseline, "trials": trials, "selected_trial": selected["id"],
                 "failed_trials": sum(trial["status"] != "ok" for trial in trials)}
        study_dir = directory / "study"
        study_dir.mkdir()
        (study_dir / "study.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        self._stage(job, "Evaluating uncompressed held-out test baseline", "evaluate", "original-evaluation",
                    checkpoint=original, extra=("--role", "test"))
        self._stage(job, "Evaluating selected reloaded artifact on held-out test", "evaluate",
                    "compressed-evaluation", checkpoint=selected_bundle, extra=("--role", "test"))
        self._stage(job, "Benchmarking uncompressed model", "benchmark", "original-benchmark",
                    checkpoint=original)
        self._stage(job, "Benchmarking selected reloaded artifact", "benchmark", "compressed-benchmark",
                    checkpoint=selected_bundle)
        self._stage(job, "Verifying generation and attention cache", "generate", "generation",
                    checkpoint=selected_bundle)
        onnx_dir = directory / "language-onnx"
        try:
            self._stage(job, "Time-boxed fixed-logits ONNX investigation", "export-language-onnx",
                        "language-onnx", checkpoint=selected_bundle,
                        extra=("--role", "validation"), timeout=900)
            with tarfile.open(directory / "language-fixed-logits-onnx.tar", "w") as output:
                for path in sorted(onnx_dir.iterdir()):
                    output.add(path, arcname=path.name)
        except Exception as error:
            onnx_dir.mkdir(exist_ok=True)
            (onnx_dir / "investigation.json").write_text(json.dumps({
                "status": "not_verified", "serving_ready": False, "time_box_seconds": 900,
                "error": str(error),
                "limitation": "No ONNX artifact is published unless fixed-shape logits parity passes. "
                              "Autoregressive cache export remains required for serving readiness."
            }, indent=2) + "\n", encoding="utf-8")
        plan = read_json(directory / "trials" / selected["id"] / "compressed" / "compression_plan.json")
        from ..comparison import build_comparison, write_comparison
        report = build_comparison(
            read_json(directory / "original-evaluation" / "evaluate.json"),
            read_json(directory / "compressed-evaluation" / "evaluate.json"),
            read_json(directory / "original-benchmark" / "benchmark.json"),
            read_json(directory / "compressed-benchmark" / "benchmark.json"), plan,
            original_bundle=original, compressed_bundle=selected_bundle)
        write_comparison(report, directory / "comparison")
        archive = directory / "selected-pytorch-bundle.tar"
        with tarfile.open(archive, "w") as output:
            output.add(selected_bundle, arcname="bundle", recursive=True)
        study.update(selected_test=read_json(directory / "compressed-evaluation" / "evaluate.json"),
                     comparison_status=report["status"], selected_bundle_archive=archive.name)
        (study_dir / "study.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        job.update(status="completed", stage="Language study complete", summary=report["status"])
        if report["status"] != "verified":
            job["error"] = "; ".join(report.get("reasons", [])) or "Selected artifact did not verify."

    def _execute(self, job):
        directory = self._directory(job["id"])
        try:
            self._stage(job, "Saving original model", "snapshot", "original", checkpoint=job["checkpoint"])
            original = directory / "original" / "bundle"
            if job["mode"] == "analyze":
                self._stage(job, "Measuring layer relevance", "relevance", "relevance", checkpoint=original)
                job.update(status="completed", stage="Layer relevance complete",
                           summary="Gradient-based layer relevance is ready. These scores are diagnostic, not a compression quality guarantee.")
            else:
                config = read_json(directory / "config.json")
                if config.get("language_study"):
                    self._language_study(job, config, original)
                    return
                if config.get("recipe"):
                    self._stage(job, "Applying selected compression methods", "compress", "compressed", checkpoint=original)
                    plan_path = directory / "compressed" / "compression_plan.json"
                else:
                    self._stage(job, "Analyzing candidates", "analyze", "analysis", checkpoint=original,
                                extra=("--level", "validated"))
                    plan_path = directory / "analysis" / "compression_plan.json"
                    plan = read_json(plan_path)
                    if plan.get("status") != "feasible" or not plan.get("transformations"):
                        job.update(status="infeasible", stage="No compression plan meets the request",
                                   summary="Review the target, quality tolerance and candidate coverage. No compressed artifact was produced.")
                        return
                    self._stage(job, "Applying compression plan", "compress", "compressed", checkpoint=original,
                                extra=("--plan", str(plan_path)))
                plan = read_json(plan_path)
                compressed = directory / "compressed" / "bundle"
                for name, bundle in (("original", original), ("compressed", compressed)):
                    self._stage(job, f"Evaluating {name} on held-out data", "evaluate", f"{name}-evaluation",
                                checkpoint=bundle, extra=("--role", "test"))
                    self._stage(job, f"Benchmarking {name}", "benchmark", f"{name}-benchmark", checkpoint=bundle)
                from ..comparison import build_comparison, write_comparison
                report = build_comparison(
                    read_json(directory / "original-evaluation" / "evaluate.json"),
                    read_json(directory / "compressed-evaluation" / "evaluate.json"),
                    read_json(directory / "original-benchmark" / "benchmark.json"),
                    read_json(directory / "compressed-benchmark" / "benchmark.json"), plan,
                    original_bundle=original, compressed_bundle=compressed,
                    synthetic_baseline=read_json(directory / "config.json")["data"]["kind"] == "synthetic")
                write_comparison(report, directory / "comparison")
                if config["task"] == "segmentation":
                    self._stage(job, "Exporting and verifying segmentation ONNX", "export", "export",
                                checkpoint=compressed, extra=("--role", "validation"))
                baseline_passed = report["status"] in {"verified", "synthetic_only"}
                if not baseline_passed and not (report["status"] == "quality_failed" and job.get("recovery") != "none"):
                    job.update(status="failed", stage="Verification failed", summary=report["status"],
                               error="; ".join(report.get("reasons", [])) or "Final verification did not pass.")
                elif job.get("recovery") == "finetune":
                    self._stage(job, "Fine-tuning compressed vision model", "finetune", "recovered", checkpoint=compressed)
                    recovered = directory / "recovered" / "bundle"
                    self._stage(job, "Evaluating recovered model on held-out data", "evaluate", "recovered-evaluation",
                                checkpoint=recovered, extra=("--role", "test"))
                    self._stage(job, "Benchmarking recovered model", "benchmark", "recovered-benchmark",
                                checkpoint=recovered)
                    recovery_report = build_comparison(
                        read_json(directory / "original-evaluation" / "evaluate.json"),
                        read_json(directory / "recovered-evaluation" / "evaluate.json"),
                        read_json(directory / "original-benchmark" / "benchmark.json"),
                        read_json(directory / "recovered-benchmark" / "benchmark.json"), plan,
                        original_bundle=original, compressed_bundle=recovered,
                        synthetic_baseline=config["data"]["kind"] == "synthetic")
                    write_comparison(recovery_report, directory / "recovered-comparison")
                    recovered_passed = recovery_report["status"] in {"verified", "synthetic_only"}
                    job.update(status="completed" if recovered_passed else "failed",
                               stage="Recovery verification complete", summary=recovery_report["status"])
                    if not recovered_passed:
                        job["error"] = "; ".join(recovery_report.get("reasons", []))
                elif job.get("recovery") == "lora":
                    self._stage(job, "Training LoRA on compressed language model", "recover-lora", "lora",
                                checkpoint=compressed)
                    self._stage(job, "Evaluating LoRA on held-out data", "evaluate-lora", "recovered-evaluation",
                                checkpoint=compressed, extra=("--adapter", str(directory / "lora" / "adapter"),
                                                              "--role", "test"))
                    recovered_eval = read_json(directory / "recovered-evaluation" / "evaluate.json")
                    original_eval = read_json(directory / "original-evaluation" / "evaluate.json")
                    metric = plan["constraints"]["quality_metric"]
                    original_value, recovered_value = original_eval.get(metric), recovered_eval.get(metric)
                    maximum = plan["constraints"]["max_quality_loss"]
                    finite_values = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                        and math.isfinite(value) for value in (original_value, recovered_value))
                    loss = (((original_value - recovered_value) if plan["constraints"]["quality_direction"] == "higher"
                             else (recovered_value - original_value)) if finite_values else None)
                    quality_passed = isinstance(loss, (int, float)) and math.isfinite(loss) and loss <= maximum
                    recovery_quality = {"metric": metric, "original": original_value,
                                        "recovered": recovered_value, "loss": loss,
                                        "maximum_loss": maximum, "passed": quality_passed,
                                        "limitation": "LoRA adapter runtime and combined artifact size were not benchmarked."}
                    (directory / "recovery-quality.json").write_text(json.dumps(recovery_quality, indent=2))
                    job.update(status="completed" if quality_passed else "failed",
                               stage="Recovery evaluation complete",
                               summary="LoRA quality measured on held-out data; deployment cost remains unverified.")
                    if not quality_passed:
                        job["error"] = "LoRA-recovered model did not meet the selected quality limit."
                else:
                    job.update(status="completed", stage="Verification complete", summary=report["status"])
        except Exception as error:
            job.update(status="failed", error=str(error))
        finally:
            job["finished_at"] = utc_now()
            self._save(job)
