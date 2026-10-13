"""Optional GPU workers for native compressed-tensors artifacts and vLLM evaluation."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import threading
import time

from .llm_pilot import read, score_probes, sha256


def versions():
    result = {"python": platform.python_version()}
    for package in ("torch", "transformers", "datasets", "llmcompressor", "compressed-tensors", "vllm"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def preflight(environment):
    installed = versions()
    package, expected = ("llmcompressor", "0.14.0") if environment == "compression" else ("vllm", "0.31.0")
    if installed.get(package) != expected:
        raise ValueError(f"Use the separate {environment} environment: {package}=={expected} required.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --dry-run locally and run GPU stages on the other PC.")
    if torch.cuda.get_device_capability() < (7, 5):
        raise RuntimeError("Pinned vLLM WNA16 requires SM 7.5 (Turing) or newer; no fallback is silently applied.")
    packages = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()
                if item.metadata.get("Name")}
    return {"status": "ok", "versions": installed, "packages": dict(sorted(packages.items())), "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()), "cuda": str(torch.version.cuda)}


def prepare(root, study):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from .tasks.language import text_records
    config = study["config"]
    model = config["model"]
    import os
    path = snapshot_download(model["name"], revision=model["revision"],
                             local_files_only=os.environ.get("HF_HUB_OFFLINE") != "0",
                             allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    records = text_records(config)
    token_ids = tokenizer("\n\n".join(row["text"] for row in records["calibration"]),
                          add_special_tokens=False)["input_ids"]
    maximum = max(config["pilot"]["prompt_lengths"])
    if len(token_ids) < maximum:
        raise ValueError("Calibration text is too short for the recorded benchmark workload.")
    inputs = {"model_path": path, "records": records,
              "benchmark_token_ids": {str(n): token_ids[:n] for n in config["pilot"]["prompt_lengths"]}}
    target = root / "inputs.json"
    target.write_text(json.dumps(inputs, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "ok", "artifact_hashes": {str(target): sha256(target)},
            "example_ids": {role: [row["id"] for row in rows] for role, rows in records.items()},
            "model_revision": model["revision"], "versions": versions()}


def recipe(method):
    if method == "rtn":
        from llmcompressor.modifiers.quantization import QuantizationModifier
        return QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    if method == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    if method == "awq":
        from llmcompressor.modifiers.transform.awq import AWQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier
        return [AWQModifier(duo_scaling="both"),
                QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])]
    raise ValueError(f"Unsupported native compression method: {method}")


def compress(root, study, method):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from llmcompressor import oneshot
    from .models import load_model
    torch.manual_seed(study["config"]["seed"])
    inputs = read(root / "inputs.json")
    tokenizer = AutoTokenizer.from_pretrained(inputs["model_path"], local_files_only=True, trust_remote_code=False)
    model = load_model({"source": "transformers", "name": inputs["model_path"], "weights": "pretrained",
                        "kwargs": {"dtype": torch.float16, "device_map": "cuda:0", "local_files_only": True}})
    maximum = study["config"]["data"]["max_length"]
    dataset = Dataset.from_list([tokenizer(row["text"], truncation=True, max_length=maximum,
                                          add_special_tokens=False)
                                 for row in inputs["records"]["calibration"]])
    oneshot(model=model, dataset=dataset, recipe=recipe(method), max_seq_length=maximum,
            num_calibration_samples=len(dataset))
    # Failed attempts remain inspectable; retry never overwrites a partial checkpoint.
    base = root / "artifacts"
    base.mkdir(exist_ok=True)
    attempt = len(list(base.glob(f"{method}-attempt-*"))) + 1
    output = base / f"{method}-attempt-{attempt:03d}"
    output.mkdir()
    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    provenance = {"model": study["config"]["model"], "git_revision": study["git_revision"],
                  "seed": study["config"]["seed"], "method": method, "scheme": "W4A16",
                  "ignore": ["lm_head"], "versions": versions(),
                  "calibration_role": "weight-only rounding; shared input pass" if method == "rtn" else "data-aware fitting",
                  "calibration_ids": [row["id"] for row in inputs["records"]["calibration"]]}
    (output / "compression-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    hashes = {str(path): sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    return {"status": "ok", "method": method, "artifact": str(output), "artifact_hashes": hashes,
            "format": "compressed-tensors", "scheme": "W4A16", "ignore": ["lm_head"],
            "calibration_ids": [row["id"] for row in inputs["records"]["calibration"]],
            "versions": versions(), "runtime_compatibility": "requires the evaluation stage"}


def prompt_nll(outputs, token_sequences):
    """vLLM prompt logprobs include the observed token, even outside the top-k."""
    if len(outputs) != len(token_sequences):
        raise ValueError("Missing likelihood outputs.")
    loss = tokens = 0
    for output, ids in zip(outputs, token_sequences):
        if output.prompt_logprobs is None or len(output.prompt_logprobs) != len(ids):
            raise ValueError("Missing or misaligned prompt likelihoods.")
        for index, token in enumerate(ids[1:], start=1):
            value = output.prompt_logprobs[index][token].logprob
            if not math.isfinite(value) or value > 1e-6:
                raise ValueError("Invalid observed-token log probability.")
            loss -= value
            tokens += 1
    if not tokens:
        raise ValueError("No valid next-token targets.")
    return {"nll": loss / tokens, "tokens": tokens, "sequences": len(outputs)}


def evaluate(root, study, method, role):
    import torch
    samples = []
    stopped = threading.Event()
    free, total = torch.cuda.mem_get_info()
    before = total - free
    def sample():
        while not stopped.is_set():
            free, total = torch.cuda.mem_get_info()
            samples.append(total - free)
            stopped.wait(.02)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        result = _evaluate(root, study, method, role)
    finally:
        stopped.set()
        sampler.join()
    result.update(device_used_before_load_bytes=before,
                  device_used_peak_bytes_sampled=max(samples, default=before),
                  memory_sampling_interval_seconds=.02)
    return result


def tokenize_probes(tokenizer, probes):
    # Transformers 5 defaults to a dictionary; vLLM requires a list of token IDs.
    return [tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                         tokenize=True, return_dict=False, add_generation_prompt=True)
            for row in probes]


def _evaluate(root, study, method, role):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    inputs = read(root / "inputs.json")
    config = study["config"]
    pilot = config["pilot"]
    source = inputs["model_path"] if method == "dense" else read(root / "stages" / f"{method}-pack.json")["artifact"]
    tokenizer = AutoTokenizer.from_pretrained(inputs["model_path"], local_files_only=True, trust_remote_code=False)
    started = time.perf_counter()
    engine = LLM(model=source, tokenizer=inputs["model_path"], dtype="half", trust_remote_code=False,
                 seed=config["seed"], max_model_len=pilot["max_length"], max_num_seqs=1,
                 gpu_memory_utilization=pilot["gpu_memory_utilization"], enforce_eager=True,
                 kv_cache_memory_bytes=pilot["kv_cache_memory_bytes"],
                 enable_prefix_caching=False)
    load_seconds = time.perf_counter() - started
    records = inputs["records"][role]
    sequences = [tokenizer(row["text"], truncation=True, max_length=config["data"]["max_length"],
                           add_special_tokens=False)["input_ids"] for row in records]
    if any(len(ids) < 2 for ids in sequences):
        raise ValueError("An evaluation example has no next-token targets.")
    likelihood = engine.generate([{"prompt_token_ids": ids} for ids in sequences],
                                 SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1), use_tqdm=False)
    quality = prompt_nll(likelihood, sequences)
    probes = study["probes"][role]
    prompts = tokenize_probes(tokenizer, probes)
    if any(len(ids) + pilot["output_tokens"] > pilot["max_length"] for ids in prompts):
        raise ValueError("Behavior prompt would be truncated; increase max_length explicitly.")
    responses = engine.generate([{"prompt_token_ids": ids} for ids in prompts],
                                SamplingParams(temperature=0, max_tokens=pilot["output_tokens"]), use_tqdm=False)
    behavior = score_probes(probes, [row.outputs[0].text for row in responses])
    workloads = []
    for length, ids in inputs["benchmark_token_ids"].items():
        params = SamplingParams(temperature=0, max_tokens=pilot["output_tokens"],
                                min_tokens=pilot["output_tokens"], ignore_eos=True, detokenize=False)
        timings = []
        for index in range(pilot["warmup"] + pilot["iterations"]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = engine.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)
            torch.cuda.synchronize()
            if len(generated[0].outputs[0].token_ids) != pilot["output_tokens"]:
                raise ValueError("Benchmark did not generate the fixed output length.")
            if index >= pilot["warmup"]:
                timings.append((time.perf_counter() - start) * 1000)
        workloads.append({"prompt_tokens": int(length), "output_tokens": pilot["output_tokens"],
                          "latencies_ms": timings, "latency_mean_ms": statistics.mean(timings),
                          "latency_p50_ms": statistics.median(timings),
                          "latency_p95_ms": sorted(timings)[math.ceil(.95 * len(timings)) - 1]})
    payload = {"role": role, "ids": [row["id"] for row in records], "token_sequences": sequences,
               "probes": probes, "benchmark_token_ids": inputs["benchmark_token_ids"], "pilot": pilot}
    files = [path for path in Path(source).iterdir() if path.is_file()]
    free, total = torch.cuda.mem_get_info()
    return {"status": "ok", "method": method, "role": role, **quality, "behavior": behavior,
            "workload_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "serialized_bytes": sum(path.stat().st_size for path in files), "artifact": source,
            "latency_mean_ms": statistics.mean(row["latency_mean_ms"] for row in workloads),
            "workloads": workloads, "load_seconds": load_seconds, "versions": versions(),
            "runtime": "vllm", "gpu": torch.cuda.get_device_name(),
            "device_used_bytes_after_workload": total - free,
            "memory_scope": "device-wide snapshots and sampled peak include vLLM KV reservation and other processes; not per-process or weight-only memory",
            "timing_scope": "offline batch-one fixed-length generation, cache disabled between requests; no exact TTFT measurement",
            "artifact_hashes": {str(path): sha256(path) for path in files}}


def benchmark_definitions(package_root, specifications):
    """Load official pinned task definitions and pin their dataset revisions."""
    from lm_eval.tasks._yaml_loader import load_yaml
    locations = {"arc_easy": "arc/arc_easy.yaml", "hellaswag": "hellaswag/hellaswag.yaml"}
    tasks = []
    for item in specifications:
        path = package_root / "tasks" / locations[item["task"]]
        definition = load_yaml(path)
        definition["dataset_kwargs"] = {**definition.get("dataset_kwargs", {}), "revision": item["revision"]}
        tasks.append(definition)
    return tasks


def benchmarks(root, study, method, limit):
    if importlib.metadata.version("lm_eval") != "0.4.13":
        raise ValueError("Final public benchmarks require lm_eval==0.4.13.")
    import lm_eval
    from lm_eval import simple_evaluate
    inputs = read(root / "inputs.json")
    source = inputs["model_path"] if method == "dense" else read(root / "stages" / f"{method}-pack.json")["artifact"]
    pilot = study["config"]["pilot"]
    definitions = benchmark_definitions(Path(lm_eval.__file__).parent, study["config"]["benchmarks"])
    result = simple_evaluate(model="vllm", model_args={"pretrained": source, "tokenizer": inputs["model_path"],
                             "dtype": "float16", "gpu_memory_utilization": pilot["gpu_memory_utilization"],
                             "kv_cache_memory_bytes": pilot["kv_cache_memory_bytes"], "max_num_seqs": 1,
                             "max_model_len": pilot["max_length"], "enforce_eager": True,
                             "enable_prefix_caching": False}, tasks=definitions,
                             num_fewshot=0, batch_size=1, limit=None if limit == "full" else int(limit),
                             bootstrap_iters=1000, log_samples=True, apply_chat_template=False,
                             random_seed=study["config"]["seed"], numpy_random_seed=study["config"]["seed"],
                             torch_random_seed=study["config"]["seed"], fewshot_random_seed=study["config"]["seed"])
    if not result:
        raise ValueError("Public benchmark returned no results.")
    return {"status": "ok", "role": "final_benchmark", "method": method, "limit": limit,
            "dataset_revisions": study["config"]["benchmarks"], "versions": versions(), "results": result,
            "note": "Never used for compression selection; a limited run is a smoke check, not a leaderboard result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "prepare", "compress", "evaluate", "benchmarks"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--environment", choices=["compression", "runtime"], required=True)
    parser.add_argument("--method", choices=["dense", "gptq", "awq"], required=True)
    parser.add_argument("--role", choices=["validation", "test"], required=True)
    parser.add_argument("--benchmark-limit", default="64")
    args = parser.parse_args()
    try:
        if args.action == "preflight":
            result = preflight(args.environment)
        else:
            study = read(args.root / "study.json")
            if args.action == "prepare":
                result = prepare(args.root, study)
            elif args.action == "compress":
                result = compress(args.root, study, args.method)
            elif args.action == "evaluate":
                result = evaluate(args.root, study, args.method, args.role)
            else:
                selection = read(args.root / "selection.json")
                if args.role != "test" or args.method not in {"dense", selection["method"]}:
                    raise ValueError("Only dense and frozen selected artifacts may run final benchmarks.")
                result = benchmarks(args.root, study, args.method, args.benchmark_limit)
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}", "versions": versions()}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, allow_nan=False,
                                     default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n", encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
