"""Isolated model measurements with explicit loading and inference phases.

Measurement conventions were reconciled with arodrigo-RAM_Usage; no stored
industrial benchmark result is reused as evidence for these implementations.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def benchmark_bundle(path, *, input_shape=(1, 3, 256, 256), device="cpu", task="segmentation",
                     warmup=20, iterations=100, threads=1, timeout=600, seed=0, output_tokens=32):
    command = [sys.executable, "-m", "tn_compression.benchmark", str(Path(path).resolve()),
               json.dumps({"input_shape": input_shape, "device": device, "task": task,
                           "warmup": warmup, "iterations": iterations, "threads": threads, "seed": seed,
                           "output_tokens": output_tokens})]
    environment = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                   "OPENBLAS_NUM_THREADS": str(threads)}
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "runtime": "pytorch", "timeout_seconds": timeout}
    try:
        metrics = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"status": "failed", "runtime": "pytorch", "error": result.stderr[-2000:]}
    if result.returncode and metrics.get("status") == "ok":
        metrics.update(status="failed", error=result.stderr[-2000:])
    return metrics


def _decode(model, inputs, output_tokens, synchronize):
    """Batch-one, fixed-length greedy decoding; EOS is ignored for timing."""
    import torch
    cache = None
    mask = torch.ones_like(inputs)
    timings = []
    for index in range(output_tokens):
        started = time.perf_counter()
        output = model(input_ids=inputs, attention_mask=mask, past_key_values=cache, use_cache=True)
        inputs = output.logits[:, -1:].argmax(-1)
        cache = output.past_key_values
        if cache is None:
            raise ValueError("Language benchmarking requires a functioning generation cache.")
        synchronize()
        timings.append((time.perf_counter() - started) * 1000)
        mask = torch.cat((mask, torch.ones_like(inputs)), dim=1)
    return {"ttft_ms": timings[0], "inter_token_ms": timings[1:], "output_tokens": output_tokens}


def _worker(path, options):
    import psutil
    import torch
    from .checkpoints import load_bundle
    from .tasks.vision import evaluation_mode
    device = options["device"]
    if options["iterations"] < 1 or options["warmup"] < 0 or options["threads"] < 1:
        raise ValueError("Iterations/threads must be positive; warmup must be nonnegative.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return {"status": "unavailable", "runtime": "pytorch", "device": device, "error": "CUDA unavailable"}
    torch.set_num_threads(options["threads"])
    torch.manual_seed(options["seed"])
    process = psutil.Process()
    phase = "load"
    samples = {"load": [], "inference": []}
    stopped = threading.Event()

    def sample():
        while not stopped.is_set():
            samples[phase].append(process.memory_info().rss)
            stopped.wait(0.005)

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        rss_before = process.memory_info().rss
        start = time.perf_counter()
        model, _ = load_bundle(path, device=device)
        synchronize()
        load_seconds = time.perf_counter() - start
        rss_after = process.memory_info().rss
        language = options["task"] == "causal_lm"
        if language:
            if len(options["input_shape"]) != 2 or options["input_shape"][0] != 1 or options["input_shape"][1] < 1:
                raise ValueError("Language timing requires input_shape=[1, prompt_length].")
            if options["output_tokens"] < 1:
                raise ValueError("output_tokens must be positive.")
            inputs = torch.randint(model.config.vocab_size, options["input_shape"], device=device)
        else:
            inputs = torch.randn(options["input_shape"], device=device, dtype=next(model.parameters()).dtype)
        if options["task"] == "detection":
            inputs = list(inputs.unbind(0))
        with evaluation_mode(model), torch.inference_mode():
            for _ in range(options["warmup"]):
                _decode(model, inputs, options["output_tokens"], synchronize) if language else model(inputs)
            synchronize()
            phase = "inference"
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            latencies = []
            requests = []
            for _ in range(options["iterations"]):
                start = time.perf_counter()
                if language:
                    requests.append(_decode(model, inputs, options["output_tokens"], synchronize))
                else:
                    model(inputs)
                synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
    finally:
        stopped.set()
        sampler.join()
    ordered = sorted(latencies)
    result = {
        "status": "ok", "runtime": "pytorch", **options, "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor(),
        "load_seconds": load_seconds, "rss_before_load_bytes": rss_before, "rss_after_load_bytes": rss_after,
        "rss_load_peak_bytes": max(samples["load"], default=rss_after),
        "rss_inference_peak_bytes": max(samples["inference"], default=process.memory_info().rss),
        "rss_sampling_interval_seconds": 0.005,
        "latency_mean_ms": statistics.mean(latencies), "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "latencies_ms": latencies, "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": str(next(model.parameters()).dtype), "timing_scope": "complete_model_forward",
    }
    if device.startswith("cuda"):
        result.update(gpu=torch.cuda.get_device_name(device),
                      cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(device))
    if language:
        result.update(timing_scope="fixed-length greedy generation including Python loop",
                      workload="seeded synthetic token IDs; batch=1; EOS ignored; no request queue",
                      requests=requests, output_tokens_per_second=options["output_tokens"] * len(latencies) / (sum(latencies) / 1000),
                      kv_cache_peak_bytes=None)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(_worker(sys.argv[1], json.loads(sys.argv[2]))))
    except Exception as error:
        status = "oom" if "out of memory" in str(error).lower() else "failed"
        print(json.dumps({"status": status, "error": f"{type(error).__name__}: {error}", "runtime": "pytorch"}))
        sys.exit(1)
