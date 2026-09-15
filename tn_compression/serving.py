"""Measure a streaming completion workload against a separately started server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


def read_events(lines, *, started, clock=time.perf_counter):
    """Time content events; never pretend one SSE chunk equals one token."""
    arrivals = []
    usage = None
    finished = False
    for line in lines:
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].strip()
        if payload == b"[DONE]":
            finished = True
            break
        event = json.loads(payload)
        if "error" in event:
            raise RuntimeError(str(event["error"]))
        if event.get("usage"):
            usage = event["usage"]
        if any(choice.get("text") for choice in event.get("choices", [])):
            arrivals.append(clock() - started)
    if not finished or not arrivals or not usage or usage.get("completion_tokens", 0) < 1:
        raise ValueError("Incomplete stream: need content, final token usage and [DONE].")
    tokens = usage["completion_tokens"]
    return {"ttft_seconds": arrivals[0], "latency_seconds": clock() - started,
            "tpot_seconds": (arrivals[-1] - arrivals[0]) / (tokens - 1) if tokens > 1 else None,
            "content_event_times_seconds": arrivals, "usage": usage}


def completion(endpoint, payload, timeout):
    started = time.perf_counter()
    request = Request(endpoint.rstrip("/") + "/v1/completions",
                      data=json.dumps({**payload, "stream": True, "stream_options": {"include_usage": True}}).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            result = read_events(response, started=started)
        return {"status": "ok", **result}
    except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as error:
        return {"status": "failed", "error": str(error), "latency_seconds": time.perf_counter() - started}


def benchmark_server(config):
    """Closed-loop concurrency with a shared work queue and per-request raw data."""
    concurrency = config.get("concurrency", 1)
    prompts = config["prompts"]
    if concurrency < 1 or not prompts or config.get("max_tokens", 32) < 1:
        raise ValueError("Positive concurrency/output length and nonempty prompts are required.")
    endpoint = config.get("endpoint", "http://127.0.0.1:8000")
    timeout = config.get("timeout_seconds", 120)

    def run(prompt):
        return completion(endpoint, {"model": config["model"], "prompt": prompt,
                          "temperature": 0, "max_tokens": config.get("max_tokens", 32)}, timeout)

    for _ in range(config.get("warmup", 1)):
        warmed = run(prompts[0])
        if warmed["status"] != "ok":
            return {"schema_version": 1, "status": "unavailable", "config": config, "warmup": warmed}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        rows = list(executor.map(run, prompts))
    elapsed = time.perf_counter() - started
    successful = [row for row in rows if row["status"] == "ok"]
    metrics = {}
    for key in ("ttft_seconds", "tpot_seconds", "latency_seconds"):
        values = [row[key] for row in successful if row[key] is not None]
        metrics[key] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if values else None
    return {"schema_version": 1, "status": "ok" if len(successful) == len(rows) else "failed",
            "config": config, "requests": rows, "wall_seconds": elapsed, "metrics": metrics,
            "output_tokens_per_second": sum(row["usage"]["completion_tokens"] for row in successful) / elapsed,
            "arrival_policy": "closed-loop shared queue; at most concurrency active requests",
            "definitions": {"ttft": "request start to first nonempty content event",
                            "tpot": "first-to-last content event time / (actual output tokens - 1)",
                            "latency": "request start through final stream marker"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark_server(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
