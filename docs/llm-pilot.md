# Native LLM compression pilot

This first expansion compares a pinned Qwen2.5-1.5B-Instruct baseline with GPTQ
and AWQ W4A16 (4-bit weights, FP16 activations), then loads their actual native
artifacts in vLLM. It does not replace the existing Qwen demo or its recorded
result. GPU packing, cross-version checkpoint compatibility and speed are
**not yet verified**. CPU checks establish controller and scoring behavior only.

## What the study measures

- Token-weighted NLL on pinned WikiText examples, with explicit calibration,
  validation and test roles and recorded IDs.
- Separate authored diagnostic probes for extraction, instruction following and
  arithmetic, including a small Spanish instruction check. These are integration
  probes, not a representative language benchmark or a multilingual quality claim.
- Serialized weights/config/tokenizer bytes, load time, fixed-length batch-one
  generation at 128 and 512 input tokens, raw times and p50/p95.
- Sampled device-wide GPU memory, including vLLM KV-cache reservation and other
  processes. Use an otherwise idle GPU; this is not isolated weight memory.
- Optional final ARC-Easy and HellaSwag through pinned lm-evaluation-harness task
  definitions and pinned dataset commits, including per-example outputs.

The controller screens NLL and per-slice probe degradation on validation only,
retains failed/slower configurations, and reports a bytes/latency/NLL Pareto
frontier including sampled device memory. Choose the trade-off explicitly before final test evaluation. No test
metric enters selection, and only dense and the frozen selected candidate run
the final public suite. HellaSwag's labeled validation split is held out from
compression selection; this does not assert absence from original pretraining.

Default bounds (NLL increase <= 0.05, probe success loss <= 0.10) are configurable
pilot settings, not statistically established guarantees. Four probes per slice
cannot establish a small quality margin. A limited public suite is a smoke check;
use the full suite and application-specific held-out data before making claims.
WikiText-only calibration is also narrower than typical instruction-model usage.

Dense and compressed inference use the same explicit 256 MiB KV cache, context
limit and batch-one workload. This prevents automatic KV allocation from filling
the same GPU-memory fraction and hiding weight-memory differences. vLLM ignores
`gpu_memory_utilization` when explicit KV-cache bytes are supplied. Memory remains
sampled device usage, so other GPU processes can invalidate its comparison.

## Environments and hardware

Use a Linux NVIDIA host with SM 7.5 or newer and a driver compatible with the
CUDA wheels. The pinned [vLLM WNA16 implementation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py)
declares SM 7.5 minimum; model shapes, kernels, drivers and available VRAM still
need actual verification. A P100 is unsupported. A T4 meets the architecture
floor but installed CUDA wheels and kernels may still be incompatible.

The compression and inference environments are separate: LLM Compressor 0.14.0
uses compressed-tensors 0.19.0, while vLLM 0.31.0 pins 0.17.0. Native W4 checkpoint
loading across them is an explicit GPU verification gate, not an assumed result.
If it fails, keep the failed logs and resolve a supported pair; do not bypass the
resolver with forced dependency versions. Core Transformers 4.51 stays unchanged.

Budget at least the roughly 3 GB FP16 model weights plus calibration/Hessian
state, activations, packing temporaries and runtime/KV memory. A 16 GB GPU is a
reasonable pilot target, not a verified memory guarantee. No measured runtime
estimate exists for this integration. Installs/downloads may dominate the first run.
Each worker has a 30-minute timeout; there are two compression candidates and one
seed. There is no training. Interrupted/failed attempts remain inspectable.

## CPU checks

The dry run needs only Python's standard library and starts no worker, download or
GPU operation. It does not create the output directory:

```bash
python -m tn_compression.llm_pilot --output-dir runs/llm-pilot --dry-run
python -m tn_compression.llm_pilot --output-dir runs/llm-pilot \
  --dry-run --finalize --select gptq --benchmarks
python -m pytest -q tests/test_llm_pilot.py tests/test_pet_pilot.py
```

On the local Windows PC, the existing CPU image can run checks without rebuilding:

```powershell
docker run --rm --network none --entrypoint python `
  -v "${PWD}:/app:ro" -w /app -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 `
  tensor-decomposition-compression-ui -m pytest -q tests/test_llm_pilot.py tests/test_pet_pilot.py -p no:cacheprovider
```

## Other PC / Colab / Kaggle

Use a reviewed checkout containing these files. Record its commit with
`git rev-parse HEAD`; Git must be available. Do not run the following installs or
study commands on the local CPU PC. They download GPU packages/model assets.

Colab: mount Drive with `from google.colab import drive; drive.mount('/content/drive')`
and place `RUN_DIR` under `/content/drive/MyDrive`. Kaggle: use `/kaggle/working`
and save a notebook version/output before ending the session. Their free GPU
availability, quotas and installed drivers vary; inspect the actual accelerator.

```bash
nvidia-smi
python -m venv /tmp/tn-compression-env
python -m venv /tmp/tn-runtime-env
/tmp/tn-compression-env/bin/python -m pip install -r requirements/llm-compression.txt
/tmp/tn-runtime-env/bin/python -m pip install -r requirements/llm-runtime.txt
# Run from the repository; module execution uses the source checkout directly.
export RUN_DIR=/content/drive/MyDrive/tn-compression/llm-pilot
# Kaggle alternative: export RUN_DIR=/kaggle/working/llm-pilot
export HF_HOME="$RUN_DIR/hf-cache"
python -m tn_compression.llm_pilot --output-dir "$RUN_DIR" --dry-run \
  --compression-python /tmp/tn-compression-env/bin/python \
  --runtime-python /tmp/tn-runtime-env/bin/python

# First real pilot; explicitly permits remote asset downloads.
python -m tn_compression.llm_pilot --output-dir "$RUN_DIR" --allow-download \
  --compression-python /tmp/tn-compression-env/bin/python \
  --runtime-python /tmp/tn-runtime-env/bin/python
python -m json.tool "$RUN_DIR/validation-report.json"
```

Inspect quality and each latency workload; do not pick a method solely by filename
size. Rerun the same command to resume. Completed results/checkpoints are hashed;
changed source, config, probes or package inventories are refused. Environments
are preflighted afresh before any completed stages are reused. Freeze resolved
dependencies with each environment's `python -m pip freeze` for rebuilding later.
Avoid concurrent controllers writing the same output directory.

After inspecting validation, choose a feasible candidate (example: `gptq`):

```bash
python -m tn_compression.llm_pilot --output-dir "$RUN_DIR" --allow-download \
  --compression-python /tmp/tn-compression-env/bin/python \
  --runtime-python /tmp/tn-runtime-env/bin/python \
  --finalize --select gptq --benchmarks --benchmark-limit 64
python -m json.tool "$RUN_DIR/test-report.json"
python -m json.tool "$RUN_DIR/stages/gptq-benchmarks-64.json"
```

The 64-example public run is only a short integration check. Expand the frozen
candidate to the complete public suite without rerunning completed compression:

```bash
python -m tn_compression.llm_pilot --output-dir "$RUN_DIR" --allow-download \
  --compression-python /tmp/tn-compression-env/bin/python \
  --runtime-python /tmp/tn-runtime-env/bin/python \
  --finalize --select gptq --benchmarks --benchmark-limit full
```

Keep this choice frozen after seeing final scores. Create a new study for a new
decision policy. For three seeds, write three config copies changing only `seed`
to 0/1/2 and use distinct output directories; report each seed and calibration IDs.
Different seeds resample calibration/evaluation IDs here: interpret the variation
as pipeline variability, not merely recovery-training randomness.

## Evidence and remaining scope

Worker logs and results live under `stages/`; native checkpoints and unsuccessful
attempts live under `artifacts/`. `study.json` records source hashes, config, probes,
revision and worker paths. Environment reports record actual installed versions.
`selection.json` freezes the chosen method and validation-report hash.

This milestone does not yet include agent selection, recovery training, optimized
CNN pruning, audio routes, serving concurrency, exact TTFT or a UI for these native
artifacts. Those remain staged work in [the expansion plan](design/compression-expansion.md).
The backend integrations follow the official [GPTQ example](https://github.com/vllm-project/llm-compressor/blob/0.14.0/examples/quantization_w4a16/llama3_example.py)
and [AWQ example](https://github.com/vllm-project/llm-compressor/blob/0.14.0/examples/awq/llama_example.py),
with both candidates using symmetric W4A16 for the matched-budget pilot.

## Local verification record (2026-10-13)

The final working-tree check in the existing CPU image, with networking disabled:

```text
python -m pytest -q -rs -p no:cacheprovider --disable-warnings
208 passed, 6 skipped, 18 warnings in 65.32s
```

Five analyzer-shell checks and one Pet-shell check were skipped because the CPU
image lacks the required Git/Bash tool combination. Existing warnings concern
Matplotlib/PyParsing, fixed-shape ONNX tracing and the tiny LoRA fixture.
Focused LLM/Pet checks passed: 14 passed, 1 shell skip. Standard-library CPU dry
runs (including final benchmark command generation), Python syntax and
`git diff --check` passed. Dry runs created no output directory.

Independent source review caught and verified fixes for Transformers 5 chat
template output, the pinned evaluation-harness YAML loader, and immutable resume
after an unselected failure. The suite included four pre-existing local UI/direct
recipe edits; those edits are outside this milestone's commits. No new GPU
packages or model weights were installed/downloaded locally. Native GPU execution,
public benchmark execution and performance results remain unverified.
