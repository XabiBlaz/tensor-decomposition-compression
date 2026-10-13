# Compression expansion: implementation and evidence plan

Status: proposed architecture, 2026-10-13. This document does not claim new methods,
GPU measurements, or audio support are implemented.

## Starting point

GitHub `main` was inspected at `33bb8044aaf7388f69cf82a33f6399f510c8c670`.
The initial planning turn could not start the local shell. Development resumed
after commands became available; pre-existing UI/direct-recipe changes were preserved.

Reuse the CLI, model loaders, task evaluators, analyzer capability reports,
compression implementations, bundles, comparison reports and benchmark workers.
The core language environment currently pins Transformers `>=4.51,<4.52`.
Do not install current GPU compression libraries into that environment without
resolving and verifying their dependency constraints.

The recorded Qwen result establishes a storage/quality trade-off, with slower
inference in the measured implementation. The Pet segmentation study is prepared;
its real GPU results remain pending.

## Evidence standard

Integrating a paper's method is implementation evidence. Calling a result
state-of-the-art requires a defined task, model, dataset, target runtime and
hardware, appropriate competing methods and repeatable measurements.

Publish the method, source/version, calibration cost, quality, actual artifact
bytes, memory and latency. Keep failed, infeasible and slower trials. Label each
capability separately as implemented, CPU-verified, GPU-verified or experimental.
CPU lifecycle tests cannot establish CUDA kernel correctness or speed.

## Architecture

```text
CLI / UI / notebook
    -> explicit experiment specification
    -> capability checks + bounded candidate selection
    -> existing PyTorch implementation OR isolated external backend worker
    -> native artifact + provenance manifest
    -> task evaluation + deployment benchmark
    -> validation selection -> frozen recipe -> held-out test report
```

Keep the existing package layout. Add a backend module only for an actual
integration, and a task module only when its first runnable study needs it.
Avoid a general plugin framework, remote service scheduler or arbitrary code tools.

Every runnable experiment identifies:

- The model revision/checksum, tokenizer or processor revision, and preprocessing.
- Dataset revision, recorded IDs and separate training, calibration, validation
  and final test roles. Recovery uses training data, never test data.
- Methods, legal layer groups, transformation order, protected layers and backend.
- The quality metric, direction and units, objective, constraints and trial budget.
- Device, inference shapes/context lengths, batch/concurrency and timing protocol.
- Exact worker environment, Git revision, seed, artifact hashes and run directory.

Extend the existing capability contract to the combination of model architecture,
layer layout, method, artifact format, runtime, GPU architecture and dtype.
Reject unsupported combinations before expensive work. Validate mixed methods in
their declared order: pruning, decomposition and quantization do not generally commute.

Keep external artifacts native: compressed-tensors/Hugging Face checkpoints,
CTranslate2 directories, ONNX files or TensorRT engines as appropriate. A small
manifest links their provenance and reports. Do not reinterpret every artifact as
an existing tensor-factor bundle. TensorRT engine portability must be checked
against the target device/runtime; retain its reproducible build configuration.

Use separately pinned optional GPU environments for backends with incompatible
dependencies. The CPU installation and existing offline examples must keep working.
Do not upgrade the core Transformers pin merely to satisfy one backend.

## Implementation sequence

### 1. Deployable LLM quantization and evaluation

Confirmed first milestone: LLM compression for open models on NVIDIA GPUs.
Retain Qwen2.5-0.5B as a regression fixture, then use one pinned 1–3B causal LM
for the first meaningful comparison if memory permits. A Qwen2.5-1.5B-Instruct
study is a sensible starting point, subject to resolving its exact revision,
license, backend support and calibration/evaluation data before execution.
The fixture proves lifecycle compatibility, not broad LLM quality.

Start with one maintained backend, vLLM LLM Compressor, for GPTQ and AWQ W4A16
(4-bit weights, 16-bit activations), with a dense baseline on the same runtime.
Build on the existing GPTQ example and provenance rather than duplicating it.
Verify native checkpoint reload and actual runtime loading. A factorized model
that only loads in PyTorch is not a vLLM-compatible artifact.

Use matched artifact budgets for RTN, GPTQ and AWQ when supported. Add AutoRound
as a later comparison, and rotation methods such as SpinQuant only with a verified
artifact/runtime path. FP8 and FP4 options must be gated by their precise scheme
and kernel requirements; a GPU supporting CUDA does not support every format.

Keep current SVD/weighted SVD and gated-MLP pruning as research comparisons.
Review SVD-LLM/V2 as a reference for low-rank improvements, accounting for its
architecture and dependency assumptions. Add sparse methods such as Wanda or
SparseGPT only with explicit sparse layout, storage and kernel support. Zeroed
weights in a dense tensor do not demonstrate compressed storage or acceleration.

Exit criteria: CPU checks pass; a prepared GPU runner can produce baseline and
compressed native artifacts, quality evaluations and matched runtime benchmarks;
unsupported kernels stop clearly. GPU verification remains an external run.

### 2. Real CNN evidence and structured pruning

Complete the Oxford Pet comparison before adding several new vision tasks.
Then integrate dependency-aware channel pruning through Torch-Pruning/DepGraph
on one supported ResNet or U-Net. Treat residual connections, normalization and
coupled channel dimensions as pruning dependencies. Save a safe reconstruction
description and test strict reload; do not rely on unsafe arbitrary module pickle.

Compare dense, existing Tucker/SVD, channel pruning and a runtime-supported INT8
path at measured storage budgets. Start with post-training quantization; add QAT
or distillation only if the initial quality guardrail fails. If recovery is used,
give a dense control equivalent additional training data/steps and record costs.

Use PyTorch as the correctness reference. Compare dense and compressed ONNX
Runtime or TensorRT artifacts on the same target runtime and workload. A native
int8 storage module with dequantization is a separate reference, not an INT8 kernel
speed claim. Measure export parity and task quality after export.

### 3. Speech recognition, then TTS/multimodal

Interpret the initial audio task as STT/ASR. Start with one pinned Whisper model,
a recorded labeled speech subset, and CTranslate2 INT8 conversion/inference.
Record sample rate, resampling, language/task tokens, segmentation and decoding.
Evaluate WER/CER, audio seconds, latency/real-time factor and peak memory.
Keep preprocessing and decoding identical across comparable candidates.

TTS is a later, separate task adapter. Evaluate transcription-based intelligibility,
speaker similarity where applicable, latency/real-time factor and human listening
on a fixed protocol. Automatic proxies alone cannot establish perceptual quality.

Add one vision-language model only after language and vision routes are stable.
Calibration and evaluation must contain real multimodal inputs and preserved
processor/chat-template behavior; text-only perplexity is insufficient.

### 4. Compression decisions and optional agent

First improve deterministic selection: capability filtering, bounded layer/group
interventions and validation-constrained selection, with measured final budgets.
Compare against uniform rank/precision and weight-energy selection. Equal ranks
or bit labels do not imply equal achieved bytes or runtime cost.

Use exhaustive search only for tiny grids, otherwise a bounded Pareto or greedy
search. Full-model validation must check cumulative transformations; sums of
individual layer damage do not account for interactions.

Add an agent only if it improves results or reduces search cost at equal budgets.
Use LangGraph for durable control when needed, with Pi restricted to structured
candidate proposals. One orchestrator owns selection and execution. Reuse the
existing controller/runner instead if graph persistence adds no value yet.

The agent receives capability summaries and validation reports, and can propose
allowlisted recipes. A deterministic broker validates and executes them with
fixed argument arrays. Test data, Git actions, credentials, arbitrary shell and
code editing are unavailable. Isolate the whole process, restrict mounts/network,
limit memory/disk/time and proposals, and make resumed stages idempotent.

## Pre/post analysis and evaluation

Before compression: inventory layers, tied/protected parameters, activation
outliers, singular-value energy, numerical conditioning and baseline task quality.
Use existing gradient/Taylor relevance and activation reconstruction diagnostics.
CKA is optional and uses matched examples/layers; it measures similarity, not a
guarantee of task quality or causal importance.

After compression: task quality and per-example deltas, logits/teacher KL when
appropriate, reconstruction and matched representation diagnostics, strict
artifact reload, export parity and runtime measurements. Use task quality as the
primary acceptance criterion. When pruning changes dimensions, select diagnostics
that remain well-defined and report mappings rather than comparing arbitrary rows.

Use the existing quality contracts. The current analyzer enforces one quality
metric; multiple simultaneous guardrails or confidence bounds require an explicit
extension and tests. Do not claim they exist merely because reports contain
several metrics.

For LLMs, supplement token NLL with a pinned EleutherAI lm-evaluation-harness
suite appropriate to the model, such as HellaSwag and ARC-Easy for language
understanding, and instruction-following checks for instruct models. Add a small
representative deployment set of summarization, extraction and question-answering
requests when those are the intended uses. Score extraction/format constraints
deterministically and audit semantic response quality with an explicit human rubric
or an independently validated judge. Fix prompts, IDs, few-shot settings, chat
template and decoding. Report each task and input slice, not only an aggregate;
avoid tasks on which the dense model is already at chance, since preserving that
score says little about useful capability. Small pilot subsets are integration
evidence; final subsets must be frozen and large enough for the claim.
Measure prefill and decode separately, time to first token, time per output token,
throughput, memory and context/concurrency. Never count an SSE chunk as one token.

For CNNs, use classification accuracy or segmentation IoU/Dice, including per-class
results, small/thin structures and a boundary metric when the application needs it.
For ASR, report WER/CER with recorded text normalization, stratified by language,
accent, noise, duration and domain; report omissions, hallucinations and silence
behavior separately. Do not infer multilingual quality from one English test set.
For TTS, combine transcription-based intelligibility, pronunciation/number checks,
speaker similarity where applicable and blinded, randomized human listening on
the same prompts, covering long text and difficult words. Preserve sample rate
and decoding, record speaker references, and distinguish automated MOS estimates
from human ratings. An ASR proxy can miss unnatural speech; speaker embeddings
can miss pronunciation errors. Both tasks need latency/real-time factor and memory.

Choose quality guardrails in domain units and justify them from the dense baseline
and application needs. A single weighted average must not hide a serious regression
in one task or subgroup. Select thresholds before search and include paired
per-example differences and uncertainty in final reports where feasible.

Measure batch-1 latency and a throughput workload, with exact image shape,
preprocessing, warmup, CUDA synchronization and repeated raw timings.

Use one seed for a pilot and expand to three for recovery/search variability.
Report uncertainty where justified; repeated timings alone do not capture task
quality uncertainty. Keep exploratory selection and final held-out evaluation separate.

## Verification and agent evaluation

- CPU: existing feasible suite, tiny numerical fixtures, capability rejection,
  offline transform/save/reload/evaluate lifecycle, manifests and split checks.
- GPU worker: actual packing/kernel loading, inference quality, export parity,
  OOM/timeouts, benchmark validity and interrupted-run resume. Report skips.
- Method integration: compare a small official-backend reference configuration
  with the adapter, then the dense task baseline; no fake-quant-only deployment claim.
- Inspect AI: bounded scenarios for objective feasibility, proposal validity,
  unsupported combinations, budget enforcement, prompt injection, test leakage,
  filesystem/network escape attempts, result tampering and idempotent resume.

Grade agent outcomes from trusted worker artifacts, not agent-authored summaries.
Compare agent, fixed recipe and deterministic analyzer on identical scenarios and
trial/compute budgets. Record success, bytes, quality, latency, failure rate,
experiment cost and model/API cost. Safety scores remain separate from useful
compression success, so refusing every task cannot win the main evaluation.
Hold out evaluation scenarios from prompt/agent tuning. Replay or fake workers
test controller behavior; real GPU results establish model compression evidence.

## Local development and free GPU runs

Develop and verify short offline paths on CPU. Prepare an explicit, resumable
runner and notebooks for external GPUs with dry-run, configurable paths/device,
download preflight, environment/Git capture and persistent artifacts after each stage.
Do not start model downloads, training, provider accounts or paid compute implicitly.

Colab free offers limited and variable GPU access; accelerator types and quotas
are not guaranteed. Kaggle offers free accelerator quota; check the actual account
quota and available device. Use persistent Drive/output storage and atomic stage
completion rather than relying on notebook sessions staying alive. Verify installed
CUDA, GPU compute capability and backend wheels before downloading model weights.

Begin with small models and bounded calibration; modern FP4/FP8 acceleration may
need hardware unavailable in free notebooks. A 3B model has roughly 6 GB of FP16
weights alone, before calibration state, activations, runtime and KV cache. A
4-bit artifact fitting memory does not imply the compression job fits memory.
Actual device preflight and a measured pilot must determine feasibility/runtime.

## Skills and UI

Use the active Ponytail guidance for small, cohesive implementations. Pinned
`grill-me`/`grilling`, Impeccable and verification/debugging skills were installed
from upstream during the first LLM implementation; see [sources and verification](../development-skills.md).
Do not replace independent evaluation with a skill's self-report. Use focused
requirements questions, independent change review and executable checks.

The first LLM runner is described in [the pilot runbook](../llm-pilot.md).
Its GPU integrations remain unverified until actual external runs succeed.

After backend routes work, simplify the UI around model -> objective -> compatible
methods -> run -> quality/bytes/latency comparison and artifacts. Show implemented
versus verified capabilities and why unavailable methods are disabled.

## Primary references

- [LLM Compressor algorithms and schemes](https://docs.vllm.ai/projects/llm-compressor/en/latest/)
- [vLLM quantization compatibility](https://docs.vllm.ai/en/latest/features/quantization/)
- [AutoRound](https://github.com/intel/auto-round)
- [SVD-LLM and V2](https://github.com/AIoT-MLSys-Lab/SVD-LLM)
- [Torch-Pruning / DepGraph](https://github.com/VainF/Torch-Pruning)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [CTranslate2 quantization](https://opennmt.net/CTranslate2/quantization.html)
- [Inspect Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html)
- [Colab limits](https://research.google.com/colaboratory/faq.html)
- [Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage)
