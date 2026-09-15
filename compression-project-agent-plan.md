# Implementation handoff: reproducible compression, LLM inference and damage-aware allocation

## 1. Objective and execution instruction

Extend `XabiBlaz/tensor-decomposition-compression`, preserving its history and the public Python namespace `tn_compression`. Build a reliable compression toolkit and a reproducible empirical study that demonstrate numerical understanding, ML systems engineering, efficient LLM inference and research judgment.

Central question: **Which compression choices preserve model quality while delivering measured memory and inference benefits, and how reliably can inexpensive diagnostics predict their damage?**

Proceed with this plan. Begin with source inventory, numerical correctness and model reconstruction; implement the milestones below as cohesive, reviewable commits. Resolve routine implementation decisions from the actual code and installed environments. Do not stop at another planning-only response. Report completed changes, evidence, limitations and the next milestone. Preserve the explicit per-push approval rule in Section 13.

This document combines and revises the earlier vision-first plan. The earlier agent's reported numerical defects and passing test counts are findings to verify in the current checkout, not freshly established facts. Do not claim unavailable source repositories or uncommitted changes were inspected or migrated.

Success does not require a new state-of-the-art algorithm. It requires correct implementations or integrations, fair comparisons, reliable deployment and a defensible empirical finding. Attribute established methods and distinguish reproductions, adaptations and original experiments.

## 2. Scope and ordering

Finish one complete segmentation workflow and add small classification/detection adapters around the same compression core, then bring LLM evaluation and vLLM forward. The full segmentation study remains a later milestone; it must not delay the first LLM result. Keep the existing classification examples and generic PyTorch API working. Classification and object detection are included as bounded demonstrations of reuse, not additional full research studies.

Initial benchmark fixtures and validation targets (not a model allowlist):

- Vision: one SMP segmentation architecture with binary/multiclass handling, recovery and ONNX export; one ResNet18 classifier; and one TorchVision Faster R-CNN/ResNet50-FPN detector. Reuse supported convolutional decomposition across all three, adding only task-specific loading/input/evaluation adapters. For the classifier and detector, initially demonstrate one validated method and compression setting against a dense baseline, with reload and PyTorch measurements.
- Language: one small pretrained dense decoder-only architecture, initially around 0.5–1.5B parameters if the available hardware supports it; tiny randomly initialized fixtures serve correctness tests only.
- Compression: existing tensor methods, ordinary SVD, activation-aware low-rank compression, one deployable weight-quantization recipe and activation-aware structured MLP pruning.
- Research: candidate-specific damage estimation and compression-budget allocation, with direct intervention measurements as the reference.

For the initial benchmark, select and pin the exact LLM model revision after checking architecture, license, available memory and the chosen vLLM version. Record the selection rationale. Users remain free to select other models through the generic loaders and capability checks below. Do not choose a large model merely for presentation. CPU work and offline fixtures should remain useful when a GPU is unavailable; do not substitute those results for GPU performance claims.

Defer the `vision-platform` monorepo, SGLang, TensorRT expansion, specialized compression adapters for unsupported transformer internals, attention-head/residual-width pruning, custom GPU kernels, broad SAE work and agent-safety work until the core results exist. Generic model selection is in scope from the start. Defer heterogeneous mixtures of methods within one model until individual methods and serialization are verified.

## 3. Source consolidation and correctness

Selectively reconcile the original sources, leaving them untouched:

| Source | Reuse or reconcile |
| --- | --- |
| Current repository | Public API, planning, demos, diagnostics, reporting and CI |
| `tn_compression_module` | DLF contract, complementary utilities and API regression tests |
| `925013_qiavs` | Decomposition fixes, segmentation workflows, fine-tuning and evaluation |
| `segmentation_models_pytorch_Lortek` | Thin project adapter, losses, construction and dataset handling |
| `arodrigo-RAM_Usage` | Isolated process measurements, ONNX and optional TensorRT tooling |

Keep one authoritative implementation per decomposition method. Preserve meaningful execution variants with accurate names and documented correspondence. Preserve the distinction between CKA-based grouping and legacy Fisher segmentation. Restore `build_dlf_compression_config` compatibility. Use upstream Segmentation Models PyTorch with a thin local adapter.

First verify and fix the reported bias-free Tucker and float64 failures. Cover device, dtype, bias, rectangular kernels, stride, padding, dilation and factor orientation. Do not replace factorized execution by silent dense reconstruction. Explicitly distinguish matrix SVD from true tensor-train decomposition.

Primary convolutional experiments compress supported `groups=1` convolutions; preserve grouped/depthwise layers and protected stems/heads, recording their size contribution. Keep experimental methods visibly marked until validated. Implement the classification/detection adapters below; defer broader architecture-specific wrappers, detection training/export expansion and TFRecord migration.

## 4. Architecture and model lifecycle

Separate reusable algorithms from workflows and command presentation. Introduce small interfaces only as needed by the supported vision tasks and language workflow:

| Component | Contract |
| --- | --- |
| Model loader | Construct a model through SMP, TorchVision, PyTorch Hub, Hugging Face Transformers or a supplied module; record source, version, constructor and weights separately from task behavior |
| Model adapter | Identify eligible modules and dependency groups; validate shapes; reconstruct/export the changed architecture |
| Compression method | Describe legal candidates; estimate cost/error; apply the transformation; serialize its resolved representation |
| Sensitivity evaluator | Measure local reconstruction error and full-model effects for a specified candidate intervention |
| Budget planner | Select supported candidates under a specified resource objective and quality constraint |
| Execution backend | Load a supported artifact; run evaluation or serving; report capabilities and failures |
| Task evaluator | Own preprocessing, data splits, loss, metrics and output interpretation |

Publish a model × method × backend × hardware validation matrix, alongside method capability rules. The matrix records tested combinations; it must not block other models solely because their names are absent. Planning inputs include task, calibration data, workload, precision, backend and resource/quality constraints.

### Open model selection and capability checks

The product must let the user choose a model independently of the benchmark fixtures. Build configurable loaders rather than separate command implementations for each benchmark model:

- TorchVision: select any model registered in the installed version, with its constructor arguments, explicit weights and task contract.
- SMP: select architecture, encoder, weights, input channels, classes and other valid constructor options. Use the library's supported architecture/encoder combinations. [SMP model configuration](https://smp.readthedocs.io/en/latest/models.html).
- PyTorch Hub/custom PyTorch: select a pinned Hub entrypoint or supply an `nn.Module`, plus the input/task adapter and reconstruction factory where needed.
- LLMs: accept a Hugging Face model ID or local checkpoint through `AutoModelForCausalLM` and `AutoTokenizer`, with pinned revisions and loading options. This covers models recognized by the selected Transformers version; custom implementations need their own explicit loading path. [Transformers Auto classes](https://huggingface.co/docs/transformers/model_doc/auto).

Compression methods declare capability predicates over module types, geometry, precision, parent usage and relevant dependencies. Discover matching modules in the selected model. Prefer shared task-contract adapters and module-family handlers; introduce model-specific logic only where the actual architecture requires it. Do not use model-name membership in the benchmark list as a capability check.

The initial broadly reusable path is shape-preserving convolutional factorization and linear low-rank replacement: the internal representation becomes smaller while external input/output dimensions stay unchanged. Start with supported ordinary `nn.Conv2d` and `nn.Linear` behavior. Parent code that accesses `.weight` directly, functional/fused projections, shared or tied parameters, grouped convolutions and custom operators may require additional handlers. Unsupported modules remain unchanged and are reported; do not apply a replacement solely because a tensor is two- or four-dimensional.

An inspection/plan report must state eligible and protected modules, skip reasons, fraction of model parameters covered, proposed changes, achieved/estimated size, and backend compatibility. If the selection cannot meet the requested target, report that explicitly. Zero modified layers is a no-op, not successful compression. Distinguish compatibility inferred from module contracts from combinations verified by execution and checkpoint round-trip. For newly selected models, run a representative forward/output-contract check and reload check before marking the artifact usable; these checks do not establish accuracy or speedup.

Keep three separate capabilities: **load the original model**, **apply this compression method**, and **execute the compressed artifact on this backend**. PyTorch/Transformers is the general research path. Quantization is constrained by the chosen implementation/kernels. Structured pruning requires a verified dependency handler for the affected module family. vLLM additionally requires support for the model and changed representation. Missing one optional method/backend must not prevent inspection or use of another compatible method/backend.

For LLMs, enable generic linear-layer inspection and supported shape-preserving low-rank transformations across compatible causal language models. Preserve attention masks, positional inputs, cache/generation behavior and tied parameters. Initially verify structural MLP pruning on one supported gated-MLP family; models with different MLP layouts, MoE experts, fused projections or unconventional attention need additional handlers. Do not silently apply that pruning recipe to every model accepted by `AutoModelForCausalLM`.

Calibration and evaluation still require representative data and a task contract. Loading an arbitrary model does not supply its labels, preprocessing, quality target or workload. A model with a nonstandard output contract should request an adapter with a precise explanation, rather than being rejected because it was not benchmarked.

Preserve existing Python entry points and script commands. Extend `tn-compress` with `inspect`, `calibrate`, `plan`, `compress`, `train`, `finetune`, `evaluate`, `export` and `benchmark`. Commands should call shared Python functionality. Planning must not mutate the original model.

Checkpoint bundles must contain weights, model construction/configuration, resolved factors/ranks, pruning indices, quantization metadata, protected modules, preprocessing/class mappings or tokenizer revision, calibration identifiers, ordered transformations and schema version. Reload without redoing decomposition or selection. Preserve existing full-module loading as a documented compatibility path.

Support full-model, decoder/head-only and explicit-layer fine-tuning. Construct optimizers after module replacement and parameter freezing. Make training/recovery policies task-specific. Record which LLM parameters/adapters are updated; do not assume vision policies transfer unchanged.

### Vision loading and bounded task adapters

PyTorch Hub is a plausible optional loading route, not the mechanism that makes compression generic. The reusable unit is an eligible module inside a supported `torch.nn.Module`; each task retains its input, output and evaluation contract. Hub can return arbitrary entrypoint results, so validate the returned object and require an adapter for wrapper objects. [PyTorch Hub](https://docs.pytorch.org/docs/stable/hub.html).

Use `torchvision.models.get_model` with explicit weight versions as the default for TorchVision models. Expose `source=torch_hub` as a thin alternative using a pinned repository tag/ref or versioned local checkout and recorded resolved commit. Preserve constructor arguments, weight identity/checksum, preprocessing and class mappings in checkpoint metadata. Reconstruct the original architecture without downloading pretrained weights before applying saved compression structure and state. Keep direct-module input supported, requiring an explicit construction factory for portable reload.

TorchVision supplies model/weight lookup and weight-associated preprocessing. Its detection models still require an installed compatible TorchVision build for custom operators even when loaded through Hub. Pin matching PyTorch/TorchVision builds and avoid mixing installed and Hub-imported versions within one process. [TorchVision model loading](https://docs.pytorch.org/vision/stable/models.html).

| Task | Minimal adapter | Initial benchmark compression boundary |
| --- | --- | --- |
| Classification | Batched images, class labels, logits, cross-entropy and top-1 accuracy; correct weight transforms and label mapping | Supported internal ResNet18 convolutions; preserve stem and classifier output dimensions |
| Segmentation | Existing SMP model/data/loss/metric adapter | Existing validated convolutional boundaries |
| Detection | Variable-sized image lists, box/label targets, prediction dictionaries and standard box AP evaluation | Eligible convolutions in the ResNet backbone body; initially preserve FPN, RPN, RoI heads, detector transforms and postprocessing |

TorchVision Faster R-CNN takes lists of image tensors and returns boxes, labels and scores in evaluation mode; training returns a loss dictionary. Preserve this interface and its internal resizing/normalization. Do not apply classification normalization a second time. [Faster R-CNN contract](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).

Share layer traversal, calibration hooks, decomposition, planning, reconstruction and benchmark reporting. Replacements must preserve external shapes and account for parent modules that inspect child attributes; test complete model execution. Detection needs its own collate function, coordinate/category handling and evaluator, but no new detector implementation. Reuse maintained evaluation utilities with attribution.

This extension does not promise generic channel pruning or quantization of every Hub model. Shape-preserving decomposition is the initial shared path. Structural pruning across residual/FPN connections requires dependency handling; vision quantization requires separately validated backend support. Detect unsupported modules and report compressed versus protected parameter coverage. Keep architecture-specific policy out of the decomposition implementation.

## 5. Default decision method: measure candidate damage, then allocate

Do not use CKA similarity or raw gradient norms as the default allocation policy. Neither directly measures the effect of a particular rank, bitwidth or removed channel group. Keep them as optional descriptive diagnostics or historical baselines.

Use a staged process: **cheap local screening → actual candidate intervention in the full model → constrained selection → cumulative re-evaluation**. This is the project's proposed engineering policy, not an established algorithm name or a claim of guaranteed optimality.

### 5.1 Separate the data roles

Maintain disjoint examples for calibration/reconstruction, allocation validation and final testing. Recovery training uses training data. A validation set repeatedly used to choose compression is part of model selection, not an untouched evaluation set. Record dataset versions, identifiers, tokenization, sequence lengths, masks and seeds. For LLMs, calibrate on representative natural text and include a distinct-domain evaluation slice.

Start with a small, recorded calibration pilot; increase its size only to resolve unstable rankings or insufficient coverage. Stream statistics, sample vision patches/tokens and process one layer/block at a time. Do not retain every full activation or full-vocabulary teacher distribution in GPU memory.

### 5.2 Local output reconstruction: first screening score

For a linear layer with input columns `X`, unchanged bias and candidate weight `W_c`, compute:

`E_local = ||(W - W_c) X||_F^2 / (||W X||_F^2 + epsilon)`.

The unnormalized mean error is `tr(DeltaW C DeltaW^T)`, with `C = XX^T / N` and `DeltaW = W - W_c`. This directly weights weight errors by observed inputs. `C` is an uncentered input second moment; the Hessian of this local quadratic reconstruction objective is proportional to it. **It is not the Hessian of the full language-model or segmentation loss.**

Use the actual module/block outputs when biases, nonlinearities, quantized activations or structural changes invalidate the simple linear expression. For convolutions, evaluate sampled real feature maps or appropriately sampled unfolded input patches. For gated MLP pruning, score the effect at the MLP output as well as individual projections.

Prefer direct sampled residual evaluation initially. Full second-moment matrices consume quadratic memory; use streamed/blockwise statistics or declared approximations when necessary. Normalize and aggregate consistently, but do not assume normalized errors are equally consequential across layers. Retain unnormalized measurements too.

### 5.3 Direct candidate interventions: reference damage measurement

Construct actual rank-reduced, quantized or pruned candidates for shortlisted layers/dependency groups. Temporarily insert each candidate and measure the full model on a fixed allocation-validation subset, then restore the model exactly.

- LLM primary selection signal: change in token-weighted negative log-likelihood. Report perplexity as an evaluation metric. Use correctly shifted targets and exclude padding.
- LLM auxiliary signal: mean token-level `KL(p_original || p_candidate)` at the same teacher-forced prefixes and temperature. Compute in chunks. KL measures fidelity to the original model, not truth or task success.
- Segmentation: change in task loss plus mean/per-class IoU and Dice; watch small/rare classes rather than relying only on a background-dominated average.
- Classification: change in cross-entropy and top-1 accuracy on mapped labels; optional output KL at identical inputs.
- Detection: screen using backbone feature reconstruction, then measure box AP/AP50 on a fixed annotated validation subset. Do not compare variable-length post-NMS predictions using naive elementwise KL. Native detector losses are optional auxiliary diagnostics only, with sampling/RNG, training flags and normalization buffers controlled and restored; calling `train()` just to obtain losses can change the evaluated computation.

Measure the proposed transformation itself. Whole-layer removal is not a reliable stand-in for modest rank truncation or INT4 quantization. For pruning, test the actual selected groups. Zero-ablation measures that particular removal intervention; mean replacement or activation patching, if investigated, are separate interventions.

This measures damage on the selected sample without recovery. It does not guarantee unseen-domain performance, post-recovery ranking or the behavior of several interventions together.

### 5.4 Allocate a resource budget using candidate curves

For each eligible layer/group, generate a small candidate set that includes leaving it unchanged. Record method, rank/width/precision, measured local error, validation damage, artifact bytes, estimated arithmetic and backend validity. For initial serving-compatible MLP pruning, candidate width is uniform across layers; per-layer channel identities may differ. Variable-width allocation is a later research-backend extension with explicit reconstruction support.

First compare fixed methods independently. For low-rank compression, allocate ranks; for pruning, allocate supported widths/groups; for quantization, assess supported precision choices or exclusions. Do not assume arbitrary per-layer mixed precision is executable by the selected quantization backend. Reject unsupported combinations before spending evaluation time.

Filter dominated candidates. Implement a simple budgeted greedy policy using marginal measured damage per positive byte saving, with a uniform-allocation baseline and explicit tie-breaking. Accept changes in small batches, measure cumulative damage, refresh affected activations/rankings and revert or reduce a step that violates the quality constraint. Retain the best validated feasible checkpoint and report if the requested budget is infeasible.

Do not sum isolated quality losses and call that the true joint loss. Track both predicted and observed cumulative effects. Byte budgets are the initial objective because they are easier to establish. For latency objectives, profile candidate shapes on the intended backend and validate the entire resulting model: layer times and FLOP reductions do not reliably add up to end-to-end gains. Do not claim a global optimum from greedy selection.

## 6. Method implementations and baselines

| Family | Initial implementation/integration | Role |
| --- | --- | --- |
| Vision decomposition | Partial Tucker, verified TT/TTPWT and CP variants | Preserve and validate existing mathematical work |
| LLM low rank | Ordinary SVD, then a documented activation-aware approach based on SVD-LLM | Compare weight-only and data-aware reconstruction |
| LLM quantization | Simple round-to-nearest reference; one maintained GPTQ W4A16 recipe compatible with the chosen vLLM stack | Practical second-order-informed quantization and deployment baseline |
| Structured pruning | Gated-MLP channel removal, comparing random, weight-based and activation/output-contribution selection | Main dense-shape pruning study |
| Sparse weight pruning | Magnitude and Wanda reference; SparseGPT as a later comparison | Separate cheap selection from reconstruction/compensation |

GPTQ uses approximate second-order information for quantization. Integrate and pin a maintained implementation rather than writing a production quantization kernel. Check supported group size, exclusions, checkpoint format and GPU architecture. If the selected GPTQ path is unsupported, record the reason and choose a documented supported recipe; do not silently relabel another algorithm as GPTQ. [GPTQ](https://arxiv.org/abs/2210.17323), [vLLM quantization](https://docs.vllm.ai/en/latest/features/quantization/).

SVD-LLM uses data whitening and sequential updates to improve low-rank compression. Reproduce the chosen components accurately and state any omissions. Label an independent weighted-reconstruction implementation accordingly, rather than claiming a full paper reproduction. [SVD-LLM](https://arxiv.org/abs/2403.07378).

Wanda uses magnitudes and input activations to select weights. SparseGPT adds a reconstruction-based pruning approach and supports unstructured and certain semi-structured patterns. Neither should be presented as equivalent to physically removing MLP channels. Zeros in a dense tensor do not establish memory or speed savings. Hardware-supported 2:4 execution is an optional later experiment. [Wanda](https://arxiv.org/abs/2306.11695), [SparseGPT](https://arxiv.org/abs/2301.00774).

AWQ is a useful later quantization comparator using activation-informed channel scaling; it is not a universal layer-importance oracle. Add it only after the initial quantization result is reproducible. [AWQ](https://arxiv.org/abs/2306.00978).

### Structured MLP pruning details

For `h = SiLU(W_gate x) * (W_up x)` and `y = W_down h`, retaining intermediate channels `J` requires the same rows `J` of gate/up projections and columns `J` of the down projection, with corresponding bias handling. Preserve residual width, attention, embeddings and output head initially. Record retained indices and architecture configuration.

A cheap channel-ranking baseline is `RMS(h_j) * ||W_down[:, j]||_2`. Its square equals isolated single-channel mean squared MLP-output removal error when other components are fixed. It ignores cross-channel terms when several channels are removed and downstream task effects. Therefore evaluate actual groups and validate in the full model. This baseline is an analytical reference, not a claimed new method.

Optionally test group-gate Taylor/Fisher approximations once the forward-only path works. Introduce gates on the structures being removed; define whether the score uses signed mean change, mean absolute derivative or per-example squared derivatives. Do not describe a raw parameter gradient norm or a squared batch-averaged gradient as a faithful Hessian estimate. Finite, large removals still need direct validation.

Test equivalence between physical slicing and the corresponding masked model before recovery. Exported parameter counts must reflect the smaller architecture. Uniform retained widths must respect the selected runtime's constraints. Broader residual-width slicing such as SliceGPT is a separate advanced method, not another name for this MLP operation. [SliceGPT](https://arxiv.org/abs/2401.15024).

## 7. Experiments and research contribution

Main hypothesis: candidate reconstruction error predicts compression damage better than weight-only heuristics for some settings; adding full-model validation and iterative allocation improves the quality–resource trade-off over uniform allocation. Treat this as a hypothesis to test, including failures.

Evaluate both prediction and decisions:

- Correlation between predicted scores and observed candidate damage, reported within comparable methods, budgets and intervention sizes.
- Harmful-candidate detection and ranking stability across calibration samples/domains.
- Quality at matched achieved storage or another explicitly specified resource budget, comparing uniform and adaptive allocation.
- Cumulative predicted versus observed damage, before and after recalibration.
- Calibration/search/recovery time and memory, not just final inference cost.

A predictor can have useful correlation yet make poor budget decisions. Report actual selected-model outcomes. Include a limited calibration-size ablation and a cross-domain check. Account for candidate-search cost when comparing policies. Bootstrap independent examples/sequences rather than treating correlated tokens or pixels as independent observations.

### Segmentation protocol

Use Oxford-IIIT Pet and an offline synthetic quickstart. First complete U-Net/ResNet34 with one validated method, then add the remaining validated methods. Use a fixed validation split from the official training partition; keep the official test partition for final evaluation. Explicitly map all trimap labels and define the binary mapping separately. Use nearest-neighbor resizing for masks.

The expanded study retains U-Net/ResNet34, FPN/ResNet18 and U-Net/MobileNetV2; seeds 0, 1 and 2; Tucker/TT energy thresholds 0.70, 0.85 and 0.95; and valid CP ranks 8, 16 and 32. These are candidate settings, not equal compression budgets. Report achieved model size and add matched-budget comparisons where feasible.

Retain the earlier proposed starting configuration: ImageNet encoder initialization, 256×256 input, batch size 8, AdamW, multiclass cross-entropy, 30 baseline epochs and 10 recovery epochs. Verify convergence/cost in a pilot before freezing it for the study; use the appropriate binary output/loss contract for binary workflows. Select checkpoints by validation mean IoU. Include original, immediately compressed, recovered compressed and dense additional-training controls.

### Small classification and detection demonstrations

Classification: start with ResNet18 and reuse Oxford-IIIT Pet images, class labels and split identifiers. Adapt the ImageNet-initialized model to the 37-class task, establish a trained dense baseline, then compress and compare it using the shared evaluator/reporting. Do not evaluate an unchanged 1,000-class ImageNet head against Pet class indices. Reuse an existing correct classification workflow if available. One seed, one validated decomposition and one nontrivial compression setting are sufficient initially; use the same training/recovery machinery where practical.

Detection: start with COCO-pretrained `fasterrcnn_resnet50_fpn` and fixed, small annotated COCO subsets, with disjoint calibration, allocation-validation and final-evaluation image IDs. Preserve the pretrained category convention and map dataset IDs explicitly. Reuse a standard COCO box evaluator, fixing score/NMS settings, maximum detections and image transforms. Report AP@[0.50:0.95] and AP50 with subset size/IDs; subset scores are not full COCO benchmark results. The first result is dense versus backbone-compressed inference with no detector retraining. Tiny synthetic images/boxes are correctness fixtures, not accuracy evidence.

For both tasks, deliver load → inspect → calibrate/plan → compress → save/reload → evaluate → benchmark, one configuration and a compact result row. Defer detector recovery, detection ONNX/TensorRT export, extra detectors, and multi-seed/task-wide sweeps. Keep these examples small enough that they demonstrate core reuse without moving the first LLM milestone behind a second vision study.

### LLM protocol

Start with one pinned pretrained model and fixed natural-text splits; add a second scale or architecture only after the pipeline is validated. Choose and pin a small set of public downstream tasks covering more than perplexity. Record evaluation harness version, prompts, chat template when applicable, few-shot settings and exact samples. Report sample sizes and uncertainty.

Pilot several legal compression settings before committing to a larger sweep. Compare achieved bytes and quality, rather than asserting that a percentage of pruned parameters equals INT4 storage or a tensor rank. Quantized scales/metadata and unchanged layers count toward actual size.

First report post-training results without recovery. Then evaluate a selected subset with a fixed token/update recovery budget and a dense control under the same training policy. Record trainable parameters, peak memory and wall time; matched tokens do not imply equal compute. Do not require full-model LLM fine-tuning as a prerequisite for the first result.

For calibration-sampling experiments use seeds 0, 1 and 2 once the pilot is stable. Distinguish calibration variation, recovery-training variation and repeated timing measurements. Do not label a single deterministic pretrained checkpoint as three independently trained baselines.

An optional interpretability extension tests whether components that appear dispensable on one domain matter on another through ablation/restoration. Make claims about observed causal effects under the intervention, not automatic neuron semantics. A later safety slice may measure harmful-request compliance and benign over-refusal before/after compression; it cannot establish general model safety or replace a dedicated safety project.

## 8. Inference and deployment

Use Transformers/PyTorch as the initial LLM research backend. Introduce vLLM early for dense and supported quantized checkpoints. Pin version, model/tokenizer revision, quantization format, precision and environment. Benchmark a real serving workload, not just an API launch demo.

Integrate the pruned architecture in a separate milestone. First try an accurately updated standard configuration with supported uniform MLP width. Verify that the selected vLLM loader actually honors it. If needed, implement a scoped model/weight-loading adapter; arbitrary replacement PyTorch modules are not automatically deployable. Native vLLM layers include fused and parallel projections whose loading conventions must be respected. [vLLM model integration](https://docs.vllm.ai/en/latest/contributing/model/basic/).

Validate compressed Transformers versus the corresponding vLLM model on fixed teacher-forced outputs/log-probabilities where available, within documented precision tolerances, plus generation smoke tests. Require exact checkpoint reconstruction within one reference backend where appropriate; do not demand bit-identical free-running generations across different kernels.

Factorized low-rank execution in vLLM is an advanced integration, with its own acceptance tests and profiling. Dense reconstruction may be offered as an explicitly labeled compatibility export but cannot substantiate factorized inference savings. Per-layer variable widths and mixed methods remain research-backend-only until supported loaders are verified.

Required segmentation runtimes: PyTorch CPU/CUDA and ONNX Runtime CPU. Classification/detection demonstrations initially require PyTorch CPU and CUDA where available. Classification ONNX export may reuse the existing path after parity checks; detection export is deferred because its dynamic outputs and operators need separate validation. Validate every advertised export against its corresponding compressed PyTorch model. TensorRT stays optional. SGLang follows only if there is a specific engine-comparison question.

## 9. Benchmarking and reproducibility

Run configurations in isolated processes. Separate loading, warmup and steady state; synchronize CUDA timings appropriately. Record hardware, driver, runtime versions, precision, compilation settings, CPU threads, workload, warmup and repetitions. Record unavailable providers and failed/OOM runs as explicit outcomes.

Vision: task-specific quality (segmentation IoU/Dice, classification top-1/loss, detection box AP/AP50), parameters, on-disk artifact bytes, process RSS, CUDA allocated/reserved memory and latency distributions. Include batch size 1 and declared image dimensions. For detection, distinguish backbone timing from complete detector inference including internal transforms, proposals, RoI operations and NMS; do not present backbone acceleration as total detector acceleration. Record post-resize dimensions and cap settings. Diagnose CPU/GPU differences using profiling before claiming a cause.

LLM: held-out NLL/perplexity and downstream scores, artifact bytes, peak process/device memory, time to first token, inter-token latency/time per output token with exact definitions, throughput and p50/p95 request latency. Separate prefill-heavy and decode-heavy workloads, and sweep a small set of prompt lengths, output lengths and concurrency levels. Record actual generated lengths, request arrival policy, prefix caching, context limit, KV-cache precision/capacity and engine memory settings.

Compare dense and compressed models within the same backend to isolate compression effects. Present best-deployable configurations separately when backend settings differ. KV-cache allocation/reservation can dominate process memory; weight compression and MLP pruning do not inherently shrink attention KV tensors. Report weights and cache-related memory separately where measurable.

Make configurations and compact machine-readable raw results authoritative. Include model/data/code revisions, resolved compression plans, seeds, failures and reproduction commands. Plot quality–size–latency frontiers and prediction-versus-observed-damage results. Publish unsuccessful settings and slowdowns.

Use optional dependencies and separate pinned environments/containers for the core research workflow and vLLM serving. Keep Python 3.11 as the core reference if compatible; do not force all optional runtimes into one dependency lock. Containers must run the documented commands from a clean environment. Mount model/data caches; keep tests offline and small. Include source attribution and dependency/license information for redistributed code.

## 10. Verification

Run the current tests to establish the actual baseline. The earlier 15 repository and 10 source-package tests are reported counts, not a required frozen total. Add focused tests for meaningful failure modes:

- Full-rank bias-free Tucker reconstruction, known low-rank CP tensors and correct SVD/TT naming/execution.
- Device/dtype, convolution geometry, unsupported modules and non-mutating planning.
- Checkpoint round-trip equivalence with preserved ranks, indices, tokenizer/configuration and transformation order.
- Binary/multiclass masks, losses and optimizer steps changing only selected parameters.
- Classification preprocessing/class mapping and detection variable-size collation, empty targets/predictions, box coordinates/category IDs and preserved output contracts. Verify one classifier and one detector after convolution replacement and checkpoint reload. Test the optional Hub loader offline with local fixtures; do not download models in routine CI.
- Tiny offline vision and language lifecycles; original-model restoration after sensitivity trials.
- Generic selection using non-benchmark model names/local configurations and a custom-module fixture; verify that capability decisions follow module contracts rather than a benchmark-name allowlist, and that skipped modules/no-op plans are reported accurately.
- Structured masking-versus-slicing equivalence, dependency handling and real parameter reduction.
- Exact local reconstruction identity on tiny examples; explicitly distinguish approximations.
- Held-out split separation, token/pixel masks, weighted loss aggregation and cache invalidation after model/data/config changes.
- Planner support constraints, unchanged candidates, infeasible budgets, cumulative quality checks and rollback.
- Quantization save/load, supported-format checks, ONNX parity and compressed-reference/backend comparisons.
- Benchmark schema and explicit unavailable/failed/OOM results.

Use CPU CI for tiny offline checks, and separate GPU verification/research workflows. Tests validate behavior, not promised speedup. Do not invent metrics or treat CPU checks as evidence of GPU correctness.

## 11. Milestones and reviewable outputs

| Milestone | Reviewable completion result |
| --- | --- |
| 1. Foundation and correctness | Source/provenance inventory, clean install, reconciled algorithms, actual baseline and numerical regression results |
| 2. Lifecycle and vision adapters | Reliable bundles, DLF compatibility, one full segmentation lifecycle, and small classification/detection load → compress → reload → evaluate examples sharing the same core |
| 3. LLM foundation | Configurable causal-LM loader and shared task adapter, one benchmark model, fixed data splits, dense evaluation, streamed calibration and candidate evaluation |
| 4. Quantization and serving | Supported quantized artifact, dense/quantized vLLM benchmark and reproducible containers |
| 5. Structured pruning | Correct MLP surgery, group selection, round-trip tests and measured immediate quality/resource effects |
| 6. Low rank and allocation | SVD/data-aware comparison, candidate curves, uniform/adaptive policies, cumulative validation and calibration-cost report |
| 7. Research and structural serving | Prediction study, controlled recovery subset, cross-domain analysis and validated pruned-model vLLM path or documented scoped blocker |
| 8. Expanded study and presentation | Three-seed selected experiments, expanded segmentation study, raw results, report, limitations and recruiter-facing README |

Publish usable local deliverables at every milestone. Milestone 4 should already support a credible demonstration; do not wait for every later experiment before making the project reviewable. Each milestone may contain several cohesive commits. If a GPU or external source is unavailable, complete the remaining implementable work and report the specific blocked evidence without fabricating results.

## 12. Presentation and follow-on project

Lead the README with the problem, a short reproducible command, open model-selection options, method capability rules and verified results. Distinguish the broad configurable interface from the small tested benchmark matrix; an unlisted model is not automatically rejected, and successful loading alone is not evidence that every method/backend works. Show how to select another TorchVision/SMP model or Hugging Face checkpoint without editing compression algorithms, and how to supply an adapter for a genuinely different contract. Explain one successful and one unsuccessful compression configuration. Include architecture, method explanations, research protocol, profiling interpretation, provenance and limitations. Maintain a concise experiment log connecting decisions to results. Attribute paper implementations separately from original integration and experiments.

The later `vision-platform` may consolidate `deep-learning-framework`, `dlgs` and `lortek-vision-system` into separate training, inference and inspection packages/containers. Share preprocessing/class/output contracts and demonstrate recorded image → inference → inspection result → dashboard without industrial hardware. Install `tn_compression` as a dependency with one authoritative implementation. Do not begin this consolidation during the flagship milestones.

## 13. Git workflow and publication boundary

Use the existing destination `XabiBlaz/tensor-decomposition-compression`. Preserve original attribution, initial release and history; leave source-repository remotes and working changes untouched. Use the already verified GitHub connection in the implementation environment after checking the actual destination; do not assume credentials from another environment exist here.

Local implementation, testing and cohesive commits may proceed under this instruction. Before **every push**, present destination, branch, exact commits, change summary and validation results, then wait for explicit user consent. Consent covers only that reviewed push. Do not force-push or rewrite history. Complete the concrete reviewable work before requesting push approval.

Start now with Milestone 1, then proceed through the next unblocked milestones under this plan.
