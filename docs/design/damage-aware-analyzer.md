# Damage-aware compression analyzer

The analyzer answers a transformation-specific question: given a loaded model,
a task, representative calibration data, a quality constraint and an execution
backend, which concrete transformations should be applied? It does not assign a
single generic importance score to a layer.

## Candidate records

A candidate is one `(layer, method, resolved configuration)` proposal. Rank 128
and rank 256 SVD replacements of the same projection are separate candidates,
as are Tucker and CP replacements of the same convolution. Candidate IDs hash
the model fingerprint, layer path, method and proposed configuration. Measurement
results do not change the ID.

Each serialized record separates:

- structural method eligibility and protected-module reasons;
- checkpoint reconstruction capability;
- requested inference-backend capability;
- estimated tensor bytes and measured replacement tensor bytes;
- normalized local reconstruction evidence;
- full-model metrics and metric deltas;
- the final decision and its reason.

The model fingerprint includes model identity metadata, state names, shapes,
dtypes and tensor values. Hashing is streamed in bounded chunks. This detects a
plan being applied to different weights, at the cost of one full pass over model
state at analyzer startup.

## Evidence levels

### Structural

Structural analysis does not run the model or require data. It discovers leaf
`Linear` and supported `Conv2d` modules, protected or unsupported modules, and
verified gated-MLP groups. It generates bounded grids for SVD, weighted SVD,
TT/TTPWT, partial Tucker, CP3/CP4, gated-MLP pruning and the dense
round-to-nearest quantization reference.

Candidates that increase parameter count are retained as rejected evidence.
Round-to-nearest quantization remains a floating-point module, so it is reported
with zero real byte saving and cannot satisfy a storage target. The analyzer
does not infer packed INT4 storage from its nominal bit width.

### Calibrated

Calibrated analysis executes bounded representative data. Linear and MLP inputs
use deterministic priority sampling, including attention-mask filtering for
language models. Convolutions use a bounded number of feature-map crops. This
avoids retaining all feature maps or materializing every unfolded image patch.

For original module output `Y` and candidate output `Y_c`, the reported score is:

```text
||Y - Y_c||² / (||Y||² + epsilon)
```

The denominator makes a score meaningful within a layer's candidate curve.
Scores are never compared as interchangeable units across unrelated layers.
The record retains the unnormalized squared error, mean squared error, output
element count, sample shape, seed, identifiers and preprocessing metadata.

Calibration restores hooks, module modes, buffers and random-number-generator
state. The supplied model is not replaced or mutated.

### Validated

Only the within-layer Pareto frontier reaches full-model validation. A candidate
is dominated when another proposal for the same layer saves at least as many
bytes and has no larger normalized local error, with one strict improvement.
`max_validated_per_layer` bounds expensive interventions further.

Task adapters measure:

- classification: cross-entropy and top-1 accuracy;
- segmentation: loss, mean/per-class IoU and Dice;
- detection: COCO AP/AP50 when a COCO API is available;
- causal language modeling: token-weighted NLL, perplexity and streamed
  teacher-to-candidate KL.

Detection loss is not currently available from the inference evaluator and is
recorded as a capability limitation rather than fabricated. Language KL streams
one batch at a time so full-vocabulary teacher distributions are not retained
for the entire validation set.

The initial allocator sorts validated candidates by measured quality damage per
estimated byte saved. It applies one candidate at a time, evaluates cumulative
task damage, and accepts it only when the configured quality constraint remains
satisfied. Overlapping transformations are rejected. Every trial restores the
original module, modes, buffers and RNG state before an accepted replacement is
installed. After planning, all accepted replacements are removed again, leaving
the caller's model unchanged.

This greedy policy is deterministic and reviewable. It is not a global optimizer.
Its plan records whether the requested byte target was feasible.

## API and CLI

Python callers use `tn_compression.analyzer.analyze_model`. The CLI invokes the
same service:

```bash
tn-compress analyze \
  --config examples/configs/analyzer-smoke.yaml \
  --output-dir runs/analyzer-smoke \
  --level validated \
  --target-size-mb 40 \
  --max-quality-loss 0.05 \
  --backend pytorch
```

The output directory contains:

- `analysis.json`, with all candidates and evidence;
- `analysis-summary.txt`, with accepted, rejected, protected and unsupported rows;
- `analysis-report.html`, a standalone table requiring no server;
- `compression_plan.json`, containing accepted transformations in application order.

Apply the resolved plan with:

```bash
tn-compress compress \
  --config examples/configs/analyzer-smoke.yaml \
  --plan runs/analyzer-smoke/compression_plan.json \
  --output-dir runs/analyzer-smoke-compressed \
  --device cpu
```

The model fingerprint must match. Weighted SVD recollects the configured bounded
calibration inputs when applying the plan. Pruning plans store resolved retained
indices. Checkpoint bundles store replacement structures and transformation
metadata.

## Relation to established methods

The analyzer's weighted SVD minimizes a ridge-regularized reconstruction
objective using an uncentered input second moment. It is related to the
activation-aware motivation in ASVD and the whitening motivation in SVD-LLM,
but it is not an exact reproduction of either paper's complete algorithm.

The activation-based gated-MLP score combines intermediate activation RMS with
the corresponding down-projection column norm. Wanda similarly combines weights
and activations for sparse-weight selection, but the implementation here removes
coordinated dense MLP channels. It is not Wanda.

SparseGPT adds approximate second-order weight reconstruction and is not
implemented by this analyzer. OWL motivates non-uniform layer-wise allocation
from activation outliers; this analyzer instead uses candidate-specific local
curves followed by measured full-model interventions. These methods belong in
later empirical comparisons. No state-of-the-art claim is made.

## Running experiments

The runner defaults to offline mode and will fail instead of downloading missing
assets:

```bash
bash scripts/run_analyzer_experiments.sh --dry-run --suite all
bash scripts/run_analyzer_experiments.sh \
  --suite smoke --device cpu --output-dir runs/analyzer
```

Use a stable `--run-id` to resume. Each successful stage has a `.done` marker;
stdout and stderr are preserved under `logs/`. The run records the Git commit,
configuration, Python packages, OS, CPU and available NVIDIA information.
Existing output is never silently replaced. A changed configuration cannot
resume an earlier run ID.

Prepare Oxford-IIIT Pet explicitly before the vision suite, then edit the
absolute `data.root` in `analyzer-vision.yaml`:

```bash
python - <<'PY'
from torchvision.datasets import OxfordIIITPet
OxfordIIITPet('/absolute/data/root', split='trainval', target_types=['segmentation', 'category'], download=True)
OxfordIIITPet('/absolute/data/root', split='test', target_types=['segmentation', 'category'], download=True)
PY
```

Prepare the pinned language model, tokenizer and dataset in a separate networked
step. Confirm the revisions in `analyzer-language.yaml`, place them in the cache,
then pass the cache locations without enabling network access:

```bash
bash scripts/run_analyzer_experiments.sh \
  --suite language --device cuda:0 \
  --model-cache /absolute/hf-cache \
  --data-cache /absolute/hf-datasets-cache
```

The smoke suite uses random model weights and synthetic data. It verifies the
workflow only and cannot support model-quality conclusions.

## Limitations

- Artifact estimates count tensor payloads, not container headers.
- Reference quantization is dense and provides no deployment saving.
- Factorized representations have no verified vLLM loader.
- ONNX candidates still require artifact parity checks after plan application.
- Detection validation reports AP/AP50 but no inference-mode detector loss.
- Candidate interaction is measured greedily; combinations not visited by the
  search remain unknown.
- Calibration and validation evidence is task- and data-dependent. It does not
  guarantee test-set, cross-domain or post-recovery behavior.
