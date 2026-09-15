# Language workflow

Install the CPU reference environment with `requirements/language.txt` and
`pip install -e .`. The public model selection uses the same Transformers loader
as custom local configurations; Qwen is a benchmark fixture, not an allowlist.

```sh
tn-compress evaluate --config examples/configs/qwen-pilot.yaml --output-dir runs/qwen-dense --role validation
tn-compress calibrate --config examples/configs/qwen-pilot.yaml --output-dir runs/qwen-calibration
tn-compress plan --config examples/configs/qwen-pilot.yaml --output-dir runs/qwen-candidates
tn-compress compress --config examples/configs/qwen-pilot.yaml --output-dir runs/qwen-compressed
tn-compress evaluate --config examples/configs/qwen-pilot.yaml --checkpoint runs/qwen-compressed/bundle --output-dir runs/qwen-reloaded
```

The example deliberately includes damaging ranks. Review `plan.json` before
using them: the first Qwen up projection is very sensitive in this pilot.
Compression is explicit; `plan` performs temporary interventions and restores
the model. It does not choose a final jointly validated allocation yet.

Qwen2.5-0.5B was selected for its standard gated MLP, tied embeddings, small CPU
footprint and Apache-2.0 model license. Model and tokenizer revision:
`060db6499f32faf8b98477b0a26969ef7d8b9987`. WikiText revision:
`b08601e04326c79dfdd32d625aee71d232d685c3`. Calibration uses train, allocation uses
validation, and test is reserved for reporting. Each role uses eight seeded
hash-selected text examples, truncated to 128 tokens. All IDs are recorded.
The loader rejects ID and normalized-text overlap between roles; use explicit
`text_json` records for custom partitions. A separate-domain slice is pending.

NLL is summed over valid shifted targets and divided by the number of tokens.
Padding and its following boundary are excluded. Calibration visits all batches
with bounded, seeded sampling of real-token inputs, one layer at a time. Sample
files are diagnostic artifacts, not a reusable cache; every run recomputes them.
The local quadratic identity uses an **uncentered input second moment**, not the
full-model loss Hessian. Optional teacher KL compares teacher to candidate at
declared temperature with CPU chunks, while model forward logits still have the
normal per-batch vocabulary allocation.

## CPU pilot

`validation/milestone3.json` records the unmodified Qwen checkpoint and three
isolated SVD trials on `model.layers.0.mlp.up_proj` in float32, PyTorch 2.6.0 CPU,
Transformers 4.51.3, two CPU threads. Baseline validation: 817 tokens, NLL 3.19436,
perplexity 24.3946. Rank 256 saved 11,534,336 tensor bytes (about 0.58% of the
model) but increased NLL by 2.23053. Ranks 128 and 64 were worse. These are useful
failed settings, not recommended compression levels. Search took 56.1 seconds,
excluding model/data loading. Tensor bytes exclude checkpoint headers and are
not latency or peak-memory measurements.

The earlier dense test pilot (1,008 tokens) gave NLL 2.98980, perplexity 19.8817;
it uses a different split and must not be compared directly with candidate
validation scores. Tiny offline Qwen/Llama tests cover reload, generation/cache,
token masks, split separation and intervention restoration. No downstream-task,
GPU-speed or serving-quality claims follow from this pilot.
