# Language workflow

Install the CPU reference environment with `requirements/language.txt` and
`pip install -e .`. The public model selection uses the same Transformers loader
as custom local configurations; Qwen is a benchmark fixture, not an allowlist.

## Model and dataset assets

The model, tokenizer and Hugging Face dataset revisions must be full 40-character
commit IDs. A local model/tokenizer directory is also accepted. Loading uses
`trust_remote_code=False`, so models that require custom repository Python code
are outside the current supported set. A model may load successfully yet have
unsupported layers for a selected compression method; review the plan and final
evaluation before treating it as a usable result.

The Docker image defaults to `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.
Thus the Qwen example fails on a fresh cache at the initial model load. Select
the UI's explicit download option for the first run, with Internet access and
enough cache space, or place a complete pinned model/tokenizer and dataset in
the mounted cache. Subsequent runs can use the cache offline. The language
preflight checks for the config, tokenizer, all weight shards and the requested
data splits before saving the original bundle. It reports missing assets rather
than substituting untrained weights. A local `text_json` dataset avoids the
dataset download; it still needs disjoint calibration, validation and test
records, and a separate `train` partition for recovery.

For another supported causal LM, replace `model.name` and `model.revision` in
the workflow. Set `tokenizer.name` and `tokenizer.revision` if the tokenizer is
not stored with the model. Choose a model that fits the available RAM/GPU memory;
the Qwen example alone requires roughly two GB of float32 parameters before
temporary analysis copies, activation memory and optimizer state.

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

[The raw Qwen pilot result](qwen-pilot-results.json) records the unmodified Qwen checkpoint and three
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

## Optional LoRA recovery

The existing `train`/`finetune` command can update selected or all base model
parameters from the explicit `train` text partition. The optional PEFT route
trains only LoRA adapter weights with a token and optimizer-update budget. It
needs a saved base bundle and produces an `adapter/` directory plus
`lora_recovery.json`. The adapter is bound to the saved base's SHA256 weight
digest. To reload it, use `load_lora_adapter(adapter_dir, base_bundle_path=...)`;
the adapter alone does not reconstruct a decomposed, pruned or otherwise
compressed base model. Always measure post-recovery NLL on held-out test data
and compare it with both the original and the compressed model. Recovery on
training text cannot prove that lost knowledge or general performance returned.

Provide a `train` role alongside calibration, validation and test, then add a
bounded recovery section to the workflow:

```yaml
lora:
  token_budget: 1024
  max_updates: 32
  rank: 8
  alpha: 16
  learning_rate: 0.0001
  target_modules: all-linear
```

```sh
tn-compress recover-lora --config YOUR_WORKFLOW.yaml --checkpoint YOUR_COMPRESSED_BUNDLE --output-dir runs/recovered
tn-compress evaluate-lora --config YOUR_WORKFLOW.yaml --checkpoint YOUR_COMPRESSED_BUNDLE --adapter runs/recovered/adapter --role test --output-dir runs/recovered-test
```

These commands require the language extra with PEFT installed. Use the same
base bundle for both commands.
