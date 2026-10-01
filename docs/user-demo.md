# Use your own model and produce a defensible result

Start the local interface with `docker compose up --build -d` and open
`http://localhost:7860`. The CPU image keeps uploaded weights, runs, and Hugging
Face caches in Docker volumes. Put labeled images under `data/` in this project;
that directory is mounted read-only at `/data`.

## Vision weights and labels

The web upload accepts a `.pt` or `.pth` **state_dict**, optionally wrapped as
`{"state_dict": ...}` or `{"model_state_dict": ...}`. Select the exact
TorchVision or Segmentation Models PyTorch architecture and constructor
arguments that produced the weights. A class-count or shape mismatch fails
strict loading; the tool does not guess a model architecture from tensors.
The upload is converted into a portable, checksum-verified model bundle before
analysis or compression.

A file containing a serialized Python `nn.Module` cannot be loaded safely by
the web service. If it is a file you created and trust, convert it in the
original environment where its model class is installed:

```sh
python scripts/export_trusted_pt_state.py old_full_model.pt model_weights.pt --trusted
```

The conversion deliberately uses pickle deserialization and must never be run
on an untrusted file. Upload the resulting `model_weights.pt`.

For a real classification demonstration, arrange images as follows. Every split
must contain the same class directories. Keep the final test images separate
from all tuning decisions.

```text
data/my-images/
  train/class_a/*.jpg
  train/class_b/*.jpg
  calibration/class_a/*.jpg
  calibration/class_b/*.jpg
  validation/class_a/*.jpg
  validation/class_b/*.jpg
  test/class_a/*.jpg
  test/class_b/*.jpg
```

Set `data.kind` to `image_folder`, `data.root` to `/data/my-images`,
`task` and `model.task` to `classification`, and set both `num_classes` and
`model.kwargs.num_classes` to the directory count. This adapter resizes images
to the selected square size and applies the standard ImageNet normalization by
default. Supply an adapter if the trained model used different preprocessing;
using the wrong preprocessing invalidates the quality comparison.

## What the controls measure

**Inspect layer importance** uses calibration examples to plot each eligible layer's
mean absolute weight-times-gradient score, with gradient RMS as a companion
number. Larger bars show stronger local loss sensitivity on those examples;
they do not prove that a layer must be kept dense. See [the metric and source
papers](relevance.md).

**Tensor decomposition** reshapes a compatible Conv2d weight into an
output-channel matrix, or uses a Linear weight directly, then keeps the smallest
SVD rank whose squared singular values retain the chosen energy. Energy is a
weight-reconstruction threshold for each layer, not a guaranteed model-accuracy
percentage. Non-beneficial ranks and protected modules are skipped. The run
reports actual selected ranks and saved bytes.

**Quantization** stores eligible weights as per-output-channel int8 values and
scales in the portable bundle. The PyTorch forward path dequantizes these
weights, so it may be slower or use more transient memory; the benchmark reports
what actually happened. The earlier round-to-nearest 4-bit reference is a
floating-point diagnostic and does not count as storage compression.

**Pruning** physically removes intermediate channels only in verified
Llama/Qwen gated MLPs. Arbitrary vision pruning is unavailable because safe
channel removal requires adapting connected layers. A mixed recipe applies
pruning first, then SVD factorization, then int8 weight storage. Unsupported
model/method pairs fail explicitly rather than silently changing semantics.

The Docker UI saves the original and compressed bundles, evaluates both on the
same held-out test examples, and benchmarks both under the same declared
workload. The comparison includes quality loss, serialized bundle bytes, tensor
bytes, latency and sampled memory. A failed quality or size limit stays a failed
result. Optional vision fine-tuning uses the training split and reports a fresh
test-and-benchmark comparison for the recovered bundle.

For a clean demonstration, start with one fixed trained baseline, analyze on
calibration images, select a recipe using validation data, and run the final
held-out test once. Show a small table with baseline, compressed, and optionally
recovered quality, serialized bytes, latency, and hardware details. Report any
regression. The [existing measured example](demo.md) is a workflow check on
synthetic images, not proof of real-world accuracy.

## Language assets and recovery

The language input accepts a local compatible Hugging Face causal LM or a
repository ID pinned to a full commit revision; the tokenizer and any Hugging
Face dataset are pinned separately. The default Docker image starts offline. If
the pinned files are absent, keep **Download missing Hugging Face files** selected
for the first run. They remain in the persistent cache for later offline runs. Custom
remote Python model code is disabled.

The Qwen preset is a fixed study rather than an open-ended test-set loop. It
records a dense calibration measurement, evaluates the dense validation
baseline, tries packed int8 and two selective SVD settings independently on
validation, and selects the lowest finite validation NLL. The test split is
evaluated only after selection, once for the dense bundle and once for the
selected reloaded bundle. See the [runbook and measured result](qwen-demo.md).

Optional language recovery trains a LoRA adapter on a separate text training
split. The adapter is saved alongside metadata containing the compressed base
bundle checksum; the adapter alone is not a complete model. The recovered
language quality is checked on held-out text. Adapter inference cost is not yet
benchmarked in this workflow.

TFRecords are a **dataset serialization format**, not a vision fine-tuning
method. The current vision adapter reads class folders or the supported
Oxford-IIIT Pet dataset; vision recovery uses PyTorch fine-tuning. A TFRecord
input adapter could be added if the training data is only available in that
format, with image decoding and label mapping specified explicitly.
