# TN Compression

Post-training tensor-decomposition compression for PyTorch image models.

TN Compression is a standalone compression lab for already-trained PyTorch models. The intended workflow is:

1. Inspect a model before compression.
2. Decide whether to compress all eligible layers, specific layers, or layer groups.
3. Apply tensor-decomposition compression.
4. Inspect the compressed model with output drift, representation similarity, and a small SAE-based activation diagnostic.

The project does not train or fine-tune task models. Compression can reduce parameter count, but a real model still needs task-specific evaluation after compression.

## Repository Layout

```text
tn_compression/       # reusable compression library
compression/          # command-line workflows and reports
compression/configs/  # editable example compression configs
tests/                # fast regression tests
environment.yml       # small CPU/GPU-capable environment
```

## Installation

Create the environment:

```bash
conda env create -f environment.yml
conda activate tn-compression
```

All commands support:

```text
--device cpu      # force CPU
--device auto     # use CUDA if available, otherwise CPU
--device cuda     # require CUDA
--device cuda:0   # require a specific CUDA device
```

## Supported Models

The model choices:

- `torchvision_resnet18`
- `torchvision_vgg16`
- `artifact`

Torchvision models support:

```text
--weights none       # random initialization, no download; default
--weights imagenet   # ImageNet pretrained torchvision weights; may download once
```

When `--weights imagenet` is used, the standard 1000-class ImageNet head is kept. With `--weights none`, `--num-classes` can change the classifier output size.

In case you want to compress any model of your choice, use `--model artifact`. Artifact mode loads a full serialized `torch.nn.Module`, not a state dict:

```bash
python compression/compress_model.py \
  --model artifact \
  --artifact-path ./model.pt \
  --input-shape 1,3,224,224 \
  --method tensor_train \
  --device auto
```

The full module is required because compression needs to inspect and replace submodules.

Full-module artifacts also require the original Python module/class definitions to be importable at load time. If `torch.load` fails with a missing module error, install that dependency or add the original model definition to the Python path before using artifact mode.

## Start Here

Run these three commands first:

```bash
python compression/analyze_compression_plan.py \
  --model torchvision_resnet18 \
  --weights none \
  --device auto

python compression/compress_model.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --device auto

python compression/representation_analysis.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --device auto
```

Use `--config compression/configs/grouped_layers.yaml` after inspection when you want CKA/Fisher-Jenks groups from `segments.json` to control compression.

## Input Shape

`--input-shape` is the tensor shape passed to `model(x)`, including batch size:

```text
1,3,224,224
```

It is needed for commands that run a forward pass: latency, output drift, CKA, representation analysis, and SAE activation collection. Torchvision models default to `1,3,224,224`. Artifact mode tries common image shapes automatically, but pass `--input-shape` for non-standard models.

## Tensor Decompositions

Compression policies use `compression.default_method`, `compression.layers`, or `compression.groups` in YAML config.

- `tensor_train` / `tt`: supports `Conv2d` and `Linear`; recommended default.
- `partial_tucker`: supports `Conv2d`; channel-mode Tucker factorization.
- `cp3`: supports `Conv2d`; experimental CP/PDP-style convolution decomposition.
- `cp4`: supports `Conv2d`; alternative CP/PDP decomposition path.
- `cp2`: supports `Linear` only.

Unsupported or unsuitable layers are skipped and recorded in `compression_plan.json` and `report.md`. Grouped convolutions are currently skipped.

## Tensor Train Structure And Rank Selection

Tensor Train structure controls how a compressed Conv2d is represented:

- `TTPWT`: pointwise, vertical, horizontal, pointwise convolution sequence. This is the recommended deployment-oriented default.
- `PLAIN_TT`: original TT-style wrapper that stores TT cores and reconstructs the convolution weight.

When you pass `--method tensor_train`, the default Conv2d structure is `TTPWT`. Select it explicitly with `--structure TTPWT`, or use the direct TT wrapper with `--structure PLAIN_TT`. The `--structure` flag mainly affects Conv2d layers; Linear layers use the linear TT decomposition path.

Rank selection controls how many latent dimensions are kept:

- Fixed rank: `--rank 8` or `--rank 4,8,8`.
- `SVD`: keeps the smallest rank that reaches the requested spectral energy threshold.
- `ENTROPY`: follows singular-value entropy/concentration.
- `VBMF` / `EVBMF`: Bayesian matrix-factorization rank estimates where implemented.
- `rank_cap` / `--rank-cap`: optional maximum automatic rank. Use `null` in YAML, or `--rank-cap 0` on the CLI, when you want the energy/rank-selection method to decide without a hard ceiling.

Example:

```bash
python compression/compress_model.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --structure TTPWT \
  --rank-method SVD \
  --energy 0.94 \
  --rank-cap 8
```

## Configuration Vs Plan

You edit YAML configs. The library generates `compression_plan.json`.

```text
YAML config -> generate_compression_plan(...) -> compression_plan.json -> apply_compression_plan(...)
```

Important: YAML files do not affect compression unless you pass them with `--config`. Without `--config`, the command ignores the YAML files and builds an in-memory config from CLI flags such as `--method`, `--rank`, `--rank-method`, `--energy`, and `--rank-cap`.

```bash
python compression/compress_model.py \
  --model torchvision_resnet18 \
  --weights none \
  --config compression/configs/individual_layers.yaml \
  --device auto
```

Example configs:

```text
compression/configs/default_all.yaml
compression/configs/individual_layers.yaml
compression/configs/grouped_layers.yaml
```

`compression_plan.json` records every inspected leaf layer, eligibility, selected policy, skip reasons, and estimated compression ratio when available.

## 1. Inspect Before Compressing

This does not compress the model:

```bash
python compression/analyze_compression_plan.py \
  --model torchvision_resnet18 \
  --weights none \
  --device auto
```

Use ImageNet weights when you want inspection on pretrained representations:

```bash
python compression/analyze_compression_plan.py \
  --model torchvision_resnet18 \
  --weights imagenet \
  --device auto
```

For an artifact:

```bash
python compression/analyze_compression_plan.py \
  --model artifact \
  --artifact-path ./model.pt \
  --input-shape 1,3,224,224 \
  --device auto
```

Outputs in `compression/outputs/`:

- `precompression_analysis.json`
- `precompression_report.md`
- `params_per_layer.png`
- `cka_heatmap.png` when activation probes run
- `activation_effective_rank.png` when activation probes run
- `cka_groups.png` when CKA grouping runs
- `segments.json` with CKA-based layer groups

How to read this:

- Large parameter layers usually matter most for compression ratio.
- Low activation effective rank may indicate redundancy.
- High CKA between nearby layers can suggest redundant representation geometry.

CKA grouping works by collecting activations from eligible probed layers, flattening each activation into a matrix, and computing pairwise linear CKA. By default, the pairwise CKA scores are segmented with Fisher-Jenks natural breaks, and the lower edge of the highest-similarity class becomes the grouping threshold. Layers connected by CKA scores above that threshold become connected components, which are written to `segments.json`.

This is better than a fixed threshold when you do not know the CKA scale beforehand, because the cutoff adapts to the trained model's observed CKA distribution. You can still force a manual cutoff with `--cka-threshold 0.90`. With synthetic inputs it is still a diagnostic; for stronger decisions, run the same inspection with representative data or a representative artifact input pipeline.

## 2. Configure Compression

Global policy: compress every eligible layer with one method.

```yaml
compression:
  mode: default_all
  default_method:
    type: tensor_train
    structure: TTPWT
    method: SVD
    energy: 0.94
    rank: null
    rank_cap: null
```

Individual policy: explicitly target layer names from the inspection report.

```yaml
compression:
  mode: individual
  layers:
    conv1:
      type: tensor_train
      structure: TTPWT
      method: SVD
      energy: 0.98
      rank: null
      rank_cap: null
    layer1/0/conv1:
      type: tensor_train
      structure: TTPWT
      rank: 4
    fc:
      type: cp2
      rank: 8
```

Grouped policy: use CKA-based `segments.json` from inspection and assign policies by group.

```yaml
compression:
  mode: cka_groups
  analysis_dir: compression/outputs
  groups:
    group_0:
      type: tensor_train
      structure: TTPWT
      method: SVD
      energy: 0.90
      rank: null
      rank_cap: null
    group_1:
      type: tensor_train
      structure: TTPWT
      method: SVD
      energy: 0.94
      rank: null
      rank_cap: null
    group_2: skip
```

Layer names can use `/` or `.` separators. For example, `layer1/0/conv1` and `layer1.0.conv1` refer to the same submodule.

## 3. Compress

Global CLI run:

```bash
python compression/compress_model.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --device auto
```

VGG-16:

```bash
python compression/compress_model.py \
  --model torchvision_vgg16 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --device auto
```

Config-driven run:

```bash
python compression/compress_model.py \
  --model torchvision_resnet18 \
  --weights none \
  --config compression/configs/individual_layers.yaml \
  --device auto
```

Dry-run plan:

```bash
python compression/compress_model.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank 8 \
  --dry-run-plan \
  --device auto
```

Save compressed state dict:

```bash
python compression/compress_model.py \
  --model artifact \
  --artifact-path ./model.pt \
  --input-shape 1,3,224,224 \
  --config compression/configs/default_all.yaml \
  --save-model \
  --device auto
```

Outputs:

- `summary.json`
- `report.md`
- `compression_plan.json`
- optional `compression_summary.png`
- optional compressed state dict with `--save-model`

## 4. Post-Compression Representation Analysis

This compares original and compressed activations. It is not fine-tuning and not proof of task accuracy.

```bash
python compression/representation_analysis.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --device auto
```

Interpretation:

- CKA near `1.0`: similar activation geometry on probed inputs.
- CKA near `0.0`: representation geometry changed strongly.
- Lower relative L2 activation drift is better.
- Effective-rank changes show whether activations became more or less concentrated.

## 5. SAE Post-Compression Demo

The SAE demo trains a tiny sparse autoencoder on activations collected from the compressed model. It is a diagnostic bridge to sparse-feature workflows, not a model repair method.

```bash
python compression/sae_postcompression.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --steps 100 \
  --l1-coef 1e-3 \
  --device auto
```

Metrics:

- Reconstruction error: how well the SAE reconstructs compressed activations.
- Explained variance: fraction of activation variance captured by the SAE reconstruction.
- Mean L0: average active latents per activation vector.
- Dead latent fraction: unused latent capacity.
- Latent activation sparsity: fraction of inactive latent entries.
- Original-vs-compressed CKA: activation geometry change before SAE training.

This demo does not claim monosemanticity or semantic feature discovery.

## Python API

```python
from tn_compression.api import compress_model, generate_compression_plan
from tn_compression.config import load_config

config = load_config("compression/configs/default_all.yaml")
plan = generate_compression_plan(model, config, device="auto")
result = compress_model(model, config, inplace=False, device="auto", return_model=True)
compressed_model = result.model
```

Use `generate_compression_plan` to inspect eligibility and skip reasons before mutating a model. Use `compress_model` when you want the compressed model and configured artifacts.

## Verify

Fast CPU checks:

```bash
python -m compileall tn_compression compression tests
python compression/analyze_compression_plan.py --model torchvision_resnet18 --weights none --input-shape 1,3,64,64 --max-layers 6
python compression/compress_model.py --model torchvision_resnet18 --weights none --num-classes 10 --input-shape 1,3,64,64 --method tensor_train --rank 2 --latency-runs 1 --warmup-runs 0
python compression/compress_model.py --model torchvision_resnet18 --weights none --num-classes 10 --input-shape 1,3,64,64 --config compression/configs/individual_layers.yaml --latency-runs 1 --warmup-runs 0
python compression/representation_analysis.py --model torchvision_resnet18 --weights none --num-classes 10 --input-shape 2,3,64,64 --method tensor_train --rank 2 --batch-size 2 --batches 1 --max-layers 3
python compression/sae_postcompression.py --model torchvision_resnet18 --weights none --num-classes 10 --input-shape 2,3,64,64 --rank 2 --steps 20 --batch-size 2 --batches 1 --max-layers 3
python -m pytest -q
```
