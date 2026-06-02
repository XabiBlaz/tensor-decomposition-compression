# Compression Lab

Runnable workflows for model inspection, tensor-decomposition compression, representation checks, and a small SAE post-compression demo.

Install from the repository root:

```bash
conda env create -f environment.yml
conda activate tn-compression
```

All commands support `--device cpu`, `--device auto`, `--device cuda`, and `--device cuda:0`.

Public model choices:

- `torchvision_resnet18`
- `torchvision_vgg16`
- `artifact`

Torchvision models support `--weights none` and `--weights imagenet`. The default is `none` to avoid automatic downloads.

Pre-compression inspection writes CKA-based groups to `segments.json`. The grouping threshold is selected with Fisher-Jenks natural breaks over pairwise CKA scores unless you pass a manual `--cka-threshold`.

## Configs And Plans

Edit YAML configs in:

```text
compression/configs/default_all.yaml
compression/configs/individual_layers.yaml
compression/configs/grouped_layers.yaml
```

Then run:

```bash
python compression/compress_model.py \
  --model torchvision_resnet18 \
  --weights none \
  --config compression/configs/individual_layers.yaml \
  --device auto
```

The `--config` flag is required when you want the YAML file to control compression. If `--config` is omitted, the compression command uses only CLI flags such as `--method`, `--rank`, `--rank-method`, `--energy`, and `--rank-cap`.

For automatic energy-based rank selection, `rank_cap` is only a ceiling. Use `rank_cap: null` in YAML when you want the energy criterion to choose ranks without that cap.

`compression_plan.json` is generated output. It records eligible layers, selected policies, and skip reasons. It is not the file you normally edit.

## Inspect

```bash
python compression/analyze_compression_plan.py \
  --model torchvision_resnet18 \
  --weights none \
  --device auto
```

For artifacts, pass the full serialized module and input shape:

```bash
python compression/analyze_compression_plan.py \
  --model artifact \
  --artifact-path ./model.pt \
  --input-shape 1,3,224,224 \
  --device auto
```

## Compress

```bash
python compression/compress_model.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --device auto
```

## Representation Analysis

```bash
python compression/representation_analysis.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --device auto
```

CKA and activation drift are diagnostics, not task-accuracy guarantees.

## SAE Demo

```bash
python compression/sae_postcompression.py \
  --model torchvision_resnet18 \
  --weights none \
  --method tensor_train \
  --rank-method SVD \
  --energy 0.94 \
  --steps 100 \
  --device auto
```

The SAE is trained on compressed-model activations. It is separate from model fine-tuning and does not repair the compressed model.
