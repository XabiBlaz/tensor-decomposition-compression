# Oxford-IIIT Pet compression pilot

This is a runbook, not a result. No real-data metric is checked in until the
GPU study has run. The pilot trains one SMP U-Net/ResNet-18 from **seeded random
initialization** (`weights: null`), then compares a dense model, the existing
validated damage-aware analyzer, and three direct weight-SVD-energy settings.
The chosen architecture and initialization do not fetch mutable pretrained
weights. The runner saves the resolved config, Git SHA, package versions,
checkpoint, per-role image IDs, trial failures, validation selection and final
test comparisons. See [the template](../examples/configs/pet-pilot.yaml).

The public [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
has image trimaps. Our mapping is `1 pet → 0`, `2 background → 1`,
`3 border → 2`; all three classes contribute to mean IoU. Images resize to
160×160 with bilinear interpolation, masks with nearest-neighbor interpolation,
and inputs use ImageNet mean/std. TorchVision's official `trainval` IDs are
SHA256-sorted with fixed data seed 42 into 70% train, 10% calibration and 20%
validation; its official `test` split stays separate. The bounded first run
uses 1,536/16/128/256 images respectively. `splits/split-ids.json` records
the exact chosen IDs. Model seed 7 affects initialization and training order,
not the data partition.

## What the pilot measures

The analyzer considers partial-Tucker ratios 0.5 and 0.75 on four named
ResNet-18 convolutions; the simple comparator uses SVD weight energies 0.55,
0.70 and 0.85 on those same layers. These are deliberately different methods:
**equal ranks do not mean equal compression**. The runner compares achieved
tensor bytes, selects the closest valid simple baseline to the analyzer, and
reports whether their achieved sizes are within 5%. If they are not, it makes
no analyzer-value claim. Infeasible plans, no-op/failed trials, worse latency,
and quality failures remain visible in `selection.json` and stage logs.

Candidate eligibility uses validation mean IoU, a maximum 0.05 absolute loss
from dense, and the 43 MiB tensor-byte target. Final test evaluation begins
only after selection is frozen. Every selected candidate and the dense model
are benchmarked with the same 160×160 batch-one input. The final report gives
test mean IoU, actual tensor and serialized-bundle bytes, latency and peak
resident/CUDA allocation through the existing comparison contract. Times and
memory are workload-specific. The current physical pruning capability is for
Llama/Qwen gated MLPs, **not arbitrary vision layers**.

Optional `--with-recovery` runs two additional epochs for each selected
compressed model **and** a dense control, all with the same train/validation
IDs, optimizer policy, learning rate, and epoch count. It writes separate
recovery-versus-equal-budget-control comparisons. It is off for the first
pilot because it adds compute; use a new run ID if enabling it later.

## Short local checks

From the repository root in an environment with its test dependencies:

```sh
python -m pytest -q tests
bash -n scripts/run_pet_pilot.sh
bash scripts/run_pet_pilot.sh --data-root data/pets --output-dir runs/pet-pilot \
  --run-id dry-check --device cpu --dry-run
python -m pytest -q tests/test_vision_tasks.py::test_offline_segmentation_lifecycle_and_frozen_buffers
```

The dry run does not need a dataset or create files. The focused test exercises
an offline train/compress/bundle-reload/evaluate lifecycle on tiny synthetic
data; it is not Pet accuracy evidence. Do not run full Pet training on this
CPU-only PC.

## Colab or Kaggle setup

Use a GPU runtime, verify `torch.cuda.is_available()` and `nvidia-smi`, and
clone the **reviewed and approved** repository revision with a clean worktree. These steps are for
you to run later; this PC has not downloaded the dataset or run the study.
Install the project and the vision extra in the GPU runtime, then verify that
the existing Torch/CUDA installation still sees the GPU:

```sh
git clone https://github.com/XabiBlaz/tensor-decomposition-compression.git
cd tensor-decomposition-compression
git checkout main  # Only after the reviewed commits have been approved and merged.
python -m pip install -e '.[vision]'
python -c 'import torch, torchvision, segmentation_models_pytorch; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
nvidia-smi
```

TorchVision's [OxfordIIITPet loader](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html)
downloads under `DATA_ROOT/oxford-iiit-pet`. Download **both** official splits
once on the GPU host, using a writable local dataset root:

```sh
export DATA_ROOT=/content/pets  # Colab; use /kaggle/working/pets on Kaggle
python - <<'PY'
import os
from torchvision.datasets import OxfordIIITPet
for split in ('trainval', 'test'):
    OxfordIIITPet(os.environ['DATA_ROOT'], split=split,
                  target_types='segmentation', download=True)
PY
```

For Colab, mount Drive in a notebook cell with
`from google.colab import drive; drive.mount('/content/drive')`, then set
`OUT=/content/drive/MyDrive/pet-pilot`. For Kaggle, set
`OUT=/kaggle/working/pet-pilot`, enable GPU and Internet for initial setup,
and save a notebook version with output or use its file-persistence setting
before the session ends. Kaggle's `/kaggle/working` output can be saved for
reuse; Colab GPU availability and session life vary. Keep the same repository
revision, dataset root and output directory when resuming. [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[Kaggle Notebooks guide](https://www.kaggle.com/docs/notebooks).

## Run, inspect, resume

In the repository root, after the data step:

```sh
export OUT=/content/drive/MyDrive/pet-pilot  # or /kaggle/working/pet-pilot
bash scripts/run_pet_pilot.sh --data-root "$DATA_ROOT" --output-dir "$OUT" \
  --run-id oxford-pilot-v1 --device cuda --seeds 7
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['OUT']) / 'oxford-pilot-v1' / 'seed-7'
for name in ('selection.json', 'final.json'):
    report = json.loads((root / name).read_text())
    print(name, json.dumps(report, indent=2)[:5000])
PY
```

An interrupted run is resumed by repeating the **same** command and run ID.
Completed stages are skipped only when their configuration, command and Git
revision match; a changed template or revision requires a new run ID. Inspect
`logs/*.log`, `.stages/*.failed`, `selection.json`, and `final.json` before
claiming any benefit. A completed run can still show infeasibility, a quality
failure, or slower inference.

After inspecting the first result, repeat the command with `--seeds 7,17,27`
and the same run ID. Seed 7 is reused; seeds 17 and 27 get separate directories
and retain the same data split. Use a **new** run ID to change the config or
enable `--with-recovery`. If comparing seeds, report all three runs, not only
the best one.

The 45.6 MB dense tensor payload was measured locally for this architecture;
the configured eight epochs mean 1,536 training steps at batch size eight.
The validation/intervention work adds several hundred forward passes, so
roughly **30–120 minutes on a single T4-class GPU** is a planning range, not a
measurement. Batch-eight 160×160 activations dominate the 45.6 MB weights;
plan for **at least 8 GB GPU memory**, preferably a 12–16 GB runtime, and
reduce batch size with a **new** run ID if an allocation fails. The runner's
benchmark records actual peak CUDA allocation and latency for later reporting.
