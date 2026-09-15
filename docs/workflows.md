# Runnable workflows

Use Python 3.11. Install matching PyTorch/TorchVision wheels for the target machine,
then `pip install -r requirements/vision.txt -e .`. The clean tested CPU pair is
PyTorch 2.6.0 / TorchVision 0.21.0. Optional language dependencies use a separate
requirements file. Dataset and model caches stay outside Git.

## Offline segmentation lifecycle

Run from the repository root; each command writes a separate directory:

```sh
tn-compress train --config examples/configs/segmentation-synthetic.yaml --output-dir runs/demo/train
tn-compress compress --config examples/configs/segmentation-synthetic.yaml --checkpoint runs/demo/train/bundle --output-dir runs/demo/compress
tn-compress finetune --config examples/configs/segmentation-synthetic.yaml --checkpoint runs/demo/compress/bundle --output-dir runs/demo/recover
tn-compress evaluate --config examples/configs/segmentation-synthetic.yaml --checkpoint runs/demo/recover/bundle --output-dir runs/demo/evaluate
tn-compress export --config examples/configs/segmentation-synthetic.yaml --checkpoint runs/demo/recover/bundle --output-dir runs/demo/export
tn-compress benchmark --config examples/configs/segmentation-synthetic.yaml --checkpoint runs/demo/recover/bundle --output-dir runs/demo/benchmark
```

These synthetic images verify integration and mechanics, not real-world quality.
Use the classification synthetic configuration for the analogous ResNet18 workflow.
The Pet configuration downloads the public dataset and requires substantive
training. For recovery, copy the configuration, set epochs to 10 and learning rate
to 0.0001, and use the compressed checkpoint explicitly. The 70/10/20 partition
of the official training data separates training, calibration and validation;
the official test data is untouched until final evaluation. Binary mode maps
pet/background to 1/0 and ignores border pixels (255); multiclass maps 1/2/3 to
0/1/2. Masks use nearest-neighbor resizing.

## Model selection and reconstruction

`model.source` supports TorchVision, SMP, Transformers, PyTorch Hub and custom
factories. Change registered model names/constructor arguments in YAML without
editing a decomposition. Pin named TorchVision weights rather than DEFAULT.
Remote Hub and Hugging Face inputs require immutable commit revisions. Hub
wrappers need an explicit adapter. A successful load does not imply method or
backend compatibility; inspection reports coverage and skip reasons.

In Python, pass a custom factory to `load_bundle(path, factory=...)`; the factory
constructs the original architecture without downloading weights. Bundles store
replacement constructors, factor shapes, state tensors, model recipe, training
flags, trainability and a weight checksum. Reload uses strict state loading and
never reruns decomposition. A bundle directory must be empty before saving.

Classification with `task: classification` and `num_classes` adapts a recognized
Linear classifier; other head layouts need `classifier_path`. Detection keeps
TorchVision's image-list and prediction-dictionary contracts. COCO evaluation
uses pycocotools AP/AP50 and records subset IDs. Backbone-only compression is
configured by explicit layer paths. Detector training and ONNX export are deferred.

## Measurement limits

The benchmark starts a separate Python process for each bundle. It records load
time separately, sampled process RSS, actual parameter count, precision, threads,
input shape and complete-model forward latency. GPU memory uses PyTorch allocator
counters; these are not total device usage. RSS sampling can miss short peaks.
Model timing is not data-loader or network serving timing. ONNX outputs are checked
against the corresponding compressed PyTorch model before export is marked verified.

The current implementation supports fixed-shape tensor-output ONNX exports.
Unavailable CUDA, failed runs and timeouts are explicit outcomes. There are no
TensorRT or vLLM performance claims in the vision smoke results.
