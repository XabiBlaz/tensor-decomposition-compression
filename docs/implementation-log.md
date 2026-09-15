# Implementation and experiment log

## 2026-09-15 — Milestone 1

Starting commit: `732e612`; local work branch: `work/compression-toolkit`.
Baseline verified on the existing Python 3.10 / torch 1.13 environment:
15 tests passed in 16.96 seconds. No network access or pretrained weights were
needed for these tests. Source inventory is in `source-inventory.json`.

Reconciled CP, partial Tucker and TT execution. New tests cover numerical
reconstruction, convolution geometry and padding, coefficient scaling, dtype,
frozen weights, true TT axis ordering and gradients, zero-rank selection and
PLAIN_TT updates. Initial combined run: 33 passed in 17.01 seconds.

A clean Python 3.11 / PyTorch 2.6 CPU installation passed 41 tests in 24.92 seconds;
`pip check` found no broken requirements. Matplotlib emitted dependency deprecation
warnings, with no test failures. DLF compatibility and shared-parameter protection
are included. The reference dependencies are in `requirements/core.txt`.

This machine has
RTX 2080 Ti GPUs and driver 460.106.00. GPU 0 is occupied and is not used. Modern
serving-runtime compatibility requires separate verification; CPU results do not
establish GPU or serving performance.

No pushes have been made. Later milestones and research results remain pending.

## Milestone 2 — lifecycle and vision adapters

Python 3.11 CPU suite: **56 passed** in 31.21 seconds. Full synthetic SMP and
ResNet18 workflows completed training, compression, recovery, reload, evaluation,
ONNX parity and subprocess benchmarks. Compact raw validation results are in
`validation/milestone2.json`; they are explicitly synthetic, not public-data
accuracy evidence. A Faster R-CNN/ResNet50-FPN fixture passed variable-image-size,
compression and exact checkpoint round-trip tests with detection output contracts.

A separate existing Python 3.10/PyTorch 1.13 CUDA environment successfully executed
SMP on GPU 1, selected by UUID. A public Pet baseline is being attempted there;
its training and any later public-data results are not claimed complete here.

Bundles reconstruct shapes without repeating decomposition. Shared task functions
provide correct mask mapping, weighted metrics and frozen-normalization recovery.
Remaining evidence for this milestone: public Pet/classification quality and a
COCO subset AP comparison. The command paths and offline verification are usable.

## Milestone 3 — causal language evaluation and candidate trials

Implemented as separate commits for evaluation/data roles, bounded calibration,
and CLI candidate trials. Python 3.11 / PyTorch 2.6 CPU suite: **66 passed** in
35.07 seconds. Qwen's three rank candidates were measured against the same 817
validation tokens; all substantially damaged quality for less than 1% whole-model
tensor-byte savings. Exact IDs and raw measurements: `validation/milestone3.json`.
See `language.md` for reproduction and limitations. The eight-example pilot is
complete; downstream tasks, cross-domain evaluation and wider sampling remain
research milestones. The Pet baseline has now finished 30 epochs; compressed
public-data comparisons remain pending.
