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
