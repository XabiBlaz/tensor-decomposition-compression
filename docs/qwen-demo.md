# Qwen2.5-0.5B UI demo

This is the reproducible language-model demonstration. It uses
`Qwen/Qwen2.5-0.5B` at commit
`060db6499f32faf8b98477b0a26969ef7d8b9987` and
`Salesforce/wikitext` at commit
`b08601e04326c79dfdd32d625aee71d232d685c3`. Remote model code is disabled.

## Runbook

The CPU UI remains available with `docker compose up --build -d`. For the GPU
path used below:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec ui python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print((torch.ones(1, device='cuda') + 1).item())"
```

This machine has the legacy 460.106.00 driver, so the overlay sets
`NVIDIA_DISABLE_REQUIRE=1`. CUDA 11 minor-version compatibility was accepted
only after the displayed PyTorch device allocation and arithmetic check passed.
On another host, use a current NVIDIA driver and remove that compatibility
override if it is unnecessary; device discovery alone is not a sufficient test.

Open `http://localhost:7860`, choose **Run the Qwen2.5 language study**, keep
**Compress a model** selected, choose **GPU**, and leave **Download missing
Hugging Face files** checked only on the first run. Click **Run fixed study**.
After the first successful asset download, uncheck that option to enforce cache-
only operation. The cache and run history live in the Docker volumes declared
by `compose.yaml`.

Allow roughly 15–30 minutes on an RTX 2080 Ti after assets are cached (the
recorded cache-only run took 15 minutes 49 seconds). The UI runs one job at a
time. Progress and actionable failures appear in the run log.
When complete, the validation trial table, final held-out comparison, generation
text, and limitations appear on the result page. **Run artifacts** provides the
comparison JSON/Markdown, per-example IDs and valid-token counts, benchmark
reports, generation/cache verification, layer inventory, and the complete
`selected-pytorch-bundle.tar`. When fixed-logits ONNX parity passes, it also
provides `language-fixed-logits-onnx.tar`.

The fixed protocol uses 32 seeded hash-selected examples for each disjoint role:
WikiText train for calibration, validation for candidate selection, and test for
the final report. Texts shorter than 80 characters are excluded and each example
is truncated to 128 tokens. The run artifact records every source row ID and its
valid shifted-target count; the fixed list is also checked in as
[example provenance](qwen-demo-example-provenance.json). Packed int8 and SVD
use weight-only transforms, so
the calibration role is recorded as a disjoint provenance/baseline measurement;
it is not mislabeled as activation calibration.

## Measured GPU result

The checked-in [machine-readable summary](qwen-demo-results.json) records the
complete settings. The run used one GeForce RTX 2080 Ti (11,019 MiB), NVIDIA
driver 460.106.00, PyTorch 2.6.0+cu118, float32 weights, three warmups, and ten
measured batch-one greedy generations. Every request used the same recorded
64-token WikiText validation prompt and generated 32 tokens with the attention
cache; EOS was ignored for timing.

Packed int8 won validation. The initial 95% SVD pilot was an honest no-op failure
because its factorization would not reduce tensor bytes; the maintained preset
uses 92% and 90% settings on only `model.layers.23.mlp.down_proj`. Those trials
saved 1,327,616 and 2,134,016 tensor bytes but raised validation NLL from 3.24785
to 3.47746 and 3.49104, respectively, so neither was selected.

| Measurement | Dense | Selected int8 | Change |
| --- | ---: | ---: | ---: |
| Held-out test NLL (3,421 valid tokens) | 3.05195 | 3.06728 | +0.01534 |
| Held-out perplexity | 21.1565 | 21.4835 | +0.3270 |
| Serialized PyTorch bundle | 2,520,795,515 B | 1,448,664,154 B | −42.53% |
| Unique tensor bytes | 1,976,131,200 B | 903,868,032 B | −54.26% |
| Mean 32-token request latency | 761.23 ms | 1,028.40 ms | +35.10% |
| Output throughput | 42.04 tok/s | 31.12 tok/s | −25.98% |
| Peak CUDA allocated | 2,027,881,984 B | 989,454,848 B | −51.21% |

This result passes the +0.1 NLL guardrail but fails as a speed optimization. The
portable int8 modules dequantize weights in the PyTorch forward path, explaining
the smaller artifact and GPU allocation alongside worse latency. Sampled host
inference RSS also rose from 2,607,751,168 to 2,833,252,352 bytes. No recovery
was used.

Generation from the reloaded selected bundle produced 24 tokens and returned an
attention cache. A cached next-token step matched a full-prefix step with maximum
logit error `1.69e-05` at `rtol=1e-3`, `atol=1e-4`.

## ONNX and segmentation boundaries

The PyTorch bundle is the primary artifact. The time-boxed language experiment
verified a 905,206,965-byte, fixed `[1, 128]` logits-only graph with ONNX Runtime
CPU. Its maximum absolute logit error was `2.046e-4` at `rtol=1e-3`,
`atol=3e-4`, with no provider fallback. The UI publishes that graph in a tar
only after parity passes, including any external tensor files.

That graph has no past-key-value interface, dynamic sequence contract, or
verified autoregressive generation, so it is explicitly **not serving-ready**
and is not benchmarked against cached PyTorch generation. ONNX Runtime CUDA and
TensorRT results are not claimed without a verified cached-generation graph.

The segmentation smoke workflow remains supported and automatically exports a
PyTorch-parity-checked ONNX file for downloadable supported models. It is only a
mechanics check. To extend it to real segmentation quality, provide the trained
checkpoint, dataset format and paths, exact preprocessing, class count and mask
value mapping (including ignored labels), and train/calibration/validation/test
split definitions. No TFRecord conversion or custom dataset adapter is assumed.
