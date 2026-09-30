# Reproducible compression demo

The web interface runs at `http://localhost:7860` and stores analysis, bundles,
benchmarks and comparisons for each run. The quickest check is **Synthetic vision
demo**. For a more useful demonstration of the quality guardrail, train the
**Trained synthetic classifier** baseline first. Both are synthetic workflow
checks; neither establishes real-world model quality.

## Start the interface

```sh
docker compose up --build -d
```

Open `http://localhost:7860`, choose **Synthetic vision demo**, select
**Compress & verify**, and start the run. The application is bound to localhost
and runs one compression job at a time. Runs and model caches are stored in
Docker volumes; files under `./data` are available read-only at `/data` in the
container. The default image is CPU only. The language preset needs separately
prepared, pinned Hugging Face assets; the first demo needs no download.

To run the UI directly with an installed package instead:

```sh
python -m tn_compression.ui --runs-dir runs/ui --data-dir data
```

## Train and compress the synthetic baseline

With Python, TorchVision and the project's core dependencies installed:

```sh
tn-compress train --config examples/configs/demo-trained-synthetic.yaml \
  --output-dir data/demo-trained --device cpu
```

The command writes `data/demo-trained/bundle`. In the UI choose **Trained
synthetic classifier**; the checkpoint field is prefilled with
`demo-trained/bundle`. Choose **Compress & verify**. The saved bundle is checked
again on the test split and benchmarked against the original bundle. An
infeasible plan stops before producing a compressed artifact.

The preset asks to save a ResNet18 below 40 MiB of tensor payload with at most
a 0.02 absolute loss of top-1 accuracy. It screens a later convolutional layer
and the classifier, then validates cumulative quality. The displayed artifact
file size can differ from the tensor-size target.

## Measured example

The [raw comparison](demo-results.json) records a Docker CPU run on 2026-09-29
using Python 3.11.11, PyTorch 2.6.0, two CPU threads, 20 measured forwards after
three warmups, and 96 held-out synthetic test images. The container ran on
Docker Desktop's WSL2 Linux engine. The baseline was trained for three epochs
on a separate deterministic split. The accepted replacement was partial Tucker
on `layer4.1.conv2` with ranks `[256, 256]`.

| Measurement | Original | Compressed | Change |
| --- | ---: | ---: | ---: |
| Top-1 accuracy on synthetic test images | 100% | 100% | 0 points |
| Tensor payload | 44,750,764 bytes | 38,721,452 bytes | −13.47% |
| Serialized bundle | 44,792,433 bytes | 38,773,875 bytes | −13.44% |
| Mean CPU forward latency | 4.629 ms | 3.632 ms | −21.54% |
| Peak sampled inference RSS | 392,036,352 bytes | 419,545,088 bytes | +7.02% |

The small timing sample is a demonstration, not a stable speedup claim. Peak
sampled process memory increased in this run even though the artifact was
smaller. The synthetic task is deliberately easy and color-coded. Use a trained
model, a representative real dataset, multiple runs, and a deployment-specific
benchmark before making a useful compression claim. The [quality contract](quality.md)
explains how to set the allowed degradation in native metric units.
