"""Render raw candidate results without pooling ranks into a correlation claim."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.results.glob("*-seed*.json")):
        result = json.loads(path.read_text())
        for candidate in result["candidates"]:
            if candidate["method"] != "unchanged":
                rows.append({"method": candidate["method"], "seed": result["calibration_seed"],
                             **{key: candidate[key] for key in ("path", "rank", "bytes_saved", "delta_nll", "relative_squared_error")}})
    if not rows:
        raise ValueError("No candidate result files found.")
    with (args.output / "candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, color in (("svd", "#345995"), ("weighted_svd", "#bd562c")):
        selected = [row for row in rows if row["method"] == method]
        for index, xkey in enumerate(("bytes_saved", "relative_squared_error")):
            scale = 2**20 if xkey == "bytes_saved" else 1
            axes[index].scatter([row[xkey] / scale for row in selected], [row["delta_nll"] for row in selected],
                                label=method, color=color, alpha=0.7, s=45)
            axes[index].set_ylabel("Validation NLL increase")
            axes[index].grid(alpha=0.2)
    axes[0].set_xlabel("Tensor payload saved (MiB)")
    axes[1].set_xlabel("Sampled relative output error")
    axes[0].legend(frameon=False)
    figure.suptitle("Qwen pilot: three calibration seeds, fixed validation examples")
    figure.savefig(args.output / "candidate-pilot.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
