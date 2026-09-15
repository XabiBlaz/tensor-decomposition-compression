"""Run bounded candidate comparisons; hold validation fixed across calibration seeds."""

import argparse
import json
from pathlib import Path
import platform
import subprocess

import torch

from tn_compression.checkpoints import load_bundle
from tn_compression.config import load_config
from tn_compression.language_experiments import evaluate_candidates
from tn_compression.models import load_model
from tn_compression.tasks.language import load_text_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=["svd", "weighted_svd"], default=["svd", "weighted_svd"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {}
    if args.checkpoint:
        model, manifest = load_bundle(args.checkpoint, device=args.device)
    else:
        model = load_model(config["model"]).to(args.device)
    model.eval()
    validation, validation_ids = load_text_split(config, "validation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        calibration, calibration_ids = load_text_split({**config, "seed": seed}, "calibration")
        for method in args.methods:
            result = evaluate_candidates(model, calibration, validation, config["candidates"], method=method,
                        device=args.device, seed=seed, max_rows=config.get("calibration", {}).get("max_rows", 256))
            result.update(workflow=config, calibration_seed=seed, validation_seed=config.get("seed", 0),
                          example_ids={"calibration": calibration_ids, "validation": validation_ids},
                          checkpoint_weights_sha256=manifest.get("weights_sha256"),
                          environment={"python": platform.python_version(), "torch": str(torch.__version__),
                                       "device": args.device, "threads": torch.get_num_threads()},
                          code_revision=code_revision)
            destination = args.output_dir / f"{method}-seed{seed}.json"
            destination.write_text(json.dumps(result, indent=2) + "\n")
            print(destination, flush=True)


if __name__ == "__main__":
    main()
