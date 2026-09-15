"""Small command handlers over the same public Python workflows."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .api import compress_model, generate_compression_plan
from .artifacts import resolve_device
from .checkpoints import load_bundle, save_bundle
from .config import load_config
from .models import load_model


def vision_batches(config, role):
    from .tasks.data import CocoDetectionDataset, OxfordPetDataset, SyntheticVisionDataset
    from .tasks.detection import detection_collate
    data = dict(config.get("data", {}))
    kind = data.pop("kind", "synthetic")
    batch_size = data.pop("batch_size", 2)
    task = config.get("task", "segmentation")
    classes = config.get("num_classes", 3)
    binary = config.get("binary", False)
    if kind == "synthetic":
        seed = data.pop("seed", 0) + {"train": 0, "calibration": 10000, "validation": 20000, "test": 30000}[role]
        dataset = SyntheticVisionDataset(task=task, num_classes=classes, binary=binary, seed=seed, **data)
    elif kind == "oxford_pet":
        dataset = OxfordPetDataset(task=task, role=role, binary=binary, **data)
    elif kind == "coco":
        dataset = CocoDetectionDataset(role=role, **data)
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=role == "train", num_workers=0,
                      collate_fn=detection_collate if task == "detection" else None)


def compression_config(config, output):
    return {"schema_version": 1, "artifact": {"kind": "live_module"},
            "compression": config.get("compression", {"default_method": {"type": "partial_tucker", "rank": 2}}),
            "output": {"dir": str(output), "plan_dir": str(output / "plans"),
                       "name": "compressed", "artifacts": [{"kind": "state_dict"}]}}


def evaluate(model, config, role, device):
    batches = vision_batches(config, role)
    if config.get("task") == "detection":
        from .tasks.detection import evaluate_detection
        return evaluate_detection(model, batches, batches.dataset.coco, device=device,
                                  category_mapping=config.get("category_mapping"))
    from .tasks.vision import evaluate_vision
    return evaluate_vision(model, batches, task=config.get("task", "segmentation"),
                           num_classes=config.get("num_classes", 3), binary=config.get("binary", False), device=device)


def run(args):
    config = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.get("seed", 0))
    device = str(resolve_device(args.device, config))
    if args.command == "benchmark":
        if not args.checkpoint:
            raise ValueError("benchmark requires --checkpoint.")
        from .benchmark import benchmark_bundle
        result = benchmark_bundle(args.checkpoint, device=device, task=config.get("task", "segmentation"),
                                  **config.get("benchmark", {}))
    else:
        model = load_bundle(args.checkpoint, device=device)[0] if args.checkpoint else load_model(config["model"]).to(device)
        model.eval()
        if args.command == "inspect":
            result = generate_compression_plan(model, compression_config(config, output), device=device).plan
        elif args.command == "compress":
            result_object = compress_model(model, compression_config(config, output), inplace=True,
                                           device=device, return_model=True)
            result = result_object.to_dict()
            if not result_object.compressed_layers:
                raise ValueError("Compression changed zero layers; see the generated plan and skip reasons.")
            save_bundle(result_object.model, output / "bundle", metadata={"workflow": config, "compression": result["summary"]})
        elif args.command in {"train", "finetune"}:
            if args.command == "finetune" and not args.checkpoint:
                raise ValueError("finetune requires an explicit compressed --checkpoint.")
            from .tasks.vision import train_vision
            training = dict(config.get("training", {}))
            result = {"history": train_vision(model, vision_batches(config, "train"), vision_batches(config, "validation"),
                                               task=config.get("task", "segmentation"), num_classes=config.get("num_classes", 3),
                                               binary=config.get("binary", False), device=device, **training)}
            save_bundle(model, output / "bundle", metadata={"workflow": config, "training": result})
        elif args.command == "evaluate":
            result = evaluate(model, config, args.role, device)
        elif args.command == "export":
            from .export import export_onnx
            inputs, _ = next(iter(vision_batches(config, args.role)))
            result = export_onnx(model, inputs.to(device), output / "model.onnx")
        else:
            raise ValueError(f"Unknown command: {args.command}")
    (output / f"{args.command}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Inspect, compress and evaluate configurable PyTorch models.")
    parser.add_argument("command", choices=["inspect", "compress", "train", "finetune", "evaluate", "export", "benchmark"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--role", choices=["calibration", "validation", "test"], default="validation")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
