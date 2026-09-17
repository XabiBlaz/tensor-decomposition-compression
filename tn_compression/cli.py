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


def analysis_batches(config, role):
    if config.get("task") == "causal_lm":
        from .tasks.language import load_text_split
        return load_text_split(config, role)
    batches = vision_batches(config, role)
    dataset = batches.dataset
    identifiers = getattr(dataset, "identifiers", None)
    if identifiers is None:
        identifiers = [f"{config.get('data', {}).get('kind', 'data')}:{role}:{index}"
                       for index in range(len(dataset))]
    return batches, identifiers


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
    if config.get("task") == "causal_lm":
        from .tasks.language import evaluate_language, load_text_split
        batches, identifiers = load_text_split(config, role)
        return {**evaluate_language(model, batches, device=device), "example_ids": identifiers, "role": role}
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
        if args.command == "analyze":
            from .analyzer.reporting import write_analysis_artifacts
            from .analyzer.service import analyze_model
            analysis = config.get("analysis", {})
            level = args.level or analysis.get("level", "structural")
            calibration_batches = validation_batches = calibration_ids = None
            detection_coco = None
            if level in {"calibrated", "validated"}:
                calibration_batches, calibration_ids = analysis_batches(config, "calibration")
            if level == "validated":
                validation_batches, _ = analysis_batches(config, "validation")
                if config.get("task") == "detection":
                    detection_coco = validation_batches.dataset.coco
            target_mb = args.target_size_mb
            if target_mb is None:
                target_mb = analysis.get("target_size_mb")
            max_quality_loss = args.max_quality_loss
            if max_quality_loss is None:
                max_quality_loss = analysis.get("max_quality_loss")
            report = analyze_model(
                model, config, level=level,
                calibration_batches=calibration_batches,
                validation_batches=validation_batches,
                calibration_ids=calibration_ids,
                target_size_bytes=round(target_mb * 1024 * 1024) if target_mb is not None else None,
                max_quality_loss=max_quality_loss,
                requested_backend=args.backend or analysis.get("backend", "pytorch"),
                device=device, detection_coco=detection_coco)
            paths = write_analysis_artifacts(report, output)
            result = report.to_dict()
            result["artifacts"] = paths
        elif args.command == "calibrate":
            if config.get("task") != "causal_lm":
                raise ValueError("CLI calibration currently uses the causal-LM adapter; vision has Python task adapters.")
            from .calibration import collect_linear_inputs
            from .tasks.language import load_text_split, model_inputs
            batches, identifiers = load_text_split(config, "calibration")
            calibration = config.get("calibration", {})
            layers = calibration.get("layers", [])
            if not layers:
                raise ValueError("Select calibration.layers explicitly from inspection.")
            samples = {}
            for index, path in enumerate(layers):
                values = collect_linear_inputs(model, path, batches,
                       lambda net, batch: net(**model_inputs(batch, device), use_cache=False),
                       max_rows=calibration.get("max_rows", 256), seed=config.get("seed", 0),
                       input_mask=lambda batch: batch.get("attention_mask"))
                filename = f"linear_inputs_{index}.pt"
                torch.save(values, output / filename)
                samples[path] = {"shape": list(values.shape), "file": filename}
            result = {"example_ids": identifiers, "layers": samples, "workflow": config,
                      "reuse": "Recomputed on every run; stored samples are not an automatic cache."}
        elif args.command == "plan" and config.get("task") == "causal_lm":
            from .language_experiments import evaluate_candidates
            from .tasks.language import load_text_split
            calibration_batches, calibration_ids = load_text_split(config, "calibration")
            validation_batches, validation_ids = load_text_split(config, "validation")
            options = config.get("calibration", {})
            result = evaluate_candidates(model, calibration_batches, validation_batches, config["candidates"],
                       device=device, max_rows=options.get("max_rows", 256), seed=config.get("seed", 0),
                       method=config.get("candidate_method", "svd"))
            result.update(workflow=config, example_ids={"calibration": calibration_ids, "validation": validation_ids})
        elif args.command in {"inspect", "plan"}:
            result = generate_compression_plan(model, compression_config(config, output), device=device).plan
        elif args.command == "compress" and args.plan:
            from .analyzer.service import apply_compression_plan
            plan = json.loads(Path(args.plan).read_text())
            calibration_batches = calibration_ids = None
            if any(item.get("method") == "weighted_svd" for item in plan.get("transformations", [])):
                calibration_batches, calibration_ids = analysis_batches(config, "calibration")
            result = apply_compression_plan(model, plan, config,
                                            calibration_batches=calibration_batches,
                                            calibration_ids=calibration_ids, device=device)
            save_bundle(model, output / "bundle", metadata={"workflow": config, "analysis_plan": plan})
        elif args.command == "compress" and config.get("allocation"):
            from .allocation import allocate_ranks
            from .tasks.language import load_text_split
            if config.get("task") != "causal_lm":
                raise ValueError("Rank allocation currently uses the causal-LM task contract.")
            calibration_batches, calibration_ids = load_text_split(config, "calibration")
            validation_batches, validation_ids = load_text_split(config, "validation")
            result = allocate_ranks(model, calibration_batches, validation_batches, config["candidates"],
                       device=device, seed=config.get("seed", 0), **config["allocation"])
            result["example_ids"] = {"calibration": calibration_ids, "validation": validation_ids}
            if result["model_tensor_bytes"] < result["original_tensor_bytes"]:
                save_bundle(model, output / "bundle", metadata={"workflow": config, "allocation": result})
        elif args.command == "compress" and config.get("pruning"):
            from .language_experiments import prune_language
            from .tasks.language import load_text_split
            if config.get("task") != "causal_lm":
                raise ValueError("Gated MLP pruning uses the causal-LM task contract.")
            calibration_batches, calibration_ids = load_text_split(config, "calibration")
            validation_batches, validation_ids = load_text_split(config, "validation")
            result = prune_language(model, calibration_batches, validation_batches, device=device,
                                    seed=config.get("seed", 0), **config["pruning"])
            result["example_ids"] = {"calibration": calibration_ids, "validation": validation_ids}
            if result["status"] == "accepted":
                save_bundle(model, output / "bundle", metadata={"workflow": config, "pruning": result})
        elif args.command == "compress":
            result_object = compress_model(model, compression_config(config, output), inplace=True,
                                           device=device, return_model=True)
            result = result_object.to_dict()
            if not result_object.compressed_layers:
                raise ValueError("Compression changed zero layers; see the generated plan and skip reasons.")
            save_bundle(result_object.model, output / "bundle", metadata={"workflow": config, "compression": result["summary"]})
        elif args.command in {"train", "finetune"}:
            if config.get("task") == "causal_lm":
                if not args.checkpoint:
                    raise ValueError("Language recovery requires an explicit --checkpoint, including for the dense control.")
                from .tasks.language import load_text_split
                from .tasks.recovery import recover_language
                training_batches, training_ids = load_text_split(config, "train")
                result = recover_language(model, training_batches, device=device, **config["recovery"])
                result["example_ids"] = training_ids
                save_bundle(model, output / "bundle", metadata={"workflow": config, "recovery": result})
                (output / f"{args.command}.json").write_text(json.dumps(result, indent=2) + "\n")
                return result
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
            if config.get("task") == "causal_lm":
                from transformers import AutoTokenizer
                from .tasks.language import load_text_split
                from .language_export import export_language
                batches, _ = load_text_split(config, args.role)
                spec = config.get("tokenizer", config["model"])
                tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec.get("revision"), trust_remote_code=False)
                result = export_language(model, output / "huggingface", batches[0], tokenizer=tokenizer, device=device)
                (output / "export.json").write_text(json.dumps(result, indent=2) + "\n")
                return result
            if config.get("task") == "detection":
                raise ValueError("ONNX export currently supports classification and segmentation.")
            from .export import export_onnx
            inputs, _ = next(iter(vision_batches(config, args.role)))
            result = export_onnx(model, inputs.to(device), output / "model.onnx")
        else:
            raise ValueError(f"Unknown command: {args.command}")
    result.setdefault("workflow", config)
    (output / f"{args.command}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Inspect, compress and evaluate configurable PyTorch models.")
    parser.add_argument("command", choices=["inspect", "calibrate", "analyze", "plan", "compress", "train", "finetune", "evaluate", "export", "benchmark"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--role", choices=["calibration", "validation", "test"], default="validation")
    parser.add_argument("--level", choices=["structural", "calibrated", "validated"])
    parser.add_argument("--target-size-mb", type=float)
    parser.add_argument("--max-quality-loss", type=float)
    parser.add_argument("--backend")
    parser.add_argument("--plan", help="Compression plan produced by tn-compress analyze.")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
