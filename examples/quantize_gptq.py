"""Pinned llm-compressor W4A16 integration; run in its separate environment.

Recipe API follows the upstream 0.5.1 quantization_w4a16 example; see
docs/serving.md for attribution, hardware requirements and validation status.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from tn_compression.config import load_config
from tn_compression.tasks.language import text_records


def quantize(config, output):
    if importlib.metadata.version("llmcompressor") != "0.5.1":
        raise ValueError("This recipe requires llmcompressor==0.5.1 in a separate environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("The pinned GPTQ recipe requires CUDA; CPU rounding is a separate reference.")
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("The selected compressed-tensors serving path needs Ampere or newer hardware.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    from datasets import Dataset
    from transformers import AutoTokenizer
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.transformers import oneshot
    from tn_compression.models import load_model

    records = text_records(config)["calibration"]
    spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"], trust_remote_code=False)
    model = load_model({**spec, "kwargs": {**spec.get("kwargs", {}), "torch_dtype": torch.float16}})
    model.to("cuda")
    maximum = config["data"].get("max_length", 128)
    dataset = Dataset.from_list([tokenizer(row["text"], truncation=True, max_length=maximum) for row in records])
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(model=model, dataset=dataset, recipe=recipe, max_seq_length=maximum,
            num_calibration_samples=len(records))
    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    metadata = {"workflow": config, "method": "GPTQ", "scheme": "W4A16", "group_size": 128,
                "format": "compressed-tensors", "llmcompressor": "0.5.1", "ignore": ["lm_head"],
                "calibration_ids": [row["id"] for row in records],
                "status": "exported; reload, quality and backend parity still required"}
    (output / "compression-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quantize(load_config(args.config), args.output)
