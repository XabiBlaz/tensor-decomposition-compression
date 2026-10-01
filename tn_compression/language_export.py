"""Standard Transformers export for verified dense-shaped language artifacts."""

import inspect
import json
from pathlib import Path

import torch

from .tasks.language import model_inputs
from .tasks.vision import evaluation_mode


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=False)[0]


@torch.no_grad()
def export_language(model, path, batch, *, tokenizer=None, device="cpu"):
    """Save and verify a standard HF artifact; factorized modules need another backend."""
    from transformers import AutoModelForCausalLM
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {path}")
    unsupported = [name for name, module in model.named_modules()
                   if getattr(module, "_tn_replacement", False) and type(module) is not torch.nn.Linear]
    if unsupported:
        raise ValueError(
            "Standard Transformers save_pretrained cannot reconstruct custom compressed modules "
            f"(including factorized or packed-int8 layers): {unsupported}. Use the reconstructible "
            "PyTorch bundle; fixed-logits ONNX is a separately limited artifact.")
    inputs = model_inputs(batch, device)
    with evaluation_mode(model):
        expected = model(**inputs, use_cache=False).logits
        model.save_pretrained(path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    restored = AutoModelForCausalLM.from_pretrained(path, torch_dtype=next(model.parameters()).dtype,
                 attn_implementation=model.config._attn_implementation, trust_remote_code=False).to(device).eval()
    actual = restored(**inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    result = {"status": "verified_transformers", "maximum_logit_error": (actual - expected).abs().max().item(),
              "rtol": 1e-5, "atol": 1e-6, "transformations": getattr(model, "_tn_transformations", []),
              "vllm": "unverified; requires separate loader and output comparison"}
    (path / "compression-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@torch.no_grad()
def export_fixed_logits_onnx(model, path, batch, *, device="cpu", rtol=1e-3, atol=3e-4):
    """Export a fixed-shape logits graph and verify CPU ORT parity.

    This deliberately does not advertise generation serving: past-key-value
    inputs/outputs and dynamic sequence lengths are outside this graph.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"ONNX artifact already exists: {path}")
    inputs = model_inputs(batch, device)
    if set(inputs) < {"input_ids", "attention_mask"}:
        raise ValueError("Language ONNX export needs input_ids and attention_mask.")
    wrapper = _LogitsOnly(model)
    with evaluation_mode(model):
        expected = wrapper(inputs["input_ids"], inputs["attention_mask"]).float().cpu().numpy()
        options = {"input_names": ["input_ids", "attention_mask"], "output_names": ["logits"],
                   "opset_version": 17, "do_constant_folding": False}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            options["dynamo"] = False
        torch.onnx.export(wrapper, (inputs["input_ids"], inputs["attention_mask"]), str(path), **options)
    # Passing the path lets ONNX validate external tensor data without trying
    # to reserialize a potentially >2 GiB in-memory protobuf.
    onnx.checker.check_model(str(path))
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    feeds = {"input_ids": inputs["input_ids"].cpu().numpy(),
             "attention_mask": inputs["attention_mask"].cpu().numpy()}
    actual = session.run(["logits"], feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    files = [item for item in path.parent.iterdir() if item.is_file()]
    return {
        "status": "verified_fixed_shape_logits_only", "serving_ready": False,
        "shape": list(inputs["input_ids"].shape), "precision": str(next(model.parameters()).dtype),
        "runtime": "onnxruntime", "requested_providers": providers,
        "active_providers": session.get_providers(), "provider_fallback": False,
        "maximum_logit_error": float(np.abs(actual - expected).max()),
        "rtol": rtol, "atol": atol,
        "artifact_bytes": sum(item.stat().st_size for item in files),
        "files": sorted(item.name for item in files),
        "limitations": [
            "Fixed input shape and logits only.",
            "No past-key-value cache inputs or outputs; autoregressive generation was not verified in ONNX.",
            "Not benchmarked against cached PyTorch generation and not serving-ready.",
        ],
    }
