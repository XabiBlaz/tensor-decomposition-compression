"""Standard Transformers export for verified dense-shaped language artifacts."""

import json
from pathlib import Path

import torch

from .tasks.language import model_inputs
from .tasks.vision import evaluation_mode


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
        raise ValueError(f"Standard Transformers export cannot execute factorized modules: {unsupported}")
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
