"""LoRA recovery tied to an immutable saved causal-LM base bundle."""

import json
from pathlib import Path

from ..checkpoints import file_digest, load_bundle
from .recovery import recover_language


def _base_identity(base_bundle_path):
    base = Path(base_bundle_path)
    manifest_path = base / "manifest.json"
    weights_path = base / "weights.pt"
    if not manifest_path.is_file() or not weights_path.is_file():
        raise ValueError("LoRA recovery needs a saved base bundle with manifest.json and weights.pt.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model", {}).get("source") != "transformers":
        raise ValueError("LoRA recovery currently supports Transformers causal LM bundles only.")
    digest = file_digest(weights_path)
    if digest != manifest.get("weights_sha256"):
        raise ValueError("Base bundle weight checksum does not match its manifest.")
    return manifest, digest


def recover_language_lora(model, batches, *, output_dir, base_bundle_path,
                          token_budget, max_updates, rank=8, alpha=16,
                          dropout=0.0, target_modules="all-linear",
                          learning_rate=1e-4, device="cpu"):
    """Train and save a PEFT adapter on the supplied, already-loaded base.

    Returns (PEFT model, JSON-compatible report). A caller must evaluate the
    returned model or reload it via ``load_lora_adapter`` using the *same* base
    bundle. The adapter alone does not contain a compressed/pruned base.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError("LoRA recovery needs peft; install the language extra or requirements/language.txt.") from error
    if not isinstance(rank, int) or rank < 1 or not isinstance(alpha, (float, int)) or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if not isinstance(dropout, (float, int)) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1).")
    if target_modules != "all-linear" and (not isinstance(target_modules, list)
                                           or not target_modules
                                           or any(not isinstance(value, str) or not value for value in target_modules)):
        raise ValueError("LoRA target_modules must be 'all-linear' or a nonempty list of module names.")
    manifest, digest = _base_identity(base_bundle_path)
    spec = getattr(model, "_tn_model_spec", None)
    if spec is None or spec.get("source") != "transformers":
        raise ValueError("LoRA recovery needs the causal LM loaded from the supplied base bundle.")
    if spec.get("name") != manifest["model"].get("name") or spec.get("revision") != manifest["model"].get("revision"):
        raise ValueError("Loaded model identity differs from the supplied base bundle.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"LoRA output directory is not empty: {output}")
    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    try:
        peft_model = get_peft_model(model, lora_config)
    except (TypeError, ValueError) as error:
        raise ValueError("LoRA could not attach to this architecture or target_modules. "
                         "Choose supported linear modules on the saved base bundle.") from error
    # PEFT marks only adapter parameters trainable; recover_language preserves
    # that selection and enforces the same real-token budget as full recovery.
    training = recover_language(peft_model, batches, token_budget=token_budget,
                                max_updates=max_updates, policy="trainable",
                                learning_rate=learning_rate, device=device)
    output.mkdir(parents=True, exist_ok=True)
    adapter = output / "adapter"
    peft_model.save_pretrained(adapter, safe_serialization=True)
    report = {
        "method": "lora", "base_model": spec.get("name", "inline-config"),
        "base_revision": spec.get("revision"), "base_weights_sha256": digest,
        "adapter_dir": str(adapter), "rank": rank, "alpha": alpha,
        "dropout": dropout, "target_modules": target_modules,
        "training": training,
        "reload": "Use load_lora_adapter with this adapter directory and the matching base bundle.",
    }
    (output / "lora_recovery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return peft_model, report


def load_lora_adapter(adapter_dir, *, base_bundle_path, device="cpu"):
    """Load an adapter only when its recorded compressed base is identical."""
    try:
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError("Loading a LoRA adapter needs peft.") from error
    adapter = Path(adapter_dir)
    metadata_path = adapter.parent / "lora_recovery.json"
    if not metadata_path.is_file() or not (adapter / "adapter_config.json").is_file():
        raise ValueError("The LoRA adapter needs adapter_config.json and its lora_recovery.json metadata.")
    report = json.loads(metadata_path.read_text(encoding="utf-8"))
    _, digest = _base_identity(base_bundle_path)
    if digest != report.get("base_weights_sha256"):
        raise ValueError("LoRA adapter belongs to a different base bundle; compressed/pruned weights must match.")
    base_model, _ = load_bundle(base_bundle_path, device=device)
    return PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
