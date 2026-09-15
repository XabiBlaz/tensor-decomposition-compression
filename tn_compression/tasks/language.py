"""Causal-language-model evaluation with explicit token masks and data roles."""

import math
import hashlib
import heapq
import json
import re
from pathlib import Path

import torch
from torch.nn import functional as F

from .vision import evaluation_mode


def shifted_targets(batch):
    labels = batch.get("labels", batch["input_ids"]).clone()
    attention = batch.get("attention_mask", torch.ones_like(labels))
    # A prediction is valid only when both its prefix position and target token
    # are real tokens; this also excludes the first token after left padding.
    valid = attention[:, 1:].bool() & attention[:, :-1].bool() & (labels[:, 1:] != -100)
    return labels[:, 1:].masked_fill(~valid, -100), valid


def model_inputs(batch, device):
    return {name: value.to(device) for name, value in batch.items()
            if name in {"input_ids", "attention_mask", "position_ids"}}


@torch.no_grad()
def evaluate_language(model, batches, *, device="cpu"):
    loss_sum = 0.0
    tokens = sequences = 0
    with evaluation_mode(model):
        for batch in batches:
            targets, valid = shifted_targets(batch)
            if not valid.any():
                continue
            outputs = model(**model_inputs(batch, device), use_cache=False)
            logits = outputs.logits[:, :-1].float()
            loss_sum += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(device).reshape(-1),
                                         ignore_index=-100, reduction="sum").item()
            tokens += valid.sum().item()
            sequences += len(batch["input_ids"])
    if not tokens:
        raise ValueError("Language evaluation has no valid next-token targets.")
    nll = loss_sum / tokens
    return {"task": "causal_lm", "nll": nll, "perplexity": math.exp(nll) if nll < 700 else None,
            "tokens": tokens, "sequences": sequences}


@torch.no_grad()
def teacher_kl(original, candidate, batches, *, device="cpu", token_chunk=32, temperature=1.0):
    if temperature <= 0 or token_chunk < 1:
        raise ValueError("temperature and token_chunk must be positive.")
    total = 0.0
    tokens = 0
    with evaluation_mode(original), evaluation_mode(candidate):
        for batch in batches:
            _, valid = shifted_targets(batch)
            inputs = model_inputs(batch, device)
            teacher = original(**inputs, use_cache=False).logits[:, :-1].detach().cpu()
            student = candidate(**inputs, use_cache=False).logits[:, :-1].detach().cpu()
            teacher, student = teacher[valid], student[valid]
            for start in range(0, len(teacher), token_chunk):
                teacher_log = (teacher[start:start + token_chunk].float() / temperature).log_softmax(-1)
                student_log = (student[start:start + token_chunk].float() / temperature).log_softmax(-1)
                total += (teacher_log.exp() * (teacher_log - student_log)).sum().item()
                tokens += len(teacher_log)
    if not tokens:
        raise ValueError("KL evaluation has no valid tokens.")
    return {"teacher_to_candidate_kl": total / tokens, "tokens": tokens, "temperature": temperature}


def text_batches(tokenizer, texts, *, max_length=128, batch_size=1):
    if max_length < 2 or batch_size < 1:
        raise ValueError("max_length must be at least two and batch_size must be positive.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a padding or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    for index in range(0, len(texts), batch_size):
        yield dict(tokenizer(texts[index:index + batch_size], padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt"))


def validate_split_ids(roles):
    seen = set()
    for name, identifiers in roles.items():
        identifiers = list(identifiers)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{name} must contain unique nonempty example IDs.")
        if seen.intersection(identifiers):
            raise ValueError("Calibration, allocation validation and test IDs must be disjoint.")
        seen.update(identifiers)


def text_records(config):
    """Resolve all data roles together so overlap fails before model selection."""
    data = config["data"]
    roles = ("calibration", "validation", "test")
    if data["kind"] == "text_json":
        records = json.loads(Path(data["path"]).read_text())
        if "train" in records:
            roles = (*roles, "train")
    elif data["kind"] == "huggingface":
        from datasets import load_dataset
        if not re.fullmatch(r"[0-9a-f]{40}", data.get("revision", "")):
            raise ValueError("Pin a full Hugging Face dataset revision.")
        splits = data["splits"]
        if "train" in splits:
            roles = (*roles, "train")
        # Split expressions can alias rows (train[:10] and train[:20]); require
        # canonical partitions here. Use explicit IDs in text_json for subsets.
        if any(not re.fullmatch(r"[A-Za-z0-9_]+", splits[role]) for role in roles):
            raise ValueError("Use canonical dataset splits or text_json with explicit IDs.")
        count = data.get("samples", 16)
        if count < 1:
            raise ValueError("samples must be positive.")
        records = {}
        for role in roles:
            split = splits[role]
            dataset = load_dataset(data["name"], data.get("subset"), revision=data["revision"], split=split)
            column = data.get("text_column", "text")
            candidates = ((index, row[column]) for index, row in enumerate(dataset)
                          if len(row[column].strip()) >= data.get("minimum_characters", 80))
            selected = heapq.nsmallest(count, candidates, key=lambda row: hashlib.sha256(
                f"{config.get('seed', 0)}:{split}:{row[0]}".encode()).hexdigest())
            records[role] = [{"id": f"{data['name']}@{data['revision']}:{split}:{index}", "text": text}
                             for index, text in selected]
    else:
        raise ValueError("Language data must be text_json or a pinned Hugging Face dataset.")
    validate_split_ids({role: [item["id"] for item in records[role]] for role in roles})
    # Distinct IDs can still hide duplicated text. Within-role repeats are allowed.
    hashes = {role: {hashlib.sha256(" ".join(item["text"].split()).encode()).hexdigest()
                     for item in records[role]} for role in roles}
    validate_split_ids(hashes)
    return records


def load_text_split(config, role):
    """Load pinned text and tokenization with overlap checks across all roles."""
    from transformers import AutoTokenizer
    records = text_records(config)[role]
    tokenizer_spec = config.get("tokenizer", config["model"])
    if not Path(tokenizer_spec["name"]).is_dir() and not re.fullmatch(
            r"[0-9a-f]{40}", tokenizer_spec.get("revision", "")):
        raise ValueError("Pin a full tokenizer revision or use a local tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec["name"], revision=tokenizer_spec.get("revision"),
                                              trust_remote_code=False)
    data = config["data"]
    batches = list(text_batches(tokenizer, [item["text"] for item in records], max_length=data.get("max_length", 128),
                                batch_size=data.get("batch_size", 1)))
    return batches, [item["id"] for item in records]
