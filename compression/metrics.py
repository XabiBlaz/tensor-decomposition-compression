from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


def set_reproducible_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def parameter_size_bytes(model: torch.nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def format_bytes(num_bytes: Optional[float]) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def summarize_output(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "device": str(output.device),
        }
    if isinstance(output, (list, tuple)):
        return {
            "type": type(output).__name__,
            "items": [summarize_output(item) for item in output],
        }
    if isinstance(output, Mapping):
        return {
            "type": "dict",
            "items": {str(key): summarize_output(value) for key, value in output.items()},
        }
    return {"type": type(output).__name__}


def _collect_tensors(output: Any, path: str = "output") -> Tuple[List[Tuple[str, torch.Tensor]], List[str]]:
    if isinstance(output, torch.Tensor):
        return [(path, output.detach())], []
    if isinstance(output, (list, tuple)):
        tensors: List[Tuple[str, torch.Tensor]] = []
        warnings: List[str] = []
        for index, item in enumerate(output):
            sub_tensors, sub_warnings = _collect_tensors(item, f"{path}[{index}]")
            tensors.extend(sub_tensors)
            warnings.extend(sub_warnings)
        return tensors, warnings
    if isinstance(output, Mapping):
        tensors = []
        warnings = []
        for key, item in output.items():
            sub_tensors, sub_warnings = _collect_tensors(item, f"{path}.{key}")
            tensors.extend(sub_tensors)
            warnings.extend(sub_warnings)
        return tensors, warnings
    return [], [f"{path}: unsupported output type {type(output).__name__}; numeric drift skipped for this item"]


def compare_outputs(before: Any, after: Any) -> Dict[str, Any]:
    before_tensors, before_warnings = _collect_tensors(before)
    after_tensors, after_warnings = _collect_tensors(after)
    warnings = before_warnings + after_warnings
    before_by_path = {path: tensor for path, tensor in before_tensors}
    after_by_path = {path: tensor for path, tensor in after_tensors}
    common_paths = sorted(set(before_by_path) & set(after_by_path))
    missing_before = sorted(set(after_by_path) - set(before_by_path))
    missing_after = sorted(set(before_by_path) - set(after_by_path))
    if missing_before or missing_after:
        warnings.append(
            f"Output structures differ; missing_before={missing_before}, missing_after={missing_after}"
        )

    per_tensor = []
    sum_abs = 0.0
    sum_count = 0
    max_abs = 0.0
    numerator_l2 = 0.0
    denominator_l2 = 0.0
    for path in common_paths:
        a = before_by_path[path].detach().float().cpu()
        b = after_by_path[path].detach().float().cpu()
        if list(a.shape) != list(b.shape):
            warnings.append(f"{path}: shape mismatch {list(a.shape)} vs {list(b.shape)}; drift skipped")
            continue
        diff = a - b
        abs_diff = diff.abs()
        item_max = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
        item_mean = float(abs_diff.mean().item()) if abs_diff.numel() else 0.0
        item_rel = _relative_l2(a, b)
        per_tensor.append(
            {
                "path": path,
                "shape": list(a.shape),
                "max_abs_diff": item_max,
                "mean_abs_diff": item_mean,
                "relative_l2_diff": item_rel,
            }
        )
        max_abs = max(max_abs, item_max)
        sum_abs += float(abs_diff.sum().item())
        sum_count += int(abs_diff.numel())
        numerator_l2 += float(torch.sum(diff * diff).item())
        denominator_l2 += float(torch.sum(a * a).item())

    numeric = bool(per_tensor)
    return {
        "numeric_drift_available": numeric,
        "max_abs_diff": max_abs if numeric else None,
        "mean_abs_diff": (sum_abs / sum_count) if sum_count else None,
        "relative_l2_diff": math.sqrt(numerator_l2 / denominator_l2) if denominator_l2 > 0 else None,
        "per_tensor": per_tensor,
        "before_signature": summarize_output(before),
        "after_signature": summarize_output(after),
        "warnings": warnings,
    }


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    diff_norm = torch.linalg.vector_norm(a - b)
    base_norm = torch.linalg.vector_norm(a)
    if float(base_norm.item()) == 0.0:
        return None
    return float((diff_norm / base_norm).item())


def benchmark_latency_ms(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    *,
    device: torch.device,
    warmup: int = 3,
    runs: int = 10,
) -> Dict[str, Any]:
    model.eval()
    sample_input = sample_input.to(device)
    timings: List[float] = []
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    timings_sorted = sorted(timings)
    return {
        "mean_ms": float(sum(timings) / len(timings)),
        "median_ms": float(timings_sorted[len(timings_sorted) // 2]),
        "min_ms": float(min(timings)),
        "runs": len(timings),
        "warmup": warmup,
    }


def plan_layer_summary(plan: Mapping[str, Any]) -> Dict[str, Any]:
    layers = list(plan.get("layers", []))
    eligible = [item for item in layers if item.get("eligible") and item.get("policy")]
    skipped = [item for item in layers if not (item.get("eligible") and item.get("policy"))]
    return {
        "total_layers": len(layers),
        "eligible_layers": len(eligible),
        "skipped_layers": len(skipped),
        "eligible_layer_names": [item.get("display_name") or item.get("name") for item in eligible],
        "skipped_layer_names": [item.get("display_name") or item.get("name") for item in skipped],
        "skip_reasons": [
            {
                "layer": item.get("display_name") or item.get("name"),
                "reason": item.get("skip_reason"),
                "type": item.get("type"),
            }
            for item in skipped
        ],
    }


def flatten_activation(activation: torch.Tensor, max_rows: int = 512) -> torch.Tensor:
    tensor = activation.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1])
    elif tensor.ndim == 3:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    elif tensor.ndim == 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    elif tensor.ndim >= 1:
        tensor = tensor.reshape(tensor.shape[0], -1)
    else:
        tensor = tensor.reshape(1, 1)
    if tensor.shape[0] > max_rows:
        tensor = tensor[:max_rows]
    return tensor


def singular_spectrum_summary(matrix: torch.Tensor, max_dim: int = 256) -> Dict[str, Any]:
    x = matrix.detach().float().cpu()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[0] == 0 or x.shape[1] == 0:
        return {"warning": "empty activation matrix"}
    if x.shape[1] > max_dim:
        x = x[:, :max_dim]
    x = x - x.mean(dim=0, keepdim=True)
    try:
        svals = torch.linalg.svdvals(x)
    except RuntimeError as exc:
        return {"warning": f"SVD failed: {exc}"}
    if svals.numel() == 0 or float(torch.sum(svals).item()) == 0.0:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "top_singular_values": [],
            "top5_energy_fraction": 0.0,
        }
    energy = svals.square()
    energy_total = torch.sum(energy)
    probs = energy / energy_total.clamp_min(1e-12)
    effective_rank = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    participation = energy_total.square() / energy.square().sum().clamp_min(1e-12)
    top_k = min(5, svals.numel())
    return {
        "effective_rank": float(effective_rank.item()),
        "participation_ratio": float(participation.item()),
        "top_singular_values": [float(v) for v in svals[:top_k].tolist()],
        "top5_energy_fraction": float((energy[:top_k].sum() / energy_total.clamp_min(1e-12)).item()),
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor, max_rows: int = 512) -> Tuple[Optional[float], Optional[str]]:
    if x.shape[0] != y.shape[0]:
        return None, f"CKA skipped: row count mismatch {x.shape[0]} vs {y.shape[0]}"
    if x.shape[0] > max_rows:
        return None, f"CKA skipped: {x.shape[0]} rows exceeds cap {max_rows}"
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    numerator = torch.linalg.matrix_norm(xty, ord="fro").square()
    denominator = torch.linalg.matrix_norm(xtx, ord="fro") * torch.linalg.matrix_norm(yty, ord="fro")
    if float(denominator.item()) == 0.0:
        return None, "CKA skipped: zero variance activation matrix"
    return float((numerator / denominator).item()), None


def cosine_and_l2(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Optional[float]]:
    if list(a.shape) != list(b.shape):
        return {"cosine_similarity": None, "relative_l2_diff": None}
    a_flat = a.detach().float().cpu().reshape(-1)
    b_flat = b.detach().float().cpu().reshape(-1)
    if a_flat.numel() == 0 or b_flat.numel() == 0:
        return {"cosine_similarity": None, "relative_l2_diff": None}
    cosine = F.cosine_similarity(a_flat, b_flat, dim=0)
    return {
        "cosine_similarity": float(cosine.item()),
        "relative_l2_diff": _relative_l2(a_flat, b_flat),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, Iterable):
        return [json_safe(item) for item in value]
    return repr(value)
