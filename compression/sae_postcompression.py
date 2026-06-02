from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tn_compression.api import apply_compression_plan, generate_compression_plan

try:
    from .compress_model import build_compression_config, parse_input_shape, parse_rank, resolve_demo_device, resolve_input_shape
    from .metrics import cosine_and_l2, flatten_activation, linear_cka, set_reproducible_seed
    from .models import available_model_names, build_model, get_model_info, load_full_module_artifact, synthetic_input
    from .report import write_json
    from .representation_analysis import collect_activations, select_layers
except ImportError:
    from compress_model import build_compression_config, parse_input_shape, parse_rank, resolve_demo_device, resolve_input_shape
    from metrics import cosine_and_l2, flatten_activation, linear_cka, set_reproducible_seed
    from models import available_model_names, build_model, get_model_info, load_full_module_artifact, synthetic_input
    from report import write_json
    from representation_analysis import collect_activations, select_layers


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = F.relu(self.encoder(x))
        reconstruction = self.decoder(code)
        return reconstruction, code


def run_sae_demo(args: argparse.Namespace) -> Dict[str, Any]:
    set_reproducible_seed(args.seed)
    if args.rank_cap is not None and args.rank_cap <= 0:
        args.rank_cap = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_demo_device(args.device)
    info = get_model_info(args.model)
    original_model = build_model(
        args.model,
        artifact_path=args.artifact_path,
        weights=args.weights,
        num_classes=args.num_classes,
    ).to(device)
    base_input_shape, input_shape_source = resolve_input_shape(
        args.input_shape,
        None if args.model == "artifact" else info.default_input_shape,
        original_model,
        device,
        require_forward=True,
    )
    assert base_input_shape is not None
    input_shape = (args.batch_size, *tuple(base_input_shape[1:]))
    layer_name = resolve_layer_name(original_model, args.layer, max_layers=args.max_layers)

    if args.compressed_artifact_path:
        compressed_model = load_full_module_artifact(args.compressed_artifact_path).to(device)
        compressed_layers = None
        skipped_layers = None
        compression_warnings: List[str] = []
        plan_path = None
        comparison_mode = "provided_compressed_artifact"
    else:
        config = build_compression_config(
            output_dir,
            method=args.method,
            rank=args.rank,
            structure=args.structure,
            device=device,
            rank_method=args.rank_method,
            energy=args.energy,
            rank_cap=args.rank_cap,
        )
        plan_result = generate_compression_plan(original_model, config, inplace=False, device=device)
        plan_path = plan_result.plan_path
        compressed_model = copy.deepcopy(original_model).to(device)
        compressed_layers, skipped_layers, compression_warnings = apply_compression_plan(compressed_model, plan_result.plan)
        comparison_mode = "internal_compression"

    input_batches = [synthetic_input(input_shape, device=device) for _ in range(args.batches)]
    original_acts, original_warnings = collect_activations(
        original_model,
        [layer_name],
        input_batches=input_batches,
        device=device,
    )
    compressed_acts, compressed_warnings = collect_activations(
        compressed_model,
        [layer_name],
        input_batches=input_batches,
        device=device,
    )
    if layer_name not in compressed_acts:
        raise RuntimeError(f"Layer '{layer_name}' did not produce tensor activations.")

    compressed_matrix = flatten_activation(compressed_acts[layer_name], max_rows=args.max_rows)
    original_matrix = original_acts.get(layer_name)
    original_flat = flatten_activation(original_matrix, max_rows=args.max_rows) if original_matrix is not None else None
    if compressed_matrix.shape[0] < 2 or compressed_matrix.shape[1] < 2:
        raise RuntimeError(f"Layer '{layer_name}' produced too few activation samples for SAE training: {list(compressed_matrix.shape)}")

    train_matrix, feature_mean, feature_std = standardize(compressed_matrix)
    hidden_dim = max(1, int(train_matrix.shape[1] * args.sae_hidden_mult))
    sae = SparseAutoencoder(int(train_matrix.shape[1]), hidden_dim).to(device)
    history = train_sae(
        sae,
        train_matrix.to(device),
        steps=args.steps,
        batch_size=args.sae_batch_size,
        lr=args.lr,
        l1_coef=args.l1_coef,
        seed=args.seed,
    )
    metrics = evaluate_sae(sae, train_matrix.to(device))
    comparison = compare_activation_sets(original_flat, compressed_matrix)
    plot_path = maybe_write_sae_activity_plot(output_dir / "sae_feature_activity.png", metrics)

    summary = {
        "model": args.model,
        "model_family": info.family,
        "weights": None if args.model == "artifact" else args.weights,
        "input_shape": list(input_shape),
        "input_shape_source": input_shape_source,
        "sae_training_source": "compressed_model_activations",
        "device": str(device),
        "layer": layer_name,
        "comparison_mode": comparison_mode,
        "method": args.method,
        "rank": args.rank,
        "rank_method": args.rank_method,
        "energy": args.energy,
        "rank_cap": args.rank_cap,
        "structure": args.structure,
        "compressed_artifact_path": args.compressed_artifact_path,
        "compressed_layers": compressed_layers,
        "skipped_layers": skipped_layers,
        "plan_path": plan_path,
        "activation_matrix_shape": list(compressed_matrix.shape),
        "feature_mean_abs": float(feature_mean.abs().mean().item()),
        "feature_std_mean": float(feature_std.mean().item()),
        "sae": {
            "input_dim": int(train_matrix.shape[1]),
            "hidden_dim": hidden_dim,
            "hidden_mult": args.sae_hidden_mult,
            "steps": args.steps,
            "lr": args.lr,
            "l1_coef": args.l1_coef,
            "train_loss_initial": history[0] if history else None,
            "train_loss_final": history[-1] if history else None,
            **metrics,
        },
        "original_vs_compressed_activation": comparison,
        "warnings": list(compression_warnings) + original_warnings + compressed_warnings + [
            "This is a toy post-compression SAE probe, not evidence of monosemantic features.",
            "Synthetic inputs do not validate task quality or semantic interpretability.",
        ],
        "plot_path": plot_path,
    }
    write_json(output_dir / "sae_summary.json", summary)
    write_sae_report(output_dir / "sae_report.md", summary)
    print(f"Layer: {layer_name}")
    print(f"Activation matrix: {list(compressed_matrix.shape)}")
    print(f"SAE reconstruction MSE: {metrics['reconstruction_mse']:.6g}")
    print(f"SAE mean L0: {metrics['mean_l0']:.6g}")
    print(f"Dead latent fraction: {metrics['dead_latent_fraction']:.6g}")
    print(f"Wrote summary: {output_dir / 'sae_summary.json'}")
    print(f"Wrote report: {output_dir / 'sae_report.md'}")
    return summary


def resolve_layer_name(model: nn.Module, selector: str, *, max_layers: int) -> str:
    if selector != "auto":
        return selector.strip().replace(".", "/")
    selected = select_layers(model, "auto", max_layers=max_layers)
    if not selected:
        raise RuntimeError("Could not find a Conv2d/Linear/ReLU/block layer for SAE activation collection.")
    conv_or_linear = []
    module_map = {name.replace(".", "/"): module for name, module in model.named_modules()}
    for name in selected:
        if isinstance(module_map.get(name), (nn.Conv2d, nn.Linear)):
            conv_or_linear.append(name)
    return conv_or_linear[min(len(conv_or_linear) - 1, max(0, len(conv_or_linear) // 2))] if conv_or_linear else selected[0]


def standardize(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (matrix - mean) / std, mean, std


def train_sae(
    sae: SparseAutoencoder,
    matrix: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    l1_coef: float,
    seed: int,
) -> List[float]:
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    history: List[float] = []
    rows = matrix.shape[0]
    for _ in range(max(1, steps)):
        indices = torch.randint(rows, (min(batch_size, rows),), generator=generator, device=matrix.device)
        batch = matrix[indices]
        reconstruction, code = sae(batch)
        mse = F.mse_loss(reconstruction, batch)
        l1 = code.abs().mean()
        loss = mse + l1_coef * l1
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu().item()))
    return history


@torch.no_grad()
def evaluate_sae(sae: SparseAutoencoder, matrix: torch.Tensor) -> Dict[str, Any]:
    sae.eval()
    reconstruction, code = sae(matrix)
    residual = matrix - reconstruction
    mse = float(F.mse_loss(reconstruction, matrix).cpu().item())
    total_variance = float(torch.var(matrix, unbiased=False).cpu().item())
    sse = torch.sum(residual.square())
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    sst = torch.sum(centered.square()).clamp_min(1e-12)
    active = code > 1e-6
    activity_per_latent = active.float().mean(dim=0)
    decoder_norms = torch.linalg.vector_norm(sae.decoder.weight.detach(), dim=0).cpu()
    return {
        "reconstruction_mse": mse,
        "normalized_reconstruction_error": mse / max(total_variance, 1e-12),
        "explained_variance": float((1.0 - sse / sst).cpu().item()),
        "mean_l0": float(active.float().sum(dim=1).mean().cpu().item()),
        "latent_activation_sparsity": float((1.0 - active.float().mean()).cpu().item()),
        "dead_latent_fraction": float((activity_per_latent == 0).float().mean().cpu().item()),
        "latent_activity_mean": float(activity_per_latent.mean().cpu().item()),
        "latent_activity_top10": [float(value) for value in torch.sort(activity_per_latent, descending=True).values[:10].cpu().tolist()],
        "decoder_norm_min": float(decoder_norms.min().item()),
        "decoder_norm_mean": float(decoder_norms.mean().item()),
        "decoder_norm_max": float(decoder_norms.max().item()),
    }


def compare_activation_sets(original: Optional[torch.Tensor], compressed: torch.Tensor) -> Dict[str, Any]:
    if original is None:
        return {"available": False, "warning": "Original activation was unavailable."}
    rows = min(original.shape[0], compressed.shape[0])
    original = original[:rows]
    compressed = compressed[:rows]
    cka, cka_warning = linear_cka(original, compressed)
    similarity = cosine_and_l2(original, compressed)
    return {
        "available": True,
        "linear_cka": cka,
        "cosine_similarity": similarity.get("cosine_similarity"),
        "relative_l2_diff": similarity.get("relative_l2_diff"),
        "warning": cka_warning,
    }


def maybe_write_sae_activity_plot(path: Path, metrics: Dict[str, Any]) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None
    activity = metrics.get("latent_activity_top10", [])
    if not activity:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(activity)), activity, color="#4c78a8")
    ax.set_xlabel("Top active latent index")
    ax.set_ylabel("Activation frequency")
    ax.set_title("Top SAE Latent Activity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def write_sae_report(path: Path, summary: Dict[str, Any]) -> None:
    sae = summary["sae"]
    comparison = summary.get("original_vs_compressed_activation", {})
    lines = [
        "# SAE Post-Compression Demo",
        "",
        "This is a toy post-compression sparse-autoencoder probe. It trains a small SAE on compressed-model activations.",
        "",
        "It does not fine-tune the compressed model, recover task performance, claim monosemanticity, or interpret semantic features.",
        "",
        "## Run",
        "",
        f"- Model: `{summary.get('model')}`",
        f"- Weights: `{summary.get('weights', 'n/a')}`",
        f"- Layer: `{summary.get('layer')}`",
        f"- Device: `{summary.get('device')}`",
        f"- Activation matrix: `{summary.get('activation_matrix_shape')}`",
        f"- Input shape source: `{summary.get('input_shape_source')}`",
        f"- SAE training source: `{summary.get('sae_training_source')}`",
        f"- Compression method: `{summary.get('method')}`",
        f"- Rank / rank policy: `{summary.get('rank')}`",
        f"- Automatic rank method: `{summary.get('rank_method')}`",
        f"- Rank cap: `{summary.get('rank_cap')}`",
        "",
        "## SAE Metrics",
        "",
        f"- Reconstruction MSE: {sae.get('reconstruction_mse'):.6g}",
        f"- Normalized reconstruction error: {sae.get('normalized_reconstruction_error'):.6g}",
        f"- Explained variance: {sae.get('explained_variance'):.6g}",
        f"- Mean L0: {sae.get('mean_l0'):.6g}",
        f"- Latent activation sparsity: {sae.get('latent_activation_sparsity'):.6g}",
        f"- Dead latent fraction: {sae.get('dead_latent_fraction'):.6g}",
        f"- Decoder norm mean/min/max: {sae.get('decoder_norm_mean'):.6g} / {sae.get('decoder_norm_min'):.6g} / {sae.get('decoder_norm_max'):.6g}",
        "",
        "## Original vs Compressed Activation",
        "",
        f"- Linear CKA: {comparison.get('linear_cka')}",
        f"- Cosine similarity: {comparison.get('cosine_similarity')}",
        f"- Relative L2 difference: {comparison.get('relative_l2_diff')}",
        "",
        "## Interpretation",
        "",
        "- Lower reconstruction error means the SAE can reconstruct compressed activations more easily.",
        "- Lower mean L0 and higher sparsity mean fewer latent units are active per activation vector.",
        "- A high dead latent fraction means the SAE hidden size or L1 penalty is too large for the collected activations.",
        "- These metrics are useful diagnostics, not proof that the compressed model retains task-relevant semantic features.",
    ]
    plot_path = summary.get("plot_path")
    if plot_path:
        lines.extend(["", "## Plot", "", f"- `{plot_path}`"])
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny SAE on post-compression activations.")
    parser.add_argument("--model", default="torchvision_resnet18", choices=available_model_names())
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--compressed-artifact-path", default=None)
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"], help="Torchvision weights for built-in models. 'imagenet' may download weights if unavailable locally.")
    parser.add_argument("--input-shape", type=parse_input_shape, default=None)
    parser.add_argument("--method", default="tensor_train", choices=["tensor_train", "partial_tucker", "cp3", "cp2", "tt"])
    parser.add_argument("--rank", type=parse_rank, default=None)
    parser.add_argument("--rank-method", default="SVD")
    parser.add_argument("--energy", type=float, default=0.94)
    parser.add_argument("--rank-cap", type=int, default=8)
    parser.add_argument("--structure", default="TTPWT")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--sae-hidden-mult", type=int, default=4)
    parser.add_argument("--sae-batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-coef", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=1000, help="Classifier classes for --weights none. ImageNet weights keep the standard 1000-class head.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_sae_demo(args)


if __name__ == "__main__":
    main()
