"""Common utilities for the three independent gradient-inversion attack setups.

This is a separate security-evaluation module. It does not modify the original
Flower training experiments. The attack uses a single-example gradient so that
a standard gradient-matching inversion objective is well defined.

The masking experiments apply the same masking generators used by the journal
code to the flattened gradient vector. This evaluates whether an individually
observed masked gradient is harder to invert; it is not a formal DP guarantee.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AttackResult:
    setup: str
    dataset: str
    target_client: int
    sample_index: int
    target_label: int
    num_clients: int
    round_id: int
    iterations: int
    attack_learning_rate: float
    reconstruction_mse: float
    reconstruction_mae: float
    reconstruction_l2: float
    reconstruction_cosine_similarity: float
    final_gradient_match_mse: float
    clean_gradient_norm: float
    observed_gradient_norm: float
    masking_norm: float
    mask_to_clean_gradient_ratio: float
    selected_masking_coordinates: int
    total_gradient_coordinates: int
    reconstructed_model_prediction: int
    reconstructed_model_confidence: float


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "nsl-kdd"],
        default="synthetic",
        help="Dataset-I synthetic or Dataset-II NSL-KDD.",
    )
    parser.add_argument("--target-client", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--attack-seed", type=int, default=123)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--attack-lr", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="attack_results",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print attack progress every N iterations. Set 0 to disable.",
    )


def load_target(
    dataset: str,
    target_client: int,
    num_clients: int,
    sample_index: int,
    model_seed: int,
):
    if target_client < 0 or target_client >= num_clients:
        raise ValueError("target_client must be in [0, num_clients-1]")

    # Seed before model construction so all three setup scripts use the same
    # initial model when invoked with identical arguments.
    seed_everything(model_seed)

    if dataset == "synthetic":
        from synthetic_common import Net, load_partition

        (x_train, y_train), _ = load_partition(target_client, num_clients)
        model = Net()
    else:
        from nsl_kdd_common import Net, load_partition

        (x_train, y_train), _, input_dim = load_partition(
            target_client, num_clients
        )
        model = Net(input_dim)

    if not 0 <= sample_index < len(x_train):
        raise IndexError(
            f"sample_index={sample_index} is outside the client training "
            f"partition of length {len(x_train)}"
        )

    x = torch.tensor(
        x_train[sample_index : sample_index + 1], dtype=torch.float32
    )
    y = torch.tensor(
        y_train[sample_index : sample_index + 1], dtype=torch.long
    )
    model.eval()
    return model, x, y


def flatten_tensors(tensors) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def compute_single_example_gradient(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    return flatten_tensors(gradients).detach()


def run_gradient_inversion(
    *,
    model: torch.nn.Module,
    observed_gradient: torch.Tensor,
    known_label: torch.Tensor,
    feature_dim: int,
    iterations: int,
    learning_rate: float,
    attack_seed: int,
    log_every: int = 100,
) -> Tuple[torch.Tensor, float]:
    """Reconstruct one tabular input by matching the observed gradient.

    The true class label is intentionally supplied to the attacker. This is a
    strong-attacker evaluation: differences are therefore driven by feature
    reconstruction from the observed gradient rather than by label inference.
    """

    seed_everything(attack_seed)
    dummy_x = torch.randn(
        (1, int(feature_dim)), dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([dummy_x], lr=float(learning_rate))
    target = observed_gradient.detach()

    final_match = float("nan")
    for step in range(1, int(iterations) + 1):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        logits = model(dummy_x)
        dummy_loss = F.cross_entropy(logits, known_label)
        dummy_gradients = torch.autograd.grad(
            dummy_loss,
            tuple(model.parameters()),
            create_graph=True,
        )
        dummy_flat = flatten_tensors(dummy_gradients)
        gradient_match = F.mse_loss(dummy_flat, target)
        gradient_match.backward()
        optimizer.step()

        final_match = float(gradient_match.detach().cpu().item())
        if log_every and (step == 1 or step % int(log_every) == 0):
            print(
                f"[attack] iteration={step:4d} "
                f"gradient_match_mse={final_match:.8e}"
            )

    return dummy_x.detach(), final_match


def reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    a = original.detach().cpu().numpy().reshape(-1).astype(np.float64)
    b = reconstructed.detach().cpu().numpy().reshape(-1).astype(np.float64)

    diff = a - b
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    l2 = float(np.linalg.norm(diff))

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > 0.0 else 0.0

    return {
        "reconstruction_mse": mse,
        "reconstruction_mae": mae,
        "reconstruction_l2": l2,
        "reconstruction_cosine_similarity": cosine,
    }


def save_attack_outputs(
    *,
    setup: str,
    args,
    model: torch.nn.Module,
    original_x: torch.Tensor,
    target_y: torch.Tensor,
    clean_gradient: torch.Tensor,
    observed_gradient: torch.Tensor,
    masking_vector: np.ndarray,
    selected_coordinates: int,
) -> AttackResult:
    reconstructed_x, final_match = run_gradient_inversion(
        model=model,
        observed_gradient=observed_gradient,
        known_label=target_y,
        feature_dim=original_x.shape[1],
        iterations=args.iterations,
        learning_rate=args.attack_lr,
        attack_seed=args.attack_seed,
        log_every=args.log_every,
    )

    metrics = reconstruction_metrics(original_x, reconstructed_x)

    with torch.no_grad():
        probabilities = torch.softmax(model(reconstructed_x), dim=1)
        confidence, prediction = probabilities.max(dim=1)

    clean_norm = float(torch.linalg.vector_norm(clean_gradient).cpu().item())
    observed_norm = float(
        torch.linalg.vector_norm(observed_gradient).cpu().item()
    )
    mask_norm = float(np.linalg.norm(masking_vector))
    ratio = mask_norm / clean_norm if clean_norm > 0.0 else float("inf")

    result = AttackResult(
        setup=setup,
        dataset=args.dataset,
        target_client=int(args.target_client),
        sample_index=int(args.sample_index),
        target_label=int(target_y.item()),
        num_clients=int(args.num_clients),
        round_id=int(args.round_id),
        iterations=int(args.iterations),
        attack_learning_rate=float(args.attack_lr),
        final_gradient_match_mse=float(final_match),
        clean_gradient_norm=clean_norm,
        observed_gradient_norm=observed_norm,
        masking_norm=mask_norm,
        mask_to_clean_gradient_ratio=float(ratio),
        selected_masking_coordinates=int(selected_coordinates),
        total_gradient_coordinates=int(clean_gradient.numel()),
        reconstructed_model_prediction=int(prediction.item()),
        reconstructed_model_confidence=float(confidence.item()),
        **metrics,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stem = (
        f"{setup}_{args.dataset}_client{args.target_client}_"
        f"sample{args.sample_index}_round{args.round_id}"
    )

    payload = {
        "result": asdict(result),
        "original_feature_vector": original_x.cpu().numpy().reshape(-1).tolist(),
        "reconstructed_feature_vector": reconstructed_x.cpu().numpy().reshape(-1).tolist(),
        "masking_vector": np.asarray(masking_vector, dtype=np.float64).tolist(),
    }
    json_path = os.path.join(args.output_dir, stem + ".json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = os.path.join(args.output_dir, stem + "_features.csv")
    feature_table = np.column_stack(
        [
            original_x.cpu().numpy().reshape(-1),
            reconstructed_x.cpu().numpy().reshape(-1),
        ]
    )
    np.savetxt(
        csv_path,
        feature_table,
        delimiter=",",
        header="original,reconstructed",
        comments="",
    )

    print("\n=== Attack result ===")
    for key, value in asdict(result).items():
        print(f"{key}: {value}")
    print(f"JSON: {json_path}")
    print(f"Features: {csv_path}")

    return result


def prepare_attack(args):
    model, original_x, target_y = load_target(
        args.dataset,
        args.target_client,
        args.num_clients,
        args.sample_index,
        args.model_seed,
    )
    clean_gradient = compute_single_example_gradient(
        model, original_x, target_y
    )
    return model, original_x, target_y, clean_gradient
