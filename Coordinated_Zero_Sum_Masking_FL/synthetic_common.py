"""Shared utilities for the synthetic-data experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


SEED = 42
SAMPLES_PER_CLIENT = 300
LOCAL_EPOCHS = 200
LEARNING_RATE = 0.001

np.random.seed(SEED)
torch.manual_seed(SEED)


class Net(nn.Module):
    """FCNN used for Dataset-I: 10 -> 16 -> 2."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for val in net.parameters()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    for param, new_val in zip(net.parameters(), parameters):
        param.data = torch.tensor(new_val, dtype=param.data.dtype)


def flatten_parameters(parameters: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(p).reshape(-1) for p in parameters]).astype(np.float64)


def reshape_like(flat_vector: np.ndarray, reference_parameters: List[np.ndarray]) -> List[np.ndarray]:
    shaped: List[np.ndarray] = []
    idx = 0
    for p in reference_parameters:
        size = p.size
        shaped.append(flat_vector[idx : idx + size].reshape(p.shape))
        idx += size
    if idx != flat_vector.size:
        raise ValueError("Flat vector length does not match model parameter shapes")
    return shaped


def train_model(
    net: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = LOCAL_EPOCHS,
    lr: float = LEARNING_RATE,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    net.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        outputs = net(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        outputs = net(X_tensor)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_tensor).float().mean().item()

    return float(loss.item()), float(accuracy)


def evaluate_model(net: nn.Module, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pure evaluation; the test set is never used for an optimization step."""

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    net.eval()
    with torch.no_grad():
        outputs = net(X_tensor)
        loss = F.cross_entropy(outputs, y_tensor).item()
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_tensor).float().mean().item()

    return float(loss), float(accuracy)


def generate_global_dataset(num_clients: int) -> Tuple[np.ndarray, np.ndarray]:
    total_samples = SAMPLES_PER_CLIENT * int(num_clients)
    X, y = make_classification(
        n_samples=total_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=SEED,
    )
    return X.astype(np.float32), y.astype(np.int64)


def load_partition(
    partition_id: int,
    num_partitions: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    X, y = generate_global_dataset(num_partitions)

    start_idx = int(partition_id) * SAMPLES_PER_CLIENT
    end_idx = (int(partition_id) + 1) * SAMPLES_PER_CLIENT
    client_X, client_y = X[start_idx:end_idx], y[start_idx:end_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        client_X,
        client_y,
        test_size=0.30,
        random_state=SEED,
    )
    return (X_train, y_train), (X_test, y_test)


def diagnostic_metrics(
    *,
    global_parameters: List[np.ndarray],
    local_parameters: List[np.ndarray],
    masking_vector: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, float]:
    global_flat = flatten_parameters(global_parameters)
    local_flat = flatten_parameters(local_parameters)
    update_vector = local_flat - global_flat
    masking_vector = np.asarray(masking_vector, dtype=np.float64)

    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)

    return {
        "update_norm": float(np.linalg.norm(update_vector)),
        "param_norm": float(np.linalg.norm(local_flat)),
        "masking_norm": float(np.linalg.norm(masking_vector)),
        "masking_sum": float(np.sum(masking_vector)),
        "num_samples": int(len(y_train)),
        "class_0_count": int(counts[0]),
        "class_1_count": int(counts[1]),
    }
