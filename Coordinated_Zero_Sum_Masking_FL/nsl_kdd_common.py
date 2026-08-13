"""Shared NSL-KDD utilities for the experiments."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
LOCAL_EPOCHS = 200
LEARNING_RATE = 0.001

np.random.seed(SEED)
torch.manual_seed(SEED)

NSLKDD_DIR = os.environ.get("NSLKDD_DIR", "/home/ripon/code")
TRAIN_FILE = os.path.join(NSLKDD_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(NSLKDD_DIR, "KDDTest+.txt")

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
CAT_IDX = [FEATURE_NAMES.index(c) for c in CATEGORICAL_COLS]
NUM_IDX = [i for i in range(len(FEATURE_NAMES)) if i not in CAT_IDX]

_DATA_CACHE = {"X": None, "y": None, "input_dim": None}


class Net(nn.Module):
    """FCNN used for Dataset-II: d_in -> 16 -> 2."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(int(input_dim), 16)
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
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    net.eval()
    with torch.no_grad():
        outputs = net(X_tensor)
        loss = F.cross_entropy(outputs, y_tensor).item()
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_tensor).float().mean().item()

    return float(loss), float(accuracy)


def _read_kdd_file(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing NSL-KDD file: {path}. Set NSLKDD_DIR to the directory "
            "containing KDDTrain+.txt and KDDTest+.txt."
        )

    rows = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(line.split(","))
    return np.asarray(rows, dtype=object)


def _split_features_and_label(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if raw.shape[1] not in (42, 43):
        raise ValueError(f"Unexpected NSL-KDD column count: {raw.shape[1]}")
    return raw[:, :41], raw[:, 41]


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_nsl_kdd_binary() -> Tuple[np.ndarray, np.ndarray, int]:
    """Preprocess using KDDTrain+ fitted transformations, then form the experiment pool.

    The scaler and one-hot encoder are fitted only on KDDTrain+.  The transformed
    KDDTrain+ and KDDTest+ records are then concatenated because the manuscript's
    experimental design partitions the processed pool across clients and performs a
    local stratified 70/30 split.
    """

    if _DATA_CACHE["X"] is not None:
        return (
            _DATA_CACHE["X"],
            _DATA_CACHE["y"],
            int(_DATA_CACHE["input_dim"]),
        )

    train_raw = _read_kdd_file(TRAIN_FILE)
    test_raw = _read_kdd_file(TEST_FILE)

    train_features, train_labels = _split_features_and_label(train_raw)
    test_features, test_labels = _split_features_and_label(test_raw)

    y_train = (train_labels != "normal").astype(np.int64)
    y_test = (test_labels != "normal").astype(np.int64)

    X_train_cat = train_features[:, CAT_IDX]
    X_test_cat = test_features[:, CAT_IDX]
    X_train_num = train_features[:, NUM_IDX].astype(np.float64)
    X_test_num = test_features[:, NUM_IDX].astype(np.float64)

    ohe = _make_ohe()
    ohe.fit(X_train_cat)

    scaler = StandardScaler()
    scaler.fit(X_train_num)

    X_train = np.hstack(
        [scaler.transform(X_train_num), ohe.transform(X_train_cat)]
    ).astype(np.float32)
    X_test = np.hstack(
        [scaler.transform(X_test_num), ohe.transform(X_test_cat)]
    ).astype(np.float32)

    X = np.vstack([X_train, X_test]).astype(np.float32)
    y = np.concatenate([y_train, y_test]).astype(np.int64)

    _DATA_CACHE["X"] = X
    _DATA_CACHE["y"] = y
    _DATA_CACHE["input_dim"] = int(X.shape[1])
    return X, y, int(X.shape[1])


def load_partition(
    partition_id: int,
    num_partitions: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    X, y, input_dim = load_nsl_kdd_binary()

    total = (len(X) // int(num_partitions)) * int(num_partitions)
    X, y = X[:total], y[:total]

    samples_per_client = total // int(num_partitions)
    start_idx = int(partition_id) * samples_per_client
    end_idx = (int(partition_id) + 1) * samples_per_client

    client_X, client_y = X[start_idx:end_idx], y[start_idx:end_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        client_X,
        client_y,
        test_size=0.30,
        random_state=SEED,
        stratify=client_y,
    )
    return (X_train, y_train), (X_test, y_test), input_dim


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
