from __future__ import annotations

import json
import os
from typing import List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.client import NumPyClient
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

SERVER_ADDRESS = os.getenv(
    "SERVER_ADDRESS",
    "127.0.0.1:8080",
)

KDD_TRAIN_PATH = os.getenv(
    "KDD_TRAIN_PATH",
    "/home/ripon/code/agentfednoise/KDDTrain+.txt",
)

KDD_TEST_PATH = os.getenv(
    "KDD_TEST_PATH",
    "/home/ripon/code/agentfednoise/KDDTest+.txt",
)

HIDDEN_DIM = 16
NUM_CLASSES = 2

TEST_SIZE = 0.30
RANDOM_STATE = 42

# Journal-paper setup.
LOCAL_EPOCHS = 200
LEARNING_RATE = 0.001

NSL_KDD_FEATURE_COUNT = 41
CATEGORICAL_INDICES = [1, 2, 3]
NUMERIC_INDICES = [
    idx
    for idx in range(NSL_KDD_FEATURE_COUNT)
    if idx not in CATEGORICAL_INDICES
]

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def _make_one_hot_encoder():
    """
    Compatibility helper for different scikit-learn versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def _read_nsl_kdd_file(
    path: str,
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NSL-KDD file not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        header=None,
    )

    # KDDTrain+/KDDTest+ normally contain:
    # 41 features + label + difficulty = 43 columns.
    if dataframe.shape[1] < 42:
        raise ValueError(
            f"Expected at least 42 columns in {path}, "
            f"found {dataframe.shape[1]}."
        )

    return dataframe


def _binary_labels(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Map normal -> 0 and every attack type -> 1.

    Handles both "normal" and "normal." labels.
    """
    labels = (
        dataframe.iloc[:, 41]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
        .str.lower()
    )

    return np.where(
        labels == "normal",
        0,
        1,
    ).astype(np.int64)


def load_processed_global_dataset(
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load KDDTrain+ and KDDTest+.

    Preprocessing is fitted ONLY on KDDTrain+:
      - 38 numerical features: StandardScaler
      - 3 categorical features: OneHotEncoder

    The transformed KDDTrain+ and KDDTest+ arrays are concatenated
    and then partitioned equally across the federated clients.
    """
    train_df = _read_nsl_kdd_file(
        KDD_TRAIN_PATH
    )

    test_df = _read_nsl_kdd_file(
        KDD_TEST_PATH
    )

    X_train_raw = train_df.iloc[
        :,
        :NSL_KDD_FEATURE_COUNT,
    ]

    X_test_raw = test_df.iloc[
        :,
        :NSL_KDD_FEATURE_COUNT,
    ]

    y_train_source = _binary_labels(
        train_df
    )

    y_test_source = _binary_labels(
        test_df
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_INDICES,
            ),
            (
                "categorical",
                _make_one_hot_encoder(),
                CATEGORICAL_INDICES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    # Fit preprocessing on KDDTrain+ only.
    X_train_processed = (
        preprocessor.fit_transform(
            X_train_raw
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test_raw
        )
    )

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32,
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float32,
    )

    X = np.concatenate(
        [
            X_train_processed,
            X_test_processed,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            y_train_source,
            y_test_source,
        ],
        axis=0,
    )

    return X, y


def load_partition(
    partition_id: int,
    num_partitions: int,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    if num_partitions < 1:
        raise ValueError(
            "num_partitions must be at least 1."
        )

    if (
        partition_id < 0
        or partition_id >= num_partitions
    ):
        raise ValueError(
            f"partition_id must be in "
            f"[0, {num_partitions - 1}], "
            f"got {partition_id}."
        )

    X, y = load_processed_global_dataset()

    # Equal contiguous partitions after truncation.
    samples_per_client = (
        len(X) // num_partitions
    )

    usable_samples = (
        samples_per_client
        * num_partitions
    )

    X = X[:usable_samples]
    y = y[:usable_samples]

    start_idx = (
        partition_id
        * samples_per_client
    )

    end_idx = (
        (partition_id + 1)
        * samples_per_client
    )

    client_X = X[
        start_idx:end_idx
    ]

    client_y = y[
        start_idx:end_idx
    ]

    # Each client uses the same stratified 70/30 split.
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        client_X,
        client_y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=client_y,
    )

    return (
        (X_train, y_train),
        (X_test, y_test),
    )


# ============================================================
# MODEL
# ============================================================

class Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            HIDDEN_DIM,
        )

        self.fc2 = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES,
        )

    def forward(self, x):
        x = F.relu(
            self.fc1(x)
        )

        return self.fc2(x)


# ============================================================
# PARAMETER AND METRIC HELPERS
# ============================================================

def get_parameters(
    net: nn.Module,
) -> List[np.ndarray]:
    return [
        value.detach().cpu().numpy().copy()
        for value in net.parameters()
    ]


def set_parameters(
    net: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    current_parameters = list(
        net.parameters()
    )

    if len(current_parameters) != len(parameters):
        raise ValueError(
            "Received parameter list length does not match model."
        )

    for parameter, new_value in zip(
        current_parameters,
        parameters,
    ):
        tensor = torch.as_tensor(
            new_value,
            dtype=parameter.data.dtype,
            device=parameter.data.device,
        )

        if tuple(tensor.shape) != tuple(
            parameter.data.shape
        ):
            raise ValueError(
                "Received parameter shape does not match model: "
                f"received={tuple(tensor.shape)}, "
                f"expected={tuple(parameter.data.shape)}"
            )

        parameter.data.copy_(tensor)


def flatten_parameters(
    parameters: List[np.ndarray],
) -> np.ndarray:
    if not parameters:
        return np.zeros(
            0,
            dtype=np.float64,
        )

    return np.concatenate(
        [
            np.asarray(
                parameter,
                dtype=np.float64,
            ).ravel()
            for parameter in parameters
        ]
    )


def reshape_flattened_vector(
    flat_vector: np.ndarray,
    reference_params: List[np.ndarray],
) -> List[np.ndarray]:
    reshaped: List[np.ndarray] = []
    index = 0

    for reference in reference_params:
        size = reference.size

        piece = flat_vector[
            index:index + size
        ]

        if piece.size != size:
            raise ValueError(
                "Flattened vector ended before all model "
                "parameters were reconstructed."
            )

        reshaped.append(
            piece.reshape(
                reference.shape
            ).astype(
                reference.dtype,
                copy=False,
            )
        )

        index += size

    if index != flat_vector.size:
        raise ValueError(
            "Masking vector size does not match "
            "the model parameter size: "
            f"used {index}, received "
            f"{flat_vector.size}."
        )

    return reshaped


def subtract_parameters(
    left: List[np.ndarray],
    right: List[np.ndarray],
) -> List[np.ndarray]:
    if len(left) != len(right):
        raise ValueError(
            "Parameter lists have different lengths."
        )

    return [
        np.asarray(l) - np.asarray(r)
        for l, r in zip(
            left,
            right,
        )
    ]


def add_parameters(
    left: List[np.ndarray],
    right: List[np.ndarray],
) -> List[np.ndarray]:
    if len(left) != len(right):
        raise ValueError(
            "Parameter lists have different lengths."
        )

    return [
        np.asarray(l) + np.asarray(r)
        for l, r in zip(
            left,
            right,
        )
    ]


def vector_l2_norm(
    values: List[np.ndarray],
) -> float:
    total = 0.0

    for value in values:
        array = np.asarray(
            value,
            dtype=np.float64,
        )

        total += float(
            np.sum(array ** 2)
        )

    return float(
        np.sqrt(total)
    )


def class_count(
    y: np.ndarray,
    class_id: int,
) -> int:
    return int(
        np.sum(y == class_id)
    )


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train(
    net: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = LOCAL_EPOCHS,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LEARNING_RATE,
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    net.train()
    last_loss = 0.0

    for _ in range(epochs):
        optimizer.zero_grad()

        outputs = net(
            X_tensor
        )

        loss = criterion(
            outputs,
            y_tensor,
        )

        loss.backward()
        optimizer.step()

        last_loss = float(
            loss.item()
        )

    # Report the local training state after the final epoch.
    net.eval()

    with torch.no_grad():
        outputs = net(
            X_tensor
        )

        final_loss = criterion(
            outputs,
            y_tensor,
        ).item()

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        accuracy = (
            predictions == y_tensor
        ).float().mean().item()

    return (
        float(final_loss),
        float(accuracy),
    )


def test(
    net: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    net.eval()

    with torch.no_grad():
        outputs = net(
            X_tensor
        )

        loss = criterion(
            outputs,
            y_tensor,
        ).item()

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        accuracy = (
            predictions == y_tensor
        ).float().mean().item()

    return (
        float(loss),
        float(accuracy),
    )


# ============================================================
# FLOWER CLIENT
# ============================================================

class CoordinatedMaskingDiagnosticClient(
    NumPyClient
):
    def __init__(
        self,
        model: nn.Module,
        train_data: Tuple[
            np.ndarray,
            np.ndarray,
        ],
        test_data: Tuple[
            np.ndarray,
            np.ndarray,
        ],
        partition_id: int,
    ):
        self.net = model

        self.X_train, self.y_train = (
            train_data
        )

        self.X_test, self.y_test = (
            test_data
        )

        self.partition_id = int(
            partition_id
        )

    def get_parameters(
        self,
        config,
    ):
        return get_parameters(
            self.net
        )

    def fit(
        self,
        parameters,
        config,
    ):
        # Incoming global model.
        set_parameters(
            self.net,
            parameters,
        )

        global_params = [
            parameter.copy()
            for parameter in get_parameters(
                self.net
            )
        ]

        train_loss, train_accuracy = train(
            self.net,
            self.X_train,
            self.y_train,
            epochs=LOCAL_EPOCHS,
        )

        # Clean locally trained model.
        local_params = get_parameters(
            self.net
        )

        # Clean local update:
        # Delta w_i = w_i(local) - w(global)
        clean_update = subtract_parameters(
            local_params,
            global_params,
        )

        clean_update_flat = flatten_parameters(
            clean_update
        )

        # The revised server sends only the revised terminology.
        masking_coefficient = float(
            config.get(
                "masking_coefficient",
                5.0,
            )
        )

        if "masking_vector" not in config:
            raise ValueError(
                "masking_vector is missing from the "
                "server fit configuration."
            )

        masking_vector = np.asarray(
            json.loads(
                config["masking_vector"]
            ),
            dtype=np.float64,
        )

        if (
            masking_vector.size
            != clean_update_flat.size
        ):
            raise ValueError(
                "Received masking vector has incorrect size: "
                f"{masking_vector.size}; expected "
                f"{clean_update_flat.size}."
            )

        # Mask the UPDATE explicitly:
        # Delta w_tilde_i = Delta w_i + eta_i
        masked_update_flat = (
            clean_update_flat
            + masking_vector
        )

        masked_update = (
            reshape_flattened_vector(
                masked_update_flat,
                clean_update,
            )
        )

        # Flower FedAvg aggregates model parameters rather than
        # explicit update vectors. Therefore transmit:
        #
        # w_global + Delta w_tilde_i
        #
        # which is algebraically equal to:
        # w_local + eta_i
        masked_model_params = add_parameters(
            global_params,
            masked_update,
        )

        param_norm = vector_l2_norm(
            local_params
        )

        update_norm = vector_l2_norm(
            clean_update
        )

        mask_norm = float(
            np.linalg.norm(
                masking_vector
            )
        )

        metrics = {
            "partition_id": int(
                self.partition_id
            ),
            "loss": float(
                train_loss
            ),
            "accuracy": float(
                train_accuracy
            ),
            "update_norm": float(
                update_norm
            ),
            "param_norm": float(
                param_norm
            ),
            "masking_norm": float(
                mask_norm
            ),
            "num_samples": int(
                len(self.X_train)
            ),
            "class_0_count": class_count(
                self.y_train,
                0,
            ),
            "class_1_count": class_count(
                self.y_train,
                1,
            ),
            "used_masking_coefficient": float(
                masking_coefficient
            ),
            "local_epochs": int(
                LOCAL_EPOCHS
            ),
        }

        return (
            masked_model_params,
            len(self.X_train),
            metrics,
        )

    def evaluate(
        self,
        parameters,
        config,
    ):
        set_parameters(
            self.net,
            parameters,
        )

        loss, accuracy = test(
            self.net,
            self.X_test,
            self.y_test,
        )

        return (
            float(loss),
            len(self.X_test),
            {
                "accuracy": float(
                    accuracy
                )
            },
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "NSL-KDD client for coordinated zero-sum "
            "masking with post-round LLM diagnostics."
        )
    )

    parser.add_argument(
        "--partition-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-partitions",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    train_data, test_data = (
        load_partition(
            partition_id=args.partition_id,
            num_partitions=args.num_partitions,
        )
    )

    input_dim = int(
        train_data[0].shape[1]
    )

    model = Net(
        input_dim=input_dim
    )

    print(
        f"[Client {args.partition_id}] "
        f"input_dim={input_dim}, "
        f"local_epochs={LOCAL_EPOCHS}, "
        f"train_samples={len(train_data[0])}, "
        f"test_samples={len(test_data[0])}, "
        f"class_0_train="
        f"{class_count(train_data[1], 0)}, "
        f"class_1_train="
        f"{class_count(train_data[1], 1)}"
    )

    client = (
        CoordinatedMaskingDiagnosticClient(
            model=model,
            train_data=train_data,
            test_data=test_data,
            partition_id=args.partition_id,
        )
    )

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )