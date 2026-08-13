"""Dataset-II (NSL-KDD) CZSM client with simulated failure support."""

import argparse
import json
import os

import flwr as fl
import numpy as np
from flwr.client import NumPyClient

from nsl_kdd_common import (
    Net,
    diagnostic_metrics,
    evaluate_model,
    flatten_parameters,
    get_parameters,
    load_partition,
    reshape_like,
    set_parameters,
    train_model,
)


LOG_DIR = "/home/ripon/code/czsm_nsl_kdd_crash_logs"


class FlowerClient(NumPyClient):
    def __init__(self, model, train_data, test_data, partition_id: int):
        self.net = model
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.partition_id = int(partition_id)
        os.makedirs(LOG_DIR, exist_ok=True)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def _maybe_crash(self, config, phase: str) -> None:
        server_round = int(config.get("server_round", -1))
        if int(config.get("crash_this_client", 0)) != 1:
            return
        crash_round = int(config.get("crash_on_round", server_round))
        crash_phase = str(config.get("crash_phase", "fit"))
        if server_round == crash_round and phase == crash_phase:
            print(
                f"[Client {self.partition_id}] simulated {phase} "
                f"failure at round {server_round}"
            )
            os._exit(1)

    def fit(self, parameters, config):
        self._maybe_crash(config, "fit")

        global_parameters = [np.asarray(p) for p in parameters]
        set_parameters(self.net, parameters)

        # Revised journal setting: 200 local epochs per round.
        train_loss, train_accuracy = train_model(
            self.net, self.X_train, self.y_train
        )
        test_loss, test_accuracy = evaluate_model(
            self.net, self.X_test, self.y_test
        )

        local_parameters = get_parameters(self.net)
        flat_global = flatten_parameters(global_parameters)
        flat_local = flatten_parameters(local_parameters)
        clean_update = flat_local - flat_global

        with open(
            f"{LOG_DIR}/client_{self.partition_id}_clean_update.json", "w"
        ) as handle:
            json.dump(clean_update.tolist(), handle)

        masking_vector = None
        if "masking_vector" in config:
            try:
                masking_vector = np.asarray(
                    json.loads(config["masking_vector"]),
                    dtype=np.float64,
                )
            except Exception:
                masking_vector = None

        if (
            masking_vector is None
            or masking_vector.shape != flat_local.shape
        ):
            masking_vector = np.zeros_like(flat_local)

        masked_local = flat_local + masking_vector
        diagnostics = diagnostic_metrics(
            global_parameters=global_parameters,
            local_parameters=local_parameters,
            masking_vector=masking_vector,
            y_train=self.y_train,
        )

        metrics = {
            "partition_id": self.partition_id,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "accuracy": float(train_accuracy),
            "loss": float(train_loss),
            "masking_coefficient": float(
                config.get("masking_coefficient", 5.0)
            ),
            "masking_probability": float(
                config.get("masking_probability", 1.0)
            ),
            **diagnostics,
        }
        return reshape_like(
            masked_local, local_parameters
        ), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        self._maybe_crash(config, "eval")
        set_parameters(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.X_test, self.y_test)
        return float(loss), len(self.X_test), {
            "accuracy": float(accuracy),
            "partition_id": self.partition_id,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--num-partitions", type=int, default=10)
    args = parser.parse_args()

    train_data, test_data, input_dim = load_partition(
        args.partition_id, args.num_partitions
    )
    client = FlowerClient(
        Net(input_dim), train_data, test_data, args.partition_id
    )
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )
