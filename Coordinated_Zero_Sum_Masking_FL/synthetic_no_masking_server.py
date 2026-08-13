"""Dataset-I server for the No Masking reference condition."""

import argparse
import json
import os
from typing import List

import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg


LOG_DIR = "/home/ripon/code/no_masking_synthetic_logs"
METRICS_FILE = os.path.join(LOG_DIR, "server_metrics.json")


def load_log() -> List[dict]:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as handle:
            return json.load(handle)
    return []


def save_log(log: List[dict]) -> None:
    tmp = METRICS_FILE + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(log, handle, indent=2)
    os.replace(tmp, METRICS_FILE)


class FedAvgNoMaskingLogging(FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        if not results:
            return aggregated

        total_examples = sum(int(res.num_examples) for _, res in results)
        weighted_loss = (
            sum(float(res.loss) * int(res.num_examples) for _, res in results)
            / total_examples
        )
        weighted_accuracy = (
            sum(
                float((res.metrics or {}).get("accuracy", 0.0))
                * int(res.num_examples)
                for _, res in results
            )
            / total_examples
        )

        log = load_log()
        log.append(
            {
                "round": int(server_round),
                "global_eval": {
                    "weighted_loss": float(weighted_loss),
                    "weighted_accuracy": float(weighted_accuracy),
                    "num_clients": int(len(results)),
                },
            }
        )
        save_log(log)

        print(
            f"[Server] R{server_round} no masking | "
            f"accuracy={weighted_accuracy:.4f} | "
            f"loss={weighted_loss:.4f}"
        )
        return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    save_log([])

    strategy = FedAvgNoMaskingLogging(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        accept_failures=False,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
