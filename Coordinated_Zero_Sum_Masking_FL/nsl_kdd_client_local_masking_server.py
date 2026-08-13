"""Dataset-II (NSL-KDD) server for the independent Client-Local Masking baseline."""

import argparse
import json
import os
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


LOG_DIR = "/home/ripon/code/client_local_masking_nsl_kdd_logs"
METRICS_FILE = os.path.join(LOG_DIR, "server_metrics.json")


class FedAvgClientLocalMasking(FedAvg):
    def __init__(
        self,
        *args,
        masking_coefficient: float = 5.0,
        masking_probability: float = 0.6,
        min_masked_coordinates: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masking_coefficient = float(masking_coefficient)
        self.masking_probability = float(masking_probability)
        self.min_masked_coordinates = int(min_masked_coordinates)
        self.global_flat_by_round = {}
        os.makedirs(LOG_DIR, exist_ok=True)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self.global_flat_by_round[server_round] = np.concatenate(
            [arr.reshape(-1) for arr in parameters_to_ndarrays(parameters)]
        ).astype(np.float64)

        instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        revised = []
        for client_proxy, fitins in instructions:
            cfg: Dict[str, Scalar] = dict(fitins.config or {})
            cfg.update(
                {
                    "server_round": int(server_round),
                    "masking_coefficient": self.masking_coefficient,
                    "masking_probability": self.masking_probability,
                    "min_masked_coordinates": self.min_masked_coordinates,
                }
            )
            revised.append(
                (client_proxy, FitIns(fitins.parameters, cfg))
            )
        return revised

    def aggregate_fit(self, server_round, results, failures):
        clean_vectors = []
        masked_vectors = []
        client_metrics = {}

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            pid = metrics.get("partition_id")
            if pid is None:
                continue

            clean_path = f"{LOG_DIR}/client_{pid}_clean_update.json"
            if os.path.exists(clean_path):
                with open(clean_path, "r") as handle:
                    clean_vectors.append(
                        np.asarray(json.load(handle), dtype=np.float64)
                    )

            received_flat = np.concatenate(
                [
                    arr.reshape(-1)
                    for arr in parameters_to_ndarrays(
                        fit_res.parameters
                    )
                ]
            ).astype(np.float64)
            global_flat = self.global_flat_by_round.get(server_round)
            if global_flat is not None and global_flat.shape == received_flat.shape:
                masked_vectors.append(received_flat - global_flat)
            else:
                masked_vectors.append(received_flat)
            client_metrics[str(pid)] = {
                "masking_sum": float(
                    metrics.get("masking_sum", 0.0)
                ),
                "masking_norm": float(
                    metrics.get("masking_norm", 0.0)
                ),
                "masked_coordinates": int(
                    metrics.get("masked_coordinates", 0)
                ),
                "total_coordinates": int(
                    metrics.get("total_coordinates", 0)
                ),
            }

        aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        avg_absolute_diff = None
        if clean_vectors and len(clean_vectors) == len(masked_vectors):
            clean_avg = np.mean(np.stack(clean_vectors), axis=0)
            masked_avg = np.mean(np.stack(masked_vectors), axis=0)
            avg_absolute_diff = float(
                np.mean(np.abs(clean_avg - masked_avg))
            )

        log = self._load_log()
        log.append(
            {
                "round": int(server_round),
                "fit": {
                    "num_successful_clients": int(len(results)),
                    "num_failed_clients": int(len(failures)),
                    "avg_absolute_clean_vs_masked_diff": avg_absolute_diff,
                    "client_masking": client_metrics,
                },
                "configuration": {
                    "masking_coefficient": self.masking_coefficient,
                    "masking_probability": self.masking_probability,
                    "min_masked_coordinates": self.min_masked_coordinates,
                    "formal_dp_claim": False,
                },
            }
        )
        self._save_log(log)
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        if not results:
            return aggregated

        total_examples = sum(int(res.num_examples) for _, res in results)
        weighted_loss = sum(
            float(res.loss) * int(res.num_examples)
            for _, res in results
        ) / total_examples
        weighted_accuracy = sum(
            float((res.metrics or {}).get("accuracy", 0.0))
            * int(res.num_examples)
            for _, res in results
        ) / total_examples

        log = self._load_log()
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
        self._save_log(log)
        return aggregated

    @staticmethod
    def _load_log():
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r") as handle:
                return json.load(handle)
        return []

    @staticmethod
    def _save_log(log):
        with open(METRICS_FILE, "w") as handle:
            json.dump(log, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument(
        "--masking-coefficient", type=float, default=5.0
    )
    parser.add_argument(
        "--masking-probability", type=float, default=0.6
    )
    parser.add_argument(
        "--min-masked-coordinates", type=int, default=2
    )
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(METRICS_FILE, "w") as handle:
        json.dump([], handle)

    strategy = FedAvgClientLocalMasking(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        accept_failures=False,
        masking_coefficient=args.masking_coefficient,
        masking_probability=args.masking_probability,
        min_masked_coordinates=args.min_masked_coordinates,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
