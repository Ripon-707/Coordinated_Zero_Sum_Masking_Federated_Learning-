"""Dataset-II (NSL-KDD) server for Coordinated Zero-Sum Masking (CZSM)."""

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

from masking_coordinator import (
    CoordinatedZeroSumMaskingCoordinator,
    aggregate_masking_residual,
)


LOG_DIR = "/home/ripon/code/czsm_nsl_kdd_logs"
METRICS_FILE = os.path.join(LOG_DIR, "server_metrics.json")


def parameter_vector_length(parameters: Parameters) -> int:
    return int(
        sum(arr.size for arr in parameters_to_ndarrays(parameters))
    )


class FedAvgCoordinatedZeroSumMasking(FedAvg):
    """FedAvg with an emulated non-colluding masking-coordinator interface."""

    def __init__(
        self,
        *args,
        masking_coefficient: float = 5.0,
        masking_probability: float = 1.0,
        min_masked_coordinates: int = 2,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masking_coefficient = float(masking_coefficient)
        self.masking_probability = float(masking_probability)
        self.min_masked_coordinates = int(min_masked_coordinates)
        self.coordinator = CoordinatedZeroSumMaskingCoordinator(
            seed=seed
        )
        self.round_assignments = {}
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
        if not instructions:
            return instructions

        client_ids = [proxy.cid for proxy, _ in instructions]
        assignment = self.coordinator.generate(
            round_id=server_round,
            num_coordinates=parameter_vector_length(parameters),
            client_ids=client_ids,
            mu=self.masking_coefficient,
            masking_probability=self.masking_probability,
            min_masked_coordinates=self.min_masked_coordinates,
        )
        self.round_assignments[server_round] = assignment

        revised = []
        for proxy, fitins in instructions:
            vector = assignment.column_for(proxy.cid)
            cfg: Dict[str, Scalar] = dict(fitins.config or {})
            cfg.update(
                {
                    "server_round": int(server_round),
                    "masking_vector": json.dumps(vector.tolist()),
                    "masking_coefficient": self.masking_coefficient,
                    "masking_probability": self.masking_probability,
                    "masked_coordinates": int(
                        assignment.coordinate_selection.sum()
                    ),
                }
            )
            revised.append(
                (proxy, FitIns(fitins.parameters, cfg))
            )
        return revised

    def aggregate_fit(self, server_round, results, failures):
        clean_by_cid = {}
        masked_by_cid = {}
        examples_by_cid = {}

        for proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            pid = metrics.get("partition_id")
            if pid is not None:
                path = f"{LOG_DIR}/client_{pid}_clean_update.json"
                if os.path.exists(path):
                    with open(path, "r") as handle:
                        clean_by_cid[proxy.cid] = np.asarray(
                            json.load(handle), dtype=np.float64
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
                masked_by_cid[proxy.cid] = received_flat - global_flat
            else:
                masked_by_cid[proxy.cid] = received_flat
            examples_by_cid[proxy.cid] = int(fit_res.num_examples)

        aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        successful_cids = [
            proxy.cid for proxy, _ in results
        ]
        assignment = self.round_assignments.get(server_round)

        masking_residual_l2 = None
        equal_weight_residual_l2 = None
        if assignment is not None and successful_cids:
            masks = [
                assignment.column_for(cid)
                for cid in successful_cids
            ]
            equal_residual = aggregate_masking_residual(masks)
            equal_weight_residual_l2 = float(
                np.linalg.norm(equal_residual)
            )

            weights = [
                examples_by_cid[cid]
                for cid in successful_cids
            ]
            weighted_residual = aggregate_masking_residual(
                masks, weights=weights
            )
            masking_residual_l2 = float(
                np.linalg.norm(weighted_residual)
            )

        clean_vs_masked_diff = None
        clean_vs_aggregated_diff = None
        common = [
            cid
            for cid in successful_cids
            if cid in clean_by_cid and cid in masked_by_cid
        ]
        if common:
            clean_avg = np.mean(
                np.stack([clean_by_cid[cid] for cid in common]),
                axis=0,
            )
            masked_avg = np.mean(
                np.stack([masked_by_cid[cid] for cid in common]),
                axis=0,
            )
            clean_vs_masked_diff = float(
                np.mean(np.abs(clean_avg - masked_avg))
            )

            if aggregated is not None:
                aggregated_parameters = aggregated[0]
                aggregated_flat = np.concatenate(
                    [
                        arr.reshape(-1)
                        for arr in parameters_to_ndarrays(
                            aggregated_parameters
                        )
                    ]
                ).astype(np.float64)
                global_flat = self.global_flat_by_round.get(server_round)
                if (
                    global_flat is not None
                    and global_flat.shape == aggregated_flat.shape
                    and clean_avg.shape == aggregated_flat.shape
                ):
                    aggregated_update = aggregated_flat - global_flat
                    clean_vs_aggregated_diff = float(
                        np.mean(
                            np.abs(clean_avg - aggregated_update)
                        )
                    )

        log = self._load_log()
        log.append(
            {
                "round": int(server_round),
                "fit": {
                    "num_successful_clients": int(len(results)),
                    "num_failed_clients": int(len(failures)),
                    "clean_vs_masked_mean_absolute_diff": (
                        clean_vs_masked_diff
                    ),
                    "clean_vs_aggregated_mean_absolute_diff": (
                        clean_vs_aggregated_diff
                    ),
                    "equal_weight_masking_residual_l2": (
                        equal_weight_residual_l2
                    ),
                    "fedavg_weighted_masking_residual_l2": (
                        masking_residual_l2
                    ),
                },
                "configuration": {
                    "masking_coefficient": self.masking_coefficient,
                    "preliminary_laplace_scale": (
                        1.0 / self.masking_coefficient
                    ),
                    "masking_probability": self.masking_probability,
                    "min_masked_coordinates": (
                        self.min_masked_coordinates
                    ),
                    "formal_dp_claim": False,
                },
            }
        )
        self._save_log(log)

        if failures:
            print(
                f"[Server] R{server_round}: {len(failures)} fit "
                "failure(s); exact cancellation is not guaranteed "
                "for masks assigned before dropout."
            )
        elif masking_residual_l2 is not None:
            print(
                f"[Server] R{server_round}: aggregate masking "
                f"residual L2={masking_residual_l2:.6e}"
            )

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
        "--masking-probability", type=float, default=1.0,
        help=(
            "Bernoulli coordinate-selection probability. "
            "p=1.0 reproduces the original all-coordinate "
            "coordinated-mask setting."
        ),
    )
    parser.add_argument(
        "--min-masked-coordinates", type=int, default=2
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(METRICS_FILE, "w") as handle:
        json.dump([], handle)

    strategy = FedAvgCoordinatedZeroSumMasking(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        accept_failures=False,
        masking_coefficient=args.masking_coefficient,
        masking_probability=args.masking_probability,
        min_masked_coordinates=args.min_masked_coordinates,
        seed=args.seed,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
