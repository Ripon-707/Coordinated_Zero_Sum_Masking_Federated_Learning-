"""Dataset-I CZSM server for a specific-client crash experiment.

Use --crash-client with a partition id (for example 4, 6, or 9) and
--crash-on-round to reproduce the selective-crash traces.  The masking matrix
is generated before the client receives its crash instruction, so a fit crash
represents dropout after mask assignment.  Exact cancellation is therefore not
guaranteed in that round unless masks are regenerated for the final active set.
"""

import argparse
import json
import os
from typing import Dict

import flwr as fl
import numpy as np
from flwr.common import EvaluateIns, FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from masking_coordinator import (
    CoordinatedZeroSumMaskingCoordinator,
    aggregate_masking_residual,
)


LOG_DIR = "/home/ripon/code/czsm_synthetic_specific_crash_logs"
METRICS_FILE = os.path.join(LOG_DIR, "server_metrics.json")


def parameter_vector_length(parameters: Parameters) -> int:
    return int(sum(a.size for a in parameters_to_ndarrays(parameters)))


class FedAvgCZSMSpecificCrash(FedAvg):
    def __init__(
        self,
        *args,
        total_clients: int = 10,
        crash_client: int = -1,
        crash_on_round: int = -1,
        crash_phase: str = "fit",
        masking_coefficient: float = 5.0,
        masking_probability: float = 1.0,
        min_masked_coordinates: int = 2,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.total_clients = int(total_clients)
        self.crash_client = int(crash_client)
        self.crash_on_round = int(crash_on_round)
        self.crash_phase = str(crash_phase)
        self.masking_coefficient = float(masking_coefficient)
        self.masking_probability = float(masking_probability)
        self.min_masked_coordinates = int(min_masked_coordinates)

        self.coordinator = CoordinatedZeroSumMaskingCoordinator(seed=seed)
        self.round_assignments = {}
        self.partition_to_cid: Dict[int, str] = {}
        os.makedirs(LOG_DIR, exist_ok=True)

    def _target_cid(self):
        return self.partition_to_cid.get(self.crash_client)

    def configure_fit(self, server_round, parameters, client_manager):
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

        target_cid = self._target_cid()
        revised = []
        for proxy, fitins in instructions:
            cfg: Dict[str, Scalar] = dict(fitins.config or {})
            cfg.update(
                {
                    "server_round": int(server_round),
                    "masking_vector": json.dumps(
                        assignment.column_for(proxy.cid).tolist()
                    ),
                    "masking_coefficient": self.masking_coefficient,
                    "masking_probability": self.masking_probability,
                }
            )

            if (
                target_cid is not None
                and proxy.cid == target_cid
                and server_round == self.crash_on_round
                and self.crash_phase == "fit"
            ):
                cfg.update(
                    {
                        "crash_this_client": 1,
                        "crash_on_round": int(self.crash_on_round),
                        "crash_phase": "fit",
                    }
                )
            else:
                cfg["crash_this_client"] = 0

            revised.append(
                (proxy, FitIns(fitins.parameters, cfg))
            )
        return revised

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        target_cid = self._target_cid()
        revised = []
        for proxy, evalins in instructions:
            crash_now = (
                target_cid is not None
                and proxy.cid == target_cid
                and server_round == self.crash_on_round
                and self.crash_phase == "eval"
            )
            cfg: Dict[str, Scalar] = dict(evalins.config or {})
            cfg.update(
                {
                    "server_round": int(server_round),
                    "crash_this_client": int(crash_now),
                    "crash_on_round": int(self.crash_on_round),
                    "crash_phase": "eval",
                }
            )
            revised.append(
                (proxy, EvaluateIns(evalins.parameters, cfg))
            )
        return revised

    def aggregate_fit(self, server_round, results, failures):
        for proxy, fit_res in results:
            pid = (fit_res.metrics or {}).get("partition_id")
            if pid is not None:
                self.partition_to_cid[int(pid)] = proxy.cid

        aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        assignment = self.round_assignments.get(server_round)
        residual_l2 = None
        if assignment is not None and results:
            masks = [
                assignment.column_for(proxy.cid)
                for proxy, _ in results
            ]
            weights = [int(res.num_examples) for _, res in results]
            residual = aggregate_masking_residual(
                masks, weights=weights
            )
            residual_l2 = float(np.linalg.norm(residual))

        log = self._load_log()
        log.append(
            {
                "round": int(server_round),
                "fit": {
                    "successful_clients": int(len(results)),
                    "failed_clients": int(len(failures)),
                    "fedavg_weighted_masking_residual_l2": residual_l2,
                },
                "specific_crash": {
                    "partition_id": int(self.crash_client),
                    "crash_on_round": int(self.crash_on_round),
                    "crash_phase": self.crash_phase,
                    "dropout_after_mask_assignment": bool(failures),
                    "exact_cancellation_guaranteed": not bool(failures),
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

        total = sum(int(res.num_examples) for _, res in results)
        loss = sum(
            float(res.loss) * int(res.num_examples)
            for _, res in results
        ) / total
        accuracy = sum(
            float((res.metrics or {}).get("accuracy", 0.0))
            * int(res.num_examples)
            for _, res in results
        ) / total

        log = self._load_log()
        log.append(
            {
                "round": int(server_round),
                "global_eval": {
                    "weighted_loss": float(loss),
                    "weighted_accuracy": float(accuracy),
                    "successful_eval_clients": int(len(results)),
                    "failed_eval_clients": int(len(failures)),
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
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--total-clients", type=int, default=10)
    parser.add_argument("--crash-client", type=int, default=4)
    parser.add_argument("--crash-on-round", type=int, default=2)
    parser.add_argument(
        "--crash-phase",
        choices=["fit", "eval"],
        default="fit",
    )
    parser.add_argument(
        "--masking-coefficient", type=float, default=5.0
    )
    parser.add_argument(
        "--masking-probability", type=float, default=1.0
    )
    parser.add_argument(
        "--min-masked-coordinates", type=int, default=2
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(METRICS_FILE, "w") as handle:
        json.dump([], handle)

    strategy = FedAvgCZSMSpecificCrash(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=max(1, args.total_clients - 1),
        min_evaluate_clients=1,
        min_available_clients=1,
        accept_failures=True,
        total_clients=args.total_clients,
        crash_client=args.crash_client,
        crash_on_round=args.crash_on_round,
        crash_phase=args.crash_phase,
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
