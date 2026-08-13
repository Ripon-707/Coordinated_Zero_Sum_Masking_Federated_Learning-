"""Dataset-II (NSL-KDD) CZSM server for overall client-crash experiments."""

import argparse
import json
import os
import random
from typing import Dict, Set

import flwr as fl
import numpy as np
from flwr.common import EvaluateIns, FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from masking_coordinator import (
    CoordinatedZeroSumMaskingCoordinator,
    aggregate_masking_residual,
)


LOG_DIR = "/home/ripon/code/czsm_nsl_kdd_crash_logs"
METRICS_FILE = os.path.join(LOG_DIR, "server_metrics.json")


def parameter_vector_length(parameters: Parameters) -> int:
    return int(sum(a.size for a in parameters_to_ndarrays(parameters)))


class FedAvgCZSMNSLKDDCrash(FedAvg):
    """Crash selected clients during fit after masks have been assigned."""

    def __init__(
        self,
        *args,
        total_clients: int = 10,
        crash_percent: int = 0,
        crash_on_round: int = 3,
        crash_seed: int = 42,
        masking_coefficient: float = 5.0,
        masking_probability: float = 1.0,
        min_masked_coordinates: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.total_clients = int(total_clients)
        self.crash_percent = int(crash_percent)
        self.crash_on_round = int(crash_on_round)
        self.crash_seed = int(crash_seed)
        self.masking_coefficient = float(masking_coefficient)
        self.masking_probability = float(masking_probability)
        self.min_masked_coordinates = int(min_masked_coordinates)

        self._chosen_crash_cids: Set[str] = set()
        self._initial_wait_done = False
        self._crash_plan_built = False

        self.coordinator = CoordinatedZeroSumMaskingCoordinator(seed=42)
        self.round_assignments = {}
        os.makedirs(LOG_DIR, exist_ok=True)

    def _wait_and_plan(self, client_manager) -> None:
        if not self._initial_wait_done:
            client_manager.wait_for(self.total_clients)
            self._initial_wait_done = True

        if self._crash_plan_built:
            return

        all_cids = sorted(client_manager.all().keys())
        k = int(round(
            self.total_clients * self.crash_percent / 100.0
        ))
        rng = random.Random(self.crash_seed)
        self._chosen_crash_cids = (
            set(rng.sample(all_cids, k)) if k > 0 else set()
        )
        self._crash_plan_built = True

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round == 1:
            self._wait_and_plan(client_manager)

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
            crash_now = (
                server_round == self.crash_on_round
                and proxy.cid in self._chosen_crash_cids
            )
            cfg: Dict[str, Scalar] = dict(fitins.config or {})
            cfg.update(
                {
                    "server_round": int(server_round),
                    "masking_vector": json.dumps(
                        assignment.column_for(proxy.cid).tolist()
                    ),
                    "masking_coefficient": self.masking_coefficient,
                    "masking_probability": self.masking_probability,
                    "crash_this_client": int(crash_now),
                    "crash_on_round": int(self.crash_on_round),
                    "crash_phase": "fit",
                }
            )
            revised.append(
                (proxy, FitIns(fitins.parameters, cfg))
            )
        return revised

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        revised = []
        for proxy, evalins in instructions:
            cfg: Dict[str, Scalar] = dict(evalins.config or {})
            cfg.update(
                {
                    "server_round": int(server_round),
                    "crash_this_client": 0,
                }
            )
            revised.append(
                (proxy, EvaluateIns(evalins.parameters, cfg))
            )
        return revised

    def aggregate_fit(self, server_round, results, failures):
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
                "crash": {
                    "crash_percent": int(self.crash_percent),
                    "crash_on_round": int(self.crash_on_round),
                    "dropout_after_mask_assignment": bool(failures),
                    "exact_cancellation_guaranteed": not bool(failures),
                },
                "configuration": {
                    "masking_coefficient": self.masking_coefficient,
                    "masking_probability": self.masking_probability,
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
    parser.add_argument("--total-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument(
        "--crash-percent",
        type=int,
        choices=list(range(0, 100, 10)),
        default=0,
    )
    parser.add_argument("--crash-on-round", type=int, default=3)
    parser.add_argument("--crash-seed", type=int, default=42)
    parser.add_argument(
        "--masking-coefficient", type=float, default=5.0
    )
    parser.add_argument(
        "--masking-probability", type=float, default=1.0
    )
    parser.add_argument(
        "--min-masked-coordinates", type=int, default=2
    )
    args = parser.parse_args()

    crashed = int(round(
        args.total_clients * args.crash_percent / 100.0
    ))
    remaining = max(1, args.total_clients - crashed)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(METRICS_FILE, "w") as handle:
        json.dump([], handle)

    strategy = FedAvgCZSMNSLKDDCrash(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=remaining,
        min_evaluate_clients=1,
        min_available_clients=1,
        accept_failures=True,
        total_clients=args.total_clients,
        crash_percent=args.crash_percent,
        crash_on_round=args.crash_on_round,
        crash_seed=args.crash_seed,
        masking_coefficient=args.masking_coefficient,
        masking_probability=args.masking_probability,
        min_masked_coordinates=args.min_masked_coordinates,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
