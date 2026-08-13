"""Dataset-I CZSM server for overall client-crash experiments."""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Set

import flwr as fl
import numpy as np
from flwr.common import EvaluateIns, FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from masking_coordinator import (
    CoordinatedZeroSumMaskingCoordinator,
    aggregate_masking_residual,
)


LOG_DIR = "/home/ripon/code/czsm_synthetic_crash_logs"
METRICS_FILE = os.path.join(LOG_DIR, "server_metrics.json")


def parameter_vector_length(parameters: Parameters) -> int:
    return int(sum(a.size for a in parameters_to_ndarrays(parameters)))


class FedAvgCZSMOverallCrash(FedAvg):
    """Simulate client failures after masking assignments have been generated."""

    def __init__(
        self,
        *args,
        drop_rate: float = 0.0,
        drop_seed: int = 42,
        num_rounds: int = 10,
        total_clients: int = 10,
        crash_rounds: Optional[List[int]] = None,
        crash_phase_mode: str = "random",
        masking_coefficient: float = 5.0,
        masking_probability: float = 1.0,
        min_masked_coordinates: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.drop_rate = float(drop_rate)
        self.drop_seed = int(drop_seed)
        self.num_rounds = int(num_rounds)
        self.total_clients = int(total_clients)
        self.crash_rounds = crash_rounds
        self.crash_phase_mode = str(crash_phase_mode)
        self.masking_coefficient = float(masking_coefficient)
        self.masking_probability = float(masking_probability)
        self.min_masked_coordinates = int(min_masked_coordinates)

        self.coordinator = CoordinatedZeroSumMaskingCoordinator(seed=42)
        self.round_assignments = {}

        self._initial_wait_done = False
        self._crash_plan_built = False
        self._crash_set: Set[str] = set()
        self._crash_on_round: Dict[str, int] = {}
        self._crash_phase: Dict[str, str] = {}

        os.makedirs(LOG_DIR, exist_ok=True)

    def _wait_for_all_clients_once(self, client_manager) -> None:
        if self._initial_wait_done:
            return
        print(
            f"[Server] waiting for {self.total_clients} clients "
            "before round 1..."
        )
        client_manager.wait_for(self.total_clients)
        self._initial_wait_done = True

    def _build_crash_plan(self, client_manager) -> None:
        if self._crash_plan_built:
            return

        all_cids = sorted(client_manager.all().keys())
        if len(all_cids) < self.total_clients:
            return

        k = int(math.ceil(self.drop_rate * len(all_cids)))
        rng = np.random.default_rng(self.drop_seed)
        if k > 0:
            chosen = list(rng.choice(all_cids, size=k, replace=False))
        else:
            chosen = []

        self._crash_set = set(chosen)

        def choose_phase() -> str:
            if self.crash_phase_mode in ("fit", "eval"):
                return self.crash_phase_mode
            return "fit" if rng.random() < 0.5 else "eval"

        for index, cid in enumerate(chosen):
            if self.crash_rounds:
                round_id = int(
                    self.crash_rounds[index % len(self.crash_rounds)]
                )
            else:
                low, high = 2, min(self.num_rounds, 9)
                round_id = int(rng.integers(low, high + 1))
            self._crash_on_round[cid] = round_id
            self._crash_phase[cid] = choose_phase()

        self._crash_plan_built = True
        print(
            f"[Server] crash plan: {k}/{len(all_cids)} clients "
            f"({self.drop_rate:.0%})"
        )

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round == 1:
            self._wait_for_all_clients_once(client_manager)
            self._build_crash_plan(client_manager)

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
            if proxy.cid in self._crash_set:
                cfg["crash_on_round"] = int(
                    self._crash_on_round[proxy.cid]
                )
                cfg["crash_phase"] = self._crash_phase[proxy.cid]
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
            cfg["server_round"] = int(server_round)
            if proxy.cid in self._crash_set:
                cfg["crash_on_round"] = int(
                    self._crash_on_round[proxy.cid]
                )
                cfg["crash_phase"] = self._crash_phase[proxy.cid]
            revised.append(
                (proxy, EvaluateIns(evalins.parameters, cfg))
            )
        return revised

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        assignment = self.round_assignments.get(server_round)
        successful_cids = [proxy.cid for proxy, _ in results]
        residual_l2 = None
        if assignment is not None and successful_cids:
            masks = [
                assignment.column_for(cid)
                for cid in successful_cids
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
                    "drop_rate": self.drop_rate,
                    "planned_crashed_clients": int(
                        len(self._crash_set)
                    ),
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

        if failures:
            print(
                f"[Server] R{server_round}: {len(failures)} fit "
                "failure(s) after assignment; residual masking may remain."
            )
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
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-seed", type=int, default=42)
    parser.add_argument("--crash-rounds", type=str, default="")
    parser.add_argument(
        "--crash-phase-mode",
        choices=["random", "fit", "eval"],
        default="random",
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
    args = parser.parse_args()

    crash_rounds = None
    if args.crash_rounds.strip():
        crash_rounds = [
            int(value.strip())
            for value in args.crash_rounds.split(",")
            if value.strip()
        ]

    dropped = int(math.ceil(args.drop_rate * args.total_clients))
    remaining = max(1, args.total_clients - dropped)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(METRICS_FILE, "w") as handle:
        json.dump([], handle)

    strategy = FedAvgCZSMOverallCrash(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=remaining,
        min_evaluate_clients=1,
        min_available_clients=1,
        accept_failures=True,
        drop_rate=args.drop_rate,
        drop_seed=args.drop_seed,
        num_rounds=args.num_rounds,
        total_clients=args.total_clients,
        crash_rounds=crash_rounds,
        crash_phase_mode=args.crash_phase_mode,
        masking_coefficient=args.masking_coefficient,
        masking_probability=args.masking_probability,
        min_masked_coordinates=args.min_masked_coordinates,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
