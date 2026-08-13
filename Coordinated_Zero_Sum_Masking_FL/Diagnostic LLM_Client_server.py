from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
from dotenv import load_dotenv
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from groq import Groq


# LOAD ENVIRONMENT

load_dotenv()


# CONFIGURATION

LOG_DIR = os.getenv("LOG_DIR", "./logs")
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")

NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "10"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "10"))

MIN_AVAILABLE_CLIENTS = NUM_CLIENTS
MIN_FIT_CLIENTS = NUM_CLIENTS
MIN_EVAL_CLIENTS = NUM_CLIENTS

FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0

# Masking-control parameters.

MASKING_COEFFICIENT = float(os.getenv("MASKING_COEFFICIENT", "5.0"))
MASKING_PROBABILITY = float(os.getenv("MASKING_PROBABILITY", "1.0"))
MIN_MASKED_COORDINATES = int(os.getenv("MIN_MASKED_COORDINATES", "2"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

os.makedirs(LOG_DIR, exist_ok=True)

METRICS_FILE = os.path.join(
    LOG_DIR,
    "llm_diagnostic_server_metrics.json",
)


# GENERIC HELPERS

def append_json_list(filepath: str, payload: Dict[str, Any]) -> None:
    """Append one JSON object to a JSON-list log file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except (json.JSONDecodeError, OSError):
            data = []
    else:
        data = []

    data.append(payload)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_string_list(value: Any) -> List[str]:
    """Normalize an LLM JSON field into a list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value]

    if value is None:
        return []

    return [str(value)]


def concat_len_of_parameters(parameters: Parameters) -> int:
    """Total scalar parameter count in a Flower Parameters object."""
    ndarrays = parameters_to_ndarrays(parameters)
    return int(sum(arr.size for arr in ndarrays))


# PROTOTYPE MASKING COORDINATOR

class PrototypeMaskingCoordinator:
    """
    Logical prototype of the masking coordinator.

    IMPORTANT:
    This class is intentionally co-located with the Flower server process
    for experimental convenience. Therefore, the code demonstrates the
    masking workflow but does NOT enforce process-level or hardware-level
    isolation between the masking coordinator and aggregation server.

    Under the threat model, a production coordinator must reside
    in a separate non-colluding trust domain and distribute each client
    only its assigned mask through authenticated/confidential channels.
    """

    def __init__(
        self,
        masking_coefficient: float,
        masking_probability: float,
        min_masked_coordinates: int,
        random_state: int,
    ):
        if masking_coefficient <= 0:
            raise ValueError("masking_coefficient must be > 0.")

        if not 0 < masking_probability <= 1:
            raise ValueError("masking_probability must satisfy 0 < p <= 1.")

        if min_masked_coordinates < 0:
            raise ValueError("min_masked_coordinates must be >= 0.")

        self.masking_coefficient = float(masking_coefficient)
        self.masking_probability = float(masking_probability)
        self.min_masked_coordinates = int(min_masked_coordinates)
        self.random_state = int(random_state)

    def _sample_coordinate_selection(
        self,
        rng: np.random.Generator,
        num_params: int,
    ) -> np.ndarray:
        """
        Bernoulli coordinate selection with the manuscript's minimum-k rule.
        """
        if num_params <= 0:
            return np.zeros(0, dtype=bool)

        required = min(
            self.min_masked_coordinates,
            num_params,
        )

        if self.masking_probability >= 1.0:
            return np.ones(num_params, dtype=bool)

        # Resample until the minimum selected-coordinate requirement is met.
        while True:
            selected = (
                rng.random(num_params) < self.masking_probability
            )

            if int(selected.sum()) >= required:
                return selected

    def generate_assignments(
        self,
        server_round: int,
        num_params: int,
        client_ids: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Generate coordinated zero-sum masking assignments.

        For every selected coordinate:
          1. The first n-1 client values are independent zero-mean
             Laplace draws with preliminary scale 1 / mu.
          2. The final client value is the negative sum of the first n-1
             values and is therefore a dependent balancing value.

        Under equal aggregation weights and full participation of the
        assigned client set, the masking vectors sum to zero.
        """
        num_clients = len(client_ids)

        if num_clients == 0:
            return {}, {
                "selected_coordinate_count": 0,
                "equal_weight_zero_sum_residual_l2": 0.0,
            }

        rng = np.random.default_rng(
            self.random_state + int(server_round)
        )

        selected = self._sample_coordinate_selection(
            rng=rng,
            num_params=num_params,
        )

        matrix = np.zeros(
            (num_params, num_clients),
            dtype=np.float64,
        )

        if num_clients >= 2 and np.any(selected):
            scale = 1.0 / self.masking_coefficient

            preliminary = rng.laplace(
                loc=0.0,
                scale=scale,
                size=(int(selected.sum()), num_clients - 1),
            )

            matrix[selected, : num_clients - 1] = preliminary

            # Dependent balancing column.
            matrix[selected, num_clients - 1] = -np.sum(
                preliminary,
                axis=1,
            )

        assignments = {
            cid: matrix[:, idx].copy()
            for idx, cid in enumerate(client_ids)
        }

        zero_sum_residual = np.sum(matrix, axis=1)

        metadata = {
            "masking_coefficient": self.masking_coefficient,
            "preliminary_laplace_scale": 1.0 / self.masking_coefficient,
            "masking_probability": self.masking_probability,
            "min_masked_coordinates": self.min_masked_coordinates,
            "selected_coordinate_count": int(selected.sum()),
            "total_parameter_count": int(num_params),
            "assigned_client_ids": list(client_ids),
            "equal_weight_zero_sum_residual_l2": float(
                np.linalg.norm(zero_sum_residual)
            ),
            "final_client_mask_is_dependent": True,
            "formal_dp_claim": False,
        }

        return assignments, metadata


# GROQ LLM WRAPPER

class GroqDiagnosticLLMClient:
    def __init__(
        self,
        model: str = GROQ_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Put it in the environment or .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=self.temperature,
        )

        content = response.choices[0].message.content
        return (content or "").strip()


# BASE DIAGNOSTIC AGENT

class BaseDiagnosticAgent:
    def __init__(self, llm_client: Any, agent_name: str):
        self.llm = llm_client
        self.agent_name = agent_name

    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}

        text = text.strip()

        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.DOTALL,
        )

        if fenced_match:
            candidate = fenced_match.group(1)

            try:
                data = json.loads(candidate)
                return data if isinstance(data, dict) else {}
            except Exception:
                pass

        brace_match = re.search(
            r"(\{.*\})",
            text,
            re.DOTALL,
        )

        if brace_match:
            candidate = brace_match.group(1)

            try:
                data = json.loads(candidate)
                return data if isinstance(data, dict) else {}
            except Exception:
                pass

        return {}

    def _call_llm_json(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        try:
            raw = self.llm.generate(prompt)
        except Exception as exc:
            return {
                "valid": False,
                "error": str(exc),
                "parsed": {},
            }

        parsed = self._extract_json(raw)

        return {
            "valid": bool(parsed),
            "error": "",
            "parsed": parsed if isinstance(parsed, dict) else {},
        }



# LLM PERFORMANCE-DIAGNOSTIC AGENT
class LLMPerformanceDiagnosticAgent(BaseDiagnosticAgent):
    """
    Post-round metric-conditioned diagnostic agent.

    It does not:
      - control optimization,
      - change aggregation,
      - generate masking vectors,
      - choose the masking coefficient,
      - provide a formal privacy guarantee,
      - verify a causal explanation.
    """

    ALLOWED_STATUS = {
        "good",
        "moderate",
        "poor",
        "unstable",
        "unknown",
    }

    def __init__(self, llm_client: Any):
        super().__init__(
            llm_client=llm_client,
            agent_name="performance_diagnostic_reasoning",
        )

    def explain(
        self,
        client_metrics: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        prev_global_acc = _safe_float(
            global_state.get("prev_global_accuracy", 0.0)
        )

        curr_global_acc = _safe_float(
            global_state.get(
                "curr_global_accuracy",
                prev_global_acc,
            )
        )

        payload = {
            "client_metrics": {
                "loss": _safe_float(
                    client_metrics.get("loss", 0.0)
                ),
                "accuracy": _safe_float(
                    client_metrics.get("accuracy", 0.0)
                ),
                "update_norm": _safe_float(
                    client_metrics.get("update_norm", 0.0)
                ),
                "param_norm": _safe_float(
                    client_metrics.get("param_norm", 0.0)
                ),
                "masking_norm": _safe_float(
                    client_metrics.get("masking_norm", 0.0)
                ),
                "num_samples": _safe_int(
                    client_metrics.get("num_samples", 0)
                ),
                "diagnostic_sensitivity_score": _safe_float(
                    client_metrics.get(
                        "diagnostic_sensitivity_score",
                        0.0,
                    )
                ),
                "volatility_score": _safe_float(
                    client_metrics.get("volatility_score", 0.0)
                ),
                "class_0_count": _safe_int(
                    client_metrics.get("class_0_count", 0)
                ),
                "class_1_count": _safe_int(
                    client_metrics.get("class_1_count", 0)
                ),
            },
            "global_state": {
                "prev_global_accuracy": prev_global_acc,
                "curr_global_accuracy": curr_global_acc,
                "global_accuracy_dropped": (
                    curr_global_acc < prev_global_acc
                ),
            },
        }

        prompt = f"""
You are a diagnostic reasoning agent for federated learning training.

Task:
Provide a short metric-conditioned interpretation of the observed
client training state.

Scientific scope:
- Use only the supplied metrics.
- Treat possible causes as candidates, not verified causes.
- Do not claim differential privacy, attack resistance, data leakage,
  or security from these metrics.
- The diagnostic_sensitivity_score and volatility_score are heuristic
  engineering features, not formal DP sensitivity or attack-risk scores.
- The confidence field is your self-reported confidence and is not
  statistically calibrated.
- State uncertainty when the supplied metrics are insufficient.
- Keep the output concise and scientific.

Possible metric-conditioned observations:
- High loss may be associated with weak local fit, difficult local data,
  poor convergence, class imbalance, or unstable training behavior.
- Low accuracy may be associated with weak local fit, heterogeneity,
  instability, or class imbalance.
- A large update_norm indicates a larger local model change.
- A large masking_norm relative to update_norm indicates a larger
  numerical masking magnitude relative to the clean local update.
- A high volatility_score indicates larger round-to-round changes in
  the monitored client metrics.
- diagnostic_sensitivity_score is a heuristic summary based on the
  relative masking/update behavior and other observed training metrics.
- High accuracy with low loss is consistent with stronger local fit.
- High accuracy with high loss can occur when many predictions are
  correct but some predictions remain uncertain or strongly incorrect.

Input:
{json.dumps(payload, indent=2)}

Return valid JSON only:
{{
  "performance_status": "good",
  "accuracy_reason": "<concise metric-conditioned interpretation>",
  "loss_reason": "<concise metric-conditioned interpretation>",
  "main_possible_causes": ["<candidate cause>"],
  "evidence_from_metrics": ["<metric-based evidence>"],
  "suggested_action": ["<analysis-oriented follow-up>"],
  "confidence": 0.0
}}
""".strip()

        output = self._call_llm_json(prompt)
        parsed = output.get("parsed", {})

        status = str(
            parsed.get("performance_status", "unknown")
        ).strip().lower()

        if status not in self.ALLOWED_STATUS:
            status = "unknown"

        return {
            "valid": bool(output.get("valid", False)),
            "error": str(output.get("error", "")),
            "performance_status": status,
            "accuracy_reason": str(
                parsed.get(
                    "accuracy_reason",
                    "No accuracy interpretation was produced.",
                )
            ),
            "loss_reason": str(
                parsed.get(
                    "loss_reason",
                    "No loss interpretation was produced.",
                )
            ),
            "main_possible_causes": _as_string_list(
                parsed.get("main_possible_causes", [])
            ),
            "evidence_from_metrics": _as_string_list(
                parsed.get("evidence_from_metrics", [])
            ),
            "suggested_action": _as_string_list(
                parsed.get("suggested_action", [])
            ),
            "confidence": _clamp(
                _safe_float(
                    parsed.get("confidence", 0.0)
                ),
                0.0,
                1.0,
            ),
            "confidence_is_self_reported_and_uncalibrated": True,
        }


# POST-ROUND REASONING COORDINATOR

class PostRoundReasoningCoordinator:
    def __init__(self, llm_client: Any):
        self.reasoning_agent = LLMPerformanceDiagnosticAgent(
            llm_client
        )

        self.prev_update_norms: Dict[str, float] = {}
        self.prev_losses: Dict[str, float] = {}
        self.prev_accuracies: Dict[str, float] = {}

    @staticmethod
    def _normalize(
        value: float,
        vmin: float,
        vmax: float,
    ) -> float:
        if vmax - vmin < 1e-12:
            return 0.0

        return (value - vmin) / (vmax - vmin)

    def _compute_diagnostic_scores(
        self,
        client_metrics_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute two heuristic engineering features.

        diagnostic_sensitivity_score:
            A normalized summary emphasizing masking magnitude relative
            to clean update magnitude, while also incorporating observed
            loss, low accuracy, and volatility.

        volatility_score:
            A normalized summary of round-to-round changes in update norm,
            loss, and accuracy.

        Neither score is a DP sensitivity quantity or validated attack-risk
        measurement.
        """
        client_ids = list(client_metrics_map.keys())

        if not client_ids:
            return {}

        update_norms = [
            _safe_float(
                client_metrics_map[cid].get(
                    "update_norm",
                    0.0,
                )
            )
            for cid in client_ids
        ]

        masking_norms = [
            _safe_float(
                client_metrics_map[cid].get(
                    "masking_norm",
                    0.0,
                )
            )
            for cid in client_ids
        ]

        losses = [
            _safe_float(
                client_metrics_map[cid].get("loss", 0.0)
            )
            for cid in client_ids
        ]

        accuracies = [
            _safe_float(
                client_metrics_map[cid].get(
                    "accuracy",
                    0.0,
                )
            )
            for cid in client_ids
        ]

        masking_to_update_ratios = [
            mask_norm / (update_norm + 1e-8)
            for mask_norm, update_norm in zip(
                masking_norms,
                update_norms,
            )
        ]

        min_ratio = min(masking_to_update_ratios)
        max_ratio = max(masking_to_update_ratios)

        min_loss = min(losses)
        max_loss = max(losses)

        min_acc = min(accuracies)
        max_acc = max(accuracies)

        result: Dict[str, Dict[str, float]] = {}

        for cid, metrics in client_metrics_map.items():
            update_norm = _safe_float(
                metrics.get("update_norm", 0.0)
            )
            masking_norm = _safe_float(
                metrics.get("masking_norm", 0.0)
            )
            loss = _safe_float(
                metrics.get("loss", 0.0)
            )
            accuracy = _safe_float(
                metrics.get("accuracy", 0.0)
            )

            mask_ratio = (
                masking_norm / (update_norm + 1e-8)
            )

            norm_mask_ratio = self._normalize(
                mask_ratio,
                min_ratio,
                max_ratio,
            )

            norm_loss = self._normalize(
                loss,
                min_loss,
                max_loss,
            )

            norm_low_accuracy = (
                1.0
                - self._normalize(
                    accuracy,
                    min_acc,
                    max_acc,
                )
            )

            prev_update = self.prev_update_norms.get(
                cid,
                update_norm,
            )
            prev_loss = self.prev_losses.get(
                cid,
                loss,
            )
            prev_accuracy = self.prev_accuracies.get(
                cid,
                accuracy,
            )

            delta_update = abs(
                update_norm - prev_update
            ) / (abs(prev_update) + 1e-8)

            delta_loss = abs(
                loss - prev_loss
            ) / (abs(prev_loss) + 1e-8)

            delta_accuracy = abs(
                accuracy - prev_accuracy
            ) / (abs(prev_accuracy) + 1e-8)

            delta_update = _clamp(
                delta_update,
                0.0,
                1.0,
            )
            delta_loss = _clamp(
                delta_loss,
                0.0,
                1.0,
            )
            delta_accuracy = _clamp(
                delta_accuracy,
                0.0,
                1.0,
            )

            volatility_score = _clamp(
                0.50 * delta_update
                + 0.30 * delta_loss
                + 0.20 * delta_accuracy,
                0.0,
                1.0,
            )

            diagnostic_sensitivity_score = _clamp(
                0.50 * norm_mask_ratio
                + 0.20 * norm_loss
                + 0.15 * norm_low_accuracy
                + 0.15 * volatility_score,
                0.0,
                1.0,
            )

            result[cid] = {
                "diagnostic_sensitivity_score": round(
                    float(diagnostic_sensitivity_score),
                    4,
                ),
                "volatility_score": round(
                    float(volatility_score),
                    4,
                ),
                "masking_to_update_ratio": round(
                    float(mask_ratio),
                    6,
                ),
            }

        return result

    def explain_clients(
        self,
        client_metrics_map: Dict[str, Dict[str, Any]],
        global_state: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        derived_scores = self._compute_diagnostic_scores(
            client_metrics_map
        )

        reasoning_meta: Dict[str, Dict[str, Any]] = {}

        for cid, metrics in client_metrics_map.items():
            enriched_metrics = dict(metrics)

            enriched_metrics[
                "diagnostic_sensitivity_score"
            ] = derived_scores[cid][
                "diagnostic_sensitivity_score"
            ]

            enriched_metrics[
                "volatility_score"
            ] = derived_scores[cid][
                "volatility_score"
            ]

            reasoning_output = self.reasoning_agent.explain(
                enriched_metrics,
                global_state,
            )

            reasoning_meta[cid] = {
                "input_metrics": enriched_metrics,
                "derived_diagnostic_sensitivity_score": (
                    derived_scores[cid][
                        "diagnostic_sensitivity_score"
                    ]
                ),
                "derived_volatility_score": (
                    derived_scores[cid][
                        "volatility_score"
                    ]
                ),
                "masking_to_update_ratio": (
                    derived_scores[cid][
                        "masking_to_update_ratio"
                    ]
                ),
                "performance_diagnostic_reasoning": (
                    reasoning_output
                ),
            }

            self.prev_update_norms[cid] = _safe_float(
                metrics.get("update_norm", 0.0)
            )
            self.prev_losses[cid] = _safe_float(
                metrics.get("loss", 0.0)
            )
            self.prev_accuracies[cid] = _safe_float(
                metrics.get("accuracy", 0.0)
            )

        return reasoning_meta



# FLOWER STRATEGY

class CoordinatedMaskingWithLLMDiagnosticsStrategy(FedAvg):
    """
    FedAvg strategy with:
      1. prototype coordinated zero-sum masking assignment, and
      2. post-round LLM diagnostic reasoning.

    The LLM does not alter training or masking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.masking_coordinator = PrototypeMaskingCoordinator(
            masking_coefficient=MASKING_COEFFICIENT,
            masking_probability=MASKING_PROBABILITY,
            min_masked_coordinates=MIN_MASKED_COORDINATES,
            random_state=RANDOM_STATE,
        )

        self.reasoning_coordinator = (
            PostRoundReasoningCoordinator(
                llm_client=GroqDiagnosticLLMClient()
            )
        )

        self.assigned_client_ids_by_round: Dict[
            int,
            List[str],
        ] = {}

        self.masking_metadata_by_round: Dict[
            int,
            Dict[str, Any],
        ] = {}

        self.fit_metrics_by_round: Dict[
            int,
            Dict[str, Dict[str, Any]],
        ] = {}

        self.global_state = {
            "prev_global_accuracy": 0.0,
            "curr_global_accuracy": 0.0,
        }

        self._has_global_accuracy = False

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        fit_instructions = super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

        if not fit_instructions:
            return fit_instructions

        num_params = concat_len_of_parameters(
            parameters
        )

        client_ids = [
            client_proxy.cid
            for client_proxy, _ in fit_instructions
        ]

        assignments, metadata = (
            self.masking_coordinator.generate_assignments(
                server_round=server_round,
                num_params=num_params,
                client_ids=client_ids,
            )
        )

        self.assigned_client_ids_by_round[
            server_round
        ] = list(client_ids)

        self.masking_metadata_by_round[
            server_round
        ] = metadata

        updated_fit_instructions = []

        for client_proxy, fitins in fit_instructions:
            cid = client_proxy.cid

            config: Dict[str, Scalar] = (
                dict(fitins.config)
                if fitins.config is not None
                else {}
            )

            config["masking_vector"] = json.dumps(
                assignments[cid].tolist()
            )

            config["masking_coefficient"] = str(
                MASKING_COEFFICIENT
            )

            updated_fit_instructions.append(
                (
                    client_proxy,
                    FitIns(
                        fitins.parameters,
                        config,
                    ),
                )
            )

        print(
            f"\n[Server] Round {server_round} "
            f"assigned clients: {client_ids}"
        )

        print(
            f"[Server] Round {server_round}: "
            f"mu={MASKING_COEFFICIENT}, "
            f"p={MASKING_PROBABILITY}, "
            f"selected_coordinates="
            f"{metadata['selected_coordinate_count']}, "
            f"zero_sum_residual_l2="
            f"{metadata['equal_weight_zero_sum_residual_l2']:.6e}"
        )

        return updated_fit_instructions

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        round_client_metrics: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for client_proxy, fit_res in results:
            round_client_metrics[
                client_proxy.cid
            ] = dict(fit_res.metrics)

        self.fit_metrics_by_round[
            server_round
        ] = round_client_metrics

        assigned = set(
            self.assigned_client_ids_by_round.get(
                server_round,
                [],
            )
        )

        returned = set(
            round_client_metrics.keys()
        )

        full_assigned_participation = (
            assigned == returned
        )

        metadata = self.masking_metadata_by_round.setdefault(
            server_round,
            {},
        )

        metadata[
            "returned_client_ids"
        ] = sorted(returned)

        metadata[
            "full_assigned_participation"
        ] = bool(full_assigned_participation)

        metadata[
            "cancellation_guaranteed_for_returned_set"
        ] = bool(full_assigned_participation)

        if not full_assigned_participation:
            print(
                f"[Server][Round {server_round}] "
                "Client dropout/failure occurred after mask assignment. "
                "Exact same-round zero-sum cancellation is not guaranteed."
            )

        return super().aggregate_fit(
            server_round,
            results,
            failures,
        )

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        aggregated = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        if not results:
            return aggregated

        total_examples = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0

        per_client_evaluate: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for client_proxy, eval_res in results:
            cid = client_proxy.cid
            num_examples = int(
                eval_res.num_examples
            )

            total_examples += num_examples

            weighted_loss += (
                float(eval_res.loss)
                * num_examples
            )

            weighted_accuracy += (
                float(
                    eval_res.metrics.get(
                        "accuracy",
                        0.0,
                    )
                )
                * num_examples
            )

            per_client_evaluate[cid] = {
                "num_examples": num_examples,
                "loss": float(eval_res.loss),
                "accuracy": float(
                    eval_res.metrics.get(
                        "accuracy",
                        0.0,
                    )
                ),
            }

        if total_examples > 0:
            weighted_loss /= total_examples
            weighted_accuracy /= total_examples

        if not self._has_global_accuracy:
            previous_global_accuracy = (
                weighted_accuracy
            )
            self._has_global_accuracy = True
        else:
            previous_global_accuracy = (
                self.global_state[
                    "curr_global_accuracy"
                ]
            )

        self.global_state[
            "prev_global_accuracy"
        ] = previous_global_accuracy

        self.global_state[
            "curr_global_accuracy"
        ] = weighted_accuracy

        fit_metrics = self.fit_metrics_by_round.get(
            server_round,
            {},
        )

        # IMPORTANT:
        # Reasoning is performed AFTER the current round's fit metrics
        # and global evaluation are available. This avoids the previous
        # one-round offset.
        reasoning = {}

        if fit_metrics:
            reasoning = (
                self.reasoning_coordinator.explain_clients(
                    client_metrics_map=fit_metrics,
                    global_state=self.global_state,
                )
            )

        round_log = {
            "round": int(server_round),
            "type": "post_round_diagnostic",
            "llm_configuration": {
                "provider": "Groq",
                "model": GROQ_MODEL,
                "temperature": LLM_TEMPERATURE,
                "confidence_is_self_reported_and_uncalibrated": True,
            },
            "masking_configuration": (
                self.masking_metadata_by_round.get(
                    server_round,
                    {},
                )
            ),
            "client_fit_metrics": fit_metrics,
            "global_state": dict(
                self.global_state
            ),
            "global_loss": float(
                weighted_loss
            ),
            "global_accuracy": float(
                weighted_accuracy
            ),
            "per_client_evaluate": (
                per_client_evaluate
            ),
            "llm_diagnostic_reasoning": reasoning,
            "scientific_scope": {
                "formal_dp_claim": False,
                "attack_resistance_claim": False,
                "llm_outputs_are_verified_causal_explanations": False,
                "llm_confidence_is_calibrated": False,
            },
        }

        append_json_list(
            METRICS_FILE,
            round_log,
        )

        print(
            f"[Server] Round {server_round}: "
            f"global_accuracy="
            f"{weighted_accuracy:.4f}, "
            f"global_loss="
            f"{weighted_loss:.4f}"
        )

        for cid, diagnostic in reasoning.items():
            output = diagnostic.get(
                "performance_diagnostic_reasoning",
                {},
            )

            print(
                f"[LLM][Round {server_round}]"
                f"[Client {cid}] "
                f"sens="
                f"{diagnostic.get('derived_diagnostic_sensitivity_score', 0.0):.4f}, "
                f"vol="
                f"{diagnostic.get('derived_volatility_score', 0.0):.4f}, "
                f"status="
                f"{output.get('performance_status', 'unknown')}"
            )

        return aggregated



# MAIN


if __name__ == "__main__":
    print(
        "GROQ_API_KEY loaded:",
        bool(os.getenv("GROQ_API_KEY")),
    )
    print(
        "Groq model:",
        GROQ_MODEL,
    )
    print(
        "Masking coefficient (engineering control):",
        MASKING_COEFFICIENT,
    )
    print(
        "Masking probability:",
        MASKING_PROBABILITY,
    )
    print(
        "Local coordinator note: prototype logical emulation only; "
        "production trust-domain isolation is not implemented here."
    )

    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)

    strategy = CoordinatedMaskingWithLLMDiagnosticsStrategy(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy=strategy,
    )