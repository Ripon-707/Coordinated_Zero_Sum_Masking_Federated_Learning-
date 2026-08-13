

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MaskingRound:
    """Masking assignment for one communication round."""

    client_ids: Tuple[str, ...]
    matrix: np.ndarray
    coordinate_selection: np.ndarray
    mu: float
    masking_probability: float

    def column_for(self, client_id: str) -> np.ndarray:
        idx = self.client_ids.index(str(client_id))
        return self.matrix[:, idx].copy()


class CoordinatedZeroSumMaskingCoordinator:
    """Generate client-specific masks with aggregate masking cancellation.

    For equal aggregation weights, each selected coordinate satisfies

        sum_i M[j, i] = 0.

    If explicit aggregation weights are supplied, the final balancing column is
    instead constructed so that

        sum_i q_i M[j, i] = 0.

    The first n-1 values are preliminary Laplace draws with scale 1/mu.  The
    final value is a dependent balancing value and is not an independent
    Laplace draw.
    """

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    @staticmethod
    def _validate_probability(p: float) -> float:
        p = float(p)
        if not (0.0 < p <= 1.0):
            raise ValueError("masking_probability must satisfy 0 < p <= 1")
        return p

    @staticmethod
    def _validate_mu(mu: float) -> float:
        mu = float(mu)
        if mu <= 0.0:
            raise ValueError("masking coefficient mu must be > 0")
        return mu

    def generate(
        self,
        *,
        round_id: int,
        num_coordinates: int,
        client_ids: Sequence[str],
        mu: float = 5.0,
        masking_probability: float = 1.0,
        min_masked_coordinates: int = 2,
        aggregation_weights: Optional[Sequence[float]] = None,
    ) -> MaskingRound:
        """Create the masking matrix for a finalized client set."""

        mu = self._validate_mu(mu)
        p = self._validate_probability(masking_probability)
        client_ids = tuple(str(cid) for cid in client_ids)
        n_clients = len(client_ids)
        d = int(num_coordinates)

        if d < 1:
            raise ValueError("num_coordinates must be >= 1")

        if n_clients < 1:
            raise ValueError("at least one client is required")

        if n_clients == 1:
            matrix = np.zeros((d, 1), dtype=np.float64)
            selection = np.zeros(d, dtype=np.int8)
            return MaskingRound(client_ids, matrix, selection, mu, p)

        target_min = max(0, min(int(min_masked_coordinates), d))

        # A round-specific generator provides deterministic reproduction while
        # keeping different rounds distinct.
        rng = np.random.default_rng(self.seed + int(round_id))

        selection = rng.binomial(1, p, size=d).astype(np.int8)
        while int(selection.sum()) < target_min:
            selection = rng.binomial(1, p, size=d).astype(np.int8)

        selected = np.flatnonzero(selection)
        matrix = np.zeros((d, n_clients), dtype=np.float64)

        if selected.size == 0:
            return MaskingRound(client_ids, matrix, selection, mu, p)

        preliminary = rng.laplace(
            loc=0.0,
            scale=1.0 / mu,
            size=(selected.size, n_clients - 1),
        )
        matrix[selected, : n_clients - 1] = preliminary

        if aggregation_weights is None:
            matrix[selected, n_clients - 1] = -np.sum(preliminary, axis=1)
        else:
            q = np.asarray(aggregation_weights, dtype=np.float64)
            if q.shape != (n_clients,):
                raise ValueError("aggregation_weights must have one value per client")
            if np.any(q <= 0):
                raise ValueError("aggregation_weights must be positive")
            last_weight = q[-1]
            matrix[selected, n_clients - 1] = -(
                preliminary @ q[: n_clients - 1]
            ) / last_weight

        return MaskingRound(client_ids, matrix, selection, mu, p)


def aggregate_masking_residual(
    masks: Iterable[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return the aggregate masking vector for diagnostic verification."""

    masks = [np.asarray(m, dtype=np.float64) for m in masks]
    if not masks:
        return np.array([], dtype=np.float64)

    stack = np.stack(masks, axis=0)

    if weights is None:
        return np.mean(stack, axis=0)

    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (stack.shape[0],):
        raise ValueError("weights must match the number of masks")
    w = w / np.sum(w)
    return np.sum(stack * w[:, None], axis=0)
