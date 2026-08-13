

from __future__ import annotations

from typing import Tuple

import numpy as np


DEFAULT_MASKING_COEFFICIENT = 5.0
DEFAULT_MASKING_PROBABILITY = 0.6
DEFAULT_MIN_MASKED_COORDINATES = 2
SEED = 42


def generate_client_local_masking_vector(
    size: int,
    *,
    partition_id: int,
    round_id: int,
    mu: float = DEFAULT_MASKING_COEFFICIENT,
    masking_probability: float = DEFAULT_MASKING_PROBABILITY,
    min_masked_coordinates: int = DEFAULT_MIN_MASKED_COORDINATES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one independently constructed client-local masking vector."""

    size = int(size)
    mu = float(mu)
    p = float(masking_probability)

    if size < 1:
        raise ValueError("size must be >= 1")
    if mu <= 0:
        raise ValueError("masking coefficient mu must be > 0")
    if not (0.0 < p <= 1.0):
        raise ValueError("masking_probability must satisfy 0 < p <= 1")

    target_min = max(0, min(int(min_masked_coordinates), size))
    rng = np.random.default_rng(SEED + 1009 * int(partition_id) + int(round_id))

    selection = rng.binomial(1, p, size=size).astype(np.int8)
    while int(selection.sum()) < target_min:
        selection = rng.binomial(1, p, size=size).astype(np.int8)

    selected = np.flatnonzero(selection)
    masking_vector = np.zeros(size, dtype=np.float64)

    if selected.size > 1:
        preliminary = rng.laplace(
            loc=0.0,
            scale=1.0 / mu,
            size=selected.size - 1,
        )
        masking_vector[selected[:-1]] = preliminary
        masking_vector[selected[-1]] = -float(np.sum(preliminary))

    return masking_vector, selection
