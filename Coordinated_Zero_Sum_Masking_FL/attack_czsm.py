"""Independent reconstruction-attack setup 3: Coordinated Zero-Sum Masking."""

import argparse
import numpy as np
import torch

from attack_common import (
    add_common_arguments,
    prepare_attack,
    save_attack_outputs,
)
from masking_coordinator import CoordinatedZeroSumMaskingCoordinator


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gradient inversion under the Coordinated Zero-Sum Masking "
            "(CZSM) condition."
        )
    )
    add_common_arguments(parser)
    parser.add_argument("--masking-coefficient", type=float, default=5.0)
    parser.add_argument("--masking-probability", type=float, default=1.0)
    parser.add_argument("--min-masked-coordinates", type=int, default=2)
    parser.add_argument("--coordinator-seed", type=int, default=42)
    args = parser.parse_args()

    model, original_x, target_y, clean_gradient = prepare_attack(args)

    client_ids = tuple(str(i) for i in range(args.num_clients))
    coordinator = CoordinatedZeroSumMaskingCoordinator(
        seed=args.coordinator_seed
    )
    assignment = coordinator.generate(
        round_id=args.round_id,
        num_coordinates=clean_gradient.numel(),
        client_ids=client_ids,
        mu=args.masking_coefficient,
        masking_probability=args.masking_probability,
        min_masked_coordinates=args.min_masked_coordinates,
    )

    # The attacker/server receives only the target client's masked gradient;
    # it is not given the target client's coordinator-assigned noise vector.
    masking_vector = assignment.column_for(str(args.target_client))
    observed_gradient = clean_gradient + torch.tensor(
        masking_vector,
        dtype=clean_gradient.dtype,
    )

    # Diagnostic check only. This verifies the full assignment is zero-sum;
    # the attack itself never receives the other clients' columns.
    residual = np.sum(assignment.matrix, axis=1)
    print(
        "[CZSM coordinator diagnostic] full-matrix zero-sum residual L2 = "
        f"{np.linalg.norm(residual):.8e}"
    )

    save_attack_outputs(
        setup="czsm",
        args=args,
        model=model,
        original_x=original_x,
        target_y=target_y,
        clean_gradient=clean_gradient,
        observed_gradient=observed_gradient,
        masking_vector=np.asarray(masking_vector, dtype=np.float64),
        selected_coordinates=int(assignment.coordinate_selection.sum()),
    )


if __name__ == "__main__":
    main()
