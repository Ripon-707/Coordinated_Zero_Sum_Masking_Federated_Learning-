"""Independent reconstruction-attack setup 2: Client-Local Masking."""

import argparse
import numpy as np
import torch

from attack_common import (
    add_common_arguments,
    prepare_attack,
    save_attack_outputs,
)
from client_local_masking import generate_client_local_masking_vector


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gradient inversion under the independent Client-Local Masking "
            "condition."
        )
    )
    add_common_arguments(parser)
    parser.add_argument("--masking-coefficient", type=float, default=5.0)
    parser.add_argument("--masking-probability", type=float, default=0.6)
    parser.add_argument("--min-masked-coordinates", type=int, default=2)
    args = parser.parse_args()

    model, original_x, target_y, clean_gradient = prepare_attack(args)

    masking_vector, selection = generate_client_local_masking_vector(
        clean_gradient.numel(),
        partition_id=args.target_client,
        round_id=args.round_id,
        mu=args.masking_coefficient,
        masking_probability=args.masking_probability,
        min_masked_coordinates=args.min_masked_coordinates,
    )

    observed_gradient = clean_gradient + torch.tensor(
        masking_vector,
        dtype=clean_gradient.dtype,
    )

    save_attack_outputs(
        setup="client_local_masking",
        args=args,
        model=model,
        original_x=original_x,
        target_y=target_y,
        clean_gradient=clean_gradient,
        observed_gradient=observed_gradient,
        masking_vector=np.asarray(masking_vector, dtype=np.float64),
        selected_coordinates=int(selection.sum()),
    )


if __name__ == "__main__":
    main()
