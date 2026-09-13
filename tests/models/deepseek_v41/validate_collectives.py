# SPDX-License-Identifier: MIT
"""Check the offline collective policy under changing shapes and rank skew.

Run with torchrun -m tests.models.deepseek_v41.validate_collectives. Integer
inputs have exact BF16/FP32 sums, so no numerical tolerance can hide stale data.
"""

import json

import torch
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
)

from atom.examples.deepseek_v41_offline import initialize_parallel


def main():
    rank = initialize_parallel()
    group = get_tp_group()
    try:
        errors = []
        for iteration in range(240):
            dtype = torch.bfloat16 if iteration % 3 == 0 else torch.float32
            rows = (1, 32, 250, 267, 640)[iteration % 5]
            values = torch.full(
                (rows, 5120), (iteration + rank) % 7, dtype=dtype, device="cuda"
            )
            # Whole-expert partitioning gives ranks unequal work between
            # collectives. Preserve that skew without a barrier per call.
            for _ in range(rank * 2):
                values = values + 0
            result = group.all_reduce(values, ca_fp8_quant=False)
            expected = sum((iteration + peer) % 7 for peer in range(group.world_size))
            errors.append((result - expected).abs().max())
        errors = torch.stack(errors).cpu()
        failures = (errors != 0).nonzero().flatten().tolist()
        if failures:
            raise AssertionError(f"Incorrect collective sums at iterations {failures}")
        if rank == 0:
            print(
                json.dumps(
                    {"calls": len(errors), "tp_size": group.world_size, "max_error": 0}
                )
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
