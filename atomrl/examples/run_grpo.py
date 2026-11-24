# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""GRPO launcher for AtomRL.

Checks that the config is a GRPO config, then runs it through
``atomrl.trainer.main`` exactly as ``python -m atomrl.trainer.main`` does: the
same trainer dispatch, callbacks and process-group setup (torchrun's, or a
single-rank group for one process on one GPU).

Usage:
    python examples/run_grpo.py --config configs/grpo_dense_bf16.yaml
    python examples/run_grpo.py --config configs/grpo_dense_fp8.yaml
    torchrun --nproc_per_node=8 examples/run_grpo.py --config configs/grpo_dense_fp8_8gpu.yaml
"""

from __future__ import annotations

import sys

from atomrl.core.config import AtomRLConfig
from atomrl.trainer.main import main as atomrl_main


def main() -> None:
    config = AtomRLConfig.from_cli()
    if config.algorithm.name != "grpo":
        print(
            f"Error: expected algorithm 'grpo', got '{config.algorithm.name}'",
            file=sys.stderr,
        )
        sys.exit(1)
    atomrl_main()


if __name__ == "__main__":
    main()
