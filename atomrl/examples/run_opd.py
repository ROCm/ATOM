# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""On-Policy Distillation (OPD) launcher for AtomRL.

Implements DeepSeek-V4 style on-policy distillation: student rollout,
teacher forward, then KL(student || teacher) training.

Checks that the config is an OPD config, then runs it through
``atomrl.trainer.main`` exactly as ``python -m atomrl.trainer.main`` does: the
same trainer dispatch, callbacks and process-group setup (torchrun's, or a
single-rank group for one process on one GPU).

Usage:
    python examples/run_opd.py --config configs/opd_dense_bf16.yaml
"""

from __future__ import annotations

import sys

from atomrl.core.config import AtomRLConfig
from atomrl.trainer.main import main as atomrl_main


def main() -> None:
    config = AtomRLConfig.from_cli()
    if config.algorithm.name != "opd":
        print(
            f"Error: expected algorithm 'opd', got '{config.algorithm.name}'",
            file=sys.stderr,
        )
        sys.exit(1)
    atomrl_main()


if __name__ == "__main__":
    main()
