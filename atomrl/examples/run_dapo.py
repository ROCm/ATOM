# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DAPO launcher for AtomRL.

Checks that the config is a DAPO config, then runs it through
``atomrl.trainer.main`` exactly as ``python -m atomrl.trainer.main`` does: the
same trainer dispatch, callbacks and process-group setup (torchrun's, or a
single-rank group for one process on one GPU).

Usage:
    python examples/run_dapo.py --config configs/dapo_dense_fp8.yaml
"""

from __future__ import annotations

import sys

from atomrl.core.config import AtomRLConfig
from atomrl.trainer.main import main as atomrl_main


def main() -> None:
    config = AtomRLConfig.from_cli()
    if config.algorithm.name != "dapo":
        print(
            f"Error: expected algorithm 'dapo', got '{config.algorithm.name}'",
            file=sys.stderr,
        )
        sys.exit(1)
    atomrl_main()


if __name__ == "__main__":
    main()
