# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Speculative Decoding Draft Model Distillation launcher for AtomRL.

Trains Eagle3 or DFlash draft models on teacher hidden states, following the
KimiV2.5 approach.

Checks that the config is a spec_distill config, then runs it through
``atomrl.trainer.main`` exactly as ``python -m atomrl.trainer.main`` does: the
same trainer dispatch, callbacks and process-group setup (torchrun's, or a
single-rank group for one process on one GPU).

Usage:
    python examples/run_spec_distill.py --config configs/spec_distill_eagle3.yaml
    python examples/run_spec_distill.py --config examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/configs/smoke_test_hf.yaml
"""

from __future__ import annotations

import sys

from atomrl.core.config import AtomRLConfig
from atomrl.trainer.main import main as atomrl_main


def main() -> None:
    config = AtomRLConfig.from_cli()
    if config.algorithm.name != "spec_distill":
        print(
            f"Error: expected algorithm 'spec_distill', got '{config.algorithm.name}'",
            file=sys.stderr,
        )
        sys.exit(1)
    atomrl_main()


if __name__ == "__main__":
    main()
