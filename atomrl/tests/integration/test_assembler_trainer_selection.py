# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

from atomrl.architecture.assembly.runtime_assembler import RuntimeAssembler
from atomrl.core.config import AtomRLConfig


def test_async_flag_selects_async_trainer() -> None:
    cfg = AtomRLConfig()
    cfg.async_training.enabled = True
    assembler = RuntimeAssembler(cfg)
    graph = assembler.build_graph()
    assert graph["trainer_cls"].__name__ == "AsyncRLTrainer"
