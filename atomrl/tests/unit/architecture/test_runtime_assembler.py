# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

from atomrl.architecture.assembly.runtime_assembler import RuntimeAssembler
from atomrl.core.config import AtomRLConfig


def test_assembler_resolves_default_actor_worker() -> None:
    cfg = AtomRLConfig()
    assembler = RuntimeAssembler(cfg)
    graph = assembler.build_graph()
    assert "actor_worker_cls" in graph
    assert graph["actor_worker_cls"].__name__ == "AtomActorWorker"
