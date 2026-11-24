# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import pytest
from atomrl.engine.inference.atom_engine import AtomEngine
from atomrl.engine.inference.vllm_engine import VLLMEngine


class BackendShutdownError(Exception):
    pass


class FailingProcess:
    pid = 12345

    @staticmethod
    def poll():
        return None

    @staticmethod
    def terminate():
        raise BackendShutdownError("backend terminate failed")

    @staticmethod
    def wait(timeout):
        raise AssertionError(f"wait({timeout}) should not be reached")

    @staticmethod
    def kill():
        raise BackendShutdownError("backend kill failed")


@pytest.mark.parametrize("engine_type", [AtomEngine, VLLMEngine])
def test_engine_shutdown_is_best_effort_for_backend_failures(
    monkeypatch, engine_type
) -> None:
    engine = object.__new__(engine_type)
    engine._proc = FailingProcess()
    engine._cmd_f = None
    engine._resp_f = None
    engine._cmd_fifo = None
    engine._resp_fifo = None
    engine._initialized = True
    engine._weight_dir = "weights"
    engine._send_cmd = lambda _command: (_ for _ in ()).throw(
        BackendShutdownError("backend command failed")
    )
    if engine_type is VLLMEngine:
        monkeypatch.setattr(
            "atomrl.engine.inference.vllm_engine._kill_proc_tree", lambda _pid: None
        )

    engine.shutdown()

    assert engine._proc is None
    assert engine._initialized is False
    assert engine._weight_dir is None
