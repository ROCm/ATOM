# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public library refusals and engine import isolation, without model weights."""

import os
import subprocess
import sys

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.minimax_m3.mono.library import AtomM3Mono, TPContext


def test_library_imports_without_serving_engines():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from atom.models.minimax_m3.mono.library import AtomM3Mono
assert not any(m.startswith(('atom.model_engine', 'atom.plugin', 'atom.model_ops', 'vllm'))
               for m in sys.modules)
""",
        ],
        env={**os.environ, "ATOM_DISABLE_VLLM_PLUGIN": "1", "VLLM_PLUGINS": ""},
        check=True,
    )


def test_library_refuses_unsupported_tp_before_allocating():
    with pytest.raises(ValueError, match="requires TP4"):
        AtomM3Mono([], [], TPContext(None, 0, 2, torch.device("cpu")))


def test_library_refuses_functionalized_execution(monkeypatch):
    # Device pointer descriptors cannot follow functionalization's replacement
    # tensors. Refuse this path instead of silently writing the old storage.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    runtime = AtomM3Mono.__new__(AtomM3Mono)
    with pytest.raises(ValueError, match="torch.compile is unsupported"):
        runtime.prepare_step(None)
    with pytest.raises(ValueError, match="torch.compile is unsupported"):
        runtime.forward_layer(3, None, None, None)
