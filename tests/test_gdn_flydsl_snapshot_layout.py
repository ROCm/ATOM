# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The GDN checkpoint writer refuses a chunk-state layout it cannot read.

`write_state_checkpoints` reads off `h`'s strides whether its states are stored
KV or VK, and rejects any other layout before it launches. The non-GPU unit
tests run without Triton, so the module is loaded from its source with Triton
stubbed for the import.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[1] / "atom/model_ops/fla_ops/state_checkpoint.py"
)


@pytest.fixture
def state_checkpoint(monkeypatch):
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.language = types.ModuleType("triton.language")
    spec = importlib.util.spec_from_file_location("_gdn_state_checkpoint_probe", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "triton", triton)
        m.setitem(sys.modules, "triton.language", triton.language)
        spec.loader.exec_module(module)
    return module


def test_checkpoint_writer_rejects_an_unknown_snapshot_layout(state_checkpoint):
    pool = torch.zeros(2, 1, 4, 3)
    # Neither [K, V] nor [V, K] contiguous per state: every other column.
    strided = torch.zeros(1, 2, 1, 4, 6)[..., ::2]
    one = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="neither a KV nor a VK layout"):
        state_checkpoint.write_state_checkpoints(
            strided,
            pool,
            torch.zeros(64, 2),
            torch.zeros(2, 2, 3),
            one,
            one,
            one,
            one,
            one,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 64], dtype=torch.int32),
            64,
            3,
        )
