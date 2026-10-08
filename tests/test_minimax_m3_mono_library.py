# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public library refusals and engine import isolation, without model weights."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.minimax_m3.mono.library import AtomM3Mono, TPContext, validate_layer


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


@pytest.mark.parametrize(
    "field,dtype", [("w_qkv", torch.bfloat16), ("gate", torch.float32)]
)
def test_library_rejects_weights_that_would_require_conversion(field, dtype):
    # Meta tensors exercise the storage contract without allocating expert weights.
    def tensor(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    spec = SimpleNamespace(
        layer_id=3,
        w_qkv=tensor((2560, 6144), torch.float8_e4m3fn),
        s_qkv=tensor((2560,), torch.float32),
        w_o=tensor((6144, 2048), torch.float8_e4m3fn),
        s_o=tensor((6144,), torch.float32),
        gate=tensor((128, 6144)),
        g_in=tensor((6144,)),
        g_post=tensor((6144,)),
        g_q=tensor((128,)),
        g_k=tensor((128,)),
        g_iq=tensor((128,)),
        g_ik=tensor((128,)),
        cos_sin=tensor((16384, 64)),
        bias=tensor((128,), torch.float32),
        w13=tensor((129, 1536, 3072), torch.float4_e2m1fn_x2),
        s13=tensor((129, 1536, 192), torch.uint8),
        w2=tensor((129, 6144, 384), torch.float4_e2m1fn_x2),
        s2=tensor((129, 6144, 24), torch.uint8),
    )
    validate_layer(spec, torch.device("meta"))
    bad_weight = tensor(getattr(spec, field).shape, dtype)
    setattr(spec, field, bad_weight)
    with pytest.raises(ValueError, match="expected contiguous aligned"):
        validate_layer(spec, torch.device("meta"))
    assert getattr(spec, field) is bad_weight
