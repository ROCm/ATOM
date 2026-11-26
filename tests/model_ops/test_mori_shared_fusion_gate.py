# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Whether DP attention + EP over MoRI fuses the shared expert into the routed
MoE. CPU only: `topK` is imported against the AITER stand-ins where AITER
cannot run."""

from types import SimpleNamespace

import pytest
from aiter_stub import atom_without_aiter
from conftest import atom_config_double


@pytest.fixture(scope="module")
def topk():
    with atom_without_aiter() as load:
        yield load("atom.model_ops.topK")


def _fused(monkeypatch, topk, *, eplb, switch, mori_v2, backend="auto", ep=True):
    monkeypatch.setattr(topk.envs, "ATOM_FUSE_SHARED_EXPERT", switch)
    monkeypatch.setattr(topk.envs, "ATOM_DISABLE_MORI_EP", False)
    monkeypatch.setattr(topk.envs, "ATOM_MORI_V2", mori_v2)
    # Stated, not inherited: the gate asks whether MoRI is importable.
    monkeypatch.setattr(topk, "_has_module", lambda name: True)
    config = atom_config_double(
        quant_config=SimpleNamespace(exclude_layers=[], quant_dtype=None),
        parallel_config=SimpleNamespace(data_parallel_size=8),
        eplb_enable=eplb,
        moe_all2all_backend=backend,
        enable_dp_attention=True,
        enable_expert_parallel=ep,
    )
    monkeypatch.setattr(topk, "get_current_atom_config", lambda: config)
    return topk.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(None)


@pytest.mark.parametrize(
    "eplb, switch, mori_v2, expected",
    [
        # EPLB fuses the shared expert as a routed expert.
        (True, False, False, True),
        (True, False, True, True),
        # Without EPLB it is a LOCAL_REPLICA: MoRI v1 follows the switch...
        (False, False, False, False),
        (False, True, False, True),
        # ...and MoRI v2 never fuses it.
        (False, True, True, False),
    ],
)
def test_mori_shared_expert_fusion(monkeypatch, topk, eplb, switch, mori_v2, expected):
    fused = _fused(monkeypatch, topk, eplb=eplb, switch=switch, mori_v2=mori_v2)
    assert fused is expected


@pytest.mark.parametrize("backend, ep", [("rccl", True), ("auto", False)])
def test_gate_is_only_for_mori_ep(monkeypatch, topk, backend, ep):
    """RCCL, or DP attention without EP, keeps fusing on MoRI v2 with the
    switch off."""
    fused = _fused(
        monkeypatch,
        topk,
        eplb=False,
        switch=False,
        mori_v2=True,
        backend=backend,
        ep=ep,
    )
    assert fused is True
