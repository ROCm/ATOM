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
def impl():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            topk=load("atom.model_ops.topK"),
            mori_v2=load("atom.model_ops.fused_moe.mori_v2_prepare_finalize"),
        )


def _fused(monkeypatch, impl, *, eplb, switch, transport, backend="auto", ep=True):
    topk = impl.topk
    monkeypatch.setattr(topk.envs, "ATOM_FUSE_SHARED_EXPERT", switch)
    monkeypatch.setattr(topk.envs, "ATOM_DISABLE_MORI_EP", False)
    monkeypatch.setattr(topk.envs, "ATOM_MORI_V2", transport != "v1")
    monkeypatch.setattr(impl.mori_v2, "_MEGA_STAGE1_FUSED", transport == "v2 compact")
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
    "eplb, switch, transport, expected",
    [
        # EPLB fuses the shared expert as a routed expert.
        (True, False, "v1", True),
        (True, False, "v2 mori", True),
        # Without EPLB it is a LOCAL_REPLICA: MoRI v1 and MegaMoE's compact
        # dispatch follow the switch...
        (False, False, "v1", False),
        (False, True, "v1", True),
        (False, False, "v2 compact", False),
        (False, True, "v2 compact", True),
        # ...and MegaMoE's MoRI dispatch never fuses it.
        (False, True, "v2 mori", False),
    ],
)
def test_mori_shared_expert_fusion(
    monkeypatch, impl, eplb, switch, transport, expected
):
    fused = _fused(monkeypatch, impl, eplb=eplb, switch=switch, transport=transport)
    assert fused is expected


@pytest.mark.parametrize("backend, ep", [("rccl", True), ("auto", False)])
def test_gate_is_only_for_mori_ep(monkeypatch, impl, backend, ep):
    """RCCL, or DP attention without EP, keeps fusing on MoRI v2 with the
    switch off."""
    fused = _fused(
        monkeypatch,
        impl,
        eplb=False,
        switch=False,
        transport="v2 mori",
        backend=backend,
        ep=ep,
    )
    assert fused is True
