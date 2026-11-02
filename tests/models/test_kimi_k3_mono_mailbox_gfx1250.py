# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU/static contracts for gfx1250 K1 mailbox memory operations."""

from pathlib import Path

import pytest

from atom.models.kimi_k3.mono.kernels.attention.mailbox_contract import (
    GFX1250_MAILBOX,
    K1_REQUIRED_CAPABILITIES,
    Capability,
)


def test_k1_required_memory_capabilities_are_source_evidenced():
    GFX1250_MAILBOX.require(*K1_REQUIRED_CAPABILITIES)
    assert set(K1_REQUIRED_CAPABILITIES) <= set(GFX1250_MAILBOX.supported)
    for capability in K1_REQUIRED_CAPABILITIES:
        evidence = GFX1250_MAILBOX.evidence_for(capability)
        assert evidence.supported
        assert evidence.source.endswith(".py")
        assert evidence.lines != "-"
        assert evidence.contract


def test_peer_tagged_pair_remains_an_explicit_capability_boundary():
    evidence = GFX1250_MAILBOX.evidence_for(Capability.PEER_TAGGED_PAIR)
    assert not evidence.supported
    assert "XGMI" in evidence.contract
    assert GFX1250_MAILBOX.unsupported == (Capability.PEER_TAGGED_PAIR,)
    with pytest.raises(ValueError, match="peer_tagged_pair"):
        GFX1250_MAILBOX.require(Capability.PEER_TAGGED_PAIR)


def test_coherent_cpol17_is_not_misrepresented_as_peer_pair_proof():
    coherent = GFX1250_MAILBOX.evidence_for(Capability.COHERENT_CPOL_17)
    peer = GFX1250_MAILBOX.evidence_for(Capability.PEER_TAGGED_PAIR)
    assert coherent.supported and "XCD" in coherent.contract
    assert not peer.supported


def test_device_source_uses_gfx12_store_counter_and_evidenced_scopes():
    source = (
        Path(__file__).parents[2]
        / "atom"
        / "models"
        / "kimi_k3"
        / "mono"
        / "kernels"
        / "attention"
        / "kda_gfx1250.py"
    ).read_text()
    assert "rocdl.s_wait_storecnt(0)" in source
    assert "rocdl.s_waitcnt(vmcnt=0)" not in source
    assert "cache_modifier=CM_DEV" in source
    assert "comm_ops.store_i32_global_system_release" in source
    assert "comm_ops.load_i32_global_system(address, acquire=True)" in source
    assert "comm_ops.fence_system_release()" in source
    assert "comm_ops.fence_system_acquire()" in source
    assert "comm_ops.atomic_add_system(address, value)" in source
    publish = source.index("def mailbox_publish_device")
    drain = source.index("wait_vector_stores()", publish)
    barrier = source.index("workgroup_barrier()", drain)
    store = source.index("mailbox_store_device(", barrier)
    assert publish < drain < barrier < store


def test_role_lds_emitter_is_architecture_separate_and_consumes_spec():
    source = (
        Path(__file__).parents[2]
        / "atom"
        / "models"
        / "kimi_k3"
        / "mono"
        / "kernels"
        / "attention"
        / "kda_gfx1250.py"
    ).read_text()
    for declaration in (
        "@fx.struct\n    class AttnLds:",
        "@fx.struct\n    class GemvLds:",
        "@fx.struct\n    class KdaLds:",
        "@fx.union\n    class RoleLds:",
        "@fx.struct\n    class Smem:",
    ):
        assert declaration in source
    assert "def emit_role_lds_types(spec):" in source
