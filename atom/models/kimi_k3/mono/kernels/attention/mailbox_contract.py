# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Source-evidenced gfx1250 memory-operation capabilities for Kimi K1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    DEVICE_MAILBOX = "device_mailbox"
    STORE_DRAIN = "store_drain"
    WORKGROUP_BARRIER = "workgroup_barrier"
    SYSTEM_EPOCH = "system_epoch"
    COHERENT_CPOL_17 = "coherent_cpol_17"
    PEER_TAGGED_PAIR = "peer_tagged_pair"


@dataclass(frozen=True)
class MemoryEvidence:
    capability: Capability
    supported: bool
    source: str
    lines: str
    contract: str


EVIDENCE = (
    MemoryEvidence(
        Capability.DEVICE_MAILBOX,
        True,
        "aiter/ops/flydsl/kernels/fused_compress_attn_hca_gfx1250.py",
        "43-46, 570, 624-649, 677-699",
        (
            "CPol SCOPE_DEV=16 writes through per-CU caches to L2 and matching "
            "loads read sibling-CTA stores at the device coherence point"
        ),
    ),
    MemoryEvidence(
        Capability.STORE_DRAIN,
        True,
        "aiter/ops/flydsl/kernels/communication_ops_utils.py",
        "86-98",
        "gfx12 store publication drains with s_wait_storecnt(0)",
    ),
    MemoryEvidence(
        Capability.WORKGROUP_BARRIER,
        True,
        "aiter/ops/flydsl/kernels/gemm_common_gfx1250.py",
        "121-132",
        "gpu.barrier is the gfx1250 non-cluster workgroup barrier",
    ),
    MemoryEvidence(
        Capability.SYSTEM_EPOCH,
        True,
        "aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py",
        "275-290, 383-401, 507-512",
        (
            "gfx1250 uses system atomic arrival plus system release/acquire "
            "fences for cross-PE generation handoff"
        ),
    ),
    MemoryEvidence(
        Capability.COHERENT_CPOL_17,
        True,
        "aiter/ops/flydsl/kernels/splitk_epilogue.py",
        "15-19, 82-88, 124",
        "CPol sc0|sc1 (0x11) is coherent across XCDs",
    ),
    MemoryEvidence(
        Capability.PEER_TAGGED_PAIR,
        False,
        "no checked-in gfx1250 AITER/ATOM implementation",
        "-",
        (
            "no source proves an XGMI peer payload+tag vector store/poll with "
            "the mono mailbox's exact atomicity and cache policy"
        ),
    ),
)


@dataclass(frozen=True)
class MailboxCapabilities:
    architecture: str
    evidence: tuple[MemoryEvidence, ...] = EVIDENCE

    def evidence_for(self, capability: Capability) -> MemoryEvidence:
        return next(item for item in self.evidence if item.capability is capability)

    def require(self, *capabilities: Capability) -> None:
        unsupported = [
            capability.value
            for capability in capabilities
            if not self.evidence_for(capability).supported
        ]
        if unsupported:
            raise ValueError(
                f"{self.architecture} unsupported mailbox capabilities: "
                f"{', '.join(unsupported)}"
            )

    @property
    def supported(self) -> tuple[Capability, ...]:
        return tuple(item.capability for item in self.evidence if item.supported)

    @property
    def unsupported(self) -> tuple[Capability, ...]:
        return tuple(item.capability for item in self.evidence if not item.supported)


GFX1250_MAILBOX = MailboxCapabilities("gfx1250")
K1_REQUIRED_CAPABILITIES = (
    Capability.DEVICE_MAILBOX,
    Capability.STORE_DRAIN,
    Capability.WORKGROUP_BARRIER,
    Capability.SYSTEM_EPOCH,
)
