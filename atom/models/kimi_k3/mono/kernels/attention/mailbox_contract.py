# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Memory-operation capabilities used by the local gfx1250 K1 pipeline."""

from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    DEVICE_MAILBOX = "device mailbox"
    STORE_DRAIN = "store drain"
    DEVICE_ACQUIRE = "device acquire"
    WORKGROUP_BARRIER = "workgroup barrier"
    SYSTEM_EPOCH = "system epoch"
    COHERENT_CPOL_17 = "coherent cpol 0x11"
    PEER_TAGGED_PAIR = "peer tagged pair"


@dataclass(frozen=True)
class MailboxCapabilities:
    architecture: str
    supported: tuple[Capability, ...]

    def require(self, *capabilities: Capability) -> None:
        unsupported = [
            capability.value
            for capability in capabilities
            if capability not in self.supported
        ]
        if unsupported:
            raise ValueError(
                f"{self.architecture} unsupported mailbox capabilities: "
                f"{', '.join(unsupported)}"
            )


GFX1250_MAILBOX = MailboxCapabilities(
    "gfx1250",
    (
        Capability.DEVICE_MAILBOX,
        Capability.STORE_DRAIN,
        Capability.DEVICE_ACQUIRE,
        Capability.WORKGROUP_BARRIER,
        Capability.SYSTEM_EPOCH,
        Capability.COHERENT_CPOL_17,
    ),
)
K1_REQUIRED_CAPABILITIES = (
    Capability.DEVICE_MAILBOX,
    Capability.STORE_DRAIN,
    Capability.DEVICE_ACQUIRE,
    Capability.WORKGROUP_BARRIER,
    Capability.SYSTEM_EPOCH,
)
