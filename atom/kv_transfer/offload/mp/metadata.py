# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache MP wire tokens and worker-resolved request metadata."""

from dataclasses import dataclass

from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta
from atom.kv_transfer.offload.mp.token_shipping import ShippedTokens, TokenPrefix


@dataclass
class MPReqMeta(LMCacheReqMeta):
    """Tokens are shipped incrementally, then resolved before transfer submission."""

    token_ids: list[int] | ShippedTokens | TokenPrefix


class MPOffloadMetadata(LMCacheOffloadMetadata):
    """An MP step also delivers token-copy retirements without any transfer."""

    WORK_FIELDS = LMCacheOffloadMetadata.WORK_FIELDS + ("token_drops",)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[MPReqMeta] = []
        self.token_drops: tuple = ()
