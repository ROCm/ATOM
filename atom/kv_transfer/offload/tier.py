# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The offload load contract used by both routing and admission."""

from dataclasses import dataclass

from atom.kv_transfer.offload.load_math import loadable_hit, tier_load_tokens


@dataclass(frozen=True)
class OffloadTierProfile:
    """What the configured offload tier offers DP routing, read off the registry
    once at startup (`KVConnectorFactory.offload_tier`).

    `loads`: a rank can load a prefix another DP rank computed from the tier
    (some configured transport, ``multi`` expanded, is an offload backend
    whose role both saves and loads and whose tier the DP ranks share).

    `min_load_tokens`: the shortest tier hit, beyond the HBM prefix, that the
    connector loads rather than recomputes.
    """

    loads: bool = False
    min_load_tokens: int = 0
    # The tier's transfer granularity in tokens (`register(chunk_tokens=...)`).
    load_chunk_tokens: int = 1

    def loadable_tokens(
        self, local: int, hit: int, num_prompt: int | None = None
    ) -> int:
        """Tokens a rank with `local` cached tokens loads from a tier hit of
        `hit` tokens: what the offload scheduler will load (`load_math`), so the
        router charges the rest as prefill. 0 when the load is under the
        minimum."""
        if not self.loads:
            return 0
        chunk = self.load_chunk_tokens
        return tier_load_tokens(
            local,
            loadable_hit(hit, num_prompt, chunk),
            chunk,
            self.min_load_tokens,
        )
