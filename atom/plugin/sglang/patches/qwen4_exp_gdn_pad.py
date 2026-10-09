"""Native GDN pad sentinels for Qwen3.8-Flash-Next on SGLang.

SGLang 0.5.20 Hybrid already pads Flash GDN like Qwen3.8-2.4T
(``_replay_metadata`` / ``_forward_metadata`` write ``mamba_cache_indices = -1``
on pad rows). Native GDN still reconstructs or clones metadata and can keep a
finished request's mamba slot. Keep this helper until Native GDN consumes
Hybrid's already-padded buffers directly (same as 2.4T), then delete this
file and its call in ``attention_gdn.py``.
"""

from __future__ import annotations

from typing import Any

import torch

from atom.plugin.sglang.attention_backend.backend_resolver import real_batch_size


def apply_gdn_pad_sentinels(
    forward_batch: Any,
    idx: torch.Tensor,
    query_start_loc: torch.Tensor,
    mode: Any,
    bs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mark CUDA-graph / DP pad rows as GDN no-write (``PAD_SLOT_ID = -1``).

    Clones so Hybrid's static CUDA-graph buffers are not mutated in place.
    """

    live_bs = real_batch_size(forward_batch)
    if live_bs < idx.shape[0]:
        idx = idx.clone()
        idx[live_bs:] = -1
    if (
        mode.is_decode_or_idle()
        and live_bs < bs
        and query_start_loc.numel() > live_bs + 1
    ):
        query_start_loc = query_start_loc.clone()
        query_start_loc[live_bs + 1 :] = live_bs
    return idx, query_start_loc
