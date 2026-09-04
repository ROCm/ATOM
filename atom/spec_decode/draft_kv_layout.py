# SPDX-License-Identifier: MIT
"""Which KV layout an Eagle3 MHA draft's pool should be allocated in.

Split out of `eagle3_kv_builder` so it can be tested: that module does
`from aiter import dtypes` at import time, and CI has no aiter, so a test
reaching the predicate through it is skipped -- on the very CI whose OOM this
fixes. This module needs nothing that pulls aiter, so it imports there.
"""

from __future__ import annotations


def use_flash_layout(impl) -> bool:
    """Whether to hand this draft module a flash (4D) view of the pool.

    Mirrors `rope_cache`'s branch condition: of its three writers only
    `fused_qk_rope_reshape_and_cache` emits the 4D V that no prefill reader
    consumes, so only those modules get flash -- same kernel, one flag flipped.
    A module reaching either other writer already gets a SHUFFLE V, which a
    flash pool would only break.

    KNOWN GAP: returns False for `rotary_emb is None`, which is right unless
    `use_triton_attn` also holds -- then rope_cache's third writer takes
    asm_layout=False and emits the same 4D V, so that draft keeps paying the
    whole-pool convert. Not fixed here: that writer is `reshape_and_cache`,
    with its own blast radius. See test_rope_less_draft_is_a_known_gap.

    `SparseMHAPagedAttentionImpl` overrides `rope_cache` wholesale and hardcodes
    SHUFFLE; it sets `use_triton_attn = False` in `__init__` too, so reading the
    flag classifies it correctly at bind time rather than relying on its q/k
    norms to trip the early return above.
    """
    if impl is None or getattr(impl, "rotary_emb", None) is None:
        return False
    # rope_cache's first branch re-views V to SHUFFLE itself.
    if (
        getattr(impl, "q_norm", None) is not None
        and getattr(impl, "k_norm", None) is not None
    ):
        return False
    # The same flag `rope_cache` branches on, set in PagedAttentionImpl.__init__
    # so it is readable here -- KV binding runs before any forward.
    return bool(impl.use_triton_attn)
