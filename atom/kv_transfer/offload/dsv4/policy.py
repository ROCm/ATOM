# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler-side policy for DSV4 terminal checkpoints (pure decisions).

These are the Phase 0 / Phase 3 decisions the scheduler connector makes, factored
out so they can be unit-tested without a live scheduler:

* **save admission** (:func:`should_save_at`) — offload only a *fully prefilled*
  prompt whose length is 128-aligned. Non-128 prompts produce no unit this round
  (capturing at an interior 128 boundary needs the deferred intermediate-capture
  machinery). See ``dsv4-lmcache-bundle-plan.md``.
* **resume-boundary selection** (:func:`select_resume_boundary`) — the largest
  stored 128-aligned boundary strictly below the new prompt length (a full-prompt
  match still forwards >= 1 token for logits, so ``B < prompt_len``).
* **candidate probing** (:func:`candidate_boundaries`) — the 128-aligned
  boundaries a lookup probes, largest first, optionally capped.
* **unit keying** (:func:`checkpoint_key`) — a stable per-(prefix, B) key so a
  terminal unit is one LMCache object addressed by the exact prefix it restores.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable, Sequence

logger = logging.getLogger("atom")

CHECKPOINT_ALIGN = 128


def should_save_at(
    num_prompt_tokens: int,
    computed_tokens: int,
    *,
    align: int = CHECKPOINT_ALIGN,
    min_len: int = CHECKPOINT_ALIGN,
) -> bool:
    """True iff this request's fully-prefilled, 128-aligned prompt is offloadable.

    ``computed_tokens`` is the scheduler's computed frontier (``num_cached_tokens``).
    Capture happens only at prefill end (CSA ring + SWA tail are live only then)
    and only when the end is 128-aligned (HCA has no interior half-block).
    """
    B = int(num_prompt_tokens)
    if B < int(min_len) or B % int(align) != 0:
        return False
    return int(computed_tokens) >= B


def candidate_boundaries(
    prompt_len: int,
    *,
    align: int = CHECKPOINT_ALIGN,
    max_probes: int | None = None,
) -> list[int]:
    """128-aligned boundaries ``0 < B < prompt_len``, largest first.

    A full-prompt match still needs >= 1 forwarded token for logits, hence the
    strict ``B < prompt_len``. ``max_probes`` caps lookup cost for long prompts;
    when it truncates, the drop is logged (never a silent cap).
    """
    L = int(prompt_len)
    highest = ((L - 1) // int(align)) * int(align) if L > 0 else 0
    boundaries = list(range(highest, 0, -int(align)))
    if max_probes is not None and len(boundaries) > int(max_probes):
        dropped = len(boundaries) - int(max_probes)
        boundaries = boundaries[: int(max_probes)]
        logger.debug(
            "DSV4 offload: capping resume-boundary probes to %d for prompt_len=%d "
            "(dropped %d smaller candidates)",
            max_probes,
            L,
            dropped,
        )
    return boundaries


def select_resume_boundary(
    available_boundaries: Iterable[int],
    prompt_len: int,
    *,
    align: int = CHECKPOINT_ALIGN,
) -> int | None:
    """Largest stored 128-aligned boundary strictly below ``prompt_len``, or None."""
    L = int(prompt_len)
    best: int | None = None
    for b in available_boundaries:
        b = int(b)
        if b <= 0 or b >= L or b % int(align) != 0:
            continue
        if best is None or b > best:
            best = b
    return best


def checkpoint_key(
    token_ids: Sequence[int],
    B: int,
    *,
    fingerprint: bytes,
    align: int = CHECKPOINT_ALIGN,
) -> str:
    """Stable key for the terminal unit restoring prefix ``token_ids[:B]``.

    Bound to the geometry ``fingerprint`` so units from an incompatible
    model/shard/layout can never key-collide with this engine's.
    """
    B = int(B)
    if B <= 0 or B % int(align) != 0:
        raise ValueError(f"checkpoint_key: B={B} must be a positive {align}-multiple")
    if len(token_ids) < B:
        raise ValueError(
            f"checkpoint_key: token_ids has {len(token_ids)} < B={B} tokens"
        )
    h = hashlib.blake2b(digest_size=16)
    h.update(bytes(fingerprint))
    h.update(B.to_bytes(8, "little"))
    # Prefix tokens as little-endian int32 (token ids fit in 32 bits).
    prefix = memoryview(
        b"".join(int(t).to_bytes(4, "little", signed=False) for t in token_ids[:B])
    )
    h.update(prefix)
    return f"dsv4:{B}:{h.hexdigest()}"
