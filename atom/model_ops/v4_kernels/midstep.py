# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Midstep state checkpoints for DeepSeek-V4 prefill.

A V4 request's per-request state is a set of rings indexed by absolute
position: the SWA window (the last `window` tokens' raw KV, per layer, plus the
DSpark draft window) and the compressor rings (the last `K_pool` tokens'
pre-pool kv/score, per Compressor). A checkpoint at prompt position `P` is the
slot as a forward ending exactly at `P` would leave it.

Without this module the scheduler gets that state by cutting the prefill chunk
at `P`, which costs a whole lockstep step per checkpoint. Every byte the
checkpoint needs is already materialized inside the uncut forward, though:
the per-token SWA KV (attention needs it) and the per-token compressor kv/score
(`wkv_gate` runs over the whole chunk). Only the ring *writes* are limited to
the chunk's tail. So a midstep checkpoint is

1. copy the request's incoming slot into a scratch slot (the rows before the
   chunk start, which the cut forward would also have inherited), then
2. replay the same ring writes the cut forward would have issued -- the tokens
   of `[start, P)` that fall in each ring's window -- into the scratch slot.

Step 2 calls the very kernels the forward uses (`swa_write`,
`update_compressor_states`), so the scratch slot ends up byte-identical to the
slot a forward cut at `P` would leave. The scheduler then files the scratch
slot as a PAGE checkpoint exactly as it would file a cut forward's slot.

`build_midstep_plan` is pure numpy so the bookkeeping is testable without a
GPU; the write helpers are no-ops unless the forward context carries a plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# Rows gathered per checkpoint for the SWA replay. Must cover every ring a
# replay writes (`write_per_batch <= ring_slots`); V4-Pro rings are
# `window_size + max_spec_steps` = 131. The replay itself writes only the last
# `write_per_batch` of these, exactly like the forward.
SWA_GATHER_ROWS = 256


@dataclass
class MidstepHostPlan:
    """Host-side description of one forward's midstep checkpoints.

    One entry per honoured reservation `r`, in batch order:
      `dst_slots[r]`   scratch slot the checkpoint is written into,
      `src_slots[r]`   the request's incoming slot (copied into `dst` first),
      `swa_rows`       flat token rows to gather, concatenated per `r`,
      `swa_cu[r:r+2]`  the span of `r` within `swa_rows`,
      `compress_plans[ratio]` write-plan rows `(ragged_id, r, position, 0)`.
    """

    dst_slots: np.ndarray
    src_slots: np.ndarray
    swa_rows: np.ndarray
    swa_cu: np.ndarray
    compress_plans: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def num(self) -> int:
        return int(self.dst_slots.shape[0])


def build_midstep_plan(
    saves: Sequence[Sequence[tuple[int, int]]],
    chunk_starts: Sequence[int],
    cu_seqlens_q: np.ndarray,
    src_slots: Sequence[int],
    ratios_overlap: Sequence[tuple[int, bool]],
    *,
    extra_write: int = 0,
    swa_gather_rows: int = SWA_GATHER_ROWS,
) -> MidstepHostPlan | None:
    """Plan the replay writes for every midstep reservation of a prefill.

    `saves[i]` lists `(scratch_slot, position)` for batch row `i`, the
    scheduler's `state_save_all`. A position must lie in `(start, end]` of that
    row's chunk; anything else cannot be produced by this forward, and a
    reservation that is published without its bytes would be a findable,
    silently wrong checkpoint -- so it raises instead of skipping.
    """
    dst, src, rows, cu = [], [], [], [0]
    plan_rows: dict[int, list[tuple[int, int, int, int]]] = {
        r: [] for r, _ in ratios_overlap
    }
    k_pool = {
        r: r * (2 if ov else 1) + extra_write for r, ov in ratios_overlap
    }
    for i, reservations in enumerate(saves):
        if not reservations:
            continue
        start = int(chunk_starts[i])
        base = int(cu_seqlens_q[i])
        end = start + int(cu_seqlens_q[i + 1]) - base
        for slot, pos in reservations:
            slot, pos = int(slot), int(pos)
            if not start < pos <= end:
                raise ValueError(
                    f"midstep checkpoint at {pos} lies outside this forward's "
                    f"chunk ({start}, {end}] for batch row {i}"
                )
            if slot < 0:
                raise ValueError(f"midstep checkpoint at {pos} has no slot")
            r = len(dst)
            dst.append(slot)
            src.append(int(src_slots[i]))
            n_in_chunk = pos - start
            take = min(n_in_chunk, swa_gather_rows)
            stop = base + n_in_chunk
            rows.extend(range(stop - take, stop))
            cu.append(cu[-1] + take)
            for ratio, _ in ratios_overlap:
                lo = max(start, pos - k_pool[ratio])
                for p in range(lo, pos):
                    plan_rows[ratio].append((base + (p - start), r, p, 0))
    if not dst:
        return None
    return MidstepHostPlan(
        dst_slots=np.asarray(dst, dtype=np.int32),
        src_slots=np.asarray(src, dtype=np.int32),
        swa_rows=np.asarray(rows, dtype=np.int64),
        swa_cu=np.asarray(cu, dtype=np.int32),
        compress_plans={
            ratio: np.asarray(v, dtype=np.int32).reshape(-1, 4)
            for ratio, v in plan_rows.items()
        },
    )


@dataclass
class MidstepPlan:
    """Device copy of `MidstepHostPlan`, carried on the attention metadata."""

    dst_slots: "torch.Tensor"  # noqa: F821  [R] int32
    swa_rows: "torch.Tensor"  # noqa: F821  [N] int64
    swa_cu: "torch.Tensor"  # noqa: F821  [R+1] int32
    compress_plans: dict  # ratio -> [n, 4] int32
    max_rows: int  # largest per-checkpoint SWA span


def to_device(host: MidstepHostPlan, device) -> MidstepPlan:
    import torch

    spans = np.diff(host.swa_cu)
    return MidstepPlan(
        dst_slots=torch.from_numpy(host.dst_slots).to(device),
        swa_rows=torch.from_numpy(host.swa_rows).to(device),
        swa_cu=torch.from_numpy(host.swa_cu).to(device),
        compress_plans={
            r: torch.from_numpy(p).to(device)
            for r, p in host.compress_plans.items()
            if p.shape[0]
        },
        max_rows=int(spans.max()) if spans.size else 0,
    )


def current_midstep_plan() -> MidstepPlan | None:
    from atom.utils.forward_context import get_forward_context

    md = getattr(get_forward_context(), "attn_metadata", None)
    return getattr(md, "v4_midstep", None)


def midstep_swa_write(
    kv,
    positions,
    pool,
    window,
    write_per_batch: int,
    *,
    k_packed=None,
    k_rope=None,
    pool_rope=None,
    prefix: str = "",
) -> None:
    """Replay a forward's `swa_write` into every midstep scratch slot.

    Same arguments as the `swa_write` it follows. Each checkpoint becomes one
    virtual request whose tokens end at its position; `swa_write` keeps the
    last `write_per_batch` of them, exactly as a forward ending there would.
    """
    plan = current_midstep_plan()
    if plan is None:
        return
    from atom.model_ops.v4_kernels.state_writes import swa_write

    idx = plan.swa_rows
    swa_write(
        kv[idx] if kv is not None else None,
        positions[idx],
        plan.swa_cu,
        plan.dst_slots,
        pool,
        window,
        max(1, min(write_per_batch, plan.max_rows)),
        k_packed=k_packed[idx] if k_packed is not None else None,
        k_rope=k_rope[idx] if k_rope is not None else None,
        pool_rope=pool_rope,
        prefix=f"{prefix}.midstep" if prefix else "",
    )


def midstep_compressor_write(
    kv, score, ape, kv_state, score_state, *, ratio: int, overlap: bool, prefix=""
) -> None:
    """Replay a Compressor's `update_compressor_states` into the scratch slots."""
    plan = current_midstep_plan()
    if plan is None:
        return
    write_plan = plan.compress_plans.get(ratio)
    if write_plan is None:
        return
    from atom.model_ops.v4_kernels.state_writes import update_compressor_states

    update_compressor_states(
        kv,
        score,
        ape,
        kv_state,
        score_state,
        write_plan=write_plan,
        state_slot_mapping=plan.dst_slots,
        ratio=ratio,
        overlap=overlap,
        prefix=f"{prefix}.midstep" if prefix else "",
    )
