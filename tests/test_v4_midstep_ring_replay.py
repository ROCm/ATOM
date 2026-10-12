# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A midstep checkpoint's slot must be byte-identical to a cut forward's.

For every checkpoint position P inside an uncut prefill chunk, compare two
ways of producing the state as of P:

- **cut**: a forward that ends exactly at P writes its rings into the request's
  own slot (which holds the incoming state);
- **midstep**: copy the incoming slot into a scratch slot, then replay the
  uncut forward's ring writes truncated at P (`v4_kernels/midstep.py`).

Every ring row of the two slots has to match bit for bit: the SWA window on the
bf16 path and on the fp8 2buff path, and the compressor kv/score rings. This is
the property that makes a midstep checkpoint indistinguishable from a cut one
to the request that later resumes from it.
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("replays Triton ring writes; needs a real GPU", allow_module_level=True)

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
    UnifiedPoolGeometry,
)
from atom.model_ops.v4_kernels import midstep
from atom.model_ops.v4_kernels.state_writes import (
    swa_write,
    update_compressor_states,
)

DEV = "cuda"
RING = 11
WINDOW = 8  # write_per_batch of the forward: min(window_size, max_seqlen_q)
D = 16
NUM_SLOTS = 12
RATIOS = [DENSE_RATIO, CSA_RATIO, HCA_RATIO]
# Three requests: chunk starts and lengths. Row 1 resumes far into the prompt,
# row 2 starts mid-ring so its checkpoints straddle the incoming state.
STARTS = [0, 5 * RING + 3, RING - 2]
LENS = [3 * RING + 1, 2 * RING, RING + 5]
SLOTS = [2, 7, 4]
# Per row: checkpoint offsets past the chunk start, including the chunk end
# and positions closer to the start than a window, which need incoming rows.
OFFSETS = [[1, WINDOW, 2 * RING, LENS[0]], [3, RING + 1, LENS[1]], [2, LENS[2] - 1]]


def _batch():
    torch.manual_seed(1)
    cu = np.zeros(len(LENS) + 1, dtype=np.int64)
    cu[1:] = np.cumsum(LENS)
    positions = torch.cat(
        [torch.arange(s, s + n, dtype=torch.int64) for s, n in zip(STARTS, LENS)]
    ).to(DEV)
    saves, scratch, nxt = [], [], max(SLOTS) + 1
    for i, offs in enumerate(OFFSETS):
        row = []
        for off in offs:
            row.append((nxt, STARTS[i] + off))
            scratch.append((i, nxt, STARTS[i] + off))
            nxt += 1
        saves.append(row)
    assert nxt <= NUM_SLOTS + 1 + 8
    return cu, positions, saves, scratch


def _install(monkeypatch, plan):
    monkeypatch.setattr(midstep, "current_midstep_plan", lambda: plan)


def _cut_cu(i, p, cu):
    """A one-request forward over [start, p) of row i, as flat token rows."""
    base = int(cu[i])
    return base, base + (p - STARTS[i])


@pytest.fixture
def geometry():
    return UnifiedPoolGeometry(
        RATIOS, num_blocks=2, num_slots=32, ring_slots=RING, block_size=256
    )


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("fp8", [False, True])
def test_swa_replay_matches_a_cut_forward(monkeypatch, geometry, ratio, fp8):
    cu, positions, saves, scratch = _batch()
    params = geometry.window_params(ratio)
    T = int(cu[-1])
    rows = geometry.plane_rows
    if fp8:
        nope_dt = torch.float8_e4m3fn
        # The real 2buff layout: 512 B fp8 NoPE (+ inline scale) and 64 bf16 RoPE.
        k_packed = torch.randn(T, 1, 512, device=DEV).to(nope_dt)
        k_rope = torch.randn(T, 1, 64, dtype=torch.bfloat16, device=DEV)
        pool0 = torch.randn(rows, 512, device=DEV).to(nope_dt)
        rope0 = torch.randn(rows, 64, dtype=torch.bfloat16, device=DEV)
        kv = None
    else:
        kv = torch.randn(T, D, dtype=torch.bfloat16, device=DEV)
        pool0 = torch.randn(rows, D, dtype=torch.bfloat16, device=DEV)
        k_packed = k_rope = rope0 = None

    def ring_rows(slot):
        return torch.tensor(
            [params.index(slot, q) for q in range(RING)], device=DEV, dtype=torch.long
        )

    # midstep: seed each scratch slot from its row's slot, then replay.
    pool, rope = pool0.clone(), (rope0.clone() if fp8 else None)
    for i, g, _p in scratch:
        pool[ring_rows(g)] = pool0[ring_rows(SLOTS[i])]
        if fp8:
            rope[ring_rows(g)] = rope0[ring_rows(SLOTS[i])]
    host = midstep.build_midstep_plan(
        saves, STARTS, cu, SLOTS, [(4, True)], swa_gather_rows=midstep.SWA_GATHER_ROWS
    )
    _install(monkeypatch, midstep.to_device(host, DEV))
    midstep.midstep_swa_write(
        kv, positions, pool, params, WINDOW, k_packed=k_packed, k_rope=k_rope,
        pool_rope=rope,
    )

    for i, g, p in scratch:
        # cut: the request's own slot, a forward ending exactly at p.
        cpool, crope = pool0.clone(), (rope0.clone() if fp8 else None)
        lo, hi = _cut_cu(i, p, cu)
        swa_write(
            kv[lo:hi] if kv is not None else None,
            positions[lo:hi],
            torch.tensor([0, hi - lo], dtype=torch.int32, device=DEV),
            torch.tensor([SLOTS[i]], dtype=torch.int32, device=DEV),
            cpool,
            params,
            WINDOW,
            k_packed=k_packed[lo:hi] if fp8 else None,
            k_rope=k_rope[lo:hi] if fp8 else None,
            pool_rope=crope,
        )
        want, got = cpool[ring_rows(SLOTS[i])], pool[ring_rows(g)]
        assert torch.equal(want.view(torch.uint8), got.view(torch.uint8)), (i, p)
        if fp8:
            assert torch.equal(crope[ring_rows(SLOTS[i])], rope[ring_rows(g)]), (i, p)


@pytest.mark.parametrize("ratio,overlap", [(4, True), (128, False), (4, False)])
def test_compressor_replay_matches_a_cut_forward(monkeypatch, ratio, overlap):
    cu, positions, saves, scratch = _batch()
    T = int(cu[-1])
    k_pool = ratio * (2 if overlap else 1)
    state_size = k_pool + 3
    kv = torch.randn(T, D, dtype=torch.float32, device=DEV)
    score = torch.randn(T, D, dtype=torch.float32, device=DEV)
    ape = torch.randn(ratio, D, dtype=torch.float32, device=DEV)
    kv0 = torch.randn(32, state_size, D, dtype=torch.float32, device=DEV)
    sc0 = torch.randn(32, state_size, D, dtype=torch.float32, device=DEV)

    kv_s, sc_s = kv0.clone(), sc0.clone()
    for i, g, _p in scratch:
        kv_s[g] = kv0[SLOTS[i]]
        sc_s[g] = sc0[SLOTS[i]]
    host = midstep.build_midstep_plan(saves, STARTS, cu, SLOTS, [(ratio, overlap)])
    _install(monkeypatch, midstep.to_device(host, DEV))
    midstep.midstep_compressor_write(
        kv, score, ape, kv_s, sc_s, ratio=ratio, overlap=overlap
    )

    for i, g, p in scratch:
        # cut: the documented prefill write window, positions >= ctx - K_pool.
        lo, hi = _cut_cu(i, p, cu)
        rows = [
            (r, 0, int(positions[r]), 0)
            for r in range(lo, hi)
            if int(positions[r]) >= max(0, p - k_pool)
        ]
        plan = torch.tensor(rows, dtype=torch.int32, device=DEV).reshape(-1, 4)
        ckv, csc = kv0.clone(), sc0.clone()
        update_compressor_states(
            kv, score, ape, ckv, csc,
            write_plan=plan,
            state_slot_mapping=torch.tensor([SLOTS[i]], dtype=torch.int32, device=DEV),
            ratio=ratio, overlap=overlap,
        )
        assert torch.equal(ckv[SLOTS[i]], kv_s[g]), (i, p)
        assert torch.equal(csc[SLOTS[i]], sc_s[g]), (i, p)


def test_builder_hands_kernels_plane_positions_and_the_copy_pool_slots():
    """The replay kernels address plane positions; the slot copy, pool slots.

    `_attach_midstep` crosses between the two exactly like
    `_populate_state_slot_mappings` does for `state_slot_out`. Passing a pool
    slot to a ring kernel writes another request's state -- the bug this pins.
    """
    from types import SimpleNamespace

    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    copies = []
    stub = SimpleNamespace(
        _unique_compress_ratios_overlap=[(4, True)],
        device=DEV,
        relocate_state_slots=lambda pairs: copies.extend(pairs),
        _physical_slots=lambda s: np.int32(99) - s,
    )
    batch = SimpleNamespace(
        state_save_all=[[(5, 300)], []],
        state_slots_committed=[2, 3],
        state_fork_srcs=[-1, -1],
        num_cached_tokens=[256, 0],
    )
    plan = DeepseekV4AttentionMetadataBuilder._attach_midstep(
        stub, batch, 2, np.array([0, 100, 140])
    )
    assert copies == [(2, 5)]  # pool slots: request's slot -> scratch
    assert plan.dst_slots.tolist() == [99 - 5]  # plane position for kernels
