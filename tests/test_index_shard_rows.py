# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a sharded MiniMax-M3 index cache's rows are written and read.

Rank r keeps block b's index rows iff ``b % P == r``, at local page ``b // P``
(`minimax_m3/index_shard.py`). Checked on the CPU against a replicated cache:

- writes: `owned_index_slots` sends each token's row to its owner's local slot
  and every other rank skips it;
- decode reads: `page_view` addresses a local plane by block id exactly where
  the CP scorers compute ``base + page * stride(0)``, for every owned block;
- prefill reads: the gathered pages, read through the gather's table, are the
  replicated cache's rows at the same logical blocks, so a reference prefill
  scorer picks the same top-k from them as from the replicated cache.
"""

import random

import pytest
import torch

from atom.model_engine.block_pool import BlockPool
from atom.model_ops.minimax_m3.index_shard import (
    INDEX_BLOCK_TOKENS,
    gathered_planes,
    owned_index_slots,
    pack_index_pages,
    page_view,
    plan_index_gather,
)

P = 4
D = 128
PAGE = INDEX_BLOCK_TOKENS


def shards(plane: torch.Tensor, world: int = P) -> list[torch.Tensor]:
    """Each rank's local plane of a replicated ``plane``: block b at b // P."""
    return [plane[rank::world].contiguous() for rank in range(world)]


def padded(plane: torch.Tensor, front: int) -> torch.Tensor:
    """``plane`` copied into a storage with ``front`` pages before it, as the
    pool's index planes sit after its K/V."""
    store = torch.zeros((front + plane.shape[0], *plane.shape[1:]), dtype=plane.dtype)
    store[front:] = plane
    return store[front:]


class TestWrites:
    def test_each_row_lands_once_at_its_owners_local_slot(self):
        rng = random.Random(1)
        slots = torch.tensor(
            [rng.randrange(0, 64 * PAGE) for _ in range(300)] + [-1] * 5
        )
        written = {}
        for rank in range(P):
            local = owned_index_slots(slots, rank, P)
            for token, (slot, mine) in enumerate(zip(slots.tolist(), local.tolist())):
                if mine < 0:
                    continue
                page, row = divmod(slot, PAGE)
                assert page % P == rank
                assert divmod(mine, PAGE) == (page // P, row)
                written[token] = written.get(token, 0) + 1
        assert written == {t: 1 for t in range(300)}

    def test_the_decode_buffer_holds_the_result(self):
        slots = torch.tensor([5 * PAGE + 3, -1, 6 * PAGE])
        out = torch.full((8,), 99, dtype=torch.int64)
        result = owned_index_slots(slots, 1, P, out=out)
        assert result.data_ptr() == out.data_ptr()
        assert out[:3].tolist() == [1 * PAGE + 3, -1, -1]


class TestDecodeView:
    @pytest.mark.parametrize("row_bytes", [64, 8])
    def test_every_owned_block_reads_its_local_page(self, row_bytes):
        pages = 32
        full = torch.randint(0, 256, (pages, PAGE, row_bytes), dtype=torch.uint8)
        for rank, local in enumerate(shards(full)):
            plane = padded(local, front=2)
            view = page_view(plane, rank, P)
            step = plane.stride(0) // P
            for block in range(rank, pages, P):
                # The scorers' address for page `block`.
                addr = view.data_ptr() + block * step
                assert addr == plane[block // P].data_ptr()
                assert torch.equal(view[block], full[block])
            last = view[view.shape[0] - 1]
            assert (
                last.data_ptr() + last.numel()
                == plane[-1].data_ptr() + plane[-1].numel()
            ), "the view runs past the plane"

    def test_a_replicated_plane_is_its_own_view(self):
        plane = torch.zeros(4, PAGE, 64, dtype=torch.uint8)
        assert page_view(plane, 0, 1) is plane

    def test_a_plane_at_the_start_of_its_storage_is_refused_off_rank_0(self):
        plane = torch.zeros(4, PAGE, 64, dtype=torch.uint8)
        assert page_view(plane, 0, P).data_ptr() == plane.data_ptr()
        with pytest.raises(ValueError, match="too close"):
            page_view(plane, 1, P)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant(payload, scale):
    lut = torch.tensor(E2M1 + tuple(-x for x in E2M1))
    p = payload.to(torch.int64)
    vals = torch.stack([lut[p & 15], lut[p >> 4]], dim=-1).flatten(-2)
    s = scale.view(torch.float8_e4m3fn).to(torch.float32)
    return (vals.unflatten(-1, (-1, 16)) * s[..., None]).flatten(-2)


def reference_topk(q, payload, scale, table, seq_lens, q_lens, topk):
    """Per query token, the ``topk`` causal block maxima of q . k over the FP4
    index rows found through ``table``: what the TP prefill scorer and selector
    compute (largest score first, the lower block on a tie)."""
    out, row = [], 0
    for b, (seq_len, q_len) in enumerate(zip(seq_lens, q_lens)):
        blocks = -(-seq_len // PAGE)
        keys = torch.cat(
            [dequant(payload[table[b][j]], scale[table[b][j]]) for j in range(blocks)]
        )
        for t in range(q_len):
            pos = seq_len - q_len + t
            dots = keys[: pos + 1] @ q[row].to(torch.float64).float()
            best = []
            for j in range((pos // PAGE) + 1):
                best.append((float(dots[j * PAGE : (j + 1) * PAGE].max()), -j))
            out.append([-j for _, j in sorted(best, reverse=True)[:topk]])
            row += 1
    return out


class TestPrefillGather:
    def _deployment(self, seed=3):
        rng = random.Random(seed)
        pages = 48
        payload = torch.randint(0, 256, (pages, PAGE, D // 2), dtype=torch.uint8)
        # E4M3 scale bytes of moderate magnitude (no NaN codes).
        scale = torch.randint(0x28, 0x48, (pages, PAGE, D // 16), dtype=torch.uint8)
        pool = BlockPool(pages, residues=P)
        for _ in range(30):  # churn, so the tables are not in id order
            held = [pool.pop(rng.randrange(P)) for _ in range(rng.randrange(1, 12))]
            for b in held:
                pool.allocate(b)
            for b in rng.sample(held, len(held)):
                pool.free(b)
        seq_lens = [700, 130, 1000, 256]
        q_lens = [100, 2, 1000, 1]
        tables = []
        for seq_len in seq_lens:
            row = []
            for j in range(-(-seq_len // PAGE)):
                block = pool.pop(j % P)
                pool.allocate(block)
                row.append(block)
            tables.append(row)
        return payload, scale, tables, seq_lens, q_lens

    def _gather(self, payload, scale, tables, seq_lens):
        payloads, scales = shards(payload), shards(scale)
        plans = [
            plan_index_gather(tables, seq_lens, rank, P, "cpu") for rank in range(P)
        ]
        for plan in plans[1:]:
            assert torch.equal(plan.score_table, plans[0].score_table)
        sends = [
            pack_index_pages(plan, payloads[rank], scales[rank])
            for rank, plan in enumerate(plans)
        ]
        # all_gather_into_tensor's layout: rank 0's buffer first.
        gathered = torch.cat(sends)
        return (*gathered_planes(gathered, payloads[0], scales[0]), plans[0])

    def test_the_gathered_pages_are_the_replicated_rows(self):
        payload, scale, tables, seq_lens, _ = self._deployment()
        g_payload, g_scale, plan = self._gather(payload, scale, tables, seq_lens)
        assert g_payload.stride(0) == g_scale.stride(0) == 64 * PAGE + 8 * PAGE
        for b, row in enumerate(tables):
            for j, block in enumerate(row):
                page = int(plan.score_table[b, j])
                assert torch.equal(g_payload[page], payload[block])
                assert torch.equal(g_scale[page], scale[block])

    def test_the_scorer_picks_the_same_blocks_from_them(self):
        payload, scale, tables, seq_lens, q_lens = self._deployment()
        g_payload, g_scale, plan = self._gather(payload, scale, tables, seq_lens)
        torch.manual_seed(0)
        topk = 4
        for _head in range(P):  # each rank scores its own head
            q = torch.randn(sum(q_lens), D)
            want = reference_topk(q, payload, scale, tables, seq_lens, q_lens, topk)
            got = reference_topk(
                q, g_payload, g_scale, plan.score_table.tolist(), seq_lens, q_lens, topk
            )
            assert got == want

    def test_a_block_off_its_lane_is_refused(self):
        plan_index_gather([[0, 5, 6, 3]], [512], 0, P, "cpu")
        # Logical block 1 holding block 6 (lane 2): its index is rank 2's.
        with pytest.raises(RuntimeError, match="logical block 1"):
            plan_index_gather([[0, 6, 2]], [384], 0, P, "cpu")

    def test_a_padding_request_sends_in_bounds_pages(self):
        plan = plan_index_gather([], [300], 2, P, "cpu")
        assert plan.send_pages.tolist() == [0]
        assert plan.score_table.tolist() == [[0, 1, 2]]
