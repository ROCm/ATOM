# SPDX-License-Identifier: MIT
"""GPU metadata and selective resets against the host walk-plan reference."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("rows", [1, 8, 13, 40, 64])
@pytest.mark.parametrize("pattern", ["single", "six", "mixed"])
def test_parallel_plan_and_reset_replay(rows, pattern):
    from atom.models.deepseek_v41.mono import index_plan as ip
    from atom.models.deepseek_v41.mono.step_meta import write_step_meta

    rng = np.random.default_rng(4140 + rows)
    owners = np.arange(rows, dtype=np.int32)
    if pattern == "six":
        owners //= 6
    elif pattern == "mixed":
        owners = np.cumsum(rng.integers(0, 2, rows)).astype(np.int32)
    if rows > 1:
        owners[-1] = -1
    positions = rng.choice([0, 510, 511, 512, 8191, 163839], rows).astype(np.int64)
    positions[owners < 0] = 0
    slots = np.arange(rows, dtype=np.int32)[::-1].copy()
    step = SimpleNamespace(
        batch_ids=torch.tensor(owners, device="cuda"),
        slots=torch.tensor(slots, device="cuda"),
        positions=torch.tensor(positions, device="cuda"),
        block_tables=torch.empty(rows, 1282, device="cuda", dtype=torch.int32),
    )
    geometry = SimpleNamespace(window_size=512)
    window = SimpleNamespace(slot_rows=8192, ring_slots=4096)
    ring = torch.empty(rows, device="cuda", dtype=torch.int32)
    meta = torch.empty(rows, 4, device="cuda", dtype=torch.int32)
    plans = {r: torch.empty(256, 4, device="cuda", dtype=torch.int32) for r in (1, 2)}
    backing = torch.full((131,), 71, device="cuda", dtype=torch.int32)
    resets = (backing[3:74], backing[83:126])

    def launch(lo, hi):
        write_step_meta(
            step,
            geometry,
            window,
            ring,
            meta,
            plans,
            (lo, hi - lo),
            resets,
            parallel_plans=True,
        )

    def check(lo, hi):
        pos = step.positions.cpu().numpy()
        live = owners >= 0
        batch = np.maximum(owners, 0)
        ref_ring = np.where(live, slots[batch] * 8192 + pos % 4096, -1)
        first = np.maximum(pos - 511, 0)
        ref_meta = np.stack(
            (
                first,
                np.where(live, pos + 1 - first, 0),
                batch * 1282,
                slots[batch] * 8192,
            ),
            axis=1,
        )
        torch.testing.assert_close(
            ring.cpu(), torch.from_numpy(ref_ring).int(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            meta.cpu(), torch.from_numpy(ref_meta).int(), rtol=0, atol=0
        )
        for ratio, plan in plans.items():
            ref = np.empty((256, 4), dtype=np.int32)
            ip.fill_walk_plan(
                (pos[lo:hi] + 1) // ratio, ip.walk_pieces(owners[lo:hi]), ref
            )
            ref[ref[:, 1] > 0, 0] += lo
            torch.testing.assert_close(
                plan.cpu(), torch.from_numpy(ref), rtol=0, atol=0
            )
        assert all(torch.count_nonzero(t).item() == 0 for t in resets)
        untouched = torch.cat((backing[:3], backing[74:83], backing[126:]))
        assert torch.equal(untouched, torch.full_like(untouched, 71))

    for lo, hi in ((0, rows), (rows // 3, rows)):
        backing.fill_(71)
        launch(lo, hi)
        check(lo, hi)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(0, rows)
    step.positions.add_(17)
    backing.fill_(71)
    graph.replay()
    check(0, rows)
