# SPDX-License-Identifier: MIT
"""Device regression tests for CSA/SWA row coordinates (run only on a GPU)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="the V4.1 cache and CSR kernels need Triton")
pytest.importorskip("aiter", reason="V4.1 device attention needs AITER")

from atom.model_ops.attentions.deepseek_v41.indices import (
    build_indices,
    build_window_rows,
    fill_step_indptrs,
)
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry


@pytest.mark.parametrize("dtype", ["fp4", "fp8", "fp4_a8w4"])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("cached,replay", [(0, 0), (12, 0), (0, 10)])
def test_csr_resolves_to_causal_main_and_window_rows(
    dtype, decode, cached, replay, position_dtype
):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    a8w4 = dtype == "fp4_a8w4"
    geo = V41PoolGeometry(
        2,
        ((0, 2),),
        32,
        8,
        512,
        128,
        kv_dtype="fp4" if a8w4 else dtype,
        speculative_tokens=3,
        layer_ratios=(2,),
        index_topk=4,
    )
    device = "cuda"
    tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    positions = [10, 11, 12, 13, 31, 32, 0]
    batches = [0, 0, 0, 0, 1, 1, -1]
    tables = [[2, 0, 1], [3, 4, 5]]
    slots = [2, 0]
    step = SimpleNamespace(
        width=7,
        max_q_len=4,
        decode=decode,
        positions=tensor(positions).to(position_dtype),
        batch_ids=tensor(batches),
        cu_seqlens_q=tensor([0, 4, 6]),
        slots=tensor(slots),
        block_tables=tensor(tables),
        swa_cached_until=tensor([cached, 0]),
        swa_replay_start=tensor([replay, 0]),
    )
    indptrs = fill_step_indptrs(
        step, geo, {2: (tensor([0] * 8), tensor([0] * 8))}, split=True if a8w4 else None
    )[2]
    selected = tensor([[0, 1, 2, 3]] * 7)
    window = geo.window(0, 6)
    ring_rows = len(slots) * geo.ring_slots
    pi, pp, wi, wp = build_indices(
        selected, indptrs, step, geo, window, 0, 2, split=True if a8w4 else None
    )
    assert pi.dtype == (torch.int64 if dtype == "fp4" else torch.int32)
    if decode and (dtype == "fp8" or a8w4):
        wi = build_window_rows(step, geo, window, wp)
    pi, pp, wi, wp = pi[0].tolist(), pp.tolist(), wi.tolist(), wp.tolist()
    for t, (pos, batch) in enumerate(zip(positions, batches)):
        if batch < 0:
            assert pp[t + 1] == pp[t] and wp[t + 1] == wp[t]
            continue
        main = [tables[batch][0] * 16 + j for j in range(4)]
        if dtype == "fp8" or a8w4:
            assert pi[pp[t] : pp[t + 1]] == main
        start = 10 if batch == 0 else 31
        floor = max(0, pos - 7, (replay if batch == 0 else 0) if not decode else 0)
        cached_end = max(start, cached if batch == 0 else 0)
        history_end = (
            pos + 1 if decode else (cached_end if pos >= cached_end else start)
        )
        history, current = [], []
        for p in range(floor, pos + 1):
            if p < history_end:
                owner = slots[batch] if decode or dtype == "fp4" else batch
                history.append(owner * geo.ring_slots + p % geo.ring_slots)
            else:
                current.append((0 if batch == 0 else 4) + p - start)
        if dtype == "fp4":
            expected = [row * 2 for row in main] + [row * 2 + 1 for row in history]
            assert pi[pp[t] : pp[t + 1]] == expected
            if not decode:
                assert wi[wp[t] : wp[t + 1]] == current
        else:
            assert wi[wp[t] : wp[t + 1]] == history + [
                ring_rows + row for row in current
            ]
