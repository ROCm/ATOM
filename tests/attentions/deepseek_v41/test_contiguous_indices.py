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
        running_tokens=7,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", ["fp8", "fp4"])
def test_prefill_request_counts_reuse_indices_kernel(dtype, monkeypatch):
    import triton

    geo = V41PoolGeometry(
        2,
        ((0, 2),),
        32,
        8,
        512,
        128,
        kv_dtype=dtype,
        speculative_tokens=3,
        layer_ratios=(2,),
        index_topk=4,
    )
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
    tokens = 16
    selected = tensor([[0, 1, 2, 3]] * tokens)
    positions = torch.empty(tokens, dtype=torch.int32, device="cuda")
    batches = torch.empty_like(positions)
    cu = torch.empty(5, dtype=torch.int32, device="cuda")
    pptr, eptr = tensor([0] * (tokens + 1)), tensor([0] * (tokens + 1))
    step = SimpleNamespace(
        running_tokens=tokens,
        max_q_len=tokens,
        decode=False,
        positions=positions,
        batch_ids=batches,
        slots=tensor([3, 1, 2, 0]),
        block_tables=tensor([[1, 0], [2, 0], [3, 0], [4, 0]]),
        swa_cached_until=tensor([0] * 4),
        swa_replay_start=tensor([0] * 4),
    )
    cache_misses = []

    def record_miss(**event):
        if event["fn"].name == "_indices":
            cache_misses.append(event["key"])

    monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", record_miss)
    # Fixed token bucket and tensor alignment; only the live request count changes.
    for iteration, requests in enumerate((1, 2, 3, 4, 1)):
        lengths = [
            tokens // requests + (b < tokens % requests) for b in range(requests)
        ]
        offsets = [0]
        pos, batch_ids = [], []
        for batch, length in enumerate(lengths):
            offsets.append(offsets[-1] + length)
            pos.extend(range(10 + 32 * batch, 10 + 32 * batch + length))
            batch_ids.extend([batch] * length)
        positions.copy_(tensor(pos))
        batches.copy_(tensor(batch_ids))
        cu[: requests + 1].copy_(tensor(offsets))
        step.cu_seqlens_q = cu[: requests + 1]
        indptrs = fill_step_indptrs(step, geo, {2: (pptr, eptr)}, split=True)[2]
        pi, pp, wi, wp = build_indices(
            selected, indptrs, step, geo, geo.window(0, 6), 0, 2, split=True
        )
        pi, pp, wi, wp = pi[0].tolist(), pp.tolist(), wi.tolist(), wp.tolist()
        for t, (position, batch) in enumerate(zip(pos, batch_ids)):
            assert pi[pp[t] : pp[t + 1]] == [(batch + 1) * 16 + j for j in range(4)]
            start = 10 + 32 * batch
            expected = [
                (
                    batch * geo.ring_slots + p % geo.ring_slots
                    if p < start
                    else requests * geo.ring_slots + offsets[batch] + p - start
                )
                for p in range(position - geo.window_size + 1, position + 1)
            ]
            assert wi[wp[t] : wp[t + 1]] == expected
        if iteration == 0:
            cache_misses.clear()  # The first request count may compile once.
        else:
            assert not cache_misses, "request count introduced a new specialization"
