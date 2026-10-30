"""New GPU ABI/write/replay checks, before reading attention checkpoint tensors."""

import gc
import weakref
from unittest.mock import patch

import torch
from aiter.ops import mla_sparse_prefill as opus
from atom.model_ops.blockscale import quantize_fp8
from atom.models.deepseek_v41.owner_service.owner_head_attention import (
    attend_fp8_head_shards as native,
)
from candidate import attend_fp8_head_shards as candidate


def byte_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def describe(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "ptr": tensor.data_ptr(),
        "dtype": str(tensor.dtype),
    }


def case(rows):
    query = quantize_fp8(torch.randn(rows, 64, 512, device="cuda").bfloat16() * 0.2)
    values, scales = quantize_fp8(torch.randn(256, 512, device="cuda").bfloat16() * 0.2)
    records = torch.cat((values.view(torch.uint8), scales.view(torch.uint8)), -1)
    lengths = [73 + i % 56 for i in range(rows)]
    window_lengths = [1 + i % 6 for i in range(rows)]
    if rows == 221:
        lengths[-1] = window_lengths[-1] = 0

    def csr(counts):
        offsets = [0]
        ids = []
        for count in counts:
            ids.extend(range(count))
            offsets.append(len(ids))
        return (
            torch.tensor(ids, device="cuda", dtype=torch.int32),
            torch.tensor(offsets, device="cuda", dtype=torch.int32),
        )

    prefix_ids, prefix_ptr = csr(lengths)
    window_ids, window_ptr = csr(window_lengths)
    sink = torch.randn(64, device="cuda")
    args = (
        query,
        records[:128],
        prefix_ids,
        prefix_ptr,
        records[128:],
        window_ids,
        window_ptr,
        sink,
        512**-0.5,
        16,
    )
    key = (query[0].dtype, values.dtype)
    forward = opus._OPUS_V41_KERNELS[key]
    calls = []

    def recorded(*bound):
        q, qs, out = bound[0], bound[1], bound[11]
        calls.append(
            {
                "q": describe(q),
                "scale": describe(qs),
                "out": describe(out),
                "prefix_nnz": bound[4].numel(),
                "window_nnz": bound[8].numel(),
                "splits": 1 if len(bound) == 13 else int(bound[13]),
                "kernel": forward.__name__,
            }
        )
        out.fill_(float("nan"))
        return forward(*bound)

    with patch.dict(opus._OPUS_V41_KERNELS, {key: recorded}):
        reference = native(*args)
        actual = candidate(*args)
    assert len(calls) == 8
    for shard, (a, b) in enumerate(zip(calls[:4], calls[4:], strict=True)):
        assert all(
            a[k] == b[k] for k in ("kernel", "splits", "prefix_nnz", "window_nnz")
        )
        assert b["q"]["ptr"] == query[0].data_ptr() + shard * 16 * 512
        assert b["scale"]["ptr"] == query[1].data_ptr() + shard * 16 * 16
        assert b["out"]["ptr"] == actual.data_ptr() + shard * 16 * 512 * 2
        assert b["q"]["stride"] == list(query[0].stride())
        assert b["scale"]["stride"] == list(query[1].stride())
        assert b["out"]["stride"] == list(actual.stride())
    assert byte_equal(actual, reference)
    assert torch.isfinite(actual).all().item() and torch.count_nonzero(actual).item()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = candidate(*args)
    reference_ref = weakref.ref(captured)
    unchanged = records.clone()
    changes = []
    try:
        previous = None
        for iteration in range(3):
            update = quantize_fp8(
                torch.randn(rows, 64, 512, device="cuda").bfloat16()
                * (0.15 + iteration * 0.05)
            )
            for dst, src in zip(query, update, strict=True):
                dst.copy_(src)
            expected = native(*args)
            captured.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert byte_equal(captured, expected)
            assert torch.isfinite(captured).all().item()
            assert torch.count_nonzero(captured).item()
            assert all(byte_equal(a, b) for a, b in zip(query, update, strict=True))
            assert byte_equal(records, unchanged)
            if previous is not None:
                assert not byte_equal(captured, previous)
            previous = captured.clone()
            changes.append(
                {"iteration": iteration, "native_exact": True, "fully_written": True}
            )
    finally:
        graph.reset()
        del captured
        gc.collect()
    assert reference_ref() is None
    return {"rows": rows, "calls": calls, "updates": changes, "graph_released": True}


def run():
    return [case(rows) for rows in (384, 221)]
