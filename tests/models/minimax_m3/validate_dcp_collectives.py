# SPDX-License-Identifier: MIT
"""Run with torchrun --standalone --nproc-per-node=4 on four idle GPUs."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from aiter.dist import parallel_state as ps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from test_m3_dcp_attention import cache_fixture

from atom.model_ops.minimax_m3.dcp import M3DCPProjection
from atom.model_ops.minimax_m3.dcp_attention import (
    attention,
    index_topk,
    local_attention,
    merge_partials,
)


def check_format(fmt, page, rank, group):
    _, _, table, caches = cache_fixture(fmt, page)
    torch.manual_seed(314)
    q = torch.randn(8, 16, 128, device="cuda", dtype=torch.bfloat16)
    req = torch.zeros(8, device="cuda", dtype=torch.int32)
    lens = torch.tensor(
        [1, 127, 128, 129, 511, 513, 767, 768], device="cuda", dtype=torch.int32
    )
    start = torch.zeros_like(lens)
    args = {
        "world": 4,
        "rank": rank,
        "group": group,
        "scale": 128**-0.5,
        "nvfp4": fmt == "nvfp4",
    }
    expected_parts = [
        local_attention(
            q,
            caches[r],
            table,
            req,
            lens,
            start,
            world=4,
            rank=r,
            scale=128**-0.5,
            nvfp4=fmt == "nvfp4",
        )
        for r in range(4)
    ]
    expected, _ = merge_partials(
        torch.stack([v[0] for v in expected_parts]),
        torch.stack([v[1] for v in expected_parts]),
    )

    def run():
        return attention(q, caches[rank], table, req, lens, start, **args)

    actual = run()
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
    with ps.graph_capture() as context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=context.stream):
            captured = run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, actual)
    # Short prefixes have three entirely empty ranks. Replaying the same
    # communication graph must overwrite every previous partial.
    lens.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, run())
    key = caches[rank][0][:, 0]
    ks = caches[rank][2][:, 0].contiguous() if fmt == "nvfp4" else None
    if fmt == "fp8":
        # Reconstruct the owned index vector from page-16 SHUFFLE.
        key = key.permute(0, 2, 1, 3).reshape(2, 128, 128)
    selected = index_topk(
        q[:, :4],
        key.contiguous(),
        ks,
        table,
        req,
        lens,
        group,
        rank,
        4,
        3,
        1,
        1,
        128**-0.5,
    )
    assert torch.all(selected[:, :, 0] == 0)
    assert torch.all(selected[:, :, 1:] == -1)

    def indexed_run():
        topk = index_topk(
            q[:, :4],
            key.contiguous(),
            ks,
            table,
            req,
            lens,
            group,
            rank,
            4,
            3,
            1,
            1,
            128**-0.5,
        )
        return attention(q, caches[rank], table, req, lens, start, topk=topk, **args)

    indexed_run()
    with ps.graph_capture() as context:
        indexed_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(indexed_graph, stream=context.stream):
            indexed_out = indexed_run()
    lens.fill_(768)
    indexed_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(indexed_out, indexed_run())
    if rank == 0:
        print(
            f"{fmt}: DCP4 projection, LSE merge, global top-k and graph replay passed",
            flush=True,
        )


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    ps.init_distributed_environment(world_size=4, rank=rank, local_rank=rank)
    ps.initialize_model_parallel(
        tensor_model_parallel_size=4, decode_context_model_parallel_size=4
    )
    group = ps.get_dcp_group()
    projection = object.__new__(M3DCPProjection)
    projection.world, projection.rank, projection.group = 4, rank, group
    packed = torch.full((8, 384), -999.0, device="cuda", dtype=torch.bfloat16)
    local = packed[:, 128:256]
    local.fill_(rank + 1.0)
    gathered = projection.gather(local, 128)
    for h in range(4):
        torch.testing.assert_close(
            gathered[:, h * 128 : (h + 1) * 128], torch.full_like(local, h + 1.0)
        )
    torch.testing.assert_close(projection.local_output(gathered), local)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    # Fullgraph rejects a Python communicator graph break. Dynamic row counts
    # exercise the fake output shape without recompiling for every batch size.
    compiled = torch.compile(
        lambda x: projection.gather(x[:, 128:256], 128),
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )
    for rows in (8, 12):
        x = torch.full((rows, 384), rank + 2.0, device="cuda", dtype=torch.bfloat16)
        result = compiled(x)
        expected = torch.cat(
            [
                torch.full((rows, 128), h + 2.0, device="cuda", dtype=torch.bfloat16)
                for h in range(4)
            ],
            dim=1,
        )
        torch.testing.assert_close(result, expected)
    assert len(graphs) == 1, (
        f"DCP projection split or recompiled into {len(graphs)} graphs"
    )
    with ps.graph_capture() as context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=context.stream):
            captured = compiled(x)
    x.add_(1)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected + 1)
    if rank == 0:
        print(
            "DCP projection fullgraph dynamic tracing and graph replay passed",
            flush=True,
        )

    for fmt, page in (("fp8", 16), ("nvfp4", 128)):
        check_format(fmt, page, rank, group)
    dist.barrier()
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
