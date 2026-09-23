# SPDX-License-Identifier: MIT
"""Multi-GPU lossless Ulysses checks and microbenchmarks.

    torchrun --master-addr=127.0.0.1 --nproc-per-node=4 tests/distributed/check_ulysses_exchange.py

Every rank checks the payload bit for bit, including the replicated sparse
index key, before timing. Report maximum rank latency, not rank zero alone.
"""

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from aiter.dist import parallel_state as ps

from atom.distributed import ulysses_sp as sp
from atom.distributed.sp_kernels import all_to_all_into, pack_fields


def timing(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    latency = torch.tensor(start.elapsed_time(end) * 1000 / repeats, device="cuda")
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return latency.item()


def check_causal_gqa(world):
    # Exercise the real wrapper with a nondivisible length. Only the attention
    # arithmetic is replaced by an independent float32 dense causal reference.
    # The full-head and sharded-head calculations see identical Q/K/V values.
    torch.manual_seed(789)
    tokens, heads, kv_heads, dim = 17, 32, 4, 128
    q = torch.randn(tokens, heads * dim, device="cuda")
    k = torch.randn(tokens, kv_heads * dim, device="cuda")
    v = torch.randn_like(k)

    def attention(query, key, value, position, **kwargs):
        assert query.shape[0] == key.shape[0] == position.numel() == tokens
        nq, nk = query.shape[1] // dim, key.shape[1] // dim
        qt = query.view(tokens, nq, dim).transpose(0, 1)
        kt = key.view(tokens, nk, dim).transpose(0, 1).repeat_interleave(nq // nk, 0)
        vt = value.view(tokens, nk, dim).transpose(0, 1).repeat_interleave(nq // nk, 0)
        mask = torch.ones(tokens, tokens, device="cuda", dtype=torch.bool).triu(1)
        scores = (qt @ kt.transpose(-1, -2)) * dim**-0.5
        out = scores.masked_fill(mask, -torch.inf).softmax(-1) @ vt
        return out.transpose(0, 1).reshape(tokens, nq * dim)

    positions = torch.arange(tokens, device="cuda")
    expected = sp.sp_split_tokens(attention(q, k, v, positions))
    layer = SimpleNamespace(head_dim=dim, impl=SimpleNamespace(forward=attention))
    actual = sp.ulysses_attention(
        layer,
        sp.sp_split_tokens(q),
        sp.sp_split_tokens(k),
        sp.sp_split_tokens(v),
        positions,
        None,
        None,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.init_distributed_environment(world_size=world, rank=rank)
    ps.initialize_model_parallel(prefill_context_model_parallel_size=world)
    sp.set_sp_world_size(world)
    group = ps.get_pcp_group()
    torch.manual_seed(1234 + rank)
    results = []
    kv_shards = min(world, 4)
    for label, spec in (
        ("qwen", ((0, 4096, 0), (4096, 512, kv_shards), (4608, 512, kv_shards))),
        (
            "m3",
            (
                (0, 8192, 0),
                (8192, 512, kv_shards),
                (8704, 512, kv_shards),
                (9216, 512, kv_shards),
                (9728, 128, 1),
            ),
        ),
    ):
        width = max(offset + size for offset, size, _ in spec)
        columns = torch.stack(
            [
                torch.cat(
                    [
                        torch.arange(size // (shards or world), device="cuda")
                        + offset
                        + (peer // (world // (shards or world)))
                        * (size // (shards or world))
                        for offset, size, shards in spec
                    ]
                )
                for peer in range(world)
            ]
        )
        for total in (world, 17 * world, 8192, 131072):
            tokens = total // world
            source = torch.randn(
                tokens, width + 16, device="cuda", dtype=torch.bfloat16
            )[:, :width]

            def reference_pack(source=source, columns=columns, tokens=tokens):
                return torch.gather(
                    source.unsqueeze(0).expand(world, -1, -1),
                    2,
                    columns[:, None, :].expand(world, tokens, -1),
                )

            expected_send = reference_pack()
            send = pack_fields(source, spec, world)
            torch.testing.assert_close(send, expected_send, rtol=0, atol=0)
            expected = torch.empty_like(send)
            dist.all_to_all_single(expected, expected_send, group=group.device_group)
            received = torch.empty_like(send)
            all_to_all_into(received, send, group)
            torch.testing.assert_close(received, expected, rtol=0, atol=0)
            owner = SimpleNamespace()
            packed = sp._exchange_columns(source, spec, owner)
            torch.testing.assert_close(packed, expected.flatten(0, 1), rtol=0, atol=0)
            # Restore Q heads, including a noncontiguous input column view.
            q_width = spec[0][1]
            restored = sp.ulysses_gather_heads(packed[:, : q_width // world])
            torch.testing.assert_close(restored, source[:, :q_width], rtol=0, atol=0)
            qkv = source[:, : q_width + 1024].split([q_width, 512, 512], dim=-1)
            separate = sp._exchange_tensors(qkv, [world, kv_shards, kv_shards])
            torch.testing.assert_close(
                separate,
                packed[:, : q_width // world + 1024 // kv_shards],
                rtol=0,
                atol=0,
            )

            def reference(received=received, reference_pack=reference_pack):
                dist.all_to_all_single(
                    received, reference_pack(), group=group.device_group
                )

            def candidate(received=received, source=source, spec=spec):
                all_to_all_into(received, pack_fields(source, spec, world), group)

            item = {
                "model": label,
                "tokens": total,
                "torch_pack_us": timing(reference_pack, args.repeats),
                "triton_pack_us": timing(
                    lambda source=source, spec=spec: pack_fields(source, spec, world),
                    args.repeats,
                ),
                "torch_exchange_us": timing(reference, args.repeats),
                "pynccl_exchange_us": timing(candidate, args.repeats),
            }
            # Capture after eager warmup has established every peer connection.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                candidate()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(received, expected, rtol=0, atol=0)
            item["graph_us"] = timing(graph.replay, args.repeats)
            results.append(item)
            if rank == 0:
                print(json.dumps(item), flush=True)
    # Nondivisible sequence lengths must preserve row order and drop padding.
    for total in (1, world - 1, world + 1, 131073):
        original = torch.arange(total * 3, device="cuda", dtype=torch.int32).view(
            total, 3
        )
        restored = sp.sp_gather_tokens(sp.sp_split_tokens(original), total)
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
    check_causal_gqa(world)
    if rank == 0 and args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(results, f, indent=2)
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


if __name__ == "__main__":
    main()
