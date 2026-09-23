# SPDX-License-Identifier: MIT
"""Multi-GPU Ulysses correctness checks, including 128K-token exchanges.

    torchrun --master-addr=127.0.0.1 --nproc-per-node=4 tests/distributed/check_ulysses_exchange.py

Every rank checks exact payload bits, KV/index-key replication, inverse
exchange, and graph replay with changed inputs.
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from aiter.dist import parallel_state as ps

from atom.distributed import ulysses_sp as sp
from atom.distributed.sp_kernels import all_to_all_into, pack_fields


def assert_bitwise_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


def check_causal_gqa(world):
    # Compare the real wrapper with an independent float32 causal reference.
    # Fewer KV heads than ranks exercise replication even in a two-rank run.
    torch.manual_seed(789)
    tokens, heads, dim = 17, 32, 128
    kv_heads = max(1, min(4, world // 2))
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
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world < 2 or 32 % world:
        raise ValueError("Run with 2, 4, 8, 16, or 32 ranks.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.init_distributed_environment(world_size=world, rank=rank)
    ps.initialize_model_parallel(prefill_context_model_parallel_size=world)
    sp.set_sp_world_size(world)
    group = ps.get_pcp_group()
    torch.manual_seed(1234 + rank)
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
            row_padding = 0 if total == world else 16
            source = torch.empty(
                tokens, width + row_padding, device="cuda", dtype=torch.bfloat16
            )[:, :width]
            # Sample arbitrary BF16 bits, including signed zero and NaN payloads.
            source.view(torch.int16).random_(-32768, 32768)

            def reference_pack(source=source, columns=columns, tokens=tokens):
                return torch.gather(
                    source.unsqueeze(0).expand(world, -1, -1),
                    2,
                    columns[:, None, :].expand(world, tokens, -1),
                )

            expected_send = reference_pack()
            send = pack_fields(source, spec, world)
            assert_bitwise_equal(send, expected_send)
            expected = torch.empty_like(send)
            dist.all_to_all_single(expected, expected_send, group=group.device_group)
            received = torch.empty_like(send)
            all_to_all_into(received, send, group)
            assert_bitwise_equal(received, expected)
            owner = SimpleNamespace()
            packed = sp._exchange_columns(source, spec, owner)
            assert_bitwise_equal(packed, expected.flatten(0, 1))
            # Cover both contiguous small gathers and strided column views.
            q_width = spec[0][1]
            q_heads = packed[:, : q_width // world]
            if total == world:
                q_heads = q_heads.contiguous()
            restored = sp.ulysses_gather_heads(q_heads)
            assert_bitwise_equal(restored, source[:, :q_width])
            qkv = source[:, : q_width + 1024].split([q_width, 512, 512], dim=-1)
            separate = sp._exchange_tensors(qkv, [world, kv_shards, kv_shards])
            assert_bitwise_equal(
                separate,
                packed[:, : q_width // world + 1024 // kv_shards],
            )

            def candidate(received=received, source=source, spec=spec):
                all_to_all_into(received, pack_fields(source, spec, world), group)

            # Eager exchanges above establish peer connections before capture.
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                candidate()
            for _ in range(3):
                source.view(torch.int16).random_(-32768, 32768)
                dist.all_to_all_single(
                    expected, reference_pack(), group=group.device_group
                )
                received.zero_()
                graph.replay()
                assert_bitwise_equal(received, expected)
            if rank == 0:
                print(f"PASS {label}: {total} tokens, SP{world}", flush=True)
    # Nondivisible sequence lengths must preserve row order and drop padding.
    for total in (1, world - 1, world + 1, 131073):
        original = torch.arange(total * 3, device="cuda", dtype=torch.int32).view(
            total, 3
        )
        restored = sp.sp_gather_tokens(sp.sp_split_tokens(original), total)
        assert_bitwise_equal(restored, original)
    check_causal_gqa(world)
    if rank == 0:
        print("PASS padding and causal GQA", flush=True)
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


if __name__ == "__main__":
    main()
