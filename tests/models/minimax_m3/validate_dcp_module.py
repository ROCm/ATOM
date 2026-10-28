# SPDX-License-Identifier: MIT
"""Four-GPU M3 sparse layer against dense attention over the same quantized KV."""

import os
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
from aiter.dist import parallel_state as ps

from atom.config import use_custom_atom_config
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.minimax_m3.sparse_attn import (
    make_sparse_decode_metadata,
    make_sparse_prefill_metadata,
)
from atom.models.minimax_m3.model import MiniMaxM3SparseAttention
from atom.utils import forward_context as fc


def unpack(payload, scales):
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="cuda",
    )
    codes = torch.stack((payload & 15, payload >> 4), -1).flatten(-2).long()
    return values[codes] * scales.view(torch.float8_e4m3fn).float().repeat_interleave(
        16, -1
    )


def cache(pages):
    pool = MhaKvPool(
        layers=1,
        block_size=128,
        num_kv_heads=4,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fn,
        kv_format="nvfp4",
    )
    pool.allocate(pages, "cuda")
    k, v = pool.kv_views(0)
    ks, vs = pool.scale_views(0)
    for x in (k, v, ks, vs):
        x.zero_()
    return pool, NS(k_cache=k, v_cache=v, k_scale=ks, v_scale=vs)


def check(rank):
    cfg = NS(
        dcp_config=NS(interleave_size=128, indexer_dcp_only=False),
        decode_context_parallel_size=4,
        kv_cache_block_size=128,
        enable_tbo=False,
        enable_tbo_decode=False,
        torch_dtype=torch.bfloat16,
        compilation_config=NS(static_forward_context={}),
    )
    hf = NS(
        hidden_size=256,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.5,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        sparse_attention_config={
            "sparse_block_size": 128,
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_topk_blocks": 16,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_attention_freq": [1],
        },
    )
    with use_custom_atom_config(cfg), torch.device("cuda"), torch.no_grad():
        torch.set_default_dtype(torch.bfloat16)
        layer = MiniMaxM3SparseAttention(
            hf, 0, prefix="test", cache_config="fp8", index_cache_config="fp4"
        )
        assert layer.attn.num_heads == 64 and layer.attn.num_kv_heads == 4
        torch.manual_seed(314)
        widths = [8192, 512, 512, 512, 128]
        w = torch.randn(sum(widths), 256, dtype=torch.bfloat16) * 0.02
        fields = w.split(widths)
        layer.qkv_proj.weight.copy_(
            torch.cat([x.chunk(4)[rank] for x in fields[:4]] + [fields[4]])
        )
        ow = torch.randn(256, 8192, dtype=torch.bfloat16) * 0.02
        layer.o_proj.weight.copy_(ow.chunk(4, 1)[rank])
        for norm in (
            layer.q_norm,
            layer.k_norm,
            layer.index_q_norm,
            layer.index_k_norm,
        ):
            norm.weight.zero_()
        layer.qkv_proj.process_weights_after_loading()
        layer.o_proj.process_weights_after_loading()
        _local_pool, local = cache(2)
        _full_pool, full = cache(8)
        impl = layer.attn.impl
        impl.kv_nvfp4 = True
        impl.kv_stage = torch.zeros(6, 2, 128, 4, 128, dtype=torch.bfloat16)
        impl.kv_stage_slots = torch.arange(768, dtype=torch.int64)
        impl.index_cache = torch.zeros(2, 128, 64, dtype=torch.uint8)
        impl.index_scale = torch.zeros(2, 128, 8, dtype=torch.uint8)
        impl.index_stage = torch.zeros(6, 128, 128, dtype=torch.bfloat16)
        impl.index_stage_slots = impl.kv_stage_slots
        n = 768
        positions = torch.arange(n, dtype=torch.int64)
        hidden = torch.randn(n, 256, dtype=torch.bfloat16)
        slots = torch.where(
            (positions // 128) % 4 == rank, positions // 512 * 128 + positions % 128, -1
        )
        cu = torch.tensor([0, n], dtype=torch.int32)
        lengths = torch.tensor([n], dtype=torch.int32)
        table = torch.tensor([[0, 1]], dtype=torch.int32)
        sm = make_sparse_prefill_metadata(
            cu_seqlens_q=cu,
            seq_lens=lengths,
            block_table=table,
            slot_mapping=slots,
            max_query_len=n,
            max_seq_len=n,
            num_prefills=1,
            num_prefill_tokens=n,
            num_idx_heads=4,
        )
        fc._forward_context = fc.ForwardContext(
            attn_metadata=NS(sparse_attention_metadata=sm),
            kv_cache_data={"layer_0": local},
            context=fc.Context(
                positions=positions,
                is_prefill=True,
                scheduled_bs=1,
                running_bs=1,
                scheduled_tokens=n,
                running_tokens=n,
            ),
        )
        actual = layer(positions, hidden)
        # The independent reference projects all heads with plain torch matmul,
        # then uses the existing fused writer without sharding. Dense softmax is
        # valid here because all six causal blocks fit the target's top-k=16.
        local_index = (impl.index_cache, impl.index_scale)
        full_index = (
            torch.zeros(8, 128, 64, dtype=torch.uint8),
            torch.zeros(8, 128, 8, dtype=torch.uint8),
        )

        def reference(inputs, pos):
            qkv = inputs @ w.T
            q, k, v, _, _ = qkv.split(widths, -1)
            impl.index_cache, impl.index_scale = full_index
            try:
                q = impl.rope_cache(
                    q,
                    k,
                    v,
                    qkv,
                    pos,
                    NS(
                        attn_metadata=NS(slot_mapping=pos),
                        kv_cache_data={"layer_0": full},
                    ),
                )[0]
            finally:
                impl.index_cache, impl.index_scale = local_index
            total = int(pos[-1]) + 1
            keys = (
                unpack(full.k_cache, full.k_scale)
                .permute(0, 2, 1, 3)
                .reshape(-1, 4, 128)[:total]
            )
            vals = (
                unpack(full.v_cache, full.v_scale)
                .permute(0, 2, 1, 3)
                .reshape(-1, 4, 128)[:total]
            )
            q = q.float().transpose(0, 1)
            keys = keys.repeat_interleave(16, 1).transpose(0, 1)
            vals = vals.repeat_interleave(16, 1).transpose(0, 1)
            causal = torch.arange(total)[None, :] <= pos[:, None]
            ref = torch.nn.functional.scaled_dot_product_attention(
                q, keys, vals, attn_mask=causal
            )
            ref = (
                ref.transpose(0, 1).reshape(len(pos), 8192).chunk(4, -1)[rank]
                @ ow.chunk(4, 1)[rank].float().T
            )
            return ref

        expected = reference(hidden, positions)
        torch.testing.assert_close(actual.float(), expected, atol=0.005, rtol=0.02)
        # Every local physical plane must equal its global-page counterpart.
        for name in ("k_cache", "v_cache", "k_scale", "v_scale"):
            got, expected = getattr(local, name), getattr(full, name)
            for page in range(rank, 6, 4):
                torch.testing.assert_close(
                    got[page // 4], expected[page], atol=0, rtol=0
                )
        for got, expected_plane in zip(local_index, full_index):
            for page in range(rank, 6, 4):
                torch.testing.assert_close(
                    got[page // 4], expected_plane[page], atol=0, rtol=0
                )

        # The next q=8 verify straddles into rank 2's second local page. Capture
        # the entire real layer, replay with changed hidden states, and compare
        # against causal dense attention including the retained prefix.
        pos = torch.arange(n, n + 8, dtype=torch.int64)
        x = torch.randn(8, 256, dtype=torch.bfloat16)
        slots = torch.where((pos // 128) % 4 == rank, pos // 512 * 128 + pos % 128, -1)
        sm = make_sparse_decode_metadata(
            seq_lens=torch.tensor([n + 8], dtype=torch.int32),
            block_table=table,
            slot_mapping=slots,
            max_seq_len=1024,
            num_idx_heads=4,
            max_query_len=8,
        )
        fc._forward_context = fc.ForwardContext(
            attn_metadata=NS(sparse_attention_metadata=sm),
            kv_cache_data={"layer_0": local},
            context=fc.Context(
                positions=pos,
                is_prefill=False,
                scheduled_bs=1,
                running_bs=1,
                scheduled_tokens=8,
                running_tokens=8,
            ),
        )
        actual = layer(pos, x)
        torch.testing.assert_close(
            actual.float(), reference(x, pos), atol=0.005, rtol=0.02
        )
        with ps.graph_capture() as capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                captured = layer(pos, x)
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(x, pos)
        torch.testing.assert_close(captured.float(), expected, atol=0.005, rtol=0.02)
        print(
            f"rank {rank}: real M3 layer prefill and q=8 graph replay match dense reference; max error {float((captured.float()-expected).abs().max()):.6f}",
            flush=True,
        )


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    ps.init_distributed_environment(world_size=4, rank=rank, local_rank=rank)
    ps.initialize_model_parallel(
        tensor_model_parallel_size=4, decode_context_model_parallel_size=4
    )
    check(rank)
    dist.barrier()
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
