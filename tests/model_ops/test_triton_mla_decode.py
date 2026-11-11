# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from aiter.ops.triton.utils.types import e4m3_dtype

from atom.model_ops.attention_mla import MLAAttention
from atom.utils import selector
from atom.utils.forward_context import AttentionMetaData


@pytest.mark.parametrize("triton", [False, True])
@pytest.mark.parametrize("family", [selector.Family.MLA, selector.Family.KIMI_MLA])
def test_mla_backend_import(monkeypatch, family, triton):
    # Resolve the real modules: importing the Kimi backend used to reach the
    # removed aiter.mla_decode module even when Triton MLA was disabled.
    monkeypatch.setenv("ATOM_USE_TRITON_MLA", str(int(triton)))
    backend = selector._cached_get_attn_backend.__wrapped__(family, False, False)
    assert backend.get_impl_cls() is MLAAttention
    assert backend.get_builder_cls() is not None


def _inputs(block_size, q_len, quant, shuffled=False, heads=16):
    torch.manual_seed(42)
    rank, rope = 512, 64
    lengths = [19, 67, 129]
    blocks_per_seq = (max(lengths) + block_size - 1) // block_size
    num_blocks = len(lengths) * blocks_per_seq + 3
    q = torch.randn(len(lengths) * q_len, heads, rank + rope, device="cuda")
    kv = torch.randn(num_blocks, block_size, rank + rope, device="cuda")
    q_scale = 0.25 if quant == "fp8_qkv" else 1.0
    kv_scale = 0.5 if quant != "bf16" else 1.0
    q = (q / q_scale).to(e4m3_dtype if quant == "fp8_qkv" else torch.bfloat16)
    kv = (kv / kv_scale).to(e4m3_dtype if quant != "bf16" else torch.bfloat16)
    # Nonsequential physical IDs and a strided table catch token/page confusion.
    table = torch.randperm(num_blocks, device="cuda")[: len(lengths) * blocks_per_seq]
    table = table.to(torch.int32).view(len(lengths), blocks_per_seq)
    table = torch.nn.functional.pad(table, (0, 2))[:, :blocks_per_seq]
    metadata = AttentionMetaData(
        cu_seqlens_q=torch.arange(
            0, (len(lengths) + 1) * q_len, q_len, device="cuda", dtype=torch.int32
        ),
        context_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        max_seqlen_q=q_len,
        max_seqlen_k=max(lengths),
        block_tables=table,
    )
    # Construct only the decode portion; projections are outside this regression.
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads = heads
    layer.head_pad = 0
    layer.dcp_world_size = 1
    layer.use_seg_mla = False
    layer.use_triton_mla = True
    layer.kv_shuffled = shuffled
    layer.kv_cache_block_size = block_size
    layer._shuffle_block_size_cached = block_size
    layer.kv_lora_rank = rank
    layer.qk_rope_head_dim = rope
    layer.scale = (rank + rope) ** -0.5
    layer.dtype = torch.bfloat16
    layer._q_scale = torch.tensor(q_scale, device="cuda")
    layer._k_scale = torch.tensor(kv_scale, device="cuda")
    layer._v_up_proj_and_o_proj = lambda output: output
    cache = kv
    if shuffled:
        # The same 16-lane tile layout used by concat_and_cache_mla.
        width = 16 if kv.element_size() == 1 else 8

        def shuffle(part):
            dim = part.shape[-1]
            return (
                part.reshape(
                    num_blocks, block_size // 16, 16, dim // (2 * width), 2, width
                )
                .permute(0, 1, 3, 4, 2, 5)
                .reshape(num_blocks, -1)
            )

        cache = torch.cat([shuffle(kv[..., :rank]), shuffle(kv[..., rank:])], dim=-1)
    return layer, q, cache.view(-1, 1, rank + rope), metadata, kv


def _reference(layer, q, kv, metadata):
    queries = q.float() * layer._q_scale
    keys = kv.float() * layer._k_scale
    output = []
    for seq, length in enumerate(metadata.context_lens.tolist()):
        seq_kv = keys[metadata.block_tables[seq].long()].flatten(0, 1)
        start, end = metadata.cu_seqlens_q[seq : seq + 2].tolist()
        for token in range(start, end):
            visible = seq_kv[: length - (end - token - 1)]
            scores = queries[token] @ visible.T * layer.scale
            output.append(scores.softmax(-1) @ visible[:, : layer.kv_lora_rank])
    return torch.stack(output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("block_size,shuffled", [(16, False), (64, False), (64, True)])
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("quant", ["bf16", "fp8_kv", "fp8_qkv"])
def test_triton_mla_decode_matches_reference(block_size, shuffled, q_len, quant):
    layer, q, cache, metadata, kv = _inputs(block_size, q_len, quant, shuffled)
    actual = layer._forward_decode(q, cache, metadata)
    expected = _reference(layer, q, kv, metadata)
    # With FP8 queries the kernel also rounds softmax probabilities to FP8
    # before PV, whereas the reference keeps the entire attention in FP32.
    atol = 0.04 if quant == "fp8_qkv" else 0.01
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=0.02)


def _fp4_decoded(kv, fmt):
    """The values a packed FP4 cache row decodes to, from aiter's quantizers."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant, dynamic_nvfp4_quant
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    rows = kv.reshape(-1, kv.shape[-1]).contiguous()
    if fmt == "mxfp4_fp8rope":
        rank = 512
        latent = _fp4_decoded(rows[:, :rank].contiguous(), "mxfp4")
        rope = rows[:, rank:].to(e4m3_dtype).float()
        return torch.cat([latent, rope], dim=-1).view(kv.shape)
    if fmt == "mxfp4_b16":
        # aiter's dynamic_mxfp4_quant is fixed at 32 values per scale: quantize
        # each 16-value block twice over, so its scale is the block's own.
        blocks = rows.reshape(-1, 16)
        packed, scales = dynamic_mxfp4_quant(torch.cat([blocks, blocks], -1))
        values = mxfp4_to_f32(packed.view(torch.uint8))[:, :16]
        scale = e8m0_to_f32(scales.view(torch.uint8)).reshape(-1, 1)
        return (values * scale).reshape(kv.shape)
    if fmt == "mxfp4":
        packed, scales = dynamic_mxfp4_quant(rows)
        scales = e8m0_to_f32(scales.view(torch.uint8)).repeat_interleave(32, dim=-1)
    else:
        packed, scales = dynamic_nvfp4_quant(rows)
        scales = scales.view(e4m3_dtype).float().repeat_interleave(16, dim=-1)
    return (mxfp4_to_f32(packed.view(torch.uint8)) * scales).view(kv.shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("fmt", ["mxfp4", "mxfp4_b16", "mxfp4_fp8rope", "nvfp4"])
@pytest.mark.parametrize(
    "q_len",
    [
        1,
        pytest.param(
            4,
            marks=pytest.mark.xfail(
                reason="the Gluon shuffled decode (BF16/FP8 too) mishandles a "
                "query whose context ends on a tile edge, as the 129-token "
                "sequence's does with 4 queries"
            ),
        ),
    ],
)
def test_triton_mla_decode_fp4_kv_matches_reference(fmt, q_len):
    """An FP4 cache filled by the shuffled writer ATOM uses for prefill and read
    back through _forward_decode attends exactly over its decoded values."""
    from aiter.ops.triton.kv_cache import cat_and_cache_mla

    from atom.model_ops.mla_fp4_kv import FP4_KV_BLOCK_SIZE, mla_kv_cache_row_width

    if fmt == "mxfp4_fp8rope" and q_len > 1:
        pytest.skip("FP8 rope is decoded one query token at a time")
    block_size = FP4_KV_BLOCK_SIZE
    # FP8 rope is read by the LDS pipeline, which takes 128 heads at once.
    heads = 128 if fmt == "mxfp4_fp8rope" else 16
    layer, q, _, metadata, _ = _inputs(
        block_size, q_len, "fp8_qkv", shuffled=True, heads=heads
    )
    layer.fp4_kv_format = fmt
    layer._k_scale = torch.tensor(1.0, device="cuda")
    rank, rope = layer.kv_lora_rank, layer.qk_rope_head_dim
    num_blocks = metadata.block_tables.numel() + 3
    torch.manual_seed(7)
    kv = torch.randn(num_blocks * block_size, 1, rank + rope, device="cuda").bfloat16()
    row = mla_kv_cache_row_width(rank + rope, rope, fmt)
    cache = torch.zeros(
        num_blocks, 1, block_size, row, dtype=torch.uint8, device="cuda"
    )
    cat_and_cache_mla(
        kv[..., :rank],
        kv[..., rank:],
        cache,
        torch.arange(kv.shape[0], device="cuda"),
        layer._k_scale,
        shuffled_kv_cache=True,
        fp4_kv_format=fmt,
    )
    actual = layer._forward_decode(q, cache.view(-1, 1, row), metadata)
    decoded = _fp4_decoded(kv, fmt).view(num_blocks, block_size, rank + rope)
    expected = _reference(layer, q, decoded, metadata)
    torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("block_size,shuffled", [(16, False), (64, True)])
def test_triton_mla_decode_ignores_query_less_padded_sequences(block_size, shuffled):
    """In a step where a DP peer prefills, a decoding rank's q holds only its
    own rows while the per-sequence metadata is padded to the group's batch
    with query-less sequences; the real sequences must still be attended."""
    layer, q, cache, metadata, kv = _inputs(block_size, 1, "fp8_qkv", shuffled)
    expected = _reference(layer, q, kv, metadata)
    pad = 5
    num_seqs = metadata.context_lens.numel()
    padded = AttentionMetaData(
        cu_seqlens_q=torch.nn.functional.pad(
            metadata.cu_seqlens_q, (0, pad), value=num_seqs
        ),
        context_lens=torch.nn.functional.pad(metadata.context_lens, (0, pad)),
        max_seqlen_q=1,
        max_seqlen_k=metadata.max_seqlen_k,
        block_tables=torch.nn.functional.pad(metadata.block_tables, (0, 0, 0, pad)),
    )
    actual = layer._forward_decode(q, cache, padded)
    torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_triton_mla_decode_graph_replays_updated_metadata():
    layer, q, cache, metadata, kv = _inputs(16, 4, "fp8_qkv")
    # The base metadata builder leaves the maximum length unknown at capture.
    metadata.max_seqlen_k = 0
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer._forward_decode(q, cache, metadata)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer._forward_decode(q, cache, metadata)
    metadata.context_lens.sub_(3)
    metadata.block_tables.copy_(metadata.block_tables.roll(1, 0))
    graph.replay()
    eager = layer._forward_decode(q, cache, metadata)
    torch.testing.assert_close(actual, eager, atol=0, rtol=0)
    expected = _reference(layer, q, kv, metadata)
    torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.02)
