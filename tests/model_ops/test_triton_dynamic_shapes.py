# SPDX-License-Identifier: MIT
"""Dynamic batch/length bounds must preserve results without new JIT variants."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


@contextmanager
def one_variant(*kernels):
    # Isolate this assertion from shapes warmed by other tests. Restore their
    # entries afterward; no disk cache or compiled module is removed.
    caches = [k.device_caches[torch.cuda.current_device()][0] for k in kernels]
    saved = [dict(cache) for cache in caches]
    for cache in caches:
        cache.clear()
    try:
        yield
        torch.cuda.synchronize()
        for kernel, cache in zip(kernels, caches):
            assert len(cache) == 1, (kernel.__name__, len(cache))
    finally:
        for cache, entries in zip(caches, saved):
            cache.clear()
            cache.update(entries)


@pytest.mark.parametrize("groups", [1, 4])
def test_gemma_rows_reuse_kernel(groups):
    from atom.model_ops import triton_gemma_rmsnorm as m

    torch.manual_seed(41)
    weight = torch.randn(groups * 128, device="cuda", dtype=torch.bfloat16)
    with one_variant(m._gemma_rmsnorm_kernel):
        for rows in (1, 15, 16, 17, 127, 305):
            x = torch.randn(rows, groups * 128, device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            out, residual_out = m.gemma_rmsnorm_triton(x, weight, 1e-6, residual, 128)
            combined = (x.float() + residual.float()).reshape(rows, groups, 128)
            expected = combined * torch.rsqrt(
                combined.square().mean(-1, keepdim=True) + 1e-6
            )
            expected *= 1 + weight.float().view(groups, 128)
            torch.testing.assert_close(out, expected.reshape_as(x).to(x.dtype))
            torch.testing.assert_close(residual_out, combined.reshape_as(x).to(x.dtype))


@pytest.mark.parametrize("tiled", [False, True])
def test_mrope_position_stride_reuses_kernel(tiled):
    from atom.model_ops import triton_mrope as m

    torch.manual_seed(42)
    angles = torch.randn(64, 32, device="cuda")
    rotary = SimpleNamespace(
        mrope_section=[8, 12, 12],
        mrope_interleaved=True,
        rotary_dim=64,
        is_neox_style=True,
        cos_cache=angles.cos().to(torch.bfloat16),
        sin_cache=angles.sin().to(torch.bfloat16),
    )
    columns = torch.arange(32, device="cuda")
    axes = torch.where(columns % 3 == 1, 1, torch.where(columns % 3 == 2, 2, 0))
    kernel = m._mrope_qk_tiled_kernel if tiled else m._mrope_qk_kernel
    with one_variant(kernel):
        for n in ((128, 129, 255, 256, 257) if tiled else (1, 15, 16, 17, 127)):
            q = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(n, 256, device="cuda", dtype=torch.bfloat16)
            positions = torch.randint(0, 64, (3, n), device="cuda")
            output = m.try_mrope_qk_fused(rotary, positions, q, k, 2, 1, 256)
            selected = positions[axes].T
            cos = rotary.cos_cache[selected, columns].float()[:, None, :]
            sin = rotary.sin_cache[selected, columns].float()[:, None, :]
            for x, actual in zip((q, k), output):
                heads = x.reshape(n, -1, 256).float()
                left, right = heads[..., :32], heads[..., 32:64]
                expected = torch.cat(
                    (
                        left * cos - right * sin,
                        right * cos + left * sin,
                        heads[..., 64:],
                    ),
                    -1,
                )
                torch.testing.assert_close(
                    actual, expected.reshape_as(x).to(x.dtype), atol=0.016, rtol=0.016
                )


@pytest.mark.parametrize(
    "sizes,heads,single",
    [
        ((17, 18, 19, 20, 21, 22), 1, True),
        ((257, 259, 261, 263, 265, 267), 4, False),
        ((1025, 2049, 3073, 4097, 6145), 12, False),
    ],
)
def test_cached_kv_lengths_reuse_kernels(sizes, heads, single):
    from atom.model_ops import triton_fused_qkv_quant as m

    kernels = (m._kv_quant,) if single else (m._kv_amax, m._kv_quant)
    with one_variant(*kernels):
        for n in sizes:
            inputs = []
            for dim, value in ((4, 2), (3, -9)) if single else ((192, 2), (128, -9)):
                storage = torch.full(
                    (n * heads * dim + 128,), 128, device="cuda", dtype=torch.bfloat16
                )
                x = storage[: n * heads * dim].reshape(n, heads, dim)
                x.fill_(value)
                # Put the maximum in the tail, including odd unrolled trips.
                x[-1].mul_(2)
                inputs.append(x)
            k8, v8, ks, vs = m.fused_kv_per_tensor_quant(*inputs)
            for actual, sign in ((k8, 1), (v8, -1)):
                assert torch.all(actual[:-1].float() == sign * 224)
                assert torch.all(actual[-1].float() == sign * 448)
            assert ks.item() == pytest.approx(4 / 448)
            assert vs.item() == pytest.approx(18 / 448)


@pytest.mark.parametrize("prefill", [False, True])
def test_mla_dynamic_bounds_preserve_request_regions(prefill):
    from atom.model_ops import attention_mla as m

    kernel = (
        m._convert_req_index_to_global_index_dsa_prefill_kernel
        if prefill
        else m._convert_req_index_to_global_index_kernel
    )
    workspace = torch.empty(64 * 128 + 16, device="cuda", dtype=torch.int32)
    dense = torch.arange(64 * 256, device="cuda", dtype=torch.int32)
    with one_variant(kernel):
        for n in (1, 15, 16, 17, 31, 32, 33):
            ptr = torch.arange(n + 1, device="cuda", dtype=torch.int32)
            topk = torch.arange(128, device="cuda", dtype=torch.int32).repeat(n, 1)
            topk[:, 3] = -1
            topk[:, 5] = 256
            workspace.fill_(7)
            if prefill:
                table = (
                    torch.arange(n * 32, device="cuda", dtype=torch.int32).reshape(
                        n, 32
                    )
                    + 100
                )
                out = m.triton_convert_req_index_to_global_index_dsa_prefill(
                    ptr,
                    ptr * 125,
                    ptr[:-1].contiguous(),
                    topk,
                    table,
                    ptr * 256,
                    PAGE_SIZE=16,
                    NUM_TOPK_TOKENS=128,
                    BLOCK_N=128,
                    out=workspace,
                    seq_local=True,
                )
                base = (ptr[:-1] * 32 + 100) * 16
            else:
                out = m.triton_convert_req_index_to_global_index(
                    ptr,
                    ptr * 256,
                    ptr * 125,
                    dense,
                    topk,
                    NUM_TOPK_TOKENS=128,
                    out=workspace,
                )
                base = ptr[:-1] * 256
            expected = base[:, None] + torch.arange(
                125, device="cuda", dtype=torch.int32
            )
            expected[:, 3] = expected[:, 5] = 0
            assert torch.equal(out[: n * 125], expected.flatten())
            assert torch.all(workspace[n * 125 :] == 7)


@pytest.mark.parametrize("sizes", [(1,), (15, 17, 31, 33), (16, 32, 48)])
def test_ple_request_count_reuses_kernel(sizes):
    from atom.model_ops.qwen4_exp.ops import ple as m

    torch.manual_seed(43)
    weight = torch.randn(32, 3, device="cuda", dtype=torch.bfloat16)
    with one_variant(m._conv):
        for n in sizes:
            x = torch.randn(n * 2, 32, device="cuda", dtype=torch.bfloat16)
            slots = torch.arange(n, device="cuda", dtype=torch.int32)
            state = torch.zeros(64, 32, 2, device="cuda", dtype=x.dtype)
            starts = torch.arange(n + 1, device="cuda", dtype=torch.int32) * 2
            actual = m.dilated_causal_conv1d(
                x,
                weight,
                state,
                starts,
                slots,
                slots,
                torch.zeros(n, device="cuda", dtype=torch.bool),
                1,
            )
            ref = F.conv1d(
                x.reshape(n, 2, 32).transpose(1, 2).float(),
                weight[:, None, :].float(),
                groups=32,
                padding=2,
            )[..., :2].to(x.dtype)
            ref = F.silu(ref.float()).to(x.dtype).transpose(1, 2).reshape_as(x)
            torch.testing.assert_close(actual, ref, atol=0.016, rtol=0.016)


def test_qsa_real_requests_change_within_and_between_buckets():
    from atom.model_ops.qwen4_exp.ops import qsa as m

    with one_variant(m._draft_decode_metadata):
        for n, real in ((64, 1), (64, 15), (64, 16), (65, 17), (128, 31), (129, 33)):
            lengths = torch.full((n,), 19, device="cuda", dtype=torch.int32)
            table = torch.arange(n * 32, device="cuda", dtype=torch.int32).reshape(
                n, 32
            )
            rejects = torch.full((real,), 3, device="cuda", dtype=torch.int32)
            slots, positions, requests, compressed = [
                torch.empty(n, device="cuda", dtype=torch.int64) for _ in range(4)
            ]
            m.qsa_draft_decode_metadata(
                lengths,
                table,
                rejects,
                slots,
                positions,
                requests,
                compressed,
                real,
                16,
                4,
            )
            ids = torch.arange(real, device="cuda")
            assert torch.equal(slots[:real], ids * 512 + 15)
            assert torch.equal(compressed[:real], (ids * 512 + 15) // 4)
            assert torch.equal(requests[:real], ids)
            assert torch.all(positions[:real] == 15) and torch.all(lengths[:real] == 16)
            for x in (slots, positions, requests, compressed):
                assert torch.all(x[real:] == -1)
            assert not lengths[real:].count_nonzero()


def test_m3_context_and_batch_reuse_kernels():
    from atom.model_ops.minimax_m3 import indexer_candidate_exchange as ex
    from atom.model_ops.minimax_m3 import indexer_context_parallel as cp

    cache = torch.zeros(64, 128, 128, device="cuda", dtype=torch.bfloat16)
    with one_variant(cp._context_score, ex._local_topk, ex._merge_topk):
        for n, blocks in ((1, 33), (15, 35), (16, 37), (17, 39), (31, 41)):
            q = torch.ones(n, 4, 128, device="cuda", dtype=torch.bfloat16)
            table = torch.arange(64, device="cuda", dtype=torch.int32).repeat(n, 1)
            lengths = torch.full((n,), blocks * 128, device="cuda", dtype=torch.int32)
            scores = cp.indexer_context_scores(
                q, cache, table, lengths, blocks * 128, 0, 4, 1, 0.1
            )
            assert not scores.count_nonzero()
            keys = ex.local_candidate_keys(scores, lengths, 4, 0, 4, 1, 64)
            indices, _, _ = ex.merge_candidate_keys(keys, table, lengths, 4, 0, 0, 1)
            assert torch.all((indices >= 0) & (indices < blocks))


def test_packed_gather_bounds_reuse_kernel():
    from atom.model_ops.attentions.deepseek_v41 import packed_rows as m
    from atom.model_ops.blockscale import quantize_fp8

    torch.manual_seed(44)
    x = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)
    pool = m.pack_rows(*quantize_fp8(x)).flatten()
    ref = quantize_fp8(x, dequantize=True)
    order = torch.randperm(128, device="cuda")
    addresses = ((order * 33 * 2) | 1).contiguous()
    addresses[111:] = 2**60
    ptr = torch.tensor([0, 7, 23, 58, 111], device="cuda", dtype=torch.int32)
    with one_variant(m._gather_prefix):
        for capacity, first, end in (
            (1, 0, 1),
            (15, 1, 2),
            (16, 1, 2),
            (17, 2, 4),
            (33, 0, 4),
            (128, 0, 4),
        ):
            backing = torch.full((capacity + 16, 32), 7, device="cuda", dtype=x.dtype)
            out = backing[:capacity]
            m.gather_prefix_rows(pool, addresses, ptr, out, first, end)
            start, stop = (0, 7, 23, 58, 111)[first], (0, 7, 23, 58, 111)[end]
            count = min(capacity, stop - start)
            assert torch.equal(out[:count], ref[order[start : start + count]])
            assert not out[count:].count_nonzero()
            assert torch.all(backing[capacity:] == 7)


def test_cached_kv_large_lengths_keep_a_single_integer_signature():
    from atom.model_ops import triton_fused_qkv_quant as m

    # Compile only: exercise signed/unsigned boundaries without making the
    # regular suite allocate multi-GB tensors or launching into tiny storage.
    x = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(1, device="cuda", dtype=torch.float8_e4m3fn)
    partial = torch.empty((2, 512), device="cuda")
    scales = torch.empty(2, device="cuda")
    with one_variant(m._kv_amax, m._kv_quant):
        for tokens in (2**31 - 1, 2**31, 2**32 + 1):
            m._kv_amax.warmup(
                x, x, partial, tokens, 192, 128, 512, 8192, grid=(512, 2), num_warps=8
            )
            m._kv_quant.warmup(
                x,
                x,
                y,
                y,
                partial,
                scales,
                tokens,
                192,
                128,
                512,
                8192,
                False,
                grid=(4096, 2),
                num_warps=8,
            )


@pytest.mark.parametrize("alignment", [0, 1])
def test_m3_local_topk_reuses_each_alignment_class(alignment):
    from atom.model_ops.minimax_m3 import indexer_candidate_exchange as m

    lengths = torch.full((4,), 1048576, device="cuda", dtype=torch.int32)
    with one_variant(m._local_topk):
        for blocks in (1040, 1056, 1072):
            scores = torch.zeros((4, 4, blocks + alignment), device="cuda")
            m.local_candidate_keys(scores, lengths, 16, 0, 4, 1, 8192)


@pytest.mark.parametrize("blocks", [1025, 1536, 2048, 2049])
def test_m3_local_topk_long_rows_match_torch(blocks):
    from atom.model_ops.minimax_m3 import indexer_candidate_exchange as m

    torch.manual_seed(45)
    # Unique signed scores make the expected ordering independent of tie rules.
    scores = torch.randperm(16 * blocks, device="cuda").reshape(4, 4, blocks).float()
    scores -= 8 * blocks
    lengths = torch.full((4,), blocks * 4 * 128, device="cuda", dtype=torch.int32)
    keys = m.local_candidate_keys(scores, lengths, 16, 0, 4, 1, blocks * 4)
    indices = (keys & 0xFFFF) - 1
    expected = scores.topk(16, dim=-1).indices * 4
    assert torch.equal(indices, expected)
