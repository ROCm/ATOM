# SPDX-License-Identifier: MIT
"""CSA2 operator/ownership differential tests against pinned upstream methods."""

import pytest
import torch

pytest.importorskip("aiter", reason="the compressor and cached RoPE are AITER kernels")

from atom.model_ops.deepseek_v41.compressor import Compressor
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding


@pytest.mark.parametrize("original_length,base", [(0, 10000), (65536, 160000)])
def test_rope_matches_reference_interleaved_and_inverse(
    reference, original_length, base
):
    torch.manual_seed(723)
    target = RotaryEmbedding(
        64, 1024, base=base, original_length=original_length, factor=16
    )
    frequencies = reference.precompute_freqs_cis(
        64, 1024, original_length, base, 16, 32, 1
    )
    torch.testing.assert_close(target.frequencies, frequencies, rtol=0, atol=0)
    positions = torch.tensor([0, 1, 97, 511, 1023])
    for shape in ((2, 5, 512), (2, 5, 8, 512)):
        source = torch.randn(shape, dtype=torch.bfloat16)
        for inverse in (False, True):
            expected = source.clone()
            reference.apply_rotary_emb(
                expected[..., -64:], frequencies[positions], inverse
            )
            actual = target(source.clone(), positions, inverse=inverse)
            assert torch.equal(actual, expected)


def test_rope_pair_rotates_both_as_two_separate_calls_would():
    """The two-channel entry is a launch count, not a different rotation.

    On CPU this takes the fallback, so what it pins is the contract every
    caller depends on: both tensors rotated, in place, in argument order.
    """
    torch.manual_seed(724)
    target = RotaryEmbedding(64, 1024, base=10000)
    positions = torch.tensor([0, 1, 97, 511, 1023])
    query = torch.randn((2, 5, 8, 512), dtype=torch.bfloat16)
    latent = torch.randn((2, 5, 512), dtype=torch.bfloat16)
    expected = target(query.clone(), positions), target(latent.clone(), positions)

    actual = target.pair(query, latent, positions)

    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert actual[0] is query and actual[1] is latent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_all_chunk_boundaries_against_official_decode(
    reference, single_rank, ratio
):
    torch.manual_seed(317)
    args = reference.ModelArgs(
        dim=64, head_dim=64, compress_ratios=(ratio,), max_batch_size=2, max_seq_len=16
    )
    with reference.set_dtype(torch.bfloat16):
        source = reference.Compressor(args, 0)
        target = Compressor(64, 64, ratio, args.norm_eps).cuda()
        upstream = dict(source.named_parameters())
        # Filled one disk tensor at a time through the merged loader, which is
        # how `packed_modules_mapping` fills `wkv_gate` -- so this also pins
        # that the pooling half is shard 0 and the gate is shard 1. Feeding
        # both halves the same numbers would make that unfalsifiable.
        for shard_id, name in enumerate(("wkv",) if ratio == 1 else ("wkv", "wgate")):
            value = (torch.randn(64, 64) * 0.1).bfloat16()
            target.wkv_gate.weight_loader(target.wkv_gate.weight, value, shard_id)
            upstream[f"{name}.weight"].data.copy_(value)
        for norm in (target.norm.weight, upstream["norm.weight"]):
            norm.data.fill_(1)
        hidden = torch.randn(2, 11, 64, dtype=torch.bfloat16)
        expected = []
        for position in range(hidden.shape[1]):
            latent = source(hidden[:, position : position + 1], position)
            if latent is not None:
                expected.append(latent)
        expected = torch.cat(expected, dim=1)
        for chunks in ((11,), (1, 3, 2, 5), (2, 1, 7, 1)):
            position, tail, actual = 0, None, []
            for length in chunks:
                latent, tail = target(
                    hidden[:, position : position + length].cuda(), position, tail
                )
                if latent is not None:
                    actual.append(latent.cpu())
                position += length
            torch.testing.assert_close(
                torch.cat(actual, dim=1), expected, rtol=1 / 128, atol=2**-10
            )
            assert (tail is None) == (ratio == 1)


@pytest.mark.parametrize("ratio,rows", [(1, 32), (2, 64)])
def test_compressor_holds_pool_and_gate_in_one_matrix(single_rank, ratio, rows):
    """One parameter, and the two disk tensors land in their own row ranges.

    A ratio-1 layer ships no `wgate` at all, so its matrix is the pooling half
    alone -- and the merged loader has to refuse a gate shard there rather
    than write past the rows it owns.
    """
    torch.manual_seed(451)
    compressor = Compressor(96, 32, ratio, 1e-6)
    assert [name for name, _ in compressor.named_parameters()] == [
        "wkv_gate.weight",
        "norm.weight",
    ]
    assert compressor.wkv_gate.weight.shape == (rows, 96)
    shards = [torch.full((32, 96), float(i + 1)) for i in range(ratio)]
    for shard_id, value in enumerate(shards):
        compressor.wkv_gate.weight_loader(compressor.wkv_gate.weight, value, shard_id)
    for shard_id, value in enumerate(shards):
        loaded = compressor.wkv_gate.weight[shard_id * 32 : (shard_id + 1) * 32]
        assert torch.equal(loaded.float(), value)
    if ratio == 1:
        with pytest.raises(AssertionError):
            compressor.wkv_gate.weight_loader(compressor.wkv_gate.weight, shards[0], 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_compressor_project_hands_back_strided_halves(single_rank):
    """The halves the two writers read: zero-copy, unit inner stride.

    `fused_compress_attn` and `update_compressor_states` both take a row
    stride but address the head dimension as `col_off + d`, so a trailing
    chunk of one GEMM is exactly what they accept -- and making the halves
    contiguous instead would put the copy back that the fusion removed.
    """
    torch.manual_seed(451)
    compressor = Compressor(96, 32, 2, 1e-6).cuda()
    compressor.wkv_gate.weight.data.normal_(std=0.1)
    values, scores = compressor.project(torch.randn(7, 96, dtype=torch.bfloat16).cuda())
    for half in (values, scores):
        assert half.shape == (7, 32)
        assert half.stride() == (64, 1)
        assert half.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()


def test_draft_context_kv_fusion_sees_the_kv_half(single_rank, small_config):
    """The fused context-KV GEMM reads a shard of `wqkv_a`, not a `wkv` layer.

    It concatenates one projection per stage and declines when a stage cannot
    supply one. Nothing else asserts that a real attention module supplies it,
    so a rename on the model side turns the fusion off and only costs speed --
    the per-stage fallback it falls back to is numerically identical.
    """
    from atom.models.deepseek_v41.config import build_attention_topology
    from atom.models.deepseek_v41.dspark import DraftAttention

    spec = build_attention_topology(small_config)[1]
    attention = DraftAttention(small_config, spec)
    shard = attention.wkv_shard
    assert shard is not None
    assert shard.output_size == small_config.head_dim
    assert shard.native_a8_group_rows is not None
    rows = attention.wqkv_a.output_sizes[0]
    assert torch.equal(
        shard.weight.view(torch.uint8),
        attention.wqkv_a.weight.view(torch.uint8)[rows:],
    )
    group = attention.wqkv_a.weight_scale_row_group
    assert torch.equal(
        shard.weight_scale.view(torch.uint8),
        attention.wqkv_a.weight_scale.view(torch.uint8)[rows // group :],
    )
