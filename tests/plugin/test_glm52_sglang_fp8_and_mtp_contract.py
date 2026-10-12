"""Contracts behind the GLM-5.2 SGLang FP8 regression and the TP4 MTP fault."""

from types import SimpleNamespace

import torch

from atom.plugin.sglang.attention_backend.glm52_dsa_backend import (
    verify_kernel_positions,
)
from atom.plugin.sglang.glm52_dsa_bridge import (
    GLM52_GRAPH_SEQ_LEN_CAPACITY,
    resolve_draft_decode_context_lens,
    resolve_target_verify_lens,
)
from atom.plugin.sglang.models.glm52_dsa import short_verify_skips_indexer


def test_fp8_decode_keeps_the_indexer():
    metadata = SimpleNamespace(max_seqlen_q=1, max_seqlen_k=128)
    assert not short_verify_skips_indexer(
        metadata, is_prefill=False, dcp_world_size=1, topk_tokens=2048
    )


def test_short_mtp_verify_skips_indexer():
    metadata = SimpleNamespace(max_seqlen_q=4, max_seqlen_k=128)
    assert short_verify_skips_indexer(
        metadata, is_prefill=False, dcp_world_size=1, topk_tokens=2048
    )


def test_prefill_and_long_context_keep_the_indexer():
    short = SimpleNamespace(max_seqlen_q=4, max_seqlen_k=128)
    long = SimpleNamespace(max_seqlen_q=4, max_seqlen_k=4096)
    assert not short_verify_skips_indexer(
        short, is_prefill=True, dcp_world_size=1, topk_tokens=2048
    )
    assert not short_verify_skips_indexer(
        long, is_prefill=False, dcp_world_size=1, topk_tokens=2048
    )


def test_verify_positions_ignore_the_graph_fill_only():
    draft = 4
    real = verify_kernel_positions(
        torch.tensor([800, 20], dtype=torch.int64), 2, draft, torch.device("cpu")
    )
    assert real.tolist() == [796, 797, 798, 799, 16, 17, 18, 19]
    sentinel = int(GLM52_GRAPH_SEQ_LEN_CAPACITY)
    filled = verify_kernel_positions(
        torch.tensor([sentinel, 800], dtype=torch.int64), 2, draft, torch.device("cpu")
    )
    assert filled.tolist()[:draft] == [0, 1, 2, 3]
    assert filled.tolist()[draft:] == [796, 797, 798, 799]


def test_graph_fill_does_not_override_a_real_mtp_context():
    sentinel = int(GLM52_GRAPH_SEQ_LEN_CAPACITY)
    batch = SimpleNamespace(
        batch_size=2,
        seq_lens_cpu=torch.tensor([sentinel, 800], dtype=torch.int32),
        spec_info=SimpleNamespace(num_tokens_per_req=1),
    )
    positions = torch.tensor([1499, 19], dtype=torch.int64)
    lens = resolve_draft_decode_context_lens(batch, positions, 2, 0)
    assert lens.tolist() == [1500, 801]

    verify_batch = SimpleNamespace(
        batch_size=2,
        seq_lens_cpu=torch.tensor([800, sentinel], dtype=torch.int32),
    )
    verify_positions = torch.tensor(
        [sentinel - 4, sentinel - 3, sentinel - 2, sentinel - 1, 796, 797, 798, 799],
        dtype=torch.int64,
    )
    prefix, context = resolve_target_verify_lens(
        verify_batch,
        verify_positions,
        2,
        4,
        use_positions=True,
    )
    assert prefix.tolist() == [796, 796]
    assert context.tolist() == [800, 800]

    # Both sources at the fill: this is the capture allocation, keep the capacity.
    wide = SimpleNamespace(
        batch_size=1,
        seq_lens_cpu=torch.tensor([sentinel], dtype=torch.int32),
    )
    wide_positions = torch.arange(sentinel - 4, sentinel, dtype=torch.int64)
    _, wide_context = resolve_target_verify_lens(
        wide, wide_positions, 1, 4, use_positions=True
    )
    assert wide_context.tolist() == [sentinel]

    # Capture staging: kernel positions are the draft only, seq_lens is still
    # the fill. The fill must not become max_seqlen_k.
    capture_batch = SimpleNamespace(
        batch_size=1,
        seq_lens_cpu=torch.tensor([sentinel], dtype=torch.int32),
    )
    capture_positions = torch.arange(4, dtype=torch.int64)
    _, capture_context = resolve_target_verify_lens(
        capture_batch,
        capture_positions,
        1,
        4,
        use_positions=True,
    )
    assert capture_context.tolist() == [4]


def test_head_chunk_keeps_the_padded_flash_stride():
    # _drop_rope_pad slices the logical head dim and leaves the padded pitch.
    # Packing that view is what makes the hd256 kernel walk off the allocation.
    padded = 320
    logical = 256
    q = torch.zeros(2, 16, padded)
    viewed = q[..., :logical]
    chunk = viewed[:, 8:16]
    assert chunk.shape[-1] == logical
    assert chunk.stride(-1) == 1
    assert viewed.stride(-2) == padded
    assert chunk.stride(-2) == padded
    packed = chunk.contiguous()
    assert packed.stride(-2) == logical
