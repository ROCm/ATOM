"""Contracts behind the GLM-5.2 SGLang FP8 regression and the TP4 MTP fault."""

from types import SimpleNamespace

import torch

from atom.plugin.sglang.attention_backend.glm52_dsa_backend import (
    verify_kernel_positions,
)
from atom.plugin.sglang.glm52_dsa_bridge import GLM52_GRAPH_SEQ_LEN_CAPACITY
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
