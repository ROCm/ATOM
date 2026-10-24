"""Contracts behind the GLM-5.2 SGLang FP8 regression and the TP4 MTP fault."""

from types import SimpleNamespace

import numpy as np
import torch

from atom.plugin.sglang.attention_backend.glm52_dsa_backend import (
    verify_kernel_positions,
)
from atom.plugin.sglang.glm52_dsa_bridge import (
    GLM52_GRAPH_SEQ_LEN_CAPACITY,
    _cap_context_to_host_seq,
    _cap_context_to_written_tokens,
    real_prefill_seq_lens,
    resolve_draft_decode_context_lens,
    resolve_target_verify_lens,
)
from atom.plugin.sglang.models.glm52_dsa import (
    glm52_short_context_is_dense,
    short_verify_skips_indexer,
)


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

    # Live replay: seq_lens is still the fill, but the host sum is the prompt.
    # Positions derived from the fill are not a second source. Walk the sum.
    stale = SimpleNamespace(
        batch_size=1,
        seq_lens_cpu=torch.tensor([sentinel], dtype=torch.int32),
        seq_lens_sum=8,
    )
    _, stale_context = resolve_target_verify_lens(
        stale,
        torch.arange(sentinel - 4, sentinel, dtype=torch.int64),
        1,
        4,
        use_positions=True,
    )
    assert stale_context.tolist() == [8]

    # A real context that lands on the capacity still matches its host sum.
    confirmed = SimpleNamespace(
        batch_size=1,
        seq_lens_cpu=torch.tensor([sentinel], dtype=torch.int32),
        seq_lens_sum=sentinel,
    )
    _, confirmed_context = resolve_target_verify_lens(
        confirmed,
        torch.arange(sentinel - 4, sentinel, dtype=torch.int64),
        1,
        4,
        use_positions=True,
    )
    assert confirmed_context.tolist() == [sentinel]

    padded = SimpleNamespace(
        batch_size=4,
        seq_lens_cpu=torch.tensor(
            [12, sentinel, sentinel, sentinel], dtype=torch.int32
        ),
        seq_lens_sum=12 + 3 * sentinel,
        num_padding=3,
    )
    padded_lens = resolve_draft_decode_context_lens(
        padded, torch.tensor([11, 0, 0, 0], dtype=torch.int64), 4, 0
    )
    assert padded_lens.tolist() == [13, 2, 2, 2]


def test_short_prefill_drops_the_graph_fill_seq_len():
    fill = GLM52_GRAPH_SEQ_LEN_CAPACITY
    batch = SimpleNamespace()
    fixed = real_prefill_seq_lens(
        np.array([fill, fill], dtype=np.int32),
        np.array([6, 4], dtype=np.int32),
        batch,
    )
    assert fixed.tolist() == [6, 4]
    kept = real_prefill_seq_lens(
        np.array([3000, 128], dtype=np.int32),
        np.array([128, 128], dtype=np.int32),
        batch,
    )
    assert kept.tolist() == [3000, 128]
    capture = SimpleNamespace(_atom_glm52_keep_graph_fill=True)
    held = real_prefill_seq_lens(
        np.array([fill], dtype=np.int32),
        np.array([6], dtype=np.int32),
        capture,
    )
    assert held.tolist() == [fill]


def test_host_seq_stops_a_graph_fill_the_token_row_still_looks_full():
    # Replay can leave both the GPU context and a stale req_to_token row at
    # the fill. The scheduler's CPU length is the one that was actually written.
    capped = _cap_context_to_host_seq(
        np.array([10240, 10240], dtype=np.int32),
        np.array([6, 10240], dtype=np.int32),
        query_len=4,
    )
    assert capped.tolist() == [10, 10240]


def test_context_stops_at_the_written_req_to_token_prefix():
    pool = SimpleNamespace(
        req_to_token=torch.tensor(
            [
                [4, 5, 6, 0, 0, 9],
                [0, 0, 0, 0, 0, 0],
                [7, 8, 9, 10, 11, 12],
            ],
            dtype=torch.int32,
        )
    )
    batch = SimpleNamespace(req_pool_indices=torch.tensor([0, 1, 2]))
    capped = _cap_context_to_written_tokens(
        np.array([10240, 10240, 4], dtype=np.int32), batch, pool
    )
    assert capped.tolist() == [3, 1, 4]


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


def test_short_glm_context_uses_dense_attention_kimi_does_not():
    from atom.plugin.sglang.attention_backend.sparse_mla_indexer import (
        sparse_attn_indexer_sglang_plugin_mode,
    )

    batch = SimpleNamespace(
        batch_size=1,
        seq_lens_cpu=torch.tensor([6], dtype=torch.int32),
    )
    assert not glm52_short_context_is_dense(batch, 2048)
    sparse_attn_indexer_sglang_plugin_mode._glm52_skip_short_topk = True
    try:
        assert glm52_short_context_is_dense(batch, 2048)
        decode = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([6], dtype=torch.int32),
            forward_mode=SimpleNamespace(
                is_decode_or_idle=lambda: True,
                is_target_verify=lambda: False,
            ),
        )
        assert not glm52_short_context_is_dense(decode, 2048)
        long = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([4096], dtype=torch.int32),
        )
        assert not glm52_short_context_is_dense(long, 2048)
    finally:
        sparse_attn_indexer_sglang_plugin_mode._glm52_skip_short_topk = False
