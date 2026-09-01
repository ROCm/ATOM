# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Which last-page-length array `_forward_prefill_mla` reads, and when.

There are two of them and they are indexed differently. `kv_last_page_lens` is
per SEQUENCE -- how many tokens sit in that sequence's last KV block -- and the
prefill builder fills it only when there is a drafter or a cached prefix,
leaving it `None` otherwise. `sparse_kv_last_page_lens` is per QUERY TOKEN, all
ones, because a DSA selection is packed at page_size 1; the sparse builder
always fills it and deliberately does not touch the dense one.

So the dense array is available exactly where the dense branch runs, and
reading it before the branch is a crash on a model whose sparse prefill has
neither a drafter nor a cached prefix. GLM-5.3-Flash served with
`--no-enable_prefix_caching` is that model: every rank died with
`'NoneType' object has no attribute 'shape'` on the first request long enough
to exceed `index_topk`, which is every 16-shot GSM8K prompt.

The dense test is the other half. `cu_seqlens_q` is padded to the step's
`running_bs` while the per-seq array is not, so the dense branch must still cut
the q-cums down to the sequences the kernel is being told about.
"""

from types import SimpleNamespace

import pytest
import torch

try:
    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention

    _IMPORT_ERR = None
except ImportError as _e:  # aiter/triton absent on a CPU-only runner
    _IMPORT_ERR = str(_e)

needs_atom_mla = pytest.mark.skipif(
    _IMPORT_ERR is not None, reason=f"requires full atom import env: {_IMPORT_ERR}"
)

KV_LORA_RANK = 8
QK_ROPE_HEAD_DIM = 4
ENTRY = KV_LORA_RANK + QK_ROPE_HEAD_DIM
NUM_HEADS = 2


def _attn(is_sparse: bool):
    """An `MLAAttention` with only the fields `_forward_prefill_mla` reads.

    Built by `__new__` on purpose: a real one needs a distributed group, a
    checkpoint and a GPU, none of which the metadata question depends on.
    """
    self = MLAAttention.__new__(MLAAttention)
    self.is_sparse_mla = is_sparse
    self.dcp_world_size = 1
    self.use_seg_mla = False
    self.head_repeat_factor = 1
    self.head_pad = 0
    self.num_heads = NUM_HEADS
    self.kv_lora_rank = KV_LORA_RANK
    self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    self.dtype = torch.float32
    self.kv_cache_dtype = "bf16"  # not fp8 -> the mla_prefill_fwd branch
    self.scale = 1.0
    self.sparse_kv_indices_buffer = torch.zeros(64, dtype=torch.int32)
    # The v-up/o projections are past the point under test.
    self._v_up_proj_and_o_proj = lambda o: o
    return self


def _capture(monkeypatch):
    """Record what reaches the prefill kernel instead of launching it."""
    seen = {}

    def fake_mla_prefill_fwd(
        q, kv, o, cu_seqlens_q, kv_indptr, kv_indices, last_page_lens, max_q_len, *a
    ):
        seen.update(
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            last_page_lens=last_page_lens,
            max_q_len=max_q_len,
        )

    monkeypatch.setattr(attention_mla, "mla_prefill_fwd", fake_mla_prefill_fwd)
    return seen


@needs_atom_mla
def test_sparse_prefill_does_not_read_the_dense_last_page_lens(monkeypatch):
    """A sparse prefill with no drafter and no cached prefix must still run."""
    seen = _capture(monkeypatch)
    attn = _attn(is_sparse=True)
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(scheduled_bs=1)),
    )

    n_tokens = 5
    sparse_cu_seqlens_q = torch.arange(n_tokens + 1, dtype=torch.int32)
    sparse_kv_last_page_lens = torch.ones(n_tokens, dtype=torch.int32)
    md = SimpleNamespace(
        # What the builder leaves behind with neither a drafter nor a prefix.
        kv_last_page_lens=None,
        kv_indptr=None,
        kv_indices=None,
        # cu_seqlens_q is per-sequence and padded; sparse must not use it.
        cu_seqlens_q=torch.tensor([0, n_tokens, n_tokens], dtype=torch.int32),
        max_seqlen_q=n_tokens,
        sparse_cu_seqlens_q=sparse_cu_seqlens_q,
        sparse_kv_indptr=torch.arange(n_tokens + 1, dtype=torch.int32),
        sparse_kv_last_page_lens=sparse_kv_last_page_lens,
    )

    q = torch.zeros(n_tokens, NUM_HEADS, ENTRY)
    kv = torch.zeros(64 * 4, ENTRY)
    out = attn._forward_prefill_mla(q, kv, md)

    assert out.shape == (n_tokens, NUM_HEADS, KV_LORA_RANK)
    assert seen["last_page_lens"] is sparse_kv_last_page_lens
    assert seen["cu_seqlens_q"] is sparse_cu_seqlens_q
    assert seen["max_q_len"] == 1


@needs_atom_mla
def test_dense_prefill_cuts_the_q_cums_to_the_per_seq_array(monkeypatch):
    """Dense keeps reading the per-seq array -- and the cut it pays for."""
    seen = _capture(monkeypatch)
    attn = _attn(is_sparse=False)

    n_seqs, n_tokens = 2, 6
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(scheduled_bs=n_seqs)),
    )
    kv_last_page_lens = torch.tensor([3, 3], dtype=torch.int32)
    md = SimpleNamespace(
        kv_last_page_lens=kv_last_page_lens,
        kv_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        kv_indices=torch.zeros(2, dtype=torch.int32),
        # Padded past the two real sequences, as a padded step leaves it.
        cu_seqlens_q=torch.tensor([0, 3, 6, 6, 6], dtype=torch.int32),
        max_seqlen_q=3,
    )

    q = torch.zeros(n_tokens, NUM_HEADS, ENTRY)
    kv = torch.zeros(64 * 4, ENTRY)
    attn._forward_prefill_mla(q, kv, md)

    assert seen["last_page_lens"] is kv_last_page_lens
    assert seen["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert seen["cu_seqlens_q"].shape[0] == n_seqs + 1
