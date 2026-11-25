# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Which last-page-length array `_forward_prefill_mla` reads, and when.

There are two of them and they are indexed differently. `kv_last_page_lens` is
per SEQUENCE -- how many tokens sit in that sequence's last KV block -- and the
prefill builder fills it only when there is a drafter or a cached prefix,
leaving it `None` otherwise. `sparse_kv_last_page_lens` is per QUERY TOKEN, all
ones, because a DSA selection is packed at page_size 1; the sparse builder
always fills it and deliberately does not touch the dense one.

Only sparse MLA reaches this function (dense prefill runs on
`_forward_prefill_mha`), and reading the dense array there is a crash on a
model whose sparse prefill has neither a drafter nor a cached prefix. GLM-5.3-Flash served with
`--no-enable_prefix_caching` is that model: every rank died with
`'NoneType' object has no attribute 'shape'` on the first request long enough
to exceed `index_topk`, which is every 16-shot GSM8K prompt.
"""

from types import SimpleNamespace

import torch
from aiter_stub import atom_under_stubs

# Against the stand-in AITER, so Pre Checkin (no AITER) runs this file.
with atom_under_stubs() as load:
    attention_mla = load("atom.model_ops.attention_mla")
    head_groups = load("atom.model_ops.attentions.sparse_mla_head_groups")
MLAAttention = attention_mla.MLAAttention
PERSISTENT_WORK_BUFFERS = head_groups.PERSISTENT_WORK_BUFFERS

KV_LORA_RANK = 8
QK_ROPE_HEAD_DIM = 4
ENTRY = KV_LORA_RANK + QK_ROPE_HEAD_DIM
# BF16 with 16 < heads < 128: the shape main's asm path padded to 128 heads.
NUM_HEADS = 32


def _attn():
    """An `MLAAttention` with only the fields `_forward_prefill_mla` reads.

    Built by `__new__` on purpose: a real one needs a distributed group, a
    checkpoint and a GPU, none of which the metadata question depends on.
    """
    self = MLAAttention.__new__(MLAAttention)
    self.is_sparse_mla = True
    self.dcp_world_size = 1
    self.use_seg_mla = False
    self.head_repeat_factor = 1
    self.head_pad = 0
    self.num_heads = NUM_HEADS
    self.kv_lora_rank = KV_LORA_RANK
    self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    self.dtype = torch.bfloat16
    self.kv_cache_dtype = "bf16"
    self.scale = 1.0
    self.sparse_prefill_head_groups = None
    self.sparse_kv_indices_buffer = torch.zeros(64, dtype=torch.int32)
    # The v-up/o projections are past the point under test.
    self._v_up_proj_and_o_proj = lambda o: o
    return self


def _capture(monkeypatch):
    """Record what reaches the prefill kernel instead of launching it."""
    seen = {}

    def fake_mla_decode_fwd(
        q, kv, o, cu_seqlens_q, kv_indptr, kv_indices, last_page_lens, max_q_len, **kw
    ):
        seen.update(
            q=q,
            num_kv_splits=kw.get("num_kv_splits"),
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            last_page_lens=last_page_lens,
            max_q_len=max_q_len,
            work={name: kw.get(name) for name in PERSISTENT_WORK_BUFFERS},
        )
        return o, None

    monkeypatch.setattr(attention_mla, "mla_decode_fwd", fake_mla_decode_fwd)
    return seen


def test_sparse_prefill_does_not_read_the_dense_last_page_lens(monkeypatch):
    """A sparse prefill with no drafter and no cached prefix must still run.

    With bf16 KV it runs on the persistent decode kernel, fed the step's
    sparse-prefill work plan and an explicit split count. Without the split
    count aiter drops the plan for large 16-head BF16 batches, and that form
    mis-addresses KV past 4 GiB on gfx950 (ROCm/ATOM#3120).
    """
    seen = _capture(monkeypatch)
    attn = _attn()

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
        sparse_kv_indices=attn.sparse_kv_indices_buffer,
        sparse_kv_last_page_lens=sparse_kv_last_page_lens,
        **{
            f"sparse_prefill_{name}": torch.zeros(1) for name in PERSISTENT_WORK_BUFFERS
        },
    )

    q = torch.zeros(n_tokens, NUM_HEADS, ENTRY, dtype=torch.bfloat16)
    kv = torch.zeros(64 * 4, ENTRY, dtype=torch.bfloat16)
    out = attn._forward_prefill_mla(q, kv, md)

    assert out.shape == (n_tokens, NUM_HEADS, KV_LORA_RANK)
    assert seen["last_page_lens"] is sparse_kv_last_page_lens
    assert seen["cu_seqlens_q"] is sparse_cu_seqlens_q
    assert seen["max_q_len"] == 1
    assert seen["q"].shape[1] == NUM_HEADS
    assert seen["num_kv_splits"] is not None
    for name in PERSISTENT_WORK_BUFFERS:
        assert seen["work"][name] is getattr(md, f"sparse_prefill_{name}")
