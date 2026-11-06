# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_UA_SPLIT_REDUCE_ONESHOT: the unified-attention decode asks aiter for the
one-shot split reduce (``split_reduce_oneshot=True``) only with ATOM_UA_SPLIT_REDUCE
on and an aiter that takes it. The kernel itself (bitwise the split reduce) is
checked in aiter's op_tests (test_unified_attn_split_reduce_oneshot_k4v8)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("the model modules import aiter", allow_module_level=True)

DEV = "cuda"


def _module(name):
    import importlib

    try:
        return importlib.import_module(name)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"{name} does not import here: {exc}")


@pytest.mark.parametrize(
    "split, oneshot, aiter_takes, expect",
    [
        ("1", "1", True, {"split_reduce": True, "split_reduce_oneshot": True}),
        ("1", "0", True, {"split_reduce": True}),
        ("1", "1", False, {"split_reduce": True}),
        ("0", "1", True, {}),
    ],
)
def test_unified_decode_asks_for_the_oneshot_reduce(
    monkeypatch, split, oneshot, aiter_takes, expect
):
    mha = _module("atom.model_ops.attention_mha")
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "1")
    monkeypatch.setenv("ATOM_UA_SPLIT_REDUCE", split)
    monkeypatch.setenv("ATOM_UA_SPLIT_REDUCE_ONESHOT", oneshot)
    seen = {}

    def fake_unified(q, k, v, out, **kw):
        seen.update(kw)

    monkeypatch.setattr(mha, "unified_attention", fake_unified)
    monkeypatch.setattr(mha, "unified_attention_takes_split_reduce", lambda: True)
    monkeypatch.setattr(
        mha, "unified_attention_takes_split_reduce_oneshot", lambda: aiter_takes
    )
    impl = object.__new__(mha.PagedAttentionImpl)
    nn.Module.__init__(impl)
    impl.kv_cache_dtype = "fp8_k4v8"
    impl.sliding_window = -1
    impl.scale, impl.k4v8, impl.kv_scale, impl.sinks = 1.0, True, None, None
    impl.use_flash_layout = False
    md = SimpleNamespace(
        cu_seqlens_q=None,
        context_lens=None,
        max_seqlen_q=4,
        max_seqlen_k=10,
        block_tables=None,
        equal_seqlen_q=True,
    )
    q = torch.zeros(8, 4, 256, dtype=torch.float8_e4m3fn, device=DEV)
    ctx = SimpleNamespace(attn_metadata=md, context=SimpleNamespace(is_draft=False))
    impl.paged_attention_unified(q, None, None, None, None, None, None, ctx)
    got = {k: seen[k] for k in ("split_reduce", "split_reduce_oneshot") if k in seen}
    assert got == expect
