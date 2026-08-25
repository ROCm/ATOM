# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the inplace o= contract on ATOM's kv-layout chunk.

Unlike the vk variant (which is a verbatim port of vLLM and can be
bit-compared against vllm.model_executor.layers.fla.ops.chunk), the kv
variant is ATOM-native and has no upstream reference with the same state
layout. So this test file focuses narrowly on the inplace o= contract:

    * o=None  (existing behavior — kernel allocates internally)  vs
    * o=preallocated  (new — kernel writes inplace, returns same storage)

must produce identical bytes for both `o` and `final_state`. We also
cover the contiguity-assert contract and the head_first + o= rejection
that kv supports (vk has no head_first kwarg).
"""

from __future__ import annotations

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()


# Evict any conftest stubs of atom.* so the real atom.model_ops imports work
# (see test_chunk_gated_delta_rule_vk.py for the same workaround).
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.fla_ops.chunk import (  # noqa: E402
    chunk_gated_delta_rule,
)

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="No GPU available"),
]


QWEN3_NEXT_H = 16
QWEN3_NEXT_K = 128
QWEN3_NEXT_V = 128


def _make_inputs(*, T, H, K, V, device="cuda", dtype=torch.bfloat16, seed=0):
    rng = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, T, H, K, dtype=dtype, device=device, generator=rng)
    k = torch.randn(1, T, H, K, dtype=dtype, device=device, generator=rng)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device, generator=rng)
    g = -torch.rand(1, T, H, dtype=torch.float32, device=device, generator=rng)
    beta = torch.rand(1, T, H, dtype=dtype, device=device, generator=rng).sigmoid()
    cu = torch.tensor([0, T], dtype=torch.int32, device=device)
    return q, k, v, g, beta, cu


# ============================================================================
# Inplace o= contract
# ============================================================================


def test_inplace_o_writes_into_caller_buffer():
    """o=preallocated must (a) return the same storage as the caller's
    buffer and (b) produce bit-exact output vs the alloc-internally call."""
    q, k, v, g, beta, cu = _make_inputs(
        T=128, H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=11
    )
    # Baseline: kernel allocates o internally.
    o_alloc, s_alloc = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # Inplace path: caller provides matching contiguous buffer.
    o_buf = torch.empty_like(v)
    o_buf_ptr = o_buf.data_ptr()
    o_inplace, s_inplace = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        o=o_buf,
    )
    assert o_inplace.data_ptr() == o_buf_ptr, (
        f"inplace o broken: returned data_ptr={o_inplace.data_ptr():x} "
        f"!= caller buffer data_ptr={o_buf_ptr:x}"
    )
    if not torch.equal(o_inplace, o_alloc):
        diff = (o_inplace.float() - o_alloc.float()).abs()
        raise AssertionError(
            f"inplace o numerically differs from alloc-internally: "
            f"max abs diff = {diff.max().item():.6g}"
        )
    assert torch.equal(s_inplace, s_alloc), (
        "final_state changed depending on whether o was inplace — "
        "shouldn't happen: the recurrence runs identically."
    )


def test_inplace_o_rejects_non_contiguous():
    """A non-contiguous o would be silently cloned by input_guard's
    .contiguous() call, defeating the inplace contract. Must assert before
    .apply() to catch it loudly."""
    q, k, v, g, beta, cu = _make_inputs(
        T=128, H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=12
    )
    backing = torch.empty(
        v.shape[0],
        v.shape[1],
        v.shape[2],
        v.shape[3] * 2,
        dtype=v.dtype,
        device=v.device,
    )
    non_contig_o = backing[..., : v.shape[3]]
    assert not non_contig_o.is_contiguous()
    with pytest.raises(AssertionError, match="contiguous"):
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            o=non_contig_o,
        )


def test_inplace_o_rejects_head_first():
    """head_first=True + o= must raise NotImplementedError because the
    trailing rearrange would alias the caller's storage as non-contiguous,
    silently breaking the inplace contract."""
    q, k, v, g, beta, cu = _make_inputs(
        T=128, H=QWEN3_NEXT_H, K=QWEN3_NEXT_K, V=QWEN3_NEXT_V, seed=13
    )
    o_buf = torch.empty_like(v)
    # head_first=True flips q/k/v to [B, H, T, ...]; we don't actually
    # reshape the inputs because the function should reject at the public
    # entry point before touching them.
    with pytest.raises(NotImplementedError, match="head_first"):
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=None,  # head_first requires cu_seqlens=None per upstream
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            head_first=True,
            o=o_buf,
        )
