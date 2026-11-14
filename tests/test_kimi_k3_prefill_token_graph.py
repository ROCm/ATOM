# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 prefill token graphs: when the dense MLA split is taken, and that a
padded bucket's split writes the rows `forward_impl` computes, the same KV
cache, and zeros past them. Kernels are replaced by references."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from itertools import pairwise
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch._dynamo  # before Triton may be faked below
from aiter_stub import stubbed_aiter
from torch import nn

# Without AITER (the CPU runner) attention_mla's AITER imports, and Triton where
# it is missing, are faked; the kernels these tests reach are references here.
_FAKE_AITER = (
    "aiter.jit",
    "aiter.jit.utils",
    "aiter.jit.utils.torch_guard",
    "aiter.mla",
    "aiter.ops",
    "aiter.ops.gated_rmsnorm_fp8_group_quant",
    "aiter.ops.quant",
    "aiter.ops.shuffle",
    "aiter.ops.triton",
    "aiter.ops.triton.attention",
    "aiter.ops.triton.attention.mla",
    "aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
    "aiter.ops.triton.fused_add_rmsnorm_pad",
    "aiter.ops.triton.fusions",
    "aiter.ops.triton.fusions.fused_kv_cache",
    "aiter.ops.triton.gemm",
    "aiter.ops.triton.gemm.batched",
    "aiter.ops.triton.gemm.batched.batched_gemm_bf16",
    "aiter.ops.triton.gather_kv_b_proj",
    "aiter.ops.triton.gluon",
    "aiter.ops.triton.gluon.pa_decode_gluon",
    "aiter.ops.triton.kv_cache",
    "aiter.ops.triton.quant",
    "aiter.ops.triton.quant.fused_mxfp8_quant",
    "aiter.tuned_gemm",
    "aiter.utility",
    "aiter.utility.fp4_utils",
)


def _imports():
    if importlib.util.find_spec("aiter") is not None:
        return contextlib.nullcontext()
    no_triton = importlib.util.find_spec("triton") is None
    return stubbed_aiter(*_FAKE_AITER, *(("triton", "triton.language") * no_triton))


with _imports():
    from atom.model_ops import attention_mla as mla
    from atom.model_ops import base_attention as base
    from atom.utils import forward_context

HEADS, NOPE, ROPE, V_DIM, LORA = 2, 8, 4, 8, 6


def test_kimi_records_prefill_token_graphs_only_without_dp():
    pytest.importorskip("einops", reason="the Kimi builder imports the GDN mixer")
    with _imports():
        from atom.model_ops.attentions.kimi_mla_gdn_attn import _KimiMLAGDNCommon

    def supports(runner):
        builder = SimpleNamespace(model_runner=runner)
        return _KimiMLAGDNCommon.supports_token_graph.fget(builder)

    def runner(dp):
        parallel = SimpleNamespace(data_parallel_size=dp)
        return SimpleNamespace(config=SimpleNamespace(parallel_config=parallel))

    assert supports(runner(1))
    assert not supports(runner(2))
    assert not supports(runner(8))
    assert not supports(SimpleNamespace())


@pytest.mark.parametrize(
    "use_mla, impl, taken",
    [
        (True, SimpleNamespace(records_prefill_split=lambda: True), "split"),
        (True, SimpleNamespace(records_prefill_split=lambda: False), "region"),
        (True, SimpleNamespace(), "region"),  # an impl without the split
        (False, SimpleNamespace(records_prefill_split=lambda: True), "region"),
    ],
)
def test_the_attention_op_takes_the_split_its_impl_asks_for(
    monkeypatch, use_mla, impl, taken
):
    impl.forward_prefill_split = lambda *args: ("split", args)
    layers = {"layer": SimpleNamespace(impl=impl)}
    compilation = SimpleNamespace(static_forward_context=layers)
    monkeypatch.setattr(
        base,
        "get_current_atom_config",
        lambda: SimpleNamespace(compilation_config=compilation),
    )
    monkeypatch.setattr(base, "_attention_region", lambda *args: ("region", args))
    q, k, v, positions = (torch.zeros(4, 2) for _ in range(4))
    kind, args = base.unified_attention_with_output_base(
        q, None, k, v, positions, "layer", use_mla, None
    )
    assert kind == taken
    expected = (q, None, k, v, positions, "layer")
    if taken == "region":
        expected += (use_mla, None)
    assert len(args) == len(expected)
    assert all(a is e for a, e in zip(args, expected))


def _bind(impl, *names):
    for name in names:
        setattr(impl, name, MethodType(getattr(mla.MLAAttention, name), impl))
    return impl


def _split_impl(**over):
    impl = SimpleNamespace(
        split_prefill_recording=True, is_sparse_mla=False, o_proj=nn.Identity()
    )
    impl.__dict__.update(over)
    return _bind(impl, "records_prefill_split", "_prefill_split_supported")


@pytest.fixture
def step(monkeypatch):
    """The forward context a test runs under; no PCP."""
    ctx = SimpleNamespace(context=None, tr_graph=None, in_hipgraph=False)
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    monkeypatch.setattr(mla, "pcp_is_enabled", lambda: False)
    return ctx


@pytest.mark.parametrize(
    "tr_graph, in_hipgraph, prefill, dummy, split",
    [
        ("record", True, True, True, True),  # the recording
        (None, True, True, True, True),  # the warmup forward before it
        ("replay", False, True, False, False),  # a replayed step
        (None, False, True, False, False),  # an eager prefill
        (None, False, True, True, False),  # a dummy prefill outside capture
        ("record", True, False, True, False),  # a decode recording
        (None, True, False, False, False),  # decode graph capture
    ],
)
def test_the_split_runs_on_a_prefill_token_graph_capture_alone(
    step, tr_graph, in_hipgraph, prefill, dummy, split
):
    step.tr_graph, step.in_hipgraph = tr_graph, in_hipgraph
    step.context = SimpleNamespace(is_prefill=prefill, is_dummy_run=dummy)
    assert _split_impl().records_prefill_split() is split


@pytest.mark.parametrize(
    "over",
    [
        {"split_prefill_recording": False},  # the model did not opt in
        {"is_sparse_mla": True},  # prefill is the MLA kernel over the cache
        {"o_proj": nn.Linear(2, 2)},  # the impl projects its own output
    ],
)
def test_the_split_needs_the_opt_in_and_a_dense_layer_without_o_proj(step, over):
    step.tr_graph = "record"
    step.context = SimpleNamespace(is_prefill=True, is_dummy_run=True)
    assert _split_impl().records_prefill_split()
    assert not _split_impl(**over).records_prefill_split()


def test_the_split_is_off_under_pcp(step, monkeypatch):
    step.tr_graph = "record"
    step.context = SimpleNamespace(is_prefill=True, is_dummy_run=True)
    monkeypatch.setattr(mla, "pcp_is_enabled", lambda: True)
    monkeypatch.setattr(mla, "get_pcp_world_size", lambda: 2)
    assert not _split_impl().records_prefill_split()


def test_the_split_is_off_under_triton_gemm(step, monkeypatch):
    # The eager prefill then takes the fused GEMM + split + cat for kv_b_proj.
    step.tr_graph = "record"
    step.context = SimpleNamespace(is_prefill=True, is_dummy_run=True)
    monkeypatch.setattr(mla, "use_triton_gemm", lambda: True)
    assert not _split_impl().records_prefill_split()


# ---- the split against forward_impl, with reference kernels ----------------


def _varlen(*, q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, **kw):
    """Varlen attention, bottom-right causal, LSE ``[heads, tokens]``."""
    out = q.new_zeros(q.shape[0], q.shape[1], v.shape[-1])
    lse = torch.full((q.shape[1], q.shape[0]), float("-inf"))
    cu_q, cu_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for (q0, q1), (k0, k1) in zip(pairwise(cu_q), pairwise(cu_k)):
        if q1 == q0 or k1 == k0:
            continue
        s = torch.einsum("qhd,khd->hqk", q[q0:q1], k[k0:k1]) * softmax_scale
        if causal:
            # Query i of the last q1 - q0 positions sees keys up to its own.
            shift = (k1 - k0) - (q1 - q0)
            late = torch.arange(k1 - k0)[None] > torch.arange(q1 - q0)[:, None] + shift
            s = s.masked_fill(late, float("-inf"))
        lse[:, q0:q1] = torch.logsumexp(s, -1)
        out[q0:q1] = torch.einsum("hqk,khd->qhd", s.softmax(-1), v[k0:k1])
    return (out, lse) if kw.get("return_lse") else out


def _merge(*, output, prefix_output, prefix_lse, suffix_output, suffix_lse, **kw):
    p, s = prefix_lse.T.unsqueeze(-1), suffix_lse.T.unsqueeze(-1)
    top = torch.maximum(p, s)
    wp, ws = (p - top).exp(), (s - top).exp()
    output.copy_((prefix_output * wp + suffix_output * ws) / (wp + ws))
    if kw.get("output_lse") is not None:
        kw["output_lse"].copy_((top + (wp + ws).log()).squeeze(-1).T)


def _cache_write(k_nope, k_rope, kv_cache, slot_mapping, **_):
    kv_cache[slot_mapping.long(), 0] = torch.cat((k_nope, k_rope), dim=-1)


@pytest.fixture
def kernels(monkeypatch):
    # The chunked path imports merge_attn_states from its module at call time.
    merge = SimpleNamespace(merge_attn_states=_merge)
    monkeypatch.setitem(
        sys.modules, "atom.model_ops.attentions.triton_merge_attn_states", merge
    )
    monkeypatch.setattr(mla, "flash_attn_varlen_func", _varlen)
    monkeypatch.setattr(mla, "concat_and_cache_mla", _cache_write)
    monkeypatch.setattr(mla, "use_triton_gemm", lambda: False)
    monkeypatch.setattr(mla, "pcp_is_enabled", lambda: False)


def _dense_impl(prefix_k, prefix_v):
    torch.manual_seed(1)
    weight = torch.randn(HEADS * (NOPE + V_DIM), LORA)

    def gather(cache, indptr, indices, cu_seqlens_k, k_out, v_out, *a, **kw):
        k_out.copy_(prefix_k[: k_out.shape[0]])
        v_out.copy_(prefix_v[: v_out.shape[0]])

    impl = SimpleNamespace(
        num_heads=HEADS,
        num_kv_heads=1,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        qk_head_dim=NOPE + ROPE,
        v_head_dim=V_DIM,
        kv_lora_rank=LORA,
        scale=(NOPE + ROPE) ** -0.5,
        dtype=torch.float32,
        layer_num=0,
        qrep_enabled=False,
        q_proj=lambda x, x_scale=None: x,
        rotary_emb=lambda positions, q_pe, k_pe: None,
        kv_b_proj=lambda x: nn.functional.linear(x, weight),
        o_proj=nn.Identity(),
        _k_scale=None,
        kv_cache_dtype="auto",
        kv_shuffled=False,
        use_seg_mla=False,
        is_sparse_mla=False,
        rope_is_zero_pad=False,
        use_flydsl_fp8_prefill_attn=False,
        _flydsl_fp8_mha_ok=None,
        split_prefill_recording=True,
        _gather_cached_kv_b_proj=gather,
    )
    return _bind(
        impl,
        "forward_impl",
        "forward_prefill_split",
        "_prefill_attend",
        "_prefill_q",
        "_write_prefill_kv",
        "_new_token_kv",
        "_quantize_new_tokens",
        "_chunked_prefill_attention",
        "_forward_prefill_cached_chunked",
        "_forward_prefill_cached_single_pass",
        "_forward_prefill_mha",
        "_check_flydsl_fp8_mha",
        "_prepare_prefill_k",
        "_drop_rope_pad",
        "_flash_attn_prefill",
    )


def _metadata(path):
    """Prompts of 3 and 2 new tokens behind 4 and 1 cached ones."""
    cu_new = torch.tensor([0, 3, 5], dtype=torch.int32)
    cu_full = torch.tensor([0, 7, 10], dtype=torch.int32)
    cu_cached = torch.tensor([0, 4, 5], dtype=torch.int32)
    has_cached = path != "new_only"
    chunks = SimpleNamespace(
        num_chunks=1,
        is_dcp=False,
        k_workspace=torch.empty(5, HEADS, NOPE + ROPE),
        v_workspace=torch.empty(5, HEADS, V_DIM),
        total_tokens=[5],
        kv_indptr=[cu_cached],
        kv_indices=[None],
        cu_seqlens_k=[cu_cached],
        max_seqlen_k=[4],
        shuffle_kv_block_indptr=None,
        shuffle_kv_block_indices=None,
    )
    return SimpleNamespace(
        has_cached=has_cached,
        mla_chunk_meta=chunks if path == "chunked" else None,
        run_sparse=False,
        slot_mapping=torch.tensor([20, 21, 22, 30, 31]),
        cu_seqlens_q=cu_new,
        cu_seqlens_k=cu_full if has_cached else cu_new,
        max_seqlen_q=3,
        max_seqlen_k=7 if has_cached else 3,
        min_seqlen_q=2,
        dropout_p=0.0,
        total_kv=10,
        kv_indptr=cu_full,
        kv_indices=None,
    )


def _run(monkeypatch, impl, metadata, rows, *, dummy=False):
    """A step of ``rows`` real tokens; returns its KV cache."""
    kv_cache = torch.zeros(40, 1, LORA + ROPE)
    ctx = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=True,
            is_dummy_run=dummy,
            scheduled_tokens=rows,
            running_tokens=rows,
            scheduled_tokens_gpu=None,
        ),
        attn_metadata=metadata,
        kv_cache_data={"layer_0": SimpleNamespace(k_cache=kv_cache)},
        tr_graph=None,
        in_hipgraph=False,
        main_stream=None,
    )
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    layers = {"layer": SimpleNamespace(impl=impl)}
    compilation = SimpleNamespace(static_forward_context=layers)
    monkeypatch.setattr(
        mla,
        "get_current_atom_config",
        lambda: SimpleNamespace(compilation_config=compilation),
    )
    return kv_cache


@pytest.mark.parametrize("path", ["new_only", "chunked", "single_pass"])
@pytest.mark.parametrize("bucket", [5, 8])
def test_a_padded_split_writes_the_rows_forward_impl_computes(
    monkeypatch, kernels, path, bucket
):
    rows = 5
    torch.manual_seed(2)
    q = torch.randn(bucket, HEADS * (NOPE + ROPE))
    k_nope = torch.randn(bucket, LORA)
    k_rope = torch.randn(bucket, 1, ROPE)
    positions = torch.arange(bucket)
    prefix_k = torch.randn(10, HEADS, NOPE + ROPE)
    prefix_v = torch.randn(10, HEADS, V_DIM)
    metadata = _metadata(path)

    impl = _dense_impl(prefix_k, prefix_v)
    eager_cache = _run(monkeypatch, impl, metadata, rows)
    eager = impl.forward_impl(
        q[:rows], k_nope[:rows], k_rope[:rows].clone(), positions[:rows]
    )

    impl = _dense_impl(prefix_k, prefix_v)
    split_cache = _run(monkeypatch, impl, metadata, rows)
    q[rows:], k_nope[rows:], k_rope[rows:] = 1e4, 1e4, 1e4
    split = impl.forward_prefill_split(q, None, k_nope, k_rope, positions, "layer")

    assert split.shape == (bucket, HEADS * V_DIM)
    assert eager.abs().sum() > 0
    torch.testing.assert_close(split[:rows], eager, rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(split[rows:]) == 0
    torch.testing.assert_close(split_cache, eager_cache, rtol=0, atol=0)


def test_the_recordings_dummy_step_leaves_the_output_and_cache_alone(
    monkeypatch, kernels
):
    impl = _dense_impl(
        torch.zeros(10, HEADS, NOPE + ROPE), torch.zeros(10, HEADS, V_DIM)
    )
    kv_cache = _run(monkeypatch, impl, _metadata("chunked"), 8, dummy=True)
    q = torch.randn(8, HEADS * (NOPE + ROPE))
    split = impl.forward_prefill_split(
        q, None, torch.randn(8, LORA), torch.randn(8, 1, ROPE), torch.arange(8), "layer"
    )
    assert split.shape == (8, HEADS * V_DIM)
    assert torch.count_nonzero(split) == 0
    assert torch.count_nonzero(kv_cache) == 0


def test_an_fp8_split_needs_the_steps_row_count(monkeypatch, kernels):
    impl = _dense_impl(
        torch.zeros(10, HEADS, NOPE + ROPE), torch.zeros(10, HEADS, V_DIM)
    )
    impl._flydsl_fp8_mha_ok = True
    _run(monkeypatch, impl, _metadata("new_only"), 5)  # no scheduled_tokens_gpu
    with pytest.raises(AssertionError, match="row count"):
        impl.forward_prefill_split(
            torch.randn(8, HEADS * (NOPE + ROPE)),
            None,
            torch.randn(8, LORA),
            torch.randn(8, 1, ROPE),
            torch.arange(8),
            "layer",
        )
