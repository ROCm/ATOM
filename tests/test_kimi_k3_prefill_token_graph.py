# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 prefill token graphs: when the model records them, when the dense
MLA split is taken, and that a padded bucket's MLA and KDA splits write the rows
the eager forward computes, the same KV cache, and zeros past them. Kernels are
replaced by references."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from itertools import pairwise
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch._dynamo  # before Triton may be faked below
from aiter_stub import atom_without_aiter, stubbed_aiter
from conftest import gpu_stack_import
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


def _mla_impl(cls=None, **over):
    """An `MLAAttention` (``cls``, default this module's) with the fields the
    split's predicates read."""
    impl = object.__new__(cls or mla.MLAAttention)
    fields = {
        "split_prefill_recording": True,
        "is_sparse_mla": False,
        "o_proj": nn.Identity(),
    }
    impl.__dict__.update({**fields, **over})
    return impl


@pytest.fixture
def step(monkeypatch):
    """The forward context a test runs under; no PCP."""
    ctx = SimpleNamespace(context=None, tr_graph=None, in_hipgraph=False)
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    monkeypatch.setattr(mla, "pcp_is_enabled", lambda: False)
    return ctx


@contextlib.contextmanager
def _einops():
    """The Kimi builder imports the GDN mixer, which imports einops; the CPU
    runner has none, and nothing here calls it."""
    if importlib.util.find_spec("einops") is not None:
        yield
        return
    einops = ModuleType("einops")
    einops.__getattr__ = lambda name: MagicMock(name=f"einops.{name}")
    sys.modules["einops"] = einops
    try:
        yield
    finally:
        sys.modules.pop("einops", None)


def test_kimi_records_prefill_token_graphs_without_dp_while_its_layers_split(
    monkeypatch,
):
    with _einops(), atom_without_aiter() as load:
        kimi = load("atom.model_ops.attentions.kimi_mla_gdn_attn")
        # The builder's attention_mla: without AITER, a stand-in import of its
        # own, not this module's `mla`.
        attn = load(kimi.MLAAttention.__module__)
    monkeypatch.setattr(attn, "pcp_is_enabled", lambda: False)
    target_layers = 4

    def supports(dp, *layers):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=dp),
            hf_config=SimpleNamespace(num_hidden_layers=target_layers),
            compilation_config=SimpleNamespace(
                static_forward_context={f"l{i}": m for i, m in enumerate(layers)}
            ),
        )
        builder = SimpleNamespace(model_runner=SimpleNamespace(config=config))
        return kimi._KimiMLAGDNCommon.supports_token_graph.fget(builder)

    def mla_layer(layer_num=0, **over):
        impl = _mla_impl(kimi.MLAAttention, layer_num=layer_num, **over)
        return SimpleNamespace(impl=impl)

    kda_layer = SimpleNamespace()
    assert supports(1, mla_layer(), kda_layer)
    assert not supports(2, mla_layer(), kda_layer)
    assert not supports(8, mla_layer(), kda_layer)
    # A recorded layer whose split stands down would run as one region.
    assert not supports(1, mla_layer(), mla_layer(is_sparse_mla=True))
    # Kimi-K3.1: every MLA layer opted in, and every one is sparse.
    assert not supports(1, mla_layer(is_sparse_mla=True), kda_layer)
    # GLM-5.3-Flash shares this builder and opts no layer in: its MLA layers
    # are all sparse and never set `split_prefill_recording`; its linear
    # attention and its MTP layer (numbered past the target's) beside them.
    glm_mla = mla_layer(split_prefill_recording=False, is_sparse_mla=True)
    glm_mtp = mla_layer(
        target_layers, split_prefill_recording=False, is_sparse_mla=True
    )
    assert not supports(1, glm_mla, kda_layer, glm_mtp)
    # A target layer the model did not opt in counts too.
    assert not supports(1, mla_layer(), mla_layer(split_prefill_recording=False))
    # A drafter's layer, numbered past the target's, does not: the recording
    # does not run it.
    drafter = mla_layer(
        target_layers, split_prefill_recording=False, o_proj=nn.Linear(2, 2)
    )
    assert supports(1, mla_layer(), drafter)
    # Nothing opted in, nothing recorded: a drafter alone, or no MLA layer.
    assert not supports(1, drafter, kda_layer)
    assert not supports(1, kda_layer)
    monkeypatch.setattr(attn, "pcp_is_enabled", lambda: True)
    monkeypatch.setattr(attn, "get_pcp_world_size", lambda: 2)
    assert not supports(1, mla_layer(), kda_layer)
    monkeypatch.setattr(attn, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(attn, "use_triton_gemm", lambda: True)
    assert not supports(1, mla_layer(), kda_layer)


def test_the_capture_step_is_a_request_less_dummy_prefill_of_the_bucket():
    with _einops(), atom_without_aiter() as load:
        kimi = load("atom.model_ops.attentions.kimi_mla_gdn_attn")
    positions = torch.full((16,), 7)
    runner = SimpleNamespace(
        config=SimpleNamespace(max_num_batched_tokens=8),
        forward_vars={"positions": SimpleNamespace(gpu=positions)},
    )
    build = kimi._KimiMLAGDNCommon.build_for_token_graph_capture
    builder = SimpleNamespace(model_runner=runner)

    attn_metadata, context = build(builder, 8)
    assert context.is_prefill and context.is_dummy_run
    assert (context.scheduled_bs, context.scheduled_tokens) == (1, 8)
    assert (context.running_bs, context.running_tokens) == (1, 8)
    assert attn_metadata.max_seqlen_q == attn_metadata.max_seqlen_k == 8
    # The runner's own positions, zeroed over the bucket and left alone past it.
    assert context.positions.data_ptr() == positions.data_ptr()
    assert torch.count_nonzero(positions[:8]) == 0
    assert bool((positions[8:] == 7).all())
    for tokens in (0, 9):
        with pytest.raises(ValueError, match="exceeds max_num_batched_tokens"):
            build(builder, tokens)


@pytest.mark.parametrize(
    "use_mla, tr_graph, taken",
    [
        (True, "record", "split"),
        (True, None, "region"),  # an eager prefill
        (False, "record", "region"),
    ],
)
def test_the_attention_op_takes_the_split_its_impl_asks_for(
    monkeypatch, step, use_mla, tr_graph, taken
):
    step.tr_graph = tr_graph
    step.context = SimpleNamespace(is_prefill=True, is_dummy_run=False)
    monkeypatch.setattr(
        mla.MLAAttention, "forward_prefill_split", lambda self, *a: ("split", a)
    )
    layers = {"layer": SimpleNamespace(impl=_mla_impl())}
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
    assert _mla_impl().records_prefill_split() is split


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
    assert _mla_impl().records_prefill_split()
    assert not _mla_impl(**over).records_prefill_split()


def test_the_split_is_off_under_pcp(step, monkeypatch):
    step.tr_graph = "record"
    step.context = SimpleNamespace(is_prefill=True, is_dummy_run=True)
    monkeypatch.setattr(mla, "pcp_is_enabled", lambda: True)
    monkeypatch.setattr(mla, "get_pcp_world_size", lambda: 2)
    assert not _mla_impl().records_prefill_split()


def test_the_split_is_off_under_triton_gemm(step, monkeypatch):
    # The eager prefill then takes the fused GEMM + split + cat for kv_b_proj.
    step.tr_graph = "record"
    step.context = SimpleNamespace(is_prefill=True, is_dummy_run=True)
    monkeypatch.setattr(mla, "use_triton_gemm", lambda: True)
    assert not _mla_impl().records_prefill_split()


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


def _bind(impl, *names):
    for name in names:
        setattr(impl, name, MethodType(getattr(mla.MLAAttention, name), impl))
    return impl


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
        use_asm_fp8_prefill_attn=False,
        use_flydsl_fp8_prefill_attn=False,
        _fp8_mha_ok=None,
        _fp8_mha_backend=None,
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
        "_write_kv",
        "_new_token_kv",
        "_quantize_new_tokens",
        "_chunked_prefill_attention",
        "_forward_prefill_cached_chunked",
        "_forward_prefill_cached_single_pass",
        "_forward_prefill_mha",
        "_check_fp8_prefill_mha",
        "_dcp_take_prefetched",
        "_dcp_prefetch_ready",
        "_dcp_start_context_gathers",
        "_dcp_prefetch_next",
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
    impl._fp8_mha_ok = True
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


# ---- the KDA split against _forward_impl, with a reference recurrence -------

KDA_HEADS, KDA_DIM, HIDDEN = 2, 4, 16


def _kda_layer(model, calls):
    """A KDA layer of linear projections whose recurrence (the one kernel,
    `_prefill_recurrence`) is a causal reference that logs its inputs."""
    torch.manual_seed(3)
    width = KDA_HEADS * KDA_DIM
    w_in = torch.randn(4 * width + KDA_HEADS + KDA_DIM, HIDDEN)
    w_f, w_o = torch.randn(width, KDA_DIM), torch.randn(HIDDEN, width)

    def recurrence(
        mixed_qkv, gate, beta, kda_metadata, conv_state, ssm_state, out=None
    ):
        calls.append((mixed_qkv.clone(), gate.clone(), beta.clone()))
        x = mixed_qkv[:, :width].unflatten(-1, (KDA_HEADS, KDA_DIM))
        return torch.cumsum(x * gate[0].sigmoid() * beta[0, ..., None], dim=0)

    layer = object.__new__(model.KimiKDAAttention)
    layer.__dict__.update(
        layer_name="kda",
        layer_num=0,
        head_dim=KDA_DIM,
        num_local_heads=KDA_HEADS,
        local_proj_size=width,
        fuse_f_a_quant=False,
        quant_f_a_strided=False,
        conv_weight=None,
        in_proj=lambda x, x_scale=None: nn.functional.linear(x, w_in),
        f_b_proj=lambda x: nn.functional.linear(x, w_f),
        o_norm=lambda out, gate, is_prefill: out * gate.sigmoid(),
        o_proj=lambda x: nn.functional.linear(x, w_o),
        _prefill_recurrence=recurrence,
    )
    return layer


def _kda_step(monkeypatch, model, layer, rows, metadata, dummy=False):
    """A prefill step of ``rows`` real tokens with KDA ``metadata`` (None:
    none); ``dummy`` is the recording's dummy step, which has none."""
    width = KDA_HEADS * KDA_DIM
    attn_metadata = SimpleNamespace()
    if metadata is not None:
        attn_metadata.kda_metadata = metadata
    ctx = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=True, scheduled_tokens=rows, is_dummy_run=dummy
        ),
        attn_metadata=attn_metadata,
        kv_cache_data={
            "layer_0": SimpleNamespace(
                k_cache=torch.zeros(4, 3 * width, 4),
                v_cache=torch.zeros(4, KDA_HEADS, KDA_DIM, KDA_DIM),
            )
        },
        tr_graph=None,
        in_hipgraph=False,
        main_stream=None,
    )
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    compilation = SimpleNamespace(static_forward_context={"kda": layer})
    monkeypatch.setattr(
        model,
        "get_current_atom_config",
        lambda: SimpleNamespace(compilation_config=compilation),
    )


def _kda_metadata(rows):
    """Prompts of 3 and 2 tokens."""
    return SimpleNamespace(
        num_actual_tokens=rows,
        num_prefills=2,
        num_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([0, 1]),
        non_spec_state_indices_in_tensor=None,
        non_spec_query_start_loc=torch.tensor([0, 3, 5]),
        has_initial_state=torch.tensor([True, False]),
    )


@pytest.mark.parametrize("bucket", [5, 8])
def test_a_padded_kda_split_writes_the_rows_forward_impl_computes(monkeypatch, bucket):
    model = gpu_stack_import("atom.models.kimi_k3.model")
    rows = 5
    torch.manual_seed(4)
    hidden = torch.randn(bucket, HIDDEN)

    eager_calls, split_calls = [], []
    layer = _kda_layer(model, eager_calls)
    _kda_step(monkeypatch, model, layer, rows, _kda_metadata(rows))
    eager = layer._forward_impl(hidden[:rows].clone(), None)

    layer = _kda_layer(model, split_calls)
    _kda_step(monkeypatch, model, layer, rows, _kda_metadata(rows))
    hidden[rows:] = 1e4
    split = layer._forward_prefill_split(hidden, None)

    assert split.shape == (bucket, HIDDEN)
    assert eager.abs().sum() > 0
    torch.testing.assert_close(split[:rows], eager, rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(split[rows:]) == 0
    # One recurrence, on the eager step's rows.
    assert len(split_calls) == len(eager_calls) == 1
    for got, want in zip(split_calls[0], eager_calls[0]):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_the_kda_recordings_dummy_step_leaves_the_output_alone(monkeypatch):
    model = gpu_stack_import("atom.models.kimi_k3.model")
    calls = []
    layer = _kda_layer(model, calls)
    _kda_step(monkeypatch, model, layer, 8, None, dummy=True)
    split = layer._forward_prefill_split(torch.randn(8, HIDDEN), None)
    assert split.shape == (8, HIDDEN)
    assert torch.count_nonzero(split) == 0
    assert not calls


def test_a_live_kda_replay_without_metadata_fails_loudly(monkeypatch):
    """Only the recording's dummy step may skip the recurrence; a live step
    with no KDA metadata must not hand back zeros as its output."""
    model = gpu_stack_import("atom.models.kimi_k3.model")
    calls = []
    layer = _kda_layer(model, calls)
    _kda_step(monkeypatch, model, layer, 5, None)
    with pytest.raises(AssertionError, match="no KDA metadata"):
        layer._forward_prefill_split(torch.randn(8, HIDDEN), None)
    assert not calls
