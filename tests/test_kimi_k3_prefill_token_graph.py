# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 prefill token graphs: when the model records them, when the dense
MLA split is taken, and that a padded bucket's MLA and KDA splits, which run
every row of the bucket, make the kernel calls the eager forward makes and give
its rows, KV cache and state, whatever the pad rows hold. Kernels are replaced
by references."""

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
from aiter_stub import atom_without_aiter
from torch import nn

# Imported for real, so every module the stand-ins load below, the Kimi model's
# included, reads the forward context a test sets.
from atom.utils import forward_context

# Without AITER (the CPU runner) attention_mla's AITER and Triton imports are
# stand-ins; the kernels these tests reach are references here.
with atom_without_aiter() as _load:
    mla = _load("atom.model_ops.attention_mla")
    base = _load("atom.model_ops.base_attention")

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
    """The Kimi model and builder import einops, which the CPU runner does not
    have; only the KDA layer's reshapes are called here (`_rearrange`)."""
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


def _kimi_model():
    """`atom.models.kimi_k3.model`, against AITER stand-ins on the CPU runner."""
    with _einops(), atom_without_aiter() as load:
        return load("atom.models.kimi_k3.model")


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
    "use_mla, tr_graph, in_hipgraph, taken",
    [
        (True, "record", True, "split"),
        (False, "record", True, "region"),
        (True, "replay", False, "region"),  # a forward under a replaying step
        (True, None, False, "call"),  # an eager prefill: no graph records it
        (False, None, False, "call"),
    ],
)
def test_the_attention_op_takes_the_split_its_impl_asks_for(
    monkeypatch, step, use_mla, tr_graph, in_hipgraph, taken
):
    step.tr_graph, step.in_hipgraph = tr_graph, in_hipgraph
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
    monkeypatch.setattr(base, "_attention", lambda *args: ("call", args))
    q, k, v, positions = (torch.zeros(4, 2) for _ in range(4))
    kind, args = base.unified_attention_with_output_base(
        q, None, k, v, positions, "layer", use_mla, None
    )
    assert kind == taken
    expected = (q, None, k, v, positions, "layer")
    if taken != "split":
        expected += (use_mla, None)
    assert len(args) == len(expected)
    assert all(a is e for a, e in zip(args, expected))


# (tr_graph, in_hipgraph, prefill, dummy, split): a forward, and whether a
# layer that records a prefill split takes it there.
_FORWARDS = [
    ("record", True, True, True, True),  # the recording
    (None, True, True, True, True),  # the warmup forward before it
    ("replay", False, True, False, False),  # a replayed step
    (None, False, True, False, False),  # an eager prefill
    (None, False, True, True, False),  # a dummy prefill outside capture
    ("record", True, False, True, False),  # a decode recording
    (None, True, False, False, False),  # decode graph capture
]


@pytest.mark.parametrize("tr_graph, in_hipgraph, prefill, dummy, split", _FORWARDS)
def test_the_split_runs_on_a_prefill_token_graph_capture_alone(
    step, tr_graph, in_hipgraph, prefill, dummy, split
):
    step.tr_graph, step.in_hipgraph = tr_graph, in_hipgraph
    step.context = SimpleNamespace(is_prefill=prefill, is_dummy_run=dummy)
    assert _mla_impl().records_prefill_split() is split


@pytest.mark.parametrize("tr_graph, in_hipgraph, prefill, dummy, split", _FORWARDS)
def test_a_kda_layer_splits_on_the_forwards_an_mla_layer_does(
    monkeypatch, step, tr_graph, in_hipgraph, prefill, dummy, split
):
    """Its warmup forward included, so the projections first run at the
    bucket's shape outside the capture. Any other forward a graph records
    runs the mixer as one region; one that none records calls it."""
    model = _kimi_model()
    layer = SimpleNamespace(_forward_prefill_split=lambda *args: "split")
    compilation = SimpleNamespace(static_forward_context={"kda": layer})
    monkeypatch.setattr(
        model,
        "get_current_atom_config",
        lambda: SimpleNamespace(compilation_config=compilation),
    )
    monkeypatch.setattr(model, "_kda", lambda *args: "call")
    monkeypatch.setattr(model, "_kda_mixer", lambda *args: "region")
    step.tr_graph, step.in_hipgraph = tr_graph, in_hipgraph
    step.context = SimpleNamespace(is_prefill=prefill, is_dummy_run=dummy)
    eager = tr_graph is None and not in_hipgraph
    taken = "split" if split else "call" if eager else "region"
    assert model.kda_attention_with_output(torch.zeros(2, 4), None, "kda") == taken


def test_kimi_records_its_prefill_buckets_apart_from_its_decode_ones(monkeypatch):
    """Under TR_HYBRID a decode bucket is recorded first, at the shapes a
    prefill bucket of its rows has. Shared, the prefill pass would count that
    recording, and a padded prefill would replay a decode forward, whose
    whole-mixer KDA region returns the step's rows rather than the bucket's
    (`token_graph(per_phase=True)`)."""
    model = _kimi_model()
    per_phase = []

    def run(owner, forward, args, kwargs, phased=False):
        per_phase.append(phased)

    monkeypatch.setitem(model.token_graph.__globals__, "_run_token_graph", run)
    rows = torch.zeros(4, dtype=torch.long)
    object.__new__(model.KimiLinearModel)(rows, rows)
    assert per_phase == [True]


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
    """Varlen attention, bottom-right causal, LSE ``[heads, tokens]``. A
    request without keys gets zeros and an LSE of -inf; a row past every
    request is left as the kernel found it, NaN here."""
    out = q.new_full((q.shape[0], q.shape[1], v.shape[-1]), float("nan"))
    lse = torch.full((q.shape[1], q.shape[0]), float("nan"))
    cu_q, cu_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for (q0, q1), (k0, k1) in zip(pairwise(cu_q), pairwise(cu_k)):
        out[q0:q1], lse[:, q0:q1] = 0, float("-inf")
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
    """Skips slot -1, as the kernels do: a pad row, or a token another rank
    holds."""
    rows = slot_mapping >= 0
    kv_cache[slot_mapping[rows].long(), 0] = torch.cat((k_nope, k_rope), dim=-1)[rows]


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
        "_write_kv",
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


def _metadata(path, bucket=5):
    """Prompts of 3 and 2 new tokens behind 4 and 1 cached ones, over
    ``bucket`` rows as the base builder lays out a padded step: KV slot -1
    past the 5 scheduled rows, and a padding request of no rows. A chunk's own
    `cu_seqlens_k` is unpadded (`_chunked_prefill_attention`)."""
    padding = int(bucket > 5)

    def cu(*offsets):
        return torch.tensor([*offsets, *offsets[-1:] * padding], dtype=torch.int32)

    cu_new, cu_full = cu(0, 3, 5), cu(0, 7, 10)
    cu_cached = torch.tensor([0, 4, 5], dtype=torch.int32)
    slots = torch.full((bucket,), -1)
    slots[:5] = torch.tensor([20, 21, 22, 30, 31])
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
        slot_mapping=slots,
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


def _run(monkeypatch, impl, metadata, rows, *, bucket=None, dummy=False, live=None):
    """A step of ``rows`` real tokens run at ``bucket`` rows (default: no
    padding), ``live`` its row count on the device; returns its KV cache."""
    kv_cache = torch.zeros(40, 1, LORA + ROPE)
    ctx = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=True,
            is_dummy_run=dummy,
            scheduled_tokens=rows,
            running_tokens=bucket or rows,
            scheduled_tokens_gpu=live,
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


def _split_inputs(bucket):
    """``(q, q_scale, k_nope, k_rope, positions, layer_name)`` of a bucket."""
    return (
        torch.randn(bucket, HEADS * (NOPE + ROPE)),
        None,
        torch.randn(bucket, LORA),
        torch.randn(bucket, 1, ROPE),
        torch.arange(bucket),
        "layer",
    )


@pytest.mark.parametrize("path", ["new_only", "chunked", "single_pass"])
@pytest.mark.parametrize("bucket", [5, 8])
def test_a_padded_split_attends_the_bucket_and_gives_forward_impls_rows(
    monkeypatch, kernels, path, bucket
):
    rows = 5
    torch.manual_seed(2)
    q, _, k_nope, k_rope, positions, _ = _split_inputs(bucket)
    prefix_k = torch.randn(10, HEADS, NOPE + ROPE)
    prefix_v = torch.randn(10, HEADS, V_DIM)

    impl = _dense_impl(prefix_k, prefix_v)
    eager_cache = _run(monkeypatch, impl, _metadata(path), rows)
    eager = impl.forward_impl(
        q[:rows], k_nope[:rows], k_rope[:rows].clone(), positions[:rows]
    )

    impl = _dense_impl(prefix_k, prefix_v)
    split_cache = _run(monkeypatch, impl, _metadata(path, bucket), rows, bucket=bucket)
    attended = []

    def varlen(**kw):
        attended.append(kw["q"].shape[0])
        return _varlen(**kw)

    monkeypatch.setattr(mla, "flash_attn_varlen_func", varlen)
    # Whatever the last step left in the pad rows reaches no real row.
    q[rows:], k_nope[rows:], k_rope[rows:] = (float("nan"),) * 3
    split = impl.forward_prefill_split(q, None, k_nope, k_rope, positions, "layer")

    assert split.shape == (bucket, HEADS * V_DIM)
    assert eager.abs().sum() > 0
    torch.testing.assert_close(split[:rows], eager, rtol=1e-5, atol=1e-5)
    # Every attention ran the whole bucket; the KV write skipped its pad rows.
    assert attended and set(attended) == {bucket}
    torch.testing.assert_close(split_cache, eager_cache, rtol=0, atol=0)


def test_the_recordings_dummy_step_attends_nothing_and_writes_no_kv(
    monkeypatch, kernels
):
    impl = _dense_impl(
        torch.zeros(10, HEADS, NOPE + ROPE), torch.zeros(10, HEADS, V_DIM)
    )
    kv_cache = _run(monkeypatch, impl, _metadata("chunked", 8), 8, dummy=True)
    attended = []
    monkeypatch.setattr(mla, "flash_attn_varlen_func", attended.append)
    split = impl.forward_prefill_split(*_split_inputs(8))
    assert split.shape == (8, HEADS * V_DIM)
    # Zeros: the warmup forward runs the rest of the layer on them.
    assert torch.count_nonzero(split) == 0
    assert not attended
    assert torch.count_nonzero(kv_cache) == 0


def test_an_fp8_split_needs_the_steps_row_count(monkeypatch, kernels):
    impl = _dense_impl(
        torch.zeros(10, HEADS, NOPE + ROPE), torch.zeros(10, HEADS, V_DIM)
    )
    impl._flydsl_fp8_mha_ok = True
    _run(monkeypatch, impl, _metadata("new_only", 8), 5, bucket=8)  # no live count
    with pytest.raises(AssertionError, match="row count"):
        impl.forward_prefill_split(*_split_inputs(8))


def test_an_fp8_split_quantizes_the_steps_rows_and_attends_on_them(
    monkeypatch, kernels
):
    """The descales see the live row count alone: on a replay the pad rows
    hold whatever the last step left there. The region attends with what the
    recorded head quantized."""
    impl = _dense_impl(
        torch.zeros(10, HEADS, NOPE + ROPE), torch.zeros(10, HEADS, V_DIM)
    )
    impl._flydsl_fp8_mha_ok = True
    live = torch.tensor([5], dtype=torch.int32)
    _run(monkeypatch, impl, _metadata("new_only", 8), 5, bucket=8, live=live)
    scales = [torch.tensor([float(i)]) for i in range(5)]
    quantized, attended = [], []

    def quantize(q, k, v, *, k_rope=None, num_rows=None):
        quantized.append(((q, k, v), num_rows))
        return (q.clone(), k.clone(), v.clone(), *scales)

    def attend(q8, k8, v8, **kw):
        attended.append(((q8, k8, v8), kw))
        return q8.new_zeros(q8.shape[0], HEADS, V_DIM)

    monkeypatch.setattr(mla, "fused_qkv_per_tensor_quant", quantize)
    monkeypatch.setattr(mla, "flydsl_flash_attn_fp8_func", attend)
    split = impl.forward_prefill_split(*_split_inputs(8))

    assert split.shape == (8, HEADS * V_DIM)
    ((inputs, num_rows),) = quantized
    assert num_rows is live
    assert all(x.shape[0] == 8 for x in inputs)
    (((q8, k8, v8), kw),) = attended
    torch.testing.assert_close(q8, inputs[0], rtol=0, atol=0)
    torch.testing.assert_close(k8, inputs[1], rtol=0, atol=0)
    torch.testing.assert_close(v8, inputs[2], rtol=0, atol=0)
    assert kw["q_descale"] is scales[0]
    assert kw["k_descale"] is scales[1]
    assert kw["v_descale"] is scales[2]


# ---- the KDA split against _forward_impl, with a reference recurrence -------

KDA_HEADS, KDA_DIM, HIDDEN = 2, 4, 16


def _rearrange(x, pattern, **sizes):
    """The einops reshapes the KDA layer makes, where einops is missing."""
    if pattern == "t (h d) -> 1 t h d":
        return x.unflatten(-1, (-1, sizes["d"]))[None]
    if pattern == "t (h d) -> t h d":
        return x.unflatten(-1, (-1, sizes["d"]))
    if pattern == "t h d -> t (h d)":
        return x.flatten(-2)
    raise NotImplementedError(pattern)


def _logged(**arguments):
    """A kernel call's arguments as its stand-in saw them."""
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in arguments.items()
    }


def _kda_layer(model, calls):
    """A KDA layer of linear projections whose recurrence kernel (`_run_kda`)
    is a causal reference logging every input into ``calls``."""
    torch.manual_seed(3)
    width = KDA_HEADS * KDA_DIM
    w_in = torch.randn(4 * width + KDA_HEADS + KDA_DIM, HIDDEN)
    w_f, w_o = torch.randn(width, KDA_DIM), torch.randn(HIDDEN, width)

    def run_kda(q, k, v, gate, beta, initial, cu_seqlens, final_state):
        arguments = _logged(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            initial=initial,
            cu_seqlens=cu_seqlens,
            final_state=final_state,
        )
        calls.append(("kda", arguments))
        out = torch.cumsum((q + k) * v * gate.sigmoid() * beta[..., None], dim=1)
        return out, initial + 1

    layer = object.__new__(model.KimiKDAAttention)
    layer.__dict__.update(
        layer_name="kda",
        layer_num=0,
        head_dim=KDA_DIM,
        num_local_heads=KDA_HEADS,
        local_proj_size=width,
        fuse_f_a_quant=False,
        _split_in_proj=False,
        conv_weight=None,
        in_proj=lambda x, x_scale=None: nn.functional.linear(x, w_in),
        f_b_proj=lambda x: nn.functional.linear(x, w_f),
        o_norm=lambda out, gate, is_prefill: out * gate.sigmoid(),
        o_proj=lambda x: nn.functional.linear(x, w_o),
        activation="silu",
        _run_kda=run_kda,
    )
    return layer


def _causal_conv(calls):
    """The causal conv as the identity -- q, k and v are the projection's --
    logging every input into ``calls``."""

    def conv(x, weight, bias, **kw):
        calls.append(("conv", _logged(x=x, weight=weight, bias=bias, **kw)))
        return x.transpose(0, 1).split(kw["k_dim_size"], dim=-1)

    return conv


def _kda_cache():
    """A KDA layer's cache entry: a random conv and SSM state, the same every
    call."""
    random = torch.Generator().manual_seed(5)
    return SimpleNamespace(
        k_cache=torch.randn(4, 3 * KDA_HEADS * KDA_DIM, 4, generator=random),
        v_cache=torch.randn(4, KDA_HEADS, KDA_DIM, KDA_DIM, generator=random),
    )


def _kda_step(monkeypatch, model, layer, metadata, calls, *, rows=5, dummy=False):
    """A prefill step of ``rows`` real tokens with KDA ``metadata`` (None:
    none); ``dummy`` is the recording's dummy step, which has none. Returns the
    layer's cache entry (`_kda_cache`)."""
    cache = _kda_cache()
    attn_metadata = SimpleNamespace()
    if metadata is not None:
        attn_metadata.kda_metadata = metadata
    ctx = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=True, scheduled_tokens=rows, is_dummy_run=dummy
        ),
        attn_metadata=attn_metadata,
        kv_cache_data={"layer_0": cache},
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
    if importlib.util.find_spec("einops") is None:
        monkeypatch.setattr(model, "rearrange", _rearrange)
    monkeypatch.setattr(model, "causal_conv1d_fn", _causal_conv(calls))
    # Imported where it is called: a fresh request's state reads as zeros.
    kernels = ModuleType("atom.model_ops.kimi_k3")
    kernels.gather_kda_initial_state = lambda state, slots, has_initial: (
        state[slots] * has_initial[:, None, None, None]
    )
    monkeypatch.setitem(sys.modules, "atom.model_ops.kimi_k3", kernels)
    return cache


def _kda_metadata():
    """Prompts of 3 and 2 tokens. The first resumes the state it reads from
    slot 1 into slot 2, as a prefix-cache fork does; the second starts afresh
    in slot 0."""
    return SimpleNamespace(
        num_actual_tokens=5,
        num_prefills=2,
        num_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([2, 0]),
        non_spec_state_indices_in_tensor=torch.tensor([1, 3]),
        non_spec_query_start_loc=torch.tensor([0, 3, 5]),
        has_initial_state=torch.tensor([True, False]),
    )


def _assert_same_call(got, want, rows, bucket):
    """A split's kernel call against the eager forward's: the same arguments,
    a token-major one (tokens on dim 1) the whole bucket tall, the eager step's
    rows first."""
    assert got.keys() == want.keys()
    for name, value in want.items():
        other = got[name]
        if not isinstance(value, torch.Tensor):
            assert other is value or other == value, name
            continue
        assert isinstance(other, torch.Tensor), name
        if other.shape != value.shape:
            assert (other.shape[1], value.shape[1]) == (bucket, rows), name
            other = other[:, :rows]
        torch.testing.assert_close(other, value, rtol=1e-6, atol=1e-6, msg=name)


@pytest.mark.parametrize("bucket", [5, 8])
def test_a_padded_kda_split_makes_the_kernel_calls_forward_impl_makes(
    monkeypatch, bucket
):
    """Every input of the causal conv and the recurrence, the slot the state
    is read from and the one it is written to included."""
    model = _kimi_model()
    rows = 5
    torch.manual_seed(4)
    hidden = torch.randn(bucket, HIDDEN)
    metadata = _kda_metadata()

    eager_calls, split_calls = [], []
    layer = _kda_layer(model, eager_calls)
    eager_cache = _kda_step(monkeypatch, model, layer, metadata, eager_calls)
    eager = layer._forward_impl(hidden[:rows].clone(), None)

    layer = _kda_layer(model, split_calls)
    split_cache = _kda_step(monkeypatch, model, layer, metadata, split_calls)
    # Whatever the last step left in the pad rows reaches no real row.
    hidden[rows:] = float("nan")
    split = layer._forward_prefill_split(hidden, None)

    assert split.shape == (bucket, HIDDEN)
    assert eager.abs().sum() > 0
    torch.testing.assert_close(split[:rows], eager, rtol=1e-5, atol=1e-5)
    assert (
        [n for n, _ in split_calls]
        == [n for n, _ in eager_calls]
        == [
            "conv",
            "kda",
        ]
    )
    for (_, got), (_, want) in zip(split_calls, eager_calls):
        _assert_same_call(got, want, rows, bucket)
    # The state written where the eager step writes it, from what it read.
    assert not torch.equal(eager_cache.v_cache, _kda_cache().v_cache)
    torch.testing.assert_close(split_cache.v_cache, eager_cache.v_cache)


def test_the_kda_recordings_dummy_step_touches_no_state(monkeypatch):
    model = _kimi_model()
    calls = []
    layer = _kda_layer(model, calls)
    cache = _kda_step(monkeypatch, model, layer, None, calls, rows=8, dummy=True)
    before = (cache.k_cache.clone(), cache.v_cache.clone())
    split = layer._forward_prefill_split(torch.randn(8, HIDDEN), None)
    assert split.shape == (8, HIDDEN)
    # Zeros through the gated norm and o_proj: the warmup forward runs on them.
    assert torch.count_nonzero(split) == 0
    assert not calls
    assert torch.equal(cache.k_cache, before[0])
    assert torch.equal(cache.v_cache, before[1])


def test_a_live_kda_replay_without_metadata_fails_loudly(monkeypatch):
    """Only the recording's dummy step may skip the recurrence; a live step
    with no KDA metadata must not hand back an unwritten output."""
    model = _kimi_model()
    calls = []
    layer = _kda_layer(model, calls)
    _kda_step(monkeypatch, model, layer, None, calls)
    with pytest.raises(AssertionError, match="no KDA metadata"):
        layer._forward_prefill_split(torch.randn(8, HIDDEN), None)
    assert not calls


def test_a_prefills_conv_map_spans_the_rows_the_step_runs(monkeypatch):
    """A split KDA layer hands the causal conv the whole bucket
    (`_kda_recurrence`); `causal_conv1d_fn` would rebuild a map sized for fewer
    rows in every layer."""
    with _einops(), atom_without_aiter() as load:
        gdn = load("atom.model_ops.attentions.gdn_attn")
    sized = []

    def conv_map(query_start_loc, num_tokens):
        sized.append(num_tokens)
        return {8: {"num_tokens": num_tokens}}, None, None

    monkeypatch.setattr(gdn, "compute_causal_conv1d_metadata", conv_map)
    slots = SimpleNamespace(gpu=torch.arange(4))
    builder = SimpleNamespace(
        use_spec_decode=False,
        device="cpu",
        non_spec_state_indices_tensor=slots,
        non_spec_state_indices_in_tensor=slots,
        _has_state_fork=False,
        replayssm=False,
    )
    batch = SimpleNamespace(
        total_seqs_num=2,
        total_seqs_num_decode=0,
        total_seqs_num_prefill=2,
        total_tokens_num=5,
        total_tokens_num_decode=0,
        total_tokens_num_prefill=5,
    )
    # Published out to a padding request (`stage_prefill`).
    attn_metadata = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 3, 5, 5]), num_cached_tokens=None
    )
    for running_tokens in (8, 5, 0):  # a bucket; unpadded; a caller passing none
        built = gdn.GDNStateMixin.prepare_gdn_metadata(
            builder,
            batch,
            attn_metadata,
            is_prefill=True,
            prepare_block_tables=False,
            running_tokens=running_tokens,
        )
        assert built.non_spec_query_start_loc.tolist() == [0, 3, 5]
    assert sized == [8, 5, 5]
