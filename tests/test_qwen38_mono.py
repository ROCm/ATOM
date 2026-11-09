# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next mono: which deployments and steps it serves and its host-side contracts (CPU), the step prep
launches and the PLE n-gram host table (the GPU tests skip without a GPU or Triton)."""

import ctypes
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from atom.models.qwen38.mono import config as mono_config
from atom.models.qwen38.mono import step_meta
from atom.models.qwen38.mono.kernels.weights import aiter_moe
from atom.models.qwen38.mono.kernels.weights.source import (
    LayoutError,
    WeightSpec,
    check,
)
from atom.mono.runtime import deployment

_HF = {
    "num_hidden_layers": 48,
    "hidden_size": 2560,
    "hc_count": 4,
    "hc_lowrank": 320,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "layer_types": [
        "qwen_sparse_attention" if i % 4 == 3 else "linear_attention" for i in range(48)
    ],
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "ple_layer_ids": [2],
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "mamba_ssm_dtype": "float32",
    "output_gate_type": "sigmoid",
}
_QUANT = {
    "quant_method": "quark",
    "global_quant_config": {
        "weight": {"dtype": "fp4"},
        "input_tensors": {"dtype": "fp4"},
    },
    "layer_quant_config": {},
}


def _atom_config(**overrides):
    """Configuration F as a worker sees it."""
    fields = {
        "tensor_parallel_size": 1,
        "enable_expert_parallel": False,
        "enable_dp_attention": False,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "compilation_config": SimpleNamespace(level=3, cudagraph_mode=None),
        "kv_cache_dtype": "fp8",
        "index_cache_dtype": "bf16",
        "kv_cache_block_size": 64,
        "speculative_config": SimpleNamespace(method="mtp", num_speculative_tokens=3),
        "quant_config": SimpleNamespace(quant_method="quark", hf_quant_config=_QUANT),
        "torch_dtype": torch.bfloat16,
        "hf_config": SimpleNamespace(**_HF),
        "max_model_len": 262144,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(deployment, "_gfx", lambda: "gfx950")
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    monkeypatch.setenv("ATOM_GDN_SSM_DTYPE", "fp32")
    monkeypatch.delenv("ATOM_QSA_DRAFT_KV_BF16", raising=False)


@pytest.mark.parametrize("replayssm", ["0", "1", None])
def test_configuration_f_is_served_on_either_verify(native, monkeypatch, replayssm):
    """ReplaySSM (ATOM's default under MTP, unset) and the classic verify alike."""
    if replayssm is None:
        monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM")
    else:
        monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", replayssm)
    assert mono_config.config_refusal(_atom_config()) is None
    offload = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
    assert mono_config.config_refusal(_atom_config(kv_transfer_config=offload)) is None


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"tensor_parallel_size": 8}, "TP 8"),
        ({"speculative_config": None}, "MTP3"),
        ({"kv_cache_dtype": "bf16"}, "KV / index caches"),
        ({"kv_cache_block_size": 16}, "KV block 16"),
        (
            {"quant_config": SimpleNamespace(quant_method="fp8", hf_quant_config={})},
            "Quark MXFP4",
        ),
        ({"hf_config": SimpleNamespace(**{**_HF, "ple_layer_ids": [3]})}, "PLE layers"),
        # a connector whose kernels can hold the CUs the resident programs wait on
        ({"kv_transfer_config": {"kv_connector": "moriio"}}, "KV connector moriio"),
    ],
)
def test_other_deployments_are_refused(native, overrides, reason):
    config = _atom_config(**overrides)
    assert reason in mono_config.config_refusal(config)
    assert not mono_config.refused_by_switch_only(config)


def test_a_compute_partition_is_refused(native, monkeypatch):
    monkeypatch.setattr(deployment, "_cu_count", lambda: 128)
    assert "128 CUs" in mono_config.config_refusal(_atom_config())


@pytest.mark.parametrize(
    "env,value,reason",
    [
        ("ATOM_MONO_ENABLE", "0", "ATOM_MONO_ENABLE"),
        ("ATOM_REPLAYSSM_CACHE_LEN", "128", "ATOM_REPLAYSSM_CACHE_LEN"),
        ("ATOM_GDN_SSM_DTYPE", "fp16", "ATOM_GDN_SSM_DTYPE"),
        ("ATOM_QSA_DRAFT_KV_BF16", "0", "draft caches"),
    ],
)
def test_engine_switches_are_checked(native, monkeypatch, env, value, reason):
    monkeypatch.setenv(env, value)
    if env == "ATOM_REPLAYSSM_CACHE_LEN":
        monkeypatch.delenv(
            "ATOM_ENABLE_REPLAYSSM"
        )  # (the capacity matters under ReplaySSM only)
    assert reason in mono_config.config_refusal(_atom_config())
    # a served deployment missing only an engine switch is warned about at install
    assert mono_config.refused_by_switch_only(_atom_config()) == (
        env != "ATOM_MONO_ENABLE"
    )


def test_install_reads_a_hub_id_checkpoint_from_its_local_directory(
    native, monkeypatch, tmp_path
):
    """``--model`` may be a hub id: the installer resolves the directory once and hands it to every reader (the
    geometry, the drafter's layer, the target's layers), and wraps the target and the drafter.
    """
    from atom.models.qwen38.mono import dispatch

    hub_id, seen = "amd/Qwen3.8-Flash-Next-Quark-MXFP4", []
    monkeypatch.setattr(
        dispatch, "local_model_dir", lambda m: str(tmp_path) if m == hub_id else m
    )
    monkeypatch.setattr(dispatch, "load_geometry", lambda d: seen.append(d) or "geo")

    class Runner:
        def __init__(self, *args, **kwargs):
            seen.append(args[-1])

    class Wrapped:
        def __init__(self, draft, runner):
            self.draft, self.runner = draft, runner

    # (the runners, the drafter and ATOM's MTP model need a GPU: stand-ins record what they are given)
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    mtp = type("Qwen4ExpMTP", (), {})
    module("atom.models.qwen38.mono.runner", TargetRunner=Runner)
    module("atom.models.qwen38.mono.draft_runner", DraftRunner=Runner)
    module("atom.models.qwen38.mono.draft_model", MonoDraftModel=Wrapped)
    module("atom.models.qwen4_exp_mtp", Qwen4ExpMTP=mtp)
    draft = mtp()
    draft.model = SimpleNamespace()
    drafter = SimpleNamespace(model=draft)
    out = dispatch.install_mono_decode(
        torch.nn.Module(), _atom_config(model=hub_id), drafter
    )
    assert isinstance(out, dispatch.MonoTargetModel)
    assert isinstance(drafter.model, Wrapped) and drafter.model.draft is draft
    assert seen == [str(tmp_path)] * 3


def _fwd(batch=2, spec=4, prefill=False, replayssm=False, decodes=0, idx_rows=None):
    gdn = SimpleNamespace(
        replayssm=replayssm,
        num_prefills=0,
        num_decodes=decodes,
        num_spec_decodes=batch,
        spec_state_indices_tensor=torch.zeros(
            idx_rows or batch, spec, dtype=torch.int32
        ),
    )
    md = SimpleNamespace(gdn_metadata=gdn, image_mask=None)
    return SimpleNamespace(
        context=SimpleNamespace(is_prefill=prefill),
        ubatch_slices=None,
        attn_metadata=md,
    )


@pytest.mark.parametrize(
    "fwd,rows,replayssm,served",
    [
        (_fwd(1), 4, False, 1),
        (_fwd(4, idx_rows=8), 16, False, 4),  # a wider slot table
        (_fwd(2, replayssm=True), 8, True, 2),
        (_fwd(5), 20, False, None),  # concurrency > 4: Native
        (_fwd(2), 6, False, None),  # not SPEC rows a request
        (_fwd(2, prefill=True), 8, False, None),
        (_fwd(2, decodes=1), 8, False, None),  # a mixed decode
        (_fwd(2, spec=2), 8, False, None),  # another slot table
        # not the verify the kernels were built for
        (_fwd(2, replayssm=True), 8, False, None),
        (_fwd(2), 8, True, None),
    ],
)
def test_verify_steps_served(fwd, rows, replayssm, served):
    assert (
        step_meta.verify_step(fwd, rows, mono_config.BATCHES, replayssm=replayssm)
        == served
    )


def test_target_puts_fill_every_metadata_arg_but_the_selection():
    batch, width = 2, 8
    m = step_meta.meta_buffers(batch, 4, width, 16, "cpu")
    table = torch.arange(batch * (width + 3), dtype=torch.int32).view(batch, width + 3)[
        :, :width
    ]
    meta = step_meta.StepMeta(
        batch,
        torch.tensor([True, False]),
        table,
        torch.tensor([9, 9], dtype=torch.int32),
        torch.tensor([3, 4], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([5, 6], dtype=torch.int32),
        torch.arange(8, dtype=torch.int32).view(2, 4),
        rope_off=torch.tensor([0, -900], dtype=torch.int32),
    )
    puts = {id(p.dst): p for p in step_meta.target_puts(m, meta)}
    assert set(puts) == {
        id(v) for k, v in m.items() if k not in ("sel", "h1_mem", "stage")
    }
    assert puts[id(m["seq_lens"])].sub == 4 and puts[id(m["seq_lens"])].fill == 0
    assert not puts[id(m["gdn.conv_off"])].masked
    bt = puts[id(m["block_table"])]
    assert (bt.group, bt.inner, bt.stride) == (width, width, width + 3)
    assert puts[id(m["gdn.ssm_write"])].group == 4
    assert puts[id(m["rope_off"])].fill == 0


def test_draft_rows_must_be_consecutive_block_table_slots():
    table = [[7, 3], [2, 5]]  # two requests, block 4
    positions = [1, 2, 3, 4, 2, 3, 4, 5]
    slots = [
        7 * 4 + 1,
        7 * 4 + 2,
        7 * 4 + 3,
        3 * 4 + 0,
        2 * 4 + 2,
        2 * 4 + 3,
        5 * 4 + 0,
        5 * 4 + 1,
    ]
    assert step_meta.draft_rows_error(positions, slots, table, 4, 2, 4) is None
    assert (
        step_meta.draft_rows_error(positions, slots[:-1] + [0], table, 4, 2, 4)
        is not None
    )
    assert (
        step_meta.draft_rows_error([1, 2, 4, 5, 2, 3, 4, 5], slots, table, 4, 2, 4)
        is not None
    )


def test_pools_refuse_another_layout_or_dtype():
    from atom.models.qwen38.mono.kernels.geometry import G

    blocks, bs = 3, 64
    kv = torch.zeros(blocks, bs, 2, 256, dtype=torch.bfloat16)
    raw = torch.zeros(blocks, bs, 1, 128, dtype=torch.bfloat16)
    cmp = torch.zeros(blocks, bs // 4, 1, 128, dtype=torch.bfloat16)
    step_meta.QSAPools(G, kv, kv.clone(), raw, cmp)
    fp8 = kv.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="fp32 k_scale"):
        step_meta.QSAPools(G, fp8, fp8.clone(), raw, cmp)
    scale = torch.ones(blocks, bs, 2)
    step_meta.QSAPools(G, fp8, fp8.clone(), raw, cmp, scale, scale.clone())
    with pytest.raises(ValueError, match="index_cmp"):
        step_meta.QSAPools(G, kv, kv.clone(), raw, raw)
    conv = torch.zeros(5, 6, G.conv_dim, dtype=torch.bfloat16)
    pools = step_meta.GDNPools(G.gdn, conv, torch.zeros(5, 48, 128, 128))
    pools.check_window(4)
    # a channel-major pool, or a view of the token-major one, is not what the kernels read
    for other in (conv.transpose(1, 2).contiguous(), conv.transpose(1, 2)):
        with pytest.raises(ValueError, match="unsupported conv pool"):
            step_meta.GDNPools(G.gdn, other, torch.zeros(5, 48, 128, 128))
    with pytest.raises(ValueError, match="dense"):  # every other slot of a wider pool
        step_meta.GDNPools(
            G.gdn, conv.repeat(2, 1, 1)[::2], torch.zeros(5, 48, 128, 128)
        )
    with pytest.raises(ValueError, match="conv rows"):
        pools.check_window(1)
    with pytest.raises(ValueError, match="state pool is torch.bfloat16"):
        step_meta.GDNPools(
            G.gdn, conv, torch.zeros(5, 48, 128, 128, dtype=torch.bfloat16)
        )
    # ReplaySSM: ATOM's record buffers of the layer and the cursor
    rec = (
        torch.zeros(5, 48, 16, 128, dtype=torch.bfloat16),
        torch.zeros(5, 48, 16, 128, dtype=torch.bfloat16),
        torch.zeros(5, 48, 16),
    )
    cursor = torch.zeros(5, dtype=torch.int32)
    ssm = torch.zeros(5, 48, 128, 128)
    assert step_meta.GDNPools(G.gdn, conv, ssm, rec, cursor).replay == 16
    assert step_meta.GDNPools(G.gdn, conv, ssm).replay == 0
    with pytest.raises(ValueError, match="ReplaySSM records"):
        step_meta.GDNPools(
            G.gdn, conv, ssm, (rec[0], rec[1], rec[2].bfloat16()), cursor
        )
    with pytest.raises(ValueError, match="cursor"):
        step_meta.GDNPools(G.gdn, conv, ssm, rec, None)


@pytest.mark.parametrize("gate_up", [False, True])
def test_expert_layout_checks_reject_dirty_padding(gate_up):
    """Bind-time checks of ATOM's shuffled, padded A8W4 experts reach the padding lanes."""
    experts, stored_n, stored_k, used_n, used_k = 3, 128, 256, 64, 128
    weights = torch.zeros(experts, stored_n, stored_k // 2, dtype=torch.uint8)
    scales = torch.zeros(experts, stored_n, stored_k // 32, dtype=torch.uint8)
    for expert in range(experts):
        ranges = (
            [(0, used_n)]
            if not gate_up
            else [(0, used_n // 2), (stored_n // 2, stored_n // 2 + used_n // 2)]
        )
        for lo, hi in ranges:
            weights[expert, lo:hi, : used_k // 2] = 17 + expert
            scales[expert, lo:hi, : used_k // 32] = (
                torch.arange(1, used_k // 32 + 1) + expert * 11
            )
    weights = aiter_moe.shuffle_weight(weights, gate_up)
    scales = aiter_moe.shuffle_scale(
        scales.reshape(-1, stored_k // 32), experts, gate_up
    )
    params = (("gate_up", gate_up), ("n", used_n), ("k", used_k), ("experts", experts))
    weight_spec = WeightSpec(
        "weights", "aiter.a8w4.moe", tuple(weights.shape), (torch.uint8,), params=params
    )
    scale_spec = WeightSpec(
        "scales", "aiter.e8m0.moe", tuple(scales.shape), (torch.uint8,), params=params
    )
    check(weights, weight_spec)
    check(scales, scale_spec)
    weights[-1].view(-1)[-1] = 1
    with pytest.raises(LayoutError, match="padding"):
        check(weights, weight_spec)
    scales[-1, -1] = 1
    with pytest.raises(LayoutError, match="padding"):
        check(scales, scale_spec)


def test_a_shape_that_does_not_build_stops_startup(monkeypatch):
    """An admitted deployment is served whole: a build failure is raised at startup, and a step never builds."""
    pytest.importorskip("flydsl")
    from atom.models.qwen38.mono import runner
    from atom.models.qwen38.mono.kernels.geometry import G
    from atom.models.qwen38.mono.kernels.program import PlanError

    def no_room(*args, **kwargs):
        raise PlanError("no room")

    monkeypatch.setattr(runner, "compose_layer", no_room)
    kernels = runner.Kernels("Qwen mono", ("delta",), G)
    with pytest.raises(RuntimeError, match="ATOM_MONO_ENABLE=0"):
        kernels.build(1, 4, 4096)
    assert not kernels.serves(1, 4, 4096)


# ------------------------------------------------------------------ the fallback draft's QSA attention path


def _qsa(requests):
    return SimpleNamespace(block_tables=torch.zeros(requests, 4096, dtype=torch.int32))


@pytest.mark.parametrize("requests", [1, 6])
@pytest.mark.parametrize("query_length", [1, 4])
def test_fallback_draft_attention_path(requests, query_length):
    """Decode rows of one or SPEC a request take the fused path only with the three-axis RoPE position cache: a
    text-only pool has none (the fused index preparation reads it), so ATOM's module serves it; six requests are
    past the draft program, where serving reached this path."""
    rows, cache = requests * query_length, object()
    assert step_meta.draft_attention_fused(_qsa(requests), False, rows, cache)
    assert not step_meta.draft_attention_fused(_qsa(requests), False, rows, None)
    assert not step_meta.draft_attention_fused(_qsa(requests), True, rows, cache)
    assert not step_meta.draft_attention_fused(None, False, rows, cache)
    assert not step_meta.draft_attention_fused(_qsa(requests), False, rows + 1, cache)


def test_fallback_draft_attention_runs_the_module_without_a_position_cache(
    monkeypatch,
):
    """The wiring: a text-only pool's step goes to ATOM's module before any fused preparation (GPU libraries)."""
    pytest.importorskip("triton")
    pytest.importorskip("aiter")
    from atom.models.qwen38.mono import draft_fallback

    calls = []

    class Module:  # ATOM's QSA module, without the position cache
        rope_position_cache = None

        def __call__(self, positions, mixed):
            calls.append(mixed)
            return "native"

    module = Module()
    fwd = SimpleNamespace(
        attn_metadata=SimpleNamespace(qsa_metadata=_qsa(6)),
        context=SimpleNamespace(is_prefill=False, is_draft=True),
    )
    monkeypatch.setattr(draft_fallback, "get_forward_context", lambda: fwd)
    mixed = torch.zeros(6, 8)
    assert draft_fallback.attention(module, None, mixed) == "native" and calls == [
        mixed
    ]


# ------------------------------------------------------------------ the step prep launches (GPU)


@pytest.fixture
def prep_ops():
    """The step prep's Triton launches."""
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from atom.models.qwen38.mono.kernels import prep

    return prep


@pytest.mark.parametrize("batch", [1, 4])
def test_verify_prep_graph_replay_selects_accepted_state(prep_ops, batch):
    spec, hidden, streams = 4, 2560, 4
    ids = torch.arange(batch * spec, device="cuda", dtype=torch.int32)
    embedding = (
        (torch.arange(32 * hidden, device="cuda") % 251)
        .to(torch.bfloat16)
        .reshape(32, hidden)
    )
    out = torch.empty(
        batch * spec, hidden * streams, device="cuda", dtype=torch.bfloat16
    )
    # Strided slot tables and a permutation make request IDs different from slot IDs.
    idx = torch.arange(100, 100 + batch * 8, device="cuda", dtype=torch.int32).reshape(
        batch, 8
    )[:, :4]
    idx_in = torch.flip(idx.clone(), [0]) + 100
    accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
    slots = torch.arange(batch * spec, device="cuda", dtype=torch.int64)
    # MRoPE positions [3, rows]: request b's rows sit `delta` past its cache rows (an image request's offset)
    context = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32) * 100
    delta = torch.tensor([0, -900, 7, -3][:batch], device="cuda", dtype=torch.int64)
    rows = (context.long() - spec)[:, None] + torch.arange(spec, device="cuda")
    positions = (rows + delta[:, None]).reshape(1, -1).repeat(3, 1).contiguous()
    prep = prep_ops.VerifyPrep(batch)

    def launch():
        return prep(ids, embedding, out, accepted, idx, idx_in, batch, streams, slots, spec,
                    positions, context)  # fmt: skip

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for count in (1, 4, 2, 3):
        accepted.fill_(count)
        slots.fill_(1)
        if batch > 1:
            slots[-spec] = -1
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, embedding[ids.long()].repeat(1, streams))
        assert torch.equal(prep.prev, accepted - 1)
        assert torch.equal(prep.slot, idx[:, 0])
        assert torch.equal(prep.ssm_read, idx_in[:, count - 1])
        assert torch.equal(prep.valid, slots[::spec] >= 0)
        assert torch.equal(
            prep.rope_off, torch.where(prep.valid, delta, 0).to(torch.int32)
        )


@pytest.mark.parametrize("spec", [1, 4])
@pytest.mark.parametrize("selected", [False, True])
def test_draft_metadata_replay_padding_and_epoch(prep_ops, spec, selected):
    batch, width = 4, 4096
    pos = torch.arange(batch * spec, device="cuda", dtype=torch.int64) + 16384
    slots = torch.ones_like(pos)
    # A strided source checks that table row stride is respected.
    table = torch.arange(batch * (width + 7), device="cuda", dtype=torch.int64).reshape(
        batch, width + 7
    )[:, :width]
    select = (
        torch.arange(batch, device="cuda", dtype=torch.int64) * spec
        if selected
        else None
    )
    metadata = {
        "seq_lens": torch.zeros(batch, device="cuda", dtype=torch.int32),
        "block_table": torch.zeros(batch, width, device="cuda", dtype=torch.int32),
        "sel": torch.full((batch,), -7, device="cuda", dtype=torch.int32),
    }
    epoch = torch.tensor(9, device="cuda", dtype=torch.int32)

    def launch():
        prep_ops.draft_meta(metadata, [epoch], pos, slots, table, select, batch, spec)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    previous = epoch.item()
    for step in range(3):
        pos.add_(1)
        slots[spec] = -1
        pos[2 * spec] = -1
        graph.replay()
        torch.cuda.synchronize()
        assert epoch.item() == previous + step + 1
        assert metadata["seq_lens"].tolist() == [
            pos[0].item(),
            0,
            0,
            pos[3 * spec].item(),
        ]
        assert torch.equal(
            metadata["block_table"][[0, 3]], table[[0, 3]].to(torch.int32)
        )
        assert bool((metadata["block_table"][[1, 2]] == -1).all())
        expected = (
            select.to(torch.int32) if selected else torch.full_like(metadata["sel"], -7)
        )
        assert torch.equal(metadata["sel"], expected)


@pytest.mark.parametrize("batch", [1, 4])
def test_target_metadata_puts_match_the_reference_fill(prep_ops, batch):
    """The fused metadata launch writes what the reference ops would: a padded request gets slots -1 and a
    context of 0 over a block-table row of -1 (no pool touched), every counter + 1."""
    from atom.models.qwen38.mono.step_meta import StepMeta, meta_buffers, target_puts

    fused_puts = prep_ops.fused_puts

    spec, width, dev = 4, 4096, "cuda"
    gen = torch.Generator().manual_seed(batch)

    def ints(*shape):
        return torch.randint(0, 1 << 20, shape, generator=gen, dtype=torch.int32).to(
            dev
        )

    valid = torch.ones(batch, dtype=torch.bool)
    valid[batch // 2] = batch == 1
    table = ints(batch, width + 7)[:, :width]
    meta = StepMeta(batch, valid.to(dev), table, ints(batch) + spec, ints(batch), ints(batch) % 4, ints(batch),
                    ints(batch, spec), rope_off=ints(batch) - (1 << 19))  # fmt: skip
    m = meta_buffers(batch, spec, width, 16, dev)
    counters = [torch.full((1,), 5, dtype=torch.int32, device=dev) for _ in range(2)]
    fused_puts(meta.valid, target_puts(m, meta), counters)
    torch.cuda.synchronize()
    v, rows = meta.valid, meta.valid.repeat_interleave(spec)
    assert torch.equal(m["gdn.conv_slot"], torch.where(v, meta.conv_slot, -1))
    assert torch.equal(m["gdn.conv_off"], meta.conv_off)
    assert torch.equal(m["rope_off"], torch.where(v, meta.rope_off, 0))
    assert torch.equal(m["gdn.ssm_read"], torch.where(v, meta.ssm_read, -1))
    assert torch.equal(
        m["gdn.ssm_write"], torch.where(rows, meta.ssm_write.reshape(-1), -1)
    )
    assert torch.equal(m["seq_lens"], torch.where(v, meta.context_lens - spec, 0))
    assert torch.equal(m["block_table"], torch.where(v[:, None], table, -1))
    assert all(c.item() == 6 for c in counters)


# ------------------------------------------------------------------ the PLE n-gram table in page-locked host memory
# The CPU half checks the shard addresses by following them the way the kernel does; the GPU half checks the kernel
# against the HBM embedding it replaces, bit for bit. The module holds the Triton lookup kernel.


def _ple():
    pytest.importorskip("triton")
    from atom.models.qwen38.mono import ple

    return ple


DIM = 160
SHARD_ROWS = 37
NUM_SHARDS = 6
# The last shard is short, like a checkpoint whose padded table is not a
# multiple of the shard width.
TABLE_ROWS = SHARD_ROWS * NUM_SHARDS - 5


def _table(dtype=torch.bfloat16, rows=TABLE_ROWS):
    gen = torch.Generator().manual_seed(0)
    return torch.randn(rows, DIM, generator=gen).to(dtype)


def _shards(table):
    return list(table.split(SHARD_ROWS))


def _probe_ids(rows, count=512):
    """Random ids plus every edge: first/last row, each side of every shard
    boundary, and the padded tail."""
    gen = torch.Generator().manual_seed(1)
    edges = {0, rows - 1, rows - 2}
    for shard in range(1, -(-rows // SHARD_ROWS)):
        edges |= {shard * SHARD_ROWS - 1, shard * SHARD_ROWS}
    ids = torch.cat(
        [torch.tensor(sorted(edges)), torch.randint(0, rows, (count,), generator=gen)]
    )
    return ids.to(torch.int64)


def _follow(bases, ids, start, end, dtype, shard_rows=SHARD_ROWS):
    """The kernel's addressing on the host: row `i` is at
    `bases[i // shard_rows] + (i % shard_rows) * row_bytes`; another rank's
    rows are zeros."""
    row_bytes = DIM * dtype.itemsize
    out = torch.zeros(ids.numel(), DIM, dtype=dtype)
    for n, i in enumerate(ids.tolist()):
        if start <= i < end:
            address = int(bases[i // shard_rows]) + (i % shard_rows) * row_bytes
            raw = bytearray(ctypes.string_at(address, row_bytes))
            out[n] = torch.frombuffer(raw, dtype=dtype)
    return out


def _bitwise_equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.fixture
def fake_pin(monkeypatch):
    """`pin_rows` as copies into fresh host memory; `fail_at` refuses one
    shard (by its first row), as a full page-locked pool would."""
    state = SimpleNamespace(calls=0, fail_at=None, copy=[])

    def pin_rows(sources, copy=True):
        state.calls += 1
        state.copy.append(copy)
        if state.fail_at is not None and state.calls == state.fail_at:
            return None
        return tuple(s.clone() for s in sources)

    monkeypatch.setattr(_ple(), "pin_rows", pin_rows)
    # Copies in plain memory: the device would see them where the host does.
    monkeypatch.setattr(_ple(), "device_address", lambda address: address)
    return state


def _load(table, host, start=0, end=TABLE_ROWS, shards=None):
    """What the shard loader hands over: each shard's overlap with the rank."""
    for index, shard in enumerate(_shards(table) if shards is None else shards):
        first = index * SHARD_ROWS
        lo, hi = max(first, start), min(first + shard.shape[0], end)
        if lo < hi:
            host.load_rows(index, lo, shard[lo - first : hi - first])


def _host(start=0, end=TABLE_ROWS, dtype=torch.bfloat16, device="cpu"):
    return _ple().MonoPleHostTable(
        start, end, DIM, dtype, SHARD_ROWS, torch.device(device), "ple"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_shard_addresses_find_every_row(fake_pin, dtype):
    table = _table(dtype)
    host = _host(dtype=dtype)
    _load(table, host)
    bases = host.shard_bases(NUM_SHARDS)
    assert host.on_host
    assert host.pinned_bytes == table.nbytes
    ids = _probe_ids(TABLE_ROWS)
    assert _bitwise_equal(_follow(bases, ids, 0, TABLE_ROWS, dtype), table[ids])


@pytest.mark.parametrize("fail_at", [1, 4])
def test_refused_page_lock_moves_the_table_to_device_memory(fake_pin, fail_at):
    fake_pin.fail_at = fail_at
    table = _table()
    host = _host()
    _load(table, host)
    bases = host.shard_bases(NUM_SHARDS)
    assert not host.on_host
    assert host.pinned_bytes == 0
    # Shards after the refusal are not offered for page-locking at all.
    assert fake_pin.calls == fail_at
    ids = _probe_ids(TABLE_ROWS)
    assert _bitwise_equal(_follow(bases, ids, 0, TABLE_ROWS, table.dtype), table[ids])


# ---- GPU: the kernel against the HBM embedding it replaces -----------------

_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="needs a ROCm GPU (UVA reads of page-locked host memory)",
)


def _ops():
    return _ple()


def _checkpoint_shards(table, directory):
    """The shards as the loader gets them: views of a safetensors mapping,
    adjacent and not page-aligned."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = directory / "ple.safetensors"
    names = [f"shard_{i}" for i in range(-(-table.shape[0] // SHARD_ROWS))]
    save_file({n: s.contiguous() for n, s in zip(names, _shards(table))}, path)
    with safe_open(path, framework="pt", device="cpu") as f:
        return [f.get_tensor(n) for n in names]


def _device_host(table, mode, monkeypatch, tmp_path, start=0, end=None):
    """`PleHostTable` loaded with the real page-locking in `mode`."""
    end = table.shape[0] if end is None else end
    if mode == "hbm":
        monkeypatch.setattr(_ple(), "pin_rows", lambda sources: None)
    host = _host(start, end, table.dtype, "cuda")
    _load(table, host, start, end, _checkpoint_shards(table, tmp_path))
    bases = host.shard_bases(NUM_SHARDS).cuda()
    assert host.on_host == (mode != "hbm")
    return host, bases


@_gpu
def test_uva_lookup_replays_in_a_cuda_graph(monkeypatch, tmp_path):
    ple = _ops()
    table = _table()
    host, bases = _device_host(table, "copy", monkeypatch, tmp_path)
    ids = torch.zeros(256, dtype=torch.int64, device="cuda")

    def lookup():
        return ple.sharded_table_lookup(
            ids, bases, 0, TABLE_ROWS, SHARD_ROWS, DIM, torch.bfloat16
        )

    lookup()  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = lookup()
    reference = table.cuda()
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        ids.copy_(torch.randint(0, TABLE_ROWS, (256,), generator=gen))
        graph.replay()
        torch.cuda.synchronize()
        assert _bitwise_equal(out, torch.nn.functional.embedding(ids, reference))
    del graph, host


@_gpu
def test_ngram_embedding_on_host_matches_hbm_embedding(monkeypatch):
    """The model's n-gram embedding: built on the device under a deployment Mono
    serves it goes to the host, otherwise it is the HBM `VocabParallelEmbedding`;
    both load the same shards and return the same bytes."""
    pytest.importorskip("aiter.dist.parallel_state", exc_type=ImportError)
    from atom.model_ops import embed_head
    from atom.model_ops.qwen4_exp import ple_layer

    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(embed_head, "get_tp_group", lambda: group)
    monkeypatch.setattr(ple_layer, "get_tp_group", lambda: group)
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        eos_token_id=0,
        vocab_size=4096,
        split_ngram_parts=16,
        ngram_vocab_size_base=997,
        make_ngram_vocab_size_divisible_by=128,
    )

    def build(uva):
        monkeypatch.setattr(_ple(), "_mono_serves", lambda: uva)
        with torch.device("cuda"):
            layer = ple_layer.Qwen4ExpNGramEmbedding(
                config, 16 * 32, 0, 1024, 8, prefix="ple"
            )
        return layer

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        host_layer, hbm_layer = build(True), build(False)
        assert isinstance(host_layer.ngram_embedding, _ple().MonoPleEmbedding)
        assert isinstance(hbm_layer.ngram_embedding, embed_head.VocabParallelEmbedding)
        rows = host_layer.table_rows
        gen = torch.Generator().manual_seed(2)
        table = torch.randn(rows, host_layer.head_dim, generator=gen).bfloat16()
        for layer in (host_layer, hbm_layer):
            weight = layer.ngram_embedding.weight
            for index, shard in enumerate(table.split(layer.checkpoint_shard_rows)):
                weight.weight_loader(weight, shard, index)
            if hasattr(layer.ngram_embedding, "process_weights_after_loading"):
                layer.ngram_embedding.process_weights_after_loading()
        assert host_layer.ngram_embedding.weight.numel() == 0
        assert host_layer.ngram_embedding.table.on_host
        tokens = torch.randint(0, config.vocab_size, (300,), generator=gen).cuda()
        starts = torch.tensor([0, 7, 120, 300], dtype=torch.int32, device="cuda")
        context = torch.randint(0, config.vocab_size, (3, 2), generator=gen).cuda()
        got = host_layer(tokens, starts, context)
        expected = hbm_layer(tokens, starts, context)
        assert _bitwise_equal(got, expected)
        # Padding rows of the table are reachable by id, and as checkpointed.
        tail = torch.arange(rows - 130, rows, device="cuda")
        assert _bitwise_equal(
            host_layer.ngram_embedding(tail), hbm_layer.ngram_embedding(tail)
        )
    finally:
        torch.set_default_dtype(previous)


@_gpu
@pytest.mark.parametrize("serves,fp8", [(False, False), (True, True)])
def test_refused_deployment_and_fp8_table_never_allocate_host_table(
    monkeypatch, serves, fp8
):
    monkeypatch.setattr(_ple(), "_mono_serves", lambda: serves)

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled host path allocated a table")

    monkeypatch.setattr(_ple(), "MonoPleEmbedding", unexpected)
    with torch.device("cuda"):
        assert _ple().make_host_embedding(128, DIM, 32, "ple", fp8) is None
