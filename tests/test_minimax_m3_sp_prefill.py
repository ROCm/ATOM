# SPDX-License-Identifier: MIT
"""MiniMax-M3 sequence-parallel prefill (`atom/models/minimax_m3/sp_prefill.py`)
on the CPU, AITER stubbed (`tests/aiter_stub.py`): when a step runs SP (pure TP,
native MXFP8 linears, token thresholds every TBO micro-batch reads alike), when
the FlyDSL MXFP4 MoE chain takes the experts, the unsharded o_proj copy and how
it loads, and the chain's warm-up."""

import builtins
import sys
import types
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import torch
from aiter_stub import QuantType, atom_under_stubs

CHAIN = "aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_mxfp4.prefill"
FP8 = torch.float8_e4m3fn


@pytest.fixture(scope="module")
def load():
    with atom_under_stubs() as load:
        yield load


@pytest.fixture(scope="module")
def sp(load):
    return load("atom.models.minimax_m3.sp_prefill")


# Patched on the module objects: a stub package resolves a submodule it has not
# imported yet to a MagicMock, so a dotted-path patch can miss it.


@pytest.fixture
def set_gfx(load, monkeypatch):
    chip_info = load("aiter.jit.utils.chip_info")
    return lambda gfx: monkeypatch.setattr(chip_info, "get_gfx", lambda: gfx)


@pytest.fixture
def run_as(load, monkeypatch):
    """Call `plan` as the TBO micro-batch of this context, or outside TBO (None)."""
    ubatching = load("atom.utils.tbo.ubatching")

    def enter(ctx):
        monkeypatch.setattr(ubatching, "tbo_active", lambda: ctx is not None)
        monkeypatch.setattr(ubatching, "_get_current_tbo_context", lambda: ctx)

    return enter


def lin(native=True, dtype=FP8, runs_bf16=False):
    return NS(
        native_a8_group_rows=32 if native else None,
        runs_bf16=runs_bf16,
        weight=torch.empty(1, dtype=dtype),
    )


def layer(moe=False, **kw):
    attn = NS(qkv_proj=lin(**kw), o_proj=lin(**kw), attn=NS(per_token_fp8_out=False))
    out = NS(self_attn=attn, is_moe_layer=moe)
    if not moe:
        out.mlp = NS(gate_up_proj=lin(**kw))
    return out


class Model:
    def __init__(self, layers):
        self.layers, self.start_layer, self.end_layer = layers, 0, len(layers)


def test_sp_support_is_known_from_the_linears_before_the_weights_load(sp):
    assert sp._native_mxfp8(lin())
    for not_native in (
        lin(native=False),
        lin(runs_bf16=True),
        lin(dtype=torch.bfloat16),
        None,
    ):
        assert not sp._native_mxfp8(not_native)
    assert sp._layer_supports_sp(layer()) and sp._layer_supports_sp(layer(moe=True))
    assert not sp._layer_supports_sp(layer(native=False))
    per_token = layer()
    per_token.self_attn.attn.per_token_fp8_out = True
    assert not sp._layer_supports_sp(per_token)


@pytest.mark.parametrize(
    "env, tp, pp, dp, wanted",
    [
        (True, 4, 1, 1, True),
        (False, 4, 1, 1, False),
        (True, 1, 1, 1, False),
        (True, 4, 2, 1, False),
        (True, 4, 1, 2, False),
    ],
)
def test_sp_only_under_pure_tp(load, sp, monkeypatch, env, tp, pp, dp, wanted):
    cfg = NS(parallel_config=NS(data_parallel_size=dp))
    monkeypatch.setattr(sp.envs, "ATOM_M3_SP_PREFILL", env, raising=False)
    monkeypatch.setattr(sp, "_tp", lambda: NS(world_size=tp))
    monkeypatch.setattr(
        load("aiter.dist.parallel_state"), "get_pp_group", lambda: NS(world_size=pp)
    )
    monkeypatch.setattr(load("atom.config"), "get_current_atom_config", lambda: cfg)
    assert bool(sp._wanted()) is wanted


# The unsharded o_proj copy


def test_no_unsharded_o_proj_where_sp_cannot_run(sp, monkeypatch):
    never = mock.Mock(side_effect=AssertionError("o_proj_full allocated"))
    linear = types.ModuleType("atom.model_ops.linear")
    linear.ReplicatedLinear = never
    monkeypatch.setitem(sys.modules, "atom.model_ops.linear", linear)
    monkeypatch.setattr(sp.envs, "ATOM_M3_SP_PREFILL", True, raising=False)
    mixed = Model([layer(), layer(moe=True, native=False)])
    with mock.patch.object(sp, "_wanted", lambda: True):
        sp.attach_o_proj_full(mixed, None)
    off = Model([layer()])
    with mock.patch.object(sp, "_wanted", lambda: False):
        sp.attach_o_proj_full(off, None)
    for model in (mixed, off):
        assert model._sp_supported is False
        assert not any(hasattr(x.self_attn, "o_proj_full") for x in model.layers)
    assert sp.plan_model(off, 8192) is None
    never.assert_not_called()


class FakeReplicatedLinear(torch.nn.Module):
    """`ReplicatedLinear`'s parameter and loader: the whole weight."""

    def __init__(self, in_features, out_features, bias, quant_config, prefix):
        super().__init__()
        self.args = (in_features, out_features, bias, quant_config, prefix)
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )
        self.weight.weight_loader = lambda param, loaded: param.data.copy_(loaded)


class Attention(torch.nn.Module):
    """An attention whose o_proj loads the first 2 of 4 input columns, its TP shard."""

    def __init__(self, i, extra_param=False, loader=True):
        super().__init__()
        o_proj = torch.nn.Module()
        o_proj.prefix = f"model.layers.{i}.self_attn.o_proj"
        o_proj.weight = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
        if loader:
            o_proj.weight.weight_loader = lambda param, loaded: param.data.copy_(
                loaded[:, :2]
            )
        if extra_param:
            o_proj.input_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.o_proj, self.qkv_proj = o_proj, lin()
        self.attn = NS(per_token_fp8_out=False)
        self.total_num_heads, self.head_dim, self.hidden_size = 2, 2, 4


class Layer(torch.nn.Module):
    def __init__(self, i, **attention_kw):
        super().__init__()
        self.self_attn = Attention(i, **attention_kw)
        self.is_moe_layer = True


class M3(torch.nn.Module):
    def __init__(self, num_layers, **attention_kw):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            Layer(i, **attention_kw) for i in range(num_layers)
        )
        self.start_layer, self.end_layer = 0, num_layers


@pytest.fixture
def sp_everywhere(sp, monkeypatch):
    linear = types.ModuleType("atom.model_ops.linear")
    linear.ReplicatedLinear = FakeReplicatedLinear
    monkeypatch.setitem(sys.modules, "atom.model_ops.linear", linear)
    monkeypatch.setattr(sp, "_wanted", lambda: True)
    monkeypatch.setattr(sp, "_native_mxfp8", lambda lin: lin is not None)
    return sp


def test_unsharded_o_proj_loads_with_o_proj_and_is_recorded_loaded(load, sp_everywhere):
    model = M3(2)
    sp_everywhere.attach_o_proj_full(model, "qc")
    assert model._sp_supported is True
    params = dict(model.named_parameters())
    dispatcher = load("atom.model_loader.weight_dispatch").WeightDispatcher(
        model=model,
        params_dict=params,
        hf_config=None,
        prefix="",
        spec_decode=False,
        submit=lambda loader, *args: loader(*args),
        staging_pool=None,
        batching_enabled=False,
        batching_excluded=None,
        default_weight_loader=None,
        packed_modules_mapping={},
        expert_index={},
        expert_weight_prefixes=(),
        has_expert_mapping=False,
        detect_fused_expert_fn=None,
        get_fused_expert_mapping_fn=None,
        load_fused_expert_weights_fn=None,
    )
    for i, x in enumerate(model.layers):
        attn = x.self_attn
        prefix = f"model.layers.{i}.self_attn.o_proj"
        assert attn.o_proj_full.args == (4, 4, False, "qc", prefix)
        w = torch.arange(16.0).view(4, 4) + i
        dispatcher.dispatch(
            f"{prefix}.weight", f"layers.{i}.self_attn.o_proj.weight", w
        )
        assert torch.equal(attn.o_proj.weight, w[:, :2])
        assert torch.equal(attn.o_proj_full.weight, w)
    assert dispatcher.loaded_weights_record == set(params)
    assert dispatcher.dropped_ckpt_keys == []


@pytest.mark.parametrize(
    "attention_kw",
    [{"extra_param": True}, {"loader": False}],
    ids=["copy-lacks-a-parameter", "no-weight-loader"],
)
def test_unsharded_o_proj_refuses_what_it_cannot_load(sp_everywhere, attention_kw):
    with pytest.raises(ValueError, match="o_proj"):
        sp_everywhere.attach_o_proj_full(M3(1, **attention_kw), None)


# The FlyDSL MXFP4 MoE chain


def fake_moe():
    method = NS(
        quant_type=QuantType.per_1x32,
        is_guinterleave=False,
        use_triton=False,
        hidden_pad=0,
    )
    experts = NS(
        quant_method=method,
        w13_bias=None,
        w2_bias=None,
        ep_size=1,
        swiglu_limit=7.0,
        w13_weight=torch.empty((8, 1536, 1024), dtype=torch.uint8),
        w2_weight=torch.empty((8, 2048, 384), dtype=torch.uint8),
    )
    return NS(
        experts=experts,
        shared_experts=None,
        fuse_shared_experts=False,
        gate=NS(weight=torch.empty(8, 2048)),
    )


@pytest.fixture
def chain_module(sp, monkeypatch, set_gfx):
    """aiter#8699's prefill module, taking hidden 2048 / intermediate 768, on gfx950."""
    fake = types.ModuleType(CHAIN)
    fake.supports_shapes = lambda hidden, inter: (hidden, inter) == (2048, 768)
    monkeypatch.setitem(sys.modules, CHAIN, fake)
    monkeypatch.setattr(sp.envs, "ATOM_M3_SP_FLYDSL_MOE", True, raising=False)
    set_gfx("gfx950")
    return fake


@pytest.mark.parametrize("env", [False, True])
def test_chain_is_off_without_aiters_prefill_module(sp, monkeypatch, set_gfx, env):
    set_gfx("gfx950")
    monkeypatch.setattr(sp.envs, "ATOM_M3_SP_FLYDSL_MOE", env, raising=False)
    monkeypatch.delitem(sys.modules, CHAIN, raising=False)
    real_import = builtins.__import__

    def no_chain(name, *args, **kwargs):
        if name.startswith("aiter.ops.flydsl.kernels.minimax_m3_moe_prefill"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_chain)
    assert sp._flydsl_moe_ok(fake_moe()) is False


def _set(path, value):
    def change(moe):
        *owners, attr = path.split(".")
        target = moe
        for owner in owners:
            target = getattr(target, owner)
        setattr(target, attr, value)

    return change


@pytest.mark.parametrize(
    "change",
    [
        _set("experts.quant_method.is_guinterleave", True),
        _set("experts.quant_method.use_triton", True),
        _set("experts.quant_method.hidden_pad", 64),
        _set("experts.quant_method.quant_type", QuantType.per_Token),
        _set("experts.w13_bias", torch.zeros(1)),
        _set("experts.ep_size", 2),
        _set("experts.swiglu_limit", None),
        _set("fuse_shared_experts", True),
        _set("experts.w13_weight", torch.empty((9, 1536, 1024), dtype=torch.uint8)),
        _set("experts.w2_weight", torch.empty((8, 3072, 384), dtype=torch.uint8)),
    ],
    ids=[
        "interleaved-gate-up",
        "triton-layout",
        "hidden-pad",
        "per-token-quant",
        "bias",
        "expert-parallel",
        "swiglu-limit",
        "fused-shared-expert",
        "expert-count",
        "hidden-size",
    ],
)
def test_chain_takes_only_flydsl_fused_moes_two_slab_layout(sp, chain_module, change):
    assert sp._flydsl_moe_ok(fake_moe()) is True
    moe = fake_moe()
    change(moe)
    assert sp._flydsl_moe_ok(moe) is False


@pytest.mark.parametrize("gfx", ["gfx942", "gfx1250"])
def test_chain_needs_gfx950(sp, chain_module, set_gfx, gfx):
    set_gfx(gfx)
    assert sp._flydsl_moe_ok(fake_moe()) is False


@pytest.fixture
def block_m_variants(sp, monkeypatch):
    """aiter#8699's BLOCK_M choice: 128 below 16384 tokens, 256 from it."""
    fake = types.ModuleType(CHAIN)
    fake.MIN_PREFILL_TOKENS, fake.BM256_FROM_TOKENS = 3072, 16384
    fake.block_m_for = lambda n: 256 if n >= 16384 else 128
    monkeypatch.setitem(sys.modules, CHAIN, fake)
    monkeypatch.setattr(sp, "_chain_warm", set())


SHAPE = (2048, 768, 8, 4)


def test_chain_warm_up_runs_each_block_m_once(sp, block_m_variants):
    ran = []

    def chain(n, out):
        assert out is None
        ran.append(n)

    sp._warm_chain(chain, SHAPE, 5120)
    sp._warm_chain(chain, SHAPE, 5120)
    # BLOCK_M 256 waits for a step with the rows for it
    assert ran == [3072]
    sp._warm_chain(chain, SHAPE, 20480)
    sp._warm_chain(chain, SHAPE, 65536)
    assert ran == [3072, 16384]
    # another shape compiles its own kernels
    sp._warm_chain(chain, (3072, 768, 8, 4), 20480)
    assert ran == [3072, 16384, 3072, 16384]


def test_chain_warm_up_retries_a_variant_that_failed(sp, block_m_variants):
    def broken(n, out):
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError):
        sp._warm_chain(broken, SHAPE, 5120)
    ran = []
    sp._warm_chain(lambda n, out: ran.append(n), SHAPE, 5120)
    assert ran == [3072]


# Which steps run SP, and their token shards


@pytest.fixture
def tp4(sp, monkeypatch):
    monkeypatch.setattr(sp, "_tp", lambda: NS(world_size=4, rank_in_group=1))
    monkeypatch.setattr(sp, "SP_MIN_TOKENS", 4096)
    monkeypatch.setattr(sp, "SP_MIN_UBATCH_TOKENS", 8192)


def test_plan_thresholds_and_shard(sp, tp4, monkeypatch, run_as):
    run_as(None)
    assert sp.plan(4095) is None and sp.plan(65537) is None
    s = sp.plan(5000)
    assert (s.num_tokens, s.rows, s.padded, s.rank) == (5000, 1280, 5120, 1)
    monkeypatch.setattr(sp, "_tp", lambda: NS(world_size=1, rank_in_group=0))
    assert sp.plan(8192) is None


def test_seq_shard_pads_the_last_rank(sp):
    x = torch.arange(10.0).unsqueeze(1)
    assert sp.SeqShard(10, 3, 4, 1).shard(x).flatten().tolist() == [3, 4, 5]
    last = sp.SeqShard(num_tokens=10, rows=3, world=4, rank=3)
    assert last.shard(x).flatten().tolist() == [9, 0, 0]
    assert last.pad(x).flatten().tolist() == list(range(10)) + [0, 0]


def tbo_ring(*running_tokens):
    ctxs = [
        NS(forward_context=NS(context=NS(running_tokens=n))) for n in running_tokens
    ]
    for i, ctx in enumerate(ctxs):
        ctx.partner = ctxs[(i + 1) % len(ctxs)]
    return ctxs


@pytest.mark.parametrize(
    "tokens, runs_sp",
    [
        ((9000, 9000), True),
        ((9000, 7000), False),
        ((70000, 30000), False),
        ((9000, 9000, 7000), False),
    ],
    ids=["both-large", "one-below-the-minimum", "one-past-the-maximum", "three"],
)
def test_tbo_micro_batches_decide_together(sp, tp4, run_as, tokens, runs_sp):
    for ctx in tbo_ring(*tokens):
        run_as(ctx)
        n = ctx.forward_context.context.running_tokens
        assert (sp.plan(n) is not None) is runs_sp


def test_tbo_decision_needs_every_micro_batch_context(sp, tp4, run_as):
    ring = tbo_ring(9000, 9000)
    ring[1].forward_context = None
    run_as(ring[0])
    with pytest.raises(AssertionError):
        sp.plan(9000)
