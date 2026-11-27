# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""DeepSeek-V4.1 mono on gfx950 under in-place weight updates, on stand-ins (no GPU, no AITER, no FlyDSL):

- the layer kernel's launches read every weight in place: the compressor its bf16 norms, the draft's context write
  each stage's own ``wkv`` rows (not DSpark's fused copy), the Engram gate ``q_weight`` and ``k_weight`` (not
  ``EngramOp.gate_weight``); its runner's audit refuses, under a runner that takes weight updates, a weight it holds
  that is neither a parameter nor registered, and its draft runner one taking DSpark's cached noise row;
- the hi hook registers what it packs (the aliased plain tensors excluded) and rebuilds it in place, its draft
  context write reads the draft stages' own ``wkv`` packs, its compressor its norms in place, and nothing aliases a
  buffer ATOM derives at load (``EngramOp.gate_weight``);
- the routing wrappers share one registry and route around the model under its own names.
"""

import types
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from conftest import gpu_stack_import
from torch import nn

from atom.models.deepseek_v41.mono.weight_refresh import (
    DerivedWeights,
    ParameterStorage,
)
from atom.mono.runtime.consensus import MonoUnsupported

FP8, E8M0, BF16 = torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.bfloat16
HIDDEN, KV_DIM, Q_LORA = 5120, 512, 1280


@pytest.fixture(scope="module")
def load():
    """The hi hook's host modules: against the stand-ins where AITER cannot run."""
    with atom_without_aiter() as load:
        yield load


@pytest.fixture
def lk():
    """The layer kernel's modules, which import its FlyDSL kernels: skipped without the GPU stack (as the other
    layer-kernel tests are)."""
    return gpu_stack_import


def _fp8(rows: int, cols: int, fill: int = 0):
    w = torch.full((rows, cols), fill, dtype=torch.uint8).view(FP8)
    w.is_shuffled = True
    return nn.Parameter(w, requires_grad=False)


def _e8m0(rows: int, cols: int):
    return nn.Parameter(
        torch.full((rows, cols), 127, dtype=torch.uint8).view(E8M0), False
    )


# ---------------------------------------------------------------- the layer kernel (LK)


class _Stage(nn.Module):
    """A draft stage's attention: ``wqkv_a`` = [wq_a; wkv] and the KV rows' view (``wkv_shard``)."""

    def __init__(self, layer_id: int, rmsnorm) -> None:
        super().__init__()
        self.wqkv_a = nn.Module()
        self.wqkv_a.weight = _fp8(Q_LORA + KV_DIM, HIDDEN, layer_id)
        self.wqkv_a.weight_scale = _e8m0((Q_LORA + KV_DIM) // 32, HIDDEN // 32)
        self.kv_norm = rmsnorm(KV_DIM, 1e-6)
        self.kv_norm.weight.data = self.kv_norm.weight.data.to(BF16)
        self.spec = SimpleNamespace(layer_id=layer_id)
        rows = self.wqkv_a.weight.data.narrow(0, Q_LORA, KV_DIM)
        rows.is_shuffled = True
        scale = self.wqkv_a.weight_scale.data.narrow(0, Q_LORA // 32, KV_DIM // 32)
        # a view, as ``LinearBase.shard_view`` makes it: kept out of ``_modules``
        object.__setattr__(
            self,
            "wkv_shard",
            SimpleNamespace(
                weight=rows, weight_scale=scale, native_a8_group_rows=32, bias=None
            ),
        )


class _Draft(nn.Module):
    """DSpark's draft as the context write reads it: without the fused caches (any use would raise)."""

    def __init__(self, rmsnorm, dc) -> None:
        super().__init__()
        self.stages = nn.ModuleList(_Stage(40 + s, rmsnorm) for s in range(3))
        stage0 = nn.Module()
        stage0.main_proj = nn.Module()
        stage0.main_proj.weight = _fp8(HIDDEN, dc.K_PROJ)
        stage0.main_proj.weight_scale = _e8m0(HIDDEN // 32, dc.K_PROJ // 32)
        stage0.main_proj.bias = None
        stage0.main_norm = rmsnorm(HIDDEN, 1e-6)
        stage0.main_norm.weight.data = stage0.main_norm.weight.data.to(BF16)
        self.mtp = nn.ModuleList([stage0])
        cache = torch.zeros(64, dc.HALF, dtype=BF16)
        self.rope = SimpleNamespace(cos_cache=cache, sin_cache=cache.clone())

    @property
    def context_layers(self):
        return list(self.stages)

    def _context_kv_fusion(self):
        raise AssertionError("mono read DSpark's fused context KV copy")

    def _draft_kv_tail_fusion(self):
        raise AssertionError("mono read DSpark's stacked KV tail")


def test_lk_draft_context_reads_each_stages_own_wkv(lk, monkeypatch):
    launch = lk("atom.models.deepseek_v41.mono.draft_ctx_launch")
    dc = lk("atom.models.deepseek_v41.mono.kernels.draft_ctx")
    rmsnorm = lk("atom.model_ops.layernorm").RMSNorm
    draft = _Draft(rmsnorm, dc)
    monkeypatch.setattr(
        launch, "uncached_bytes", lambda n, dev: torch.zeros(n, dtype=torch.uint8)
    )

    assert launch.MonoDraftContext._refusal(draft) is None
    ctx = launch.MonoDraftContext(draft, 8, 6, "cpu")

    for s, stage in enumerate(draft.stages):
        w, ws = ctx.wkv[s]
        # the KV rows of the stage's own wqkv_a, in place: one contiguous run of the stored layout
        assert w.data_ptr() == stage.wqkv_a.weight.data_ptr() + Q_LORA * HIDDEN
        assert (
            ws.data_ptr()
            == stage.wqkv_a.weight_scale.data_ptr() + Q_LORA // 32 * HIDDEN // 32
        )
    storage = ParameterStorage(draft)
    assert all(t in storage for t in ctx.weight_pointers().values())
    assert set(dc.ABI.names) >= {"wk0", "wk1", "wk2", "wks0", "wks1", "wks2"}
    assert "wk" not in dc.ABI.names


def test_lk_draft_context_refuses_stages_of_another_layout(lk):
    launch = lk("atom.models.deepseek_v41.mono.draft_ctx_launch")
    dc = lk("atom.models.deepseek_v41.mono.kernels.draft_ctx")
    draft = _Draft(lk("atom.model_ops.layernorm").RMSNorm, dc)
    draft.stages[1].wkv_shard.weight.is_shuffled = False
    assert "stage 1 wkv" in launch.MonoDraftContext._refusal(draft)


def test_lk_engram_gate_reads_q_and_k_in_place(lk):
    eg = lk("atom.models.deepseek_v41.mono.kernels.engram")
    launch = lk("atom.models.deepseek_v41.mono.engram_launch")
    assert {"gate_q", "gate_k"} <= set(eg.ABI.names) and "gate_w" not in eg.ABI.names
    op = SimpleNamespace(
        wkv=SimpleNamespace(weight=torch.zeros(4), weight_scale=torch.zeros(4)),
        q_weight=torch.ones(4, HIDDEN, dtype=BF16),
        k_weight=torch.ones(4, HIDDEN, dtype=BF16),
        gate_weight=torch.ones(4, HIDDEN),
    )
    engram = launch.MonoEngram.__new__(launch.MonoEngram)
    engram.blocks = {1: SimpleNamespace(engram=op)}
    got = engram.weight_pointers()
    assert (
        got["engram 1 q_weight"] is op.q_weight
        and got["engram 1 k_weight"] is op.k_weight
    )
    assert not any(t is op.gate_weight for t in got.values())


def test_lk_compressor_reads_its_norms_in_place(lk):
    launch = lk("atom.models.deepseek_v41.mono.compress_launch")
    ip = lk("atom.models.deepseek_v41.mono.index_plan")
    comp = SimpleNamespace(
        ratio=2,
        wkv_gate=SimpleNamespace(weight=torch.zeros(2 * KV_DIM, HIDDEN, dtype=BF16)),
        ape=torch.zeros(2, KV_DIM),
        norm=SimpleNamespace(weight=torch.ones(KV_DIM, dtype=BF16), eps=1e-6),
    )
    indexer = SimpleNamespace(
        wk=SimpleNamespace(weight=torch.zeros(ip.DIM, KV_DIM, dtype=BF16)),
        k_norm=SimpleNamespace(weight=torch.ones(ip.DIM, dtype=BF16), eps=1e-6),
    )
    w = launch.compress_weights(SimpleNamespace(compressor=comp, indexer=indexer))
    assert w.norm_w is comp.norm.weight and w.k_norm_w is indexer.k_norm.weight
    # an f32 norm (a model that keeps it so) is not one the launch reads in place: refused
    comp.norm.weight = comp.norm.weight.float()
    assert (
        launch.compress_weights(SimpleNamespace(compressor=comp, indexer=indexer))
        is None
    )


def _lk_runner(runner_module, layer_tensors, derived, model):
    runner = runner_module.V41MonoDecodeRunner.__new__(
        runner_module.V41MonoDecodeRunner
    )
    runner.layers = [SimpleNamespace(layer_id=0, tensors=layer_tensors)]
    runner.derived, runner._read_models = derived, (model, None)
    return runner


def test_lk_runner_audit_passes_in_place_weights_and_refuses_a_copy(lk, monkeypatch):
    runner_module = lk("atom.models.deepseek_v41.mono.runner")
    monkeypatch.setattr(runner_module, "group_agree", lambda ok, group: ok)
    model = nn.Linear(4, 4)

    passing = _lk_runner(
        runner_module, dict(model.named_parameters()), DerivedWeights(True), model
    )
    passing._audit_agreed(None)
    assert passing.derived.refusal() is None

    copied = {"weight": model.weight.detach().float().clone()}
    refused = _lk_runner(runner_module, copied, DerivedWeights(True), model)
    with pytest.raises(MonoUnsupported, match="layer 0 weight"):
        refused._audit_agreed(None)


def test_lk_draft_runner_refuses_the_cached_noise_row(lk):
    draft_runner = lk("atom.models.deepseek_v41.mono.draft_runner")
    runner = draft_runner.DraftMonoRunner.__new__(draft_runner.DraftMonoRunner)
    runner.derived, runner.embed_peer = DerivedWeights(), None
    runner._audit()
    assert "_noise_row" in runner.derived.refusal()


# ---------------------------------------------------------------- the hi hook


def _hi_self(hi_runner, *models):
    fake = SimpleNamespace(
        _in_params=ParameterStorage(*models), derived=DerivedWeights(), _weight_sets=[]
    )
    fake._register_packed = types.MethodType(hi_runner.HiLayers._register_packed, fake)
    return fake


def test_hi_registers_what_it_packs_and_rebuilds_it_in_place(load):
    hi_runner = load("atom.models.deepseek_v41.mono.hi_runner")
    block = nn.Linear(8, 8)
    fake = _hi_self(hi_runner, block)

    def pack():
        return {
            "norm.g": block.bias.detach(),
            "w.packed": (block.weight.t() * 2).contiguous(),
        }

    packed = pack()
    fake._register_packed("hi layer 0", packed, tuple(block.parameters()), pack)
    (entry,) = fake.derived.entries
    assert set(entry.dests) == {"w.packed"}  # the in-place one is not a copy
    address = packed["w.packed"].data_ptr()
    with torch.no_grad():
        block.weight.fill_(1.5)
    assert fake.derived.rebuild([block.weight]).rebuilt == ("hi layer 0",)
    assert packed["w.packed"].data_ptr() == address
    assert torch.equal(packed["w.packed"], torch.full((8, 8), 3.0))
    # the rebuild ran under the install's torch defaults, restored after
    assert torch.get_default_dtype() == torch.float32
    # a shared destination is registered once, by its first set
    fake._register_packed(
        "hi draft head", {"mk.w": packed["w.packed"]}, (block.weight,), pack
    )
    assert len(fake.derived.entries) == 1
    weights = {f"{n} {k}": t for n, ws in fake._weight_sets for k, t in ws.items()}
    assert fake.derived.audit("hi", weights, block.parameters()) is None


def test_hi_draft_context_reads_the_stages_own_wkv_packs(load):
    hi_runner = load("atom.models.deepseek_v41.mono.hi_runner")
    hi_draft = load("atom.models.deepseek_v41.mono.hi_draft")
    draft = nn.Module()
    draft.mtp = nn.ModuleList([nn.Module()])
    draft.mtp[0].main_proj = nn.Linear(4, 4)
    draft.mtp[0].main_norm = nn.LayerNorm(4)
    stages = [
        SimpleNamespace(wqkv_a=nn.Linear(4, 4), kv_norm=nn.LayerNorm(4))
        for _ in range(3)
    ]
    draft.context_layers = stages
    fake = _hi_self(
        hi_runner, draft, *(nn.ModuleList([s.wqkv_a, s.kv_norm]) for s in stages)
    )
    fake.D = SimpleNamespace(STAGES=3)
    fake.dr_w = [
        {"wkv.w": torch.full((16,), s), "wkv.ws": torch.full((2,), s)} for s in range(3)
    ]
    for s, packed in enumerate(fake.dr_w):  # the stages' packs: registered first
        fake.derived.register(f"stage {s}", (stages[s].wqkv_a.weight,), packed, print)
    fake.cx_w = {
        "cx_main_proj.w": torch.zeros(8),
        "cx.main_norm.g": draft.mtp[0].main_norm.weight,
    }
    for s in range(3):
        fake.cx_w[f"cx_wkv{s}.w"] = fake.dr_w[s]["wkv.w"].clone()
        fake.cx_w[f"cx_wkv{s}.ws"] = fake.dr_w[s]["wkv.ws"].clone()
    fake._pack_ctx = lambda d: dict(fake.cx_w)

    freed = hi_draft.HiDraft._ctx_reads_stage_wkv(fake, draft)

    assert freed == sum(
        t.numel() * t.element_size() for ws in fake.dr_w for t in ws.values()
    )
    for s in range(3):
        assert fake.cx_w[f"cx_wkv{s}.w"] is fake.dr_w[s]["wkv.w"]
    ctx = fake.derived.entries[-1]
    assert ctx.name == "hi draft ctx" and set(ctx.dests) == {"cx_main_proj.w"}


def test_hi_aliases_parameters_only(load):
    hw = load("atom.models.deepseek_v41.mono.hi_weights")
    assert "eng.gw" not in hw.PLAIN
    assert {"comp.norm", "comp.knorm"} <= set(hw.PLAIN)
    block = nn.Module()
    block.engram = nn.Module()
    block.engram.register_buffer("gate_weight", torch.ones(4, 8))
    block.attn_norm = nn.Module()
    block.attn_norm.weight = nn.Parameter(torch.ones(8, dtype=BF16), False)
    packed = {"eng.gw": torch.ones(4, 8), "attn_norm.g": torch.ones(8, dtype=BF16)}
    assert hw.alias_plain(block, packed) == 1
    assert packed["attn_norm.g"].data_ptr() == block.attn_norm.weight.data_ptr()
    assert packed["eng.gw"].data_ptr() != block.engram.gate_weight.data_ptr()


def test_hi_compressor_reads_its_norms_in_place(load):
    base = load("atom.models.deepseek_v41.mono.hi.blocks.base")
    comp_mod = load("atom.models.deepseek_v41.mono.hi.blocks.gfx950.compressor")
    program = load("atom.models.deepseek_v41.mono.hi.plan.program")
    Channel, Fmt = base.Channel, base.Fmt
    comp = comp_mod.Compressor(
        "comp", Channel("xab", HIDDEN, Fmt.BF16), 2, Channel("ckvn", KV_DIM, Fmt.BF16), "pos", "rope", "ckv",
        4096, kv_state="comp.kv_state", score_state="comp.score_state",
        index_out=Channel("ikey", comp_mod.IHD, Fmt.BF16),
    )  # fmt: skip
    layer = {
        "attn.compressor.wkv.weight": torch.zeros(KV_DIM, HIDDEN, dtype=BF16),
        "attn.compressor.wgate.weight": torch.zeros(KV_DIM, HIDDEN, dtype=BF16),
        "attn.compressor.norm.weight": torch.ones(KV_DIM, dtype=BF16),
        "attn.indexer.wk.weight": torch.arange(comp_mod.IHD * KV_DIM)
        .view(comp_mod.IHD, KV_DIM)
        .to(BF16),
        "attn.indexer.k_norm.weight": torch.ones(comp_mod.IHD, dtype=BF16),
    }
    out = comp.pack(layer)
    assert (
        out["comp.norm"].data_ptr() == layer["attn.compressor.norm.weight"].data_ptr()
    )
    assert (
        out["comp.knorm"].data_ptr() == layer["attn.indexer.k_norm.weight"].data_ptr()
    )
    assert torch.equal(
        out["comp.aux"], comp_mod.pack_wk(layer["attn.indexer.wk.weight"]).flatten()
    )
    pool = comp.stages(program.Scenario(1, 6, 8))[-1]
    assert {"comp.norm", "comp.aux", "comp.knorm"} <= set(pool.args)
    # without the index keys: the norm alone, no aux
    plain = comp_mod.Compressor(
        "comp", Channel("xab", HIDDEN, Fmt.BF16), 1, Channel("ckvn", KV_DIM, Fmt.BF16), "pos", "rope", "ckv", 4096,
    )  # fmt: skip
    pool = plain.stages(program.Scenario(1, 6, 8))[-1]
    assert "comp.norm" in pool.args and "comp.aux" not in pool.args
    assert set(plain.pack(layer)) == {"comp.wkv", "comp.norm"}


# ---------------------------------------------------------------- the routing wrappers


def test_the_v41_wrapper_answers_the_hooks(lk):
    """``MonoDecodeModel`` answers the update path's model hooks (``weight_updater``) with its registry: the model's
    parameters under the model's own names, the audit's refusal, the rebuild of what changed.
    """
    dispatch = lk("atom.models.deepseek_v41.mono.dispatch")
    model, derived = nn.Linear(2, 2), DerivedWeights(strict=False)
    wrapper = dispatch.MonoDecodeModel(
        model, None, 1024, False, 0, None, None, derived=derived
    )
    assert wrapper.mono_derived is derived
    assert wrapper.weight_sync_target is model
    assert dict(wrapper.weight_sync_target.named_parameters()).keys() == {
        "weight",
        "bias",
    }
    packed = (model.weight.detach() * 2).contiguous()
    derived.register(
        "w x2",
        (model.weight,),
        {"w": packed},
        lambda: packed.copy_(model.weight.detach() * 2),
    )
    with torch.no_grad():
        model.weight.fill_(3.0)
    wrapper.after_weight_update([model.weight])
    assert torch.equal(packed, torch.full((2, 2), 6.0))
    assert wrapper.weight_update_refusal() is None
    derived.audit("path", {"copy": model.weight.detach().clone()}, model.parameters())
    assert "copy" in wrapper.weight_update_refusal()
