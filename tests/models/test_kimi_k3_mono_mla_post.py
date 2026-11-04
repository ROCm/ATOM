# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's MLA post stage (``ATOM_MONO_MLA_POST``).

An MLA layer's v-up BMM, g_proj, sigmoid and mul run as the first stage of its
K2 launch. These tests pin the plain-Python parts: the stage's task geometry,
the bind-time refusals, the elementwise tail's arithmetic against torch on a CPU,
and the flag defaults. Whether the stage is right and faster on a GPU (and what
the changed GEMM summation order does to accuracy) needs a GPU.
"""

import inspect
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono import weights
from atom.models.kimi_k3.mono.config import HIDDEN, Dims
from atom.models.kimi_k3.mono.kernels.stages import mla_post_plan as pl
from atom.mono.plan.execution import BLOCKS, WAVES
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs

DIMS = Dims(8)
FP8 = torch.float8_e4m3fn


# --------------------------------------------------------------- task geometry
def test_post_tasks_cover_the_core_once():
    assert pl.PT == 96
    rows = [r for j in range(pl.PT) for r in range(*_span(pl.task_rows(j)[0]))]
    assert sorted(rows) == list(range(pl.PROJ))


def _span(row0):
    return row0, row0 + pl.ROWS


def test_each_task_reads_one_16_row_tile_of_one_heads_w_v():
    seen = set()
    for j in range(pl.PT):
        row0, h, vr0 = pl.task_rows(j)
        assert 0 <= h < pl.HEADS and 0 <= vr0 < pl.VH and vr0 % pl.ROWS == 0
        # core column row0 + r is head h's value column vr0 + r
        assert row0 == h * pl.VH + vr0
        seen.add((h, vr0))
    assert len(seen) == pl.PT == pl.HEADS * pl.VH // pl.ROWS


# K2's CTA roles (``kernels.layer`` / ``stages.moe``, FlyDSL modules: restated
# here, cross-checked by ``test_placement_constants_match_k2`` where FlyDSL is)
NA, O0, OTASKS, E0 = 56, 56, 448, 160
G_TASKS, R_TASKS, L_TASKS = 96, 28, 56


def _o_tasks(bid):
    """CTA bid's o_proj task count (``layer.o_slot``)."""
    slot = (bid + BLOCKS - O0) % BLOCKS
    return sum(slot + BLOCKS * j < OTASKS for j in range(2))


def test_tasks_sit_on_distinct_ctas_inside_the_grid():
    ctas = [pl.task_cta(j) for j in range(pl.PT)]
    assert len(set(ctas)) == pl.PT
    assert min(ctas) == 0 and max(ctas) == BLOCKS - 1
    assert [pl.cta_task(c) for c in ctas] == list(range(pl.PT))
    assert sum(pl.cta_task(b) >= 0 for b in range(BLOCKS)) == pl.PT
    with pytest.raises(ValueError):
        pl.task_cta(pl.PT)


def test_tasks_avoid_the_shared_and_router_gemv_ctas():
    """No post task on a shared gate/up CTA, nor on a router CTA (both on the
    path to the MoE); every one-o_proj-task CTA takes one."""
    for j in range(pl.PT):
        c = pl.task_cta(j)
        assert not O0 <= c < O0 + G_TASKS
        assert not E0 <= c < E0 + R_TASKS
    single = [b for b in range(BLOCKS) if _o_tasks(b) == 1]
    assert len(single) == 64
    assert all(pl.cta_task(b) >= 0 for b in single)
    # the AttnRes CTAs keep the first tasks (``task_cta(j) == j``)
    assert [pl.task_cta(j) for j in range(NA)] == list(range(NA))


def test_placement_constants_match_k2():
    pytest.importorskip("flydsl", reason="the K2 modules are FlyDSL")
    from atom.models.kimi_k3.mono.kernels import layer as k2
    from atom.models.kimi_k3.mono.kernels.stages import moe as k2b

    assert (k2.O0, k2.OTASKS, k2.E0) == (O0, OTASKS, E0)
    assert (k2b.G_TASKS, k2b.R_TASKS, k2b.L_TASKS) == (G_TASKS, R_TASKS, L_TASKS)
    for b in range(BLOCKS):
        slot = k2.o_slot(b)
        assert sum(slot + BLOCKS * j < OTASKS for j in range(2)) == _o_tasks(b)


def test_width_bound_is_inside_mono():
    from atom.models.kimi_k3.mono.config import MAX_ROWS, TOKEN_GROUP

    assert 1 <= pl.MAX_ROWS <= MAX_ROWS
    # one token group: the stage's 16-token tiles are the o_proj groups
    assert pl.MAX_ROWS <= TOKEN_GROUP


@pytest.mark.parametrize("s", [1, 2, 6, 8, 12, 16])
def test_x_slabs_cover_k_once_and_fit(s):
    budget = 132 * 1024
    slabs = pl.slabs(s, budget)
    assert slabs[0][0] == 0 and slabs[-1][1] == pl.G_GROUPS
    assert all(a[1] == b[0] for a, b in pairwise(slabs))
    widest = max(g1 - g0 for g0, g1 in slabs) * pl.GROUP_COLS
    assert s * (widest + pl.XPAD) <= budget
    assert pl.lds_words(s, budget) * 4 >= s * (widest + pl.XPAD)


def test_slabs_split_when_the_rows_do_not_fit():
    assert len(pl.slabs(16, 132 * 1024)) == 1
    many = pl.slabs(16, 16 * (2 * pl.GROUP_COLS + pl.XPAD))
    assert len(many) > 1
    with pytest.raises(AssertionError):
        pl.slabs(16, 16 * pl.GROUP_COLS)


def test_geometry_matches_the_kernel_constants():
    assert pl.GROUP_COLS == WAVES * pl.KCH
    assert pl.HIDDEN == HIDDEN
    assert pl.PROJ == DIMS.proj
    assert pl.V_CH * pl.KCH == pl.LORA


# ---------------------------------------------------------- the elementwise tail
def test_gate_tail_is_torchs_bf16_sigmoid_then_mul():
    """core = bf16(av * bf16(1 / (1 + exp(-gate)))): what the stage computes
    after the GEMMs, against torch's bf16 ``av * sigmoid(gate)`` on a CPU."""
    g = torch.Generator().manual_seed(0)
    gate = (torch.randn(64, 1536, generator=g) * 4).to(torch.bfloat16)
    av = torch.randn(64, 1536, generator=g).to(torch.bfloat16)
    ref = av * torch.sigmoid(gate)
    sg = (1.0 / (1.0 + torch.exp(-gate.float()))).to(torch.bfloat16)
    mine = (av.float() * sg.float()).to(torch.bfloat16)
    assert torch.equal(mine.view(torch.int16), ref.view(torch.int16))


# ------------------------------------------------------------------ bind refusals
def _param(t, shuffled=True):
    p = torch.nn.Parameter(t, requires_grad=False)
    p.is_shuffled = shuffled
    return p


def _forward_impl(q, k_nope, k_rope, positions=None, q_scale=None, latent_out=False):
    raise NotImplementedError


def _layer(**over):
    impl = SimpleNamespace(
        W_V=torch.zeros(12, 128, 512, dtype=torch.uint8).view(FP8),
        W_V_scale=torch.ones(1),
        forward_impl=_forward_impl,
        dcp_world_size=1,
        num_heads=12,
        kv_lora_rank=512,
        v_head_dim=128,
    )
    g_proj = SimpleNamespace(
        weight=_param(torch.zeros(1536, HIDDEN, dtype=torch.uint8).view(FP8)),
        weight_scale=torch.ones(1536),
        quant_type=SimpleNamespace(name="per_Token"),
        bias=None,
    )
    sa = SimpleNamespace(
        attn=SimpleNamespace(impl=impl),
        forward_latent=lambda *a: None,
        fuse_input_norm_quant=True,
        g_proj=g_proj,
    )
    for k, v in over.items():
        target = {"impl": impl, "g_proj": g_proj, "sa": sa}[k.split("__")[0]]
        setattr(target, k.split("__")[1], v)
    return SimpleNamespace(self_attn=sa)


def test_bind_accepts_the_plain_fp8_decode_layer():
    post = weights.bind_mla_post(_layer(), 3, DIMS)
    assert post.layer_id == 3
    assert tuple(post.w_v.shape) == (12, 128, 512) and post.w_v_s.shape == (1,)
    assert tuple(post.w_g.shape) == (1536, HIDDEN) and post.w_g_s.numel() == 1536


@pytest.mark.parametrize(
    "over, why",
    [
        ({"impl__dcp_world_size": 2}, "decode context parallel"),
        ({"impl__is_sparse_mla": True}, "is_sparse_mla"),
        ({"impl__use_seg_mla": True}, "use_seg_mla"),
        ({"impl__num_heads": 16}, "MLA heads"),
        ({"impl__kv_lora_rank": 256}, "kv_lora_rank"),
        ({"impl__W_V": torch.zeros(12, 128, 512, dtype=torch.bfloat16)}, "W_V"),
        ({"impl__W_V_scale": torch.ones(2)}, "W_V_scale"),
        ({"sa__fuse_input_norm_quant": False}, "input RMSNorm"),
        (
            {"g_proj__weight": _param(torch.zeros(1536, HIDDEN, dtype=torch.bfloat16))},
            "g_proj",
        ),
        (
            {
                "g_proj__weight": _param(
                    torch.zeros(1536, HIDDEN, dtype=torch.uint8).view(FP8), False
                )
            },
            "not preshuffled",
        ),
        ({"g_proj__quant_type": SimpleNamespace(name="per_1x128")}, "per-token"),
        ({"g_proj__weight_scale": torch.ones(7)}, "weight_scale"),
        ({"g_proj__bias": torch.zeros(1)}, "bias"),
    ],
)
def test_bind_refuses_what_the_stage_cannot_reproduce(over, why):
    with pytest.raises(MonoUnsupported, match=why):
        weights.bind_mla_post(_layer(**over), 3, DIMS)


def test_bind_refuses_an_mla_without_latent_out():
    def old(q, k_nope, k_rope, positions=None, q_scale=None):
        return None

    layer = _layer(impl__forward_impl=old)
    with pytest.raises(MonoUnsupported, match="latent_out"):
        weights.bind_mla_post(layer, 3, DIMS)


def test_attention_hooks_exist():
    try:
        from atom.model_ops.attention_mla import MLAAttention
    except ImportError as e:  # the AITER / triton runtime, absent or without a GPU
        pytest.skip(f"attention_mla does not import here: {e!r}")

    for fn in (MLAAttention.forward_impl, MLAAttention._forward_decode):
        assert "latent_out" in inspect.signature(fn).parameters


# ---------------------------------------------------------------------- the flags
def test_flags_default_off(monkeypatch):
    for name in ("ATOM_MONO_MLA_POST", "ATOM_MONO_MLA_POST_CHECK"):
        monkeypatch.delenv(name, raising=False)
        assert getattr(envs, name) is False
        monkeypatch.setenv(name, "1")
        assert getattr(envs, name) is True
