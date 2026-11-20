import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    build_packed_finish,
)
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_bytes


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=4
    )


def test_task_pairs_never_straddle_two_weights():
    assert front.ROUTER_TASKS % 2 == 0
    assert (front.ROUTER_TASKS + front.LATENT_TASKS) % 2 == 0
    assert front.SHARED_TASKS % 2 == 0
    assert front.PAIRS == 188


@pytest.mark.parametrize("rows", (1, 16, 17, 40, 48))
def test_replicated_front_is_one_launch_of_one_cta_per_task_pair(rows):
    kernel = front.build_replicated_front(_plan(rows))
    assert kernel.grid == (front.PAIRS,)
    assert kernel.block == (256,)
    assert kernel.launch_count == 1
    assert kernel.name.endswith(f"_s{rows}")
    assert kernel.abi.names[0] == "x"
    assert "router_logits" in kernel.abi.names
    assert tdm_lds_bytes(rows, 2) <= 320 * 1024


@pytest.mark.parametrize("rows", (1, 40, 48))
def test_fp8_shared_front_takes_the_block_scales(rows):
    kernel = front.build_replicated_front(_plan(rows), fp8_shared=True)
    assert kernel.name.endswith(f"_s{rows}_f8")
    names = kernel.abi.names
    assert names.index("shared_scale") == names.index("shared_weight") + 1


@pytest.mark.parametrize("rows", (1, 10, 20, 40, 48))
@pytest.mark.parametrize("fp4_latent", (False, True))
def test_split_front_fits_every_row_count(rows, fp4_latent):
    assert front.split_front_fits(_plan(rows), True, fp4_latent=fp4_latent)


def test_mxfp4_latent_front_refuses_a_grid_without_its_clusters():
    plan = Gfx1250KimiBuildPlan(
        threads=256, blocks=front.SPLIT_CTAS - 4, lds_bytes=320 * 1024, tokens=16
    )
    assert not front.split_front_fits(plan, fp4_latent=True)
    with pytest.raises(ValueError, match="MXFP4"):
        front.build_replicated_front(plan, fp4_latent=True)
    assert front.REPLICATED_FRONT_FP4_ABI.names[-1] == "latent_scale"


def test_replicated_front_feeds_only_the_grouped_moe_finish():
    with pytest.raises(ValueError, match="grouped-MoE"):
        build_packed_finish(_plan(16), replicated_front=True)


@pytest.mark.parametrize("rows", (1, 16, 40, 48))
def test_replicated_finalize_covers_every_hidden_row_once(rows):
    from atom.models.kimi_k3.mono.kernels.stages import k2_finalize_tdm_gfx1250 as fin

    kernel = fin.build_replicated_finalize(_plan(rows))
    assert kernel.grid == (fin.PAIRS,) == (7168 // 32,)
    assert kernel.abi.names == ("routed_normed", "shared", "up_weight", "output")


@pytest.mark.parametrize(
    ("rows", "front_paired", "oproj_paired", "finalize_paired"),
    (
        (5, True, True, True),
        (30, True, True, True),
        (31, False, True, True),
        (42, False, True, True),
        (43, False, False, True),
        (46, False, False, True),
        (47, False, False, False),
        (64, False, False, False),
    ),
)
def test_k2_gemvs_pair_their_slabs_where_four_token_slabs_fit(
    monkeypatch, rows, front_paired, oproj_paired, finalize_paired
):
    from atom.models.kimi_k3.mono.kernels.stages import k2_finalize_tdm_gfx1250 as fin
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    def names():
        oproj_plan = Gfx1250KimiBuildPlan(
            threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=1
        )
        return (
            front.build_replicated_front(_plan(rows), fp8_shared=True).name,
            oar.build_oproj_allreduce(oproj_plan, 8, 1536, oneshot=True).name,
            fin.build_replicated_finalize(_plan(rows)).name,
        )

    paired = names()
    assert ["_ps" in name for name in paired] == [
        front_paired,
        oproj_paired,
        finalize_paired,
    ]
    monkeypatch.setenv("ATOM_KIMI_MONO_PAIRED_SLABS", "0")
    assert not any("_ps" in name for name in names())
    monkeypatch.setenv("ATOM_KIMI_MONO_PAIRED_SLABS", "oproj")
    assert ["_ps" in name for name in names()] == [False, oproj_paired, False]


def _paired_by_default(rows):
    kernel = front.build_replicated_front(
        _plan(rows), fp8_shared=True, pipe_sites=frozenset()
    )
    return "_ps" in kernel.name


@pytest.mark.parametrize("rows", range(1, 65))
def test_late_front_plans_only_where_they_won_and_fit(rows):
    lds = 320 * 1024
    pipe = front.front_pipe(rows, True, lds, frozenset({"late"}), _paired_by_default(rows))
    expected = (
        (True, 6) if rows <= 9
        else None if rows <= 30
        else (False, 5) if rows <= 40
        else (False, 4) if rows <= 50
        else None
    )  # fmt: skip
    if expected is None:
        assert pipe == {}
        return
    assert (pipe["paired"], pipe["depth"]) == expected
    assert pipe["late"] and pipe["token_steps"] == 3
    assert pipe["fp8_depth"] == 6
    words = front._front_words(rows, pipe["depth"], pipe["paired"], 3)
    assert words + rows * 16 <= lds // 4


@pytest.mark.parametrize(
    ("rows", "name"),
    (
        (8, "k3_k2_gfx1250_replicated_front_w_d6_ps_late_es_f6_s8_f8"),
        (16, "k3_k2_gfx1250_replicated_front_w_d4_ps_es_s16_f8"),
        (40, "k3_k2_gfx1250_replicated_front_w_d5_late_es_f6_s40_f8"),
        (48, "k3_k2_gfx1250_replicated_front_w_d4_late_es_f6_s48_f8"),
        (64, "k3_k2_gfx1250_replicated_front_w_d4_es_s64_f8"),
    ),
)
def test_front_pipe_switch_names_its_build(monkeypatch, rows, name):
    default = front.build_replicated_front(_plan(rows), fp8_shared=True).name
    assert "_late" not in default and "_es" not in default
    monkeypatch.setenv("ATOM_KIMI_MONO_FRONT_PIPE", "late,early")
    kernel = front.build_replicated_front(_plan(rows), fp8_shared=True)
    assert kernel.name == name
    assert kernel.grid == (front.PAIRS,) and kernel.abi == front.REPLICATED_FRONT_ABI


def test_front_pipe_sites_parse_delays_and_epilogues():
    pipe = front.front_pipe(8, True, 320 * 1024, frozenset({"epi", "hw4", "rd16"}))
    assert pipe == {"fused": True, "diag": frozenset({"hw4"}), "delay": {"router": 16}}
    assert front._pipe_suffix(pipe) == "_fe_rd16_hw4"


def test_herd_candidates_feed_only_the_grouped_moe_finish():
    with pytest.raises(ValueError, match="HERD"):
        build_packed_finish(_plan(16), herd=True)
    kernel = build_packed_finish(
        _plan(16), include_routed=False, replicated_front=True, herd=True
    )
    # its 16 routing CTAs leave the shared down to the other 240
    assert kernel.name.endswith("_rf_herd_ra")
