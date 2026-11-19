# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.models.kimi_k3.mono.kernels.attention import (
    kda_builder_gfx1250 as builder,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)


def _plan(rows=16):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=rows,
        nblocks=4,
    )


def _spec(qlen=16):
    return builder.K1LocalBuildSpec(
        qlen=qlen,
        delta=True,
        state_len=qlen + 2,
        eps=1e-5,
        out_eps=1e-5,
        onorm_eps=1e-5,
        lower_bound=-5.0,
        write_idx=-1,
    )


def test_build_local_loads_complete_product_pipeline(monkeypatch):
    plan = _plan()
    spec = replace(_spec(), kda0=7)
    kernel = SimpleNamespace(complete=True, launch_count=1)
    observed = {}

    def build_resident_k1(got_plan, declaration, got_spec):
        observed.update(
            plan=got_plan,
            declaration=declaration,
            spec=got_spec,
        )
        return kernel

    monkeypatch.setattr(
        builder,
        "import_module",
        lambda name: SimpleNamespace(build_resident_k1=build_resident_k1),
    )

    result = builder.build_local(plan, spec=spec)
    assert result.runnable
    assert result.complete
    assert result.kernel is kernel
    assert result.launch_count == 1
    assert result.declaration.protocol.kda0 == 7
    assert result.refusal.code == "none"
    assert observed == {
        "plan": plan,
        "declaration": result.declaration,
        "spec": spec,
    }


def test_local_scratch_covers_every_protocol_region():
    declaration = builder.declare_local(_plan(48), qlen=16)
    layout = builder.local_scratch_layout(declaration)

    assert tuple(layout) == (
        *declaration.protocol.mailbox_pairs,
        "sync",
        "x",
    )
    assert all(offset % 16 == 0 for offset, _size in layout.values())
    for name, pairs in declaration.protocol.mailbox_pairs.items():
        assert layout[name][1] == pairs * 8


def test_split_projection_owns_whole_clusters_past_attention():
    assert builder.split_projection_ctas(256) == 200
    with pytest.raises(ValueError, match="whole TDM clusters"):
        builder.split_projection_ctas(254)
    with pytest.raises(ValueError, match="tasks"):
        builder.split_projection_ctas(240)


@pytest.mark.parametrize("rows", [1, 10, 20, 40, 48])
def test_fp8_split_projection_fits_every_row_count(rows):
    plan = _plan(rows)
    assert builder.split_projection_fits(plan, qlen=1)
    lds = builder.role_lds_plan(plan, qlen=1, split="fp8")
    assert lds.split_format == "fp8"
    assert lds.split_stages == (3 if rows <= 46 else 2)
    assert lds.smem.bytes + lds.role.bytes <= plan.lds_bytes


def test_bf16_split_projection_refuses_rows_it_cannot_hold():
    assert builder.split_projection_fits(_plan(36), qlen=1, fp8=False)
    assert not builder.split_projection_fits(_plan(40), qlen=1, fp8=False)
    with pytest.raises(ValueError, match="does not fit 40 rows"):
        builder.role_lds_plan(_plan(40), qlen=1, split="bf16")


def test_split_scratch_follows_the_x_handoff():
    plan = _plan(40)
    declaration = builder.declare_local(plan, qlen=1, split="fp8")
    layout = builder.local_scratch_layout(declaration)
    x_offset, x_bytes = layout["x"]
    offset, size = layout["split"]
    assert tuple(layout)[-2:] == ("x", "split")
    assert offset % 16 == 0 and offset >= x_offset + x_bytes
    assert size == 200 * 8 * 3 * 256 * 4
    assert "split" not in builder.local_scratch_layout(
        builder.declare_local(plan, qlen=1)
    )


def test_split_projection_requires_the_tdm_projection():
    spec = replace(
        _spec(qlen=1), state_len=3, split_projection=True, tdm_projection=False
    )
    with pytest.raises(ValueError, match="split"):
        spec.validate(_plan(40))
    assert replace(spec, tdm_projection=True).split_format == "bf16"
    assert replace(spec, fp8_projection=True).split_format == "fp8"


def test_resident_projection_reuses_each_weight_slab_across_s48_tiles():
    plan = _plan(48)
    spec = _spec()
    declaration = builder.declare_local(
        plan,
        qlen=spec.qlen,
        gemv_unit=spec.gemv_unit,
    )
    slabs = declaration.lds.arena.gemv_slabs[0]

    assert declaration.gemv_unit == 2
    assert slabs == ((0, 2), (2, 2), (4, 2), (6, 1))
    assert sum(groups for _first, groups in slabs) == 7
    assert spec.launch_count == 1
    # 6284 rows x 7168 BF16 columns, loaded once rather than once per M16 tile.
    assert builder.HIDDEN * 6284 * 2 == 90_087_424


def test_resident_launch_pins_one_eight_wave_cta_per_cu():
    source = (
        Path(__file__).parents[2]
        / "atom"
        / "models"
        / "kimi_k3"
        / "mono"
        / "kernels"
        / "attention"
        / "kda_resident_gfx1250.py"
    ).read_text()
    assert 'CompilationContext.compile_hints({"waves_per_eu": 2})' in source
    generation = source[source.index("def _arrive") : source.index("def _task_start")]
    kept = generation.index("fx.ptr_store(fx.Int64(previous + fx.Int32(1)), slots)")
    barrier = generation.index("gpu.barrier()", kept)
    read = generation.index("fx.ptr_load(slots)", barrier)
    assert kept < barrier < read

    projection = source.index("if bid >= ATTENTION_CTAS:")
    ready = source.index("mailbox_poll_device(", projection)
    weights = source.index('compute["kda_gemv_loads"](', ready)
    assert projection < ready < weights
    declaration = builder.declare_local(_plan(), qlen=16)
    assert declaration.candidate_grid == (256,)
    assert declaration.known_block_size == (256, 1, 1)


def test_resident_attention_refuses_more_than_sixteen_sources():
    with pytest.raises(ValueError, match="at most 16 sources"):
        builder.declare_local(replace(_plan(), nblocks=16), qlen=16)


def test_local_build_accepts_the_production_conv_window():
    plan = _plan()
    _spec(qlen=16).validate(plan)
    with pytest.raises(ValueError, match="two retained history"):
        replace(_spec(qlen=16), state_len=17).validate(plan)


@pytest.mark.parametrize(
    ("switch", "heads", "split"),
    [
        ("", 48, 1),
        ("1", 48, 1),
        ("2", 48, 2),
        ("4", 96, 1),
        ("auto", 48, 4),
        ("auto", 96, 2),
        ("auto", 192, 1),
    ],
)
def test_head_split_counts_only_parts_the_grid_holds_at_once(
    monkeypatch, switch, heads, split
):
    from atom.models.kimi_k3.mono.kernels.attention import kda_resident_gfx1250

    monkeypatch.setenv("ATOM_KIMI_MONO_K1_HEAD_SPLIT", switch)
    assert kda_resident_gfx1250._head_split(heads, 256) == split


def test_head_split_defaults_to_auto(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_K1_HEAD_SPLIT", raising=False)
    assert envs.ATOM_KIMI_MONO_K1_HEAD_SPLIT == "auto"


def test_whole_heads_replace_the_chunk_tasks_by_default(monkeypatch):
    from atom.models.kimi_k3.mono.kernels.attention import kda_resident_gfx1250
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_K1_WHOLE_HEADS", raising=False)
    assert envs.ATOM_KIMI_MONO_K1_WHOLE_HEADS
    # one or two requests' chunk tasks fit the grid: forced whole heads
    assert kda_resident_gfx1250._whole_heads(5, 96, 256, False) == (True, True)
    assert kda_resident_gfx1250._whole_heads(8, 192, 256, False) == (True, True)
    # past one pass, plain decode and ReplaySSM: whole heads anyway
    assert kda_resident_gfx1250._whole_heads(5, 288, 256, False) == (True, False)
    assert kda_resident_gfx1250._whole_heads(1, 96, 256, False) == (True, False)
    assert kda_resident_gfx1250._whole_heads(5, 96, 256, True) == (True, False)


def test_whole_heads_switch_off_keeps_the_chunk_tasks(monkeypatch):
    from atom.models.kimi_k3.mono.kernels.attention import kda_resident_gfx1250

    monkeypatch.setenv("ATOM_KIMI_MONO_K1_WHOLE_HEADS", "0")
    assert kda_resident_gfx1250._whole_heads(5, 96, 256, False) == (False, False)
    assert kda_resident_gfx1250._whole_heads(5, 288, 256, False) == (True, False)


def test_head_split_refuses_other_counts(monkeypatch):
    from atom.models.kimi_k3.mono.kernels.attention import kda_resident_gfx1250

    monkeypatch.setenv("ATOM_KIMI_MONO_K1_HEAD_SPLIT", "3")
    with pytest.raises(ValueError):
        kda_resident_gfx1250._head_split(48, 256)


def test_replay_records_bound_what_a_step_finds_committed():
    # the serving cache_len 16, DSpark 4's window 5: a flush once h > 6
    assert builder.replay_records(16, 5) == 11
    assert builder.replay_staged(16, 5) == 12
    assert builder.replay_staged(16, 8) == 8


def test_replay_lds_appends_to_the_recurrence_only():
    plain = builder.role_lds_plan(_plan(40), qlen=5)
    replay = builder.role_lds_plan(_plan(40), qlen=5, replay_cap=16, replay_window=5)
    assert plain == builder.role_lds_plan(_plan(40), qlen=5, replay_cap=0)
    assert plain != replay
    fields = repr(replay)
    assert "glog" in fields and "rec" in fields
    assert "glog" not in repr(plain)


@pytest.mark.parametrize(
    ("cap", "window", "message"),
    (
        (16, 4, "window"),  # narrower than the query
        (8, 5, "window"),  # two windows do not fit the buffer
        (0, 5, "without a record buffer"),
    ),
)
def test_replay_spec_refuses_an_inconsistent_window(cap, window, message):
    spec = replace(_spec(5), replay_cap=cap, replay_window=window)
    with pytest.raises(ValueError, match=message):
        spec.validate(_plan(40))
