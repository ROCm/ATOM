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
        state_len=qlen + 3,
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
    generation = source[source.index("def _opening_generation") : source.index(
        "def _task_start"
    )]
    load = generation.index("generation = fx.ptr_load(ticket)")
    barrier = generation.index("gpu.barrier()", load)
    returned = generation.index("return generation", barrier)
    assert load < barrier < returned

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
