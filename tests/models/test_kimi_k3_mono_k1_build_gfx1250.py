# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

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
    spec = _spec()
    kernel = SimpleNamespace(complete=True, launch_count=7)
    observed = {}

    def build_local_pipeline(got_plan, declaration, got_spec):
        observed.update(
            plan=got_plan,
            declaration=declaration,
            spec=got_spec,
        )
        return kernel

    monkeypatch.setattr(
        builder,
        "import_module",
        lambda name: SimpleNamespace(build_local_pipeline=build_local_pipeline),
    )

    result = builder.build_local(plan, spec=spec)
    assert result.runnable
    assert result.complete
    assert result.kernel is kernel
    assert result.launch_count == 7
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
        "vals",
        "vloc",
        "ol",
    )
    assert all(offset % 16 == 0 for offset, _size in layout.values())
    for name, pairs in declaration.protocol.mailbox_pairs.items():
        assert layout[name][1] == pairs * 8
