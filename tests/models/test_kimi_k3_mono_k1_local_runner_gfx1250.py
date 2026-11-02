# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU contracts for the product gfx1250 rank-local K1 adapter."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono.k1_local_runner_gfx1250 import (
    Gfx1250K1LocalRunner,
    K1LocalRuntime,
)
from atom.models.kimi_k3.mono.kernels.attention import (
    kda_builder_gfx1250 as builder,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)


def _plan():
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=1,
        nblocks=1,
    )


def _spec():
    return builder.K1LocalBuildSpec(
        qlen=1,
        delta=True,
        state_len=4,
        eps=1e-5,
        out_eps=1e-5,
        onorm_eps=1e-5,
        lower_bound=-5.0,
        write_idx=-1,
    )


def _runtime():
    def bf16(*shape):
        return torch.zeros(shape, dtype=torch.bfloat16)

    def f32(*shape):
        return torch.zeros(shape, dtype=torch.float32)

    def i32(*shape):
        return torch.zeros(shape, dtype=torch.int32)

    return K1LocalRuntime(
        prefix=bf16(1, 7168),
        delta=bf16(1, 7168),
        blocks=bf16(1, 1, 7168),
        blk_sm=7168,
        blk_sr=7168,
        ares_nw=bf16(1),
        ares_qk=bf16(1),
        in_nw=bf16(1),
        w_in=bf16(1),
        w_fb=bf16(1),
        conv_w=f32(1),
        conv_st=bf16(1),
        cs_seq=1,
        cs_dim=4,
        cs_tok=1,
        a_log=f32(1),
        dt_bias=f32(1),
        on_w=bf16(1),
        rstate=f32(1),
        rs_seq=1,
        st_idx=i32(1),
        st_stride=1,
        num_acc=i32(1),
        core_out=bf16(1),
        x_dbg=bf16(1),
        proj_dbg=bf16(1),
        layer=2,
        tl=torch.zeros(1, dtype=torch.int64),
    )


def test_product_runner_packs_exact_local_abi_and_owns_scratch():
    plan = _plan()
    declaration = builder.declare_local(plan, qlen=1)
    observed = []

    def launch(*arguments, **kwargs):
        observed.append((arguments, kwargs))

    kernel = SimpleNamespace(
        complete=True,
        launcher=launch,
        launch_count=6,
    )
    result = SimpleNamespace(
        runnable=True,
        kernel=kernel,
        declaration=declaration,
    )
    runner = Gfx1250K1LocalRunner(plan, _spec(), result)
    runtime = _runtime()

    assert runner(runtime) is runtime.core_out
    assert runner(runtime, stream=7) is runtime.core_out
    assert runner.launch_count == 6
    assert len(observed) == 2
    assert all(len(arguments) == len(builder.K1_ABI_NAMES) for arguments, _ in observed)
    scratch_index = builder.K1_ABI_NAMES.index("scratch")
    assert observed[0][0][scratch_index] == observed[1][0][scratch_index]
    assert observed[1][1] == {"stream": 7}


def test_plan_exposes_typed_product_build_api(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(builder, "build_local", lambda plan, spec: sentinel)
    assert _plan().build_k1_local(_spec()) is sentinel


def test_product_runner_rejects_dtype_drift():
    plan = _plan()
    declaration = builder.declare_local(plan, qlen=1)
    kernel = SimpleNamespace(
        complete=True,
        launcher=lambda *_args, **_kwargs: None,
        launch_count=6,
    )
    result = SimpleNamespace(
        runnable=True,
        kernel=kernel,
        declaration=declaration,
    )
    runtime = replace(
        _runtime(),
        delta=torch.zeros((1, 7168), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="delta must be bfloat16"):
        Gfx1250K1LocalRunner(plan, _spec(), result)(runtime)
