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
        state_len=3,
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
        on_w=f32(1),
        rstate=f32(1),
        rs_seq=1,
        st_idx=i32(1),
        st_stride=1,
        num_acc=i32(1),
        core_out=bf16(1),
        core_q=None,
        core_scale=None,
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
        launch_count=1,
    )
    result = SimpleNamespace(
        runnable=True,
        kernel=kernel,
        declaration=declaration,
        launch_count=1,
    )
    runner = Gfx1250K1LocalRunner(plan, _spec(), result).prepare(torch.device("cpu"))
    runtime = _runtime()

    assert runner._scratch is not None
    assert runner(runtime) is runtime.core_out
    assert runner(runtime, stream=7) is runtime.core_out
    assert runner.launch_count == 1
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
        launch_count=1,
    )
    result = SimpleNamespace(
        runnable=True,
        kernel=kernel,
        declaration=declaration,
        launch_count=1,
    )
    runtime = replace(
        _runtime(),
        delta=torch.zeros((1, 7168), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="delta must be bfloat16"):
        Gfx1250K1LocalRunner(plan, _spec(), result)(runtime)


def test_block_write_requires_the_new_residual_row():
    plan = _plan()
    spec = replace(_spec(), write_idx=1)
    declaration = builder.declare_local(plan, qlen=1)
    kernel = SimpleNamespace(
        complete=True,
        launcher=lambda *_args, **_kwargs: None,
        launch_count=1,
    )
    result = SimpleNamespace(
        runnable=True,
        kernel=kernel,
        declaration=declaration,
        launch_count=1,
    )

    with pytest.raises(ValueError, match="at least 2 residual rows"):
        Gfx1250K1LocalRunner(plan, spec, result)(_runtime())


def test_quantized_core_outputs_are_typed_and_packed_into_the_abi():
    plan = _plan()
    spec = replace(_spec(), quant_core=True)
    declaration = builder.declare_local(plan, qlen=1)
    observed = []
    kernel = SimpleNamespace(
        complete=True,
        launcher=lambda *args, **_kwargs: observed.append(args),
        launch_count=1,
    )
    result = SimpleNamespace(
        runnable=True,
        kernel=kernel,
        declaration=declaration,
        launch_count=1,
    )
    quant_bytes = torch.empty(12 * 128 + 4, dtype=torch.uint8)
    runtime = replace(
        _runtime(),
        core_q=quant_bytes[: 12 * 128].view(1, 12 * 128),
        core_scale=quant_bytes[12 * 128 :].view(torch.float32).view(1, 1),
    )

    Gfx1250K1LocalRunner(plan, spec, result)(runtime)

    assert (
        observed[0][builder.K1_ABI_NAMES.index("core_q")] == runtime.core_q.data_ptr()
    )
