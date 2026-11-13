# SPDX-License-Identifier: MIT
import sys
from types import SimpleNamespace

import pytest
from aiter_stub import atom_under_stubs


@pytest.fixture(scope="module")
def atom_modules():
    with atom_under_stubs() as load:
        yield SimpleNamespace(
            moe=load("atom.model_ops.moe"),
            comm=load("atom.model_ops.fused_moe.comm_fused_moe"),
            deepseek_v4=load("atom.models.deepseek_v4"),
            host=load("aiter.ops.flydsl.comm_fused_moe_host"),
            runtime=load("aiter.ops.comm_fused_moe_runtime"),
        )


@pytest.fixture
def sp_modules(request):
    # Preserve the external types used by an already imported MoE module.
    # Replacing AITER then would create a second, incompatible QuantType enum.
    moe = sys.modules.get("atom.model_ops.moe")
    if moe is None:
        moe = request.getfixturevalue("atom_modules").moe
    return SimpleNamespace(moe=moe)
