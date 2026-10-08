# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono kernels' hand-off contracts on real builds: traced compile-only
for gfx950 (no GPU launch), every mailbox access checked against
``contract``'s region tables.

One build of each served model (the V4.1 target and its DSpark draft), each
traced once; the negative controls check the target's trace against a wrong
table instead of building again."""

import dataclasses

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.deepseek_v41.mono import contract
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.mono.plan.check import ContractError, check
from atom.mono.runtime.compile import trace

# one verify request (5 drafts + 1) at TP 4, the indexer bounded by a 1M context
TARGET_K1 = k1.AttnPreBuild(tokens=6, feeds_index=True, tp=4)
TARGET_K2 = k2.LayerPostBuild(tokens=6, index_bound_max=1 << 20, tp=4)
# one request's 10-row block: 128 experts top-3, no indexer
DRAFT_K1 = k1.AttnPreBuild(tokens=10, tp=4)
DRAFT_K2 = k2.LayerPostBuild(tokens=10, experts=128, topk=3, tp=4, ug_groups=1)


@pytest.fixture(scope="module")
def traces():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ARCH", "gfx950")
        patch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        return {
            "target_k1": trace(k1.build_attn_pre(TARGET_K1), k1.ABI),
            "target_k2": trace(k2.build_layer_post(TARGET_K2), k2.ABI),
            "draft_k1": trace(k1.build_attn_pre(DRAFT_K1), k1.ABI),
            "draft_k2": trace(k2.build_layer_post(DRAFT_K2), k2.ABI),
        }


def test_the_target_holds(traces):
    check(traces["target_k1"], contract.k1_regions(TARGET_K1))
    check(traces["target_k2"], contract.k2_regions(TARGET_K2))


def test_the_draft_holds(traces):
    check(traces["draft_k1"], contract.k1_regions(DRAFT_K1))
    check(traces["draft_k2"], contract.k2_regions(DRAFT_K2))


def _changed(decls, name, **change):
    assert any(d.name == name for d in decls), name
    return [dataclasses.replace(d, **change) if d.name == name else d for d in decls]


@pytest.mark.parametrize(
    "name, change, caught",
    [
        # K2b's route table declared as the router's
        ("route", {"writer": "k2b.router"}, "route put by k2b.route"),
        # a selection can be final at any tree level: naming only the first
        # level as the ready flag's writer is caught by the second's put
        ("irdy", {"writer": "k2a.isel0"}, "irdy put by k2a.isel1"),
        # the all-reduce partials declared as this GPU's scratch
        ("ffn", {"space": contract.S}, "ffn in k2b.down as peer"),
    ],
)
def test_a_wrong_table_is_caught(traces, name, change, caught):
    decls = _changed(contract.k2_regions(TARGET_K2), name, **change)
    with pytest.raises(ContractError, match=caught):
        check(traces["target_k2"], decls)
