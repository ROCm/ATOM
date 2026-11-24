# SPDX-License-Identifier: MIT
"""MiniMax-M3 SP prefill ``_expert_tables``: a MoE block without the FHMoE attribute (builds that lack the FHMoE
integration) takes the regular expert tables and applies the routed scale after the chain.
"""

from types import SimpleNamespace

import pytest

sp = pytest.importorskip("atom.models.minimax_m3.sp_prefill")


def _experts():
    return SimpleNamespace(
        w13_weight="w13",
        w2_weight="w2",
        w13_weight_scale="w13s",
        w2_weight_scale="w2s",
    )


def test_expert_tables_without_fhmoe_attribute():
    moe = SimpleNamespace(experts=_experts(), routed_scaling_factor=2.5)
    assert not hasattr(moe, "fhmoe")
    tables, scale = sp._expert_tables(moe)
    assert tables == ("w13", "w2", "w13s", "w2s") and scale == 2.5


def test_expert_tables_fhmoe_none_keeps_the_scale():
    moe = SimpleNamespace(experts=_experts(), routed_scaling_factor=2.5, fhmoe=None)
    assert sp._expert_tables(moe)[1] == 2.5


def test_expert_tables_with_fhmoe_folds_the_scale():
    fh = SimpleNamespace(routed_tables=("a", "b", "c", "d"))
    moe = SimpleNamespace(experts=_experts(), routed_scaling_factor=2.5, fhmoe=fh)
    assert sp._expert_tables(moe) == (("a", "b", "c", "d"), 1.0)
