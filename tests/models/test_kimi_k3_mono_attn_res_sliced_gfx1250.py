# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry, exchange-record and ABI contracts of the column-sliced
gfx1250 AttnRes mix."""

from types import SimpleNamespace

import pytest

from atom.models.kimi_k3.mono.kernels.attention.layout import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import attn_res_sliced_layout as plan
from atom.mono.plan.architecture import GFX1250

GEOMETRIES = [plan.SlicedGeometry(cluster) for cluster in plan.CLUSTERS]


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: f"c{g.cluster}")
def test_thread_groups_fill_the_cta_in_whole_waves(geometry):
    assert geometry.columns * geometry.cluster == HIDDEN
    assert geometry.lanes * plan.ELEMENTS == geometry.columns
    assert geometry.token_threads % GFX1250.wave_size == 0
    assert geometry.lanes <= geometry.token_threads < geometry.lanes + GFX1250.wave_size
    # 4-CTA: one 7-wave group a pass, a wave idle
    passing = geometry.tokens_a_pass
    assert passing * geometry.token_threads <= plan.THREADS
    assert plan.THREADS < (passing + 1) * geometry.token_threads
    assert geometry.max_clusters * geometry.cluster == plan.MAX_CTAS


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: f"c{g.cluster}")
def test_each_column_has_one_owner_and_idle_slots_stay_in_the_slice(geometry):
    owners = [0] * HIDDEN
    for rank in range(geometry.cluster):
        for slot in range(geometry.token_threads):
            first = geometry.column(rank, slot)
            assert rank * geometry.columns <= first
            assert first + plan.ELEMENTS <= (rank + 1) * geometry.columns
            if slot < geometry.lanes:
                for column in range(first, first + plan.ELEMENTS):
                    owners[column] += 1
    assert owners == [1] * HIDDEN


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: f"c{g.cluster}")
@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 10, 20, 40, 48, 63, 64, 65, 100, 257])
def test_clusters_cover_every_token_once(geometry, tokens):
    seen = []
    for home in range(geometry.clusters(tokens)):
        for step in range(geometry.passes(tokens, home)):
            for local in range(geometry.tokens_a_pass):
                token = geometry.token(tokens, home, step, local)
                if token < tokens:
                    seen.append(token)
    assert sorted(seen) == list(range(tokens))
    assert geometry.ctas(tokens) <= plan.MAX_CTAS


@pytest.mark.parametrize("candidates", range(2, plan.MAX_CANDIDATES + 1))
def test_gram_pairs_follow_the_squares_and_dots(candidates):
    crossed = plan.pairs(candidates)
    assert len(crossed) == candidates * (candidates - 1) // 2
    assert crossed == sorted(crossed) and all(c < d for c, d in crossed)
    assert plan.sums(candidates, False) == 2 * candidates
    assert plan.sums(candidates, True) == 2 * candidates + len(crossed)


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: f"c{g.cluster}")
@pytest.mark.parametrize("candidates", [2, 5, 7, plan.MAX_CANDIDATES])
@pytest.mark.parametrize("gram", [False, True])
def test_gate_reads_the_sums_the_slices_store(geometry, candidates, gram):
    """The addresses ``build_attn_res_sliced`` forms in a token's record: the
    slice stage stores partial k of rank r at ``k C + r``, partials ordered
    square, dot a candidate, then the Gram pairs; the gate reads candidate c's
    square at ``2 c C``, its dot at ``(2 c + 1) C``, gate item i (pair i - NS)
    at ``(i + NS) C``; without ``gram`` the mix's square sums follow."""
    cluster = geometry.cluster
    record = geometry.record_words(candidates, gram)
    stored = []
    for c in range(candidates):
        stored += [("square", c), ("dot", c)]
    if gram:
        stored += [("pair", pair) for pair in plan.pairs(candidates)]
    assert len(stored) == plan.sums(candidates, gram)
    if not gram:
        stored.append(("norm",))
    assert len(stored) * cluster == record
    for c in range(candidates):
        assert stored[2 * c * cluster // cluster] == ("square", c)
        assert stored[(2 * c * cluster + cluster) // cluster] == ("dot", c)
    if gram:
        for item in range(candidates, candidates + len(plan.pairs(candidates))):
            at = (item + candidates) * cluster
            assert stored[at // cluster] == (
                "pair",
                plan.pairs(candidates)[item - candidates],
            )
            assert at + cluster <= record
    else:
        assert stored[plan.sums(candidates, False)] == ("norm",)


@pytest.mark.parametrize("cluster", plan.CLUSTERS)
@pytest.mark.parametrize("gram", [False, True])
def test_scratch_bytes_cover_every_token_record(cluster, gram):
    for count in (1, 4, 6, plan.MAX_CANDIDATES - 1):
        for tokens in (1, 40, 100):
            record = plan.SlicedGeometry(cluster).record_words(count + 1, gram)
            assert (
                plan.scratch_bytes(count, tokens, cluster, gram) == record * tokens * 4
            )


@pytest.mark.parametrize(
    ("count", "addends"), ((0, 1), (plan.MAX_CANDIDATES, 1), (4, 3), (4, -1))
)
def test_refuses_what_it_cannot_hold(count, addends):
    with pytest.raises(ValueError):
        plan.check(count, addends)


def test_gram_needs_the_output_norm():
    plan.check(4, 1, out_norm=True, gram=True)
    with pytest.raises(ValueError):
        plan.check(4, 1, out_norm=False, gram=True)


@pytest.mark.parametrize("cluster", (1, 2, 12, 32))
def test_refuses_other_slicings(cluster):
    with pytest.raises(ValueError):
        plan.SlicedGeometry(cluster)


def test_lds_fits_the_largest_gram_build():
    candidates = plan.MAX_CANDIDATES
    reductions = plan.sums(candidates, True)
    crosses = len(plan.pairs(candidates))
    for geometry in GEOMETRIES:
        passing = geometry.tokens_a_pass
        words = (
            plan.WAVES * reductions
            + 2 * passing * candidates
            + passing * crosses
            + plan.WAVES
        )
        assert words * 4 <= GFX1250.max_workgroup_lds_bytes


# ---------------------------------------------------------------- the kernel


@pytest.mark.parametrize("cluster", plan.CLUSTERS)
@pytest.mark.parametrize("count", (1, 4, 6))
@pytest.mark.parametrize("addends", (0, 1, 2))
@pytest.mark.parametrize("gram", (False, True))
def test_builds_take_the_mix_abi_plus_scratch(cluster, count, addends, gram):
    pytest.importorskip("flydsl", reason="building the kernel needs FlyDSL")
    from atom.models.kimi_k3.mono.kernels.stages import attn_res_gfx1250 as mono
    from atom.models.kimi_k3.mono.kernels.stages import (
        attn_res_sliced_gfx1250 as sliced,
    )

    kernel = sliced.build_attn_res_sliced(
        count, addends, True, 1e-6, 1e-5, cluster, gram
    )
    assert kernel.abi is sliced.ATTN_RES_SLICED_ABI
    assert sliced.ATTN_RES_SLICED_ABI.names == (
        *mono.ATTN_RES_ABI.names,
        "scratch",
    )
    assert kernel.geometry == plan.SlicedGeometry(cluster)
    assert kernel.name == (
        f"k3_gfx1250_attn_res_sliced_c{cluster}_b{count}_a{addends}_n"
        f"{'_g' if gram else ''}"
    )
    # the serving bench files every AttnRes kernel under one profiler key
    assert kernel.name.startswith("k3_gfx1250_attn_res")


# ---------------------------------------------------------------- the runner


def _runner():
    pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
    pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")
    import atom.models.kimi_k3.mono.runner_gfx1250 as runner_module

    return runner_module


def test_scratch_only_grows_and_none_is_ever_freed(monkeypatch):
    runner = _runner()
    import torch

    monkeypatch.setattr(runner, "_ATTN_RES_SCRATCH", [])
    device = torch.device("cpu")
    small = runner._sliced_scratch(64, device)
    assert runner._sliced_scratch(32, device) is small
    large = runner._sliced_scratch(4096, device)
    assert large is not small and large.numel() >= 4096
    assert runner._sliced_scratch(64, device) is large
    assert runner._ATTN_RES_SCRATCH == [small, large]


def test_sliced_mix_launches_the_mix_arguments_then_scratch(monkeypatch):
    runner = _runner()
    import torch

    calls = []
    geometry = plan.SlicedGeometry(8)
    fake = SimpleNamespace(
        geometry=geometry,
        launcher=lambda *args, stream=None: calls.append(args),
    )
    key = (4, 1, 1e-6, 1e-5)
    monkeypatch.setattr(runner, "_ATTN_RES_SLICED", {(key, 8): fake})
    monkeypatch.setattr(runner, "_ATTN_RES_SCRATCH", [])
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    tokens = 3
    prefix = torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16)
    add = torch.zeros_like(prefix)
    blocks = torch.zeros(tokens, 5, HIDDEN, dtype=torch.bfloat16)[:, :4]
    site = SimpleNamespace(score_weight=torch.zeros(HIDDEN))
    norm = SimpleNamespace(weight=torch.ones(HIDDEN, dtype=torch.bfloat16))
    out, prefix_out = runner._sliced_attn_res(key, prefix, [add], blocks, site, norm, 8)
    (args,) = calls
    assert len(args) == 11
    assert args[0] == blocks.data_ptr() and args[8] == blocks.stride(0)
    # one addend: the unused second addend slot holds the prefix
    assert args[1] == args[3] == prefix.data_ptr() and args[2] == add.data_ptr()
    assert args[6] == out.data_ptr() and args[7] == prefix_out.data_ptr()
    assert args[9] == tokens
    (scratch,) = runner._ATTN_RES_SCRATCH
    assert args[10] == scratch.data_ptr()
    assert scratch.numel() >= plan.scratch_bytes(4, tokens, 8)


@pytest.mark.parametrize(
    ("value", "cluster"), (("0", 0), ("1", 8), ("4", 4), ("8", 8), ("16", 16))
)
def test_the_flag_names_the_cluster(monkeypatch, value, cluster):
    from atom.utils import envs

    monkeypatch.setenv("ATOM_KIMI_MONO_ATTN_RES_SLICED", value)
    assert envs.ATOM_KIMI_MONO_ATTN_RES_SLICED == cluster
    if cluster:
        plan.SlicedGeometry(cluster)


def test_the_flag_defaults_to_four_ctas(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_ATTN_RES_SLICED", raising=False)
    assert envs.ATOM_KIMI_MONO_ATTN_RES_SLICED == 4
