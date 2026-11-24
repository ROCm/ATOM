# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Dual-copy regions (``atom.mono.plan.layout.dual_layout``): each copy in runs
of one home, none across a page, the table a build key's scalar; V4.1's one
table for every kernel sharing a scratch (``numa_plan.numa_table``)."""

import pytest

from atom.models.deepseek_v41.mono.numa_plan import DUAL_REGIONS, numa_table
from atom.mono.plan.layout import (
    NUMA_PAGE,
    NUMA_RUN,
    dual_layout,
    dual_places,
    dual_spec,
    dual_word,
)

RUN_WORDS = NUMA_RUN // 4


def test_copy_zero_takes_the_even_runs_copy_one_the_odd():
    for w in (0, 1, RUN_WORDS - 4, RUN_WORDS, 3 * RUN_WORDS + 5):
        run, at = divmod(dual_word(w), RUN_WORDS)
        assert run == 2 * (w // RUN_WORDS) and at == w % RUN_WORDS
        assert (dual_word(w) + RUN_WORDS) // RUN_WORDS == run + 1


def test_a_16_byte_store_never_straddles_a_run():
    for w in range(0, 3 * RUN_WORDS, 4):
        assert dual_word(w) // RUN_WORDS == dual_word(w + 3) // RUN_WORDS


def test_regions_start_on_even_runs_and_stay_in_their_page():
    sizes = [("a", 800), ("b", 393216), ("c", 8), ("d", 960000), ("e", 40000)]
    start = 3 * NUMA_PAGE
    table, area = dual_layout(sizes, start)
    end = start
    for (name, off, page), (_, nbytes) in zip(table, sizes):
        span = 2 * -(-nbytes // NUMA_RUN) * NUMA_RUN
        assert off >= end and (off - start) % (2 * NUMA_RUN) == 0, name
        assert (off - start) // NUMA_PAGE == page
        assert (off + span - 1 - start) // NUMA_PAGE == page, f"{name} crosses a page"
        end = off + span
    assert start + area == end


def test_the_area_starts_on_a_page():
    with pytest.raises(AssertionError):
        dual_layout([("a", 8)], start=NUMA_RUN)


def test_a_region_larger_than_a_page_is_refused():
    with pytest.raises(AssertionError):
        dual_layout([("a", NUMA_PAGE)])


def test_the_spec_round_trips():
    table, _ = dual_layout([("lin", 1000), ("as", 300000)], NUMA_PAGE)
    spec = dual_spec(table)
    assert isinstance(spec, str)
    assert dual_places(spec) == {name: (off, page) for name, off, page in table}
    assert dual_places("") == {}


def test_v41_one_table_whatever_kernel_names_a_region():
    k1 = {"lin": (0, 4096), "pmix": (4096, 800), "x8": (4896, 64), "qkv": (5000, 16)}
    k2 = {"lin": (0, 4096), "pmix": (4096, 800), "as": (6000, 512)}
    spec, total = numa_table([k1, k2], 10_000)
    places = dual_places(spec)
    assert set(places) == {"lin", "pmix", "qkv", "as"}, "plain regions stay single"
    assert list(places) == [n for n in DUAL_REGIONS if n in places]
    assert min(off for off, _ in places.values()) == NUMA_PAGE
    assert numa_table([k2, k1], 10_000) == (spec, total)


def test_v41_a_region_too_large_for_a_page_stays_single():
    spec, _ = numa_table([{"as": (0, NUMA_PAGE // 2 + 8), "xo": (0, 64)}], 0)
    assert set(dual_places(spec)) == {"xo"}


def test_v41_a_region_the_builds_size_apart_takes_the_widest():
    # a step's layer kernel builds size IQDONE apart (an indexing layer's a
    # group a flag): one slot, the widest, whichever build comes first
    widest = numa_table([{"lin": (0, 16), "xo": (0, 64)}], 0)
    assert numa_table([{"lin": (0, 8)}, {"lin": (0, 16), "xo": (0, 64)}], 0) == widest
    assert numa_table([{"lin": (0, 16), "xo": (0, 64)}, {"lin": (0, 8)}], 0) == widest
