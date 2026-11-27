# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""DeepSeek-V4.1 mono K2b's task queues (``kernels.moe.QUEUES``): a count each
launch, taken by atomic add from zero. The layer builds of one step share the
step scratch but lay it out apart -- an indexing layer's K2a holds regions a
plain layer's does not, so K2b's regions sit elsewhere -- and one build's
tagged pairs at another's queue offset start that count past zero: the launch
skips its first units and waits for ever on what they would have written (the
133K-context hang, 2026-11-20). The queues live outside every build's layout,
a launch its own words. CPU only."""

import itertools

import pytest

pytest.importorskip("flydsl")

from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.models.deepseek_v41.mono.kernels import moe as k2b
from atom.models.deepseek_v41.mono.runner import MOE_TAG0


def _post_key(rows, tp, bound):
    return k2.LayerPostBuild(
        tokens=rows, index_bound_max=bound, index_fp4=True,
        index_mfma=bool(bound), experts=384, topk=6, tp=tp, ug_groups=1,
        kv_fp8=True, shard=tp > 1,
    )  # fmt: skip


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("rows", [6, 12, 24, 48])
@pytest.mark.parametrize("bound", [0, 1 << 20])
def test_no_build_lays_a_queue_word_out_in_the_scratch(rows, tp, bound):
    layout = k2.scratch_layout(_post_key(rows, tp, bound))
    assert not set(k2b.QUEUES) & set(layout)


CACHE_LINE = 128


def test_each_launch_owns_its_queue_words():
    # every tag a launch can take, the last one too: a word a queue region
    starts = [
        k2b.queue_offset(MOE_TAG0 + layer) for layer in range(k2b.TAGS - MOE_TAG0)
    ]
    assert all(b - a >= 4 for a, b in itertools.pairwise(starts))
    assert starts[-1] + 4 <= k2b.QUEUE_REGION
    last = starts[-1] + (len(k2b.QUEUES) - 1) * k2b.QUEUE_REGION
    assert last + 4 <= k2b.QUEUE_BYTES


def test_a_launchs_queue_words_sit_in_separate_cache_lines():
    # its CTAs' atomics on one queue do not contend with the others'
    for layer in range(k2b.TAGS - MOE_TAG0):
        base = k2b.queue_offset(MOE_TAG0 + layer)
        lines = {
            (base + i * k2b.QUEUE_REGION) // CACHE_LINE for i in range(len(k2b.QUEUES))
        }
        assert len(lines) == len(k2b.QUEUES)
