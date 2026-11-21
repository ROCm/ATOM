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


def test_each_launch_owns_its_queue_words():
    span = len(k2b.QUEUES) * 4
    # every tag a launch can take, the last one too
    starts = [
        k2b.queue_offset(MOE_TAG0 + layer) for layer in range(k2b.TAGS - MOE_TAG0)
    ]
    assert all(b - a >= span for a, b in itertools.pairwise(starts))
    assert starts[-1] + span <= k2b.QUEUE_BYTES
