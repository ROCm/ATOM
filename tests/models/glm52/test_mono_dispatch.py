# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: which steps are routed."""

from pathlib import Path

from atom.models.glm52.mono.config import AGENTX_BATCHES, GRAPH_BATCHES


def test_only_batch_one_and_two_are_routed():
    # Review Focus 2: batches 3..96 build but must not be routed. If this ever
    # widens, the GSM8K gate at concurrency 1 stops covering what ships.
    assert AGENTX_BATCHES == (1, 2)
    assert set(AGENTX_BATCHES) < set(GRAPH_BATCHES)
    for batch in (3, 4, 8, 96):
        assert batch in GRAPH_BATCHES and batch not in AGENTX_BATCHES


def test_routing_reads_nothing_rank_local():
    # Review Focus 3: a TP-divergent routing decision hangs the other ranks
    # in-kernel with no error and no log.
    source = (
        Path(__file__).parents[3] / "atom" / "models" / "glm52" / "mono" / "dispatch.py"
    ).read_text()
    routing = source[source.index("def supports") :]
    for rank_local in ("is_dummy_run", "get_tensor_model_parallel_rank"):
        assert rank_local not in routing, rank_local
