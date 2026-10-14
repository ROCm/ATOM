# SPDX-License-Identifier: MIT
"""V2 STATE and graph contracts, collected by CPU CI without importing vLLM."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BRIDGE = Path(__file__).resolve().parents[1] / "atom/plugin/vllm/deepseek_v41_bridge.py"


def _functions(*names, **namespace):
    tree = ast.parse(BRIDGE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    scope = {"np": np, "SimpleNamespace": SimpleNamespace, **namespace}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(BRIDGE), "exec"), scope
    )
    return SimpleNamespace(**{name: scope[name] for name in names})


def _snapshot(ids, slots, positions=None):
    count = len(ids)
    positions = np.asarray(positions if positions is not None else [8] * count)
    return SimpleNamespace(
        num_reqs=count,
        req_ids=ids,
        state_slots=None if slots is None else np.asarray(slots),
        query_lens=np.ones(count, dtype=np.int32),
        num_computed=positions,
        ends=positions + 1,
        block_rows=[(i + 1,) for i in range(count)],
        total_tokens=count,
    )


def test_state_slots_follow_v2_lifetime_not_the_scheduled_subset():
    functions = _functions("_v41_request_state_slots", "_v41_scheduled_batch")
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2))
    # A remains live while only B runs. C reuses B's released request index,
    # never A's older (but still owned) STATE slot.
    for ids, slots in [(["A", "B"], [0, 1]), (["B"], [1]), (["C"], [1]), (["A"], [0])]:
        snapshot = _snapshot(ids, slots)
        mapping = functions._v41_request_state_slots(snapshot, config)
        batch = functions._v41_scheduled_batch(snapshot, slots=mapping)
        assert list(batch.state_slots_committed) == slots
    # Position-zero recomputation still reaches the cache's normal reset path.
    snapshot = _snapshot(["A"], [0], positions=[0])
    batch = functions._v41_scheduled_batch(
        snapshot, slots=functions._v41_request_state_slots(snapshot, config)
    )
    assert list(batch.context_lens - batch.num_scheduled_tokens) == [0]


@pytest.mark.parametrize("slots", [None, [0], [0, 0], [-1, 1], [0, 2], [0.0, 1.0]])
def test_invalid_real_state_mapping_fails_closed(slots):
    functions = _functions("_v41_request_state_slots")
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2))
    with pytest.raises(ValueError, match="STATE"):
        functions._v41_request_state_slots(_snapshot(["A", "B"], slots), config)


def test_state_mapping_is_a_snapshot_not_a_borrowed_host_view():
    functions = _functions("_v41_request_state_slots")
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2))
    snapshot = _snapshot(["B", "A"], [1, 0])
    slots = functions._v41_request_state_slots(snapshot, config)
    snapshot.state_slots[:] = 0
    assert list(slots) == [1, 0]


@pytest.mark.parametrize("capturing,full_graph", [(True, False), (False, True)])
@pytest.mark.parametrize("active,padded,width", [(4, 4, 1), (2, 4, 1), (2, 4, 6)])
def test_full_preparation_uses_bucket_capacity(
    capturing, full_graph, active, padded, width
):
    functions = _functions("_v41_prepare_shape")
    snapshot = SimpleNamespace(
        num_reqs=active,
        num_reqs_padded=padded,
        max_query_len=width,
        full_graph=full_graph,
    )
    batch = SimpleNamespace(num_scheduled_tokens=[width] * active)
    running_bs, kwargs = functions._v41_prepare_shape(
        snapshot, batch, {"tentative": width > 1}, capturing=capturing
    )
    assert running_bs == padded
    assert kwargs == {"tentative": width > 1, "max_q_len": width}


def test_piecewise_keeps_tight_prefill_planning():
    functions = _functions("_v41_prepare_shape")
    snapshot = SimpleNamespace(
        num_reqs=2, num_reqs_padded=4, max_query_len=128, full_graph=False
    )
    batch = SimpleNamespace(num_scheduled_tokens=[128, 1])
    assert functions._v41_prepare_shape(snapshot, batch, {}, capturing=False) == (2, {})


def test_full_capture_never_defers_engram_join_even_with_mode_none():
    # The explicit capture signal must win before consulting runtime mode:
    # V2 passes NONE while preparing an ordinary FULL graph.
    functions = _functions("_v41_join_covers_the_layers")
    assert not functions._v41_join_covers_the_layers(capturing=True)
