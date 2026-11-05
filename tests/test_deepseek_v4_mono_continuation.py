# SPDX-License-Identifier: MIT
"""A padded decode tail must keep its live-row execution on every step."""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

MONO = Path(__file__).parents[1] / "atom/models/deepseek_v4/mono"


@pytest.fixture
def session_types():
    # Exercise the production session class and frontier/profile definitions
    # without importing the model, cache adapter or GPU kernel builders.
    namespace = {
        **runpy.run_path(str(MONO / "contract.py")),
        **runpy.run_path(str(MONO / "profile.py")),
    }
    path = MONO / "session.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    declarations = [
        node
        for node in tree.body
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        or (isinstance(node, ast.ClassDef) and node.name == "V4MonoSession")
    ]
    exec(  # noqa: S102 -- executes the production class, without its GPU imports
        compile(ast.Module(body=declarations, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("live_rows", [3, 4])
def test_continuation_reselects_live_rows_after_forward_clears_route(
    live_rows, session_types
):
    # The first C3 tail step already selected eager execution under the C4
    # capture key. Model forward clears that selection before the next step.
    key = (4, 1)
    mono = session_types.V4MonoSession.__new__(session_types.V4MonoSession)
    mono.profile = session_types.ORDINARY_PROFILE
    mono.owner_generation = 1
    mono.graph_keys = {key}
    mono.graph_generations = {key: 1}
    mono.graph_layouts = {key: (4, ())}
    mono.graph_is_current = lambda candidate: candidate == key
    mono.slot_generations = {slot: 1 for slot in range(live_rows)}
    mono.batch_continuation = session_types.BatchContinuation(
        1,
        key,
        1,
        tuple(
            session_types.ContinuationRow(slot, 1, 400, (0, 1))
            for slot in range(live_rows)
        ),
    )
    mono.builder = SimpleNamespace(
        model_runner=SimpleNamespace(capture_sizes=[1, 2, 4])
    )
    mono.runner = SimpleNamespace(
        _lane_graphs={},
        index_policy=lambda: {"refresh_csa_layers": 30, "reuse_csa_layers": 0},
    )
    mono.finish_capture_publication = lambda: None
    mono.eager_layout_selected = mono.lane_graph_selected = False
    mono.decode_attempts = mono.selected_steps = mono.microbucket_stacks = 0
    mono.index_refresh_executions = mono.index_reuse_executions = 0
    mono.index_suppressed_rows = 0
    mono.fast_publications_by_width = {}
    bucket = SimpleNamespace(width=live_rows, tokens_per_seq=1, ordinal=0)
    adapter = mono.adapter = SimpleNamespace(
        position=401,
        prepared_widths=(1, 2, 3, 4),
        published_buckets=((0, bucket),),
        graph_layout=lambda: (live_rows, ()),
    )
    adapter.continuation_contract = lambda context: tuple(
        (adapter.position, slot, (0, 1)) for slot in range(live_rows)
    )

    def write(*args, **kwargs):
        adapter.published_contract = tuple(
            (123, adapter.position, slot, (0, 1)) for slot in range(live_rows)
        )

    adapter.write = write
    context = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=False,
            is_dummy_run=False,
            running_bs=4,
            scheduled_bs=live_rows,
        ),
        attn_metadata=SimpleNamespace(max_seqlen_q=1),
    )

    for _ in range(3):
        assert mono.publish(None, None, context)
        # Engine.prepare_step may replay the FULL C4 graph only when its
        # captured rows match the live publication. A C3 tail must stay eager.
        assert mono.eager_layout_selected is (live_rows < key[0])
        assert not mono.lane_graph_selected
        mono.eager_layout_selected = False  # session.forward's finally block
        adapter.position += 1

    assert mono.selected_steps == 3
    assert mono.fast_publications_by_width == {4: 3}
