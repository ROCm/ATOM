# SPDX-License-Identifier: MIT
"""V2 STATE and graph contracts, collected by CPU CI without importing vLLM."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

BRIDGE = Path(__file__).resolve().parents[1] / "atom/plugin/vllm/deepseek_v41_bridge.py"
OVERRIDES = BRIDGE.with_name("v41_core_overrides.py")


@pytest.fixture
def scratch_allocator():
    cls = next(
        node
        for node in ast.parse(OVERRIDES.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "VllmDeepseekV41MetadataBuilder"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_capture_dummy_cache"
    )

    def allocate(geometry, pages, slots, device, *, max_tokens, workspace):
        return SimpleNamespace(num_pages=pages, num_slots=slots, max_tokens=max_tokens)

    scope = {"PagedAttentionCache": Mock(side_effect=allocate)}
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=[method], type_ignores=[]), str(OVERRIDES), "exec"),
        scope,
    )
    builder = SimpleNamespace(
        geometry=object(),
        block_size=32,
        device="cpu",
        score_workspace=None,
        max_num_batched_tokens=96,
        max_bs=4,
        model_runner=SimpleNamespace(v41_graphs_enabled=True),
    )
    return builder, scope[method.name].__get__(builder), scope["PagedAttentionCache"]


def test_capture_scratch_covers_per_request_tail_pages(scratch_allocator):
    builder, allocate, constructor = scratch_allocator
    first = allocate(1, 1, 1)
    # An artificial helper ordering, not a claim about V2's capture order:
    # equal token totals can need different PAGE counts (48+48 vs. 32+64).
    for lengths, expected_pages in [((48, 48), 4), ((32, 64), 3)]:
        pages = sum(-(-length // builder.block_size) for length in lengths)
        assert pages == expected_pages
        cache = allocate(pages, len(lengths), sum(lengths))
        assert cache is first
        assert cache.num_pages >= pages
        assert cache.num_slots >= len(lengths)
        assert cache.max_tokens >= sum(lengths)
    constructor.assert_called_once()


@pytest.mark.parametrize("dimension", ["num_pages", "num_slots", "max_tokens"])
def test_capture_scratch_rejects_growth_without_replacing_storage(
    scratch_allocator, dimension
):
    builder, allocate, constructor = scratch_allocator
    first = allocate(1, 1, 1)
    capacities = {
        "num_pages": first.num_pages,
        "num_slots": first.num_slots,
        "max_tokens": first.max_tokens,
    }
    request = dict(capacities)
    request[dimension] += 1
    with pytest.raises(RuntimeError, match="capture scratch capacity"):
        allocate(*request.values())
    assert builder._capture_dummy is first
    assert vars(first) == capacities
    constructor.assert_called_once()
    assert allocate(1, 1, 1) is first


@pytest.mark.parametrize("pages,expected_pages", [(9, 11), (20, 20)])
def test_capture_scratch_respects_larger_initial_request(
    scratch_allocator, pages, expected_pages
):
    _, allocate, constructor = scratch_allocator
    first = allocate(pages, 6, 160)
    assert (first.num_pages, first.num_slots, first.max_tokens) == (
        expected_pages,
        6,
        160,
    )
    assert allocate(pages, 6, 160) is first
    constructor.assert_called_once()


def test_eager_scratch_remains_per_call_after_capture(scratch_allocator):
    builder, allocate, constructor = scratch_allocator
    first = allocate(1, 1, 1)
    builder.model_runner.v41_graphs_enabled = False
    for request in [(1, 1, 1), (12, 8, 192)]:
        cache = allocate(*request)
        assert cache is not first and builder._capture_dummy is first
        assert (cache.num_pages, cache.num_slots, cache.max_tokens) == request
    assert constructor.call_count == 3


def _functions(*names, **namespace):
    tree = ast.parse(BRIDGE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    scope = {"np": np, "SimpleNamespace": SimpleNamespace, **namespace}
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
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


@pytest.mark.parametrize(
    "bound,profiling,capturing,staged_on",
    [
        (False, False, False, None),
        (False, False, True, None),
        (False, True, False, None),
        (False, True, True, "scratch"),
        (True, False, False, "serving"),
        (True, False, True, "serving"),
        (True, True, False, "serving"),
        (True, True, True, "serving"),
    ],
)
def test_proxy_factory_stages_unbound_profiling_full_only(
    bound, profiling, capturing, staged_on
):
    # Execute the production entry method without importing optional GPU/vLLM
    # dependencies. The full metadata/forward integration lives in tests/plugin.
    cls = next(
        node
        for node in ast.parse(BRIDGE.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "AtomDeepseekV41ProxyMetadataBuilder"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_and_attach"
    )
    model, builder, prepared, snapshot = object(), object(), object(), object()
    stage = Mock(return_value=prepared)
    proxy = SimpleNamespace(
        _atom_v41_model=model,
        _atom_v41_builder=builder,
        _atom_v4_profiling_kv_cache=profiling,
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"proxy": proxy})
    )
    scope = {
        "ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME": "proxy",
        "snapshot_v41_batch": lambda _: snapshot,
        "bind_deepseek_v41_proxy_cache": lambda *_: bound,
        "_v41_stage_outside_forward": stage,
    }
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=[method], type_ignores=[]), str(BRIDGE), "exec"), scope
    )
    common = SimpleNamespace()
    returned = scope[method.name](
        SimpleNamespace(vllm_config=config), common, capturing=capturing
    )
    assert returned is common and common.atom_v41_snapshot is snapshot
    if staged_on is None:
        stage.assert_not_called()
        assert not hasattr(common, "atom_v41_prepared")
    else:
        stage.assert_called_once_with(
            builder,
            model,
            config,
            snapshot,
            capturing=capturing,
            profiling=staged_on == "scratch",
        )
        assert common.atom_v41_prepared is prepared
