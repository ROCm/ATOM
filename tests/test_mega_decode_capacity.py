# SPDX-License-Identifier: MIT
"""Capacity changes must preserve Mega's cross-rank wire protocol.

Exercise the real host dispatch and instance cache with CPU tensors. Only the
AITER constructor/kernel and per-forward distributed context are replaced;
these tests do not claim to validate GPU collectives or graph replay numerics.
"""

import sys
from itertools import product
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="MegaMoE imports its Triton row-zeroing kernel")

from atom.model_ops.fused_moe import flydsl_mega_experts as mega
from atom import config as atom_config
from atom.plugin import prepare as plugin_prepare

LARGE_CAPACITY = 16384
SMALL_M_MTPR = 32


def _context(rows, **overrides):
    values = {
        "running_bs": 64,
        "running_tokens": rows,
        "running_tokens_are_unified": True,
        "running_tokens_across_dp": None,
        "is_prefill": False,
        "is_dummy_run": False,
        "is_draft": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _layer(value=0):
    return SimpleNamespace(
        **{
            name: torch.full((48, 4), value, dtype=torch.uint8)
            for name in ("_mega_w1", "_mega_w1_scale", "_mega_w2", "_mega_w2_scale")
        }
    )


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        manager=SimpleNamespace(rank=0, world_size=8),
        forward_context=SimpleNamespace(
            context=None, ubatch_slices=None, in_hipgraph=False
        ),
        tbo=False,
        capturing=False,
        heap_initialized=False,
        events=[],
        instances={},
        layer=_layer(),
    )
    monkeypatch.setenv("ATOM_MEGA_DECODE_FAST_PATH", "1")
    monkeypatch.setattr(plugin_prepare, "_CURRENT_FRAMEWORK", "atom")
    monkeypatch.setattr(mega, "_MEGA_CACHE", {})
    monkeypatch.setattr(mega, "_MEGA_CAPACITY_LOGGED", set())
    monkeypatch.setattr(mega, "_MEGA_LAYER_SLOTS", {})
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )

    class Communicator:
        @property
        def all2all_manager(self):
            # The real lazy property initializes MORI's symmetric heap.
            state.heap_initialized = True
            return state.manager

    ep_group = SimpleNamespace(device_communicator=Communicator())

    class FakeMegaMoEV2:
        def __init__(self, **kwargs):
            assert state.heap_initialized
            assert not state.capturing, "constructor reached during capture"
            self.rank = kwargs["rank"]
            self.capacity = kwargs["max_tok_per_rank"]
            self.kwargs = kwargs
            state.events.append(("build", self.rank, self.capacity))
            state.instances[self.rank, self.capacity] = self

        def forward(self, x, weights, ids, *, mask_invalid_slots):
            # Pad rows are routed to -1, so combine must skip those slots.
            assert mask_invalid_slots is True
            state.events.append(("forward", self.rank, self.capacity))
            return torch.full_like(x, self.capacity)

    class RoutedMegaMoEV2(FakeMegaMoEV2):
        """An AITER build that picks its route from a max_tokens hint."""

        small_m_mtpr = SMALL_M_MTPR

        @property
        def large_m_available(self):
            return self.capacity == LARGE_CAPACITY

        def select_layer_state(self, layer_slot):
            assert self.large_m_available
            state.events.append(("slot", self.rank, layer_slot))

        def forward(self, x, weights, ids, *, mask_invalid_slots, max_tokens=None):
            assert mask_invalid_slots is True
            state.events.append(("forward", self.rank, self.capacity))
            state.events.append(("hint", self.rank, max_tokens))
            return torch.full_like(x, self.capacity)

    def stub_module(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    stub_module("aiter.dist.parallel_state", get_ep_group=lambda: ep_group)
    stub_module("aiter.ops.flydsl.kernels.mega_moe", MegaMoEV2=FakeMegaMoEV2)
    stub_module(
        "atom.utils.forward_context", get_forward_context=lambda: state.forward_context
    )
    stub_module("atom.utils.tbo.ubatching", tbo_active=lambda: state.tbo)

    def run(
        context,
        rows=None,
        *,
        rank=0,
        world=8,
        experts=384,
        mtpr=LARGE_CAPACITY,
        layer=None,
    ):
        state.manager.rank = rank
        state.manager.world_size = world
        state.forward_context.context = context
        if rows is None:
            rows = context.running_tokens
        return mega.run_mega_moe(
            state.layer if layer is None else layer,
            torch.zeros(rows, 8, dtype=torch.bfloat16),
            torch.ones(rows, 6, dtype=torch.float32),
            torch.zeros(rows, 6, dtype=torch.int64),
            model_dim=8,
            inter_dim=8,
            experts=experts,
            topk=6,
            mtpr=mtpr,
            swiglu_limit=0.0,
        )

    def use_routed_aiter(layers=61):
        sys.modules["aiter.ops.flydsl.kernels.mega_moe"].MegaMoEV2 = RoutedMegaMoEV2
        config = SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=layers))
        monkeypatch.setattr(
            atom_config, "get_current_atom_config", lambda: config, raising=False
        )

    state.run = run
    state.use_routed_aiter = use_routed_aiter
    return state


def _built(runtime, rank=0):
    return [
        capacity
        for event, pe, capacity in runtime.events
        if event == "build" and pe == rank
    ]


def _forwarded(runtime):
    return [capacity for event, _, capacity in runtime.events if event == "forward"]


def test_mtp_verification_and_intermediate_draft_use_different_capacities(runtime):
    # B=64, K=3: target and draft step 0 process 256 token rows. Only the
    # intermediate serial draft steps reduce to one row per request.
    target = _context(256)
    draft_first = _context(256, is_draft=True, running_tokens_are_unified=False)
    draft_next = _context(64, is_draft=True)

    for context in (target, draft_first, draft_next, target):
        runtime.run(context)

    assert _forwarded(runtime) == [LARGE_CAPACITY, LARGE_CAPACITY, 128, LARGE_CAPACITY]
    assert _built(runtime) == [LARGE_CAPACITY, 128]


def test_padded_context_prevents_a_local_tensor_from_switching_protocol(runtime):
    # A scheduled-token table can be smaller than the graph's padded height;
    # draft contexts can also retain the target's old table.
    for rank, local_rows in enumerate((64, 256)):
        runtime.run(
            _context(256, running_tokens_across_dp=(64,) * 8),
            rows=local_rows,
            rank=rank,
        )

    assert _forwarded(runtime) == [LARGE_CAPACITY, LARGE_CAPACITY]
    assert _built(runtime, 0) == _built(runtime, 1) == [LARGE_CAPACITY, 128]

    runtime.run(_context(64, is_draft=True, running_tokens_across_dp=(256,) * 8))
    assert _forwarded(runtime)[-1] == 128


def test_rank_local_parent_flags_cannot_split_a_unified_draft(runtime):
    for rank, (dummy, prefill) in enumerate(product((False, True), repeat=2)):
        runtime.run(
            _context(64, is_draft=True, is_dummy_run=dummy, is_prefill=prefill),
            rank=rank,
        )

    assert _forwarded(runtime) == [128] * 4


def test_mixed_rank_shapes_keep_one_protocol_and_construction_order(runtime):
    for rank, rows in enumerate((64, 512)):
        runtime.run(
            _context(rows, running_tokens_are_unified=False, is_prefill=rank == 1),
            rank=rank,
        )

    assert _forwarded(runtime) == [LARGE_CAPACITY, LARGE_CAPACITY]
    assert _built(runtime, 0) == _built(runtime, 1) == [LARGE_CAPACITY, 128]

    # The later unified pass must find both instances already allocated,
    # including on the rank whose first local batch was small.
    runtime.capturing = True
    for rank in range(2):
        runtime.run(_context(64), rank=rank)
    assert _forwarded(runtime)[-2:] == [128, 128]


def test_tbo_child_does_not_allocate_small_workspace(runtime):
    # Child ForwardContext.ubatch_slices is None and Context.unified defaults
    # to True. Only the active TBO thread tells dispatch this is a microbatch.
    runtime.tbo = True
    runtime.run(_context(64))
    assert _built(runtime) == [LARGE_CAPACITY]
    assert _forwarded(runtime) == [LARGE_CAPACITY]

    runtime.tbo = False
    runtime.run(_context(64))
    assert _built(runtime) == [LARGE_CAPACITY, 128]
    assert _forwarded(runtime) == [LARGE_CAPACITY, 128]


@pytest.mark.parametrize(
    ("enabled", "world", "experts", "mtpr"),
    [
        ("0", 8, 384, LARGE_CAPACITY),
        ("1", 4, 192, LARGE_CAPACITY),
        ("1", 8, 392, LARGE_CAPACITY),
        ("1", 8, 256, LARGE_CAPACITY),
        ("1", 8, 384, 128),
        ("1", 8, 384, 64),
    ],
)
def test_disabled_or_unsupported_cases_keep_the_original_instance(
    runtime, monkeypatch, enabled, world, experts, mtpr
):
    monkeypatch.setenv("ATOM_MEGA_DECODE_FAST_PATH", enabled)
    runtime.run(_context(32), world=world, experts=experts, mtpr=mtpr)

    assert _built(runtime) == [mtpr]
    assert _forwarded(runtime) == [mtpr]


@pytest.mark.parametrize("framework", ["vllm", "sglang", "sgl", "rtpllm"])
@pytest.mark.parametrize("prefill", [False, True])
def test_plugin_context_cannot_enable_native_capacity_selection(
    runtime, monkeypatch, framework, prefill
):
    monkeypatch.setattr(plugin_prepare, "_CURRENT_FRAMEWORK", framework)
    # Some bridges report requests rather than token rows for prefill while
    # inheriting unified=True. The native fast path must not trust that shape.
    runtime.run(
        _context(1 if prefill else 64, is_prefill=prefill),
        rows=512 if prefill else 64,
    )

    assert _built(runtime) == [LARGE_CAPACITY]
    assert _forwarded(runtime) == [LARGE_CAPACITY]


@pytest.mark.parametrize(
    ("rows", "expected_capacity"),
    [(0, LARGE_CAPACITY), (1, 128), (128, 128), (129, LARGE_CAPACITY)],
)
def test_capacity_boundary_uses_the_published_row_count(
    runtime, rows, expected_capacity
):
    output = runtime.run(_context(rows), rows=max(1, rows))

    assert _forwarded(runtime) == [expected_capacity]
    assert torch.all(output == expected_capacity)


@pytest.mark.parametrize("decode_mtpr", [256, 512, 1024])
def test_configured_decode_capacity_moves_the_boundary(
    runtime, monkeypatch, decode_mtpr
):
    monkeypatch.setattr(mega, "_MEGA_DECODE_MTPR", decode_mtpr)
    runtime.run(_context(decode_mtpr))
    runtime.run(_context(decode_mtpr + 1))

    assert _forwarded(runtime) == [decode_mtpr, LARGE_CAPACITY]
    assert _built(runtime) == [LARGE_CAPACITY, decode_mtpr]


def test_missing_context_retains_original_capacity(runtime):
    runtime.run(None, rows=64)
    assert _forwarded(runtime) == [LARGE_CAPACITY]


def test_tensor_overflow_raises_without_a_rank_local_fallback(runtime):
    with pytest.raises(ValueError, match="DP-agreed.*capacity=128"):
        runtime.run(_context(64), rows=129)

    assert _forwarded(runtime) == []


def test_warmup_builds_both_capacities_before_any_forward_or_capture(runtime):
    # Target capture warmup sets in_hipgraph early; draft capture can leave it
    # False. Actual stream capture state must govern allocation in both cases.
    runtime.forward_context.in_hipgraph = True
    runtime.run(_context(256))
    assert runtime.events == [
        ("build", 0, LARGE_CAPACITY),
        ("build", 0, 128),
        ("forward", 0, LARGE_CAPACITY),
    ]

    runtime.forward_context.in_hipgraph = False
    runtime.capturing = True
    runtime.run(_context(64, is_draft=True))
    assert _built(runtime) == [LARGE_CAPACITY, 128]
    assert _forwarded(runtime) == [LARGE_CAPACITY, 128]


@pytest.mark.parametrize("large_already_built", [False, True])
def test_capture_cache_miss_is_rejected_before_symmetric_allocation(
    runtime, monkeypatch, large_already_built
):
    if large_already_built:
        monkeypatch.setenv("ATOM_MEGA_DECODE_FAST_PATH", "0")
        runtime.run(_context(256))
        monkeypatch.setenv("ATOM_MEGA_DECODE_FAST_PATH", "1")

    before_capture = list(runtime.events)
    runtime.capturing = True
    missing_capacity = 128 if large_already_built else LARGE_CAPACITY
    with pytest.raises(
        RuntimeError, match=f"capacity {missing_capacity}.*before graph capture"
    ):
        runtime.run(_context(64))

    assert runtime.events == before_capture


def test_cached_capacities_rebind_layer_weights_without_breaking_eplb_aliases(runtime):
    runtime.run(_context(256))
    next_layer = _layer(value=7)
    runtime.capturing = True
    runtime.run(_context(64), layer=next_layer)

    assert _built(runtime) == [LARGE_CAPACITY, 128]
    for instance in runtime.instances.values():
        assert instance._s1_w1.data_ptr() == next_layer._mega_w1.data_ptr()
        assert instance._s1_w1_scale.data_ptr() == next_layer._mega_w1_scale.data_ptr()
        assert instance.w2 is next_layer._mega_w2
        assert instance.w2_scale is next_layer._mega_w2_scale

    next_layer._mega_w1[0].fill_(9)
    for instance in runtime.instances.values():
        assert torch.all(instance._s1_w1[0] == 9)


def _events(runtime, kind):
    return [value for event, _, value in runtime.events if event == kind]


def test_aiter_without_route_hints_keeps_the_staged_call(runtime):
    runtime.run(_context(256))

    assert all(
        "large_m_state_slots" not in instance.kwargs
        for instance in runtime.instances.values()
    )
    assert _events(runtime, "hint") == _events(runtime, "slot") == []


@pytest.mark.parametrize(
    ("context", "rows", "hint"),
    [
        (_context(64), 64, 64),
        (_context(256), 200, 256),
        (_context(64, running_tokens_across_dp=(256,) * 8), 64, 64),
        (
            _context(
                512,
                running_tokens_are_unified=False,
                running_tokens_across_dp=(512, 4096),
            ),
            512,
            4096,
        ),
        (_context(512, running_tokens_are_unified=False), 512, None),
        (_context(0), 1, None),
        (None, 64, None),
    ],
)
def test_route_hint_is_the_dp_group_max_rows(runtime, context, rows, hint):
    runtime.use_routed_aiter()
    runtime.run(context, rows=rows)

    assert _events(runtime, "hint") == [hint]


def test_tbo_and_plugin_steps_take_no_route_hint(runtime, monkeypatch):
    runtime.use_routed_aiter()
    runtime.tbo = True
    runtime.run(_context(64))
    runtime.tbo = False
    monkeypatch.setattr(plugin_prepare, "_CURRENT_FRAMEWORK", "sglang")
    runtime.run(_context(64))

    assert _events(runtime, "hint") == [None, None]


def test_capture_hints_only_the_small_input_route(runtime):
    runtime.use_routed_aiter()
    runtime.run(_context(256))
    runtime.capturing = True
    runtime.run(_context(SMALL_M_MTPR))
    runtime.run(_context(SMALL_M_MTPR + 1))

    assert _events(runtime, "hint") == [256, SMALL_M_MTPR, None]


def test_each_layer_selects_its_own_large_input_slot(runtime):
    runtime.use_routed_aiter(layers=61)
    layers = [_layer(value) for value in range(3)]
    for layer in layers + layers:
        runtime.run(_context(512, running_tokens_are_unified=False), layer=layer)
    # The decode capacity has no large-input route and selects no slot.
    runtime.run(_context(64), layer=layers[2])

    assert {
        capacity: instance.kwargs["large_m_state_slots"]
        for (_, capacity), instance in runtime.instances.items()
    } == {LARGE_CAPACITY: 69, 128: 69}
    assert _events(runtime, "slot") == [0, 1, 2, 0, 1, 2]
