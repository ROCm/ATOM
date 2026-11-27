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
from atom.plugin import prepare as plugin_prepare

LARGE_CAPACITY = 16384
SMALL_M_MTPR = 32
N_LAYERS = 61
# DSpark builds 3 draft stages, numbered after the target layers.
N_DRAFT = 3


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


def _layer(value=0, layer_id=0, *, pcp_merge=False):
    return SimpleNamespace(
        layer_id=layer_id,
        prefix=f"model.layers.{layer_id}.ffn.experts",
        moe_parallel_config=SimpleNamespace(pcp_merge=pcp_merge),
        **{
            name: torch.full((48, 4), value, dtype=torch.uint8)
            for name in ("_mega_w1", "_mega_w1_scale", "_mega_w2", "_mega_w2_scale")
        },
    )


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        manager=SimpleNamespace(rank=0, world_size=8),
        forward_context=SimpleNamespace(
            context=None,
            ubatch_slices=None,
            in_hipgraph=False,
        ),
        tbo=False,
        capturing=False,
        heap_initialized=False,
        events=[],
        instances={},
        layer=_layer(),
        small_m_mtpr=SMALL_M_MTPR,
        pcp=1,
    )
    monkeypatch.setenv("ATOM_MEGA_DECODE_FAST_PATH", "1")
    monkeypatch.setattr(plugin_prepare, "_CURRENT_FRAMEWORK", "atom")
    monkeypatch.setattr(mega, "_MEGA_CACHE", {})
    monkeypatch.setattr(mega, "_MEGA_CAPACITY_LOGGED", set())
    monkeypatch.setattr(mega, "_MEGA_LAYER_IDS", set())
    monkeypatch.setattr(mega, "_MEGA_LAYERS_WITHOUT_ID", [])
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
        """AITER's MegaMoEV2 (ROCm/aiter#7788): builds its fused routes with
        routed=True and picks one per call from the max_tokens hint, which
        must lie in [this rank's rows, capacity]."""

        def __init__(self, *, routed, large_m_state_slots, **kwargs):
            assert state.heap_initialized
            assert not state.capturing, "constructor reached during capture"
            assert routed is True
            self.rank = kwargs["rank"]
            self.capacity = kwargs["max_tok_per_rank"]
            self.large_m_state_slots = large_m_state_slots
            self.small_m_mtpr = state.small_m_mtpr
            self.kwargs = kwargs
            state.events.append(("build", self.rank, self.capacity))
            state.instances[self.rank, self.capacity] = self

        @property
        def large_m_available(self):
            return self.capacity == LARGE_CAPACITY

        def select_layer_state(self, layer_slot):
            assert self.large_m_available
            # AITER refuses a slot outside the ones it built.
            if not 0 <= layer_slot < self.large_m_state_slots:
                raise ValueError(
                    f"layer_slot={layer_slot} outside [0, {self.large_m_state_slots})"
                )
            state.events.append(("slot", self.rank, layer_slot))

        def forward(self, x, weights, ids, *, mask_invalid_slots, max_tokens):
            # Pad rows are routed to -1, so combine must skip those slots.
            assert mask_invalid_slots is True
            if max_tokens is not None and not (
                x.shape[0] <= max_tokens <= self.capacity
            ):
                raise ValueError(
                    f"max_tokens={max_tokens} must be in "
                    f"[{x.shape[0]} (this rank's rows), {self.capacity}]"
                )
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
    stub_module(
        "atom.distributed.pcp_utils",
        pcp_pad_len=lambda rows: -(-rows // state.pcp) * state.pcp,
    )

    def run(
        context,
        rows=None,
        *,
        rank=0,
        world=8,
        experts=384,
        mtpr=LARGE_CAPACITY,
        layer=None,
        **kwargs,
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
            **{"swiglu_limit": 0.0, **kwargs},
        )

    # Target layers, then the draft model's: both are built before any forward.
    for layer_id in range(N_LAYERS + N_DRAFT):
        mega.register_mega_layer(_layer(layer_id=layer_id))
    state.run = run
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
    assert [e for e in runtime.events if e[0] in ("build", "forward")] == [
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
    runtime.run(context, rows=rows)

    assert _events(runtime, "hint") == [hint]


def test_tbo_and_plugin_steps_take_no_route_hint(runtime, monkeypatch):
    runtime.tbo = True
    runtime.run(_context(64))
    runtime.tbo = False
    monkeypatch.setattr(plugin_prepare, "_CURRENT_FRAMEWORK", "sglang")
    runtime.run(_context(64))

    assert _events(runtime, "hint") == [None, None]


def test_capture_hints_the_rows_as_eager_does(runtime):
    # The route depends on the rows alone: a capture and its warmup hint what
    # eager hints whether or not in_hipgraph is set (a draft capture leaves it
    # unset), so the warmup compiles exactly what the recording launches,
    # large-input route included.
    runtime.run(_context(SMALL_M_MTPR + 1))
    runtime.forward_context.in_hipgraph = True
    runtime.run(_context(SMALL_M_MTPR))
    runtime.run(_context(SMALL_M_MTPR + 1))
    runtime.run(_context(SMALL_M_MTPR + 1, is_draft=True))
    runtime.capturing = True
    runtime.run(_context(SMALL_M_MTPR + 1))
    runtime.run(_context(SMALL_M_MTPR + 1, is_draft=True))

    assert _events(runtime, "hint") == [
        SMALL_M_MTPR + 1,
        SMALL_M_MTPR,
        SMALL_M_MTPR + 1,
        SMALL_M_MTPR + 1,
        SMALL_M_MTPR + 1,
        SMALL_M_MTPR + 1,
    ]


def test_each_layer_selects_the_large_input_slot_of_its_layer_id(runtime):
    # Target layers and the draft layers after them (MTP / DSpark number them
    # num_hidden_layers + i) each own one slot, whatever order they first run in.
    ids = (2, 0, N_LAYERS + N_DRAFT - 1, N_LAYERS, 1)
    layers = [_layer(value, layer_id) for value, layer_id in enumerate(ids)]
    for layer in layers + layers:
        runtime.run(_context(512, running_tokens_are_unified=False), layer=layer)
    # The decode capacity has no large-input route and selects no slot.
    runtime.run(_context(64), layer=layers[0])

    assert {
        capacity: instance.large_m_state_slots
        for (_, capacity), instance in runtime.instances.items()
    } == {LARGE_CAPACITY: N_LAYERS + N_DRAFT, 128: N_LAYERS + N_DRAFT}
    assert _events(runtime, "slot") == list(ids) * 2


def test_slots_cover_the_layer_ids_built_and_no_more(runtime, monkeypatch):
    # A target without a draft model: the slots end at its last layer, and a
    # layer id nobody registered is refused by AITER rather than aliased.
    monkeypatch.setattr(mega, "_MEGA_LAYER_IDS", set())
    for layer_id in range(N_LAYERS):
        mega.register_mega_layer(_layer(layer_id=layer_id))
    runtime.run(_context(512, running_tokens_are_unified=False))

    assert runtime.instances[0, LARGE_CAPACITY].large_m_state_slots == N_LAYERS
    with pytest.raises(ValueError, match=f"layer_slot={N_LAYERS} outside"):
        runtime.run(
            _context(512, running_tokens_are_unified=False),
            layer=_layer(layer_id=N_LAYERS),
        )


def test_a_layer_without_a_layer_id_is_refused_by_the_large_input_route(runtime):
    # The DeepSeek-V3/R1 MTP block keeps layer_id None: an instance without the
    # large-input route selects no slot and serves it; one with it is refused
    # before it is cached, the same on every rank.
    mega.register_mega_layer(
        SimpleNamespace(layer_id=None, prefix="model.layers.61.mlp.experts")
    )
    runtime.run(_context(64), mtpr=128)
    assert _forwarded(runtime) == [128]

    with pytest.raises(ValueError, match="layer_id.*'model.layers.61.mlp.experts'"):
        runtime.run(_context(512, running_tokens_are_unified=False))
    assert (0, LARGE_CAPACITY) not in mega._MEGA_CACHE


@pytest.mark.parametrize("pcp", [2, 4])
@pytest.mark.parametrize("prefill_rows", [101, 102, 4093])
def test_pcp_merge_hints_the_gathered_padded_rows(runtime, pcp, prefill_rows):
    # Under ATOM_PCP_MOE_MERGE a split prefill gathers every PCP shard of the
    # step's rows padded to a PCP multiple; other passes hold the step's rows.
    # Every PCP rank hints the padded count, which bounds both.
    runtime.pcp = pcp
    layer = _layer(pcp_merge=True)
    padded = -(-prefill_rows // pcp) * pcp
    context = _context(
        prefill_rows,
        running_tokens_are_unified=False,
        running_tokens_across_dp=(prefill_rows,),
        is_prefill=True,
    )
    runtime.run(context, rows=padded, layer=layer)
    runtime.run(context, rows=prefill_rows, layer=layer)

    assert _events(runtime, "hint") == [padded, padded]


@pytest.mark.parametrize(("rows", "capacity"), [(126, 128), (127, LARGE_CAPACITY)])
def test_pcp_merge_selects_the_decode_capacity_by_the_padded_rows(
    runtime, rows, capacity
):
    # The decode capacity must hold the padded rows the hint names: with PCP 3,
    # 127 rows pad to 129 > 128 and stay on the configured capacity.
    runtime.pcp = 3
    padded = -(-rows // 3) * 3
    runtime.run(_context(rows), rows=padded, layer=_layer(pcp_merge=True))

    assert _forwarded(runtime) == [capacity]
    assert _events(runtime, "hint") == [padded]


def test_unpadded_hint_is_refused_by_the_route_bounds(runtime):
    # Without pcp_merge the hint is the step's own rows: a gathered, padded
    # input above it is outside AITER's [run_tokens, mtpr] bounds.
    runtime.pcp = 2
    context = _context(
        101, running_tokens_are_unified=False, running_tokens_across_dp=(101,)
    )
    with pytest.raises(ValueError, match="max_tokens=101 must be in \\[102"):
        runtime.run(context, rows=102)


@pytest.mark.parametrize(
    ("small_m_mtpr", "decode_mtpr", "builds_decode"),
    [
        (0, 128, True),  # no small-input route: the decode capacity serves
        (SMALL_M_MTPR, 128, True),
        (512, 128, False),
        (512, 512, False),
        (512, 1024, True),
    ],
)
def test_decode_capacity_is_skipped_when_the_small_route_covers_it(
    runtime, monkeypatch, small_m_mtpr, decode_mtpr, builds_decode
):
    runtime.small_m_mtpr = small_m_mtpr
    monkeypatch.setattr(mega, "_MEGA_DECODE_MTPR", decode_mtpr)
    for rank in range(2):
        runtime.run(_context(64), rank=rank)

    expected = [LARGE_CAPACITY, decode_mtpr] if builds_decode else [LARGE_CAPACITY]
    assert _built(runtime, 0) == _built(runtime, 1) == expected
    assert _forwarded(runtime) == [expected[-1]] * 2
    # The configured capacity routes the step by its hint.
    assert _events(runtime, "hint") == [64, 64]


@pytest.mark.parametrize(
    ("context", "hint"),
    [
        # dp_size 1 never publishes a draft shape: the table still holds the
        # target's rows, and the draft's own unified count is the bound.
        (_context(6, is_draft=True, running_tokens_across_dp=(4096,)), 6),
        (
            _context(
                6,
                is_draft=True,
                running_tokens_are_unified=False,
                running_tokens_across_dp=(6, 18),
            ),
            18,
        ),
    ],
)
def test_draft_pass_hints_its_own_rows(runtime, context, hint):
    runtime.run(context, layer=_layer(layer_id=N_LAYERS))

    assert _events(runtime, "hint") == [hint]


@pytest.mark.parametrize(
    ("arch", "fused"),
    [("gfx950", True), ("gfx950:sramecc+:xnack-", True), ("gfx942", False)],
)
def test_fused_route_is_decided_by_aiter_query(monkeypatch, arch, fused):
    layout = {
        "world_size": 8,
        "model_dim": 7168,
        "inter_dim": 3072,
        "experts": 384,
        "topk": 6,
        "quant": "a8w4",
        "swiglu_limit": 10.0,
    }
    queried = []

    def fused_layout_supported(*, arch, **kwargs):
        queried.append(kwargs)
        return arch.split(":", 1)[0] == "gfx950"

    def stub_module(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    stub_module("aiter.jit.utils.chip_info", get_gfx_runtime=lambda: arch)
    stub_module(
        "aiter.ops.flydsl.kernels.mega_moe.mega_moe_config",
        fused_layout_supported=fused_layout_supported,
    )

    assert mega.mega_builds_fused_route(**layout) is fused
    assert queried == [layout]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("num_cu", [256, 304])
@pytest.mark.parametrize("mtpr", [128, 1024, LARGE_CAPACITY, 32768])
@pytest.mark.parametrize("decode_mtpr", [128, 1024])
def test_fused_route_query_covers_every_instance_built(
    runtime, monkeypatch, fused, num_cu, mtpr, decode_mtpr
):
    # The instances run_mega_moe builds, each deciding its routes the way
    # AITER's MegaMoEV2.__init__ does (routed=True): a fused route in any of
    # them -- the decode capacity's large-input one included -- must be
    # answered by mega_builds_fused_route, and on 256-CU parts (the small-input
    # route) no other answer is given.
    monkeypatch.setattr(mega, "_MEGA_DECODE_MTPR", decode_mtpr)
    large_m_mtprs = (1024, 2048, 4096, 8192, 16384)

    def fused_routes(capacity):
        small = fused and num_cu == 256
        large = fused and capacity in large_m_mtprs
        return small, large

    built = []
    real_get_or_build = mega.get_or_build_mega_moe

    def get_or_build(**kwargs):
        instance = real_get_or_build(**kwargs)
        small, large = fused_routes(kwargs["mtpr"])
        instance.small_m_mtpr = min(512, kwargs["mtpr"]) if small else 0
        instance.large_m = large
        built.append(instance)
        return instance

    monkeypatch.setattr(mega, "get_or_build_mega_moe", get_or_build)
    runtime.run(_context(64), mtpr=mtpr)
    holds_fused = any(i.small_m_mtpr or fused_routes(i.capacity)[1] for i in built)

    assert not holds_fused or fused
    if num_cu == 256:
        assert holds_fused is fused


def test_without_model_settings_the_build_is_upstreams(runtime):
    runtime.run(_context(64))

    assert runtime.instances
    for instance in runtime.instances.values():
        assert instance.kwargs["act"] == "silu"
        # An aiter before combine_wire takes the build unchanged.
        assert "combine_wire" not in instance.kwargs


def test_cache_key_separates_activations_limits_and_wires(runtime):
    builds = [
        ("silu", 7.0, "auto"),
        ("swigluoai", 7.0, "auto"),
        ("swigluoai", 10.0, "auto"),
        ("swigluoai", 7.0, "bf16"),
    ]
    for act, limit, wire in builds * 2:
        runtime.run(
            _context(64),
            world=4,
            experts=128,
            act=act,
            swiglu_limit=limit,
            combine_wire=wire,
        )

    # One instance per (act, limit, wire), each built once and reused after.
    assert len(_built(runtime)) == len(builds)
    assert sorted(
        (
            m.kwargs["act"],
            m.kwargs["swiglu_limit"],
            m.kwargs.get("combine_wire", "auto"),
        )
        for m in mega._MEGA_CACHE.values()
    ) == sorted(builds)
