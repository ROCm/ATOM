# SPDX-License-Identifier: MIT
"""Slot-indexed MoE route capture: scatter, gather, CUDA-graph pad slots."""

from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atom.config import Config, check_return_routed_experts
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.model_ops.fused_moe import routed_experts_capturer as cap_mod
from atom.model_ops.fused_moe.routed_experts_capturer import (
    UNCAPTURED,
    RoutedExpertsCapturer,
    RoutedExpertsPatch,
    capture_bytes_per_kv_block,
    capture_row_bytes,
    maybe_capture_routed_experts,
    merge_routed_patches,
    prefix_slots,
)
from atom.utils import forward_context as fc


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _segments(patch: RoutedExpertsPatch) -> list[tuple[int, int, np.ndarray]]:
    off = patch.offsets.tolist()
    return [
        (int(r), int(s), patch.rows[off[j] : off[j + 1]])
        for j, (r, s) in enumerate(zip(patch.req_ids, patch.starts))
    ]


def _store_take(capturer, slots, req_ids, starts, lens, **kw):
    """The step's patch as the scheduler receives it: rows on the host."""
    patch = capturer.gather_step(slots, req_ids, starts, lens, **kw)
    return patch._replace(rows=patch.rows.cpu().numpy())


@contextmanager
def _forward(slots):
    """A forward whose attention metadata carries ``slots``."""
    fc._forward_context = fc.ForwardContext(
        attn_metadata=SimpleNamespace(slot_mapping=slots)
    )
    try:
        yield
    finally:
        fc.reset_forward_context()


def _capture(capturer, layer_id, ids, slots, top_k=None):
    """What a FusedMoE bound to ``layer_id`` does in a forward over ``slots``."""
    layer = SimpleNamespace(top_k=capturer.top_k if top_k is None else top_k)
    capturer.bind(layer, layer_id)
    with _forward(slots):
        maybe_capture_routed_experts(layer, ids)


def test_config_flag_defaults_off():
    names = {f.name for f in fields(Config)}
    assert "enable_return_routed_experts" in names
    field = next(f for f in fields(Config) if f.name == "enable_return_routed_experts")
    assert field.default is False


def _r3_config(**overrides):
    base = {
        "enable_return_routed_experts": True,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "sequence_parallel_size": 1,
        "kv_transfer_config": {},
        "enable_rapidserve": False,
        "plugin_config": None,
    }
    return SimpleNamespace(**{**base, **overrides})


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"decode_context_parallel_size": 2}, "decode_context_parallel_size"),
        ({"prefill_context_parallel_size": 2}, "prefill_context_parallel_size"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
        ({"sequence_parallel_size": 2}, "sequence_parallel_size"),
        ({"kv_transfer_config": {"kv_connector": "moriio"}}, "KV transfer"),
        ({"enable_rapidserve": True}, "never initializes"),
        ({"plugin_config": SimpleNamespace(is_plugin_mode=True)}, "plugin mode"),
    ],
)
def test_unsupported_topologies_fail_closed(overrides, match):
    with pytest.raises(ValueError, match=match):
        check_return_routed_experts(_r3_config(**overrides))


def test_servers_refuse_the_flag():
    """No server endpoint returns routes, so a server must not pay for them."""
    from atom.model_engine.arg_utils import EngineArgs, refuse_server_routed_experts
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    refuse_server_routed_experts(parser, parser.parse_args([]))
    with pytest.raises(SystemExit):
        refuse_server_routed_experts(
            parser, parser.parse_args(["--enable-return-routed-experts"])
        )


def test_supported_topologies_pass():
    check_return_routed_experts(_r3_config())
    check_return_routed_experts(_r3_config(enable_dp_attention=True))
    check_return_routed_experts(
        _r3_config(enable_return_routed_experts=False, pipeline_parallel_size=2)
    )


def _import_runner():
    return pytest.importorskip(
        "atom.model_engine.model_runner",
        reason="model_runner imports aiter at module load",
        exc_type=ImportError,
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_only_tp_rank0_binds_the_capture_buffer(rank, monkeypatch):
    """Every rank plans the same geometry; only rank 0 exports, so only its
    MoE layers get a view and the other ranks' hook stays a no-op."""
    mr = _import_runner()
    moes = [
        SimpleNamespace(prefix=f"model.layers.{i}.mlp.experts", top_k=2) for i in (0, 2)
    ]
    monkeypatch.setattr(mr, "fused_moe_modules", lambda model: moes)
    runner = mr.ModelRunner.__new__(mr.ModelRunner)
    runner.model = object()
    runner._routed_experts_geo = (3, 2)
    runner._routed_experts_capturer = None
    runner.rank = rank
    runner.block_size = 4
    runner.device = "cpu"

    runner._bind_routed_experts_layers(num_kvcache_blocks=2)

    capturer = runner._routed_experts_capturer
    if rank == 0:
        assert capturer.buffer.shape == (9, 3, 2)
        for moe, layer in zip(moes, (0, 2)):
            view = getattr(moe, cap_mod.ROUTES_VIEW_ATTR)
            assert view.data_ptr() == capturer.buffer[:, layer].data_ptr()
    else:
        assert capturer is None
        assert not any(hasattr(m, cap_mod.ROUTES_VIEW_ATTR) for m in moes)


@pytest.mark.parametrize("rank", [0, 1])
def test_kv_cross_check_expects_only_the_rows_a_rank_allocates(rank, monkeypatch):
    """Every rank is charged the capture rows but only rank 0 allocates them,
    so the other ranks' cross-check must not expect them."""
    mr = _import_runner()
    moes = [SimpleNamespace(prefix="model.layers.0.mlp.experts", top_k=2)]
    monkeypatch.setattr(mr, "fused_moe_modules", lambda model: moes)
    runner = mr.ModelRunner.__new__(mr.ModelRunner)
    runner.model = object()
    runner._routed_experts_geo = (3, 2)
    runner._routed_experts_capturer = None
    runner.rank = rank
    runner.block_size = 4
    runner.device = "cpu"
    kv_bytes = 1 << 20
    blocks = 2
    capture = blocks * capture_bytes_per_kv_block(4, 3, 2)
    runner.pool_plan = SimpleNamespace(total_reserved_bytes=kv_bytes + capture)

    runner._bind_routed_experts_layers(num_kvcache_blocks=blocks)
    capturer = runner._routed_experts_capturer
    # What the allocation adds on top of the KV pools (the pad row is charged
    # outside the plan).
    allocated = (
        0 if capturer is None else capturer.buffer.nbytes - capture_row_bytes(3, 2)
    )
    assert runner._expected_kv_bytes(blocks) == kv_bytes + allocated


def test_flag_off_leaves_field_absent():
    seq = Sequence([1, 2, 3], 3)
    assert seq.routed_experts is None


def test_two_request_batch_order_segments():
    """A step's rows come back in batch token order, one segment per request."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=64, num_layers=3, top_k=2, device=device)
    # Request A on block 0, request B on block 1, two prefill tokens each.
    slots = torch.tensor([0, 1, 16, 17], device=device)
    ids = torch.tensor(
        [[10, 11], [12, 13], [20, 21], [22, 23]], dtype=torch.int32, device=device
    )
    _capture(capturer, 0, ids, slots)
    _capture(capturer, 1, ids + 100, slots)
    patch = _store_take(capturer, slots, [7, 9], [0, 0], [2, 2])
    assert patch.rows.dtype == np.int16
    assert patch.rows.shape == (4, 3, 2)
    (ra, sa, a), (rb, sb, b) = _segments(patch)
    assert (ra, sa, rb, sb) == (7, 0, 9, 0)
    np.testing.assert_array_equal(a[:, 0, :], [[10, 11], [12, 13]])
    np.testing.assert_array_equal(a[:, 1, :], [[110, 111], [112, 113]])
    np.testing.assert_array_equal(b[:, 0, :], [[20, 21], [22, 23]])

    # Next step decodes A alone at position 2.
    slots2 = torch.tensor([2], device=device)
    ids2 = torch.tensor([[14, 15]], dtype=torch.int32, device=device)
    _capture(capturer, 0, ids2, slots2)
    patch2 = _store_take(capturer, slots2, [7], [2], [1])
    [(r, s, rows)] = _segments(patch2)
    assert (r, s) == (7, 2)
    np.testing.assert_array_equal(rows[:, 0, :], [[14, 15]])


def test_patches_rebuild_request_history():
    seq = Sequence([1, 2, 3], 3)
    seq.id = 7
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=32, num_layers=1, top_k=1, device=device)
    for start, n in ((0, 3), (3, 1), (4, 1)):
        slots = torch.arange(start, start + n, device=device)
        ids = slots.to(torch.int32).unsqueeze(-1) + 40
        _capture(capturer, 0, ids, slots)
        patch = _store_take(capturer, slots, [7], [start], [n])
        for _r, s, rows in _segments(patch):
            seq.apply_routed_expert_patch(s, rows)
    assert seq.routed_experts[:, 0, 0].tolist() == [40, 41, 42, 43, 44]


def test_length_contract_prompt_plus_completion_minus_one():
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=32, num_layers=1, top_k=1, device=device)
    prompt_len, completion_len = 4, 2
    # Routes exist for every forwarded token: all prompt tokens plus every
    # completion token except the last sampled id (never forwarded).
    n_routes = prompt_len + completion_len - 1
    slots = torch.arange(n_routes, device=device)
    ids = torch.arange(n_routes, device=device, dtype=torch.int32).unsqueeze(-1)
    _capture(capturer, 0, ids, slots)
    patch = _store_take(capturer, slots, [1], [0], [n_routes])
    assert patch.rows.shape[0] == prompt_len + completion_len - 1


def test_graph_dummy_does_not_clobber_slot_zero():
    """Trailing -1 must not restore the pre-write value of physical slot 0."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=2, device=device)
    capturer.buffer[:, 0, :] = -7
    slots = torch.tensor([0, -1], device=device)
    ids = torch.tensor([[9, 8], [1, 2]], dtype=torch.int32, device=device)
    _capture(capturer, 0, ids, slots)
    assert capturer.buffer[0, 0].tolist() == [9, 8]
    assert capturer.buffer[capturer.num_slots, 0].tolist() == [1, 2]


def test_graph_dummy_negative_slots_ignored():
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=2, device=device)
    capturer.buffer[:, 0, :] = -7
    slots = torch.tensor([-1, -1, 3], device=device)
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device=device)
    _capture(capturer, 0, ids, slots)
    # Pad rows must not clobber slot 0.
    got0 = capturer.buffer[0, 0].tolist()
    assert got0 == [-7, -7]
    got3 = capturer.buffer[3, 0].tolist()
    assert got3 == [5, 6]


def test_prefix_hit_segment_reads_cached_slots():
    """B prefix-hits A's block: its cached rows come first, then its new token."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=32, num_layers=1, top_k=2, device=device)
    slots = torch.tensor([0, 1, 2], device=device)
    ids = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int32, device=device)
    _capture(capturer, 0, ids, slots)
    a = _store_take(capturer, slots, [1], [0], [3])

    b_slots = torch.tensor([3], device=device)
    _capture(
        capturer,
        0,
        torch.tensor([[5, 6]], dtype=torch.int32, device=device),
        b_slots,
    )
    prefix = (
        np.array([2]),
        np.array([0]),
        np.array([3]),
        prefix_slots([0, 1], 0, 3, 16),
    )
    b = _store_take(capturer, b_slots, [2], [3], [1], prefix=prefix)
    (r0, s0, cached), (r1, s1, new) = _segments(b)
    assert (r0, s0, r1, s1) == (2, 0, 2, 3)
    np.testing.assert_array_equal(cached, a.rows)
    np.testing.assert_array_equal(new[:, 0, :], [[5, 6]])


def test_every_quant_methods_routing_is_captured(monkeypatch):
    """Quant methods' `select_experts` all go through `_select_logical`, so
    its capture covers FusedMoE and the comm-fused path alike."""
    moe = pytest.importorskip(
        "atom.model_ops.moe",
        reason="moe imports aiter at module load",
        exc_type=ImportError,
    )
    topk_weights = torch.rand(2, 2)
    topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    monkeypatch.setattr(
        moe.FusedMoE,
        "select_experts",
        staticmethod(lambda **_: (topk_weights, topk_ids)),
    )
    captured = []
    monkeypatch.setattr(
        moe,
        "maybe_capture_routed_experts",
        lambda layer, ids: captured.append((layer, ids)),
    )
    layer = SimpleNamespace(
        use_grouped_topk=False,
        top_k=2,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
        routed_scaling_factor=1.0,
    )

    out = moe._select_logical(layer, torch.zeros(2, 4), torch.zeros(2, 8), 8, 0)

    assert out == (topk_weights, topk_ids)
    assert captured == [(layer, topk_ids)]


def test_maybe_capture_skips_an_unbound_layer():
    """Outside a forward, so reading slots would raise: it must not get that far."""
    layer = SimpleNamespace(top_k=2)
    maybe_capture_routed_experts(layer, torch.zeros((2, 2), dtype=torch.int32))


def test_moe_hook_writes_its_layer_at_this_forwards_slots():
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=2, top_k=2, device=device)
    layer = SimpleNamespace(top_k=2)
    capturer.bind(layer, 1)
    with _forward(torch.tensor([3], device=device)):
        maybe_capture_routed_experts(
            layer, torch.tensor([[5, 6]], dtype=torch.int32, device=device)
        )
    assert capturer.buffer[3, 1].tolist() == [5, 6]
    assert (capturer.buffer[3, 0] == UNCAPTURED).all()


def test_moe_outside_the_target_model_never_captures():
    """A drafter's MoE has a `layer_id` but is never bound a view."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=2, top_k=2, device=device)

    class _DraftMoE:
        layer_id = 0

    with _forward(torch.tensor([0], device=device)):
        maybe_capture_routed_experts(
            _DraftMoE(), torch.tensor([[5, 6]], dtype=torch.int32, device=device)
        )
    assert (capturer.buffer == UNCAPTURED).all()


def _forward_metadata(monkeypatch, attn_metadata):
    from types import SimpleNamespace

    ctx = SimpleNamespace(attn_metadata=attn_metadata)
    monkeypatch.setattr(cap_mod, "get_forward_context", lambda: ctx)


def test_current_slot_mapping_reads_the_builder_metadata(monkeypatch):
    from types import SimpleNamespace

    slots = torch.tensor([3, 4])
    _forward_metadata(monkeypatch, SimpleNamespace(slot_mapping=slots))
    assert cap_mod.current_slot_mapping() is slots


@pytest.mark.parametrize("kind", ["none", "empty-dict", "per-layer-dict", "no-slots"])
def test_current_slot_mapping_refuses_without_one_slot_space(monkeypatch, kind):
    from types import SimpleNamespace

    attn_metadata = {
        "none": None,
        "empty-dict": {},
        "per-layer-dict": {
            "full_attn": SimpleNamespace(slot_mapping=torch.tensor([0, 1])),
            "gdn": SimpleNamespace(slot_mapping=torch.tensor([7, 8])),
        },
        "no-slots": object(),
    }[kind]
    _forward_metadata(monkeypatch, attn_metadata)
    with pytest.raises(RuntimeError, match="no single per-token slot_mapping"):
        cap_mod.current_slot_mapping()


def test_prefix_slots_match_gpu_formula():
    table = [4, 9, 2, 7]
    pos = torch.arange(50, dtype=torch.long)
    blocks = torch.tensor(table, dtype=torch.long)
    full = (blocks[pos // 16] * 16 + (pos % 16)).tolist()
    for lo, hi in ((0, 18), (5, 16), (15, 17), (16, 50), (33, 34)):
        assert prefix_slots(table, lo, hi, 16).tolist() == full[lo:hi]


def test_gather_step_drops_graph_pad_tail():
    """Graph-padded slot_mapping: only the step's real tokens are returned."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=2, device=device)
    slots = torch.tensor([0, -1, -1], device=device)
    _capture(
        capturer,
        0,
        torch.tensor([[9, 8], [1, 2], [3, 4]], dtype=torch.int32, device=device),
        slots,
    )
    patch = _store_take(capturer, slots, [1], [0], [1])
    np.testing.assert_array_equal(patch.rows[:, 0, :], [[9, 8]])


def test_uncaptured_rows_are_minus_one():
    """Layers or slots no MoE layer wrote read back as UNCAPTURED, not expert 0."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=2, top_k=2, device=device)
    slots = torch.tensor([0, 1], device=device)
    _capture(
        capturer,
        0,
        torch.tensor([[7, 8], [9, 10]], dtype=torch.int32, device=device),
        slots,
    )
    patch = _store_take(capturer, slots, [1], [0], [2])
    np.testing.assert_array_equal(patch.rows[:, 0, :], [[7, 8], [9, 10]])
    assert (patch.rows[:, 1, :] == UNCAPTURED).all()


def test_short_slot_mapping_raises_at_store():
    """A metadata builder exposing fewer slots than tokens is an error, not -1 rows."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=1, device=device)
    slots = torch.tensor([2], device=device)
    _capture(capturer, 0, torch.tensor([[5]], dtype=torch.int32, device=device), slots)
    with pytest.raises(
        RuntimeError, match="forwarded 3 tokens but its slot_mapping has 1"
    ):
        capturer.gather_step(slots, [1, 2], [0, 0], [2, 1])


@pytest.mark.parametrize("slot_count", [2, 4])
def test_capture_raises_when_topk_rows_and_slots_differ(slot_count):
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=2, device="cpu")
    with pytest.raises(RuntimeError, match="routed 3 tokens"):
        _capture(
            capturer,
            0,
            torch.zeros((3, 2), dtype=torch.int32),
            torch.arange(slot_count),
        )


@pytest.mark.parametrize("layer_id", [-1, 2])
def test_capture_raises_on_layer_outside_buffer(layer_id):
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=2, top_k=2, device="cpu")
    with pytest.raises(RuntimeError, match="outside the buffer's 2 layers"):
        _capture(
            capturer,
            layer_id,
            torch.zeros((1, 2), dtype=torch.int32),
            torch.tensor([0]),
        )


def test_capture_raises_when_fewer_columns_than_layer_top_k():
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=4, device="cpu")
    with pytest.raises(RuntimeError, match="top_k=4 with 3 top-k columns"):
        _capture(
            capturer,
            0,
            torch.zeros((1, 3), dtype=torch.int32),
            torch.tensor([0]),
            top_k=4,
        )


def test_capture_drops_fused_shared_columns_past_layer_top_k():
    """A top-2 layer with one fused shared column, in a buffer sized for a top-3
    layer: the shared id must not land in the third route column."""
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=3, device="cpu")
    slots = torch.tensor([1])
    _capture(capturer, 0, torch.tensor([[4, 7, 99]], dtype=torch.int32), slots, top_k=2)
    patch = _store_take(capturer, slots, [1], [0], [1])
    assert patch.rows[0, 0].tolist() == [4, 7, UNCAPTURED]


def test_empty_step_gathers_nothing():
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=2, device="cpu")
    assert capturer.gather_step(torch.tensor([], dtype=torch.long), [], [], []) is None
    # Outside a forward there are no slots to read: refused, not an empty step.
    with pytest.raises(RuntimeError, match="no single per-token slot_mapping"):
        capturer.gather_step(None, [1], [0], [1])


def test_gathered_steps_are_independent():
    """A later gather neither replaces nor reorders an earlier one."""
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=1, device="cpu")
    _capture(capturer, 0, torch.tensor([[4]], dtype=torch.int32), torch.tensor([1]))
    a = capturer.gather_step(torch.tensor([1]), [1], [0], [1])
    _capture(capturer, 0, torch.tensor([[6]], dtype=torch.int32), torch.tensor([2]))
    b = capturer.gather_step(torch.tensor([2]), [2], [0], [1])
    assert b.req_ids.tolist() == [2]
    assert a.req_ids.tolist() == [1]
    assert a.rows[:, 0, 0].tolist() == [4]
    assert b.rows[:, 0, 0].tolist() == [6]


def test_gather_snapshots_slots_before_staging_reuse():
    """The next forward rewrites slot_mapping in place on the compute stream,
    behind the gather: the gathered rows are this step's."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=1, top_k=2, device=device)
    slots = torch.tensor([1], device=device)
    _capture(
        capturer, 0, torch.tensor([[7, 8]], dtype=torch.int32, device=device), slots
    )
    patch = capturer.gather_step(slots, [1], [0], [1])
    slots.fill_(2)
    _capture(
        capturer, 0, torch.tensor([[3, 4]], dtype=torch.int32, device=device), slots
    )
    np.testing.assert_array_equal(patch.rows.cpu()[:, 0, :], [[7, 8]])


def _prefix_batch(block_tables):
    from types import SimpleNamespace

    return SimpleNamespace(
        req_ids=[7, 9],
        block_tables=block_tables,
        routed_prefix_ranges=[(1, 0, 4)],
    )


def test_cached_prefix_slots_read_the_requests_own_table():
    req_ids, starts, lens, slots = cap_mod.batch_prefix_slots(
        _prefix_batch([[0], [5]]), block_size=4
    )
    assert req_ids.tolist() == [9] and starts.tolist() == [0] and lens.tolist() == [4]
    assert slots.tolist() == [20, 21, 22, 23]


def test_cached_prefix_slots_refuse_a_dropped_block_table():
    """`ScheduledBatch` omits an empty table, which shifts every later row onto
    another request's blocks; the lookup refuses rather than read them."""
    with pytest.raises(RuntimeError, match="looked up by batch position"):
        cap_mod.batch_prefix_slots(_prefix_batch([[5]]), block_size=4)


def test_reused_slot_keeps_dense_layers_uncaptured():
    """Layer 1 is dense: no request writes it, so a reused slot reads -1."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=3, top_k=2, device=device)
    slot = torch.tensor([3], device=device)
    for ids in ([[1, 2]], [[5, 6]]):  # request A, then B on the same slot
        for layer in (0, 2):
            _capture(
                capturer,
                layer,
                torch.tensor(ids, dtype=torch.int32, device=device),
                slot,
            )
    row = capturer.buffer[3].cpu().tolist()
    assert row == [[5, 6], [UNCAPTURED, UNCAPTURED], [5, 6]]


def test_capture_buffer_matches_its_budget():
    """The pool charge and the allocation come from one element type."""
    block_size, num_blocks, layers, top_k = 16, 3, 5, 4
    capturer = RoutedExpertsCapturer(
        num_slots=block_size * num_blocks,
        num_layers=layers,
        top_k=top_k,
        device="cpu",
    )
    assert capturer.buffer.dtype == cap_mod.CAPTURE_DTYPE
    assert capturer.buffer.nbytes == num_blocks * capture_bytes_per_kv_block(
        block_size, layers, top_k
    ) + capture_row_bytes(layers, top_k)


def test_capture_page_bytes_are_budgeted_per_block():
    item = cap_mod.CAPTURE_DTYPE.itemsize
    assert capture_bytes_per_kv_block(16, 58, 8) == 16 * 58 * 8 * item
    from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
        PAGED_CLASS,
        page_pool,
        plan_pools,
    )

    kv = page_pool(1024)
    capture = page_pool(capture_bytes_per_kv_block(16, 2, 2))
    plan = plan_pools([kv, capture], available_bytes=1024 * 10 + 256, max_num_seqs=1)
    assert plan.entry_bytes[PAGED_CLASS] == 1024 + capture_bytes_per_kv_block(16, 2, 2)


def test_lazy_wrapper_fused_moe_is_detected(monkeypatch):
    """Plugin LazyMoEWrapper instances are not isinstance(FusedMoE).

    Do not import ``atom.model_ops.moe``: FusedMoE pulls AITER, which needs a
    GPU (rocminfo) even to load the class.
    """
    import sys
    import types

    from torch import nn

    from atom.model_ops.fused_moe.routed_experts_capturer import is_fused_moe_module

    class InnerFusedMoE(nn.Module):
        pass

    class LazyMoEWrapper(InnerFusedMoE):
        def __new__(cls, *args, **kwargs):
            return InnerFusedMoE()

    fake_moe = types.ModuleType("atom.model_ops.moe")
    fake_moe.FusedMoE = LazyMoEWrapper
    monkeypatch.setitem(sys.modules, "atom.model_ops.moe", fake_moe)

    module = InnerFusedMoE()
    assert is_fused_moe_module(module)
    assert not isinstance(module, LazyMoEWrapper)
    assert not is_fused_moe_module(nn.Identity())


def test_is_fused_moe_module_undecorated_rejects_plain_modules(monkeypatch):
    import sys
    import types

    from torch import nn

    from atom.model_ops.fused_moe.routed_experts_capturer import is_fused_moe_module

    class FusedMoE(nn.Module):
        pass

    fake_moe = types.ModuleType("atom.model_ops.moe")
    fake_moe.FusedMoE = FusedMoE
    monkeypatch.setitem(sys.modules, "atom.model_ops.moe", fake_moe)

    assert is_fused_moe_module(FusedMoE())
    assert not is_fused_moe_module(nn.Identity())
    assert not is_fused_moe_module(nn.Linear(2, 2))


def test_in_place_slot_rewrite_between_forwards():
    """The staging slot buffer is rewritten in place between forwards; each
    forward's layers write and read that forward's slots."""
    device = _device()
    capturer = RoutedExpertsCapturer(num_slots=32, num_layers=3, top_k=2, device=device)
    staging = torch.tensor([0, 1, -1], device=device)
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device=device)
    for layer in range(3):
        _capture(capturer, layer, ids + 10 * layer, staging)
    first = _store_take(capturer, staging, [0], [0], [2])
    staging.copy_(torch.tensor([2, 3, -1], device=device))
    for layer in range(3):
        _capture(capturer, layer, ids + 100 + 10 * layer, staging)
    second = _store_take(capturer, staging, [0], [2], [2])
    for layer in range(3):
        np.testing.assert_array_equal(
            first.rows[:, layer], (ids[:2] + 10 * layer).cpu()
        )
        np.testing.assert_array_equal(
            second.rows[:, layer], (ids[:2] + 100 + 10 * layer).cpu()
        )
    # Pad slot (-1) lands in the sacrificial row, never a real slot.
    assert (capturer.buffer[4:32] == UNCAPTURED).all()


def test_apply_patch_extends_none_and_appends():
    seq = Sequence([1, 2, 3], 3)
    assert seq.routed_experts is None
    first = np.arange(4, dtype=np.int16).reshape(2, 1, 2)
    seq.apply_routed_expert_patch(0, first)
    np.testing.assert_array_equal(seq.routed_experts, first)
    second = np.arange(4, 6, dtype=np.int16).reshape(1, 1, 2)
    seq.apply_routed_expert_patch(2, second)
    assert seq.routed_expert_rows == 3
    np.testing.assert_array_equal(
        seq.routed_experts, np.concatenate([first, second], axis=0)
    )


def test_apply_patch_gap_raises():
    seq = Sequence([1, 2, 3], 3)
    seq.apply_routed_expert_patch(0, np.zeros((2, 1, 1), dtype=np.int16))
    with pytest.raises(RuntimeError, match="positions 2..3 were never returned"):
        seq.apply_routed_expert_patch(4, np.full((1, 1, 1), 5, dtype=np.int16))
    assert seq.routed_expert_rows == 2


def test_apply_patch_overlap_overwrites_and_keeps_high_water():
    seq = Sequence([1, 2, 3], 3)
    first = np.arange(6, dtype=np.int16).reshape(3, 1, 2)
    seq.apply_routed_expert_patch(0, first)
    replacement = np.array([[[9, 8]]], dtype=np.int16)
    seq.apply_routed_expert_patch(1, replacement)
    assert seq.routed_expert_rows == 3
    np.testing.assert_array_equal(seq.routed_experts[1:2], replacement)
    np.testing.assert_array_equal(seq.routed_experts[2:], first[2:])


def test_truncate_routed_experts_drops_tail():
    seq = Sequence([1, 2, 3], 3)
    seq.apply_routed_expert_patch(0, np.arange(8, dtype=np.int16).reshape(4, 1, 2))
    seq.truncate_routed_experts(2)
    assert seq.routed_experts.shape[0] == 2
    seq.truncate_routed_experts(0)
    assert seq.routed_experts.shape == (0, 1, 2)


def test_finish_to_zero_rows_keeps_shape():
    """A one-token prompt that emits nothing still returns [0, layers, top_k]."""
    seq = Sequence([1], 3)
    seq.apply_routed_expert_patch(0, np.zeros((1, 2, 2), dtype=np.int16))
    seq.finish_routed_experts(0)
    routes = seq.routed_experts
    assert routes.shape == (0, 2, 2)
    assert routes.dtype == np.int16


def test_apply_patch_deferred_overlap_keeps_prefix():
    """PP=1 deferred attach reports start = have-1; overwrite last row, keep prefix."""
    seq = Sequence([1, 2, 3], 3)
    first = np.arange(6, dtype=np.int16).reshape(3, 1, 2)
    seq.apply_routed_expert_patch(0, first)
    nxt = np.array([[[90, 91], [92, 93]]], dtype=np.int16).reshape(2, 1, 2)
    seq.apply_routed_expert_patch(seq.routed_expert_rows - 1, nxt)
    assert seq.routed_expert_rows == 4
    np.testing.assert_array_equal(seq.routed_experts[:2], first[:2])
    np.testing.assert_array_equal(seq.routed_experts[2:], nxt)


def test_apply_patch_does_not_alias_caller_array():
    seq = Sequence([1, 2, 3], 3)
    full = np.arange(12, dtype=np.int16).reshape(3, 2, 2)
    seq.apply_routed_expert_patch(0, full)
    seq.apply_routed_expert_patch(2, np.full((1, 2, 2), 99, dtype=np.int16))
    np.testing.assert_array_equal(full[2], [[8, 9], [10, 11]])
    np.testing.assert_array_equal(seq.routed_experts[2], [[99, 99], [99, 99]])


def test_apply_patch_grows_capacity_across_many_decode_steps():
    seq = Sequence([1, 2, 3], 3)
    seq.apply_routed_expert_patch(0, np.zeros((8, 2, 2), dtype=np.int16))
    for i in range(8, 64):
        seq.apply_routed_expert_patch(
            seq.routed_expert_rows - 1,
            np.full((2, 2, 2), i, dtype=np.int16),
        )
    assert seq.routed_expert_rows == 64
    out = seq.routed_experts
    assert out.shape == (64, 2, 2)
    np.testing.assert_array_equal(out[-2:], np.full((2, 2, 2), 63, dtype=np.int16))
    assert seq._routed_expert_buf.shape[0] >= 64


def _grown_seq(rows: int) -> Sequence:
    seq = Sequence([1, 2, 3], 3)
    for i in range(rows):
        seq.apply_routed_expert_patch(i, np.full((1, 4, 2), i, dtype=np.int16))
    return seq


def test_finished_routes_are_read_without_copying():
    seq = _grown_seq(9)
    seq.finish_routed_experts(7)
    first, second = seq.routed_experts, seq.routed_experts
    assert np.shares_memory(first, second)
    assert first.shape == (7, 4, 2)
    assert not first.flags.writeable


def test_finished_routes_pickle_without_the_spare_capacity():
    import pickle

    seq = _grown_seq(9)
    assert seq._routed_expert_buf.shape[0] > 9
    live = seq.routed_experts.nbytes
    seq.finish_routed_experts(9)
    restored = pickle.loads(pickle.dumps(seq))
    assert restored._routed_expert_buf.nbytes == live
    np.testing.assert_array_equal(restored.routed_experts, seq.routed_experts)


def test_late_patch_after_finish_is_dropped():
    """A deferred step scheduled before the finish still reports its routes."""
    seq = _grown_seq(5)
    seq.finish_routed_experts(4)
    seq.status = SequenceStatus.FINISHED
    seq.apply_routed_expert_patch(4, np.full((2, 4, 2), 99, dtype=np.int16))
    assert seq.routed_expert_rows == 4
    assert (seq.routed_experts != 99).all()


def _host_patch(req_id: int, start: int, n: int, fill: int) -> RoutedExpertsPatch:
    rows = torch.full((n, 1, 2), fill, dtype=torch.int16)
    return RoutedExpertsPatch(
        np.array([req_id]), np.array([start]), np.array([0, n]), rows
    )


def test_merge_routed_patches_keeps_segment_order_and_offsets():
    a = _host_patch(1, 0, 2, 5)
    b = RoutedExpertsPatch(
        np.array([1, 2]),
        np.array([2, 0]),
        np.array([0, 3, 4]),
        np.arange(8, dtype=np.int16).reshape(4, 1, 2),
    )
    merged = merge_routed_patches([a, b])
    np.testing.assert_array_equal(merged.req_ids, [1, 1, 2])
    np.testing.assert_array_equal(merged.starts, [0, 2, 0])
    np.testing.assert_array_equal(merged.offsets, [0, 2, 5, 6])
    np.testing.assert_array_equal(merged.rows[2:], b.rows)
    assert isinstance(merged.rows, np.ndarray)
    alone = merge_routed_patches([a])
    assert isinstance(alone.rows, np.ndarray)
    np.testing.assert_array_equal(alone.rows, a.rows.numpy())


def test_incremental_decode_payload_smaller_than_full_history():
    import pickle

    layers, top_k, ctx, batch = 48, 8, 512, 8
    full = {i: np.zeros((ctx, layers, top_k), dtype=np.int16) for i in range(batch)}
    patch = RoutedExpertsPatch(
        np.arange(batch),
        np.full(batch, ctx - 1),
        np.arange(batch + 1),
        np.zeros((batch, layers, top_k), dtype=np.int16),
    )
    full_size = len(pickle.dumps(full))
    patch_size = len(pickle.dumps(patch))
    assert patch_size * 20 < full_size


@pytest.mark.parametrize("in_kernel_routing", [True, False])
@pytest.mark.parametrize("capturing", [True, False])
def test_qwen4_exp_single_token_moe_captures_its_routes(
    monkeypatch, capturing, in_kernel_routing
):
    """Both batch-1 MoE branches capture the ids their kernel ran: the
    in-kernel routing's own ids, or `select_experts`'."""
    q4 = pytest.importorskip(
        "atom.models.qwen4_exp",
        reason="qwen4_exp imports aiter at module load",
        exc_type=ImportError,
    )

    capturer = RoutedExpertsCapturer(num_slots=8, num_layers=2, top_k=2, device="cpu")
    kernel_ids = torch.tensor([[3, 1, 4]], dtype=torch.int32)
    selected_ids = torch.tensor([[2, 0, 4]], dtype=torch.int32)
    monkeypatch.setattr(
        q4.FusedMoE,
        "select_experts",
        staticmethod(lambda **kw: (torch.ones((1, 3)), selected_ids)),
    )

    def kernel(x, topk_ids, *a, return_topk_ids=False, **kw):
        return (x, kernel_ids) if return_topk_ids else x

    monkeypatch.setattr(q4, "moe_decode_single_token", kernel)
    n = 5
    experts = SimpleNamespace(
        w13_weight=torch.zeros((n, 2)),
        w13_weight_scale=torch.zeros((n, 2)),
        w2_weight=torch.zeros((n, 2)),
        w2_weight_scale=torch.zeros((n, 2)),
        renormalize=True,
        scoring_func="softmax",
        use_grouped_topk=False,
        custom_routing_function=None,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0 if in_kernel_routing else 2.0,
        top_k=2,
        topk_group=None,
        num_expert_group=None,
        shared_expert_scoring_func=None,
        global_num_experts=n,
        num_fused_shared_experts=1,
    )
    if capturing:
        capturer.bind(experts, 1)
    block = q4.Qwen4ExpSparseMoeBlock.__new__(q4.Qwen4ExpSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.experts = experts
    block.n_routed_experts = n - 1
    x = torch.zeros((1, 2))
    with _forward(torch.tensor([6])):
        assert block._single_token_experts(x, torch.zeros((1, n))) is x
    if capturing:
        ran = kernel_ids if in_kernel_routing else selected_ids
        assert capturer.buffer[6, 1].tolist() == ran[0, :2].tolist()
    else:
        assert (capturer.buffer == UNCAPTURED).all()
