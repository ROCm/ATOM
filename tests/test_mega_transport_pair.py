# SPDX-License-Identifier: MIT
"""Prefill and decode must not share one MegaMoE recv layout.

Compaction is fixed when a transport is constructed and its row capacity comes
from that capacity rather than the step's tokens, so the two phases get two
transports: prefill keeps token-major rows, decode gets a small compact one.
These tests drive the real host selection and cache with stubs for the AITER
constructor, the cco communicator and the per-forward context; they do not
claim to validate collectives, arena bytes or graph replay.
"""

from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

try:
    from atom.model_ops.fused_moe import mori_v2_prepare_finalize as mv2
except ImportError as _e:  # aiter absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")

PREFILL_CAPACITY = 16384
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 48, 64, 128, 256]
DP_SIZE = 4

RECIPE = {
    "ep_rank": 0,
    "ep_size": DP_SIZE,
    "ep_src_global_rank": 0,
    "hidden_dim": 7168,
    "num_experts": 384,
    "num_experts_per_token": 6,
    "data_type_itemsize": 2,
    "inter_dim": 3072,
    "activation": 1,
    "gate_mode": 0,
    "quant_type": 2,
    "hidden_pad": 0,
    "intermediate_pad": 0,
    "swiglu_limit": 0.0,
}


def _context(tokens, **overrides):
    values = {
        "running_tokens": tokens,
        "running_tokens_are_unified": True,
        "running_tokens_across_dp": (tokens,) * DP_SIZE,
        "is_prefill": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        built=[],
        warmed=[],
        vmm=[],
        context=None,
        capture_sizes=list(CAPTURE_SIZES),
        speculative_config=None,
    )

    class FakeMega:
        def __init__(self, **kwargs):
            self.max_tokens_per_rank = kwargs["max_tokens_per_rank"]
            self._config = SimpleNamespace(
                dispatch_backend=kwargs.get("dispatch_backend", "flydsl"),
                dispatch_wire=kwargs["dispatch_wire"],
                compact_plan=kwargs.get("compact_plan", False),
            )
            state.built.append(
                (
                    self.max_tokens_per_rank,
                    self._config.dispatch_backend,
                    self._config.compact_plan,
                )
            )

        def warmup_compact_plan(self, recv_token_bound=None):
            state.warmed.append((self.max_tokens_per_rank, recv_token_bound))

    monkeypatch.setattr(mv2, "_import_mega", lambda: FakeMega)
    monkeypatch.setattr(mv2, "_MEGA_TRANSPORTS", {})
    monkeypatch.setattr(mv2, "_MEGA_DISPATCH_WIRE", "fp4")
    monkeypatch.setenv("ATOM_MEGA_DISPATCH_TDM", "0")
    monkeypatch.setenv("ATOM_MEGA_DECODE_COMPACT", "1")
    monkeypatch.setenv("ATOM_MEGA_DECODE_MTPR", "0")

    def fake_comm(ep_size, ep_rank, ep_src_global_rank, per_rank_vmm):
        state.vmm.append(per_rank_vmm)
        return SimpleNamespace(
            create_dev_comm=lambda: SimpleNamespace(per_rank_size=1 << 20),
            barrier=lambda: None,
        )

    monkeypatch.setattr(mv2, "_init_cco_comm", fake_comm)
    monkeypatch.setattr(
        mv2, "get_dp_group", lambda: SimpleNamespace(world_size=DP_SIZE)
    )
    monkeypatch.setattr(
        mv2,
        "get_forward_context",
        lambda: SimpleNamespace(context=state.context),
    )

    import atom.config as atom_config

    monkeypatch.setattr(
        atom_config,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            capture_sizes=list(state.capture_sizes),
            speculative_config=state.speculative_config,
        ),
    )

    def build(**overrides):
        kwargs = {
            "max_num_inp_token_per_rank": PREFILL_CAPACITY,
            **RECIPE,
            **overrides,
        }
        return mv2.build_mega_transports(**kwargs)

    state.build = build
    return state


def _pf(prefill, decode):
    """A prepare/finalize carrying an already-built transport pair."""
    obj = object.__new__(mv2.MoriV2PrepareAndFinalize)
    obj.mega = prefill
    obj.mega_decode = decode
    return obj


def test_pair_splits_capacity_layout_and_dispatch_kernel(runtime):
    prefill, decode = runtime.build()

    assert runtime.built == [
        (PREFILL_CAPACITY, "mori", False),
        (max(CAPTURE_SIZES), "tdm", True),
    ]
    assert prefill.max_tokens_per_rank == PREFILL_CAPACITY
    assert prefill._config.compact_plan is False
    # Sized from the capture ladder, so every captured decode step fits.
    assert decode.max_tokens_per_rank == max(CAPTURE_SIZES)
    assert decode._config.compact_plan is True


def test_prefill_dispatch_kernel_is_switchable_without_moving_decode(
    runtime, monkeypatch
):
    # Compaction only exists on TDM, so the decode half stays there whatever
    # prefill runs on.
    monkeypatch.setenv("ATOM_MEGA_DISPATCH_TDM", "1")

    prefill, decode = runtime.build()

    assert prefill._config.dispatch_backend == "tdm"
    assert prefill._config.compact_plan is False
    assert decode._config.dispatch_backend == "tdm"
    assert decode._config.compact_plan is True


def test_decode_transport_reuses_the_prefill_symmetric_window(runtime):
    runtime.build()

    # A smaller window would be a second collective allocation, not a smaller
    # slice of the existing one.
    assert len(runtime.vmm) == 2
    assert runtime.vmm[0] == runtime.vmm[1]


def test_speculative_q_len_widens_the_decode_capacity(runtime):
    runtime.speculative_config = SimpleNamespace(num_speculative_tokens=3)

    _, decode = runtime.build()

    assert decode.max_tokens_per_rank == max(CAPTURE_SIZES) * 4


def test_every_captured_bucket_is_warmed_before_capture(runtime):
    _, decode = runtime.build()

    bounds = {bound for capacity, bound in runtime.warmed}
    # Each rung as _recv_bound reports it, plus the unbounded case it returns
    # when the bound would not shrink the arena.
    assert bounds == {None} | {bs * DP_SIZE for bs in CAPTURE_SIZES}
    assert all(capacity == decode.max_tokens_per_rank for capacity, _ in runtime.warmed)


def test_warmup_runs_once_across_the_model_s_layers(runtime):
    runtime.build()
    warmed_after_first = list(runtime.warmed)
    runtime.build()

    # Later layers hit the transport cache; the warmup must not repeat per layer.
    assert runtime.warmed == warmed_after_first


def test_compaction_disabled_builds_only_prefill(runtime, monkeypatch):
    monkeypatch.setenv("ATOM_MEGA_DECODE_COMPACT", "0")

    prefill, decode = runtime.build()

    assert decode is None
    assert runtime.built == [(PREFILL_CAPACITY, "mori", False)]
    assert prefill._config.compact_plan is False


def test_a_bf16_wire_has_no_compact_decode_transport(runtime, monkeypatch):
    # TDM is only reachable on a wire that carries the e8m0 scale row, and
    # compaction is TDM-only.
    monkeypatch.setattr(mv2, "_MEGA_DISPATCH_WIRE", "bf16")

    prefill, decode = runtime.build()

    assert decode is None
    assert prefill._config.compact_plan is False


def test_triton_experts_veto_the_compact_transport(runtime):
    # triton_mega_moe reads the recv rows itself and knows nothing of the
    # compact layout, so it must never be handed one.
    prefill, decode = runtime.build(triton_experts=True)

    assert decode is None
    assert prefill._config.compact_plan is False


@pytest.mark.parametrize(
    "capture_sizes", [[], [PREFILL_CAPACITY], [PREFILL_CAPACITY * 2]]
)
def test_a_decode_capacity_that_does_not_shrink_is_not_built(runtime, capture_sizes):
    runtime.capture_sizes = capture_sizes

    _, decode = runtime.build()

    assert decode is None


def test_explicit_capacity_overrides_the_ladder_but_not_the_prefill_bound(
    runtime, monkeypatch
):
    monkeypatch.setenv("ATOM_MEGA_DECODE_MTPR", "1024")
    _, decode = runtime.build()
    assert decode.max_tokens_per_rank == 1024

    monkeypatch.setattr(mv2, "_MEGA_TRANSPORTS", {})
    monkeypatch.setenv("ATOM_MEGA_DECODE_MTPR", str(PREFILL_CAPACITY * 4))
    _, clamped = runtime.build()
    assert clamped is None


@pytest.mark.parametrize(
    ("tokens", "compact"),
    [(1, True), (256, True), (257, False), (PREFILL_CAPACITY, False)],
)
def test_selection_follows_the_decode_capacity(runtime, tokens, compact):
    prefill, decode = runtime.build()
    runtime.context = _context(tokens)

    chosen = _pf(prefill, decode).select_mega(tokens)

    assert chosen is (decode if compact else prefill)


def test_a_padded_local_tensor_cannot_switch_transport_behind_the_dp_group(runtime):
    # The compact dispatch and its plan barrier across ranks, so the choice may
    # only rest on DP-reduced terms. Were an oversized local tensor allowed to
    # fall back, this rank would run token-major dispatch while its peers sat in
    # the compact barrier waiting for it -- a hang, not a slowdown. Report the
    # broken premise instead.
    prefill, decode = runtime.build()
    runtime.context = _context(256)
    pf = _pf(prefill, decode)

    with pytest.raises(ValueError, match=r"capacity=256.*257 rows"):
        pf.select_mega(257)


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_prefill": True},
        {"running_tokens_are_unified": False},
        {"running_tokens_across_dp": None},
        # One peer above the capacity: dispatch still delivers its rows here.
        {"running_tokens_across_dp": (64, 64, 64, 4096)},
    ],
)
def test_steps_the_decode_capacity_cannot_bound_stay_on_prefill(runtime, overrides):
    prefill, decode = runtime.build()
    runtime.context = _context(64, **overrides)

    assert _pf(prefill, decode).select_mega(64) is prefill


def test_missing_context_stays_on_prefill(runtime):
    prefill, decode = runtime.build()
    runtime.context = None

    assert _pf(prefill, decode).select_mega(64) is prefill


def test_without_a_decode_transport_every_step_uses_prefill(runtime):
    prefill, _ = runtime.build(triton_experts=True)
    runtime.context = _context(64)

    assert _pf(prefill, None).select_mega(64) is prefill
