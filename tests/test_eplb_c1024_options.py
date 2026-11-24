# SPDX-License-Identifier: MIT
# Tests for the gate-permute EPLB options of a high decode load (DEP8 c1024):
# the swap objective's tile term (ATOM_EPLB_TILE_ROWS), the cold-first slot
# order of the prefill layers (ATOM_EPLB_COLD_FIRST), plans too small to be
# worth a migration (ATOM_EPLB_MIN_SWAPS, ATOM_EPLB_MTP_MIN_SWAPS) and the
# uncounted start (ATOM_EPLB_SKIP_STEPS).

import logging
import math

import numpy as np
import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    import atom.config  # noqa: F401
    from atom.model_ops import eplb
    from atom.model_ops import eplb_gate_permute as gp
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")

from test_eplb_gate_permute import (  # noqa: E402
    _skewed,
    clean_switches,  # noqa: F401  (autouse: every switch unset)
)
from test_eplb_mtp import (  # noqa: E402
    KEY,
    _mtp_manager,
    _rebalance,
    _record,
    no_record_scope,  # noqa: F401  (autouse)
)
from test_eplb_rearm import _lines, _manager, _step  # noqa: E402

OPTIONS = (
    "ATOM_EPLB_TILE_ROWS",
    "ATOM_EPLB_COLD_FIRST",
    "ATOM_EPLB_COLD_FIRST_MIN_SWAPS",
    "ATOM_EPLB_MIN_SWAPS",
    "ATOM_EPLB_MTP_MIN_SWAPS",
    "ATOM_EPLB_SKIP_STEPS",
)


@pytest.fixture(autouse=True)
def options_unset(monkeypatch):
    for name in OPTIONS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- tile term
def _tiles_by_sum(lam: float, tile: int) -> float:
    """E[ceil(X / tile)] for X ~ Poisson(lam), term by term."""
    reach = int(lam + 12 * math.sqrt(lam + 1) + 2 * tile)
    total = cdf = 0.0
    for x in range(reach + 1):
        cdf += math.exp(x * math.log(lam) - lam - math.lgamma(x + 1.0))
        if x % tile == 0:
            total += min(max(1.0 - cdf, 0.0), 1.0)
    return total


def test_expected_tiles_is_the_poisson_tile_count():
    dec = np.array([0.0, 1.0, 40.0, 400.0, 4000.0, 15559.0])
    rows, top_k, tile = 500.0, 6, 64
    got = gp.expected_tiles(dec, rows, top_k, tile) / float(1 << 20)
    lam = dec / dec.sum() * rows * top_k
    assert got[0] == 0.0
    for value, mean in zip(got[1:], lam[1:]):
        assert value == pytest.approx(_tiles_by_sum(float(mean), tile), abs=2e-6)
    # integers in units of 2^-20: the same on every rank
    units = gp.expected_tiles(dec, rows, top_k, tile)
    assert np.array_equal(units, np.round(units))
    assert gp.expected_tiles(np.zeros(4), rows, top_k, tile).sum() == 0.0
    assert gp.expected_tiles(dec, 0.0, top_k, tile).sum() == 0.0


def test_below_a_tile_it_is_the_active_chance_and_above_it_keeps_counting():
    uniform = np.full(384, 100.0)
    # 8 rows a step: an expert sees a pick in few steps, and never a second tile
    low = gp.expected_tiles(uniform, 8.0, 6, 64) / float(1 << 20)
    assert low == pytest.approx(gp.active_proxy(uniform, 8.0, 6), rel=2e-3)
    # 3000 rows a step: every expert is active in every step, so the active
    # chance cannot tell a hot expert from an average one -- the tiles can
    dec = uniform.copy()
    dec[0] *= 20.0
    active = gp.active_proxy(dec, 3000.0, 6)
    tiles = gp.expected_tiles(dec, 3000.0, 6, 64)
    assert active[0] == pytest.approx(1.0, abs=1e-9)
    assert active[1] == pytest.approx(1.0, abs=1e-9)
    assert tiles[0] > 5 * tiles[1]


def test_swap_terms_takes_the_tile_term_only_when_asked():
    gen = np.random.default_rng(5)
    pf, dec = gen.integers(1, 1000, 64), gen.integers(1, 1000, 64)
    plain = gp.swap_terms(pf, dec, 3000.0, 6, 8)
    tiled = gp.swap_terms(pf, dec, 3000.0, 6, 8, tile_rows=64)
    assert np.array_equal(plain, gp.swap_terms(pf, dec, 3000.0, 6, 8, tile_rows=0))
    assert np.array_equal(plain[:, [0, 2]], tiled[:, [0, 2]])
    assert not np.allclose(plain[:, 1], tiled[:, 1])
    assert tiled[:, 1].sum() == pytest.approx(8.0)


def _loads(layers=3, experts=64, seed=2):
    gen = torch.Generator().manual_seed(seed)
    pf = (torch.rand(layers, experts, generator=gen) ** 4 * 1e5).long() + 1
    dec = (torch.rand(layers, experts, generator=gen) ** 4 * 1e5).long() + 1
    return pf, dec


def _busiest(load: torch.Tensor, p2l: torch.Tensor, ranks: int) -> float:
    per = p2l.numel() // ranks
    sums = load[p2l.long()].view(ranks, per).sum(dim=1).double()
    return float(sums.max() / sums.mean())


def test_tile_term_balances_the_tiles_of_a_high_decode_load():
    pf, dec = _loads()
    ident = torch.arange(64, dtype=torch.int32).repeat(3, 1)
    args = dict(num_gpus=8, top_k=6, budget=None, eps=0.0)
    plain = gp.swap_rebalance(pf, dec, 3000.0, ident, **args)[0]
    tiled = gp.swap_rebalance(pf, dec, 3000.0, ident, tile_rows=64, **args)[0]
    again = gp.swap_rebalance(pf, dec, 3000.0, ident, tile_rows=64, **args)[0]
    assert torch.equal(tiled, again)  # the same plan from the same loads
    with_tiles = without = 0.0
    for layer in range(3):
        assert sorted(tiled[layer].tolist()) == list(range(64))
        tiles = torch.from_numpy(gp.expected_tiles(dec[layer].numpy(), 3000.0, 6, 64))
        assert _busiest(tiles, tiled[layer], 8) < _busiest(tiles, ident[layer], 8)
        with_tiles += _busiest(tiles, tiled[layer], 8)
        without += _busiest(tiles, plain[layer], 8)
    # the busiest rank's tiles, over the layers: no worse than with a term
    # that is 1 for every expert at this load
    assert with_tiles <= without


# ---------------------------------------------------------------- cold-first
def test_cold_first_orders_the_prefill_layers_it_swapped_and_no_other():
    pf, dec = _loads(layers=4)
    pf[3] //= 1000  # a layer with a small share of the prefill picks
    pf[3] += 1
    ident = torch.arange(64, dtype=torch.int32).repeat(4, 1)
    args = dict(num_gpus=8, top_k=6, budget=16, eps=0.0, tile_rows=64)
    plain, _, _, swaps = gp.swap_rebalance(pf, dec, 3000.0, ident, **args)
    cold, l2p, _, same = gp.swap_rebalance(
        pf, dec, 3000.0, ident, cold_first=0.5, **args
    )
    assert swaps == same and all(n > 0 for n in swaps.values())
    for layer in range(4):
        # which rank holds an expert is the swaps' alone
        assert torch.equal(
            plain[layer].view(8, 8).sort(dim=1).values,
            cold[layer].view(8, 8).sort(dim=1).values,
        )
        assert torch.equal(l2p[layer, cold[layer].long(), 0].long(), torch.arange(64))
        by_slot = pf[layer][cold[layer].long()].view(8, 8)
        ascending = bool((by_slot[:, 1:] >= by_slot[:, :-1]).all())
        assert ascending == (layer < 3)
    assert torch.equal(cold[3], plain[3])
    # a layer the policy does not touch keeps its order
    kept = gp.swap_rebalance(pf, dec, 3000.0, cold, cold_first=0.5, **args)[0]
    idle = gp.swap_rebalance(
        pf, dec, 3000.0, ident, cold_first=0.5, **{**args, "budget": 0}
    )[0]
    assert torch.equal(idle, ident)
    for layer in range(3):
        by_slot = pf[layer][kept[layer].long()].view(8, 8)
        assert bool((by_slot[:, 1:] >= by_slot[:, :-1]).all())


def test_cold_first_min_swaps_leaves_a_layer_with_fewer_swaps_as_it_is():
    pf, dec = _loads(layers=2)
    ident = torch.arange(64, dtype=torch.int32).repeat(2, 1)
    args = dict(num_gpus=8, top_k=6, budget=2, eps=0.0)
    plain = gp.swap_rebalance(pf, dec, 3000.0, ident, **args)[0]
    high = gp.swap_rebalance(
        pf, dec, 3000.0, ident, cold_first=0.5, cold_first_min_swaps=3, **args
    )[0]
    low = gp.swap_rebalance(
        pf, dec, 3000.0, ident, cold_first=0.5, cold_first_min_swaps=2, **args
    )[0]
    assert torch.equal(high, plain) and not torch.equal(low, plain)


# ---------------------------------------------------------------- small plans
def _skew(manager):
    runtime = manager.gate_permute
    runtime.pf_total.copy_(_skewed(1))
    runtime.dec_load.copy_(_skewed(2))
    runtime.dec_rows, runtime.dec_steps = 480, 20
    runtime.window_rows, runtime.window_steps = 400, 4


def _gate_rows(manager, layer=0):
    return [t.clone() for t in gp.gate_row_tensors(manager.gate_permute.gates[layer])]


def test_without_min_swaps_the_plan_migrates(monkeypatch):
    manager = _manager(monkeypatch)
    _skew(manager)
    before = manager.live_metadata.physical_to_logical_map.clone()
    assert _rebalance(manager) >= 1
    assert manager.rebalance_count == 1 and manager._idle_evaluations == 0
    assert not torch.equal(manager.live_metadata.physical_to_logical_map, before)


def test_plan_below_min_swaps_is_dropped_and_comes_back(monkeypatch, caplog):
    manager = _manager(monkeypatch, ATOM_EPLB_MIN_SWAPS=100000)
    _skew(manager)
    before = manager.live_metadata.physical_to_logical_map.clone()
    gate = _gate_rows(manager)
    with caplog.at_level(logging.INFO, logger="atom"):
        assert _rebalance(manager) == 0  # no chunk step: nothing migrates
    assert torch.equal(manager.live_metadata.physical_to_logical_map, before)
    assert all(torch.equal(a, b) for a, b in zip(gate, _gate_rows(manager)))
    assert manager.rebalance_count == 0 and manager._experts_moved_total == 0
    assert manager._evaluations == 1 and manager._idle_evaluations == 1
    dropped = _lines(caplog, "ATOM_EPLB_MIN_SWAPS")
    assert len(dropped) == 1 and "is dropped" in dropped[0]
    assert dropped[0].startswith(gp.TAG)
    # the next evaluation plans afresh; with the bar under the plan it migrates
    monkeypatch.setenv("ATOM_EPLB_MIN_SWAPS", "1")
    manager.gate_permute.pf_total[0, 0] += 1
    assert _rebalance(manager) >= 1
    assert manager.rebalance_count == 1 and manager._idle_evaluations == 0
    assert not torch.equal(manager.live_metadata.physical_to_logical_map, before)
    assert not all(torch.equal(a, b) for a, b in zip(gate, _gate_rows(manager)))


def test_dropped_plans_in_a_row_freeze_the_placement_as_converged(monkeypatch):
    manager = _manager(
        monkeypatch,
        ATOM_EPLB_MIN_SWAPS=100000,
        ATOM_EPLB_MAX_REBALANCES=8,
        ATOM_EPLB_FREEZE_IDLE=2,
    )
    _skew(manager)
    for idle in (1, 2):
        manager.gate_permute.pf_total[0, 0] += 1
        assert _rebalance(manager) == 0
        assert manager._idle_evaluations == idle
        assert manager._freeze_due() == (idle == 2)


def _draft_load(manager, draft):
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    held = _record(mtp, draft)
    # every live row of layer 40 picks the experts of rank 0
    held[40][:8] = torch.tensor([0, 1, 0], dtype=torch.int32)
    held[41][:8] = torch.tensor([0, 2, 4], dtype=torch.int32)
    held[42][:8] = torch.tensor([1, 3, 5], dtype=torch.int32)
    for _ in range(5):
        runtime.after_decode_replay("target", 6)
        mtp.after_draft_replay(KEY, 8)


def test_draft_plan_alone_migrates_without_the_options(monkeypatch):
    manager, draft = _mtp_manager(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=1)
    monkeypatch.setattr(eplb, "_migrate_single_layer", lambda **kw: [])
    mtp = manager.gate_permute.mtp
    _draft_load(manager, draft)
    manager.gate_permute.pf_total.add_(1)  # even: the target layers stay
    before = mtp.p2l.clone()
    assert _rebalance(manager) == 1 and manager.rebalance_count == 1
    assert not torch.equal(mtp.p2l, before)


@pytest.mark.parametrize("target_plan", ["empty", "dropped"])
def test_small_draft_plan_is_dropped_with_the_target_layers_staying(
    monkeypatch, target_plan
):
    manager, draft = _mtp_manager(
        monkeypatch,
        ATOM_EPLB_DECODE_SAMPLE_EVERY=1,
        ATOM_EPLB_MIN_SWAPS=100000,
        ATOM_EPLB_MTP_MIN_SWAPS=100000,
    )
    monkeypatch.setattr(eplb, "_migrate_single_layer", lambda **kw: [])
    mtp = manager.gate_permute.mtp
    _draft_load(manager, draft)
    if target_plan == "empty":
        manager.gate_permute.pf_total.add_(1)
    else:
        manager.gate_permute.pf_total.copy_(_skewed(1))
    target_before = manager.live_metadata.physical_to_logical_map.clone()
    before = mtp.p2l.clone()
    assert _rebalance(manager) == 0
    assert torch.equal(mtp.p2l, before)
    assert torch.equal(manager.live_metadata.physical_to_logical_map, target_before)
    assert manager.rebalance_count == 0 and manager._idle_evaluations == 1


def test_draft_plan_of_enough_swaps_migrates_by_itself(monkeypatch):
    manager, draft = _mtp_manager(
        monkeypatch,
        ATOM_EPLB_DECODE_SAMPLE_EVERY=1,
        ATOM_EPLB_MIN_SWAPS=100000,
        ATOM_EPLB_MTP_MIN_SWAPS=1,
    )
    monkeypatch.setattr(eplb, "_migrate_single_layer", lambda **kw: [])
    mtp = manager.gate_permute.mtp
    _draft_load(manager, draft)
    manager.gate_permute.pf_total.copy_(_skewed(1))  # a target plan below the bar
    target_before = manager.live_metadata.physical_to_logical_map.clone()
    before = mtp.p2l.clone()
    assert _rebalance(manager) == 1  # the draft chunk alone
    assert not torch.equal(mtp.p2l, before)
    assert torch.equal(manager.live_metadata.physical_to_logical_map, target_before)
    assert manager.rebalance_count == 1 and manager._idle_evaluations == 0


# ---------------------------------------------------------------- skip steps
def test_skip_steps_drops_what_the_first_steps_counted(monkeypatch, caplog):
    manager, draft = _mtp_manager(
        monkeypatch, ATOM_EPLB_SKIP_STEPS=5, ATOM_EPLB_DECODE_SAMPLE_EVERY=1
    )
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    _draft_load(manager, draft)
    runtime.pf_total.copy_(_skewed(1))
    runtime.dec_load.copy_(_skewed(2))
    runtime.dec_rows, runtime.dec_steps = 480, 20
    runtime.window_rows, runtime.window_steps = 400, 4
    with caplog.at_level(logging.INFO, logger="atom"):
        _step(manager, 5)
        assert int(runtime.pf_total.sum()) > 0 and manager._evaluations == 0
        assert not _lines(caplog, "ATOM_EPLB_SKIP_STEPS")
        _step(manager)  # the step after the skipped ones
    assert len(_lines(caplog, "ATOM_EPLB_SKIP_STEPS")) == 1
    assert int(runtime.pf_total.sum()) == 0 and int(runtime.dec_load.sum()) == 0
    assert int(runtime.dec_raw.sum()) == 0 and not runtime._raw_dirty
    assert (runtime.dec_rows, runtime.dec_steps) == (0, 0)
    assert (runtime.window_rows, runtime.window_steps) == (0, 0)
    assert int(mtp.dec_load.sum()) == 0 and int(mtp.dec_raw.sum()) == 0
    assert (mtp.rows, mtp.steps, mtp.window_rows, mtp.window_steps) == (0, 0, 0, 0)
    assert manager._evaluations == 0 and manager.rebalance_count == 0


def test_skip_steps_off_by_default_keeps_the_schedule(monkeypatch):
    manager = _manager(monkeypatch)
    runtime = manager.gate_permute
    runtime.pf_total.copy_(_skewed(1))
    _step(manager, 2)  # the steps before the first rebalance
    assert int(runtime.pf_total.sum()) == int(_skewed(1).sum())
    assert manager._evaluations == 0


def test_option_defaults():
    from atom.utils import envs

    assert envs.ATOM_EPLB_TILE_ROWS == 0
    assert envs.ATOM_EPLB_COLD_FIRST == 0.0
    assert envs.ATOM_EPLB_COLD_FIRST_MIN_SWAPS == 1
    assert envs.ATOM_EPLB_MIN_SWAPS == 0
    assert envs.ATOM_EPLB_MTP_MIN_SWAPS == 0
    assert envs.ATOM_EPLB_SKIP_STEPS == 0
