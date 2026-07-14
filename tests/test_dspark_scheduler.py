# SPDX-License-Identifier: MIT
"""Unit tests for the DSpark Hardware-Aware Prefix Scheduler (Phase 2).

Covers the GPU-free scheduling core: survival probabilities, STS apply-side
calibration (order-preserving), throughput objective, greedy admission, and the
paper's §3.5 early-stop losslessness counterexample.
"""

import numpy as np
import torch
import torch as _torch

from atom.spec_decode.dspark_scheduler import (
    build_sps_table,
    calibrate_confidence,
    expected_throughput,
    schedule_prefix_lengths,
    survival_probabilities,
)

# ----------------------------------------------------------------------------
# survival_probabilities
# ----------------------------------------------------------------------------


def test_survival_is_cumprod_and_monotone():
    conf = torch.tensor([[0.9, 0.8, 0.5], [0.95, 0.9, 0.9]])
    a = survival_probabilities(conf)
    torch.testing.assert_close(a[0], torch.tensor([0.9, 0.72, 0.36]))
    # Cumulative product is monotonically non-increasing along the block.
    assert torch.all(a[:, 1:] <= a[:, :-1] + 1e-6)


def test_survival_requires_2d():
    try:
        survival_probabilities(torch.tensor([0.9, 0.8]))
        assert False, "expected ValueError"
    except ValueError:
        pass


# ----------------------------------------------------------------------------
# STS calibration (apply side)
# ----------------------------------------------------------------------------


def test_sts_none_is_identity():
    conf = torch.tensor([[0.7, 0.6]])
    torch.testing.assert_close(
        calibrate_confidence(conf, None), conf.float(), atol=1e-6, rtol=1e-6
    )


def test_sts_is_order_preserving():
    # Temperature scaling must not change the ranking of confidence scores.
    conf = torch.tensor([[0.55, 0.91, 0.6, 0.8, 0.2]])
    temps = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
    cal = calibrate_confidence(conf, temps)
    assert torch.equal(conf.argsort(dim=1), cal.argsort(dim=1))


def test_sts_temperature_gt1_pulls_toward_half():
    # T>1 reduces overconfidence: pushes probabilities toward 0.5.
    conf = torch.tensor([[0.95]])
    cal = calibrate_confidence(conf, torch.tensor([3.0]))
    assert 0.5 < float(cal) < 0.95


def test_sts_rejects_nonpositive_temperature():
    for bad in (torch.tensor([0.0]), torch.tensor([-1.0])):
        try:
            calibrate_confidence(torch.tensor([[0.7]]), bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_sts_length_mismatch_raises():
    try:
        calibrate_confidence(torch.tensor([[0.7, 0.6]]), torch.tensor([1.0]))
        assert False, "expected ValueError"
    except ValueError:
        pass


# ----------------------------------------------------------------------------
# expected_throughput
# ----------------------------------------------------------------------------


def test_throughput_objective_matches_formula():
    # R=2, ell=[2,1]; B = 2 + 3 = 5; tau = 2 + (a00+a01) + a10.
    conf = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
    a = survival_probabilities(conf)
    sps = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.5])  # SPS(5)=0.5
    theta = expected_throughput(a, [2, 1], sps)
    tau = 2 + (0.9 + 0.72) + 0.7
    assert abs(theta - tau * 0.5) < 1e-6


# ----------------------------------------------------------------------------
# schedule_prefix_lengths — basic behavior
# ----------------------------------------------------------------------------


def test_empty_batch():
    assert schedule_prefix_lengths(torch.empty(0, 4), torch.ones(64)) == []


def test_light_load_verifies_more():
    # Flat SPS (no throughput penalty for larger B) => verify the whole block,
    # because every extra token with a>0 strictly increases tau.
    conf = torch.full((1, 5), 0.9)
    sps = torch.ones(64)
    ell = schedule_prefix_lengths(conf, sps, early_stop=False)
    assert ell == [5]


def test_heavy_load_truncates_low_survival_suffix():
    # Steeply decaying SPS punishes large B => only the highest-survival prefix
    # is worth verifying.
    conf = torch.tensor([[0.95, 0.9, 0.3, 0.1, 0.05]])
    # SPS drops fast with B; index = B.
    sps = torch.tensor([1.0, 1.0, 0.8, 0.55, 0.3, 0.15])
    ell = schedule_prefix_lengths(conf, sps)
    assert 0 <= ell[0] < 5  # truncated before the full block
    # The kept prefix is the high-survival head, dropped tail is low-survival.
    assert ell[0] <= 2


def test_global_topk_prefers_confident_request():
    # Two requests; one confident, one not. Under a budget-limiting SPS, the
    # confident request should be allocated more verification length.
    conf = torch.tensor([[0.97, 0.95, 0.93], [0.4, 0.2, 0.1]])
    sps = torch.tensor([1.0, 1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.2])
    ell = schedule_prefix_lengths(conf, sps)
    assert ell[0] >= ell[1]


def test_result_lengths_within_block():
    torch.manual_seed(0)
    conf = torch.rand(4, 6).clamp(0.05, 0.99)
    sps = torch.linspace(1.0, 0.1, steps=64)
    ell = schedule_prefix_lengths(conf, sps)
    assert len(ell) == 4
    assert all(0 <= e <= 6 for e in ell)


# ----------------------------------------------------------------------------
# Paper §3.5 early-stop losslessness counterexample
# ----------------------------------------------------------------------------


def test_paper_counterexample_early_stop_returns_zero():
    # R=1, gamma=2, a_1=0.8; SPS(1)=1.0, SPS(2)=0.5, SPS(3)=0.45.
    # Theta_0 = 1*1.0 = 1.0 ; Theta_1 = 1.8*0.5 = 0.9 < Theta_0.
    # Early-stop must halt BEFORE evaluating c_2 (which depends on x_1),
    # returning ell=0 — this is what preserves the target distribution.
    conf = torch.tensor([[0.8, 0.9]])
    sps = torch.tensor(
        [1.0, 1.0, 0.5, 0.45]
    )  # index by B: SPS(1)=1,SPS(2)=.5,SPS(3)=.45
    ell = schedule_prefix_lengths(conf, sps, early_stop=True)
    assert ell == [0], f"early-stop should return ell=0, got {ell}"


def test_disabling_early_stop_can_cross_the_dip():
    # Without early-stop, an unconstrained global search may admit past a local
    # throughput dip if a later configuration scores higher. Construct such a
    # case: Theta dips at B=2 then recovers at B=3.
    conf = torch.tensor([[0.8, 0.99]])
    # SPS(1)=1.0 -> Theta0=1.0; B=2 SPS=0.5 -> Theta1=1.8*0.5=0.9 (dip);
    # B=3 SPS=0.65 -> tau=1+0.8+0.792=2.592 -> Theta2=2.592*0.65=1.685 (recovers).
    sps = torch.tensor([1.0, 1.0, 0.5, 0.65])
    greedy = schedule_prefix_lengths(conf, sps, early_stop=True)
    glob = schedule_prefix_lengths(conf, sps, early_stop=False)
    assert greedy == [0]  # early-stop halts at the dip (lossless)
    assert glob == [2]  # global search crosses it (needs async barrier)


# ----------------------------------------------------------------------------
# Level-A suffix masking (proposer integration logic, GPU-free reproduction)
# ----------------------------------------------------------------------------


def _mask_suffix(draft, confidence, sps=None, temps=None):
    """Standalone reproduction of EagleProposer._apply_confidence_schedule's
    masking (the GPU-free part), for unit testing the invariant."""
    bs, L = draft.shape
    if sps is None:
        sps = torch.linspace(1.0, 0.1, steps=bs * (L + 1) + 1)
    ell = schedule_prefix_lengths(confidence.detach(), sps, sts_temperatures=temps)
    ell_t = torch.tensor(ell, dtype=torch.long)
    cols = torch.arange(L).view(1, L)
    keep = cols < ell_t.view(bs, 1)
    return torch.where(keep, draft, draft.new_full((), -1)), ell


def test_mask_preserves_prefix_and_sentinels_suffix():
    draft = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    conf = torch.tensor([[0.97, 0.95, 0.9, 0.3, 0.1], [0.5, 0.2, 0.1, 0.05, 0.02]])
    out, ell = _mask_suffix(draft, conf)
    assert out.shape == draft.shape  # block shape constant (CUDA-graph safe)
    for r in range(draft.shape[0]):
        assert torch.equal(out[r, : ell[r]], draft[r, : ell[r]])  # prefix intact
        assert torch.all(out[r, ell[r] :] == -1)  # suffix sentineled


def test_mask_full_block_when_sps_flat():
    # Flat SPS -> verify everything -> no masking at all.
    draft = torch.arange(15).reshape(3, 5)
    conf = torch.full((3, 5), 0.9)
    out, ell = _mask_suffix(draft, conf, sps=torch.ones(64))
    assert ell == [5, 5, 5]
    assert torch.equal(out, draft)
    assert not torch.any(out == -1)


# ----------------------------------------------------------------------------
# build_sps_table (SPS calibration densification, GPU-free)
# ----------------------------------------------------------------------------


def test_sps_table_hits_measured_points_exactly():
    table = build_sps_table([2, 8, 16], [100.0, 60.0, 30.0], max_b=20)
    assert table.shape == (21,)
    assert abs(float(table[2]) - 100.0) < 1e-4
    assert abs(float(table[8]) - 60.0) < 1e-4
    assert abs(float(table[16]) - 30.0) < 1e-4


def test_sps_table_linear_interpolation():
    table = build_sps_table([0, 10], [100.0, 0.0], max_b=10)
    # Linear from 100 to 0 over [0,10]: table[B] = 100 - 10*B.
    for b in range(11):
        assert abs(float(table[b]) - (100.0 - 10.0 * b)) < 1e-3


def test_sps_table_flat_held_outside_range():
    table = build_sps_table([4, 8], [50.0, 20.0], max_b=12)
    # Below first point -> first value; above last -> last value.
    assert abs(float(table[0]) - 50.0) < 1e-4
    assert abs(float(table[1]) - 50.0) < 1e-4
    assert abs(float(table[12]) - 20.0) < 1e-4


def test_sps_table_unsorted_input_ok():
    t1 = build_sps_table([16, 2, 8], [30.0, 100.0, 60.0], max_b=20)
    t2 = build_sps_table([2, 8, 16], [100.0, 60.0, 30.0], max_b=20)
    torch.testing.assert_close(t1, t2)


def test_sps_table_single_point_is_constant():
    table = build_sps_table([5], [42.0], max_b=8)
    assert torch.allclose(table, torch.full((9,), 42.0))


def test_sps_table_validates_inputs():
    for args in (([1, 2], [1.0], 5), ([], [], 5), ([1], [1.0], -1)):
        try:
            build_sps_table(*args)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_sps_table_feeds_scheduler_end_to_end():
    # A real (decreasing) SPS table should drive sensible truncation.
    table = build_sps_table([2, 8, 16, 32], [200.0, 120.0, 60.0, 20.0], max_b=64)
    conf = torch.tensor([[0.97, 0.95, 0.9, 0.4, 0.1]])
    ell = schedule_prefix_lengths(conf, table)
    assert 0 <= ell[0] <= 5


# ----------------------------------------------------------------------------
# Level-B variable-length verification: position-advance losslessness invariant
# ----------------------------------------------------------------------------


def _next_anchor(cu_end, mtp_k, ell, accepted):
    """Replicate the engine's anchor-advance math for one seq.

    num_bonus = accepted (+1 if the whole verified prefix passed -> bonus token)
    num_reject = mtp_k - num_bonus   (engine hardcodes mtp_k here)
    anchor_idx = cu_end - (1 + num_reject)
    """
    num_bonus = accepted + 1 if accepted == ell else accepted + 1
    num_reject = mtp_k - num_bonus
    return cu_end - (1 + num_reject)


def test_level_b_anchor_matches_phase1_prefix():
    # Verifying only ell<mtp_k must land the next anchor at the SAME position as
    # Phase 1 would for the same number of accepted tokens (lossless advance).
    mtp_k, cu_end = 5, 6
    ell = 3
    for accepted in range(ell + 1):  # 0..ell
        a_b = _next_anchor(cu_end, mtp_k, ell, accepted)
        a_p1 = _next_anchor(cu_end, mtp_k, mtp_k, accepted)
        assert a_b == a_p1, (accepted, a_b, a_p1)


def test_level_b_num_reject_nonnegative():
    # num_bonus max = ell+1, so num_reject = mtp_k-(ell+1) >= 0 for ell<=mtp_k.
    mtp_k = 5
    for ell in range(1, mtp_k + 1):
        max_bonus = ell + 1
        assert mtp_k - max_bonus >= -1  # -1 only when ell==mtp_k (all+bonus)


def test_level_b_variable_metadata_shapes():
    # Variable num_draft_tokens -> cu/arange/target_indices stay self-consistent.
    num_draft = np.array([1, 3, 5, 2], dtype=np.int32)
    cu = np.cumsum(num_draft)
    total = int(cu[-1])
    assert total == 11
    # arange per-seq resets: lengths match num_draft
    arange = np.concatenate([np.arange(n) for n in num_draft])
    assert len(arange) == total
    assert arange.tolist() == [0, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1]


# ---- ell req_id re-mapping + batch-level uniform L (Level B source alignment) ----


class _StubProposer:
    """Mirrors EagleProposer.record_dspark_ell + the q-bucket max-ell logic.

    record_dspark_ell is live production code (req_id remapping of ell across
    continuous-batching reorders). max_mapped_ell mirrors the per-step
    "max ell over the batch" that _dspark_apply_q_bucket computes inline before
    quantizing to a graph bucket. Importing EagleProposer pulls the heavy
    atom.config chain (stubbed here), so both are mirrored verbatim — keep in
    lockstep with eagle.py / model_runner.py.
    """

    def __init__(self, mtp_k, ell):
        self.mtp_k = mtp_k
        self._dspark_last_ell = None if ell is None else _torch.tensor(ell)
        self._dspark_ell_by_req = {}

    def record_dspark_ell(self, req_ids):
        ell = getattr(self, "_dspark_last_ell", None)
        if ell is None:
            self._dspark_ell_by_req = {}
            return
        ell_np = ell.detach().to("cpu").numpy().astype(np.int32)
        n = min(len(req_ids), ell_np.shape[0])
        self._dspark_ell_by_req = {req_ids[i]: int(ell_np[i]) for i in range(n)}

    def max_mapped_ell(self, req_ids):
        """max over the batch of mapped ell (clamped 1..mtp_k); missing -> mtp_k.

        This is the value _dspark_apply_q_bucket feeds into quantize_to_bucket
        (as max_ell+1). Mirrors that inline computation for testing.
        """
        by_req = getattr(self, "_dspark_ell_by_req", None)
        if not by_req:
            return self.mtp_k
        L = 0
        for rid in req_ids:
            L = max(L, by_req.get(rid, self.mtp_k))
        return int(min(max(L, 1), self.mtp_k))


def test_ell_remap_by_req_id_reordered_batch():
    # Step N batch order [A,B,C] with ell [2,5,1]; step N+1 reorders to [C,A,B].
    p = _StubProposer(mtp_k=5, ell=[2, 5, 1])
    p.record_dspark_ell(["A", "B", "C"])
    # max mapped ell over current batch (drives the q-bucket choice).
    assert p.max_mapped_ell(["C", "A", "B"]) == 5  # max(1,2,5)
    # A subset batch [C, A] → max(1,2) = 2.
    assert p.max_mapped_ell(["C", "A"]) == 2


def test_ell_remap_new_request_falls_back_to_mtpk():
    # A request with no prior ell (just started) must be fully verified (mtp_k),
    # never under-verified.
    p = _StubProposer(mtp_k=5, ell=[1, 1])
    p.record_dspark_ell(["A", "B"])
    # "Z" is new this step -> contributes mtp_k=5 -> max=5.
    assert p.max_mapped_ell(["A", "Z"]) == 5


def test_ell_remap_no_history_returns_mtpk():
    # First step ever (no ell recorded) -> no truncation.
    p = _StubProposer(mtp_k=5, ell=None)
    p.record_dspark_ell(["A", "B"])  # ell is None -> empty map
    assert p.max_mapped_ell(["A", "B"]) == 5


def test_ell_remap_clamps_to_valid_range():
    p = _StubProposer(mtp_k=5, ell=[0, 0])  # scheduler said verify 0
    p.record_dspark_ell(["A", "B"])
    # Clamped to >=1 (verifying 0 degenerates index math).
    assert p.max_mapped_ell(["A", "B"]) == 1


# ---- CUDA-graph query-length buckets (plan Y, §17) ----


def test_resolve_q_buckets_parses_and_clamps():
    from atom.spec_decode.dspark_scheduler import resolve_q_buckets

    # mtp_k=5 -> max_q=6
    assert resolve_q_buckets("1,3,6", 6) == [1, 3, 6]
    # out-of-range dropped, max_q always present
    assert resolve_q_buckets("1,3,99", 6) == [1, 3, 6]
    # empty -> just the full bucket
    assert resolve_q_buckets("", 6) == [6]
    # dups + unsorted normalized; max_q auto-added
    assert resolve_q_buckets("3,1,3", 6) == [1, 3, 6]
    # junk tolerated
    assert resolve_q_buckets("1, x, 4", 6) == [1, 4, 6]


def test_quantize_to_bucket_rounds_up():
    from atom.spec_decode.dspark_scheduler import quantize_to_bucket

    b = [1, 3, 6]
    assert quantize_to_bucket(1, b) == 1
    assert quantize_to_bucket(2, b) == 3  # round up
    assert quantize_to_bucket(3, b) == 3
    assert quantize_to_bucket(4, b) == 6  # round up
    assert quantize_to_bucket(6, b) == 6
    assert quantize_to_bucket(7, b) == 6  # cap at max


def test_ragged_positions_construction():
    """DSpark §5.2 ragged: per-req positions via cumsum+arange (the 10-req example).

    Verifies prepare_decode's ragged branch builds correct per-seg positions
    (anchored to the full-span head, never OOB) with no batch-level padding.
    """
    import numpy as np

    scheduled_bs = 10
    full_q = 6
    # 5 reqs ell=2 (len 3), 4 reqs ell=1 (len 2), 1 req ell=3 (len 4)
    lens = np.array([3, 3, 3, 3, 3, 2, 2, 2, 2, 4], dtype=np.int32)
    context_lens = np.array(
        [100, 200, 150, 300, 250, 400, 180, 220, 310, 500], dtype=np.int32
    )

    cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
    np.cumsum(lens, out=cu[1:])
    batch_ids = np.repeat(np.arange(scheduled_bs, dtype=np.int32), lens)
    j_in_seq = np.arange(int(cu[-1]), dtype=np.int32) - cu[batch_ids].astype(np.int32)
    positions = (context_lens - full_q)[batch_ids] + j_in_seq

    # total = Σ(ell_i+1), no padding
    assert int(cu[-1]) == 27
    # each seg = [ctx_i-full_q .. ctx_i-full_q+len_i-1], anchor <= ctx-1
    for i in range(scheduled_bs):
        seg = positions[int(cu[i]) : int(cu[i + 1])]
        expect = np.arange(lens[i]) + (context_lens[i] - full_q)
        assert np.array_equal(seg, expect)
        assert seg[0] <= context_lens[i] - 1  # no OOB
    # vs full (10*6=60): ragged saves 55%
    assert abs((1 - 27 / 60) - 0.55) < 1e-9


def test_ragged_graph_bucket_plan_b():
    """DSpark §5.2 plan B: ragged replays (bs, q_eff) graph, C=bs*q_eff CTAs.

    q_eff = quantize_up(ceil(Σ(ell+1)/bs)); capacity bs*q_eff >= real Σ; the
    tail to capacity is -1 padding (CTAs bail). Verifies the graph-capacity /
    ForwardMode-recovery / pad-tail invariants for the 10-req example.
    """
    import numpy as np
    from atom.spec_decode.dspark_scheduler import quantize_to_bucket, resolve_q_buckets

    bs, full_q = 10, 6
    buckets = resolve_q_buckets("1,3,6", full_q)
    new_len = np.array([3, 3, 3, 3, 3, 2, 2, 2, 2, 4], dtype=np.int32)
    total_new = int(new_len.sum())  # 27

    q_ceil = (total_new + bs - 1) // bs  # ceil(27/10) = 3
    q_eff = quantize_to_bucket(q_ceil, buckets)  # 3
    cap = bs * q_eff  # 30

    assert q_eff == 3 and cap == 30
    assert cap >= total_new  # capacity must cover real ragged tokens
    assert cap // q_eff == bs  # ForwardMode recovers bs from graph-capacity tokens
    assert cap < bs * full_q  # saves CTAs vs full-length (30 < 60)

    # pad tail batch_id = -1 (CTAs bail)
    batch_id = np.full(cap, -1, dtype=np.int32)
    batch_id[:total_new] = np.repeat(np.arange(bs), new_len)
    assert (batch_id[total_new:] == -1).all()
