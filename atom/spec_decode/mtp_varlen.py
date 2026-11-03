"""Adaptive ragged verification, ported from DeepInfer (deepseek-ai/DeepInfer
@ 5537415).

A decode step verifies ``seqlens_q[i] = 1 + d_i`` tokens per request, where
``d_i`` is how many of its drafts it checks. The step's total ``total_q`` is
picked on the host from the draft confidence and a measured cost model, and
the step replays the CUDA graph captured for the smallest token count that
holds it.

Sources:
  * `MTPVarlen`, `ForwardRuntimeEstimator`, `SamplerRuntimeEstimator`:
    ``csrc/engine/mtp_varlen.hpp``.
  * `build_graph_batch_sizes`, `select_graph_size`:
    ``deepinfer/worker/cuda_graph_runner.py``.
  * `compute_seqlens_q_by_confidence_rates(_ref)`:
    ``deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py``.

This module imports nothing from the engine, so the planner runs in the
scheduler process and in CPU tests alike. torch/triton are imported lazily.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Graph grid
# ---------------------------------------------------------------------------


def build_graph_batch_sizes(max_size: int, interval: int) -> list[int]:
    """Power-of-two sizes below ``interval``, then multiples of it, then max."""
    interval = min(interval, max_size)
    sizes: list[int] = []
    size = 1
    while size < interval:
        sizes.append(size)
        size *= 2
    sizes.extend(range(interval, max_size, interval))
    sizes.append(max_size)
    return sizes


def select_graph_size(sizes: Sequence[int], total_q: int) -> int | None:
    """Smallest captured token count holding ``total_q``; None when none does."""
    idx = bisect.bisect_left(sizes, total_q)
    return int(sizes[idx]) if idx < len(sizes) else None


def graph_num_seqs(graph_tokens: int, max_num_seqs: int) -> int:
    """Per-sequence width of the graph captured at ``graph_tokens``.

    Every request forwards at least one token, so a step that fits in G tokens
    has at most G requests; the scheduler caps it at ``max_num_seqs``.
    """
    return min(int(graph_tokens), int(max_num_seqs))


def capture_seqlens_q(graph_tokens: int, num_seqs: int, full_q: int) -> np.ndarray:
    """A capture layout: ``graph_tokens`` spread over ``num_seqs`` requests.

    Each request gets between 1 and ``full_q`` tokens, so the capture runs the
    same kernels a real ragged step does.
    """
    assert 1 <= num_seqs <= graph_tokens <= num_seqs * full_q, (
        graph_tokens,
        num_seqs,
        full_q,
    )
    base, extra = divmod(graph_tokens, num_seqs)
    lens = np.full(num_seqs, base, dtype=np.int32)
    lens[:extra] += 1
    return lens


# ---------------------------------------------------------------------------
# Runtime estimators
# ---------------------------------------------------------------------------


class SamplerRuntimeEstimator:
    """Median over the last 9 samples, reported once 5 have arrived."""

    WINDOW = 9
    MIN_SAMPLES = 5

    def __init__(self) -> None:
        self._samples: deque[float] = deque(maxlen=self.WINDOW)
        self._median = 0.0

    def update(self, value_ms: float) -> None:
        self._samples.append(float(value_ms))
        s = sorted(self._samples)
        mid = len(s) // 2
        self._median = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])

    def get(self) -> float | None:
        return self._median if len(self._samples) >= self.MIN_SAMPLES else None


class _Median:
    """A sorted multiset with a median, for the two-level estimator."""

    __slots__ = ("_v",)

    def __init__(self) -> None:
        self._v: list[float] = []

    def insert(self, x: float) -> None:
        bisect.insort(self._v, x)

    def erase(self, x: float) -> None:
        i = bisect.bisect_left(self._v, x)
        assert i < len(self._v) and self._v[i] == x
        del self._v[i]

    def __len__(self) -> int:
        return len(self._v)

    def get(self) -> float:
        n = len(self._v)
        mid = n // 2
        # DeepInfer keeps the lower half one larger, so odd sizes take v[mid].
        return self._v[mid] if n % 2 else 0.5 * (self._v[mid - 1] + self._v[mid])


class ForwardRuntimeEstimator:
    """Slope of forward time over the startup curve, per graph token.

    The startup curve is measured at a near-empty context. Each sample is
    ``(forward_ms - baseline_ms(graph)) / graph_size``; per graph size the
    median of its samples counts once it has 10, and the estimate is the median
    of those medians over a 5000-step window, graphs weighted equally.
    """

    WINDOW = 5000
    MIN_SAMPLES = 10

    def __init__(self) -> None:
        self._window: deque[tuple[int, float]] = deque()
        self._by_graph: dict[int, _Median] = {}
        self._medians = _Median()

    def update(self, graph_size: int, normalized_residual: float) -> None:
        if len(self._window) == self.WINDOW:
            old_graph, old_value = self._window.popleft()
            self._update_graph(old_graph, old_value, insert=False)
        value = float(normalized_residual)
        self._window.append((int(graph_size), value))
        self._update_graph(int(graph_size), value, insert=True)

    def get(self) -> float:
        return self._medians.get() if len(self._medians) else 0.0

    def _update_graph(self, graph_size: int, value: float, insert: bool) -> None:
        samples = self._by_graph.setdefault(graph_size, _Median())
        if len(samples) >= self.MIN_SAMPLES:
            self._medians.erase(samples.get())
        if insert:
            samples.insert(value)
        else:
            samples.erase(value)
        if len(samples) >= self.MIN_SAMPLES:
            self._medians.insert(samples.get())
        elif len(samples) == 0:
            del self._by_graph[graph_size]


class ForwardResidualModel:
    """``forward_ms - curve_ms(G) ~= offset + slope * G`` over recent steps.

    The startup curve is measured at a near-empty context, so a live step at
    long context runs far above it. Most of that gap is attention over the KV,
    paid once per step whatever the verify width; DeepInfer's estimator divides
    the whole gap by ``G`` and so prices it per token, which at long context
    makes every draft look too expensive to verify.

    The slope comes from back-to-back steps whose ``G`` differ: the pair shares
    the context length and whatever else the GPU is doing at the time, so the
    difference isolates the per-token cost. A regression over the whole window
    would not -- context grows and prefill bursts come and go while ``G``
    drifts, and that correlation reads as a per-token cost several times the
    real one. The offset then tracks the recent residual at that slope.
    """

    PAIRS = 64
    MIN_PAIRS = 8
    #: Pairs closer than this in G mostly measure noise.
    MIN_PAIR_SPREAD = 16
    MAX_SLOPE_MS = 1.0
    OFFSET_WINDOW = 64
    MIN_SAMPLES = 8

    def __init__(self) -> None:
        self._pair_slopes: deque[float] = deque(maxlen=self.PAIRS)
        self._g: deque[float] = deque(maxlen=self.OFFSET_WINDOW)
        self._r: deque[float] = deque(maxlen=self.OFFSET_WINDOW)
        self._offset = 0.0
        self._slope = 0.0

    def update(self, graph_size: int, residual_ms: float) -> None:
        g, r = float(graph_size), float(residual_ms)
        if self._g and abs(g - self._g[-1]) >= self.MIN_PAIR_SPREAD:
            self._pair_slopes.append((r - self._r[-1]) / (g - self._g[-1]))
        self._g.append(g)
        self._r.append(r)
        self._fit()

    def _fit(self) -> None:
        if len(self._pair_slopes) >= self.MIN_PAIRS:
            self._slope = float(
                np.clip(np.median(self._pair_slopes), 0.0, self.MAX_SLOPE_MS)
            )
        if len(self._g) >= self.MIN_SAMPLES:
            g = np.asarray(self._g)
            r = np.asarray(self._r)
            self._offset = float(np.median(r - self._slope * g))

    @property
    def offset_ms(self) -> float:
        return self._offset

    @property
    def slope_ms(self) -> float:
        return self._slope


class ConfidenceCalibrator:
    """Per-position map from the confidence head's output to observed acceptance.

    Each verified draft position ``k`` whose earlier drafts were all accepted is
    one Bernoulli trial of ``P(accept k | accepted < k)``, which is what the head
    claims to predict. Trials are binned by the head's value; a bin reports
    ``(accepted + prior * c) / (trials + prior)``, so an empty bin returns the
    head unchanged. Counts decay so the map follows drift in the workload.
    """

    def __init__(self, num_positions: int, num_bins: int = 10, prior: float = 8.0):
        self.num_positions = int(num_positions)
        self.num_bins = int(num_bins)
        self.prior = float(prior)
        self._trials = np.zeros((self.num_positions, self.num_bins), np.float64)
        self._accepts = np.zeros((self.num_positions, self.num_bins), np.float64)
        #: Per position, the trial count at which both counts are halved.
        self.decay_at = 20000.0

    def bins_of(self, conf: np.ndarray) -> np.ndarray:
        idx = np.floor(np.asarray(conf, np.float64) * self.num_bins).astype(np.int64)
        return np.clip(idx, 0, self.num_bins - 1)

    def update(self, trials: np.ndarray, accepts: np.ndarray) -> None:
        k = min(self.num_positions, trials.shape[0])
        self._trials[:k] += trials[:k]
        self._accepts[:k] += accepts[:k]
        heavy = self._trials.sum(axis=1) > self.decay_at
        self._trials[heavy] *= 0.5
        self._accepts[heavy] *= 0.5

    def calibrate(self, conf: np.ndarray) -> np.ndarray:
        """``conf`` is ``[n, k]`` per-position head output; returns the same shape."""
        conf = np.asarray(conf, np.float32)
        k = min(conf.shape[-1], self.num_positions)
        out = conf.copy()
        if k == 0:
            return out
        b = self.bins_of(conf[..., :k])
        pos = np.arange(k)
        t = self._trials[pos, b]
        a = self._accepts[pos, b]
        out[..., :k] = (a + self.prior * conf[..., :k]) / (t + self.prior)
        return np.clip(out, 0.0, 1.0)

    def observed(self) -> np.ndarray:
        """Observed per-position acceptance over all bins (nan where untried)."""
        t = self._trials.sum(axis=1)
        a = self._accepts.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(t > 0, a / t, np.nan)


def acceptance_trials(
    conf: np.ndarray,
    known: np.ndarray,
    lens: np.ndarray,
    accepted: np.ndarray,
    num_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin one verify step's outcome against the confidence it was planned from.

    Row ``i`` verified ``lens[i] - 1`` drafts and accepted the first
    ``accepted[i]``. Position ``k`` is a trial when it was verified and every
    draft before it was accepted; it succeeded when ``k < accepted[i]``. Rows
    with ``known[i]`` False were planned without confidence and are skipped.
    Returns ``(trials, accepts)``, each ``[num_positions, num_bins]``.
    """
    conf = np.asarray(conf, np.float32)
    n, k = conf.shape
    trials = np.zeros((k, num_bins), np.float64)
    accepts = np.zeros((k, num_bins), np.float64)
    if n == 0 or k == 0:
        return trials, accepts
    drafts = np.clip(np.asarray(lens, np.int64)[:n] - 1, 0, k)
    acc = np.clip(np.asarray(accepted, np.int64)[:n], 0, None)
    acc = np.minimum(acc, drafts)
    pos = np.arange(k)[None, :]
    tried = (pos < drafts[:, None]) & (pos <= acc[:, None])
    tried &= np.asarray(known, bool)[:n, None]
    won = tried & (pos < acc[:, None])
    bins = np.clip(np.floor(conf * num_bins).astype(np.int64), 0, num_bins - 1)
    cols = np.broadcast_to(pos, (n, k))
    np.add.at(trials, (cols[tried], bins[tried]), 1.0)
    np.add.at(accepts, (cols[won], bins[won]), 1.0)
    return trials, accepts


# ---------------------------------------------------------------------------
# Opus MLA row chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkCost:
    """What the opus MLA decode's work plan makes a verify width cost.

    The plan (``opus_build_plan`` in atom_opus_mla) splits a request's
    ``q * heads`` query rows into ``slot``-row chunks and lays each XCD out in
    columns of ``chunks`` workers that walk the same KV tiles, one chunk each.
    The chunk count is the batch's DEEPEST request's, and a shallower request
    leaves the extra workers of its column idle -- so an XCD's ``columns``
    workers make ``columns // chunks`` columns, and the step's attention time
    goes as ``context / (columns // chunks(max q))``. One q=6 request costs
    the whole batch what every request at q=6 would: 1.6-2x the k=3 (q=4)
    attention at C128, where that attention is ~4 ms of a ~75 ms forward.

    The startup curve is profiled at an empty context and the residual model
    reads this as noise, so the planner prices it here: per step,
    ``ms_per_token * context_tokens * excess(max q)``, ``excess`` being how far
    ``1 / columns`` sits above its value at ``free_q``. Widths up to
    ``free_q`` cost what they cost before. ``heads``/``slot``/``columns``
    mirror the kernel build (H_KERNEL 96 = Kimi-K3 TP8 x DCP8 gathered heads,
    SLOT 128, NWG 256 / 8 XCDs).
    """

    heads: int = 96
    slot: int = 128
    columns: int = 32
    free_q: int = 4

    def chunks(self, q: int) -> int:
        return -(-max(int(q), 1) * self.heads // self.slot)

    def inv_columns(self, q: int) -> float:
        return 1.0 / max(1, self.columns // self.chunks(q))

    def excess(self, q) -> np.ndarray:
        """Per-step price units of a widest request ``q`` (array or int)."""
        q = np.maximum(np.asarray(q, np.int64), 1)
        chunks = -(-q * self.heads // self.slot)
        inv = 1.0 / np.maximum(1, self.columns // chunks)
        return np.maximum(inv - self.inv_columns(self.free_q), 0.0)

    def caps(self, full_q: int) -> list[int]:
        """Widest-request caps worth comparing: the widest q of every chunk
        count above ``free_q``'s, up to ``full_q``; the uncapped plan first."""
        out = [int(full_q)]
        for level in range(self.chunks(self.free_q), self.chunks(full_q)):
            q = level * self.slot // self.heads
            if self.free_q <= q < full_q and q not in out:
                out.append(q)
        return out


class ChunkCostModel:
    """``ms_per_token`` of `ChunkCost`, seeded and refined from the steps.

    Back-to-back timed steps whose price units differ (an explore step beside
    a capped one) share the context and whatever else runs at the time, so
    their residual difference, less the per-token slope, over their price
    difference is one sample of the price. The median of recent samples
    replaces the seed once there are enough, within a factor of the seed.
    """

    PAIRS = 32
    MIN_PAIRS = 8
    MAX_FACTOR = 4.0

    def __init__(self, cost: ChunkCost, seed_ms_per_mtok: float) -> None:
        self.cost = cost
        self.seed = float(seed_ms_per_mtok) * 1e-6
        self._pairs: deque[float] = deque(maxlen=self.PAIRS)
        self._last: tuple[float, float, float] | None = None

    @property
    def ms_per_token(self) -> float:
        if len(self._pairs) < self.MIN_PAIRS:
            return self.seed
        return float(np.clip(np.median(self._pairs), 0.0, self.MAX_FACTOR * self.seed))

    def units(self, max_q: int | None, ctx_tokens: int | None) -> float:
        if not max_q or not ctx_tokens:
            return 0.0
        return float(ctx_tokens) * float(self.cost.excess(int(max_q)))

    def update(
        self, graph_size: int, residual_ms: float, units: float, slope_ms: float
    ):
        g, r, x = float(graph_size), float(residual_ms), float(units)
        last = self._last
        self._last = (g, r, x)
        if last is None:
            return
        dx = x - last[2]
        # A pair of equal (usually zero) prices says nothing about the price.
        if abs(dx) < 0.5 * max(x, last[2]) or abs(dx) <= 0.0:
            return
        self._pairs.append(((r - last[1]) - slope_ms * (g - last[0])) / dx)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass
class SeqDraftRates:
    """One request's view for the planner.

    ``rates`` are the cumulative acceptance estimates of its drafts (the
    cumprod of the confidence head's per-position output), or None when the
    request has none yet.
    """

    num_drafts: int
    rates: np.ndarray | None = None
    has_draft: bool = True


class MTPVarlen:
    """Per-step total-token budget and its split (DeepInfer `MTPVarlen`)."""

    def __init__(
        self,
        explore_interval: int = 0,
        chunk: ChunkCostModel | None = None,
        max_q: int = 0,
    ) -> None:
        self._curve: list[tuple[int, float]] = []
        #: Prices the widest request's opus row chunks (None: not priced).
        self.chunk = chunk
        #: Hard cap on any request's ``seqlens_q`` (0: none).
        self.max_q = int(max_q)
        #: The widest-request cap the last plan took (None: uncapped).
        self.last_cap: int | None = None
        self._curve_sizes: list[int] = []
        self._forward = ForwardResidualModel()
        self._sampler_by_num_seqs: dict[int, SamplerRuntimeEstimator] = {}
        self._sampler_fallback = SamplerRuntimeEstimator()
        #: Every this many plans verify every draft (0: never). Those steps are
        #: the only samples of large graphs and of low-confidence positions
        #: once the plan has settled, so both the forward model and the
        #: calibrator need them.
        self.explore_interval = int(explore_interval)
        self._num_plans = 0
        self.last_explore = False
        self.calibrator: ConfidenceCalibrator | None = None
        #: Wall time between back-to-back decode schedules, and the GPU time
        #: (forward + sampler) of recent steps. Their gap is what a step costs
        #: beyond the timed GPU work -- the draft's host side, metadata,
        #: launch -- and it is paid per step, so leaving it out makes short
        #: verifies look cheaper than they are.
        self._step_wall: deque[float] = deque(maxlen=self.OVERHEAD_WINDOW)
        self._step_gpu: deque[float] = deque(maxlen=self.OVERHEAD_WINDOW)

    OVERHEAD_WINDOW = 128
    MAX_OVERHEAD_MS = 200.0

    def enable_calibration(self, num_positions: int) -> None:
        self.calibrator = ConfidenceCalibrator(num_positions)

    @property
    def decode_perf_curve(self) -> list[tuple[int, float]]:
        return self._curve

    def set_decode_perf_curve(self, curve: Sequence[tuple[int, float]]) -> None:
        points = [(int(g), float(ms)) for g, ms in curve]
        sizes = [g for g, _ in points]
        assert sizes == sorted(set(sizes)), "graph sizes must strictly increase"
        # A bigger graph is never cheaper; a dip in the startup profile is
        # noise the planner would otherwise lock onto.
        ms = np.maximum.accumulate(np.asarray([m for _, m in points], np.float64))
        self._curve = [(g, float(m)) for g, m in zip(sizes, ms)]
        self._curve_sizes = sizes

    def update_runtime(
        self,
        num_seqs: int,
        graph_size: int,
        forward_ms: float,
        sampler_ms: float,
        max_q: int | None = None,
        ctx_tokens: int | None = None,
    ) -> None:
        """One timed step. ``max_q``/``ctx_tokens`` (its widest request and
        the batch's context) let the chunk price come off the residual before
        the forward model sees it, and refine that price."""
        idx = bisect.bisect_left(self._curve_sizes, graph_size)
        if idx == len(self._curve):
            return
        baseline_ms = self._curve[idx][1]
        residual_ms = forward_ms - baseline_ms
        if self.chunk is not None:
            units = self.chunk.units(max_q, ctx_tokens)
            self.chunk.update(graph_size, residual_ms, units, self.slope_ms())
            residual_ms -= self.chunk.ms_per_token * units
        self._forward.update(graph_size, residual_ms)
        self._step_gpu.append(float(forward_ms) + float(sampler_ms))
        self._sampler_by_num_seqs.setdefault(
            int(num_seqs), SamplerRuntimeEstimator()
        ).update(sampler_ms)
        self._sampler_fallback.update(sampler_ms)

    def sampler_ms(self, num_seqs: int) -> float:
        est = self._sampler_by_num_seqs.get(int(num_seqs))
        value = est.get() if est is not None else None
        if value is None:
            value = self._sampler_fallback.get()
        return 0.0 if value is None else value

    def update_step_wall(self, wall_ms: float) -> None:
        """Wall time from one decode schedule to the next, with nothing between."""
        self._step_wall.append(float(wall_ms))

    def overhead_ms(self) -> float:
        if len(self._step_wall) < 8 or len(self._step_gpu) < 8:
            return 0.0
        gap = float(np.median(self._step_wall)) - float(np.median(self._step_gpu))
        return float(np.clip(gap, 0.0, self.MAX_OVERHEAD_MS))

    def slope_ms(self) -> float:
        return self._forward.slope_ms

    def offset_ms(self) -> float:
        return self._forward.offset_ms

    def calibrate(self, conf: np.ndarray) -> np.ndarray:
        return conf if self.calibrator is None else self.calibrator.calibrate(conf)

    @staticmethod
    def _queue(seqs: Sequence[SeqDraftRates]):
        q: list[tuple[float, int]] = []
        for i, s in enumerate(seqs):
            nd = int(s.num_drafts)
            if s.rates is None or len(s.rates) == 0:
                fill = 1.0 if s.has_draft else 0.0
                q.extend((fill, i) for _ in range(nd))
                continue
            assert len(s.rates) == nd, (len(s.rates), nd)
            q.extend((float(r), i) for r in s.rates)
        # Higher rate first; a tie goes to the lower request index.
        q.sort(key=lambda e: (-e[0], e[1]))
        return q

    @staticmethod
    def _queue_arrays(seqs: Sequence[SeqDraftRates]):
        """`_queue` as arrays, plus each entry's draft position."""
        rates, reqs, pos = [], [], []
        for i, s in enumerate(seqs):
            nd = int(s.num_drafts)
            if nd <= 0:
                continue
            if s.rates is None or len(s.rates) == 0:
                r = np.full(nd, 1.0 if s.has_draft else 0.0, np.float64)
            else:
                assert len(s.rates) == nd, (len(s.rates), nd)
                r = np.asarray(s.rates, dtype=np.float64)
            rates.append(r)
            reqs.append(np.full(nd, i, np.int64))
            pos.append(np.arange(nd, dtype=np.int64))
        if not rates:
            z = np.zeros(0, np.int64)
            return np.zeros(0, np.float64), z, z
        rates, reqs, pos = (np.concatenate(a) for a in (rates, reqs, pos))
        # Higher rate first; a tie goes to the lower request index, then the
        # earlier position (the stable order `_queue` keeps).
        order = np.lexsort((pos, reqs, -rates))
        return rates[order], reqs[order], pos[order]

    def schedule(
        self,
        seqs_for_bsz: Sequence[SeqDraftRates],
        seqs_for_q: Sequence[SeqDraftRates] | None = None,
        ctx_tokens: int = 0,
    ) -> np.ndarray:
        """Per-request ``seqlens_q``; their sum is the step's ``total_q``.

        ``seqs_for_bsz`` sizes the budget and ``seqs_for_q`` splits it. DeepInfer
        passes rates one round apart; when the host holds only one round, pass
        the same list (or omit the second). ``ctx_tokens`` is the batch's
        context, which the chunk price scales with.
        """
        if seqs_for_q is None:
            seqs_for_q = seqs_for_bsz
        n = len(seqs_for_bsz)
        assert len(seqs_for_q) == n
        full = np.asarray([1 + int(s.num_drafts) for s in seqs_for_bsz], np.int32)
        self.last_explore = False
        self.last_cap = None
        if not self._curve:
            return full
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        self._num_plans += 1
        if self.explore_interval > 0 and self._num_plans % self.explore_interval == 0:
            self.last_explore = True
            return full

        rate_b, _, pos_b = self._queue_arrays(seqs_for_bsz)
        _, req_q, pos_q = self._queue_arrays(seqs_for_q)

        sampler_ms = self.sampler_ms(n) + self.overhead_ms()
        slope_ms = self.slope_ms()
        offset_ms = self.offset_ms()
        sizes = np.asarray(self._curve_sizes, np.int64)
        base = np.asarray([ms for _, ms in self._curve], np.float64)

        full_q = int(full.max())
        caps = [full_q]
        if self.chunk is not None:
            caps = self.chunk.cost.caps(full_q)
        if self.max_q > 0:
            caps = sorted({min(c, self.max_q) for c in caps}, reverse=True)
        price = 0.0
        if self.chunk is not None and ctx_tokens > 0:
            price = self.chunk.ms_per_token * float(ctx_tokens)

        best = (0.0, 0, full_q)  # (throughput, bsz, cap)
        for cap in caps:
            keep = pos_b + 2 <= cap
            rates = rate_b[keep]
            # bsz n + j holds the first j entries: every request's anchor, then
            # the j most likely drafts. DeepInfer's running sum, from n.
            expected = np.cumsum(np.concatenate(([float(n)], rates)))
            bsz = n + np.arange(rates.shape[0] + 1, dtype=np.int64)
            idx = np.searchsorted(sizes, bsz, side="left")
            ok = idx < sizes.shape[0]
            if not ok.any():
                continue
            idx = np.minimum(idx, sizes.shape[0] - 1)
            time_ms = base[idx] + offset_ms + slope_ms * sizes[idx] + sampler_ms
            if price > 0.0:
                widest = np.concatenate(([1], np.maximum.accumulate(pos_b[keep] + 2)))
                time_ms = time_ms + price * self.chunk.cost.excess(widest)
            throughput = np.where(ok, expected / np.maximum(time_ms, 1e-3), -np.inf)
            j = int(np.argmax(throughput))
            if throughput[j] > best[0]:
                best = (float(throughput[j]), int(bsz[j]), cap)

        _, best_bsz, cap = best
        if best_bsz < n:
            # The curve cannot hold even one token per request: verify every
            # draft and let the runner fall back to an eager forward.
            return full

        if cap < full_q:
            self.last_cap = cap
        take = np.flatnonzero(pos_q + 2 <= cap)[: best_bsz - n]
        lens = np.ones(n, dtype=np.int32)
        np.add.at(lens, req_q[take], 1)
        return lens


def cumulative_rates(confidence: np.ndarray) -> np.ndarray:
    """Per-position confidence -> cumulative acceptance estimate."""
    return np.cumprod(np.asarray(confidence, dtype=np.float32), axis=-1)


# ---------------------------------------------------------------------------
# Batch repack (host)
# ---------------------------------------------------------------------------


def repack_decode_batch(batch, lens: np.ndarray) -> None:
    """Rewrite a pure-decode batch to forward ``lens[i]`` tokens per request.

    Each request keeps the head of its scheduled span (anchor, then drafts),
    so a shorter span only drops trailing drafts. KV stays reserved at the
    scheduler's ``mtp_k + 1``: the rejection count is reported against that
    reservation, so the scheduler's accounting does not change.
    """
    bs = int(batch.total_seqs_num_decode)
    assert int(batch.total_tokens_num_prefill) == 0 and bs == batch.total_seqs_num
    lens = np.asarray(lens, dtype=np.int32)
    assert lens.shape == (bs,), (lens.shape, bs)
    old = np.asarray(batch.num_scheduled_tokens, dtype=np.int32)
    assert np.all(lens >= 1) and np.all(lens <= old[:bs]), (lens, old)
    if np.array_equal(lens, old[:bs]):
        return
    sched = np.asarray(batch.scheduled_tokens)
    old_cu = np.zeros(bs + 1, dtype=np.int64)
    np.cumsum(old[:bs], out=old_cu[1:])
    keep = np.repeat(old_cu[:-1], lens) + (
        np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
    )
    batch.scheduled_tokens = sched[keep]
    nst = old.copy()
    nst[:bs] = lens
    batch.num_scheduled_tokens = nst
    total = int(lens.sum())
    batch.total_tokens_num_decode = total
    batch.total_tokens_num = total


# ---------------------------------------------------------------------------
# GPU allocation (DeepInfer `compute_seqlens_q_by_confidence_rates`)
# ---------------------------------------------------------------------------


def compute_seqlens_q_by_confidence_rates_ref(
    total_q: int,
    remap,
    prev_confidence_rates,
    seqlens_q,
    fallback_rates=None,
) -> None:
    """Split ``total_q`` by the previous step's confidence.

    ``remap[i]`` is request i's row in ``prev_confidence_rates`` or -1. DeepInfer
    gives a -1 row rate 0; ``fallback_rates`` (``[num_seqs, nextn]``, per
    position) replaces that 0 when the host has rates of its own for the row.
    The flattened cumulative rates are ranked (higher wins, ties to the lower
    flat index) and each request gets ``1 + count(rank < total_q - num_seqs)``.
    """
    import torch

    num_seqs = remap.shape[0]
    mask = remap != -1
    gathered = prev_confidence_rates[remap.clamp(min=0).long()].float()
    other = (
        torch.zeros_like(gathered)
        if fallback_rates is None
        else fallback_rates.to(gathered.dtype)
    )
    gathered = torch.where(mask.unsqueeze(1), gathered, other)
    cum = torch.cumprod(gathered, dim=1)
    flat = cum.reshape(-1)
    order = torch.argsort(flat, descending=True, stable=True)
    ranks = torch.empty(flat.numel(), device=flat.device, dtype=torch.int32)
    ranks[order] = torch.arange(flat.numel(), device=flat.device, dtype=torch.int32)
    wins = (ranks.reshape(num_seqs, -1) < (total_q - num_seqs)).to(torch.int32)
    seqlens_q.copy_(1 + wins.sum(dim=1))


_TRITON_KERNEL = None


def _get_triton_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _seqlens_q_by_confidence_kernel(
        remap_ptr,
        prev_ptr,
        prev_stride,
        fallback_ptr,
        fallback_stride,
        out_ptr,
        num_seqs,
        total_q,
        NEXTN: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_FALLBACK: tl.constexpr,
    ):
        # One program per request. Each recomputes the whole [num_seqs, nextn]
        # cumulative-rate tile (tiny: <= a few thousand floats) so no second
        # launch is needed to publish it, then ranks its own row against it.
        pid = tl.program_id(0)
        rows = tl.arange(0, BLOCK_S)
        cols = tl.arange(0, BLOCK_N)
        row_ok = rows < num_seqs
        col_ok = cols < NEXTN
        ok = row_ok[:, None] & col_ok[None, :]
        src = tl.load(remap_ptr + rows, mask=row_ok, other=-1)
        has_src = src >= 0
        safe = tl.where(has_src, src, 0).to(tl.int64)
        rates = tl.load(
            prev_ptr + safe[:, None] * prev_stride + cols[None, :],
            mask=ok & has_src[:, None],
            other=0.0,
        ).to(tl.float32)
        if HAS_FALLBACK:
            fb = tl.load(
                fallback_ptr
                + rows[:, None].to(tl.int64) * fallback_stride
                + cols[None, :],
                mask=ok,
                other=0.0,
            ).to(tl.float32)
            rates = tl.where(has_src[:, None], rates, fb)
        # Padding columns multiply by 1 so the scan leaves real columns alone,
        # then drop out of the ranking at -inf.
        rates = tl.where(col_ok[None, :], rates, 1.0)
        cum = tl.cumprod(rates, axis=1)
        cum = tl.where(ok, cum, float("-inf"))
        flat = rows[:, None] * NEXTN + cols[None, :]
        budget = total_q - num_seqs
        won = 0
        for j in tl.static_range(NEXTN):
            mine = tl.sum(
                tl.where((rows[:, None] == pid) & (cols[None, :] == j), cum, 0.0)
            )
            my_flat = pid * NEXTN + j
            beats = (cum > mine) | ((cum == mine) & (flat < my_flat))
            rank = tl.sum(beats.to(tl.int32))
            won += (rank < budget).to(tl.int32)
        tl.store(out_ptr + pid, 1 + won)

    _TRITON_KERNEL = (_seqlens_q_by_confidence_kernel, triton)
    return _TRITON_KERNEL


def compute_seqlens_q_by_confidence_rates(
    total_q: int,
    remap,
    prev_confidence_rates,
    seqlens_q,
    fallback_rates=None,
    max_num_seqs: int | None = None,
) -> None:
    """Triton version of `compute_seqlens_q_by_confidence_rates_ref`.

    ``max_num_seqs`` fixes the tile height so one compiled variant serves every
    batch size.
    """
    num_seqs = int(remap.shape[0])
    if num_seqs == 0:
        return
    if not remap.is_cuda:
        compute_seqlens_q_by_confidence_rates_ref(
            total_q, remap, prev_confidence_rates, seqlens_q, fallback_rates
        )
        return
    kernel, triton = _get_triton_kernel()
    nextn = int(prev_confidence_rates.shape[1])
    block_s = triton.next_power_of_2(max(num_seqs, int(max_num_seqs or 0), 1))
    block_n = triton.next_power_of_2(max(nextn, 1))
    assert prev_confidence_rates.stride(1) == 1
    has_fb = fallback_rates is not None
    if has_fb:
        assert fallback_rates.stride(1) == 1 and fallback_rates.shape[1] == nextn
    kernel[(num_seqs,)](
        remap,
        prev_confidence_rates,
        prev_confidence_rates.stride(0),
        fallback_rates if has_fb else prev_confidence_rates,
        fallback_rates.stride(0) if has_fb else 0,
        seqlens_q,
        num_seqs,
        int(total_q),
        NEXTN=nextn,
        BLOCK_S=block_s,
        BLOCK_N=block_n,
        HAS_FALLBACK=has_fb,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Device -> host feedback
# ---------------------------------------------------------------------------


class ConfidenceFeedback:
    """Ship each step's draft confidence to the host without a sync.

    `push` enqueues a non-blocking copy behind an event; `poll` hands back the
    newest copy whose event has completed, keyed by request id, so the
    scheduler sees confidence about one step late (DeepInfer's ZOS case).
    """

    def __init__(self, max_num_seqs: int, width: int, depth: int = 3):
        import torch

        self._bufs = [
            torch.empty(
                (max_num_seqs, width),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(depth)
        ]
        self._events = [torch.cuda.Event() for _ in range(depth)]
        self._pending: deque[tuple[int, list[int]]] = deque()
        self._next = 0

    def push(self, confidence, req_ids: Sequence[int]) -> None:
        n = len(req_ids)
        if n == 0:
            return
        if len(self._pending) == len(self._bufs):
            self._pending.popleft()
        slot = self._next
        self._next = (self._next + 1) % len(self._bufs)
        self._bufs[slot][:n].copy_(confidence[:n], non_blocking=True)
        self._events[slot].record()
        self._pending.append((slot, list(req_ids)))

    def poll(self) -> dict[int, np.ndarray] | None:
        ready = None
        while self._pending and self._events[self._pending[0][0]].query():
            ready = self._pending.popleft()
        if ready is None:
            return None
        slot, req_ids = ready
        host = self._bufs[slot][: len(req_ids)].numpy().copy()
        return {rid: host[i] for i, rid in enumerate(req_ids)}


class StepTimer:
    """CUDA-event timing of a decode step's forward and sampler phases.

    ``forward`` spans the graph replay; ``sampler`` spans everything after it
    up to the end of drafting (rejection sampling, the draft pass), which is
    the part DeepInfer models per request count.
    """

    def __init__(self, depth: int = 4):
        import torch

        self._ev = [
            tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
            for _ in range(depth)
        ]
        self._pending: deque[tuple[int, int, int, int, int]] = deque()
        self._next = 0
        self._open: tuple[int, int, int, int, int] | None = None

    def start(
        self, num_seqs: int, graph_size: int, max_q: int = 0, ctx_tokens: int = 0
    ) -> None:
        if len(self._pending) == len(self._ev):
            self._pending.popleft()
        slot = self._next
        self._next = (self._next + 1) % len(self._ev)
        self._ev[slot][0].record()
        self._open = (slot, int(num_seqs), int(graph_size), int(max_q), int(ctx_tokens))

    def forward_done(self) -> None:
        if self._open is not None:
            self._ev[self._open[0]][1].record()

    def finish(self) -> None:
        if self._open is None:
            return
        self._ev[self._open[0]][2].record()
        self._pending.append(self._open)
        self._open = None

    def cancel(self) -> None:
        self._open = None

    def poll(self) -> list[tuple[int, int, float, float, int, int]]:
        """``(num_seqs, graph_size, forward_ms, sampler_ms, max_q, ctx_tokens)``
        per finished step, the argument list of `MTPVarlen.update_runtime`."""
        out = []
        while self._pending and self._ev[self._pending[0][0]][2].query():
            slot, num_seqs, graph_size, max_q, ctx = self._pending.popleft()
            s, m, e = self._ev[slot]
            out.append(
                (num_seqs, graph_size, s.elapsed_time(m), m.elapsed_time(e), max_q, ctx)
            )
        return out
