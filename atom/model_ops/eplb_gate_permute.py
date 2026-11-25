# SPDX-License-Identifier: MIT
"""Gate-permute EPLB (no redundant experts) and the "swap" placement policy.

EPLB's default keeps the router on logical expert ids and remaps them to
physical slots in a launch of its own, on every forward of every placed layer.
With ``ATOM_EPLB_GATE_PERMUTE=1`` a rebalance instead moves the gate's rows
(``weight`` and the routing biases, `_GATE_ROWS`) together with the expert
weights, in place, so the router emits physical ids directly. Nothing is then
launched for EPLB in a decode graph.

The load statistics are logical and come from two places:

* prefill: a count launch a placed layer in every pass that is not a decode
  request-graph recording -- eager passes and the recordings a prefill replays
  -- into the monitor's pass buffer, folded to logical ids when a pure-prefill
  pass commits it (`ExpertLoadMonitor.on_forward_end`);
* decode: a decode request graph's top-k id tensors are kept referenced when
  it is recorded, which pins their graph-pool memory, so after a replay they
  hold that step's ids; one replay in ``ATOM_EPLB_DECODE_SAMPLE_EVERY`` is
  counted from them, eagerly (`GatePermuteRuntime.after_decode_replay`).

``placement_policy: "swap"`` re-places a layer by a few pairwise swaps from its
live placement (`swap_rebalance`), in numpy float64 from the group's integer
loads, so every rank derives the same plan.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from atom.utils import envs
from atom.utils.graph_holders import register_graph_holder

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

logger = logging.getLogger("atom")

TAG = "[EPLB-GP]"

# The gate's per-expert tensors: a placement moves their rows with the experts.
_GATE_ROWS = ("weight", "e_score_correction_bias", "bias_vl")

# The swap objective's terms -- prefill rows, the decode active-expert proxy,
# decode rows -- each a rank load whose mean over the ranks is 1, their weights,
# and the weight of the smoothing term (the sum of squared rank loads) that
# lets the greedy leave a plateau of the max.
SWAP_WEIGHTS = (1.0, 1.0, 0.25)
SWAP_SMOOTH = 1.0


# ---------------------------------------------------------------- swap policy
def rank_loads(values: np.ndarray, rk: np.ndarray, num_ranks: int) -> np.ndarray:
    """[experts, terms] summed over the experts of each rank -> [ranks, terms],
    each sum in expert order (no BLAS: the same bits on every rank)."""
    out = np.zeros((num_ranks, values.shape[1]), dtype=np.float64)
    for m in range(values.shape[1]):
        out[:, m] = np.bincount(rk, weights=values[:, m], minlength=num_ranks)
    return out


def active_proxy(dec: np.ndarray, rows: float, top_k: int) -> np.ndarray:
    """a_e = 1 - (1 - q_e)^(top_k * rows): the chance expert e is active in a
    decode step of ``rows`` rows, q_e its share of the decode picks (picks as
    if independent)."""
    total = float(dec.sum())
    if total <= 0 or rows <= 0:
        return np.zeros(dec.shape, dtype=np.float64)
    share = np.clip(dec / total, 0.0, 1.0)
    return 1.0 - (1.0 - share) ** (float(rows) * top_k)


def swap_terms(
    pf: np.ndarray, dec: np.ndarray, rows: float, top_k: int, num_ranks: int
) -> np.ndarray:
    """[experts, 3] objective terms of one layer from its prefill and decode
    pick counts; a term with no data is zero (it then weighs nothing)."""
    pf = np.asarray(pf, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    values = np.zeros((pf.shape[0], 3), dtype=np.float64)
    if pf.sum() > 0:
        values[:, 0] = pf / pf.sum() * num_ranks
    if dec.sum() > 0:
        proxy = active_proxy(dec, rows, top_k)
        if proxy.sum() > 0:
            values[:, 1] = proxy / proxy.sum() * num_ranks
        values[:, 2] = dec / dec.sum() * num_ranks
    return values


def greedy_swaps(
    values: np.ndarray,
    rk: np.ndarray,
    num_ranks: int,
    *,
    budget: int | None,
    eps: float,
    weights=SWAP_WEIGHTS,
) -> list[tuple[int, int]]:
    """Greedy pairwise swaps, in place on ``rk`` (expert -> rank).

    Each round takes the heaviest rank of every term and tries every swap of
    one of its experts with one of another rank; the swap that lowers
    ``sum_m w_m * (max_r L[r, m] + SWAP_SMOOTH * sum_r L[r, m]^2)`` most is applied
    if it lowers it by more than ``eps``. Stops at ``budget`` swaps (None: no
    bound). Ties go to the first candidate in (heavy rank, other rank, expert
    of the heavy rank, expert of the other rank) order.

    A round is a dozen passes over [terms, experts a rank, the other experts]
    float64 buffers that are allocated once: it runs in the forward path.
    """
    w = np.asarray(weights, dtype=np.float64)
    num_experts, num_terms = values.shape
    per = num_experts // num_ranks
    assert per * num_ranks == num_experts
    other = num_experts - per
    by_term = np.ascontiguousarray(values.T)  # [terms, experts]
    loads = rank_loads(values, rk, num_ranks)  # [ranks, terms]
    delta = np.empty((num_terms, per, other), dtype=np.float64)
    gain = np.empty_like(delta)
    work = np.empty_like(delta)
    score = np.empty((per, other), dtype=np.float64)
    part = np.empty_like(score)
    no_rest = np.full(num_terms, -np.inf)
    swaps: list[tuple[int, int]] = []
    while budget is None or len(swaps) < budget:
        # members[r]: the experts of rank r, ascending.
        members = np.argsort(rk, kind="stable").reshape(num_ranks, per)
        top = loads.max(0)
        best_gain, best = float(eps), None
        heavy = sorted({int(loads[:, m].argmax()) for m in range(len(w)) if w[m] > 0})
        for h in heavy:
            others = [r for r in range(num_ranks) if r != h]
            if not others:
                continue
            mine, theirs = members[h], members[others].reshape(-1)
            load_h = loads[h][:, None, None]
            # Per expert of the other ranks: its rank's load, and the heaviest
            # of the ranks that are neither its own nor h.
            load_l = np.repeat(loads[others], per, axis=0).T[:, None, :]
            rest = np.repeat(
                np.stack(
                    [
                        (
                            np.delete(loads, [h, r], axis=0).max(0)
                            if num_ranks > 2
                            else no_rest
                        )
                        for r in others
                    ]
                ),
                per,
                axis=0,
            ).T[:, None, :]
            # delta[m, i, j]: the load of term m that a swap of mine[i] with
            # theirs[j] moves onto rank h.
            np.subtract(
                by_term[:, theirs][:, None, :], by_term[:, mine][:, :, None], out=delta
            )
            np.add(delta, load_h, out=gain)
            np.subtract(load_l, delta, out=work)
            np.maximum(gain, work, out=gain)
            np.maximum(gain, rest, out=gain)
            np.subtract(top[:, None, None], gain, out=gain)  # what the max drops
            # less the change of the sum of squares: 2 d (L_l - L_h - d)
            np.subtract(load_l - load_h, delta, out=work)
            np.multiply(delta, SWAP_SMOOTH * 2, out=delta)
            np.multiply(delta, work, out=work)
            np.add(gain, work, out=gain)
            np.multiply(gain[0], w[0], out=score)
            for m in range(1, num_terms):
                np.multiply(gain[m], w[m], out=part)
                np.add(score, part, out=score)
            flat_score = score.reshape(-1)
            flat = int(flat_score.argmax())
            value = float(flat_score[flat])
            if value > best_gain:
                tied = np.flatnonzero(flat_score == value)
                if len(tied) > 1:
                    i, j = np.divmod(tied, other)
                    flat = int(tied[np.lexsort((j, i, j // per))[0]])
                i, j = divmod(flat, other)
                best_gain = value
                best = (h, others[j // per], int(mine[i]), int(theirs[j]))
        if best is None:
            break
        h, low, a, b = best
        loads[h] += values[b] - values[a]
        loads[low] += values[a] - values[b]
        rk[a], rk[b] = low, h
        swaps.append((a, b))
    return swaps


def swap_rebalance(
    pf_load: torch.Tensor,
    dec_load: torch.Tensor | None,
    rows_per_step: float,
    old_p2l: torch.Tensor,
    *,
    num_gpus: int,
    top_k: int,
    budget: int | None,
    eps: float,
    layer_ids: list[int] | None = None,
    weights=SWAP_WEIGHTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    """The "swap" policy: `rebalance_experts`' result for a placement that
    differs from ``old_p2l`` by at most ``budget`` pairwise swaps a layer.

    ``pf_load`` / ``dec_load``: [layers, logical] pick counts (prefill, sampled
    decode; None: none), ``rows_per_step`` the group's decode rows a step,
    ``weights`` the objective's (prefill rows, active proxy, decode rows).
    A swap exchanges two experts' physical slots; a layer with none keeps its
    row of ``old_p2l`` as is. Returns ``(physical_to_logical [L, P],
    logical_to_physical [L, Lg, 1], logcnt [L, Lg], swaps a layer)``.
    """
    p2l = old_p2l.detach().to("cpu", torch.int64).numpy().copy()
    num_layers, num_physical = p2l.shape
    pf = pf_load.detach().to("cpu").numpy().astype(np.float64)
    dec = (
        np.zeros_like(pf)
        if dec_load is None
        else dec_load.detach().to("cpu").numpy().astype(np.float64)
    )
    assert pf.shape == dec.shape == (num_layers, num_physical), (
        "the swap policy places as many logical experts as there are slots: "
        f"loads {pf.shape} / {dec.shape}, placement {p2l.shape}"
    )
    assert num_physical % num_gpus == 0
    per = num_physical // num_gpus
    slots = np.arange(num_physical, dtype=np.int64)
    l2p = np.empty_like(p2l)
    swaps: dict[int, int] = {}
    for layer in range(num_layers):
        assert np.array_equal(np.sort(p2l[layer]), slots), (
            f"layer {layer}: the swap policy needs a permutation placement "
            "(no redundant experts)"
        )
        l2p[layer, p2l[layer]] = slots
    for layer in range(num_layers) if layer_ids is None else layer_ids:
        values = swap_terms(pf[layer], dec[layer], rows_per_step, top_k, num_gpus)
        rk = l2p[layer] // per
        done = greedy_swaps(
            values, rk, num_gpus, budget=budget, eps=eps, weights=weights
        )
        for a, b in done:
            slot_a, slot_b = l2p[layer, a], l2p[layer, b]
            p2l[layer, slot_a], p2l[layer, slot_b] = b, a
            l2p[layer, a], l2p[layer, b] = slot_b, slot_a
        swaps[layer] = len(done)
    dev = old_p2l.device
    return (
        torch.from_numpy(p2l).to(dev, torch.int32),
        torch.from_numpy(l2p).to(dev, torch.int32).unsqueeze(-1),
        torch.ones((num_layers, num_physical), dtype=torch.int32, device=dev),
        swaps,
    )


def head_share(p2l: np.ndarray, load: np.ndarray, per_rank: int, head: int) -> float:
    """How hot-first one layer's placement is: the mean over ranks of the
    share of the rank's ``load`` that sits in its first ``head`` slots."""
    shares = []
    for start in range(0, p2l.shape[0], per_rank):
        mine = load[p2l[start : start + per_rank]]
        total = float(mine.sum())
        if total > 0:
            shares.append(float(mine[:head].sum()) / total)
    return float(np.mean(shares)) if shares else 0.0


# ---------------------------------------------------------------- metrics
def placement_metrics(
    pf: np.ndarray,
    dec: np.ndarray,
    rows: float,
    rk: np.ndarray,
    num_ranks: int,
    top_k: int,
) -> tuple[float, float, float, float]:
    """One layer under ``rk`` (logical expert -> rank): prefill-row max/mean
    over the ranks, the active-expert proxy's max and mean rank sum, decode-row
    max/mean. NaN where the layer has no such load."""
    nan = float("nan")

    def imbalance(x):
        load = np.bincount(rk, weights=x, minlength=num_ranks)
        return float(load.max() / load.mean()) if load.sum() > 0 else nan

    proxy = np.bincount(rk, weights=active_proxy(dec, rows, top_k), minlength=num_ranks)
    has_proxy = proxy.sum() > 0
    return (
        imbalance(pf),
        float(proxy.max()) if has_proxy else nan,
        float(proxy.mean()) if has_proxy else nan,
        imbalance(dec),
    )


def _mean(values) -> float:
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def metrics_line(
    pf: np.ndarray,
    dec: np.ndarray,
    rows: float,
    rk: np.ndarray,
    layer_ids: list[int],
    num_ranks: int,
    top_k: int,
) -> str:
    """The metrics of one placement (``rk`` [layers, logical] -> rank) on the
    loads ``pf`` / ``dec`` [layers, logical], averaged over the layers: the
    prefill figure over the lower half of the layer ids and over the upper
    half less the last layer (which a bounded-replay prefill runs on one row
    a request)."""
    per_layer = {
        i: placement_metrics(pf[i], dec[i], rows, rk[i], num_ranks, top_k)
        for i in layer_ids
    }
    half = (max(layer_ids) + 1) // 2
    low = [i for i in layer_ids if i < half]
    high = [i for i in layer_ids if half <= i < max(layer_ids)]

    def span(ids):
        return f"L{ids[0]}-{ids[-1]}" if ids else "L-"

    return (
        f"prefill max/mean {span(low)} {_mean(per_layer[i][0] for i in low):.3f} "
        f"{span(high)} {_mean(per_layer[i][0] for i in high):.3f} | "
        f"active proxy max {_mean(m[1] for m in per_layer.values()):.2f} "
        f"mean/rank {_mean(m[2] for m in per_layer.values()):.2f} | "
        f"decode-row max/mean {_mean(m[3] for m in per_layer.values()):.3f}"
    )


def ranks_of(p2l: np.ndarray, per_rank: int) -> np.ndarray:
    """[layers, physical] slot -> logical  ->  [layers, logical] -> rank."""
    rk = np.empty_like(p2l)
    slots = np.arange(p2l.shape[1]) // per_rank
    for layer in range(p2l.shape[0]):
        rk[layer, p2l[layer]] = slots
    return rk


# ---------------------------------------------------------------- operating point
def rows_per_step(rows: int, steps: int, world: int) -> float:
    """The group's rows a step from the ranks' summed rows and steps (every
    rank counts the same steps, so the steps of one are ``steps / world``)."""
    return rows * world / steps if steps > 0 else 0.0


# ---------------------------------------------------------------- gate rows
def stage_rows(
    tensors: list[torch.Tensor], old_p2l: torch.Tensor, new_p2l: torch.Tensor
) -> list[torch.Tensor] | None:
    """The rows of every tensor as the placement ``new_p2l`` holds them, read
    from tensors that hold them as ``old_p2l`` does (physical slot -> logical
    expert, permutations): row p of the result is the row of logical
    ``new_p2l[p]``, which row ``old_l2p[new_p2l[p]]`` holds now. Copies, with
    nothing written; None when the two placements agree."""
    old = old_p2l.to(torch.int64)
    new = new_p2l.to(torch.int64)
    if torch.equal(old, new):
        return None
    slots = torch.arange(old.numel(), dtype=torch.int64, device=old.device)
    if not (
        torch.equal(old.sort().values, slots) and torch.equal(new.sort().values, slots)
    ):
        raise ValueError("a gate moves between permutation placements only")
    old_l2p = torch.empty_like(old)
    old_l2p[old] = slots
    source = old_l2p[new]
    staged = [t.index_select(0, source.to(t.device)) for t in tensors]
    for tensor, rows in zip(tensors, staged):
        if tensor.shape != rows.shape or tensor.dtype != rows.dtype:
            raise ValueError("a staged gate tensor does not match its target")
    return staged


def gate_row_tensors(gate: Any) -> list[torch.Tensor]:
    """The gate's per-expert tensors a placement moves (`_GATE_ROWS`)."""
    return [
        param.data
        for param in (getattr(gate, name, None) for name in _GATE_ROWS)
        if param is not None
    ]


def collect_gates(model: Any, layers: dict[int, Any]) -> dict[int, Any]:
    """Layer id -> the gate that routes ``layers[layer id]`` (an EP MoE layer
    EPLB places). Raises for a layer gate-permute cannot serve."""
    by_experts = {
        id(module.experts): module.gate
        for module in model.modules()
        if getattr(module, "gate", None) is not None
        and getattr(module, "experts", None) is not None
    }
    gates: dict[int, Any] = {}
    for layer_id, layer in layers.items():
        name = getattr(layer, "prefix", f"layer {layer_id}")

        def refuse(why, name=name):
            raise RuntimeError(f"ATOM_EPLB_GATE_PERMUTE cannot place {name}: {why}")

        gate = by_experts.get(id(layer))
        if gate is None:
            refuse("no module holds it as `experts` beside a `gate`")
        if hasattr(gate, "tid2eid"):
            refuse("its gate is hash-routed")
        layout = getattr(layer, "expert_layout", None)
        if layout is not None and (
            layout.shared_is_routed or layout.num_fused_shared_experts
        ):
            refuse("a shared expert is fused into its routed experts")
        if not getattr(layer.quant_method, "eplb_gate_permute_capable", False):
            refuse(f"{type(layer.quant_method).__name__} does not place for EPLB")
        parallel = getattr(layer, "moe_parallel_config", None)
        if parallel is not None and (
            getattr(parallel, "pcp_merge", False)
            or (
                getattr(layer, "dp_size", 1) > 1
                and not getattr(parallel, "use_all2all_kernels", True)
            )
        ):
            refuse("its rows are gathered across ranks before the experts")
        rows = gate.weight.shape[0]
        if rows != layer.global_num_experts:
            refuse(f"its gate has {rows} rows for {layer.global_num_experts} slots")
        bias = getattr(layer, "e_score_correction_bias", None)
        gate_bias = getattr(gate, "e_score_correction_bias", None)
        if bias is not None and (
            gate_bias is None or bias.data_ptr() != gate_bias.data_ptr()
        ):
            refuse("its routing bias is a copy of the gate's, which would go stale")
        gates[layer_id] = gate
    return gates


# ---------------------------------------------------------------- counting
if _HAS_TRITON:

    @triton.jit
    def _count_ids_kernel(
        ids_ptr,  # [numel] expert ids; < 0 or >= num_physical is not counted
        load_ptr,  # [num_physical] int32 counts, added to
        flag_ptr,  # [1] int32: 0 leaves the counts alone (nothing is counting)
        num_physical,
        numel,
        stride,  # every ``stride``-th block of ids is counted, ``stride``-fold
        BLOCK: tl.constexpr,
        NUM_BINS: tl.constexpr,  # next_pow2(num_physical + 1)
    ):
        # `eplb._eplb_route_kernel`'s record half, over a sample of the ids:
        # every program adds to the same ``num_physical`` counters, so a
        # prefill pass counted id by id would queue up on them.
        pid = tl.program_id(0)
        on = tl.load(flag_ptr) != 0
        offs = pid * stride * BLOCK + tl.arange(0, BLOCK)
        mask = (offs < numel) & on
        ids = tl.load(ids_ptr + offs, mask=mask, other=-1).to(tl.int64)
        in_range = mask & (ids >= 0) & (ids < num_physical)
        bin_idx = tl.where(in_range, ids, num_physical).to(tl.int32)
        hist = tl.histogram(bin_idx, NUM_BINS)
        bins = tl.arange(0, NUM_BINS)
        tl.atomic_add(
            load_ptr + bins,
            (hist * stride).to(tl.int32),
            mask=(bins < num_physical) & (hist > 0),
        )


def count_stride(numel: int, block: int, blocks: int) -> int:
    """Every how many blocks of ``block`` ids a pass of ``numel`` ids is
    counted so that about ``blocks`` of them are."""
    return max(1, -(-numel // block) // blocks)


class GatePermuteRuntime:
    """What gate-permute EPLB holds beside the manager: the gates, the decode
    load and the decode graphs' id tensors."""

    def __init__(
        self,
        *,
        manager: Any,
        monitor: Any,
        layers: dict[int, Any],
        gates: dict[int, Any],
        meta: Any,
    ):
        if meta.num_physical_experts != meta.num_logical_experts:
            raise RuntimeError(
                "ATOM_EPLB_GATE_PERMUTE needs num_redundant_experts == 0, got "
                f"{meta.num_physical_experts - meta.num_logical_experts}"
            )
        self.manager = manager
        self.monitor = monitor
        self.layers = layers
        self.gates = gates
        self.meta = meta
        # The sampled Triton count launch; without Triton the monitor's own.
        self.count_triton = _HAS_TRITON
        # The count launch recorded into the prefill graphs reads this: 1
        # while loads are counted, 0 otherwise (`set_count_flag`) -- a frozen
        # placement's graphs still launch it, and it must then cost nothing.
        self._count_flag = torch.ones(
            1, dtype=torch.int32, device=meta.physical_to_logical_map.device
        )
        self._count_flag_on = True
        self.sample_every = int(envs.ATOM_EPLB_DECODE_SAMPLE_EVERY)
        dev = meta.physical_to_logical_map.device
        shape = (meta.num_layers, meta.num_logical_experts)
        # Logical expert -> its slot, the index that folds a pass's physical
        # counts to logical ones. Fixed address, rewritten a layer a commit.
        self.fold_index = (
            meta.logical_to_physical_map[:, :, 0].to(torch.int64).contiguous()
        )
        # The decode samples and the prefill passes, summed since the start.
        self.dec_load = torch.zeros(shape, dtype=torch.int64, device=dev)
        self.pf_total = torch.zeros(shape, dtype=torch.int64, device=dev)
        self.dec_rows = 0  # this rank's live rows over the sampled replays
        self.dec_steps = 0  # the sampled replays
        # The same two since the last rebalance: the operating point the
        # next one is planned for.
        self.window_rows = 0
        self.window_steps = 0
        self._replays = 0
        # A decode sample is added here by slot, one column a slot after the
        # column pad picks (-1) fall into, and folded into `dec_load` before
        # that is read and before a layer's placement changes (`fold_decode`):
        # the host runs a sampled step's few launches while the step's own
        # recording is still being launched, so each one lengthens that step.
        slots = meta.num_physical_experts + 1
        self.dec_raw = torch.zeros(
            (meta.num_layers, slots), dtype=torch.int64, device=dev
        )
        self._raw_base = (
            torch.arange(meta.num_layers, dtype=torch.int64, device=dev) * slots + 1
        )
        self._raw_dirty = False
        self._ones = torch.ones(1, dtype=torch.int64, device=dev)
        # Request bucket -> (its held id tensors, their layers' bases), where
        # the tensors share a shape.
        self._plans: dict[Any, tuple[tuple[torch.Tensor, ...], torch.Tensor] | None] = (
            {}
        )
        # Request bucket -> [(layer id, the recording's top-k ids)].
        self._captures: dict[Any, list[tuple[int, torch.Tensor]]] = {}
        self._layer_indices: dict[tuple[int, ...], torch.Tensor] = {}
        self._capture_key = None
        self._warned: set = set()
        # The drafter's MoE layers' placement (ATOM_EPLB_MTP), else None.
        self.mtp: MtpPlacement | None = None
        register_graph_holder(self)

    @property
    def counting(self) -> bool:
        """Whether loads are being counted: while the placement is learnt."""
        return not self.manager.rebalancing_stopped

    # -- routing ---------------------------------------------------------
    def owns(self, layer: Any) -> bool:
        return self.layers.get(getattr(layer, "layer_id", None)) is layer

    def place(self, layer: Any, topk_ids: torch.Tensor) -> None:
        """A placed layer's routed ids, which are physical already: kept for
        the decode sampler inside a decode request-graph recording (nothing
        launched), counted into the monitor's pass buffer otherwise."""
        if not self.owns(layer):
            if self.mtp is not None:
                self.mtp.place(layer, topk_ids)
            return
        if self._capture_key is not None:
            self._captures[self._capture_key].append((layer.layer_id, topk_ids))
            return
        if not self.counting:
            return
        num_physical = self.meta.num_physical_experts
        if not self.count_triton:
            self.monitor.record(
                layer_id=layer.layer_id,
                topk_physical=topk_ids,
                num_physical=num_physical,
            )
            return
        load = self.monitor.pass_count_buffer(layer.layer_id, num_physical)
        numel = topk_ids.numel()
        if load is None or numel == 0:
            return
        # About 32 of the pass's 256-id blocks, evenly spread and scaled back
        # up: every block adds to the same counters, so a pass counted whole
        # queues on them.
        stride = count_stride(numel, 256, 32)
        _count_ids_kernel[(triton.cdiv(triton.cdiv(numel, 256), stride),)](
            topk_ids.contiguous(),
            load,
            self._count_flag,
            num_physical,
            numel,
            stride,
            BLOCK=256,
            NUM_BINS=1 << num_physical.bit_length(),
        )

    def sync_count_flag(self) -> None:
        """The device flag the recorded count launches read, brought in line
        with `counting` (one small write, and only when that changed)."""
        on = self.counting
        if on != self._count_flag_on:
            self._count_flag.fill_(1 if on else 0)
            self._count_flag_on = on

    # -- decode sampling -------------------------------------------------
    @contextmanager
    def decode_capture(self, key: Any):
        """Around the recording of the decode request graph ``key``."""
        self._captures[key] = []
        self._plans.pop(key, None)
        self._capture_key = key
        try:
            yield
        finally:
            self._capture_key = None

    def after_decode_replay(self, key: Any, scheduled_tokens: int) -> None:
        """After the decode request graph ``key`` replayed a step of
        ``scheduled_tokens`` live rows on this rank: every
        ``ATOM_EPLB_DECODE_SAMPLE_EVERY``-th replay is counted."""
        if self.manager.rebalancing_stopped:
            return
        # The first replay after a prefill-bearing step and every
        # ``sample_every``-th from there: `sync_phase` restarts the count on
        # every rank in the same step, so the ranks sample the same decode
        # steps (they step together, and a sampled step is a little longer).
        self._replays += 1
        mtp = self.mtp
        if mtp is not None:
            # The step's draft pass is sampled where this step is.
            mtp.due = False
        if (self._replays - 1) % self.sample_every:
            return
        if mtp is not None:
            mtp.due = True
        held = self._captures.get(key)
        if not held:
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(
                    "%s decode graph %s holds no top-k ids of a placed layer; "
                    "its replays are not sampled",
                    TAG,
                    key,
                )
            return
        rows = int(scheduled_tokens)
        self.dec_steps += 1
        self.window_steps += 1
        if rows <= 0:
            return
        self.dec_rows += rows
        self.window_rows += rows
        if key not in self._plans:
            self._plans[key] = self._plan(held)
        plan = self._plans[key]
        if plan is None:
            self.count_decode(held, rows)
            return
        # Three launches: the held tensors stacked, their slots' addresses in
        # `dec_raw`, and the add. Rows past the live ones are cut off; a pad
        # pick (-1) lands in its layer's pad column.
        tensors, base = plan
        flat = (torch.stack(tensors)[:, :rows] + base).view(-1)
        count = flat.numel()
        if self._ones.numel() < count:
            self._ones = torch.ones(count, dtype=torch.int64, device=flat.device)
        self.dec_raw.view(-1).scatter_add_(0, flat, self._ones[:count])
        self._raw_dirty = True

    def _plan(self, held: list[tuple[int, torch.Tensor]]):
        """The held id tensors of a request bucket and their layers' first
        slot address in `dec_raw` ([layers, 1, 1]) -- None where the tensors
        differ in shape (several launches a layer), which `count_decode`
        counts one by one."""
        if len({(tuple(ids.shape), ids.dtype) for _, ids in held}) != 1:
            return None
        if held[0][1].dim() != 2 or held[0][1].dtype not in (torch.int32, torch.int64):
            return None
        layers = self._layer_index(tuple(i for i, _ in held), held[0][1].device)
        base = self._raw_base.index_select(0, layers).view(-1, 1, 1)
        return tuple(ids for _, ids in held), base

    def fold_decode(self) -> None:
        """The decode samples held by slot (`dec_raw`) added to `dec_load` by
        logical expert, under the placement they were sampled under: before
        `dec_load` is read, and before a layer's fold index changes."""
        if not self._raw_dirty:
            return
        self.dec_load.add_(torch.gather(self.dec_raw[:, 1:], 1, self.fold_index))
        self.dec_raw.zero_()
        self._raw_dirty = False

    def sync_phase(self) -> None:
        """A prefill-bearing step ended -- on every rank in the same step
        (`EPLBManager.on_forward_pass_end`): the decode sampler's phase
        restarts."""
        self._replays = 0

    def count_decode(self, held: list[tuple[int, torch.Tensor]], rows: int) -> None:
        """Add the ids ``held`` -- (layer id, [bucket rows, top_k] physical
        ids) -- of a step's ``rows`` live rows to the decode load, by logical
        expert. Pad rows are routed to -1 and trail the live ones; both are
        used."""
        shapes = {tuple(ids.shape) for _, ids in held}
        groups = (
            [held]
            if len(shapes) == 1
            # Several launches a layer (micro-batches): count each as it is.
            else [[item] for item in held]
        )
        for group in groups:
            whole = len(groups) == 1
            ids = torch.stack([t[:rows] if whole else t for _, t in group]).to(
                torch.int64
            )
            self._add_physical(
                self.dec_load,
                self._layer_index(tuple(i for i, _ in group), ids.device),
                ids.reshape(len(group), -1),
            )

    def _layer_index(self, layers: tuple[int, ...], device: Any) -> torch.Tensor:
        """``layers`` as an int64 tensor on ``device``, built once: a tensor
        made from host data is a synchronous copy, which would hold the host
        until the replay just launched has finished -- on every sampled step."""
        index = self._layer_indices.get(layers)
        if index is None or index.device != device:
            index = torch.tensor(layers, dtype=torch.int64, device=device)
            self._layer_indices[layers] = index
        return index

    def _add_physical(
        self, load: torch.Tensor, layer: torch.Tensor, ids: torch.Tensor
    ) -> None:
        """load[layer[n], logical of ids[n, :]] += 1 for the ids in range."""
        num_physical = self.meta.num_physical_experts
        valid = (ids >= 0) & (ids < num_physical)
        safe = torch.where(valid, ids, torch.zeros_like(ids))
        p2l = self.meta.physical_to_logical_map.index_select(0, layer)
        logical = torch.gather(p2l.to(torch.int64), 1, safe)
        flat = logical + (layer * load.shape[1]).unsqueeze(1)
        load.view(-1).scatter_add_(
            0, flat.reshape(-1), valid.reshape(-1).to(load.dtype)
        )

    def release_graphs(self) -> int:
        """The graphs are being dropped (`graph_holders`): so are the id
        tensors their recordings left in the pool."""
        dropped = len(self._captures)
        self._captures.clear()
        self._plans.clear()
        if self.mtp is not None:
            dropped += self.mtp.release_graphs()
        return dropped

    # -- statistics ------------------------------------------------------
    def dump_global(
        self, group: Any
    ) -> tuple[torch.Tensor, torch.Tensor, float, int, float, int]:
        """The group's sums: the decode load and the cumulative prefill load
        [L, Lg], the group's live decode rows a sampled step and the sampled
        steps of a rank (their mean) -- and those two over the samples since
        the last rebalance (`start_window`), the window's operating point."""
        self.fold_decode()
        dev = self.dec_load.device
        parts = [self.dec_load.reshape(-1), self.pf_total.reshape(-1)]
        counters = [self.dec_rows, self.dec_steps, self.window_rows, self.window_steps]
        mtp = self.mtp
        if mtp is not None:
            # The drafter's layers ride the same collective: their decode
            # load and their four counters, ahead of the main ones.
            mtp.fold()
            parts.append(mtp.dec_load.reshape(-1))
            counters = mtp.counters() + counters
        parts.append(torch.tensor(counters, dtype=torch.int64, device=dev))
        payload = torch.cat(parts)
        world = int(getattr(group, "world_size", 1)) if group is not None else 1
        device_group = getattr(group, "device_group", None)
        if world > 1 and getattr(group, "reduce_on_cpu", False):
            # No device collective (ATOM_EPLB_MIGRATE_VIA=shm): the sums go
            # over the CPU group -- one read back from the device a rebalance.
            host = payload.cpu()
            torch.distributed.all_reduce(
                host, op=torch.distributed.ReduceOp.SUM, group=group.cpu_group
            )
            payload = host.to(dev)
        elif world > 1:
            if device_group is None:
                raise RuntimeError("gate-permute EPLB needs the EP device group")
            torch.distributed.all_reduce(
                payload, op=torch.distributed.ReduceOp.SUM, group=device_group
            )
        size = self.dec_load.numel()
        dec = payload[:size].view_as(self.dec_load)
        pf = payload[size : 2 * size].view_as(self.pf_total)
        tail = [int(v) for v in payload[-len(counters) :].tolist()]
        rows, steps, window_rows, window_steps = tail[-4:]
        if mtp is not None:
            start = payload.numel() - len(counters) - mtp.dec_load.numel()
            mtp.take_reduced(
                payload[start : start + mtp.dec_load.numel()], tail[:4], world
            )
        return (
            dec,
            pf,
            rows_per_step(rows, steps, world),
            steps // world,
            rows_per_step(window_rows, window_steps, world),
            window_steps // world,
        )

    def start_window(self) -> None:
        """A rebalance was planned: the samples of the next operating point
        start here."""
        self.window_rows = self.window_steps = 0
        if self.mtp is not None:
            self.mtp.window_rows = self.mtp.window_steps = 0

    # -- commit ----------------------------------------------------------
    def stage_commit(self, layer_id: int, old_p2l: torch.Tensor, new_p2l: torch.Tensor):
        """Before ``layer_id``'s weights move from ``old_p2l`` to ``new_p2l``:
        its gate rows in the new order, as copies. Everything of the gate's
        move that can fail happens here, with nothing written."""
        tensors = gate_row_tensors(self.gates[layer_id])
        staged = stage_rows(tensors, old_p2l, new_p2l)
        return None if staged is None else (tensors, staged)

    def apply_commit(self, layer_id: int, staged) -> bool:
        """``layer_id``'s weights and maps are the new placement's: write the
        gate rows `stage_commit` staged, and the layer's fold index. True when
        anything moved."""
        if staged is None:
            return False
        # The decode samples still held by slot were taken under the old
        # placement.
        self.fold_decode()
        for tensor, rows in zip(*staged):
            tensor.copy_(rows)
        self.fold_index[layer_id].copy_(
            self.meta.logical_to_physical_map[layer_id, :, 0]
        )
        return True

    def commit_layer(
        self, layer_id: int, old_p2l: torch.Tensor, new_p2l: torch.Tensor
    ) -> bool:
        """`stage_commit` and `apply_commit` in one, for a layer whose weights
        and maps already moved."""
        return self.apply_commit(
            layer_id, self.stage_commit(layer_id, old_p2l, new_p2l)
        )

    def describe(self) -> str:
        return (
            f"count={'triton' if self.count_triton else 'torch'} "
            f"decode_sample_every={self.sample_every} "
            f"mtp={int(self.mtp is not None)} "
            f"mtp_layers={0 if self.mtp is None else len(self.mtp.layers)}"
        )


class MtpPlacement:
    """Gate-permute EPLB's placement of the drafter's MoE layers
    (ATOM_EPLB_MTP): the DSpark draft stages, whose expert count is their
    own, so their maps, loads and sampler are theirs too.

    They start at the identity and are placed on decode samples alone: a
    draft pass is its own recording (`DraftGraph`), whose top-k id tensors
    are kept like a decode request graph's, and the draft pass of a step the
    target sampler takes is added by slot (`after_draft_replay`) and folded
    to logical experts lazily (`fold`). The sums ride the main loads'
    collective (`GatePermuteRuntime.dump_global`).
    """

    def __init__(
        self,
        *,
        layers: dict[int, Any],
        gates: dict[int, Any],
        ep_size: int,
        device: Any,
    ):
        self.layer_ids = sorted(layers)
        self.layers = [layers[i] for i in self.layer_ids]
        self.gates = [gates[i] for i in self.layer_ids]
        experts = {int(layer.global_num_experts) for layer in self.layers}
        top_ks = {int(layer.top_k) for layer in self.layers}
        if len(experts) != 1 or len(top_ks) != 1:
            raise RuntimeError(
                f"ATOM_EPLB_MTP: the drafter's MoE layers differ ({experts} "
                f"experts, top-{top_ks})"
            )
        self.num_experts = experts.pop()
        self.top_k = top_ks.pop()
        self.ep_size = int(ep_size)
        if self.num_experts % self.ep_size:
            raise RuntimeError(
                f"ATOM_EPLB_MTP: {self.num_experts} experts over {ep_size} ranks"
            )
        self.per_rank = self.num_experts // self.ep_size
        count, slots = len(self.layers), self.num_experts + 1
        identity = torch.arange(self.num_experts, device=device)
        # Slot -> logical expert, and logical expert -> slot (the fold index).
        self.p2l = identity.to(torch.int32).repeat(count, 1)
        self.fold_index = identity.to(torch.int64).repeat(count, 1)
        self.dec_load = torch.zeros(
            (count, self.num_experts), dtype=torch.int64, device=device
        )
        # A sample lands here by slot, column 0 taking the pad picks (-1).
        self.dec_raw = torch.zeros((count, slots), dtype=torch.int64, device=device)
        self._raw_base = (
            torch.arange(count, dtype=torch.int64, device=device) * slots + 1
        )
        self._raw_dirty = False
        self._ones = torch.ones(1, dtype=torch.int64, device=device)
        # This rank's live draft rows over the sampled draft passes, and the
        # passes: since the start, and since the last rebalance.
        self.rows = self.steps = 0
        self.window_rows = self.window_steps = 0
        # Set by the target sampler for the step it takes.
        self.due = False
        self._index = {id(layer): i for i, layer in enumerate(self.layers)}
        # Draft recording -> [(layer index, its top-k ids)], and its plan.
        self._captures: dict[Any, list[tuple[int, torch.Tensor]]] = {}
        self._plans: dict[Any, Any] = {}
        self._capture_key = None
        self._warned: set = set()
        # The group's sums as last reduced: (decode load [layers, experts],
        # rows a step since the start, a rank's samples, rows a step in the
        # window, a rank's samples in it).
        self.reduced: tuple[torch.Tensor, float, int, float, int] | None = None

    # -- routing ---------------------------------------------------------
    def owns(self, layer: Any) -> bool:
        return id(layer) in self._index

    def place(self, layer: Any, topk_ids: torch.Tensor) -> None:
        """A draft layer's routed ids, physical already: kept inside the
        recording of a draft pass, nothing otherwise (and never a launch)."""
        if self._capture_key is None:
            return
        index = self._index.get(id(layer))
        if index is not None:
            self._captures[self._capture_key].append((index, topk_ids))

    @contextmanager
    def capture(self, key: Any):
        """Around the recording of the draft pass ``key``."""
        self._captures[key] = []
        self._plans.pop(key, None)
        self._capture_key = key
        try:
            yield
        finally:
            self._capture_key = None

    def after_draft_replay(self, key: Any, rows: int) -> None:
        """After the draft pass ``key`` ran a step of ``rows`` live rows on
        this rank: counted where the target sampler took the step."""
        if not self.due:
            return
        self.due = False
        held = self._captures.get(key)
        if not held:
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(
                    "%s draft pass %s holds no top-k ids of a draft layer (not "
                    "recorded, or not this flavor); it is not sampled",
                    TAG,
                    key,
                )
            return
        rows = int(rows)
        self.steps += 1
        self.window_steps += 1
        if rows <= 0:
            return
        self.rows += rows
        self.window_rows += rows
        if key not in self._plans:
            self._plans[key] = self._plan(held)
        raw = self.dec_raw.view(-1)
        plan = self._plans[key]
        if plan is None:
            # Several launches a layer: one add a tensor, each as it is.
            for index, ids in held:
                flat = (ids.to(torch.int64) + self._raw_base[index]).view(-1)
                raw.scatter_add_(0, flat, self._count(flat))
        else:
            # Three launches, as the target layers' sample: the held tensors
            # stacked and cut to the live rows, their slots' addresses, the add.
            tensors, base = plan
            flat = (torch.stack(tensors)[:, :rows] + base).view(-1)
            raw.scatter_add_(0, flat, self._count(flat))
        self._raw_dirty = True

    def _count(self, flat: torch.Tensor) -> torch.Tensor:
        if self._ones.numel() < flat.numel():
            self._ones = torch.ones(flat.numel(), dtype=torch.int64, device=flat.device)
        return self._ones[: flat.numel()]

    def _plan(self, held: list[tuple[int, torch.Tensor]]):
        if len({(tuple(ids.shape), ids.dtype) for _, ids in held}) != 1:
            return None
        if held[0][1].dim() != 2 or held[0][1].dtype not in (torch.int32, torch.int64):
            return None
        base = torch.stack([self._raw_base[i] for i, _ in held]).view(-1, 1, 1)
        return tuple(ids for _, ids in held), base

    def fold(self) -> None:
        """The samples held by slot added to `dec_load` by logical expert,
        under the placement they were taken under: before the load is read
        and before a layer's placement changes."""
        if not self._raw_dirty:
            return
        self.dec_load.add_(torch.gather(self.dec_raw[:, 1:], 1, self.fold_index))
        self.dec_raw.zero_()
        self._raw_dirty = False

    def release_graphs(self) -> int:
        dropped = len(self._captures)
        self._captures.clear()
        self._plans.clear()
        return dropped

    # -- statistics ------------------------------------------------------
    def counters(self) -> list[int]:
        return [self.rows, self.steps, self.window_rows, self.window_steps]

    def take_reduced(self, dec: torch.Tensor, counters: list[int], world: int) -> None:
        rows, steps, window_rows, window_steps = counters
        self.reduced = (
            dec.view_as(self.dec_load),
            rows_per_step(rows, steps, world),
            steps // world,
            rows_per_step(window_rows, window_steps, world),
            window_steps // world,
        )

    # -- commit ----------------------------------------------------------
    def stage_commit(self, index: int, new_p2l: torch.Tensor):
        """Before layer ``index``'s weights move to ``new_p2l``: its gate
        rows in the new order, as copies -- everything of the gate's move
        that can fail, with nothing written."""
        tensors = gate_row_tensors(self.gates[index])
        staged = stage_rows(tensors, self.p2l[index], new_p2l)
        return None if staged is None else (tensors, staged)

    def apply_commit(self, index: int, staged, new_p2l: torch.Tensor) -> bool:
        """Layer ``index``'s weights are ``new_p2l``'s: write its staged gate
        rows, its map and its fold index. True when anything moved."""
        if staged is None:
            return False
        self.fold()  # the samples held by slot are the old placement's
        for tensor, rows in zip(*staged):
            tensor.copy_(rows)
        new = new_p2l.to(self.p2l.device)
        self.p2l[index].copy_(new)
        self.fold_index[index, new.to(torch.int64)] = torch.arange(
            self.num_experts, dtype=torch.int64, device=self.p2l.device
        )
        return True
