# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The "swap" placement policy: a rebalance moves a few pairs of experts a
layer from the live placement, chosen to lower the busiest rank of each load
term (prefill rows, decode M-tiles, decode rows) separately. Plans are numpy
float64 from the group's integer loads, so every rank derives the same one.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import torch

from atom.model_ops.eplb_policy import (
    PlacementPlan,
    PlacementPolicy,
    PlanContext,
    physical_load_to_logical_load,
)

logger = logging.getLogger("atom")

# Weights of the objective's terms (prefill rows, decode M-tiles, decode rows),
# each a rank load whose mean over the ranks is 1, and of the smoothing term
# (the sum of squared rank loads) that lets the greedy leave a plateau of the
# max.
SWAP_WEIGHTS = (1.0, 1.0, 0.25)
SWAP_SMOOTH = 1.0
# Swaps a layer a rebalance, and the least objective gain a swap must bring.
SWAP_BUDGET = 16
SWAP_EPS = 0.005
# One pure-decode pass a rank in this many is counted.
DECODE_SAMPLE_EVERY = 8
# A decode step's expert M-tiles are of this many rows (the objective's model
# of the MoE kernel, not read from it).
DECODE_TILE_ROWS = 32


def rank_loads(values: np.ndarray, rk: np.ndarray, num_ranks: int) -> np.ndarray:
    """[experts, terms] summed over the experts of each rank -> [ranks, terms]
    (no BLAS: the same bits on every rank)."""
    out = np.zeros((num_ranks, values.shape[1]), dtype=np.float64)
    for m in range(values.shape[1]):
        out[:, m] = np.bincount(rk, weights=values[:, m], minlength=num_ranks)
    return out


def expected_tiles(
    dec: np.ndarray, rows: float, top_k: int, tile_rows: int
) -> np.ndarray:
    """E[ceil(X / tile_rows)] for X ~ Poisson(q_e * top_k * rows): an expert's
    expected M-tiles in a decode step of ``rows`` rows, q_e its share of the
    decode picks."""
    total = float(dec.sum())
    if total <= 0 or rows <= 0:
        return np.zeros(dec.shape, dtype=np.float64)
    lam = np.clip(dec / total, 0.0, 1.0) * float(rows) * top_k
    top = float(lam.max())
    n = math.ceil(top + 12.0 * math.sqrt(top + 1.0) + 12.0) + tile_rows
    j = np.arange(n, dtype=np.float64)
    log_fact = np.concatenate(([0.0], np.cumsum(np.log(j[1:]))))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lam = np.log(lam)[:, None]
        pmf = np.exp(j[None, :] * log_lam - lam[:, None] - log_fact[None, :])
    pmf[lam == 0] = 0.0
    pmf[lam == 0, 0] = 1.0
    # P(X > k * tile_rows), summed over k >= 0, is E[ceil(X / tile_rows)].
    survival = 1.0 - np.cumsum(pmf, axis=1)
    return np.clip(survival[:, ::tile_rows], 0.0, None).sum(axis=1)


def swap_terms(
    pf: np.ndarray,
    dec: np.ndarray,
    rows: float,
    top_k: int,
    num_ranks: int,
    tile_rows: int,
) -> np.ndarray:
    """[experts, 3] objective terms of one layer from its prefill and decode
    pick counts; a term with no data is zero (it then weighs nothing)."""
    pf = np.asarray(pf, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    values = np.zeros((pf.shape[0], 3), dtype=np.float64)
    if pf.sum() > 0:
        values[:, 0] = pf / pf.sum() * num_ranks
    if dec.sum() > 0:
        tiles = expected_tiles(dec, rows, top_k, tile_rows)
        if tiles.sum() > 0:
            values[:, 1] = tiles / tiles.sum() * num_ranks
        values[:, 2] = dec / dec.sum() * num_ranks
    return values


def greedy_swaps(
    values: np.ndarray,
    rk: np.ndarray,
    num_ranks: int,
    *,
    budget: int,
) -> list[tuple[int, int]]:
    """Greedy pairwise swaps, in place on ``rk`` (expert -> rank).

    Each round takes the heaviest rank of every term and tries every swap of
    one of its experts with one of another rank; the swap that lowers
    ``sum_m w_m * (max_r L[r, m] + SWAP_SMOOTH * sum_r L[r, m]^2)`` most is
    applied if it lowers it by more than `SWAP_EPS`. Stops at ``budget`` swaps.
    Ties go to the first candidate in (heavy rank, other rank, expert of the
    heavy rank, expert of the other rank) order.
    """
    w = np.asarray(SWAP_WEIGHTS, dtype=np.float64)
    num_experts, num_terms = values.shape
    per = num_experts // num_ranks  # swap_rebalance checks it divides
    other = num_experts - per
    by_term = np.ascontiguousarray(values.T)
    loads = rank_loads(values, rk, num_ranks)
    delta = np.empty((num_terms, per, other), dtype=np.float64)
    gain = np.empty_like(delta)
    work = np.empty_like(delta)
    score = np.empty((per, other), dtype=np.float64)
    part = np.empty_like(score)
    no_rest = np.full(num_terms, -np.inf)
    swaps: list[tuple[int, int]] = []
    while len(swaps) < budget:
        members = np.argsort(rk, kind="stable").reshape(num_ranks, per)
        top = loads.max(0)
        best_gain, best = SWAP_EPS, None
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
            np.subtract(top[:, None, None], gain, out=gain)
            # Less the change of the sum of squares: 2 d (L_l - L_h - d).
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
    prefill: torch.Tensor,
    decode: torch.Tensor,
    rows_per_step: float,
    old_p2l: torch.Tensor,
    *,
    num_gpus: int,
    top_k: int,
    tile_rows: int,
    budget: int = SWAP_BUDGET,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    """A placement that differs from ``old_p2l`` by at most ``budget`` swaps a
    layer, each rank's experts in ascending prefill load (the large-input MoE
    kernel walks a rank's experts in slot order) where the layer moved.

    ``prefill`` / ``decode``: [layers, logical] pick counts, ``rows_per_step``
    the group's decode rows a step. Returns ``(physical_to_logical [L, P],
    logical_to_physical [L, Lg, 1], logcnt [L, Lg], swaps a layer)``; a layer
    with no swap keeps its row of ``old_p2l``.
    """
    p2l = old_p2l.detach().to("cpu", torch.int64).numpy().copy()
    num_layers, num_physical = p2l.shape
    pf = prefill.detach().to("cpu").numpy().astype(np.float64)
    dec = decode.detach().to("cpu").numpy().astype(np.float64)
    if not pf.shape == dec.shape == (num_layers, num_physical):
        raise ValueError(
            "the swap policy places as many logical experts as there are slots: "
            f"loads {pf.shape} / {dec.shape}, placement {p2l.shape}"
        )
    if num_physical % num_gpus:
        raise ValueError(f"{num_physical} slots do not split over {num_gpus} ranks")
    per = num_physical // num_gpus
    slots = np.arange(num_physical, dtype=np.int64)
    l2p = np.empty_like(p2l)
    swaps: dict[int, int] = {}
    for layer in range(num_layers):
        if not np.array_equal(np.sort(p2l[layer]), slots):
            raise ValueError(
                f"layer {layer}: the swap policy needs a permutation placement"
            )
        l2p[layer, p2l[layer]] = slots
        values = swap_terms(
            pf[layer], dec[layer], rows_per_step, top_k, num_gpus, tile_rows
        )
        rk = l2p[layer] // per
        done = greedy_swaps(values, rk, num_gpus, budget=budget)
        swaps[layer] = len(done)
        if not done:
            continue
        for g in range(num_gpus):
            experts = np.flatnonzero(rk == g)
            order = np.lexsort((experts, pf[layer][experts]))
            p2l[layer, g * per : (g + 1) * per] = experts[order]
        l2p[layer, p2l[layer]] = slots
    dev = old_p2l.device
    return (
        torch.from_numpy(p2l).to(dev, torch.int32),
        torch.from_numpy(l2p).to(dev, torch.int32).unsqueeze(-1),
        torch.ones((num_layers, num_physical), dtype=torch.int32, device=dev),
        swaps,
    )


class SwapPolicy(PlacementPolicy):
    """`placement_policy: "swap"`: no redundant experts, one node, dp_size ==
    ep_size. A rebalance swaps up to `SWAP_BUDGET` expert pairs across ranks a
    layer from the live placement (`swap_rebalance`), on the group's logical
    pick counts since the start (prefill and sampled decode, never decayed);
    a layer that moved has every rank's slots reordered cold first. It places
    every layer, as it balances decode load too, and reads neither the load
    window (`load_window_size`) nor the balancedness gate. Without prefill
    load since the last plan a rebalance is skipped and not counted; the
    first plan without a gainful swap freezes the placement for good."""

    places_all_layers = True
    gated = False

    def __init__(self) -> None:
        # This rank's logical pick counts since the start, [layers, logical]
        # int64: every pure-prefill pass, and one pure-decode pass in
        # DECODE_SAMPLE_EVERY, folded through the live placement. They never
        # decay: a plan weighs all the load since the start.
        self.placement: torch.Tensor | None = None
        self.prefill: torch.Tensor | None = None
        self.decode: torch.Tensor | None = None
        self.decode_passes = 0
        # The decode rows of the sampled passes since the last plan, and
        # their number: the operating point is the window's.
        self.window_rows = 0
        self.window_samples = 0
        # The group's prefill picks the last plan saw, and the decode rows a
        # step it planned for.
        self.planned_prefill_picks = 0
        self.decode_rows = 0.0

    def check_deployment(
        self,
        *,
        num_logical: int,
        num_physical: int,
        ep_size: int,
        dp_size: int,
        multi_node: bool,
    ) -> None:
        if num_physical != num_logical:
            raise RuntimeError(
                "EPLB placement_policy 'swap' places one slot per logical "
                f"expert, but the MoE layout has {num_physical - num_logical} "
                "redundant slots (an EPLB-routed shared expert pads them); use "
                "'naive' or 'biased'"
            )
        # Validated on one node with DP attention over the EP group (DEP); the
        # plan is derived on every rank alike, which holds on one host.
        if dp_size != ep_size or multi_node:
            raise RuntimeError(
                "EPLB placement_policy 'swap' runs on one node with DP attention "
                f"over the EP group (dp_size == ep_size); got dp_size={dp_size}, "
                f"ep_size={ep_size}, multi_node={multi_node}"
            )

    def start(self, monitor: Any, live: Any) -> None:
        # The live placement, which migration updates in place.
        self.placement = live.physical_to_logical_map
        shape = (live.num_layers, live.num_logical_experts)
        device = self.placement.device
        self.prefill = torch.zeros(shape, dtype=torch.int64, device=device)
        self.decode = torch.zeros_like(self.prefill)
        monitor.set_pass_observer(self.observe_pass)

    def stop(self, monitor: Any) -> None:
        monitor.set_pass_observer(None)
        self.placement = self.prefill = self.decode = None

    def observe_pass(
        self,
        counts: torch.Tensor,
        *,
        is_pure_prefill: bool,
        is_pure_decode: bool,
        decode_rows: int,
    ) -> None:
        if is_pure_prefill:
            target = self.prefill
        elif is_pure_decode:
            self.decode_passes += 1
            if self.decode_passes % DECODE_SAMPLE_EVERY:
                return
            self.window_samples += 1
            self.window_rows += decode_rows
            target = self.decode
        else:
            return
        target += physical_load_to_logical_load(counts, self.placement, target.shape[1])

    def group_counts(self, monitor: Any) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """The group's prefill and decode counts and the window's decode rows
        and sampled passes, each summed over the ranks in one all-reduce."""
        window = torch.tensor(
            [self.window_rows, self.window_samples],
            dtype=torch.int64,
            device=self.prefill.device,
        )
        payload = torch.cat([self.prefill.reshape(-1), self.decode.reshape(-1), window])
        payload = monitor.all_reduce_counts(payload)
        size = self.prefill.numel()
        shape = self.prefill.shape
        return (
            payload[:size].view(shape),
            payload[size : 2 * size].view(shape),
            int(payload[-2].item()),
            int(payload[-1].item()),
        )

    def plan(self, monitor: Any, ctx: PlanContext) -> PlacementPlan | None:
        live = ctx.live
        prefill, decode, rows, samples = self.group_counts(monitor)
        picks = int(prefill.sum().item())
        if picks <= self.planned_prefill_picks:
            logger.info(
                "EPLB swap plan skipped: no prefill load since the last plan "
                "(%d picks); not counted",
                picks,
            )
            return None
        self.planned_prefill_picks = picks
        first_id = min(ctx.moe_layers)
        top_k = int(ctx.moe_layers[first_id].top_k)
        # The group's decode rows a step since the last plan: a rank's mean
        # rows a sampled pass, times the row sets the group serves. A window
        # without a sample keeps the last one's.
        if samples:
            self.decode_rows = rows / samples * ctx.dp_size
        self.window_rows = self.window_samples = 0
        t0 = time.perf_counter()
        p2l, l2p, logcnt, swaps = swap_rebalance(
            prefill,
            decode,
            self.decode_rows,
            live.physical_to_logical_map,
            num_gpus=live.ep_size,
            top_k=top_k,
            tile_rows=DECODE_TILE_ROWS,
        )
        rearrange_ms = (time.perf_counter() - t0) * 1000.0
        moved = sorted(i for i, n in swaps.items() if n)
        logger.info(
            "EPLB swap plan: %d swaps over %d layers (max %d a layer) on %d "
            "prefill picks, %d decode samples, %.0f decode rows a step",
            sum(swaps.values()),
            len(moved),
            max(swaps.values(), default=0),
            picks,
            samples,
            self.decode_rows,
        )
        return PlacementPlan(
            physical_to_logical=p2l,
            logical_to_physical=l2p,
            replica_count=logcnt,
            layer_ids=moved,
            stop_reason="the swap policy converged: no swap gains",
            rearrange_ms=rearrange_ms,
        )
