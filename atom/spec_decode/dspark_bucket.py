# SPDX-License-Identifier: MIT
"""Executable-graph costs for the delayed DSpark verification scheduler.

This policy must only feed VerifyScheduler's fixed N-2 adoption path. A global
search using confidence from the tokens being verified is not lossless.
"""

import torch

from atom.spec_decode.dspark_scheduler import calibrate_confidence


def graph_row_indices(captured, q_buckets, max_rows):
    """CPU lookup [required per-request width][logical rows] -> graph rows.

    Match ForwardMode's lower bound AND q-divisibility rule. Uncovered shapes
    are -1; they must never borrow a smaller graph's cost by clamping.
    """
    sizes = sorted(set(captured))
    qs = sorted(set(q_buckets))
    if not qs or qs[0] < 1:
        raise ValueError("Need positive query-width buckets")
    result = [[-1] * (max_rows + 1)]
    for required in range(1, qs[-1] + 1):
        q = next(q for q in qs if q >= required)
        eligible = [g for g in sizes if g % q == 0 and g <= max_rows]
        index = 0
        row = []
        for n in range(max_rows + 1):
            while index < len(eligible) and eligible[index] < n:
                index += 1
            row.append(eligible[index] if index < len(eligible) else -1)
        result.append(row)
    return result


def schedule_bucket_prefix_lengths_tensor(
    confidence, sps_table, graph_rows, q_buckets, *, sts_temperatures=None
):
    """Global optimum for the profiled graph-row objective, with no host read.

    Search each per-request q cap. Within a cap, top-m cumulative survival
    probabilities maximize expected accepted tokens for each total width.
    Searching all caps matters: a different allocation of the same m tokens
    can reduce q and reach a smaller divisible graph. Stable sorting preserves
    prefix order for equal survival probabilities.
    """
    if confidence.ndim != 2:
        raise ValueError("confidence must be [R, gamma]")
    batch, gamma = confidence.shape
    if not batch or not gamma:
        return torch.zeros(batch, dtype=torch.long, device=confidence.device)
    if graph_rows.shape[0] <= gamma + 1 or graph_rows.shape[1] <= batch * (gamma + 1):
        raise ValueError("Graph lookup does not cover this candidate batch")
    if graph_rows.shape[1] > sps_table.numel():
        raise ValueError("Profile does not cover graph lookup")
    a = calibrate_confidence(confidence, sts_temperatures).cumprod(dim=1)
    flat = a.reshape(-1)
    n = flat.numel()
    pos = torch.arange(n, device=flat.device) % gamma
    allowed = torch.stack([pos < q - 1 for q in q_buckets])
    scores = flat.unsqueeze(0).expand(len(q_buckets), n).masked_fill(~allowed, -1.0)
    values, order = scores.sort(dim=1, descending=True, stable=True)
    positions = order % gamma
    required = torch.cat(
        [torch.ones_like(positions[:, :1]), positions.cummax(dim=1).values + 2],
        dim=1,
    )
    m = torch.arange(n + 1, device=flat.device)
    tau = batch + torch.cat(
        [values.new_zeros((len(q_buckets), 1)), values.cumsum(1)], dim=1
    )
    rows = graph_rows[required, batch + m]
    valid = torch.stack([m <= batch * (q - 1) for q in q_buckets]) & (rows >= 0)
    theta = (tau * sps_table[rows.clamp(min=0)]).masked_fill(~valid, -torch.inf)
    best = theta.reshape(-1).argmax()
    cap, count = best // (n + 1), best % (n + 1)
    request = order.gather(0, cap.expand(1, n)).squeeze(0) // gamma
    admitted = torch.arange(n, device=flat.device) < count
    ell = torch.zeros(batch, dtype=torch.long, device=flat.device)
    ell.scatter_add_(0, request, admitted.long())
    # During startup/no coverage, retain the complete verification layout.
    return torch.where(valid.any(), ell, torch.full_like(ell, gamma))


class BucketPolicy:
    """`bucket_global`: verification lengths priced by the graph each layout
    actually replays, and the guard that keeps the reserved width when the
    floors of today's bonus tokens erase the predicted saving."""

    def __init__(self, ragged_graph_sizes: str):
        self.ragged_graph_sizes = ragged_graph_sizes
        self._lookup = None
        self._signature = None

    def ell(self, confidence, sps_table, sts_temperatures):
        """Per-request verify lengths [bs]; the full width while no decode
        graph is recorded or a graph is being captured."""
        from atom.spec_decode.dspark_scheduler import resolve_q_buckets
        from atom.utils.region_graph import TokenGraph

        bs, width = confidence.shape
        sizes = TokenGraph.captured_sizes("decode")
        if not sizes or torch.cuda.is_current_stream_capturing():
            return torch.full((bs,), width, dtype=torch.long, device=confidence.device)
        qs = resolve_q_buckets(self.ragged_graph_sizes, width + 1)
        signature = (sizes, tuple(qs), sps_table.numel())
        if signature != self._signature:
            self._lookup = torch.tensor(
                graph_row_indices(sizes, qs, sps_table.numel() - 1),
                dtype=torch.long,
                device="cpu",
            ).to(device=confidence.device)
            self._signature = signature
        return schedule_bucket_prefix_lengths_tensor(
            confidence.detach(),
            sps_table,
            self._lookup,
            qs,
            sts_temperatures=sts_temperatures,
        )


def keeps_reserved_width(ragged_rows: int, q: int, full_rows: int, full_q: int) -> bool:
    """Whether a single-rank step keeps its reserved tokens instead of the
    ragged layout.

    N-2 decisions meet today's bonus floors here; if those floors erase the
    predicted graph saving, the reserved tokens stay. This only restores the
    existing rectangular path; no current draft confidence or speculative token
    value is consulted. Keep them only when shrinking cannot help: it would
    leave a recorded graph for an eager step, or land on a graph no smaller. An
    uncovered full width (e.g. past a DP graph cap) does not veto a covered
    ragged layout.
    """
    from atom.utils.region_graph import TokenGraph

    short = TokenGraph.bucket(ragged_rows, "decode", multiple_of=q)
    full = TokenGraph.bucket(full_rows, "decode", multiple_of=full_q)
    return full is not None and (short is None or short >= full)
