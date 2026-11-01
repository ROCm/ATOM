# SPDX-License-Identifier: MIT
"""Fit DSpark's confidence temperatures (STS) online, from serving traffic.

The verify budget ranks drafts by cumulative survival, the product of the
per-position confidences after Sequential Temperature Scaling
(`calibrate_confidence`). Raw confidence is far from calibrated (offline fits
land around T = [2.0, 1.5, 0.7, 1.0, 0.85] on V4.1), and T=1 makes the
scheduler mis-rank and mis-size verification. Without a calibration profile
the temperatures are fitted here instead.

Data, with no host synchronization: every few forwards the TP leader copies the
raw confidence of the drafts it just proposed (async D2H, tagged with the
forward). Two forwards later the deferred output reports how those drafts
fared -- per request, the drafts accepted, against the drafts the verifying
batch actually scheduled (its ragged length minus the anchor).
Position j is observed only if it was verified; there its label is
"accepted > j". Positions the scheduler chose not to verify are censored, not
negatives.

The fit minimizes the cumulative-survival ECE over the observed positions,
coordinate descent over a temperature grid, on the TP leader in a background
thread. Results reach every TP worker through one small CPU broadcast at a
forward count every worker shares, so all of them size verification with the
same temperatures. Temperatures only size verification, so this is lossless.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np
import torch

from atom.spec_decode.dspark_delayed import DelayedRows

logger = logging.getLogger("atom")


_GRID = np.array([0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
# ECE noise falls as 1/sqrt(n): a pull toward T=1 of that size keeps a thinly
# observed position from wandering to the grid edge on noise, and fades as the
# evidence grows (offline fits sit inside 0.7..2).
_SHRINK = 0.1
_BINS = 15


def _survival(logits, temperatures):
    return np.cumprod(1.0 / (1.0 + np.exp(-logits / temperatures)), axis=1)


def censored_ece(confidence, verified, accepted, temperatures, bins=_BINS):
    """Cumulative-survival ECE over the positions that were verified."""
    c = np.clip(np.asarray(confidence, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(c) - np.log1p(-c)
    return _censored_ece(
        logits, np.asarray(verified), np.asarray(accepted), temperatures, bins
    )


def _censored_ece(logits, verified, accepted, temperatures, bins=_BINS):
    predicted = _survival(logits, np.asarray(temperatures, dtype=np.float64))
    error, used = 0.0, 0
    for j in range(predicted.shape[1]):
        seen = verified > j
        if not seen.any():
            continue
        p, y = predicted[seen, j], (accepted[seen] > j).astype(np.float64)
        groups = np.minimum((p * bins).astype(np.int64), bins - 1)
        count = np.bincount(groups, minlength=bins)
        gap = np.abs(
            np.bincount(groups, weights=p, minlength=bins)
            - np.bincount(groups, weights=y, minlength=bins)
        )
        error += gap.sum() / count.sum()
        used += 1
    return error / max(used, 1)


def fit_temperatures(confidence, verified, accepted, start, *, min_rows=3000):
    """Coordinate descent over `_GRID`; positions with fewer than ``min_rows``
    verified observations keep their starting temperature."""
    c = np.clip(np.asarray(confidence, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(c) - np.log1p(-c)
    verified, accepted = np.asarray(verified), np.asarray(accepted)
    temperatures = np.asarray(start, dtype=np.float64).copy()
    seen = [int((verified > j).sum()) for j in range(c.shape[1])]
    fit = [j for j in range(c.shape[1]) if seen[j] >= min_rows]
    for _ in range(3):
        for j in fit:
            scores = []
            for value in _GRID:
                trial = temperatures.copy()
                trial[j] = value
                scores.append(
                    (
                        _censored_ece(logits, verified, accepted, trial)
                        + _SHRINK * abs(np.log(value)) / np.sqrt(seen[j]),
                        abs(value - 1),
                        value,
                    )
                )
            temperatures[j] = min(scores)[2]
    return temperatures, fit


class OnlineSTS:
    def __init__(
        self,
        scheduler,
        gamma,
        *,
        comm=None,
        sample_interval=4,
        refit_interval=256,
        window=20000,
        min_rows=3000,
    ):
        self.scheduler = scheduler
        self.gamma = gamma
        # TP only. DP ranks fit independently: a DP rank's forward count is not
        # the group's (idle ranks run dummies on paths that report no verdict),
        # so a DP collective keyed on it would deadlock. Each rank's
        # temperatures only size its own requests; the joint plan reads each
        # rank's own survival, so per-rank temperatures need no agreement.
        if comm is None:
            from atom.spec_decode.dspark_comm import DSparkComm

            comm = DSparkComm()
        self.comm = comm
        self.sample_interval = sample_interval
        self.refit_interval = refit_interval
        self.min_rows = min_rows
        self.rows = deque(maxlen=window)
        self.forward = 0
        self.copies = DelayedRows(
            lambda: scheduler.runner.tokenID_processor.async_copy_stream, gamma
        )
        self._confidence = None
        self._thread = None
        self._result = None
        self.temperatures = np.ones(gamma)
        self.version = 0

    # -- collection (TP leader) -------------------------------------------
    def capture(self, confidence):
        if self.comm.tp_leader and self.forward % self.sample_interval == 0:
            self._confidence = confidence.detach()

    def record(self, req_ids):
        source, self._confidence = self._confidence, None
        if source is None or source.shape[1] != self.gamma:
            return
        if source.shape[0] != len(req_ids):
            return
        self.copies.push(self.forward, req_ids, source)

    def observe(self, req_ids=None, accepted=None, verified=None):
        """Once per forward, before this forward's drafts are recorded.

        ``req_ids``/``accepted``/``verified``: per request of the previous
        forward's decode batch -- the drafts it accepted and the drafts it
        actually verified (its scheduled length minus the anchor, after any
        ragged shrink). The sampler's own reject count is against the full
        reservation and would label unverified drafts as rejections.
        """
        self.forward += 1
        # A capture belongs to the forward that drafted it: one its forward
        # never recorded (warmup, a skipped draft) must not be paired with a
        # later batch's request ids.
        self._confidence = None
        if self.comm.tp_leader and req_ids is not None:
            self._pair(req_ids, accepted, verified)
        if self.forward % self.refit_interval == 0:
            self._publish()

    def _pair(self, req_ids, accepted, verified):
        drafted = self.forward - 2
        taken = self.copies.take(drafted, wait=False, release=True)
        if taken is not None:
            ids, confidence = taken
            where = {r: i for i, r in enumerate(ids)}
            for k, r in enumerate(req_ids):
                i = where.get(r)
                if i is None or k >= len(accepted) or k >= len(verified):
                    continue
                a, v = int(accepted[k]), min(int(verified[k]), self.gamma)
                if 0 < v and 0 <= a <= v:
                    self.rows.append((confidence[i].copy(), v, a))
        self.copies.reap(before=drafted)

    # -- fitting and publication ------------------------------------------
    def _fit(self, rows, start):
        c = np.stack([r[0] for r in rows])
        v = np.array([r[1] for r in rows])
        a = np.array([r[2] for r in rows])
        self._result = fit_temperatures(c, v, a, start, min_rows=self.min_rows)[0]

    def _publish(self):
        # [log T * w, w] per position; w = verified observations behind it.
        values = torch.zeros(2 * self.gamma, dtype=torch.float64)
        if self.comm.tp_leader:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
                if self._result is not None:
                    self._fitted = self._result
            fitted, weight = getattr(self, "_fitted", None), self._weights()
            if fitted is not None:
                values[: self.gamma] = torch.from_numpy(np.log(fitted) * weight)
                values[self.gamma :] = torch.from_numpy(weight)
            if self._thread is None and len(self.rows) >= self.min_rows:
                self._thread = threading.Thread(
                    target=self._fit,
                    args=(list(self.rows), self.temperatures.copy()),
                    daemon=True,
                )
                self._thread.start()
        self.comm.broadcast_tp(values)
        log_sum, weight = values[: self.gamma].numpy(), values[self.gamma :].numpy()
        new = np.where(weight > 0, np.exp(log_sum / np.maximum(weight, 1)), 1.0)
        new = np.round(new, 3)
        if not np.array_equal(new, self.temperatures):
            self.temperatures = new.copy()
            self.version += 1
            self.scheduler.sts_temperatures = torch.tensor(
                new, dtype=torch.float32, device=self.scheduler.runner.device
            )
            if self.comm.tp_leader:
                logger.info(
                    "DSpark online STS v%d from %d verified blocks: %s",
                    self.version,
                    len(self.rows),
                    [round(float(t), 2) for t in new],
                )

    def _weights(self):
        """Verified observations per position, zero below the fit threshold."""
        if not self.rows:
            return np.zeros(self.gamma)
        verified = np.array([r[1] for r in self.rows])
        counts = np.array([(verified > j).sum() for j in range(self.gamma)], float)
        return np.where(counts >= self.min_rows, counts, 0.0)
