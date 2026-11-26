# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Debug aids of the single-launch step, off unless their switch is set (the same switches as the other mono backends).

- ``ATOM_MONO_CHECK=1`` (eager): the production forward serves every step and the launch runs beside it on the same
  inputs; ``RunnerDebug.check_*`` log the kernel's head input, logits, index selection, draft taps and drafts against
  the production's (``CheckModel.check_forward``).
- ``ATOM_MONO_TIMELINE=<prefix>``: where a served step's time goes -- host preparation, GPU gaps, the kernel and its
  layer / tail stamps -- averaged over 64 steps, logged and appended to ``<prefix>_r<rank>.jsonl`` (``StepTimer``).
"""

import json
import logging
import time

import torch

from atom.models.deepseek_v41.mono.gfx1250.config import NL, ROWS_PER_REQ
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

H = 5120  # hidden size: the rows of xh / aux8


def _rel(got, ref) -> float:
    return float((got - ref).norm() / ref.norm().clamp_min(1e-30))


class StepTimer:
    """Events on the compute stream at the step's entry (``a``), before the kernel (``k``) and after it (``z``), with
    the host times at the same points; resolved once the events complete (no sync on the hot path).
    """

    EVERY = 64
    # the kernel's step stamps (max over CTAs, 100 MHz): 0 layer 0 start, 12 layer 1 start, 25 target end / head,
    # 19 sampling, 20 verify done, 21 draft context / metadata, 26 draft layers start, 22 draft head done, 24 Markov done
    SPANS = (
        ("k_layer0", 0, 12),
        ("k_target_per_layer", 12, 25),
        ("k_head", 25, 19),
        ("k_sample", 19, 20),
        ("k_verify", 20, 21),
        ("k_ctx_meta", 21, 26),
        ("k_draft", 26, 22),
        ("k_markov", 22, 24),
        ("k_all", 0, 24),
    )

    def __init__(self, prefix, rank):
        self.path, self.rank = f"{prefix}_r{rank}.jsonl", rank
        self.pend, self.acc, self.n = [], {}, 0
        self.last_z = self.last_wall = None
        self.pool = []  # pinned host copies of the kernel's stamps (st [256][32] int64)

    def mark(self, name, rec):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        rec[name], rec["w_" + name] = ev, time.perf_counter()

    def copy_st(self, rec, st):
        buf = (
            self.pool.pop()
            if self.pool
            else torch.empty(256 * 32, dtype=torch.int64, pin_memory=True)
        )
        buf.copy_(st.view(torch.int64)[: 256 * 32], non_blocking=True)
        rec["st"] = buf

    def step(self, rec, rows):
        rec["prev_z"], rec["prev_wall"], rec["rows"] = self.last_z, self.last_wall, rows
        self.last_z, self.last_wall = rec["z"], rec["w_a"]
        self.pend.append(rec)
        while self.pend and self.pend[0]["z"].query():
            self._add(self.pend.pop(0))

    def _add(self, r):
        d = {
            "gpu_prep": r["a"].elapsed_time(r["k"]),
            "kernel": r["k"].elapsed_time(r["z"]),
            "host_prep": 1e3 * (r["w_k"] - r["w_a"]),
            "host_launch": 1e3 * (r["w_z"] - r["w_k"]),
            "rows": r["rows"],
        }
        prev = r["w_a"]
        for name, t in r.get(
            "hm", []
        ):  # ``StepInputs._hm``: marks inside the preparation
            d["h_" + name], prev = 1e3 * (t - prev), t
        if r["prev_z"] is not None:
            d["gpu_gap"] = r["prev_z"].elapsed_time(r["a"])
            d["host_period"] = 1e3 * (r["w_a"] - r["prev_wall"])
        m = r["st"].view(256, 32).max(0).values.double()
        for name, a, b in self.SPANS:  # ms (k_target_per_layer: a layer's)
            d[name] = (
                float(m[b] - m[a])
                / 1e5
                / ((NL - 1) if name == "k_target_per_layer" else 1)
            )
        self.pool.append(r["st"])
        for k, v in d.items():
            self.acc[k] = self.acc.get(k, 0.0) + v
        self.n += 1
        if self.n % self.EVERY == 0:
            avg = {k: round(v / self.EVERY, 3) for k, v in self.acc.items()}
            logger.warning(
                "V4.1 single-launch timeline (rank %d, %d steps, ms a step): %s",
                self.rank,
                self.EVERY,
                avg,
            )
            with open(self.path, "a") as f:
                f.write(json.dumps(avg) + "\n")
            self.acc = {}


class RunnerDebug:
    """``SingleStepRunner``'s ATOM_MONO_CHECK comparisons (it inherits them): the launch ran on this step's inputs,
    the production forward served it."""

    def check_head(self, rows, hidden) -> None:
        """The kernel's head input (xh: the target layers, the final collapse and norm) against the production's
        ``norm(hidden)``, and what the head makes of both."""
        torch.cuda.synchronize()
        got = self.scratch["xh"].view(torch.bfloat16).view(-1, H)[:rows].float()
        ref = self.base.norm(hidden.reshape(-1, H)[:rows]).float()
        lg = self.base.head.get_logits(got.to(torch.bfloat16)).float()
        lr = self.base.head.get_logits(ref.to(torch.bfloat16)).float()
        if self.rank != 0:
            return
        top5 = sum(
            len(set(a.tolist()) & set(b.tolist()))
            for a, b in zip(torch.topk(lg, 5).indices, torch.topk(lr, 5).indices)
        ) / (5 * rows)
        logger.warning(
            "V4.1 single-launch check (%d rows): xh relL2 %.3e; logits relL2 %.3e, top-1 agree %.4f, top-5 overlap %.4f",
            rows,
            _rel(got, ref),
            _rel(lg, lr),
            float((lg.argmax(-1) == lr.argmax(-1)).float().mean()),
            top5,
        )

    def check_selection(self, rows, forward_context) -> None:
        """Each selecting layer's top-512 (the kernel's index columns) against the production scorer's
        ``step.selected``: the mean / min per-row overlap of the two sets."""
        T_ = self.tables
        if not T_.idx or self.rank != 0:
            return
        torch.cuda.synchronize()
        step = forward_context.attn_metadata.step
        out = {}
        for Ly in sorted(T_.idx_w):
            nat = step.selected.get(T_.specs[Ly].layer_id)
            if nat is None:
                continue
            nat = nat.reshape(-1, nat.shape[-1])[:rows].long().cpu()
            sel = T_.isel[Ly].view(-1, 512)[:rows].long().cpu()
            ov = []
            for q in range(rows):
                a, b = set(nat[q][nat[q] >= 0].tolist()), set(
                    sel[q][sel[q] >= 0].tolist()
                )
                ov.append(len(a & b) / max(len(a), len(b), 1))
            out[T_.specs[Ly].layer_id] = (round(sum(ov) / rows, 4), round(min(ov), 4))
        logger.warning(
            "V4.1 single-launch check selection overlap (layer: mean, min): %s", out
        )

    def check_taps(self, rows) -> None:
        """The kernel's draft-context taps (aux8 / auxs: target layers 37..39 inputs, MXFP8 with 32-wide E8M0 groups)
        against the production drafter's (``Drafter._aux_buffers``, bf16) of this forward.
        """
        bufs = getattr(self.drafter, "_aux_buffers", None)
        if not bufs or self.rank != 0:
            return
        torch.cuda.synchronize()
        a8 = (
            self.scratch["aux8"]
            .view(torch.float8_e4m3fn)[: rows * 3 * H]
            .view(rows, -1, 32)
            .float()
        )
        code = (
            self.scratch["auxs"]
            .view(torch.uint8)[: rows * 3 * H // 32]
            .view(rows, -1)
            .float()
        )
        got = (a8 * torch.exp2(code - 127)[..., None]).view(rows, 3, H)
        rel = [
            round(_rel(got[:, j], b[:rows].float()), 4) for j, b in enumerate(bufs[:3])
        ]
        logger.warning(
            "V4.1 single-launch check taps (%d rows): relL2 per tap %s", rows, rel
        )

    def check_drafts(self, input_ids, forward_context) -> None:
        """The kernel's drafts of the step before (``mids``, by request slot) against the production drafter's, which
        this step's verify rows carry, for the requests whose anchor agrees; then this step's are kept.
        """
        torch.cuda.synchronize()
        req = input_ids.numel() // ROWS_PER_REQ
        ids = input_ids.view(req, ROWS_PER_REQ).cpu()
        slots = forward_context.attn_metadata.step.slots[:req].tolist()
        prev = getattr(self, "_prev_drafts", {})
        same, n = [0] * (ROWS_PER_REQ - 1), 0
        for q, sl in enumerate(slots):
            if sl in prev and prev[sl][0] == int(ids[q, 0]):
                n += 1
                for k in range(ROWS_PER_REQ - 1):
                    same[k] += int(prev[sl][1][k] == int(ids[q, 1 + k]))
        if n and self.rank == 0:
            logger.warning(
                "V4.1 single-launch check drafts (%d requests, same anchor): kernel = production by position %s",
                n,
                [round(c / n, 3) for c in same],
            )
        out = (
            self.scratch["outtok"]
            .view(torch.int32)[: 8 * ROWS_PER_REQ]
            .view(8, ROWS_PER_REQ)[:req]
            .cpu()
        )
        acc = self.scratch["verdict"].view(torch.int32)[:req].cpu()
        mids = self.scratch["mids"].view(torch.int32)[: 8 * 5].view(8, 5)[:req].cpu()
        self._prev_drafts = {
            sl: (int(out[q, int(acc[q])]), mids[q].tolist())
            for q, sl in enumerate(slots)
        }


class CheckModel:
    """``SingleStepModel``'s ATOM_MONO_CHECK forward."""

    def check_forward(self, input_ids, positions):
        """The production forward, with the launch on the same inputs before it (the production forward then rewrites
        the KV rows the launch wrote) and the comparisons logged after it."""
        fc = get_forward_context()
        rows = input_ids.numel()
        runner = None
        if (
            not torch.cuda.is_current_stream_capturing()
            and fc.spec_decode_metadata is not None
            and not getattr(fc.context, "is_dummy_run", False)
            and self.step_ready(rows)
        ):
            runner = self._single.runner
            try:
                runner.launch(input_ids, fc, self._step_batch)
            except MonoUnsupported as why:
                logger.warning(
                    "single-launch check step skipped (%d rows): %s", rows, why
                )
                runner = None
        out = self.model(input_ids, positions)
        if runner is not None:
            runner.check_head(rows, out)
            runner.check_selection(rows, fc)
            runner.check_taps(rows)
            runner.check_drafts(input_ids, fc)
        return out


def timeline(rank):
    """The StepTimer when ATOM_MONO_TIMELINE is set, else None."""
    prefix = envs.ATOM_MONO_TIMELINE
    return StepTimer(prefix, rank) if prefix else None
