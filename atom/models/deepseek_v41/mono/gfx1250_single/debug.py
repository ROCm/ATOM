# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Diagnostics of the single-launch step, all off unless their switch is set (none runs on a served step by default).

- ``ATOM_MONO_SINGLE_TIMING=1`` (``ATOM_MONO_SINGLE_TIMING_EVERY``, default 64): ``StepTimer``, where a served step's
  time goes (host preparation, GPU gaps, the kernel and its stamped spans), logged every N steps.
- ``ATOM_MONO_SINGLE_SHADOW=1`` (eager; ``ATOM_MONO_SINGLE_SHADOW_NL`` / ``_AFTER`` / ``_NOENGRAM``): the production
  forward serves and the launch runs beside it on the same inputs; ``RunnerDebug.shadow_*`` log the kernel's head input,
  selection, draft context, taps and drafts against the production's.
- ``ATOM_MONO_SINGLE_CHECK_ROWS=<layer>``: that layer's table row against the checkpoint, at construction.
- ``ATOM_MONO_SINGLE_PROWDUMP=<dir>`` (``_PROWDUMP_N``, ``_PROWDUMP_QLOG``): per served step, the verification's
  inputs / outputs and the non-finite rows of the head inputs, one ``.npz`` a rank (``RunnerDebug.dump_step``).
"""

import logging
import os
import time

import numpy as np
import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.deepseek_v41.mono.gfx1250_single import buffers
from atom.models.deepseek_v41.mono.gfx1250_single.config import NDR, NL, ROWS_PER_REQ
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def model_dir():
    """The checkpoint directory (the engine's ``--model``), for ``_check_rows``."""
    from atom.config import get_current_atom_config

    return get_current_atom_config().model


class StepTimer:
    """ATOM_MONO_SINGLE_TIMING=1: where a served step's time goes. Events on the compute stream at the step's
    entry (``a``), just before the kernel (``k``) and after it (``z``); host wall times at the same points. Logged every
    ``every`` steps (GPU spans resolved once the events complete, no sync on the hot path).
    """

    def __init__(self, rank, every=64):
        self.time, self.rank, self.every = time, rank, every
        self.pend, self.acc, self.n, self.last_z, self.last_wall = [], {}, 0, None, None
        self.pool = (
            []
        )  # pinned host copies of the kernel's stamps (st [256][32] int64, 100 MHz)

    # the kernel's stamp slots (max over CTAs): 0 layer 0 start, 12 layer 1 start, 25 target end / head, 19 sampling,
    # 20 verify done (then the draft context / metadata), 26 draft layers start, 22 draft head done, 24 Markov done
    SPANS = (
        ("k_layer0", 0, 12),
        ("k_target_per_layer", 12, 25),
        ("k_head", 25, 19),
        ("k_sample", 19, 20),
        ("k_verify", 20, 21),
        ("k_ctx_meta", 21, 26),
        ("k_draft3_dhead", 26, 22),
        ("k_markov", 22, 24),
        ("k_all", 0, 24),
    )

    def copy_st(self, rec, st):
        buf = (
            self.pool.pop()
            if self.pool
            else torch.empty(256 * 32, dtype=torch.int64, pin_memory=True)
        )
        buf.copy_(st.view(torch.int64)[: 256 * 32], non_blocking=True)
        rec["st"] = buf

    def mark(self, name, rec):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        rec[name] = ev
        rec["w_" + name] = self.time.perf_counter()

    def step(self, rec, rows):
        rec["prev_z"], rec["prev_wall"], rec["rows"] = self.last_z, self.last_wall, rows
        self.last_z, self.last_wall = rec["z"], rec["w_a"]
        self.pend.append(rec)
        while self.pend and self.pend[0]["z"].query():
            r = self.pend.pop(0)
            d = {
                "gpu_prep": r["a"].elapsed_time(r["k"]),
                "kernel": r["k"].elapsed_time(r["z"]),
                "host_prep": 1e3 * (r["w_k"] - r["w_a"]),
                "host_launch": 1e3 * (r["w_z"] - r["w_k"]),
            }
            prev = r["w_a"]
            for nm, tt in r.get("hm", []):
                d["h_" + nm] = 1e3 * (tt - prev)
                prev = tt
            if "pre" in r:
                d["h_runstep_pre"] = 1e3 * r["pre"][0]
                d["h_engram_stage"] = 1e3 * r["pre"][1]
            if r["prev_z"] is not None:
                d["gpu_gap"] = r["prev_z"].elapsed_time(r["a"])
                d["host_period"] = 1e3 * (r["w_a"] - r["prev_wall"])
            if "st" in r:
                m = r["st"].view(256, 32).max(0).values.double()
                for nm, a_, b_ in self.SPANS:
                    us = float(m[b_] - m[a_]) / 100.0
                    d[nm] = (
                        us / (39 if nm == "k_target_per_layer" else 1) / 1e3
                    )  # ms (per layer: ms too)
                self.pool.append(r["st"])
            for k_, v in d.items():
                self.acc[k_] = self.acc.get(k_, 0.0) + v
            self.acc["rows"] = self.acc.get("rows", 0) + r["rows"]
            self.n += 1
            if self.n % self.every == 0:
                logger.warning(
                    "V4.1 single-launch timing (rank %d, %d steps, ms/step): %s",
                    self.rank,
                    self.every,
                    {k_: round(v / self.every, 3) for k_, v in self.acc.items()},
                )
                self.acc = {}


class RunnerDebug:
    """The runner's diagnostic methods (``SingleStepRunner`` inherits them); see the module docstring."""

    def _check_rows(self, model, li) -> None:
        """Layer li's layer-table row (``layer_weights.layer_row``) against the checkpoint read here: the FP8 rank shards
        and scales byte for byte, the f32 / bf16 vectors exactly, the hc fn's hi + lo to its f32; logged, not raised.
        """
        import json

        from safetensors import safe_open

        from atom.models.deepseek_v41.mono.gfx1250_single.weights import (
            layer as layer_weights,
        )

        blocks = [m for n, m in model.named_modules() if n.endswith(f"layers.{li}")]
        nxt = [m for n, m in model.named_modules() if n.endswith(f"layers.{li + 1}")]
        row = layer_weights.layer_row(blocks[0], nxt[0])
        root = model_dir()
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]

        def ck(name):
            with safe_open(os.path.join(root, wm[name]), "pt") as f:
                return f.get_tensor(name)

        r, P = self.rank, f"layers.{li}."
        res = {}

        def same(k, got, want):
            got = got.cpu()
            want = want.cpu()
            res[k] = bool(
                got.shape == want.shape
                and torch.equal(
                    got.view(torch.uint8), want.contiguous().view(torch.uint8)
                )
            )

        def slab(t, rows=None, cols=None):
            if rows is not None:
                n = t.shape[0] // 4
                t = t[r * n : (r + 1) * n]
            if cols is not None:
                n = t.shape[1] // 4
                t = t[:, r * n : (r + 1) * n]
            return t

        wq = torch.cat([ck(P + "attn.wq_a.weight"), ck(P + "attn.wkv.weight")])
        wqs = (
            torch.cat([ck(P + "attn.wq_a.scale"), ck(P + "attn.wkv.scale")])
            if (P + "attn.wq_a.scale") in wm
            else None
        )
        same("wqkv", row["wqkv"], wq)
        if wqs is not None:
            same("wqkv_s", row["wqkv_s"], wqs)
        same("wqb", row["wqb"], slab(ck(P + "attn.wq_b.weight"), rows=1))
        same("woa", row["woa"], slab(ck(P + "attn.wo_a.weight"), rows=1))
        same("wob", row["wob"], slab(ck(P + "attn.wo_b.weight"), cols=1))
        for k, n in (("wqb_s", "attn.wq_b.scale"), ("woa_s", "attn.wo_a.scale")):
            if P + n in wm:
                same(k, row[k], slab(ck(P + n), rows=1))
        if P + "attn.wo_b.scale" in wm:
            same("wob_s", row["wob_s"], slab(ck(P + "attn.wo_b.scale"), cols=1))
        q = ck(P + "attn.q_norm.weight").float()
        same("qnw", row["qnw"][: q.numel()], q)
        res["qnw_pad_zero"] = bool((row["qnw"][q.numel() :] == 0).all())
        same("kvnw", row["kvnw"], ck(P + "attn.kv_norm.weight").float())
        same(
            "sink",
            row["sink"],
            slab(ck(P + "attn.attn_sink").view(-1, 1), rows=1).view(-1).float(),
        )
        same("hsc", row["hsc"], ck(P + "hc_ffn_scale").float())
        same("hbs", row["hbs"], ck(P + "hc_ffn_base").float())
        same("nw", row["nw"], ck(P + "ffn_norm.weight").to(torch.bfloat16))
        same("gate", row["gate"], ck(P + "ffn.gate.weight").to(torch.bfloat16))
        same("gbias", row["gbias"], ck(P + "ffn.gate.bias").float())
        fn = ck(P + "hc_ffn_fn").float().reshape(-1)
        hl = (row["fnh"].float() + row["fnl"].float()).cpu()[: fn.numel()]
        res["fn_hi_lo_maxrel"] = float(
            ((hl - fn).abs() / fn.abs().clamp_min(1e-30)).max()
        )
        Q = f"layers.{li + 1}."
        same("ahsc", row["ahsc"], ck(Q + "hc_attn_scale").float())
        same("ahbs", row["ahbs"], ck(Q + "hc_attn_base").float())
        same("anw", row["anw"], ck(Q + "attn_norm.weight").to(torch.bfloat16))
        logger.warning(
            "V4.1 single-launch step row check (layer %d, rank %d): %s",
            li,
            self.rank,
            json.dumps(res),
        )

    def _draft_window_rows(self, forward_context, rows):
        """(pool view [N][512] bf16, per draft layer the pool rows of this step's ``rows`` target rows): the draft
        context KV the kernel's ctx stage writes and ``DSpark.write_context_kv`` (swa_write) writes after the forward.
        """
        cache = forward_context.attn_metadata.cache
        pool = cache.pool.view(torch.bfloat16).view(-1, 512)
        kvw = self.tables.step["kvw"][:rows].long()
        lids = self._layer_ids()[NL : NL + NDR]
        out = []
        for lid in lids:
            start = int(cache.geometry.window(lid, cache.num_pages).ring_start)
            out.append(torch.where(kvw >= 0, start + kvw, torch.full_like(kvw, -1)))
        return pool, out

    def shadow_ctx_after_launch(self, forward_context, rows) -> None:
        """Keep the draft context rows the kernel just wrote (compared with the production's next step)."""
        torch.cuda.synchronize()
        pool, idx = self._draft_window_rows(forward_context, rows)
        self._ctx_kernel = [(i, pool[i.clamp_min(0)].clone(), i >= 0) for i in idx]
        self._ctx_pos = forward_context.attn_metadata.step.positions[:rows].tolist()

    def arm_native_draft_capture(self) -> None:
        """Wrap the production drafter's ``draft_hidden`` once: keep its anchors and its normed block hidden (the
        head's input) of the last call, for ``shadow_draft_hidden``."""
        model = self.drafter.model
        if getattr(model, "_single_capture", False):
            return
        orig = model.draft_hidden

        def wrapped(anchor_ids, anchors, *a, **k):
            out = orig(anchor_ids, anchors, *a, **k)
            self._native_draft = (
                anchor_ids.detach().clone(),
                anchors.detach().clone(),
                out[0].detach().float().clone(),
            )
            return out

        model.draft_hidden = wrapped
        model._single_capture = True

    def shadow_draft_after_launch(self, rows) -> None:
        """Keep the kernel's draft block of this launch: its anchors (token, position) and normed hidden (dh)."""
        torch.cuda.synchronize()
        req = rows // ROWS_PER_REQ
        anc = self.scratch["anc"].view(torch.int32)[:req].long().clone()
        pos = self.tables.step["rpos_d"][: req * 5].view(req, 5)[:, 0].clone() - 1
        dh = (
            self.scratch["dh"]
            .view(torch.bfloat16)
            .view(-1, 5120)[: req * 5]
            .float()
            .clone()
        )
        self._kernel_draft = (anc, pos, dh)

    def shadow_draft_hidden(self) -> None:
        """The kernel's draft hidden of the step before against the production's (same request: same anchor token and
        position), relL2 / cosine per block row k."""
        kd, nd = getattr(self, "_kernel_draft", None), getattr(
            self, "_native_draft", None
        )
        self._native_draft = None
        if kd is None or nd is None or self.rank != 0:
            return
        anc, pos, dh = kd
        n_ids, n_pos, n_h = nd
        n_ids, n_pos = n_ids.view(-1).long(), n_pos.view(-1).long()
        W = 5
        stats = [[] for _ in range(W)]
        matched = 0
        for q in range(anc.numel()):
            hit = torch.nonzero((n_ids == anc[q]) & (n_pos == pos[q])).flatten()
            if hit.numel() != 1:
                continue
            j = int(hit[0])
            matched += 1
            for k in range(W):
                g, r = dh[q * W + k], n_h[j * W + k]
                stats[k].append(
                    float(torch.nn.functional.cosine_similarity(g, r, dim=0))
                )
        if matched:
            logger.warning(
                "V4.1 single-launch shadow draft hidden (%d requests matched of %d): cos by block row %s",
                matched,
                anc.numel(),
                [round(sum(v) / len(v), 4) for v in stats],
            )
        else:
            logger.warning(
                "V4.1 single-launch shadow draft hidden: no request matched (kernel anchors %s pos %s; "
                "production %s pos %s)",
                anc.tolist()[:4],
                pos.tolist()[:4],
                n_ids.tolist()[:4],
                n_pos.tolist()[:4],
            )

    def shadow_ctx_compare(self, forward_context) -> None:
        """The production's draft context rows (written after its forward: compute_draft_kv) at the rows the kernel
        wrote in the step before, against the kernel's values; relL2 per draft layer."""
        prev = getattr(self, "_ctx_kernel", None)
        if not prev or self.rank != 0:
            return
        torch.cuda.synchronize()
        pool = forward_context.attn_metadata.cache.pool.view(torch.bfloat16).view(
            -1, 512
        )
        rel = []

        def r(a, b):
            return round(float((a - b).norm() / b.norm().clamp_min(1e-30)), 4)

        for i, got, ok in prev:
            if not ok.any():
                rel.append(None)
                continue
            ref = pool[i.clamp_min(0)].float()[ok]
            g = got.float()[ok]
            cos = torch.nn.functional.cosine_similarity(g, ref, dim=1)
            # the 448 latent columns and the 64 RoPE ones apart; and the kernel's row against the reference's
            # neighbours (a position / ring offset error shows as a better match one row over)
            rel.append(
                {
                    "all": r(g, ref),
                    "nope": r(g[:, :448], ref[:, :448]),
                    "rope": r(g[:, 448:], ref[:, 448:]),
                    "cos_min": round(float(cos.min()), 4),
                    "rms": (
                        round(float(g.pow(2).mean().sqrt()), 4),
                        round(float(ref.pow(2).mean().sqrt()), 4),
                    ),
                }
            )
        logger.warning(
            "V4.1 single-launch shadow draft ctx KV (rows of the step before): per draft layer %s",
            rel,
        )
        # a row-placement error: each kernel row's best-matching production row within +-8 rows of where it was written
        i, got, ok = prev[0]
        best = {}
        for k in torch.nonzero(ok).flatten().tolist():
            g = got[k].float()
            cs = []
            for d in range(-8, 9):
                ref = pool[int(i[k]) + d].float()
                cs.append(float(torch.nn.functional.cosine_similarity(g, ref, dim=0)))
            d = max(range(17), key=lambda j: cs[j]) - 8
            best[d] = best.get(d, 0) + 1
        cos0 = [
            (
                round(
                    float(
                        torch.nn.functional.cosine_similarity(
                            got[k].float(), pool[int(i[k])].float(), dim=0
                        )
                    ),
                    3,
                )
                if bool(ok[k])
                else None
            )
            for k in range(len(i))
        ]
        logger.warning(
            "V4.1 single-launch shadow draft ctx KV row cos (draft layer 1, by row: request-row k = row %% 6, "
            "position %s): %s",
            getattr(self, "_ctx_pos", None),
            cos0,
        )
        cache = forward_context.attn_metadata.cache
        geo = [
            (int(w.ring_start), int(w.slot_rows), int(w.ring_slots))
            for w in (
                cache.geometry.window(lid, cache.num_pages)
                for lid in self._layer_ids()[NL - 1 : NL + NDR]
            )
        ]
        logger.warning(
            "V4.1 single-launch shadow draft ctx KV: best-matching row offset (draft layer 1) %s; "
            "windows (ring_start, slot_rows, ring_slots) of the last target layer and the draft layers %s",
            dict(sorted(best.items())),
            geo,
        )

    def shadow_taps(self, rows) -> None:
        """The kernel's draft-context taps (aux8 / auxs: the inputs of target layers 37..39, MXFP8 with 32-wide E8M0
        groups) against the production drafter's captured taps (``Drafter._aux_buffers``, bf16) of this forward.
        """
        bufs = getattr(self.drafter, "_aux_buffers", None)
        if not bufs or self.rank != 0:
            return
        torch.cuda.synchronize()
        H = 5120
        a8 = (
            self.scratch["aux8"]
            .view(torch.float8_e4m3fn)[: rows * 3 * H]
            .view(rows, 3 * H)
            .float()
        )
        code = (
            self.scratch["auxs"]
            .view(torch.uint8)[: rows * 3 * H // 32]
            .view(rows, 3 * H // 32)
            .float()
        )
        got = (a8.view(rows, -1, 32) * torch.exp2(code - 127)[..., None]).view(
            rows, 3, H
        )
        rel = []
        for j, b in enumerate(bufs[:3]):
            ref = b[:rows].float()
            rel.append(
                round(float((got[:, j] - ref).norm() / ref.norm().clamp_min(1e-30)), 4)
            )
        logger.warning(
            "V4.1 single-launch shadow taps (%d rows): relL2 per tap (layers 37 / 38 / 39 inputs) %s",
            rows,
            rel,
        )

    def shadow_drafts(self, input_ids, forward_context) -> None:
        """The kernel's drafts of the step before (``mids``, by request slot) against the production drafter's, which
        this step's verify rows carry (rows q·6 + 1..5), for the requests whose anchor token agrees (row q·6); per
        draft position, logged. Then this step's drafts are kept for the next call."""
        torch.cuda.synchronize()
        step = forward_context.attn_metadata.step
        req = input_ids.numel() // ROWS_PER_REQ
        ids = input_ids.view(req, ROWS_PER_REQ).cpu()
        slots = step.slots[:req].tolist()
        prev = getattr(self, "_prev_drafts", {})
        same = [0] * 5
        n = 0
        for q, sl in enumerate(slots):
            if sl in prev and prev[sl][0] == int(ids[q, 0]):
                n += 1
                for k in range(5):
                    same[k] += int(prev[sl][1][k] == int(ids[q, 1 + k]))
        if n and self.rank == 0:
            logger.warning(
                "V4.1 single-launch shadow drafts (%d requests with the same anchor): kernel = production by position %s",
                n,
                [round(c / n, 3) for c in same],
            )
        out = (
            self.scratch["outtok"]
            .view(torch.int32)[: 8 * ROWS_PER_REQ]
            .view(8, ROWS_PER_REQ)[:req]
            .cpu()
        )
        acc = self.scratch["verdict"].view(torch.int32)[:8][:req].cpu()
        mids = self.scratch["mids"].view(torch.int32)[: 8 * 5].view(8, 5)[:req].cpu()
        self._prev_drafts = {
            sl: (int(out[q, int(acc[q])]), mids[q].tolist())
            for q, sl in enumerate(slots)
        }

    def shadow_selection(self, rows, forward_context) -> None:
        """The kernel's top-512 of each selecting layer it ran (``tables`` isel: index columns; irow: pool rows) against
        the production scorer's ``step.selected[layer]``; logged as the mean / min per-row overlap of the two sets.
        """
        T_ = self.tables
        if not T_.idx:
            return
        torch.cuda.synchronize()
        step = forward_context.attn_metadata.step
        for Ly in sorted(T_.idx_w):
            if Ly >= self.shadow_layers:
                continue
            nat = step.selected.get(T_.specs[Ly].layer_id)
            if nat is None:
                continue
            nat = nat.reshape(-1, nat.shape[-1])[:rows].long().cpu()
            sel = T_.isel[Ly].view(-1, 512)[:rows].long().cpu()
            row = T_.irow[Ly].view(-1, 512)[:rows].long().cpu()
            out = {}
            for name, got in (("isel", sel), ("irow", row)):
                ov = []
                for q in range(rows):
                    a = set(nat[q][nat[q] >= 0].tolist())
                    b = set(got[q][sel[q] >= 0].tolist())
                    ov.append(len(a & b) / max(len(a), len(b), 1))
                out[name] = (sum(ov) / len(ov), min(ov))
            # the pool rows the attention reads: page(btab[request][col // irpp]) * cprow + imain + col % irpp
            L = self.K["step"].LKEYS
            irpp, imain, cprow = (
                T_.rows[Ly][L.index(k)] for k in ("irpp", "imain", "cprow")
            )
            bt = step.block_tables.long().cpu()
            req = step.batch_ids[:rows].long().cpu().clamp_min(0)
            col = sel.clamp_min(0)
            want = bt[req[:, None], col // irpp] * cprow + imain + col % irpp
            ok = (want == row) | (sel < 0)
            logger.info(
                "single-launch shadow layer %d irow vs host page rows: %d / %d match (irpp %d imain %d cprow %d); "
                "row0 sel %s irow %s want %s",
                T_.specs[Ly].layer_id,
                int(ok.sum()),
                ok.numel(),
                irpp,
                imain,
                cprow,
                sel[0, :6].tolist(),
                row[0, :6].tolist(),
                want[0, :6].tolist(),
            )
            n_nat = int((nat >= 0).sum(1).float().mean())
            n_got = int((sel >= 0).sum(1).float().mean())
            logger.info(
                "single-launch shadow layer %d selection (%d rows): native %d / kernel %d valid a row; overlap "
                "vs isel mean %.4f min %.4f, vs irow mean %.4f min %.4f",
                T_.specs[Ly].layer_id,
                rows,
                n_nat,
                n_got,
                *out["isel"],
                *out["irow"],
            )

    def shadow_compare(self, rows, hidden, ref=None, parts=None) -> None:
        """This step's head input (the kernel's xh: target layers + the final collapse and norm) against the production
        forward's ``model.norm(hidden)`` (or ``ref``: the shadow_layers check); logged.
        """
        torch.cuda.synchronize()
        got = self.scratch["xh"].view(torch.bfloat16).view(-1, 5120)[:rows].float()
        if ref is None:
            ref = self.base.norm(hidden.reshape(-1, 5120)[:rows])
        ref = ref.reshape(-1, 5120)[:rows].float()
        rel = float((got - ref).norm() / ref.norm().clamp_min(1e-30))
        for name in (
            "axn",
            "y1",
            "y2",
        ):  # the last layer's attention / FFN outputs (shadow_layers < NL)
            r = (parts or {}).get(name)
            if r is None:
                continue
            g = self.scratch[name].view(torch.bfloat16).view(-1, 5120)[:rows].float()
            r = r.reshape(-1, 5120)[:rows].float()
            cos = torch.nn.functional.cosine_similarity(g, r, dim=1)
            ratio = g.norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)
            # the per-row agreement, and the worst columns of row 0 (a layout / weight / scale error shows there)
            d0 = (g[0] - r[0]).abs()
            top = torch.topk(d0, 4)
            logger.warning(
                "V4.1 single-launch shadow (rank %d) %s rows cos [%.4f, %.4f] norm ratio [%.4f, %.4f]; row0 worst cols %s got %s ref %s",
                self.rank,
                name,
                float(cos.min()),
                float(cos.max()),
                float(ratio.min()),
                float(ratio.max()),
                top.indices.tolist(),
                [round(float(x), 4) for x in g[0][top.indices]],
                [round(float(x), 4) for x in r[0][top.indices]],
            )
            logger.warning(
                "V4.1 single-launch shadow (rank %d) %s relL2 %.3e, got rms %.4f ref rms %.4f",
                self.rank,
                name,
                float((g - r).norm() / r.norm().clamp_min(1e-30)),
                float(g.pow(2).mean().sqrt()),
                float(r.pow(2).mean().sqrt()),
            )
        logger.warning(
            "V4.1 single-launch shadow (rank %d, %d rows): xh relL2 %.3e, got rms %.4f ref rms %.4f",
            self.rank,
            rows,
            rel,
            float(got.pow(2).mean().sqrt()),
            float(ref.pow(2).mean().sqrt()),
        )
        if not parts and hasattr(
            self.base, "head"
        ):  # the head input: what the logits make of the difference
            lg = self.base.head.get_logits(got.to(torch.bfloat16)).float()
            lr = self.base.head.get_logits(ref.to(torch.bfloat16)).float()
            # the kernel's own head (head over its rank's vocab shard): its logits against the same columns of lg,
            # and its global argmax (the ranks' (max, index) gathered) against its sampled ids
            n = lg.shape[0]
            vr = self.scratch["logits"].numel() // 4 // buffers.T_MAX
            kl = self.scratch["logits"].view(torch.float32).view(buffers.T_MAX, vr)[:n]
            ref_sh = lg[:, self.rank * vr : (self.rank + 1) * vr]
            k_rel = float((kl - ref_sh).norm() / ref_sh.norm().clamp_min(1e-30))
            mx, ix = kl.max(-1)
            g = get_tp_group().device_group
            mxs = [torch.empty_like(mx) for _ in range(get_tp_group().world_size)]
            ixs = [torch.empty_like(ix) for _ in range(get_tp_group().world_size)]
            torch.distributed.all_gather(mxs, mx.contiguous(), group=g)
            torch.distributed.all_gather(ixs, ix.contiguous(), group=g)
            best = torch.stack(mxs).argmax(0)
            k_arg = torch.stack(ixs).gather(0, best[None])[0] + best * vr
            sids_all = self.scratch["sids"].view(torch.int32)[:n].long()
            logger.warning(
                "V4.1 single-launch shadow kernel head (rank %d): logits shard vs head(kernel xh) relL2 %.3e; "
                "sids = argmax(kernel logits) %.4f",
                self.rank,
                k_rel,
                float((sids_all == k_arg).float().mean()),
            )
            if self.rank == 0:
                t1 = (lg.argmax(-1) == lr.argmax(-1)).float().mean()
                k5 = torch.topk(lg, 5).indices
                r5 = torch.topk(lr, 5).indices
                o5 = sum(
                    len(set(a.tolist()) & set(b.tolist())) for a, b in zip(k5, r5)
                ) / (5 * lg.shape[0])
                # the kernel's own sampled ids (greedy rows: the argmax of its head over the same xh)
                sids = self.scratch["sids"].view(torch.int32)[: lg.shape[0]].long()
                gr = self.tables.step["sparam"].view(torch.int32)[: lg.shape[0], 3] != 0
                s_own = (
                    (sids == lg.argmax(-1))[gr].float().mean()
                    if gr.any()
                    else torch.tensor(float("nan"))
                )
                s_ref = (
                    (sids == lr.argmax(-1))[gr].float().mean()
                    if gr.any()
                    else torch.tensor(float("nan"))
                )
                logger.warning(
                    "V4.1 single-launch shadow sampled ids (%d greedy rows): = argmax(head(kernel xh)) %.4f, "
                    "= argmax(head(native)) %.4f",
                    int(gr.sum()),
                    float(s_own),
                    float(s_ref),
                )
                logger.warning(
                    "V4.1 single-launch shadow logits (%d of %d rows): top-1 agree %.4f, top-5 overlap %.4f, logits relL2 %.3e",
                    lg.shape[0],
                    rows,
                    float(t1),
                    o5,
                    float((lg - lr).norm() / lr.norm().clamp_min(1e-30)),
                )

    def dump_step(self, bs, out, verdict) -> None:
        """ATOM_MONO_SINGLE_PROWDUMP=<dir>: this step's verification (inputs, outputs, per-row p) and which head-input
        rows are non-finite, for the first ATOM_MONO_SINGLE_PROWDUMP_N (default 80) steps.
        """
        dd = envs.ATOM_MONO_SINGLE_PROWDUMP
        if (
            not dd
            or getattr(self, "_ndump_prow", 0) >= envs.ATOM_MONO_SINGLE_PROWDUMP_N
        ):
            return
        self._ndump_prow = getattr(self, "_ndump_prow", 0) + 1
        torch.cuda.synchronize()
        n = bs * ROWS_PER_REQ
        sc = self.scratch

        def nanrows(name, dtype, width):  # per row: any NaN / inf
            t_ = sc[name].view(dtype)
            t_ = t_[: (t_.numel() // width) * width].view(-1, width).float()
            return (~torch.isfinite(t_)).any(dim=1).cpu().numpy()

        qlog_n = envs.ATOM_MONO_SINGLE_PROWDUMP_QLOG
        np.savez(
            os.path.join(dd, f"prow_r{self.rank}_{self._ndump_prow}.npz"),
            xh_bad=nanrows("xh", torch.bfloat16, 5120)[
                :48
            ],  # the target head's input rows
            dh_bad=nanrows("dh", torch.bfloat16, 5120)[:48],  # the draft head's
            dlog_bad=nanrows("dlog", torch.float32, 32320)[:40],
            pos=getattr(self, "_dbg_pos", torch.zeros(1)).cpu().numpy(),
            prow=sc["prow"].view(torch.float32).reshape(-1, 8)[:n].cpu().numpy(),
            tids=self.tables.step["tids"].view(torch.int32)[:n].cpu().numpy(),
            verdict=verdict.cpu().numpy(),
            out=out.cpu().numpy(),
            dids=self.tables.step["dids"].view(torch.int32)[: bs * 5].cpu().numpy(),
            sparam=self.tables.step["sparam"]
            .view(torch.float32)
            .reshape(-1, 8)[:n]
            .cpu()
            .numpy(),
            qids=self.tables.state["qids"].cpu().numpy(),
            mids=sc["mids"].view(torch.int32)[:40].cpu().numpy(),
            qlog=(
                self.tables.state["qlog"].view(40, -1).cpu().numpy()
                if self._ndump_prow <= qlog_n
                else np.zeros(1)
            ),
        )


class ShadowModel:
    """``SingleStepModel``'s ATOM_MONO_SINGLE_SHADOW forward (eager): the production forward serves the step, the launch
    runs beside it on the same inputs and ``RunnerDebug.shadow_*`` compare the two."""

    def _layer_hook(self, k, ref):
        """The production state after layer k - 1, collapsed and normed with layer k's attn_norm (eps as the kernel's
        front, 1e-20): what a k-layer launch's xh holds."""
        from atom.models.deepseek_v41.mono.gfx1250_single.runner import _base_model

        base = _base_model(self.model)
        w = base.layers[k].attn_norm.weight

        def hook(module, args, output):
            x = output.collapse().float()
            ref["x"] = (
                x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-20) * w.float()
            ).to(torch.bfloat16)

        last = base.layers[k - 1]
        hooks = [last.register_forward_hook(hook)]
        # the last layer's attention output (the kernel's y1: after wo_b and the TP all-reduce) and its FFN output (y2)
        attn_fwd = last.attention_forward

        def attention_forward(*a, **kw):
            out = attn_fwd(*a, **kw)
            ref["y1"] = out
            return out

        last.attention_forward = attention_forward
        prep = last.prepare_attention

        def prepare_attention(*a, **kw):
            out = prep(*a, **kw)
            normed = out[0]
            ref["axn"] = (
                normed[0] if isinstance(normed, tuple) else normed
            )  # the attention input (bf16 rows)
            return out

        last.prepare_attention = prepare_attention

        def ffn_hook(module, args, output):
            ref["y2"] = output

        hooks.append(last.ffn.register_forward_hook(ffn_hook))

        class _Undo:
            def remove(self_):
                for h in hooks:
                    h.remove()
                del last.attention_forward
                del last.prepare_attention

        return _Undo()

    def _shadow(self, input_ids):
        """Bring-up (ATOM_MONO_SINGLE_SHADOW=1, eager): the runner of a decode verify forward the single launch shadows,
        or None."""
        if not envs.ATOM_MONO_SINGLE_SHADOW or torch.cuda.is_current_stream_capturing():
            return None
        fc = get_forward_context()
        rows = input_ids.numel()
        if (
            fc.spec_decode_metadata is None
            or getattr(fc.context, "is_dummy_run", False)
            or not self.step_ready(rows)
        ):
            return None
        return self._single.runner

    def _shadow_launch(self, runner, input_ids):
        """The single launch on the production forward's inputs (False: refused, logged; the production forward goes
        on). Before that forward its KV writes are rewritten by it; after it (ATOM_MONO_SINGLE_SHADOW_AFTER=1) it reads
        the rows that forward wrote this step -- the compressed KV and index rows a kernel without COMP cannot.
        """
        try:
            runner.launch(
                input_ids, get_forward_context(), getattr(self, "_step_batch", None)
            )
        except MonoUnsupported as why:
            logger.warning(
                "single-launch shadow step skipped (%d rows): %s",
                input_ids.numel(),
                why,
            )
            return False
        return True

    def shadow_forward(self, input_ids, positions):
        """ATOM_MONO_SINGLE_SHADOW: the production forward, with the launch beside it on the same inputs and the
        comparisons logged (``RunnerDebug.shadow_*``)."""
        shadow = self._shadow(input_ids)
        after = envs.ATOM_MONO_SINGLE_SHADOW_AFTER
        ctx_check = shadow is not None and shadow.shadow_layers == NL and not after
        if (
            ctx_check
        ):  # the production's draft context rows of the step before, before the launch rewrites them
            shadow.arm_native_draft_capture()
            shadow.shadow_draft_hidden()
            shadow.shadow_ctx_compare(get_forward_context())
        if (
            shadow is not None
            and not after
            and not self._shadow_launch(shadow, input_ids)
        ):
            shadow = None
        if shadow is not None and ctx_check:
            shadow.shadow_ctx_after_launch(get_forward_context(), input_ids.numel())
            shadow.shadow_draft_after_launch(input_ids.numel())
        ref, hook = {}, None
        if shadow is not None and shadow.shadow_layers < NL:
            hook = self._layer_hook(shadow.shadow_layers, ref)
        # (bring-up: the kernel has no Engram yet; ATOM_MONO_SINGLE_SHADOW_NOENGRAM=1 drops it from the production
        # forward too, so the layers after it compare -- that forward's tokens are then not the model's)
        no_engram = []
        if shadow is not None and envs.ATOM_MONO_SINGLE_SHADOW_NOENGRAM:
            for blk in shadow.tables.target_blocks:
                if getattr(blk, "engram", None) is not None:
                    blk.engram_forward = (
                        lambda residual, embeddings, image_mask: residual.clone()
                    )
                    no_engram.append(blk)
        try:
            out = self.model(input_ids, positions)
        finally:
            if hook is not None:
                hook.remove()
            for blk in no_engram:
                del blk.engram_forward
        if shadow is not None and after and not self._shadow_launch(shadow, input_ids):
            shadow = None
        if shadow is not None:
            shadow.shadow_compare(input_ids.numel(), out, ref.get("x"), ref)
            shadow.shadow_selection(input_ids.numel(), get_forward_context())
            if shadow.shadow_layers == NL:
                shadow.shadow_taps(input_ids.numel())
                shadow.shadow_drafts(input_ids, get_forward_context())
        return out
