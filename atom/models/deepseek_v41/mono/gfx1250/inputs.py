# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A served step's inputs: the host checks (``precheck``), the step tables' host-to-device refills and the device
preparation the launch's graph captures (``_host_inputs`` / ``_device_inputs``), and each target row's sampling
parameters (``_sampling``). ``SingleStepRunner`` inherits them."""

import logging

import numpy as np
import torch

from atom.models.deepseek_v41.mono.gfx1250 import buffers
from atom.models.deepseek_v41.mono.gfx1250.config import ROWS_PER_REQ
from atom.models.deepseek_v41.mono.gfx1250.step_plans import fill_step_plans
from atom.models.deepseek_v41.mono.gfx1250.tables import BCOLS, ILW
from atom.models.deepseek_v41.mono.gfx1250.tables import T as T_MAX
from atom.models.deepseek_v41.mono.step_meta import write_step_meta
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")


class StepInputs:
    """``SingleStepRunner``'s per-step input preparation (it owns ``tables``, ``scratch``, ``drafter``)."""

    def precheck(self, batch, forward_context) -> bool:
        """Can the kernel serve this step's request shapes (no device work, nothing staged)? The same limits
        ``_sampling`` / ``_host_inputs`` refuse with MonoUnsupported, checked before the step commits to the kernel (an
        exception after the Engram rows are staged left the worker without a fallback: a 4.4k-token prompt with 4096
        outputs past the index bound). Identical on every rank (the batch and its spans are).
        """
        why = None
        if (
            batch is not None
            and (np.asarray(batch.top_ks[: batch.total_seqs_num]) != -1).any()
        ):
            why = "top_k"
        md = getattr(forward_context, "attn_metadata", None)
        if why is None and self.tables.idx and md is not None:
            step, geom = md.step, md.cache.geometry
            end = max((sp.end for sp in step.requests), default=1)
            bt = step.block_tables
            if any(end // r > ILW for r in (1, 2)):
                why = f"index bound {end} > {ILW}"
            elif -(-end // geom.block_size) > BCOLS or bt.shape[0] > 8:
                why = f"block table {bt.shape[0]} x {-(-end // geom.block_size)}"
        if why is None:
            self._precheck_why = None
        else:
            if why.split()[0] != getattr(
                self, "_precheck_why", None
            ):  # once a kind (the bound values change a step)
                self._precheck_why = why.split()[0]
                logger.warning(
                    "V4.1 single-launch step: not served (%s), the production path runs (rank %d)",
                    why,
                    self.rank,
                )
            return False
        return True

    def _host_inputs(self, input_ids, forward_context):
        """The step's host-side checks and the rare host-to-device refills (geometry), before the device preparation;
        returns (the Engram layers' embedding rows, the key of every address / shape ``_device_inputs`` captures).
        """
        md = forward_context.attn_metadata
        cache, step = md.cache, md.step
        rows = input_ids.numel()
        S = self.tables.step
        geom = cache.geometry
        self.tables.set_geometry(cache, self._layer_ids())
        # the window geometry key, recomputed only when the cache's geometry / page count changes (geom.window() every
        # step was most of this function's host time)
        wk = (id(geom), cache.num_pages, cache.num_slots)
        if wk != getattr(self, "_wkey", None):
            window = geom.window(0, cache.num_pages)
            self._wkey, self._gk = wk, (
                geom.window_size,
                window.slot_rows,
                window.ring_slots,
                cache.num_slots,
            )
        gk = self._gk
        if gk != getattr(
            self, "_dgeom_key", None
        ):  # (static for a cache: an H2D copy only when it changes)
            self._dgeom_key = gk
            S["dgeom"][:4].copy_(torch.tensor(gk, dtype=torch.int32))
        rope = self.drafter.model.rope
        if (
            getattr(self, "_ctx_rope", None) is None
        ):  # f32 copies of the draft's RoPE rows, once: a gather a step
            self._ctx_rope = (
                rope.cos_cache.view(-1, 32).float(),
                rope.sin_cache.view(-1, 32).float(),
            )
        key = [rows, input_ids.data_ptr(), gk, geom.block_size, self.tables._geom_key]
        if self.tables.idx:
            # the step's furthest position, from the host spans (no device read): a row at position p sees (p + 1) // r
            # compressed rows, p < span.end
            end = max((sp.end for sp in step.requests), default=1)
            for r in (1, 2):
                if end // r > ILW:
                    raise MonoUnsupported(f"index bound {end // r} > {ILW}")
            bt = step.block_tables
            if -(-end // geom.block_size) > BCOLS or bt.shape[0] > 8:
                raise MonoUnsupported(
                    f"block table: {bt.shape[0]} requests x {-(-end // geom.block_size)} columns > 8 x {BCOLS}"
                )
            key += [
                t.data_ptr()
                for t in (
                    step.positions,
                    step.slots,
                    step.batch_ids,
                    step.visible[1],
                    step.visible[2],
                )
            ]
            key += [bt.data_ptr(), tuple(bt.shape), bt.stride(0)]
        # the Engram layers' embedding rows (staged on Engram's side stream: ``get`` makes this stream wait for them)
        emb = getattr(md, "engram_embeddings", None)
        engs = {}
        for k, w in self.tables.engram.items():
            e = (
                (dict.get(emb, k) if self._eng_rows is not None else emb.get(k))
                if emb is not None
                else None
            )
            if e is None or e.shape[-2] < rows or e.shape[-1] != w["engemb"].shape[1]:
                raise MonoUnsupported(
                    f"engram layer {k}: embedding rows {None if e is None else tuple(e.shape)}"
                )
            engs[k] = e
            key += [e.data_ptr(), tuple(e.shape)]
        er = self._eng_rows
        if (
            er is not None
        ):  # the staged Engram inputs the graph captures (fixed addresses: EngramStep)
            st = er.staging.step
            key += [er.width, id(er.staging)] + [
                None if t_ is None else t_.data_ptr()
                for t_ in (
                    st.input_ids,
                    st.live,
                    st.dead,
                    st.batch_ids,
                    st.cu_seqlens,
                    st.positions,
                    st.cursor,
                    st.history_index,
                    st.candidates,
                )
            ]
        return engs, tuple(key)

    def _device_inputs(self, input_ids, forward_context, engs):
        """Every device operation of the step's preparation: from fixed addresses only (captured in the width's
        graph, ``launch``), sizes fixed by the key of ``_host_inputs``."""
        md = forward_context.attn_metadata
        cache, step = md.cache, md.step
        rows = input_ids.numel()
        req = rows // 6
        S = self.tables.step
        geom = cache.geometry
        window = geom.window(0, cache.num_pages)
        S["sparam"].copy_(self._sp_pin, non_blocking=True)
        write_step_meta(
            step,
            geom,
            window,
            S["kvw"][:rows],
            S["kmeta"][: rows * 4].view(rows, 4),
            {},
            (0, rows),
        )
        S["rpos"][:rows].copy_(step.positions[:rows])
        self.first_positions[:req].copy_(step.positions[:rows:ROWS_PER_REQ])
        S["tids"][:rows].copy_(input_ids.view(-1))
        S["dids"][: req * 5].copy_(input_ids.view(req, 6)[:, 1:].reshape(-1))
        # each request's state slot (the draft context's rows): not its batch position once requests have finished
        S["dgeom"][8 : 8 + req].copy_(step.slots[:req])
        S["ctx_kvw"][:rows].copy_(S["kvw"][:rows])
        pos = step.positions[:rows]
        torch.index_select(
            self._ctx_rope[0], 0, pos, out=S["ctx_rcos"][: rows * 32].view(rows, 32)
        )
        torch.index_select(
            self._ctx_rope[1], 0, pos, out=S["ctx_rsin"][: rows * 32].view(rows, 32)
        )
        if self.tables.idx:
            S["ibat"][:rows].copy_(step.batch_ids[:rows])
            if self.tables.comp:
                # the visible counts and the compress / write plans from the step's positions, on the device (the
                # host's arithmetic, ``step_plans``): they hold on a step whose positions the device advanced
                fill_step_plans(
                    step.positions[:rows],
                    ROWS_PER_REQ,
                    (1, 2),
                    geom.speculative_tokens,
                    {r: S[f"ibound{r}"] for r in (1, 2)},
                    {r: S[f"cplan{r}"] for r in (1, 2)},
                    {r: S[f"wplan{r}"] for r in (1, 2)},
                    T_MAX,
                )
            else:
                for r in (1, 2):
                    S[f"ibound{r}"][:rows].copy_(step.visible[r][:rows])
            # the requests' page tables: a fixed BCOLS columns (past the step's reach: never read)
            bt = step.block_tables
            n = bt.shape[0]
            cc = min(bt.shape[1], BCOLS)
            S["btab"][: n * BCOLS].view(n, BCOLS)[:, :cc].copy_(bt[:, :cc])
            if self.tables.comp:
                S["slots"][:n].copy_(step.slots[:n])
        for k, w in self.tables.engram.items():
            e = engs[k]
            w["engemb"][:rows].copy_(e.reshape(-1, e.shape[-1])[:rows])
        self.scratch["cnt"].zero_()
        self.scratch["ss"].zero_()
        # Markov's per-(position, request) global argmax is an atomic max: it must restart from INT64_MIN every step
        # (as the fixture does), else the drafts keep earlier steps' winners
        self.scratch["mbest"].view(torch.int64).fill_(-(1 << 63))

    def _sampling(self, batch, req) -> None:
        """Each target row's sampling from its request (``ScheduledBatch`` temperatures / top_ps / top_ks), as the
        recipe's prob drafting + block verification (``dspark_prob_sample``, PROB_DRAFT / BLOCK_VERIFY): greedy
        (temperature 0) -> argmax drafts and the argmax match; else the drafts are sampled from q (logits / temperature) and verified against p (the top-p set of logits / temperature) by the
        probability ratio. q is the kernel's own (``qlog``) when the request's drafts came from the last kernel launch
        (``hasq``, its row there ``qsrc``), else the one-hot rule. Noise seeds mix a step counter and the batch row.
        Top-k is not in the kernel: refused."""
        if batch is None:
            raise MonoUnsupported(
                "single-launch step: no batch for the sampling parameters"
            )
        temps = np.asarray(batch.temperatures[:req], dtype=np.float32)
        top_ps = np.asarray(batch.top_ps[:req], dtype=np.float32)
        top_ks = np.asarray(batch.top_ks[:req])
        if (top_ks != -1).any():
            raise MonoUnsupported(f"single-launch step: top_k {top_ks.tolist()}")
        rids = list(batch.req_ids[:req])
        self._step_no = getattr(self, "_step_no", 0) + 1
        n = req * ROWS_PER_REQ
        # the batch's static columns (inverse temperatures, top-p, greedy) cached by request set: in steady decode only
        # the seeds and the draft-source columns change a step (rebuilding all with numpy was ~0.06 ms of host time
        # after the verdict)
        skey = (req, tuple(rids), temps.tobytes(), top_ps.tobytes())
        sc = getattr(self, "_sp_static", None)
        if sc is None or sc[0] != skey:
            greedy = temps == 0
            sp = np.zeros((buffers.T_MAX, 8), dtype=np.float32)
            sp[:, 0], sp[:, 1], sp[:, 4] = 1.0, 1.0, 1.0
            sp.view(np.int32)[:, 3] = 1  # rows past the batch: greedy
            inv_t = np.where(greedy, 1.0, 1.0 / np.maximum(temps, 1e-6)).astype(
                np.float32
            )
            for c, v in (
                (0, inv_t),
                (1, np.where(greedy, 1.0, top_ps)),
                (4, inv_t),  # the drafts' inverse temperature: the request's
            ):
                sp[:n, c] = np.repeat(
                    v, ROWS_PER_REQ
                )  # a request's 6 verify rows share its sampling
            sp.view(np.int32)[:n, 3] = np.repeat(greedy.astype(np.int32), ROWS_PER_REQ)
            rr = np.repeat(np.arange(req, dtype=np.int64), ROWS_PER_REQ)
            sc = self._sp_static = (
                skey,
                sp,
                rr * 0x85EBCA77,
                rr * 0x27D4EB2F + 0x165667B1,
            )
        sp = sc[1]
        ints = sp.view(np.int32)
        prev = getattr(self, "_qrow", {})
        ints[:n, 5] = np.repeat(
            np.array([r in prev for r in rids], dtype=np.int32), ROWS_PER_REQ
        )
        ints[:n, 6] = np.repeat(
            np.array([prev.get(r, 0) for r in rids], dtype=np.int32), ROWS_PER_REQ
        )
        ints[:n, 2] = ((self._step_no * 0x9E3779B1 + sc[2]) & 0x7FFFFFFF).astype(
            np.int32
        )
        ints[:n, 7] = ((self._step_no * 0xC2B2AE35 + sc[3]) & 0x7FFFFFFF).astype(
            np.int32
        )
        # staged in a fixed pinned buffer; ``_device_inputs`` copies it (non-blocking: a pageable copy waited for the
        # stream to drain -- the previous step's draft launch -- stalling this step's host preparation). Rewritten a
        # step later only after that step's first launch, which follows the copy, has completed (``verify_ready``).
        if getattr(self, "_sp_pin", None) is None:
            self._sp_pin = torch.empty(sp.shape, dtype=torch.float32, pin_memory=True)
        self._sp_pin.numpy()[:] = sp
        # this launch's drafts (and their q in qlog / qids) by request: committed once the launch is issued
        self._qrow_next = {r: i for i, r in enumerate(rids)}

    def forget_drafts(self, req_ids) -> None:
        """The production drafter drew these requests' next drafts: their verification has no kernel q."""
        kept = getattr(self, "_qrow", {})
        for r in req_ids:
            kept.pop(r, None)
