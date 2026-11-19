# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A served step's inputs: the host checks (``precheck``), the step tables' host-to-device refills and the device
preparation the launch's graph captures (``_host_inputs`` / ``_device_inputs``), and each target row's sampling
parameters (``_sampling``). ``SingleStepRunner`` inherits them."""

import logging
import os

import numpy as np
import torch

from atom.models.deepseek_v41.mono.gfx1250 import buffers
from atom.models.deepseek_v41.mono.gfx1250.config import ROWS_PER_REQ
from atom.models.deepseek_v41.mono.gfx1250.step_plans import fill_step_plans
from atom.models.deepseek_v41.mono.gfx1250.step_prep import prep_step
from atom.models.deepseek_v41.mono.gfx1250.tables import BCOLS, ENG_W, ILW
from atom.models.deepseek_v41.mono.gfx1250.tables import T as T_MAX
from atom.models.deepseek_v41.mono.step_meta import write_step_meta
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")

# the prob drafter's temperature as a multiple of the request's (T_d = T * scale; single launch only: the production
# DSpark sampler drafts at the request's temperature). Lossless for any value: the drafts are sampled from q = softmax(z / T_d) and verified
# against that same q. Default 1 (T_d = T).
DRAFT_TSCALE = float(os.environ.get("ATOM_DSPARK_PROB_DRAFT_TSCALE", "1.0"))

# forced acceptance's marker in a request's bonus row (``kernels.sample.SIM_MAGIC``; that row verifies no draft, so its
# sp words 5..7 -- hasq, qsrc, seed_acc -- are free)
SIM_MAGIC = 0x53494D41


def sim_acc_words(rates, drafts=ROWS_PER_REQ - 1):
    """ATOM's forced acceptance (``SpeculativeConfig.synthetic_acceptance_rates``: per-position *unconditional* rates,
    ``rejection_sampler.acceptance_length_to_rates``) -> the kernel's ``(nf, fr)``: the conditional rate of draft p is 1
    for p < nf, fr at p = nf, 0 past it -- what ``rejection_synthetic_sample_kernel`` walks (``_get_synthetic_cond_rates``).
    None -> None (real verification). Any other curve shape is refused (the kernel holds two words, not a curve)."""
    if rates is None:
        return None
    rates = [float(r) for r in rates]
    if len(rates) != drafts:
        raise MonoUnsupported(f"forced acceptance over {len(rates)} positions (the step verifies {drafts} drafts)")
    cond, prev = [], 1.0
    for r in rates:
        cond.append(r / prev if prev > 0.0 else 0.0)
        prev = r
    nf = 0
    while nf < drafts and cond[nf] >= 1.0:
        nf += 1
    fr = cond[nf] if nf < drafts else 0.0
    if not 0.0 <= fr < 1.0 or any(c != 0.0 for c in cond[nf + 1 :]):
        raise MonoUnsupported(f"forced acceptance curve {rates} (only 1 .. 1, f, 0 .. 0)")
    return nf, np.float32(fr)


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
            if e is None or e.shape[-2] < rows or e.shape[-1] != ENG_W:
                raise MonoUnsupported(
                    f"engram layer {k}: embedding rows {None if e is None else tuple(e.shape)}"
                )
            engs[k] = e
            key += [e.data_ptr(), tuple(e.shape)]
        er = self._eng_rows
        if self.tables.engram and getattr(self, "_eng_staged", None) is None:
            # (englook) the entry hashes the staging's snapshot and looks the rows up itself
            raise MonoUnsupported("engram: no staged rows (the step kernel reads their snapshot)")
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
        if self.tables.idx and self.tables.comp:
            # the sampling parameters, the step metadata, the plans and the per-row copies / fills: one launch
            # (kernels/prep.py; the values of the four below)
            self._prep_one(input_ids, step, geom, window, rows, req)
            return
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
        if self.tables.idx and self.tables.comp:
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
        elif self.tables.idx:
            for r in (1, 2):
                S[f"ibound{r}"][:rows].copy_(step.visible[r][:rows])
        # every other per-row copy, the draft context's RoPE rows, the page tables (a fixed BCOLS columns: past the
        # step's reach never read) and the scratch the launch counts / maxes into: one launch (``step_prep``)
        prep_step(
            positions=step.positions,
            input_ids=input_ids.view(-1),
            batch_ids=step.batch_ids,
            # each request's state slot (the draft context's rows): not its batch position once requests have finished
            slots=step.slots,
            kvw=S["kvw"],
            rope=self._ctx_rope,
            block_tables=step.block_tables,
            out={
                "rpos": S["rpos"],
                "first_pos": self.first_positions,
                "tids": S["tids"],
                "dids": S["dids"],
                "dgeom_slots": S["dgeom"][8:],
                "ctx_kvw": S["ctx_kvw"],
                "ctx_cos": S["ctx_rcos"],
                "ctx_sin": S["ctx_rsin"],
                "ibat": S["ibat"] if self.tables.idx else S["tids"],
                "btab": S["btab"] if self.tables.idx else S["tids"],
                "slots": S["slots"] if self.tables.comp else S["tids"],
            },
            rows=rows,
            req=req,
            rows_per_request=ROWS_PER_REQ,
            bcols=BCOLS,
            idx=bool(self.tables.idx),
            comp=bool(self.tables.comp),
            cnt=self.scratch["cnt"].view(torch.int32),
            ss=self.scratch["ss"].view(torch.int32),
            # Markov's per-(position, request) global argmax is an atomic max: it must restart from INT64_MIN every
            # step (as the fixture does), else the drafts keep earlier steps' winners
            mbest=self.scratch["mbest"].view(torch.int64),
        )

    def _prep_takes_engram(self, er) -> bool:
        """Whether kernels/prep.py's launch runs the Engram stage's compress and snapshot (``EngramStaging._snapshot``):
        a verify step (cursor candidates staged) on the one-launch preparation, integer rows of 4 / 8 bytes."""
        st = er.staging.step
        if not (self.tables.idx and self.tables.comp) or st is None or st.candidates is None or st.dead is None:
            return False
        ts = (st.input_ids, st.live, st.dead, st.batch_ids, st.cu_seqlens, st.positions, st.cursor, st.history_index,
              st.candidates)
        return all(t.element_size() in (4, 8) and not t.dtype.is_floating_point for t in ts) and (
            er.staging.compressed.dtype == torch.int32 and er.staging.snapshot.dtype == torch.int64)

    def _prep_one(self, input_ids, step, geom, window, rows, req) -> None:
        """kernels/prep.py's launch (idx and comp tables): its build keyed by the integer widths of its buffers."""
        S = self.tables.step
        ids = input_ids.view(-1)
        bt = step.block_tables
        nbt, cc = bt.shape[0], min(bt.shape[1], BCOLS)
        bufs = dict(pos=step.positions, ids=ids, bids=step.batch_ids, slots=step.slots, bt=bt, rpos=S["rpos"],
                    firstp=self.first_positions, tids=S["tids"], dids=S["dids"], ctxkvw=S["ctx_kvw"], ibat=S["ibat"],
                    btab=S["btab"], slotsd=S["slots"], vis=S["ibound1"])
        er = getattr(self, "_eng_prep", None)
        eng, ex = None, (0,) * 10 + (0, 0, 0, 0)
        # the target's verify commits the accepted prefix's Engram cursor (sample.commit: ATOM's _commit_cursors) when
        # this launch stages the candidates of every request of the step, int64 rows (``committed_in_kernel``)
        ccur, ncom, cw = 0, 0, 0
        if er is not None:
            # the Engram stage (EngramStaging._snapshot: engram_compress + engram_snapshot) in this launch
            est, stg = er.staging.step, er.staging
            w_ = er.width
            tabs = stg.uva.hash_tables
            hist = est.cursor[:, 1:]
            bufs.update(eids=est.input_ids, elive=est.live, edead=est.dead, ecu=est.cu_seqlens, ehist=hist,
                        ehidx=est.history_index, ecand=est.candidates)
            if est.batch_ids.data_ptr() != step.batch_ids.data_ptr() or est.positions.data_ptr() != step.positions.data_ptr():
                raise MonoUnsupported("engram: the staging's batch ids / positions are not the step's")
            eng = (int(tabs.ngram), True, True)
            cand, cur = est.candidates, est.cursor
            if (cand.dtype == torch.int64 and cur.dtype == torch.int64 and cur.stride(1) == 1 and cand.stride(2) == 1
                    and cur.shape[1] <= self.K["sample"].COM_WMAX and cand.shape[2] == cur.shape[1]
                    and step.scheduled_bs == req):
                ccur, ncom, cw = cur.data_ptr(), req, int(cur.shape[1])
            ex = (tabs.lookup.data_ptr(), est.input_ids.data_ptr(), est.live.data_ptr(), est.dead.data_ptr(),
                  stg.compressed.data_ptr(), est.cu_seqlens.data_ptr(), hist.data_ptr(), est.history_index.data_ptr(),
                  stg.snapshot.data_ptr(), est.candidates.data_ptr(), hist.stride(0), w_,
                  est.candidates.stride(0), est.candidates.stride(1))
        widths = tuple(sorted((k, v.element_size()) for k, v in bufs.items()))
        for k, v in bufs.items():
            if v.element_size() not in (4, 8) or v.dtype.is_floating_point:
                raise MonoUnsupported(f"step preparation: {k} {v.dtype}")
        assert S["kvw"].element_size() == 4 and S["kmeta"].element_size() == 4 and S["dgeom"].element_size() == 4
        assert all(S[f"{p}{r}"].dtype == torch.int32 for p in ("cplan", "wplan") for r in (1, 2))
        assert S["ibound2"].dtype == S["ibound1"].dtype and self._sp_pin.is_pinned()
        # the device round's advance in this launch (``round.SingleStepRound.laid_out``: this step's positions are the
        # previous verdict's, the host laid them out as if no draft had been rejected)
        adv = bool(getattr(self, "_adv", False))
        self._com = ncom > 0
        launch = self._prep_builds.get((widths, eng, adv)) if hasattr(self, "_prep_builds") else None
        if launch is None:
            self._prep_builds = getattr(self, "_prep_builds", {})
            launch = self._prep_builds[(widths, eng, adv)] = self.K["prep"].build(
                widths, ROWS_PER_REQ, BCOLS, (1, 2), geom.speculative_tokens, T_MAX, eng, adv
            )
        cnt = self.scratch["cnt"].view(torch.int32)
        ss = self.scratch["ss"].view(torch.int32)
        mbest = self.scratch["mbest"].view(torch.int64)
        cos, sin = self._ctx_rope
        P = lambda x: x.data_ptr()
        launch(
            P(step.positions), P(ids), P(step.batch_ids), P(step.slots), P(cos), P(sin), P(bt), P(S["kvw"]),
            P(S["kmeta"]), P(S["ibound1"]), P(S["ibound2"]), P(S["cplan1"]), P(S["wplan1"]), P(S["cplan2"]),
            P(S["wplan2"]), P(self._sp_pin), P(S["sparam"]), P(S["rpos"]), P(self.first_positions), P(S["tids"]),
            P(S["dids"]), P(S["dgeom"]), P(S["ctx_kvw"]), P(S["ctx_rcos"]), P(S["ctx_rsin"]), P(S["ibat"]),
            P(S["btab"]), P(S["slots"]), P(cnt), P(ss), P(mbest),
            rows, req, nbt, cc, bt.stride(0), int(window.slot_rows), int(window.ring_slots), int(geom.window_size),
            cnt.numel(), ss.numel(), mbest.numel(), self._sp_pin.numel(), *ex,
            P(self.scratch["tcom"]), ccur, ncom, cw, P(self.scratch["verdict"]),
            torch.cuda.current_stream(),
        )

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
                (4, inv_t / np.float32(DRAFT_TSCALE)),  # the drafts' inverse temperature: 1 / (T * DRAFT_TSCALE)
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
        sim = getattr(self, "sim_acc", None)
        if sim is not None:
            # forced acceptance: each request's bonus row carries (SIM_MAGIC, nf, fr) -- it verifies no draft, so the
            # three words are its own sampling's unused ones (hasq, qsrc, seed_acc); rewritten each step (above)
            b = slice(ROWS_PER_REQ - 1, n, ROWS_PER_REQ)
            ints[b, 5] = SIM_MAGIC
            ints[b, 6] = sim[0]
            sp[b, 7] = sim[1]
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
