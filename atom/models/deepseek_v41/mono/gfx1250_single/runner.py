# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The single-launch step's runner: per-width builds, the bound pointer tables, one launch a step.

What ``ModelRunner`` asks of it on a served step (``ModelRunner._single_step``):
- ``run_step(running_bs, forward_context)``: the step metadata (``write_step_meta`` / ``write_draft_step_meta``) and the
  one launch, on the current stream (graph-capturable: every pointer it binds stays put);
- ``verify_outputs(bs)``: the rejection sampler's ``(output_token_ids [bs, k + 1] int32, -1 past the emitted tokens;
  verdict [3, bs] int32: accepted drafts, rejected drafts, anchor row)``;
- ``draft_ids(bs)``: the next step's draft tokens, as ``Drafter.propose`` returns them.

Construction (collective over the TP group, ``LazyRunner``): the kernel modules (``kernels/``: the whole step is
``kernels.layer4l_step_fly``), the scratch (``buffers.scratch_bytes``, once at the widest step),
the peer receive buffers (``SymmetricPeerBuffer``), the routed-expert planes (``planes.PLANES``). The weight tables
and the step metadata (``buffers``: WEIGHT / STEP) are not bound yet, so construction still refuses at its end and
``LazyRunner`` keeps the production path."""

import importlib
import logging
import os
import sys
import time

import numpy as np
import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.models.deepseek_v41.mono.gfx1250_single import buffers
from atom.models.deepseek_v41.mono.gfx1250_single.config import (
    NDR,
    NL,
    ROWS_PER_REQ,
    TP,
)
from atom.models.deepseek_v41.mono.gfx1250_single.debug import RunnerDebug, StepTimer
from atom.models.deepseek_v41.mono.gfx1250_single.weights import planes
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.peer_memory import SymmetricPeerBuffer
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _base_model(model):
    for m in model.modules():
        if hasattr(m, "embed") and hasattr(m, "head") and hasattr(m, "norm") and hasattr(m, "layers"):
            return m
    raise MonoUnsupported("no V4.1 base model")


KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
# the receive regions: 4 TP regions x 4 sources x 40960 16-byte words, the sampling and Markov exchange after them
# (step_tail_ct3 SOFF / SOFF2), 2 MB-rounded, doubled for the step epoch
# the recipe's draft temperature scale (dspark_prob_sample.DRAFT_TSCALE): q = softmax(z * invT / TSCALE)
DRAFT_TSCALE = float(os.environ.get("ATOM_DSPARK_PROB_DRAFT_TSCALE", "1.0"))
RECV_BYTES = 2 * (
    (16 * 40960 * 16 + 4 * 4 * 48 * 256 * 16 + (2 << 20) - 1) // (2 << 20) * (2 << 20)
)


def import_kernels() -> dict:
    """The kernel modules (this package's kernels/: the whole step is layer4l_step_fly)."""
    if KERNELS not in sys.path:
        sys.path.insert(0, KERNELS)
    names = (
        "hx",
        "attn_stages",
        "ffn_stages",
        "moe_stages",
        "ctx_stages",
        "head_fly",
        "sample4r_fly",
        "sample4p_fly",
        "markov_prob_fly",
        "step_tail_ct3",
        "layer4l_step_fly",
    )
    return {n: importlib.import_module(n) for n in names}


class SingleStepRunner(RunnerDebug):
    def __init__(self, model, drafter, atom_config) -> None:
        if get_tensor_model_parallel_world_size() != TP:
            raise MonoUnsupported(
                f"TP {get_tensor_model_parallel_world_size()} (the step kernel is TP{TP})"
            )
        self.rank = get_tensor_model_parallel_rank()
        self.device = torch.device("cuda", torch.cuda.current_device())
        cache = get_forward_context().attn_metadata.cache
        window = cache.geometry.window(0, cache.num_pages)
        try:
            self.K = import_kernels()
        except (
            Exception
        ) as err:  # a kernel source that does not import: the deployment cannot run it
            raise MonoUnsupported(
                f"step kernel import: {type(err).__name__}: {err}"
            ) from err
        # the kernel's ring slots (window + block width) and pool row width (bf16 elements) are compile-time constants
        AS = self.K["attn_stages"]
        if int(window.ring_slots) != AS.KG_RING:
            raise MonoUnsupported(f"window ring {window.ring_slots} slots (the step kernel's: {AS.KG_RING})")
        miss = buffers.check_catalog(self.K)
        if miss:
            raise MonoUnsupported(f"pointer catalog out of date: {miss[:4]}")
        self.scratch = {
            n: torch.zeros(b, dtype=torch.uint8, device=self.device)
            for n, b in buffers.scratch_bytes(self.K, NL + NDR).items()
        }
        # fine-grained, not uncached: the step's few-word peer pushes (Markov, sampling) waited ~80-100 us each for an
        # uncached store to become visible (CT12 11-08: the fixture's fine-grained receive buffers did not)
        self.peer = SymmetricPeerBuffer(
            RECV_BYTES, rank=self.rank, npes=TP, group=get_tp_group().cpu_group, fine_grained=True
        )
        self.recv = [int(a) for a in self.peer.addresses.tolist()]
        # the routed experts a layer, as the loader packed them (``planes.capture``): target "layers.{i}.ffn", the
        # draft's "mtp.{j}.ffn"
        want = [f"layers.{i}.ffn" for i in range(NL)] + [
            f"mtp.{j}.ffn" for j in range(NDR)
        ]
        missing = [p for p in want if not any(k.endswith(p) for k in planes.PLANES)]
        if missing:
            raise MonoUnsupported(
                f"no expert plane for {missing[:3]} (planes.capture not run at load)"
            )
        self.planes = {
            p: next(v for k, v in planes.PLANES.items() if k.endswith(p)) for p in want
        }
        logger.info(
            "V4.1 single-launch step: %.1f MB scratch, %.1f MB receive, %d planes (rank %d)",
            sum(t.numel() for t in self.scratch.values()) / 2**20,
            RECV_BYTES / 2**20,
            len(self.planes),
            self.rank,
        )
        if envs.ATOM_MONO_SINGLE_CHECK_ROWS is not None:  # (debug) a layer's table row against the checkpoint
            self._check_rows(model, envs.ATOM_MONO_SINGLE_CHECK_ROWS)
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import StepTables

        self.tables = StepTables(
            self.K, model, drafter, self.scratch, self.recv, self.rank, get_tp_group().device_group
        )
        logger.info(
            "V4.1 single-launch step tables: fp %d, tp %d, ctx %d pointers, %d layer rows (rank %d)",
            self.tables.fptab.numel(),
            self.tables.tp.numel(),
            self.tables.ctxtab.numel(),
            len(self.tables.rows),
            self.rank,
        )
        self.model, self.drafter = model, drafter
        self.base = _base_model(model)
        self.builds = {}
        # the step's device preparation and first launch are captured as one graph a width (``launch``): the ~20 small
        # preparation operations cost ~0.3 ms of host time and GPU gaps a step otherwise
        self._graphs, self._seen = {}, {}
        self._fast = {}  # rows -> the width's launchers (``_fast_launchers``)
        # the step is two launches (``prepare``): target .. verification, then the draft; the verdict's copies and the
        # output D2H wait for the first only (``verify_outputs`` / ``verify_ready``)
        self.split = True   # (ModelRunner reads it: the step's output ids are the kernel's, ready at ``verify_ready``)
        self._ev_verify = torch.cuda.Event()
        self.verify_ready = torch.cuda.Event()
        self._verify_stream = torch.cuda.Stream()
        self._timer = StepTimer(self.rank, envs.ATOM_MONO_SINGLE_TIMING_EVERY) if envs.ATOM_MONO_SINGLE_TIMING else None
        self._eng_rows = None
        # (debug) ATOM_MONO_SINGLE_SHADOW: the production forward serves and the launch runs beside it (``debug``)
        self.shadow = envs.ATOM_MONO_SINGLE_SHADOW
        # its target layers (< NL: no step tail / draft; its xh is the normed collapse after layer shadow_layers - 1
        # with layer shadow_layers' attn_norm, compared to the production state there)
        self.shadow_layers = envs.ATOM_MONO_SINGLE_SHADOW_NL
        if not self.shadow and self.shadow_layers != NL:
            raise MonoUnsupported("single-launch step: serving runs all the layers (ATOM_MONO_SINGLE_SHADOW_NL unset)")
        # every width built now (each a compile or a disk-cache load, then a TP barrier: ``prepare``), not on its first
        # step: a width first met mid-serve stalled every request for its JIT (~18 s cold)
        t0 = time.perf_counter()
        for rows in range(ROWS_PER_REQ, buffers.T_MAX + 1, ROWS_PER_REQ):
            self.prepare(rows)
        logger.info("V4.1 single-launch step: %d widths built in %.1f s (rank %d)", len(self.builds),
                    time.perf_counter() - t0, self.rank)

    def prepare(self, rows: int) -> bool:
        if rows % 6 or not 0 < rows <= buffers.T_MAX:
            return False
        if rows not in self.builds:
            M = self.K["layer4l_step_fly"]
            nl = self.shadow_layers
            kw = dict(NL=nl, ndr=NDR if nl == NL else 0, trd=rows // 6 * 5)
            # two launches: target .. verification, then the draft (the host reads the verdict between)
            launch = tuple(M.build(rows, self.rank, part=pt, **kw)[0] for pt in ("target", "draft"))
            # compile (JIT or disk cache) before the first launch of this width, then wait for every rank: the step
            # kernel's cross-rank polls give up after 2 s, and ranks compiling a width seconds apart made that first
            # launch time out and compute on stale / zero peer data (KV / index / compressor state stays wrong after)
            from atom.mono.runtime.compile import compile_only

            with compile_only():
                for ln in launch:
                    ln(*self._launch_args(), stream=torch.cuda.current_stream())
            torch.distributed.barrier(group=get_tp_group().cpu_group)
            self.builds[rows] = launch
        return True

    def _fast_launchers(self, rows):
        """The width's launchers: each @flyc.jit launcher's first call goes through flyc.compile (which launches it)
        and the CompiledFunction it returns is called after that -- no per-call signature binding / cache key / globals
        check (~0.15 ms of host time a launch)."""
        fast = self._fast.get(rows)
        if fast is None:
            import flydsl.compiler as flyc

            def first(i):
                def call(*a):
                    fn = flyc.compile(self.builds[rows][i], *a)
                    if fn is None:
                        raise MonoUnsupported(f"single-launch step: no compiled launcher for {rows} rows")
                    self._fast[rows][i] = fn
                return call

            fast = self._fast[rows] = [first(i) for i in range(len(self.builds[rows]))]
        return fast

    def _launch_args(self) -> tuple:
        T_ = self.tables
        return (*self.recv, self.scratch["cnt"].data_ptr(), self.scratch["st"].data_ptr(),
                float(self.base.layers[0].attn.softmax_scale), T_.fptab.data_ptr(), T_.tp.data_ptr())

    def _check_timeouts(self) -> None:
        """The kernel's poll-timeout counter (hx.note_timeout: scratch cyc int32 [0] count, [1] last site), read a
        launch late without a sync; any timeout means that step computed on stale peer data."""
        prev = getattr(self, "_to_pend", None)
        if prev is not None and prev[1].query():
            n, site = prev[0].tolist()
            if n != getattr(self, "_to_seen", 0):
                logger.error("V4.1 single-launch step: %d cross-rank poll timeouts so far (rank %d, last site %d)",
                             n, self.rank, site)
                self._to_seen = n
            prev = None
        if prev is None:
            buf = getattr(self, "_to_buf", None)
            if buf is None:
                buf = self._to_buf = torch.empty(2, dtype=torch.int32, pin_memory=True)
            buf.copy_(self.scratch["cyc"].view(torch.int32)[:2], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            self._to_pend = (buf, ev)

    def _step_inputs(self, input_ids, forward_context):
        """The STEP buffers (``tables.StepTables.step``) of this step, on the current stream."""
        engs, _ = self._host_inputs(input_ids, forward_context)
        self._device_inputs(input_ids, forward_context, engs)

    def precheck(self, batch, forward_context) -> bool:
        """Can the kernel serve this step's request shapes (no device work, nothing staged)? The same limits
        ``_sampling`` / ``_host_inputs`` refuse with MonoUnsupported, checked before the step commits to the kernel (an
        exception after the Engram rows are staged left the worker without a fallback: CT11 11-08, a 4.4k-token prompt
        with 4096 outputs past the index bound). Identical on every rank (the batch and its spans are)."""
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import BCOLS, ILW
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import T as T_MAX

        why = None
        if batch is not None and (np.asarray(batch.top_ks[: batch.total_seqs_num]) != -1).any():
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
            elif self.tables.comp:
                for r in (1, 2):
                    plan = step.plans[r]
                    if plan.num_compress > T_MAX or plan.num_write > T_MAX:
                        why = f"plan {r} rows"
        if why is None:
            self._precheck_why = None
        else:
            if why.split()[0] != getattr(self, "_precheck_why", None):   # once a kind (the bound values change a step)
                self._precheck_why = why.split()[0]
                logger.warning("V4.1 single-launch step: not served (%s), the production path runs (rank %d)", why, self.rank)
            return False
        return True

    def _host_inputs(self, input_ids, forward_context):
        """The step's host-side checks and the rare host-to-device refills (geometry), before the device preparation;
        returns (the Engram layers' embedding rows, the key of every address / shape ``_device_inputs`` captures)."""
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import BCOLS, ILW
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import T as T_MAX

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
            self._wkey, self._gk = wk, (geom.window_size, window.slot_rows, window.ring_slots, cache.num_slots)
        gk = self._gk
        self._hm("hi_geom")
        if gk != getattr(self, "_dgeom_key", None):   # (static for a cache: an H2D copy only when it changes)
            self._dgeom_key = gk
            S["dgeom"][:4].copy_(torch.tensor(gk, dtype=torch.int32))
        rope = self.drafter.model.rope
        if getattr(self, "_ctx_rope", None) is None:   # f32 copies of the draft's RoPE rows, once: a gather a step
            self._ctx_rope = (rope.cos_cache.view(-1, 32).float(), rope.sin_cache.view(-1, 32).float())
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
                raise MonoUnsupported(f"block table: {bt.shape[0]} requests x {-(-end // geom.block_size)} columns > 8 x {BCOLS}")
            key += [t.data_ptr() for t in (step.positions, step.slots, step.batch_ids, step.visible[1], step.visible[2])]
            key += [bt.data_ptr(), tuple(bt.shape), bt.stride(0)]
            if self.tables.comp:
                for r in (1, 2):
                    plan = step.plans[r]
                    for src, nl in ((plan.compress_plan_gpu, plan.num_compress), (plan.write_plan_gpu, plan.num_write)):
                        if nl > T_MAX:
                            raise MonoUnsupported(f"plan {r}: {nl} rows > {T_MAX}")
                        key += [src.data_ptr(), tuple(src.shape)]
        self._hm("hi_key")
        # the Engram layers' embedding rows (staged on Engram's side stream: ``get`` makes this stream wait for them)
        emb = getattr(md, "engram_embeddings", None)
        engs = {}
        for k, w in self.tables.engram.items():
            e = (dict.get(emb, k) if self._eng_rows is not None else emb.get(k)) if emb is not None else None
            if e is None or e.shape[-2] < rows or e.shape[-1] != w["engemb"].shape[1]:
                raise MonoUnsupported(f"engram layer {k}: embedding rows {None if e is None else tuple(e.shape)}")
            engs[k] = e
            key += [e.data_ptr(), tuple(e.shape)]
        er = self._eng_rows
        if er is not None:   # the staged Engram inputs the graph captures (fixed addresses: EngramStep)
            st = er.staging.step
            key += [er.width, id(er.staging)] + [None if t_ is None else t_.data_ptr() for t_ in (
                st.input_ids, st.live, st.dead, st.batch_ids, st.cu_seqlens, st.positions, st.cursor, st.history_index,
                st.candidates)]
        self._hm("host_inputs")
        return engs, tuple(key)

    def _device_inputs(self, input_ids, forward_context, engs):
        """Every device operation of the step's preparation: from fixed addresses only (captured in the width's
        graph, ``launch``), sizes fixed by the key of ``_host_inputs``."""
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import BCOLS
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import T as T_MAX
        from atom.models.deepseek_v41.mono.step_meta import write_step_meta

        md = forward_context.attn_metadata
        cache, step = md.cache, md.step
        rows = input_ids.numel()
        req = rows // 6
        S = self.tables.step
        geom = cache.geometry
        window = geom.window(0, cache.num_pages)
        S["sparam"].copy_(self._sp_pin, non_blocking=True)
        write_step_meta(step, geom, window, S["kvw"][:rows], S["kmeta"][: rows * 4].view(rows, 4), {}, (0, rows))
        S["rpos"][:rows].copy_(step.positions[:rows])
        S["tids"][:rows].copy_(input_ids.view(-1))
        S["dids"][: req * 5].copy_(input_ids.view(req, 6)[:, 1:].reshape(-1))
        # each request's state slot (the draft context's rows): not its batch position once requests have finished
        S["dgeom"][8 : 8 + req].copy_(step.slots[:req])
        S["ctx_kvw"][:rows].copy_(S["kvw"][:rows])
        pos = step.positions[:rows]
        torch.index_select(self._ctx_rope[0], 0, pos, out=S["ctx_rcos"][: rows * 32].view(rows, 32))
        torch.index_select(self._ctx_rope[1], 0, pos, out=S["ctx_rsin"][: rows * 32].view(rows, 32))
        if self.tables.idx:
            S["ibat"][:rows].copy_(step.batch_ids[:rows])
            for r in (1, 2):
                S[f"ibound{r}"][:rows].copy_(step.visible[r][:rows])
            # the requests' page tables: a fixed BCOLS columns (past the step's reach: never read)
            bt = step.block_tables
            n = bt.shape[0]
            cc = min(bt.shape[1], BCOLS)
            S["btab"][: n * BCOLS].view(n, BCOLS)[:, :cc].copy_(bt[:, :cc])
            if self.tables.comp:   # the compress / ring-write plans, -1 padded (pos < 0: skipped); the plan's rows past
                # its live count are sentinels (its fixed-capacity layout) or absent (a tight one)
                for r in (1, 2):
                    plan = step.plans[r]
                    for name, src in (("cplan", plan.compress_plan_gpu), ("wplan", plan.write_plan_gpu)):
                        dst = S[f"{name}{r}"]
                        dst.fill_(-1)
                        m = min(src.shape[0], T_MAX)
                        if m:
                            dst[: m * 4].copy_(src[:m].reshape(-1))
                S["slots"][:n].copy_(step.slots[:n])
        for k, w in self.tables.engram.items():
            e = engs[k]
            w["engemb"][:rows].copy_(e.reshape(-1, e.shape[-1])[:rows])
        self.scratch["cnt"].zero_()
        self.scratch["ss"].zero_()
        # Markov's per-(position, request) global argmax is an atomic max: it must restart from INT64_MIN every step
        # (as the fixture does), else the drafts keep earlier steps' winners
        self.scratch["mbest"].view(torch.int64).fill_(-(1 << 63))
        self._hm("device_inputs")

    def _hm(self, name) -> None:
        """Bring-up timing (ATOM_MONO_SINGLE_TIMING): a host time mark inside the step's preparation."""
        r = getattr(self, "_t_rec", None)
        if r is not None:
            import time

            r.setdefault("hm", []).append((name, time.perf_counter()))

    def _sampling(self, batch, req) -> None:
        """Each target row's sampling from its request (``ScheduledBatch`` temperatures / top_ps / top_ks), as the
        recipe's prob drafting + block verification (``dspark_prob_sample``, PROB_DRAFT / BLOCK_VERIFY): greedy
        (temperature 0) -> argmax drafts and the argmax match; else the drafts are sampled from q (1 / temperature
        times ``ATOM_DSPARK_PROB_DRAFT_TSCALE``) and verified against p (the top-p set of logits / temperature) by the
        probability ratio. q is the kernel's own (``qlog``) when the request's drafts came from the last kernel launch
        (``hasq``, its row there ``qsrc``), else the one-hot rule. Noise seeds mix a step counter and the batch row.
        Top-k is not in the kernel: refused."""
        if batch is None:
            raise MonoUnsupported("single-launch step: no batch for the sampling parameters")
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
            inv_t = np.where(greedy, 1.0, 1.0 / np.maximum(temps, 1e-6)).astype(np.float32)
            for c, v in ((0, inv_t), (1, np.where(greedy, 1.0, top_ps)), (4, inv_t / DRAFT_TSCALE)):
                sp[:n, c] = np.repeat(v, ROWS_PER_REQ)  # a request's 6 verify rows share its sampling
            sp.view(np.int32)[:n, 3] = np.repeat(greedy.astype(np.int32), ROWS_PER_REQ)
            rr = np.repeat(np.arange(req, dtype=np.int64), ROWS_PER_REQ)
            sc = self._sp_static = (skey, sp, rr * 0x85EBCA77, rr * 0x27D4EB2F + 0x165667B1)
        sp = sc[1]
        ints = sp.view(np.int32)
        prev = getattr(self, "_qrow", {})
        ints[:n, 5] = np.repeat(np.array([r in prev for r in rids], dtype=np.int32), ROWS_PER_REQ)
        ints[:n, 6] = np.repeat(np.array([prev.get(r, 0) for r in rids], dtype=np.int32), ROWS_PER_REQ)
        ints[:n, 2] = ((self._step_no * 0x9E3779B1 + sc[2]) & 0x7FFFFFFF).astype(np.int32)
        ints[:n, 7] = ((self._step_no * 0xC2B2AE35 + sc[3]) & 0x7FFFFFFF).astype(np.int32)
        # staged in a fixed pinned buffer; ``_device_inputs`` copies it (non-blocking: a pageable copy waited for the
        # stream to drain -- the previous step's draft launch -- stalling this step's host preparation). Rewritten a
        # step later only after that step's first launch, which follows the copy, has completed (``verify_ready``).
        if getattr(self, "_sp_pin", None) is None:
            self._sp_pin = torch.empty(sp.shape, dtype=torch.float32, pin_memory=True)
        self._sp_pin.numpy()[:] = sp
        # this launch's drafts (and their q in qlog / qids) by request: committed once the launch is issued
        self._qrow_next = {r: i for i, r in enumerate(rids)}

    def forget_drafts(self, req_ids) -> None:
        """These requests' next drafts are not the kernel's (the production drafter ran): no q for their verification."""
        prev = getattr(self, "_qrow", None)
        if prev:
            for r in req_ids:
                prev.pop(r, None)

    def _layer_ids(self):
        ids = getattr(self, "_layer_ids_c", None)
        if ids is None:
            ids = [blk.attn.spec.layer_id for blk in self.tables.target_blocks]
            ids = self._layer_ids_c = ids + [blk.attn.spec.layer_id for blk in self.tables.draft_blocks]
        return ids

    def launch(self, input_ids, forward_context, batch=None) -> None:
        rows = input_ids.numel()
        if not self.prepare(rows):
            raise MonoUnsupported(f"single-launch step: {rows} rows")
        self._sampling(batch, rows // ROWS_PER_REQ)
        self._hm("sampling")
        engs, key = self._host_inputs(input_ids, forward_context)
        if getattr(self, "_t_rec", None) is not None:
            self._timer.mark("k", self._t_rec)
        launches = self._fast_launchers(rows)
        args = (*self._launch_args(), torch.cuda.current_stream())

        er = self._eng_rows

        def first():   # the step's device preparation (the Engram stage first) and its first launch
            if er is not None:
                er.stage()
                # join Engram's side stream BEFORE the launch: the step kernel occupies every WGP and spins on its peers,
                # so a side-stream collective (the embedding all-gather) queued behind it on one GPU while its peers' step
                # kernels wait for this GPU's is a cross-GPU circular wait; the launch also reads the embedding rows
                er.join()
            self._device_inputs(input_ids, forward_context, engs)
            launches[0](*args[:-1], torch.cuda.current_stream())

        g = self._graphs.get(rows)
        if g is not None and g[0] == key:
            g[1].replay()
        elif self._seen.get(rows) == key:
            # the same addresses / shapes as the width's previous (eager) step: capture its preparation and first
            # launch, then replay -- a key change (a reallocation, a geometry or plan-capacity change) recaptures
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, capture_error_mode="thread_local"):
                first()
            self._graphs[rows] = (key, gr)
            logger.info("V4.1 single-launch step: captured the preparation graph of %d rows (rank %d)", rows, self.rank)
            gr.replay()
        else:
            self._seen[rows] = key
            first()
        self._ev_verify.record()   # the verdict is ready here; the draft launch follows on the same stream
        launches[1](*args)
        self._check_timeouts()
        self._qrow = self._qrow_next

    def run_step(self, input_ids, running_bs: int, forward_context, batch) -> None:
        """The served step: one launch (target layers, head, sampling, verify, draft layers, Markov drafts)."""
        from atom.model_ops.engram.device.staging import EngramStagedRows

        if input_ids.numel() != running_bs * ROWS_PER_REQ:
            raise MonoUnsupported(f"single-launch step: {input_ids.numel()} rows for {running_bs} requests")
        # what the production forward opens and closes around its layers (runtime.v41_begin_forward / end_forward): the
        # Engram stage -- its history snapshot, each verify prefix's cursor candidate (the TentativeState staging that
        # commit_tentative picks from by the verdict's anchors) and the embedding rows' lookup on its side stream
        _w0 = time.perf_counter() if self._timer is not None else None
        md = forward_context.attn_metadata
        if envs.ATOM_MONO_SINGLE_PROWDUMP:   # (debug) this step's positions for ``dump_step``
            self._dbg_pos = md.step.positions[: input_ids.numel()].clone()
        rows = md.engram_embeddings
        staged = isinstance(rows, EngramStagedRows) and not getattr(md, "staged_outside_forward", False)
        # the Engram stage (its lookups, the side-stream gather and the join) runs inside the launch's preparation graph
        # (``launch``): issued op by op here it was ~0.26 ms of host time between the verdict and the next launch
        self._eng_rows = rows if staged else None
        _w1 = time.perf_counter() if self._timer is not None else None
        self._t_rec = {} if self._timer is not None else None
        if self._t_rec is not None:
            self._timer.mark("a", self._t_rec)
            self._t_rec["pre"] = (self._t_rec["w_a"] - _w0, _w1 - _w0)   # (run_step entry -> a, of which the Engram stage)
        self.launch(input_ids, forward_context, batch)
        if self._t_rec is not None:
            self._timer.copy_st(self._t_rec, self.scratch["st"])
            self._timer.mark("z", self._t_rec)
            self._timer.step(self._t_rec, input_ids.numel())

    def verify_outputs(self, bs: int):
        """``(output_token_ids [bs, 6] int32, -1 past the emitted tokens; verdict [3, bs] int32: accepted drafts,
        rejected drafts, anchor row)`` -- the kernel's verify (step_tail_ct3), copied out of the scratch the next launch
        rewrites: on a side stream right after the first launch, so the host's wait for them (the step's output D2H,
        ``verify_ready``) overlaps the draft launch; the caller's stream waits for the copies."""
        cur = torch.cuda.current_stream()
        with torch.cuda.stream(self._verify_stream):
            self._verify_stream.wait_event(self._ev_verify)
            out = self.scratch["outtok"].view(torch.int32)[: 8 * ROWS_PER_REQ].view(8, ROWS_PER_REQ)[:bs].clone()
            verdict = self.scratch["verdict"].view(torch.int32)[:24].view(3, 8)[:, :bs].clone()
            self.verify_ready.record()
        out.record_stream(cur)
        verdict.record_stream(cur)
        cur.wait_event(self.verify_ready)
        self.dump_step(bs, out, verdict)
        return out, verdict

    def draft_ids(self, bs: int):
        """The next step's drafts [bs, 5] (the kernel's Markov block sampling, mids), as ``Drafter.propose`` returns."""
        return self.scratch["mids"].view(torch.int32)[: 8 * 5].view(8, 5)[:bs].to(torch.int64)
