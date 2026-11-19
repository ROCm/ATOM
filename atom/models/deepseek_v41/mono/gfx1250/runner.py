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
``kernels.step``), the scratch (``buffers.scratch_bytes``, once at the widest step),
the peer receive buffers (``SymmetricPeerBuffer``). The weight tables
and the step metadata (``buffers``: WEIGHT / STEP) are not bound yet, so construction still refuses at its end and
``LazyRunner`` keeps the production path."""

import dataclasses
import importlib
import logging
import time

import flydsl.compiler as flyc
import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.engram.device.staging import EngramStagedRows, EngramStaging
from atom.models.deepseek_v41.mono.gfx1250 import buffers
from atom.models.deepseek_v41.mono.gfx1250.config import (
    NDR,
    NL,
    ROWS_PER_REQ,
    TP,
)
from atom.models.deepseek_v41.mono.gfx1250.inputs import StepInputs, sim_acc_words
from atom.models.deepseek_v41.mono.gfx1250.tables import StepTables
from atom.mono.plan.build_key import source_digest
from atom.mono.runtime.compile import compile_only
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.peer_memory import SymmetricPeerBuffer
from atom.utils.forward_context import RoundService, get_forward_context

logger = logging.getLogger("atom")


# The TileRT-structured step (kernels/step.py: one row cluster a request, the mix partials' [8]-row layout) is the
# c1 shape: one request (6 rows). Wider steps take the production path.
MAX_STEP_ROWS = 6


def _base_model(model):
    for m in model.modules():
        if (
            hasattr(m, "embed")
            and hasattr(m, "head")
            and hasattr(m, "norm")
            and hasattr(m, "layers")
        ):
            return m
    raise MonoUnsupported("no V4.1 base model")


KERNELS = __package__ + ".kernels"
# the receive regions: 4 TP regions x 4 sources x 40960 16-byte words, the sampling and Markov exchange after them
# (tail SOFF / SOFF2), 2 MB-rounded, doubled for the step epoch
RECV_BYTES = 2 * (
    (16 * 40960 * 16 + 4 * 4 * 48 * 256 * 16 + (2 << 20) - 1) // (2 << 20) * (2 << 20)
)


# the kernel sources' digest: part of every build's FlyDSL JIT key (an edit to any kernel module rebuilds)
SOURCES = source_digest("models/deepseek_v41/mono/gfx1250/kernels")


def import_kernels() -> dict:
    """The kernel modules (this package's kernels/: the whole step is step)."""
    names = (
        "device",
        "attention",
        "front",
        "moe",
        "draft_ctx",
        "head",
        "sample_common",
        "sample",
        "markov",
        "tail",
        "step",
        "prep",
    )
    return {n: importlib.import_module(f"{KERNELS}.{n}") for n in names}


class SingleStepRunner(StepInputs):
    def __init__(self, model, drafter, atom_config) -> None:
        if get_tensor_model_parallel_world_size() != TP:
            raise MonoUnsupported(
                f"TP {get_tensor_model_parallel_world_size()} (the step kernel is TP{TP})"
            )
        self.rank = get_tensor_model_parallel_rank()
        # forced acceptance (--spec-decode-acceptance-length / -rate): the kernel's verify draws it (``_sampling``)
        spec = getattr(atom_config, "speculative_config", None)
        self.sim_acc = sim_acc_words(getattr(spec, "synthetic_acceptance_rates", None))
        if self.sim_acc is not None:
            logger.info(
                "V4.1 single-launch step: forced acceptance, %d drafts then one with probability %.4f",
                *self.sim_acc,
            )
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
        # (englook) the target entry hashes and looks the Engram rows up itself: of ATOM's EngramStaging it drives only
        # the snapshot and the cursor advance (``launch``), two private steps -- if that class no longer has them, the
        # single-launch step declines (ATOM's own path serves)
        if not all(
            callable(getattr(EngramStaging, m, None)) for m in ("_snapshot", "_advance_cursor")
        ):
            raise MonoUnsupported(
                "engram: EngramStaging has no _snapshot / _advance_cursor (the step kernel's lookup drives them)"
            )
        # the kernel's ring slots (window + block width) and pool row width (bf16 elements) are compile-time constants
        AS = self.K["attention"]
        if int(window.ring_slots) != AS.KG_RING:
            raise MonoUnsupported(
                f"window ring {window.ring_slots} slots (the step kernel's: {AS.KG_RING})"
            )
        # the index plane's format is compiled in (a build key parameter): FP8 tiles, or FP4 natural rows + scales
        self.index_fp4 = bool(getattr(cache.geometry, "index_fp4", False))
        if self.index_fp4 and getattr(cache.geometry, "index_layout", None) != "natural":
            raise MonoUnsupported(f"FP4 index layout {getattr(cache.geometry, 'index_layout', None)} (natural only)")
        # the KV format is compiled in too: BF16 rows in the one pool, or (FP4) the main pool's E2M1 rows and the window
        # rings' MXFP8 rows, each source its own data / scale planes (cache.main_view / ring_view), row-addressed
        g = cache.geometry
        self.kv_fp4 = g.kv_dtype == "fp4"
        if g.kv_dtype not in ("bf16", "fp4"):
            raise MonoUnsupported(f"kv cache {g.kv_dtype}")
        if self.kv_fp4:
            src = (getattr(cache, "main_sources", None), getattr(cache, "window_sources", None))
            if not (
                g.packed
                and g.window_fp8
                and g.main_row_bytes == AS.KG_MAIN_FP4_BYTES
                and g.window_row_bytes == AS.KG_WIN_FP8_BYTES
                and g.head_dim == AS.HD
                and src[0] is not None
                and src[1] is not None
                and window.ring_start == 0
                and window.slot_rows == window.ring_slots
            ):
                raise MonoUnsupported(
                    f"FP4 KV geometry: main row {g.main_row_bytes} B, window row {g.window_row_bytes} B, "
                    f"ring start {window.ring_start}, slot rows {window.slot_rows} (planar sources expected)"
                )
        miss = buffers.check_catalog(self.K)
        if miss:
            raise MonoUnsupported(f"pointer catalog out of date: {miss[:4]}")
        self.scratch = {
            n: torch.zeros(b, dtype=torch.uint8, device=self.device)
            for n, b in buffers.scratch_bytes(self.K, NL + NDR).items()
        }
        # fine-grained, not uncached: the step's few-word peer pushes (Markov, sampling) waited ~80-100 us each for an
        # uncached store to become visible (fine-grained receive buffers did not, measured on the kernel's fixture)
        self.peer = SymmetricPeerBuffer(
            RECV_BYTES,
            rank=self.rank,
            npes=TP,
            group=get_tp_group().cpu_group,
            fine_grained=True,
        )
        self.recv = [int(a) for a in self.peer.addresses.tolist()]
        # the STOP word, past the kernel's two parity halves (kernels.step.STOP_OFF; device.fail_ctx): zeroed once, never
        # a step; a kernel poll that times out writes it on every rank, and the host fails the run at the next launch
        STOP_OFF = self.K["step"].STOP_OFF
        if STOP_OFF + 4 > RECV_BYTES:
            raise MonoUnsupported("receive buffer too small for the STOP word")
        # the STOP word: zeroed before the first launch (the construction's collectives order it before any peer's
        # write); its host copy is taken after each verify (``verify_outputs``) and checked at the next launch
        self._stop = self.peer.storage[STOP_OFF : STOP_OFF + 4].view(torch.int32)
        self._stop.zero_()
        self._stop_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._stop_ready = torch.cuda.Event()
        self._stop_ready.record()
        # an in-rank poll that timed out (a local producer's ready never came) leaves no STOP (in-rank polls stay as
        # small as before: code size is the step's cost), but costs its launch >= POLL_TICKS: the step's GPU time
        # bounds it (POLL_TICKS 20 s; a normal step is ~4 ms)
        # the previous target launch's steady-counter stamps (prep's entry, each CTA's exit: sample.COM_T0 / COM_EXIT),
        # copied with the STOP word after its verify (``verify_outputs``) and checked at the next launch: no timing events
        # on the step's stream (each record there cost the GPU ~3 us between the step's operations, E0)
        SPF = self.K["sample"]
        if SPF.COM_NEXIT < self.K["attention"].CTAS:
            raise MonoUnsupported("tcom: fewer exit stamps than the step's CTAs")
        self._ts_words = (SPF.COM_T0, SPF.COM_EXIT + SPF.COM_NEXIT)
        self._ts_host = torch.zeros(self._ts_words[1] - self._ts_words[0], dtype=torch.int64, pin_memory=True)
        self._ts_owed = False
        logger.info(
            "V4.1 single-launch step: %.1f MB scratch, %.1f MB receive (rank %d)",
            sum(t.numel() for t in self.scratch.values()) / 2**20,
            RECV_BYTES / 2**20,
            self.rank,
        )
        self.tables = StepTables(
            self.K,
            model,
            drafter,
            self.scratch,
            self.recv,
            self.rank,
            get_tp_group().device_group,
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
        self._ev_verify = torch.cuda.Event()
        # ModelRunner's output D2H waits for it (its ``forward_done_event`` on a served step; marked as this runner's)
        self.verify_ready = torch.cuda.Event()
        # the stream the verdict's copies run on: ModelRunner's output D2H waits for it, not for the draft launch
        self.output_stream = self._verify_stream = torch.cuda.Stream()
        self._eng_rows = None
        self._eng_staged = None
        self._adv = self._adv_next = self._com = False
        # the verify's outputs for the host: a 2-slot pair the kernel writes (sample.OUT_SLOT_*), the step's slot in sparam
        # row OUT_SLOT_ROW word 7 (``_sampling``); the draft launch waits until the step's output copies are queued
        # (``launch_draft``: SingleStepRound.committed, else draft_ids, else the next launch)
        SPF = self.K["sample"]
        if MAX_STEP_ROWS > SPF.OUT_SLOT_ROW:
            raise MonoUnsupported("sparam row of the output slot is a served row")
        self._out_slot = 1
        self._draft_pending = None
        # each request's first position of the step (its preparation writes it): where the decode round's next step
        # advances from (``round.SingleStepRound``)
        self.first_positions = torch.zeros(
            buffers.T_MAX // ROWS_PER_REQ, dtype=torch.int32, device="cuda"
        )
        # every width built now (each a compile or a disk-cache load, then a TP barrier: ``prepare``), not on its first
        # step: a width first met mid-serve stalled every request for its JIT (~18 s cold)
        t0 = time.perf_counter()
        for rows in range(ROWS_PER_REQ, buffers.T_MAX + 1, ROWS_PER_REQ):
            self.prepare(rows)
        logger.info(
            "V4.1 single-launch step: %d widths built in %.1f s (rank %d)",
            len(self.builds),
            time.perf_counter() - t0,
            self.rank,
        )

    def prepare(self, rows: int) -> bool:
        if rows % 6 or not 0 < rows <= MAX_STEP_ROWS:
            return False
        if rows not in self.builds:
            M = self.K["step"]
            kw = {"NL": NL, "ndr": NDR, "trd": rows // 6 * 5, "index_fp4": self.index_fp4, "kv_fp4": self.kv_fp4}
            # two launches: target .. verification, then the draft (the host reads the verdict between)
            launch = tuple(
                M.build(rows, self.rank, part=pt, sources=SOURCES, **kw)[0]
                for pt in ("target", "draft")
            )
            # compile (JIT or disk cache) before the first launch of this width, then wait for every rank: the step
            # kernel's cross-rank polls give up after 2 s, and ranks compiling a width seconds apart made that first
            # launch time out and compute on stale / zero peer data (KV / index / compressor state stays wrong after)
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

            def first(i):
                def call(*a):
                    fn = flyc.compile(self.builds[rows][i], *a)
                    if fn is None:
                        raise MonoUnsupported(
                            f"single-launch step: no compiled launcher for {rows} rows"
                        )
                    self._fast[rows][i] = fn

                return call

            fast = self._fast[rows] = [first(i) for i in range(len(self.builds[rows]))]
        return fast

    def _launch_args(self) -> tuple:
        T_ = self.tables
        return (
            *self.recv,
            self.scratch["cnt"].data_ptr(),
            float(self.base.layers[0].attn.softmax_scale),
            T_.fptab.data_ptr(),
            T_.tp.data_ptr(),
            0,  # lo / hi: the target layer range of a split launch (one launch: every layer)
            len(self.base.layers),
        )

    def _layer_ids(self):
        ids = getattr(self, "_layer_ids_c", None)
        if ids is None:
            ids = [blk.attn.spec.layer_id for blk in self.tables.target_blocks]
            ids = self._layer_ids_c = ids + [
                blk.attn.spec.layer_id for blk in self.tables.draft_blocks
            ]
        return ids

    def launch_draft(self) -> None:
        """The step's draft launch, held back by ``launch`` until the host queued the copies of the step's outputs
        (ModelRunner's sampled-id and status D2H on its copy stream, ``verify_outputs``' copies): queued after the draft,
        a copy stream that shares the main stream's hardware queue ran them -- and every later operation -- after the
        draft (profiler, CT11: one rank, a whole request, +30 us a step between the draft's end and the next target)."""
        p = self._draft_pending
        if p is None:
            return
        self._draft_pending = None
        p[0](*p[1])
        if not torch.cuda.is_current_stream_capturing() and self.K["attention"].DSTAMP:
            self._log_dstamps()

    def launch(self, input_ids, forward_context, batch=None) -> None:
        self.launch_draft()   # (a step whose draft no hook launched: before this step's target, in stream order)
        self._out_slot ^= 1
        # the previous step's STOP copy (done long before: its output D2H waited for it; never waited for while a graph
        # captures -- the check then runs at the next eager launch)
        if not torch.cuda.is_current_stream_capturing():
            self._stop_ready.synchronize()
        stop = int(self._stop_host[0])
        # the previous target launch's GPU time, prep's entry to its last CTA's exit (its stamps were copied with the STOP
        # word, waited for just above: a hung launch still blocks there, as its end event did; a timed-out poll ends it)
        if self._ts_owed and not torch.cuda.is_current_stream_capturing():
            self._ts_owed = False
            ts = self._ts_host
            ms = (int(ts[1:].max()) - int(ts[0])) / 1e5   # (steady counter: 100 MHz)
            if ms > 18000.0:   # (a timed-out poll costs >= POLL_TICKS = 20 s; the first launch's module-load rank skew stays below it)
                raise RuntimeError(
                    f"V4.1 single-launch step: a step took {ms:.0f} ms on the GPU (a device poll timed out, "
                    f"rank {self.rank}); its outputs are invalid"
                )
        if stop != 0:
            # a step's poll timed out (a peer's message never came): its outputs are not this step's; stop the run
            raise RuntimeError(
                f"V4.1 single-launch step: a device poll timed out (STOP generation {stop & 0x7FFFFFFF:#x}, "
                f"rank {self.rank}); the step's outputs are invalid"
            )
        rows = input_ids.numel()
        if not self.prepare(rows):
            raise MonoUnsupported(f"single-launch step: {rows} rows")
        self._sampling(batch, rows // ROWS_PER_REQ)
        engs, key = self._host_inputs(input_ids, forward_context)
        launches = self._fast_launchers(rows)
        args = (*self._launch_args(), torch.cuda.current_stream())

        er = self._eng_rows
        if getattr(self, "_eng_staged", None) is not None:
            self._englook(self._eng_staged)

        def first():  # the step's device preparation (the Engram stage first) and its first launch
            self._eng_prep = None
            if er is not None:
                # the launch reads the staged rows: a step the builder's round served stages in full anyway (the builder
                # marks a served step's staging cursor-only, for models that look their rows up themselves)
                st = er.staging
                if st.step is not None and st.step.served_by_round:
                    st.step = dataclasses.replace(st.step, served_by_round=False)
                # (englook) the entry hashes the snapshot and looks the rows up itself: of EngramStaging.start only the
                # snapshot and the cursor advance (no side stream, lookup or all-gather); its join then waits on nothing,
                # as for a step the model's own round serves
                st = er.staging
                st._served = True
                # a verify step on the one-launch preparation (kernels/prep.py): its compress and snapshot run in that
                # launch (no cursor advance in between: a verify step's _advance_cursor launches nothing)
                self._eng_prep = er if self._prep_takes_engram(er) else None
                if self._eng_prep is None:
                    st._snapshot(er.width)
                st._advance_cursor(er.width)
                # join Engram's side stream BEFORE the launch: the step kernel occupies every WGP and spins on its peers,
                # so a side-stream collective (the embedding all-gather) queued behind it on one GPU while its peers' step
                # kernels wait for this GPU's is a cross-GPU circular wait; the launch also reads the embedding rows
                er.join()
            self._device_inputs(input_ids, forward_context, engs)
            launches[0](*args[:-1], torch.cuda.current_stream())

        # the device round's advance runs in this step's preparation (``advances``): its own build, so its own graph
        # (only on the device-round step the round laid out for it: a flag left by a step the launch did not serve lapses)
        md = forward_context.attn_metadata
        adv = self._adv = bool(self._adv_next) and md.served_by_round is RoundService.DEVICE
        self._adv_next = False
        gk = (rows, adv)
        g = self._graphs.get(gk)
        if g is not None and g[0] == key:
            g[1].replay()
            self._com = g[2]
        elif self._seen.get(gk) == key:
            # the same addresses / shapes as the width's previous (eager) step: capture its preparation and first
            # launch, then replay -- a key change (a reallocation, a geometry or plan-capacity change) recaptures
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, capture_error_mode="thread_local"):
                first()
            self._graphs[gk] = (key, gr, self._com)
            logger.info(
                "V4.1 single-launch step: captured the preparation graph of %d rows (rank %d)",
                rows,
                self.rank,
            )
            gr.replay()
        else:
            self._seen[gk] = key
            first()
        # the verdict is ready here (and the target launch's stamps); the draft launch follows on the same stream
        self._ev_verify.record()
        self._draft_pending = (launches[1], args)
        self._qrow = self._qrow_next

    def _englook(self, er) -> None:
        """(englook) The entry's Engram lookup inputs, once: engram.LW int64 words a slot -- the layer's hash constants
        (multipliers, column sizes / offsets, pad id), this rank's page-locked host shard (codes, scales, first row) and
        the staging snapshot's address (fixed: graph-captured)."""
        if getattr(self, "_englook_set", False):
            return
        ES = self.K["step"].ES
        st = er.staging
        host, tabs = getattr(st, "host", None), getattr(getattr(st, "uva", None), "hash_tables", None)
        lids = sorted(self.tables.engram)
        if (
            host is None
            or tabs is None
            or getattr(st, "snapshot", None) is None
            or list(host.layer_ids) != lids
            or len(lids) != ES.NSLOT
            or tabs.ngram != ES.NGR
            or tabs.heads != ES.NCOL
            or host.local_heads != ES.HPR
            or host.head_start != self.rank * ES.HPR
            or st.snapshot.shape[1] != ES.NGR
        ):
            raise MonoUnsupported("engram: staging geometry (the step kernel's lookup: 4-grams, 24 columns, 6 a rank)")
        w = torch.zeros(ES.NSLOT, ES.LW, dtype=torch.int64)
        for s_, lid in enumerate(lids):
            mult, sizes, offs = (x.detach().cpu().to(torch.int64) for x in tabs.layers[lid])
            table = host.prefetcher._tables[lid]
            if table.head_dim != 256 or (table.block_size or 0) != 32:
                raise MonoUnsupported(f"engram: table rows {table.head_dim} / blocks {table.block_size}")
            weight, scales, row0, _ = table.registered_shard()
            w[s_, ES.LW_MULT : ES.LW_MULT + ES.NGR] = mult[: ES.NGR]
            w[s_, ES.LW_SIZE : ES.LW_SIZE + ES.NCOL] = sizes[: ES.NCOL]
            w[s_, ES.LW_OFF : ES.LW_OFF + ES.NCOL] = offs[: ES.NCOL]
            w[s_, ES.LW_PAD] = int(tabs.pad_id)
            w[s_, ES.LW_ROW0] = int(row0)
            w[s_, ES.LW_WP] = weight.data_ptr()
            w[s_, ES.LW_SP] = scales.data_ptr()
            w[s_, ES.LW_SNAP] = st.snapshot.data_ptr()
        self.scratch["englook"].view(torch.int64)[: w.numel()].copy_(w.view(-1))
        torch.cuda.current_stream().synchronize()
        self._englook_set = True

    def _log_dstamps(self):
        """(diagnostic, ATOM_MONO_DSTAMP=1) every 50 launches: D CTA 0's mean steady-counter time (us, 100 MHz) from D's
        entry to each phase point, over the target layers and the draft layers (attention._dstamp)."""
        self._dst_n = getattr(self, "_dst_n", 0) + 1
        if self._dst_n % 50 or self.rank != 0:
            return
        torch.cuda.synchronize()
        v = self.scratch["cyc"][: (NL + NDR) * 32].view(torch.int32).view(NL + NDR, 8).double().cpu()
        for name, rows in (("target", v[:NL]), ("draft", v[NL:])):
            n = rows[:, 0].clamp(min=1)
            mean = (rows[:, 1:] / n[:, None]).mean(0) / 100.0
            logger.info(
                "V4.1 single-launch step D stamps (%s, %d passes): %s us",
                name,
                int(rows[:, 0].sum()),
                " ".join(f"p{i + 1}={x:.2f}" for i, x in enumerate(mean.tolist())),
            )

    def run_step(self, input_ids, running_bs: int, forward_context, batch) -> None:
        """The served step: one launch (target layers, head, sampling, verify, draft layers, Markov drafts)."""
        if input_ids.numel() != running_bs * ROWS_PER_REQ:
            raise MonoUnsupported(
                f"single-launch step: {input_ids.numel()} rows for {running_bs} requests"
            )
        # what the production forward opens and closes around its layers (runtime.v41_begin_forward / end_forward): the
        # Engram stage -- its history snapshot, each verify prefix's cursor candidate (the TentativeState staging that
        # commit_tentative picks from by the verdict's anchors) and the embedding rows' lookup on its side stream
        md = forward_context.attn_metadata
        rows = md.engram_embeddings
        staged = isinstance(rows, EngramStagedRows) and not getattr(
            md, "staged_outside_forward", False
        )
        # the Engram stage (its lookups, the side-stream gather and the join) runs inside the launch's preparation graph
        # (``launch``): issued op by op here it was ~0.26 ms of host time between the verdict and the next launch
        self._eng_rows = rows if staged else None
        # (englook) the staging whose snapshot the entry hashes (staged here or, outside the forward, already)
        self._eng_staged = rows if isinstance(rows, EngramStagedRows) else None
        self.launch(input_ids, forward_context, batch)

    def verify_outputs(self, bs: int):
        """``(output_token_ids [bs, 6] int32, -1 past the emitted tokens; verdict [3, bs] int32: accepted drafts,
        rejected drafts, anchor row)`` -- the kernel's verify (tail), in place: the step's slot of the 2-slot pair the
        verify writes (the next launch writes the other; the one after reuses this, by when the host received every copy
        of it). Produced on the caller's stream: its readers there need no wait. The output stream waits for the target
        launch (``verify_ready``: ModelRunner's output D2H), and takes the STOP word and the stamps.
        """
        SPF = self.K["sample"]
        sl = self._out_slot
        ob = self.scratch["outsl"]
        out = ob[sl * SPF.OUT_SLOT_BYTES : (sl + 1) * SPF.OUT_SLOT_BYTES].view(torch.int32).view(8, ROWS_PER_REQ)[:bs]
        vo = 2 * SPF.OUT_SLOT_BYTES + sl * SPF.VER_SLOT_BYTES
        verdict = ob[vo : vo + SPF.VER_SLOT_BYTES].view(torch.int32).view(3, 8)[:, :bs]
        with torch.cuda.stream(self._verify_stream):
            self._verify_stream.wait_event(self._ev_verify)
            self.verify_ready.record()
            self._stop_host.copy_(self._stop, non_blocking=True)
            a_, b_ = self._ts_words
            self._ts_host.copy_(self.scratch["tcom"].view(torch.int64)[a_:b_], non_blocking=True)
            self._ts_owed = True
            self._stop_ready.record()
        # the anchors this launch's verify committed in the kernel (``committed_in_kernel``) and that its successor's
        # preparation may advance from (``advances``): the slot's row, as the round hands them back
        self._verdict_anchors = verdict[2].data_ptr() if bs else None
        self._verdict_ref = verdict
        return out, verdict

    def committed_in_kernel(self, anchors) -> bool:
        """Whether the last launch's verify committed the Engram cursors at ``anchors`` (sample.commit: its preparation
        staged every request's candidates, int64 rows) -- the runner's own verdict, as ``verify_outputs`` gave it."""
        return bool(
            getattr(self, "_com", False)
            and anchors is not None
            and anchors.data_ptr() == getattr(self, "_verdict_anchors", None)
        )

    def advances(self, anchors, requests: int) -> bool:
        """The next launch's preparation advances its positions from the last launch's verdict (kernels/prep.py adv; in
        place of ``device_round.advance``) when ``anchors`` are that verdict's (still in the scratch: no launch since)
        and the preparation is the one-launch kind."""
        ok = bool(
            self.tables.idx
            and self.tables.comp
            and anchors is not None
            and anchors.data_ptr() == getattr(self, "_verdict_anchors", None)
            and tuple(anchors.shape) == (requests,)
        )
        self._adv_next = ok
        return ok

    def draft_ids(self, bs: int):
        """The next step's drafts [bs, 5] int64 (the kernel's Markov block sampling, mids64), as ``Drafter.propose``
        returns: a view (the next draft launch rewrites it, after the next step's preparation has read it)."""
        self.launch_draft()
        return self.scratch["mids64"].view(torch.int64)[: 8 * 5].view(8, 5)[:bs]
