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

from atom.model_ops.engram.device.staging import EngramStagedRows
from atom.models.deepseek_v41.mono.gfx1250 import buffers
from atom.models.deepseek_v41.mono.gfx1250.config import (
    NDR,
    NL,
    ROWS_PER_REQ,
    TP,
)
from atom.models.deepseek_v41.mono.gfx1250.inputs import StepInputs
from atom.models.deepseek_v41.mono.gfx1250.tables import StepTables
from atom.mono.plan.build_key import source_digest
from atom.mono.runtime.compile import compile_only
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.peer_memory import SymmetricPeerBuffer
from atom.utils.forward_context import get_forward_context

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
    )
    return {n: importlib.import_module(f"{KERNELS}.{n}") for n in names}


class SingleStepRunner(StepInputs):
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
        AS = self.K["attention"]
        if int(window.ring_slots) != AS.KG_RING:
            raise MonoUnsupported(
                f"window ring {window.ring_slots} slots (the step kernel's: {AS.KG_RING})"
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
            kw = {"NL": NL, "ndr": NDR, "trd": rows // 6 * 5}
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

    def launch(self, input_ids, forward_context, batch=None) -> None:
        rows = input_ids.numel()
        if not self.prepare(rows):
            raise MonoUnsupported(f"single-launch step: {rows} rows")
        self._sampling(batch, rows // ROWS_PER_REQ)
        engs, key = self._host_inputs(input_ids, forward_context)
        launches = self._fast_launchers(rows)
        args = (*self._launch_args(), torch.cuda.current_stream())

        er = self._eng_rows

        def first():  # the step's device preparation (the Engram stage first) and its first launch
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
            logger.info(
                "V4.1 single-launch step: captured the preparation graph of %d rows (rank %d)",
                rows,
                self.rank,
            )
            gr.replay()
        else:
            self._seen[rows] = key
            first()
        self._ev_verify.record()  # the verdict is ready here; the draft launch follows on the same stream
        launches[1](*args)
        self._qrow = self._qrow_next

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
        self.launch(input_ids, forward_context, batch)

    def verify_outputs(self, bs: int):
        """``(output_token_ids [bs, 6] int32, -1 past the emitted tokens; verdict [3, bs] int32: accepted drafts,
        rejected drafts, anchor row)`` -- the kernel's verify (tail), copied out of the scratch the next launch
        rewrites: on a side stream right after the first launch, so the host's wait for them (the step's output D2H,
        ``verify_ready``) overlaps the draft launch; the caller's stream waits for the copies.
        """
        cur = torch.cuda.current_stream()
        with torch.cuda.stream(self._verify_stream):
            self._verify_stream.wait_event(self._ev_verify)
            out = (
                self.scratch["outtok"]
                .view(torch.int32)[: 8 * ROWS_PER_REQ]
                .view(8, ROWS_PER_REQ)[:bs]
                .clone()
            )
            verdict = (
                self.scratch["verdict"]
                .view(torch.int32)[:24]
                .view(3, 8)[:, :bs]
                .clone()
            )
            self.verify_ready.record()
        out.record_stream(cur)
        verdict.record_stream(cur)
        cur.wait_event(self.verify_ready)
        return out, verdict

    def draft_ids(self, bs: int):
        """The next step's drafts [bs, 5] (the kernel's Markov block sampling, mids), as ``Drafter.propose`` returns."""
        return (
            self.scratch["mids"]
            .view(torch.int32)[: 8 * 5]
            .view(8, 5)[:bs]
            .to(torch.int64)
        )
