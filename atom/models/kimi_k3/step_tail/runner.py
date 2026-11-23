# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One TP rank's step tail: its buffers, the peer regions, the per-width
builds and the launch after the target forward.

What ``dispatch.StepTailDecodeModel`` and, through it, ``ModelRunner`` ask of
it on a served step:

- ``launch_verify(rows, qlen, hidden, input_ids)``: the head + verify launch on
  the current stream;
- ``verify_outputs(bs)``: the rejection sampler's ``(output_token_ids
  [bs, qlen] int32, -1 past the emitted tokens; verdict [3, bs] int32:
  accepted drafts, rejected drafts, anchor row)``;
- ``output_stream``: the stream those are ready on.

And of the draft block's epilogue (``dispatch``'s ``head_and_sample`` wrap):

- ``prepare_draft(requests, drafts)``: whether the block's shape has an agreed
  draft launch (the draft LM head and the greedy Markov sampling);
- ``launch_draft(requests, drafts, hidden, anchors)``: that launch on the
  current stream (a recording takes it), returning the draft ids
  ``[requests, drafts]`` int32.

A cross-rank wait that gives up is fatal (``check_waits``, a RuntimeError):
the launch went on with a stale peer line and its ids may be wrong. The
kernels' count reaches the host behind each verify launch without a sync and
is settled before the engine reads that step's ids on the host: with deferred
output in the next step (at its offer if the count has landed, at the latest
in its ``verify_outputs``, where the engine is about to wait for the same
launch), else in the step's own ``verify_outputs``. A draft block's give-up
is in the next verify launch's count.

Construction and ``prepare`` are collective over the TP group (the peer
mapping, each width's agreed build): every rank reaches them with the same
arguments, outside a graph capture.
"""

import logging

import torch

from atom.models.kimi_k3.step_tail import layout as L
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")


def step_outputs(outtok, verdict, requests: int, qlen: int, bs: int):
    """The kernel's flat output buffers as the rejection sampler lays its
    results out, cut to the step's ``bs`` scheduled requests: ids
    ``[bs, qlen]``, verdict ``[3, bs]``."""
    return (
        outtok[: requests * qlen].view(requests, qlen)[:bs],
        verdict[: 3 * requests].view(3, requests)[:, :bs],
    )


def head_weight(model) -> torch.Tensor:
    """This rank's LM head rows, as the kernel reads them in place."""
    head = getattr(model, "language_model", model).lm_head
    weight = getattr(head, "weight", None)
    if weight is None or getattr(head, "bias", None) is not None:
        raise MonoUnsupported("LM head without a plain weight (or with a bias)")
    if (
        tuple(weight.shape) != (L.VR, L.HIDDEN)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
    ):
        raise MonoUnsupported(
            f"LM head shard {tuple(weight.shape)} {weight.dtype} "
            f"(the tail reads [{L.VR}, {L.HIDDEN}] bf16)"
        )
    return weight


class StepTailRunner:
    def __init__(self, model, drafter=None) -> None:
        from aiter.dist.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )

        from atom.models.kimi_k3.mono.symm_peers import map_symm_peers

        tp = get_tensor_model_parallel_world_size()
        if tp != L.TP:
            raise MonoUnsupported(f"TP {tp} (the step tail is TP{L.TP})")
        self.rank = get_tensor_model_parallel_rank()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.weight = head_weight(model)
        self._group = get_tp_group().cpu_group
        # the transport the model's in-kernel all-reduces use across trays
        # (HIP IPC reaches only the local one); None on every rank otherwise
        self.peers = map_symm_peers(L.PEER_BYTES, 1, self._group, tp, self.device)
        if self.peers is None:
            raise MonoUnsupported("no symmetric peer memory for the step tail's lines")
        zeros = torch.zeros
        self.cnt = zeros(L.CNT_WORDS, dtype=torch.int32, device=self.device)
        self.rec = zeros(L.REC_BYTES, dtype=torch.uint8, device=self.device)
        self.outtok = zeros(
            L.MAX_REQ * L.MAX_QLEN, dtype=torch.int32, device=self.device
        )
        self.verdict = zeros(3 * L.MAX_REQ, dtype=torch.int32, device=self.device)
        self.drafts = zeros(
            L.MAX_REQ * L.MAX_DRAFTS, dtype=torch.int32, device=self.device
        )
        # a Markov round's x_k, CTA 0 to the rank's other CTAs
        self.round_ids = zeros(L.MAX_REQ, dtype=torch.int32, device=self.device)
        self.tab = zeros(len(L.POINTERS), dtype=torch.int64, device=self.device)
        for name, tensor in (
            ("w", self.weight),
            ("rec", self.rec),
            ("cnt", self.cnt),
            ("peers", self.peers.peers),
            ("outtok", self.outtok),
            ("verdict", self.verdict),
            ("drafts", self.drafts),
            ("mbest", self.round_ids),
        ):
            self.tab[L.POINTERS.index(name)] = tensor.data_ptr()
        self.builds: dict[tuple[int, int], object] = {}
        self._refused: dict[tuple[int, int], str] = {}
        # the draft launch's builds, by (requests, drafts); its Markov tables
        # are bound with the first one
        self._drafter = drafter
        self.markov_rank = 0
        self.draft_builds: dict[tuple[int, int], object] = {}
        self._draft_refused: dict[tuple[int, int], str] = {}
        self._step = (0, 0)
        # the verdict is ready after the head + verify launch: its copies run
        # on a stream of their own, so the output copy does not wait for what
        # the step queues after the launch (the draft's context KV)
        self._ev_verify = torch.cuda.Event()
        self.verify_ready = torch.cuda.Event()
        self.output_stream = self._verify_stream = torch.cuda.Stream()
        # the kernels' count of cross-rank waits that gave up (and the last
        # site), copied to the host behind each verify launch without a sync
        self._wait_buf = torch.empty(2, dtype=torch.int32, pin_memory=True)
        self._wait_event = torch.cuda.Event()
        self._wait_pending = False
        self._gave_up: tuple[int, int] | None = None
        # whether the engine reads a step's ids on the host a step late
        # (``scheduler.defers_output``): the dispatch says so each served step
        self.outputs_deferred = False
        logger.info("Kimi-K3 step tail bound (TP%d rank %d)", tp, self.rank)

    def prepare(self, rows: int, qlen: int) -> bool:
        """Whether this width has a build, building it on first sight (a
        collective: never inside a capture)."""
        key = (rows, qlen)
        if key in self.builds:
            return True
        if key in self._refused or torch.cuda.is_current_stream_capturing():
            return False
        from atom.mono.runtime.consensus import bind_agreed

        try:
            bind_agreed(lambda: self._build(rows, qlen), self._group)
        except MonoUnsupported as why:
            self._refused[key] = str(why)
            logger.warning(
                "Kimi-K3 step tail: no build for %d rows of %d a request (%s)",
                rows,
                qlen,
                why,
            )
            return False
        return True

    def _build(self, rows: int, qlen: int) -> None:
        from atom.models.kimi_k3.step_tail import head_dispatch
        from atom.models.kimi_k3.step_tail import kernels as K
        from atom.models.kimi_k3.step_tail.sources import SOURCES
        from atom.mono.runtime.compile import compile_only, launcher_resources

        why = L.step_refusal(rows, qlen)
        if why:
            raise MonoUnsupported(why)
        try:
            splits = head_dispatch.head_splits(rows)
        except ValueError as err:
            raise MonoUnsupported(str(err)) from err
        launch = K.build_head_verify(rows, qlen, splits, sources=SOURCES)
        with compile_only():
            launch(0, 0, 0, 0)
        for res in launcher_resources(launch):
            if res["scratch"] or res["vgpr_spill"] or res["sgpr_spill"]:
                raise MonoUnsupported(f"resident hazard in {res}")
        self.builds[(rows, qlen)] = launch
        logger.info(
            "Kimi-K3 step tail: %s built (rank %d)",
            K.head_verify_name(rows, qlen, splits),
            self.rank,
        )

    def launch_verify(self, rows: int, qlen: int, hidden, input_ids) -> None:
        """The head + verify launch of a ``rows``-row step on the current
        stream: ``hidden`` the target's final normed rows, ``input_ids`` the
        step's input ids (the drafts it verifies)."""
        self.check_waits(wait=False)
        self.builds[(rows, qlen)](
            self.tab.data_ptr(),
            hidden.data_ptr(),
            input_ids.data_ptr(),
            self.rank,
            stream=torch.cuda.current_stream(),
        )
        self._ev_verify.record()
        self._step = (rows // qlen, qlen)

    def verify_outputs(self, bs: int):
        """The served step's ``(output ids, verdict)`` for the engine. First the
        launches whose give-up count is already queued are settled
        (``check_waits``): with deferred output that is the previous step's,
        whose ids the engine is about to wait for anyway; without it this
        step's own count is settled too, where the engine is about to read the
        ids."""
        self.check_waits(wait=True)
        requests, qlen = self._step
        cur = torch.cuda.current_stream()
        with torch.cuda.stream(self._verify_stream):
            self._verify_stream.wait_event(self._ev_verify)
            out, verdict = (
                t.clone()
                for t in step_outputs(self.outtok, self.verdict, requests, qlen, bs)
            )
            # the give-up count as of this launch (the draft block before it
            # included), to the host behind the outputs
            self._wait_buf.copy_(
                self.cnt[L.CNT_TIMEOUT : L.CNT_TIMEOUT + 2], non_blocking=True
            )
            self._wait_event.record()
            self._wait_pending = True
            self.verify_ready.record()
        out.record_stream(cur)
        verdict.record_stream(cur)
        cur.wait_event(self.verify_ready)
        if not self.outputs_deferred:
            self.check_waits(wait=True)
        return out, verdict

    def check_waits(self, wait: bool) -> None:
        """Fail the step if a cross-rank wait gave up: that launch went on with
        a stale peer line. Looks at the count once it has reached the host;
        ``wait`` blocks for one still queued. Once seen, every call raises."""
        if (
            self._gave_up is None
            and self._wait_pending
            and (wait or self._wait_event.query())
        ):
            self._wait_event.synchronize()
            self._wait_pending = False
            count, site = self._wait_buf.tolist()
            if count:
                self._gave_up = (count, site)
        if self._gave_up is not None:
            count, site = self._gave_up
            raise RuntimeError(
                f"Kimi-K3 step tail: {count} cross-rank waits gave up so far "
                f"(rank {self.rank}, last site {site})"
            )

    # ---- the draft block's head and Markov sampling ------------------------

    def _bind_markov(self) -> int:
        """The draft's Markov tables as the draft launch reads them in place
        (replicated, whole); returns their rank."""
        head = getattr(getattr(self._drafter, "model", None), "markov_head", None)
        if head is None:
            raise MonoUnsupported("no DSpark Markov head")
        w1, w2 = head.markov_w1.weight, head.markov_w2.weight
        rank = int(w1.shape[1])
        for name, weight in (("W1", w1), ("W2", w2)):
            if (
                tuple(weight.shape) != (L.VOCAB, rank)
                or weight.dtype != torch.bfloat16
                or not weight.is_contiguous()
            ):
                raise MonoUnsupported(
                    f"Markov {name} {tuple(weight.shape)} {weight.dtype} "
                    f"(the draft launch reads [{L.VOCAB}, {rank}] bf16)"
                )
            self.tab[L.POINTERS.index(name.lower())] = weight.data_ptr()
        return rank

    def prepare_draft(self, requests: int, drafts: int) -> bool:
        """Whether a block of ``requests`` x ``drafts`` has a draft launch,
        building it on first sight (a collective: never inside a capture)."""
        key = (requests, drafts)
        if key in self.draft_builds:
            return True
        if key in self._draft_refused or torch.cuda.is_current_stream_capturing():
            return False
        from atom.mono.runtime.consensus import bind_agreed

        try:
            bind_agreed(lambda: self._build_draft(requests, drafts), self._group)
        except MonoUnsupported as why:
            self._draft_refused[key] = str(why)
            logger.warning(
                "Kimi-K3 step tail: no draft launch for %d requests of %d drafts (%s)",
                requests,
                drafts,
                why,
            )
            return False
        return True

    def _build_draft(self, requests: int, drafts: int) -> None:
        from atom.models.kimi_k3.step_tail import head_dispatch
        from atom.models.kimi_k3.step_tail import markov_kernels as MK
        from atom.models.kimi_k3.step_tail import markov_layout as M
        from atom.models.kimi_k3.step_tail.sources import MARKOV_SOURCES
        from atom.mono.runtime.compile import compile_only, launcher_resources

        if not self.markov_rank:
            self.markov_rank = self._bind_markov()
        table = self.tab.tolist()
        unset = [
            name
            for name in ("w", "w1", "w2", "rec", "cnt", "peers", "drafts", "mbest")
            if not table[L.POINTERS.index(name)]
        ]
        if unset:
            raise MonoUnsupported(f"draft launch pointers unset: {unset}")
        why = M.draft_refusal(requests, drafts, self.markov_rank)
        if why:
            raise MonoUnsupported(why)
        try:
            splits = head_dispatch.head_splits(requests * drafts)
        except ValueError as err:
            raise MonoUnsupported(str(err)) from err
        launch = MK.build_draft_markov(
            requests, drafts, self.markov_rank, splits, sources=MARKOV_SOURCES
        )
        with compile_only():
            launch(0, 0, 0, 0)
        for res in launcher_resources(launch):
            if res["scratch"] or res["vgpr_spill"] or res["sgpr_spill"]:
                raise MonoUnsupported(f"resident hazard in {res}")
        self.draft_builds[(requests, drafts)] = launch
        logger.info(
            "Kimi-K3 step tail: %s built (rank %d)",
            MK.draft_markov_name(requests, drafts, self.markov_rank, splits),
            self.rank,
        )

    def launch_draft(self, requests: int, drafts: int, hidden, anchors):
        """The draft launch on the current stream: ``hidden`` the draft's
        final-normed rows ``[requests * drafts, HIDDEN]`` (row = request *
        drafts + position), ``anchors`` the requests' anchor ids. Returns the
        drafts ``[requests, drafts]`` int32, a copy (the next launch rewrites
        the buffer)."""
        self.draft_builds[(requests, drafts)](
            self.tab.data_ptr(),
            hidden.data_ptr(),
            anchors.data_ptr(),
            self.rank,
            stream=torch.cuda.current_stream(),
        )
        return self.drafts[: requests * drafts].view(requests, drafts).clone()
