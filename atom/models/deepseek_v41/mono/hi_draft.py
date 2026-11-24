# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DSpark draft on the hi path (``HiDraft``, a part of ``hi_runner.HiLayers``): the draft's context-KV write
(``CTX`` a width: main_proj, the stages' wkv and norms over the target's DSpark taps, written into ATOM's draft rings
inside the target graph) and the block draft (three chained stage launches and the Markov head: ``draft_run``), with
their peer groups, buffers and weights (the stages' packed from ATOM's draft model, ``hi_weights``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import triton

from atom.models.deepseek_v41.mono import hi_bridge as ml
from atom.models.deepseek_v41.mono import hi_weights as hw
from atom.models.deepseek_v41.mono.hi_compose import _DraftLaunch
from atom.models.deepseek_v41.mono.hi_gate import log_once

logger = logging.getLogger("atom")

HC, MIX, HEAD_DIM = ml.HC, ml.MIX, ml.HEAD_DIM
CTX_LAUNCH = 42  # the DSpark context write's launch id (40: the head, 41: the Engram kv launch; its own peer group)


# ---- shared with the runner (``hi_runner``)


class _Arena:
    """One scratch arena the programs of a step counter share (``emit.kernel``'s kernel ABI): every launch's tags are
    new and the programs run one after another on one stream, so each sees only other launches' stale pairs, which
    carry other tags (their cross-launch hand-offs are peer regions). Sized for the largest program, uncached, 2 MiB
    aligned, its dual-copy word calibrated once (a run's copy depends on the arena's page only). Two arenas: the
    target's (layers, heads, the Engram kv launch, CTX: the step counter ``step``) and the draft's (its own counter).
    """

    def __init__(self, name: str, kerns, dev) -> None:
        from .hi.plan.program import DUAL_PAGE
        from .hi.runtime import uncached
        from .hi.runtime.dual import scratch_word

        progs = [k.program for k in kerns]
        self.nbytes = max([p.scratch_bytes for p in progs] + [256])
        raw = uncached.zeros(self.nbytes + DUAL_PAGE, dev)
        shift = -raw.data_ptr() % DUAL_PAGE
        self.raw, self.scratch = raw, raw[shift : shift + self.nbytes]
        dual = max(progs, key=lambda p: p.dual_bytes)
        self.word = (
            scratch_word(self.scratch, dual.dual_bytes, name) if dual.dual_on else None
        )
        log_once(
            logging.INFO,
            "V4.1 mono hi: the %s programs' scratch arena: %d programs, %.1f MiB",
            name,
            len(progs),
            raw.numel() / 2**20,
        )


def _buffers(kerns: dict, arena: _Arena, dev, step) -> dict:
    """``kerns``' buffers (key -> ``Buffers`` on the step counter ``step``): the arena's scratch, each program its own
    private buffer (the kernel ABI's ``priv``: the dual-copy word, the atomics' regions).
    """
    from .hi.runtime.kernel import Buffers

    out = {}
    for key, kern in kerns.items():
        own = kern.buffers(
            device=dev, scratch=arena.scratch, uncached=True, word=arena.word
        )
        out[key] = Buffers(scratch=own.scratch, step=step, private=own.private)
    return out


@dataclass
class _Streams:
    """A launch's output streams (the next seam's R and gates, the MoE output)."""

    r_f: torch.Tensor  # bf16 [R, 4, H]
    gate_f: torch.Tensor  # f32 [R, 24]
    ffn_out: torch.Tensor  # bf16 [R, H]


def _end_step(peers) -> None:
    """A step's last launch on a peer group is issued: the hi runtime's step phase (``PeerGroup.end_step``: after an odd
    number of launches the next step's first launch takes the other parity, so it cannot land on this step's last
    slots)."""
    if peers is not None:
        peers.end_step()


class HiDraft:
    """The draft's methods of ``hi_runner.HiLayers`` (module docstring); its state is the runner's."""

    # ------------------------------------------------------------ the DSpark draft's context-KV write

    def _ctx_build(self, draft, dev) -> int:
        """``CTX`` a width (module docstring), its own peer group, the weights (the hi runtime's ``draft.pack`` of
        ATOM's draft model ``draft``'s tensors: this rank's main_proj columns, the stages' wkv, the norms) -> bytes
        held."""
        from .hi.runtime.peer import PeerGroup

        D = self.D
        pb = max(k.program.peer_bytes for k in self.ctx_programs.values())
        self.cx_peers = PeerGroup(pb, self._cpu_group) if self.tp > 1 and pb else None
        self.cx_bufs = _buffers(self.ctx_programs, self.arena, dev, self.step)
        self.cx_w = D.pack(
            hw.ctx_tensors(draft, D.STAGES), self.tp, self.rank, "shard", device=dev
        )
        self._packed.append(self.cx_w)
        self.cx_geo = torch.zeros(D.STAGES, D.GEO, dtype=torch.int32, device=dev)
        self.cx_geo_key, self.cx_captured = None, False
        return sum(t.numel() * t.element_size() for t in self.cx_w.values())

    def bind_draft(self, drafter) -> None:
        """The DSpark proposer (``install_mono_decode``): the draft's context layers (their window rings; their count
        and the norms' eps checked at install, ``hi_runner.check``)."""
        self.cx_drafter = drafter
        self.cx_layers = [int(a.spec.layer_id) for a in drafter.model.context_layers]

    def _ctx_taps(self) -> list | None:
        """ATOM's aux buffers of the tapped layers (the draft's ``dspark_target_layer_ids`` order; bound from the
        drafter on first use), or None without a drafter."""
        aux = self.aux
        if aux is None and self.cx_drafter is not None:
            from atom.models.deepseek_v41.dspark import DeepseekV41DSpark
            from atom.mono.runtime.aux_taps import bind_aux_taps

            aux = self.aux = bind_aux_taps(
                self.cx_drafter,
                self.h_in0.shape[-1],
                DeepseekV41DSpark._target_layer_input,
            )
        if aux is None or len(aux.buffers) != self.D.TAPS:
            return None
        return list(aux.buffers)

    def _ctx_rings(self, cache) -> tuple[list, list]:
        """Per draft stage: (``cx.geo`` row, the ring base address: ATOM's pool at the layer's window start)."""
        if self.cx_layers is None:
            raise RuntimeError("V4.1 mono hi draft: no DSpark draft bound (bind_draft)")
        pool = cache.pool
        if (
            pool.dim() != 2
            or pool.shape[-1] != HEAD_DIM
            or not pool.is_contiguous()
            or pool.dtype != torch.bfloat16
        ):
            raise RuntimeError(
                f"pool {tuple(pool.shape)} {pool.dtype}: not the bf16 [rows, 512] view"
            )
        geo, rings = [], []
        for layer in self.cx_layers:
            w = cache.geometry.window(layer, cache.num_pages)
            geo.append([int(cache.num_slots), int(w.slot_rows), int(w.ring_slots), int(w.ring_stride), int(w.run_rows),
                        0, 0, 0])  # fmt: skip
            rings.append(
                pool.data_ptr()
                + int(w.ring_start) * pool.stride(0) * pool.element_size()
            )
        return geo, rings

    def _ctx_table(self, rows: int, cache, step):
        """The pointer table of ``CTX`` at width ``rows`` for this pool / these step buffers (built eagerly the
        first time: a captured graph replays the table its capture used, the buffers' addresses persistent).
        """
        taps = self._ctx_taps()
        if taps is None:
            return None
        # (in the key: a cache of other geometry at the same address)
        geo, rings = self._ctx_rings(cache)
        key = (rows, cache.pool.data_ptr(), *(t.data_ptr() for t in taps), step.batch_ids.data_ptr(),
               step.slots.data_ptr(), repr(geo))  # fmt: skip
        table = self.cx_tables.get(key)
        if table is not None:
            return table
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "V4.1 mono hi draft: CTX's pointer table first needed inside a graph capture"
            )
        self._set_cx_geo(geo)
        io = {"pos": self.pos, **self.rope_d, "cx.geo": self.cx_geo}
        io.update({f"cx.tap{i}": t for i, t in enumerate(taps)})
        io.update({f"cx.ring{s}": r for s, r in enumerate(rings)})
        io.update({"cx.bids": step.batch_ids, "cx.slots": step.slots})
        kern = self.ctx_programs[rows]
        missing = {
            a
            for a in set(kern.program.args) - set(self.cx_w) - set(io)
            if not a.startswith("_")
        }
        if missing:
            raise KeyError(f"CTX: no pointer for {sorted(missing)}")
        table = kern.table({**self.cx_w, **io}, device=self.dev, peers=self.cx_peers)
        self.cx_tables[key] = table
        return table

    def _set_cx_geo(self, geo) -> None:
        """The draft rings' geometry (``cx.geo``, one device tensor every CTX / draft table reads). A new geometry
        before any graph captured a launch reading it (the engine's warm-up / profile forwards run on a cache of
        their own: a batch width's two-request dummy step takes the hi path there) drops the tables of the
        earlier cache; after a capture it raises (a replay would read the moved rings).
        """
        if self.cx_geo_key not in (None, repr(geo)):
            if self.cx_captured:
                raise RuntimeError(
                    f"V4.1 mono hi draft: the draft rings moved ({self.cx_geo_key} -> {geo})"
                )
            self.cx_tables.clear()
            getattr(self, "dr_tables", {}).clear()
        self.cx_geo.copy_(torch.tensor(geo, dtype=torch.int32))
        self.cx_geo_key = repr(geo)

    def _ctx_launch(self, rows: int, table) -> None:
        bufs = self.cx_bufs[rows]
        bufs.launched.clear()  # one CTX launch a step (its own Buffers)
        self.cx_captured |= torch.cuda.is_current_stream_capturing()
        if self.cx_peers is not None:
            self.cx_peers.last_parity = (
                None  # its own group; the previous CTX launch is a step back
            )
        self.ctx_programs[rows].launch(
            table, bufs, layer=CTX_LAUNCH, stream=torch.cuda.current_stream()
        )
        # (one launch a step: the step phase flips, the next CTX launch has the other parity)
        _end_step(self.cx_peers)

    def _ctx_after_target(self, cache, step, rows: int) -> None:
        """A served step's end (``end_step``): ``CTX`` inside the target graph, after the end-step launch (the taps
        are written there), before the verdict. Every served step writes the draft's context rows here: the DSpark
        proposer skips ATOM's write on it (``AttentionMetaData.served_by_round``).
        """
        if rows not in self.ctx_programs or step.replay is not None:
            raise RuntimeError(
                f"V4.1 mono hi draft: a served step of {rows} rows without the draft's context write"
            )
        table = self._ctx_table(rows, cache, step)
        if table is None:
            raise RuntimeError(
                "V4.1 mono hi draft: a served step without the draft's taps bound"
            )
        self._ctx_launch(rows, table)

    # ------------------------------------------------------------ the block draft

    def _draft_build(self, model, draft, dev) -> int:
        """The draft's stages 0-2 and its head (hi ``draft.stage_program`` / ``head_program``: chained, a peer
        group of their own, launch ids ``DRAFT_LAUNCH``), their buffers and weights (the stages' from ATOM's draft
        model ``draft``, ``mtp[s]``, 128 experts; the head's ``norm`` = ``mtp[2].norm``, its LM head the verify head's
        packed tensor or this rank's ``model.head.weight`` packed the same way). -> bytes held.
        """
        from .hi.runtime.peer import PeerGroup
        from .hi.v41 import head as H

        D, W = self.D, self.dr_width
        self.dr_peers = PeerGroup(
            max(k.program.peer_bytes for k in self.dr_kerns.values()), self._cpu_group
        )
        self.dr_step = torch.zeros(1, dtype=torch.int32, device=dev)
        self.dr_bufs = _buffers(
            self.dr_kerns,
            _Arena("draft", self.dr_kerns.values(), dev),
            dev,
            self.dr_step,
        )
        Hd, rows = self.h_in0.shape[-1], self.dr_bmax * W

        def z(*shape, dt=torch.bfloat16):
            return torch.zeros(*shape, dtype=dt, device=dev)

        # the block streams of every request (request b's rows b W ..), each request's first position
        self.dr_h_in, self.dr_gin = z(rows, HC, Hd), z(rows, MIX, dt=torch.float32)
        self.dr_gin[:, 0] = 1.0  # the block streams' one-hot pre (build_block_state's)
        self.dr_pos = z(self.dr_bmax, dt=torch.int32)
        self.dr_sets = [
            _Streams(z(rows, HC, Hd), z(rows, MIX, dt=torch.float32), z(rows, Hd))
            for _ in range(2)
        ]
        # stages 0-2's output streams (the next seam's R / the head's)
        self.dr_out = [z(rows, HC, Hd) for _ in range(3)]
        self.dr_ra0 = z(rows, HC, Hd)  # stage 0's seam-A R (unread)
        # request b's draft ids at row b
        self.mk_ids = z(self.dr_bmax, W, dt=torch.int32)
        self.dr_requests = 1
        log_once(
            logging.INFO,
            "V4.1 mono hi: block draft plans by requests: %s",
            {n: p.describe() for n, p in self.dr_plans.items()},
        )
        self.dr_w, nbytes, plain = [], 0, 0
        inter = model.config.moe_intermediate_size
        for s in range(3):
            block = draft.mtp[s]
            w, raw = hw.block_tensors(block, f"mtp.{s}.")
            moe = hw.expert_packs(block.ffn.experts, inter, self.tp)
            packed = D.pack_stage(
                s, w, raw, self.tp, self.rank, device=dev, local=True, moe=moe
            )
            del w, raw, moe
            plain += hw.alias_plain(block, packed)
            self.dr_w.append(packed)
            nbytes += sum(
                t.numel() * t.element_size()
                for t in packed.values()
                if not getattr(t, "_atom_alias", False)
            )
        log_once(
            logging.INFO,
            "V4.1 mono hi: the draft stages' weights packed from ATOM's draft tensors; their routed experts' "
            "and %d plain tensors read in place",
            plain,
        )
        norm = draft.mtp[2].norm.weight.detach()
        head_w = getattr(self, "head_w_packed", None)
        if head_w is None:
            head_w = H.pack(
                model.norm.weight.detach(), model.head.weight.detach(), device=dev
            )["head.w"]
            nbytes += head_w.numel() * head_w.element_size()
        self.dr_head_w = {
            "norm.g": norm.bfloat16().contiguous().to(dev),
            "mk.w": head_w,
        }
        self.dr_tables = {}
        self._packed.extend(self.dr_w)
        return nbytes

    def _draft_tables(self, cache, anchor_ids, slots, mk_ids, plan=None):
        """The four launches' pointer tables for this pool, these staged buffers and this request's ids row (built
        eagerly the first time, as ``_ctx_table``)."""
        # (in the key: a cache of other geometry at the same address)
        geo, rings = self._ctx_rings(cache)
        key = (
            cache.pool.data_ptr(),
            anchor_ids.data_ptr(),
            slots.data_ptr(),
            mk_ids.data_ptr(),
            repr(geo),
            plan.n if plan is not None else 1,
        )
        tables = self.dr_tables.get(key)
        if tables is not None:
            return tables
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "V4.1 mono hi draft: the draft's tables first needed inside a graph capture"
            )
        self._set_cx_geo(geo)
        markov = self.cx_drafter.model.mtp[-1].markov_head
        w1, w2 = markov.markov_w1.weight.detach(), markov.markov_w2.weight.detach()
        if (
            w1.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or not (w1.is_contiguous() and w2.is_contiguous())
        ):
            raise ValueError(
                f"V4.1 mono hi draft: Markov tables {w1.dtype} / {w2.dtype}, want contiguous bf16"
            )
        plan = plan or self.dr_plans[1]
        tables, W = [], self.dr_width
        for d in plan.launches:
            if d.stage < 3:
                s = d.stage
                src, dst = self.dr_sets[(s + 1) % 2], self.dr_sets[s % 2]
                io = {"pos": self.dr_pos, **self.rope_d, "dr.ring": rings[s], "dr.geo": self.cx_geo[s],
                      "dr.slot": slots, "r_f": dst.r_f, "gate_f": dst.gate_f, "ffn_out": dst.ffn_out,
                      "r_a": self.dr_out[s - 1] if s else self.dr_ra0}  # fmt: skip
                io.update(
                    {"h_in": self.dr_h_in, "gin": self.dr_gin}
                    if s == 0
                    else {"h_in": src.r_f, "gin": src.gate_f}
                )
                tables.append(self._draft_table(d, {**self.dr_w[s], **io}))
                continue
            # (a head of requests r0 ..: their rows)
            last, a = (
                self.dr_sets[0],
                d.r0 * W,
            )
            io = {**self.dr_head_w, "r_f": last.r_f[a:], "gate_f": last.gate_f[a:], "ffn_out": last.ffn_out[a:],
                  "r_h": self.dr_out[2][a:], "mk.w1": w1, "mk.w2": w2, "mk.anchor": anchor_ids[d.r0 :],
                  "mk.ids": mk_ids[d.r0 :] if mk_ids.dim() > 1 else mk_ids}  # fmt: skip
            tables.append(self._draft_table(d, io))
        self.dr_tables[key] = tables
        return tables

    def _draft_table(self, d: _DraftLaunch, pointers):
        kern = d.kern
        missing = {
            a for a in set(kern.program.args) - set(pointers) if not a.startswith("_")
        }
        if missing:
            raise KeyError(f"draft launch {d.key}: no pointer for {sorted(missing)}")
        return kern.table({n: pointers[n] for n in kern.program.args if n in pointers}, device=self.dev,
                          peers=self.dr_peers)  # fmt: skip

    def draft_serves(self, requests: int, width: int) -> bool:
        """Host-side, TP-uniform: whether the block draft of ``requests`` requests at ``width`` runs on the hi path (up
        to ``dr_bmax``)."""
        return 1 <= requests <= self.dr_bmax and width == self.dr_width

    def draft_run(
        self, cache, anchor_ids, anchor_rows, anchor_positions, slots, noise_row
    ) -> None:
        """The block draft on the hi path (inside the draft graph's capture) by the plan of its request count
        (``_draft_plan``): each request's block streams and first position (``_draft_begin_kernel``, its anchor's
        embedding row, with the launch-tag bump), then stages 0-2 and the head(s); request b's ids land in
        ``mk_ids[b]``. ``anchor_ids`` / ``anchor_positions`` / ``slots``: a row a request; ``anchor_rows``: their
        embedding rows (ATOM's lookup).
        """
        requests = int(anchor_ids.numel())
        self.dr_requests = requests
        self.cx_captured |= torch.cuda.is_current_stream_capturing()
        W, Hd = self.dr_width, self.dr_h_in.shape[-1]
        ids, apos, sl = anchor_ids.view(-1), anchor_positions.view(-1), slots.view(-1)
        stream = torch.cuda.current_stream()
        self.dr_peers.last_parity = (
            None  # its own group; the previous draft is a step back
        )
        plan = self.dr_plans[requests]

        def launches(dl, tables):
            for d, table in zip(dl, tables, strict=True):
                bufs = self.dr_bufs[d.key]
                bufs.launched.clear()  # (the begin kernel bumped the draft's step: these are new tags)
                d.kern.launch(table, bufs, layer=d.lid, stream=stream)

        # one request at a time (its own tags: each begin kernel bumps the step)
        if plan.per_request:
            for b in range(requests):
                tables = self._draft_tables(
                    cache, ids[b : b + 1], sl[b : b + 1], self.mk_ids[b]
                )
                ml._draft_begin_kernel[(W, triton.cdiv(Hd, 1024))](
                    anchor_rows[b], noise_row, self.dr_h_in, apos[b : b + 1], self.dr_pos, self.dr_step, Hd,
                    self.draft_max_pos, NHC=HC, BLOCK=1024,
                )  # fmt: skip
                launches(plan.launches, tables)
            return
        tables = self._draft_tables(cache, ids, sl, self.mk_ids, plan)
        # request b's streams (rows b W ..) and first position; one tag bump a draft
        for b in range(requests):
            ml._draft_begin_kernel[(W, triton.cdiv(Hd, 1024))](
                anchor_rows[b], noise_row, self.dr_h_in[b * W :], apos[b : b + 1], self.dr_pos[b:], self.dr_step, Hd,
                self.draft_max_pos, NHC=HC, BLOCK=1024, BUMP=int(b == 0),
            )  # fmt: skip
        launches(plan.launches, tables)
