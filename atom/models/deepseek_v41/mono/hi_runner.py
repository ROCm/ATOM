# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's high-interactivity (hi) monokernels inside ATOM's V4.1 mono decode
(``ATOM_DSV41_MONO_HI``, MI355X / gfx950, TP2 or TP4).

On a step this hook serves, every one of V4.1's 40 decoder layers runs its
hi layer program (the ``mono/hi`` package: ``layer.window_layer_blocks``, one
persistent launch a layer at TP; the plan: ``hi_plan.plan_layers``), then
the hi verify head.
Prefill and every step the hook does not serve stay ATOM's (its own mono
decode, K1 / K2).

One served step:

1. ``begin_step`` (one launch, ``hi_bridge._begin_step_kernel``): the
   launch-tag step bump, ATOM's token embedding rows of the step expanded into
   layer 0's streams, the window rings' import from ATOM's pool, each
   request's first position, its request words (batch id, slot, liveness:
   the in-place programs') and the indexer / candidate bounds, and the Engram history and cursor candidates;
   then the Engram token-only halves (one launch, ``engram_kv_stages``) looked
   up from ATOM's page-locked host shards.
2. ``layer``: the 40 layer launches, chained (a layer pushes its MoE partials,
   the next one's seam A sums them), then the verify head (greedy, or one draw
   a sampled row; ``head.py``) on the last layer's pushed partials.
3. ``end_step`` (one launch, ``_end_step_kernel``): the ring export (the
   step's ring rows back into ATOM's pool), the DSpark taps of layers 37-39, the last layer's gates planar and
   ATOM's collapse into the runner's output rows; then the DSpark draft's
   context-KV write (``CTX``, launch ``CTX_LAUNCH``) and the post-verify
   (``post_verify``: verdict and Engram cursor commit), all inside
   the target graph.

The block draft runs on the hi path too (``draft_run``: three chained stage
launches and the Markov head a step, from ``MonoDraftModel.block_backbone``).

**Widths.** ``hi_compose.ROWS`` lists the step widths served: a width of
B x 6 rows serves B requests verifying 6 rows each (DSpark's 5 drafts and the
anchor; a bs-B graph whose query bucket is 6, ATOM laying request b's rows out
at b x 6), the hi runtime's ``Scenario(B, 6, tp)``. Up to ``MAX_BATCH`` requests a
width has its own programs; past it ``hi_compose.BANDS`` (6 and 8) runs
the smallest band of at least its requests (``hi_bands``: one hi
``Program.dyn_rows`` build a (kind, band), launched with the step's live words
after the step counter). Per request, by its batch slot b, the hi-side
buffers hold a slot each (rings ``[layer, b, 128, 512]``, ``pos[b]``, the
request words); the bridge imports and exports each request's ring from its
own batch id, slot and first position.

**ATOM's storage in place.** The owners' compressed KV (the main rows of
ATOM's pool), their FP4 index keys (ATOM's index plane) and the ratio-2
compressor state (ATOM's compress rings) are read and written by the programs
where ATOM keeps them (``hi.device.gfx950.paged``), through the step's block
table: nothing of theirs is imported or exported, and a request's first served
step after a prefill costs no copy. Every store into that storage is masked by
its row's liveness (the begin-step's request words), so a padded request or a
band's dead one writes none of it. The storage's addresses are bound into the
programs' pointer tables when the first step sees the KV cache
(``_bind_cache``).

**Incremental ring import.** A served step that continues the last served step
of its request slot (the same ATOM slot, its first position at most a step's
rows past it) imports only the ring slots that step's rows wrote
(``hi_bridge``'s incremental-import comment), over a fixed grid
(``IMPORT_RING`` programs a layer); any other step imports the whole ring. The
end-step launch records the step (``imp``); a forward the hook does not serve
clears the records of its slots (``invalidate``, from
``HiDecodeModel.forward``).

**Verify head.** The hi runtime's head (the fold of the last layer's MoE output, the
collapse, the final RMSNorm, the bf16 vocab-sharded GEMV with a top-1 epilogue
and the exact cross-rank argmax with ATOM's rule: the bf16 logit, ties to the
lowest id) runs right after the last layer inside the target graph, its weights
packed from ATOM's own ``model.norm.weight`` / ``model.head.weight``. Its ids
land in a persistent int32 buffer (``head_ids``) that the in-graph post-verify
(``round.post_verify``) turns into the verify's output ids and verdict: the
forward's head output (``HiDecodeModel.compute_head``), which ATOM's runner
takes instead of logits on the verify steps that read nothing but each row's
pick. It samples (the hi runtime's ``head_blocks(sample=True)``): per row,
temperature 0 takes the greedy pick, else one draw of softmax(bf16 logits / T)
in ATOM's top-p nucleus, the rows with top_p >= 1 drawn in the dot epilogue
(``race``); Philox counters keyed by the seed, the launch tag, the row and the
id (the same draw on every rank). The rows' (T, top_p) are a persistent buffer
(``head.par``) published with each served step's metadata
(``set_head_sampling``, from the builder's routing decision,
``round.HiRound.service``). Under forced acceptance
(``--spec-decode-acceptance-length``) the head's ids feed the same synthetic
acceptance schedule (``round.post_verify``).

Graph capture: weights, buffers, programs, pointer tables and the peer groups
are bound at install, so a step only launches; the launch-tag step counter is
bumped once a step on the device inside the capture.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import triton
import triton.language as tl

from atom.model_ops.engram.host_memory import device_address
from atom.models.deepseek_v41.mono import hi_bridge as ml
from atom.models.deepseek_v41.mono import hi_plan
from atom.models.deepseek_v41.mono import hi_weights as hw
from atom.models.deepseek_v41.mono.hi_bands import LiveWords, plan_widths  # noqa: F401
from atom.models.deepseek_v41.mono.hi_compose import (  # noqa: F401 (re-exported)
    BANDS,
    DRAFT_LAUNCH,
    HEAD_MORE,
    MAX_BATCH,
    ROWS,
    SLOTS,
    HiPrograms,
    _DraftLaunch,
    _DraftPlan,
)
from atom.models.deepseek_v41.mono.hi_draft import (
    CTX_LAUNCH,
    HiDraft,
    _Arena,
    _buffers,
    _end_step,
    _Streams,
)
from atom.models.deepseek_v41.mono.hi_gate import (  # noqa: F401 (REQ_ROWS re-exported)
    BLOCK_SIZE,
    MAX_CTX,
    REQ_ROWS,
    log_once,
)
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")

HC, MIX, WINDOW, HEAD_DIM = ml.HC, ml.MIX, ml.WINDOW, ml.HEAD_DIM
ENGKV = ("engkv",)  # the Engram token-only halves' program key
ENGKV_LAUNCH = (
    41  # its launch id (odd: layer 0, launched next, has the other tag parity)
)
# the verify head's program key (``tables[HEAD, width, request]``); its launch ids: ``head_launch_ids``
HEAD = ("head",)
IMPORT_RING = 16  # the incremental ring import's programs a layer
HEAD_SEED = 0  # the sampling head's Philox seed


class HiLayers(HiDraft, HiPrograms):
    """the hi runtime's layer programs for one TP rank (``install``) and the per-step
    hand-off (module docstring)."""

    def __init__(
        self, model, atom_config, widths: list[int], max_ctx: int, engram_mapping, draft
    ):
        from aiter.dist.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )

        from .hi.runtime.peer import PeerGroup
        from .hi.v41 import layer as L

        t0 = time.time()
        self.tp = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        cfg = model.config
        # (what this rank loaded is checked, agreed over the ranks, before: ``check``)
        # the programs (``hi_compose``: the plan, the widths, every program built)
        self._cpu_group = get_tp_group().cpu_group
        HiPrograms.__init__(self, cfg, model.topology, widths, max_ctx, self.tp, self._cpu_group,
                            index_layout=atom_config.index_cache_layout)  # fmt: skip
        # lean host prep (``lean_host``) with every layer on the hi path; ``served_widths``: the widths a forward took this
        # hook's path at (a capture's included), the builder's lean-step test
        self.runs_every_layer = True
        self.lean_host, self.served_widths, self.lean_steps = False, set(), 0
        # the rows of this forward's in-graph verify (``end_step``; 0: none), its head output (``head_output``)
        self.verified = 0
        dev = self.dev = torch.device("cuda", torch.cuda.current_device())
        # (a band's programs: the band's rows)
        R, H = max(self.pwidth(s) for s in self.widths), cfg.hidden_size
        self.RB = 1 << (R - 1).bit_length()
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32

        def z(*shape, dt=bf):
            return torch.zeros(*shape, dtype=dt, device=dev)

        # ---- the streams: layer 0's input, the ping-pong pair
        self.h_in0, self.gin0 = z(R, HC, H), z(R, MIX, dt=f32)
        self.gin0[:, 0] = 1.0  # the embedding streams' one-hot pre
        self.sets = [
            _Streams(z(R, HC, H), z(R, MIX, dt=f32), z(R, H)) for _ in range(2)
        ]
        self.pre, self.post, self.comb = (
            z(R, HC, dt=f32),
            z(R, HC, dt=f32),
            z(R, HC * HC, dt=f32),
        )
        self.r_a = {i: z(R, HC, H) for i in self.mf if i in self.aux_ids}
        self.r_a_shared = z(R, HC, H)
        # ---- per step metadata, rings, rope tables
        # each request's first position (its batch slot)
        self.pos = z(self.bmax, dt=i32)
        self.bounds = z(3, self.RB, dt=i32)  # (p + 1) // 2, p + 1, ceil((p + 1) / 8)
        # [layer, request slot, ring slot, 512]
        self.rings = z(len(self.mf), self.bmax, WINDOW, HEAD_DIM)
        self.ring_layers = torch.tensor(self.mf, dtype=i32, device=dev)
        self.ring_of = {i: m for m, i in enumerate(self.mf)}
        # the block draft's first position clamp (``_draft_begin_kernel``)
        self.draft_max_pos = max_ctx - 16
        # the RoPE tables, ATOM's own in place (``RotaryEmbedding``: bf16 cos / sin [positions, 32]; ``rope_cs``):
        # the window layers' ``window_rope``, the compressing layers' ``global_rope``, the draft's its own ``rope``
        self.rope_w = hw.rope_args(model.window_rope)
        self.rope_c = hw.rope_args(model.global_rope)
        self.rope_d = hw.rope_args(draft.rope)
        # ---- ATOM's storage in place (module docstring): the KV owners (layer -> ratio) and the request words
        # (``hi_bridge.REQ_WORDS``: the page geometry, bound by ``_bind_cache``; each request's batch id, slot and
        # liveness and each row's, written by the begin-step), the owners' addresses bound by ``_bind_cache``
        self.kv_owners = {
            i: self.plan[i].ratio
            for i in self.mf
            if self.plan[i].kind == "kv+index source"
        }
        if self.bmax > ml.MAX_REQ or self.RB > ml.MAX_ROWS:
            raise ValueError(f"V4.1 mono hi: {self.bmax} requests / {self.RB} rows, the request words hold "
                             f"{ml.MAX_REQ} / {ml.MAX_ROWS}")  # fmt: skip
        from .hi.device.gfx950 import paged

        if (paged.BID, paged.SLOT, paged.LIVE, paged.ROW_LIVE, paged.REQ_WORDS) != (
            ml.REQ_BID,
            ml.REQ_SLOT,
            ml.REQ_LIVE,
            ml.ROW_LIVE,
            ml.REQ_WORDS,
        ):
            raise RuntimeError(
                "V4.1 mono hi: the request words' layout differs between the bridge and the programs"
            )
        self.req = z(ml.REQ_WORDS, dt=i32)
        self.req[ml.REQ_BID : ml.REQ_SLOT].fill_(-1)
        self._bound = None  # the bound cache's key (``_bind_cache``)
        self._late: list = (
            []
        )  # (table, index, arg name, owner) of every pointer the bind fills
        self.sel_out = {i: torch.full((R, 512), -1, dtype=i32, device=dev) for i in self.mf
                        if self.plan[i].kind in ("kv+index source", "index source")}  # fmt: skip
        self.cand = z(R, -(-self.cl[1] // 8 // 32), dt=i32)
        # ---- Engram (module docstring): the token-only halves' launch before layer 0, its inputs and outputs; the
        # table lookup from ATOM's page-locked host shards (``use_host_tables``), its history and cursor candidates
        # written by the begin-step (``_fold_args``)
        self._eng_mapping = engram_mapping  # ATOM's NgramHashMapping: the hash constants (``engram_consts``)
        # ``post_verify`` (set by ``install``); the runner and the step's input ids
        self.pv, self.runner, self.step_ids = None, None, None
        if self.eng:
            from .hi.blocks.gfx950.engram import NOUT

            self.ids = z(R, dt=i32)
            self.hist = torch.full((self.bmax, max_ctx + 16), -1, dtype=i32, device=dev)
            # layer i's ``eng.kv``
            self.eng_kv = {i: z(R, NOUT) for i in self.eng.values()}
        # the last layer's gates go planar in the end-step launch (ATOM's collapse, also there, reads them)
        self.planar_at_end = True
        # ---- the incremental ring import (module docstring): the last served step a request slot, recorded by the
        # end-step launch
        self.incr = True
        self.ring_p = IMPORT_RING
        # a request slot each
        self.imp = torch.full((self.bmax * ml.IMP_WORDS,), -1, dtype=i32, device=dev)
        peer_bytes = max(
            k.program.peer_bytes
            for k in [*self.programs.values(), *self._heads.values()]
        )
        self._cpu_group = get_tp_group().cpu_group
        self.peers = (
            PeerGroup(peer_bytes, self._cpu_group)
            if self.tp > 1 and peer_bytes
            else None
        )
        # the launch-tag step counter, every program's (one counter a peer group); with bands, the live words after it
        # (rows, spec, batch: ``Program.dyn_rows``, the band programs' live requests, staged each step by
        # ``begin_step``; the static programs read the counter only)
        self.step = z(4 if self.band_of else 1, dt=i32)
        self.live = LiveWords(self.step, self.width_plan) if self.band_of else None
        if self.peers is not None and self.band_of:
            # (the group's counter is this tensor, live words and all)
            self.peers.step = self.step
        # every program on the step counter shares one scratch arena (``hi_draft._Arena``): the layers, the heads, the
        # Engram kv launch and CTX
        self.arena = _Arena("target", [*self.programs.values(), *self._heads.values(),
                                       *self.eng_programs.values(), *self.ctx_programs.values()], dev)  # fmt: skip
        self.bufs = _buffers(self.programs, self.arena, dev, self.step)
        self.head_bufs = _buffers(self.head_programs, self.arena, dev, self.step)
        # ---- weights (the hi runtime's own packing, from ATOM's loaded tensors: ``hi_weights``) and the pointer tables
        self.tables, self._packed = {}, []
        nbytes = 0
        # the routed experts' weights, the plain bf16 / f32 tensors (``hi_weights.alias_plain``) read in place from ATOM's
        self.plain_aliased = 0
        inter = cfg.moe_intermediate_size
        for i in self.mf:
            block = model.layers[i]
            w, raw = hw.block_tensors(block, f"layers.{i}.")
            moe = hw.expert_packs(block.ffn.experts, inter, self.tp)
            packed = L.pack(i, w, raw, device=dev, tp=self.tp, rank=self.rank, local=True, moe=moe,
                            source=self.sources.get(i), engram=self.plan[i].engram)  # fmt: skip
            del w, raw, moe
            self.plain_aliased += hw.alias_plain(block, packed)
            nbytes += sum(
                t.numel() * t.element_size()
                for t in packed.values()
                if not getattr(t, "_atom_alias", False)
            )
            io = self._io(i)
            for s in self.widths:
                key = self._key(i, s)
                kern = self.programs[key]
                missing = (
                    set(kern.program.args)
                    - set(packed)
                    - set(io)
                    - {"_peers", "_rank", "_peer_top"}
                )
                if missing:
                    raise KeyError(
                        f"layer {i} ({self.plan[i].kind}): no pointer for {sorted(missing)}"
                    )
                self.tables[i, s] = kern.table(
                    {**packed, **io}, device=dev, peers=self.peers
                )
                self._late += [(self.tables[i, s], kern.program.args.index(n), n, own)
                               for n, own in self._late_args(i) if n in kern.program.args]  # fmt: skip
            self._packed.append(packed)
        if self.eng:
            nbytes += self._engram_tables(model, dev)
        nbytes += self._head_tables(model, dev)
        # ---- the DSpark draft on the hi path: its context-KV write and the block draft (module docstring)
        self.cx_tables = {}
        self.cx_layers, self.cx_drafter = None, None
        nbytes += self._ctx_build(draft, dev)
        nbytes += self._draft_build(model, draft, dev)
        self.aux = None
        logger.info(
            "V4.1 mono hi: %d layers on the hi path (TP%d rank %d), widths %s (requests x rows), ctx bucket %d, "
            "%d programs, weights %.1f GiB, peer %d B, draft context write (launch %d) and block draft (launches %s) "
            "on the hi path, residual %s, built in %.0f s",
            len(self.mf), self.tp, self.rank, self.split, max_ctx, len(self.programs), nbytes / 2**30, peer_bytes,
            CTX_LAUNCH, DRAFT_LAUNCH, {s: L.residual_layout(self.tp, rows=s) for s in self.widths}, time.time() - t0,
        )  # fmt: skip
        log_once(
            logging.INFO,
            "V4.1 mono hi: weights packed from ATOM's tensors; the routed experts' and %d plain tensors of the "
            "%d layers read in place",
            self.plain_aliased,
            len(self.mf),
        )
        torch.cuda.empty_cache()

    # ------------------------------------------------------------ build

    def _io(self, i: int) -> dict:
        """Layer i's non-weight pointers (module docstring)."""
        p = self.plan[i]
        out = self.sets[i % 2]
        io = {"pos": self.pos, "ring": self.rings[self.ring_of[i]], "r_f": out.r_f, "gate_f": out.gate_f,
              "ffn_out": out.ffn_out, **(self.rope_w if p.ratio == 0 else self.rope_c)}  # fmt: skip
        if i == 0:
            io.update(h_in=self.h_in0, gin=self.gin0)
        else:  # the previous hi layer's outputs, in place (the ping-pong pair)
            prev = self.sets[(i - 1) % 2]
            io.update(h_in=prev.r_f, gin=prev.gate_f, pend=prev.ffn_out)
        io["r_a"] = self.r_a.get(i, self.r_a_shared)
        # the gate half reads the token-only half's kv (``begin_step``'s launch)
        if p.engram:
            io["eng.kv"] = self.eng_kv[i]
        bound = self.bounds[0] if p.ratio == 2 else self.bounds[1]
        if (
            p.ratio
        ):  # ATOM's storage in place: the request words, the page geometry; its addresses (``_bind_cache``)
            io["req"] = self.req
            io.update({n: 0 for n, _ in self._late_args(i)})
        if p.kind == "consumer":
            io.update(sel=self.sel_out[p.topk_owner])
        elif p.kind in ("kv+index source", "index source"):
            io.update({"idx.bound": bound, "sel_out": self.sel_out[i], "cand": self.cand,
                       "cand.bound": self.bounds[2]})  # fmt: skip
        return io

    def _late_args(self, i: int) -> list:
        """Layer i's pointers into ATOM's storage, (arg name, its KV owner; None: the block table), bound when the
        first step sees the cache (``_bind_cache``): its owner's main rows (``ckv``), the step's block table
        (``ptab``); an index source its owner's FP4 plane (``index_k`` values / ``index_k.s`` scales); a ratio-2 KV
        source its compress rings."""
        p = self.plan[i]
        if not p.ratio:
            return []
        own = i if p.kind == "kv+index source" else p.kv_owner
        out = [("ckv", own), ("ptab", None)]
        if p.kind in ("kv+index source", "index source"):
            out += [("index_k", own), ("index_k.s", own)]
        if p.kind == "kv+index source" and p.ratio == 2:
            out += [("comp.kv_state", own), ("comp.score_state", own)]
        return out

    def _engram_consts(self) -> dict:
        """The Engram hash constants from ATOM's own hash mapping (``engram_consts``)."""
        if self._eng_mapping is None:
            raise RuntimeError(
                "V4.1 mono hi: the Engram table lookup needs ATOM's Engram runtime (its hash mapping)"
            )
        consts = engram_consts(self._eng_mapping)
        log_once(
            logging.INFO,
            "V4.1 mono hi: Engram hash constants from ATOM's mapping (layers %s, %d token ids)",
            sorted(self.eng.values()),
            consts["token_map"].numel(),
        )
        return consts

    def _engram_tables(self, model, dev) -> int:
        """The token-only halves' pointer tables (one a width): each Engram layer's wkv column share (from ATOM's
        ``engram.wkv``) and hash words, its table shard ATOM's page-locked host one (``use_host_tables``) + the step
        inputs. -> bytes held.
        """
        from .hi.v41 import engram_io

        consts = self._engram_consts()
        args = {"ids": self.ids, "pos": self.pos, "hist": self.hist}
        args["tmap"] = consts["token_map"].to(torch.int32).to(dev)
        for n, i in self.eng.items():
            kv = engram_io.pack_kv(
                n,
                i,
                hw.engram_tensors(model.layers[i].engram),
                consts,
                self.tp,
                self.rank,
                dev,
            )
            args.update(kv)
            # ATOM's host shards, set by ``use_host_tables`` before any launch
            args[f"{n}.tq"] = torch.zeros(1, 256, dtype=torch.uint8, device=dev)
            args[f"{n}.ts"] = torch.zeros(1, 8, dtype=torch.uint8, device=dev)
            args[f"{n}.kv"] = self.eng_kv[i]
        self._packed.append(args)
        self._eng_args, self._eng_consts = args, consts
        from .hi.runtime.peer import PeerGroup

        # a peer group of its own: its regions never overlap the chain's pinned moe.part (module docstring)
        pb = max(k.program.peer_bytes for k in self.eng_programs.values())
        self.eng_peers = PeerGroup(pb, self._cpu_group) if self.tp > 1 and pb else None
        for s, kern in self.eng_programs.items():
            missing = (
                set(kern.program.args) - set(args) - {"_peers", "_rank", "_peer_top"}
            )
            if missing:
                raise KeyError(f"Engram kv launch: no pointer for {sorted(missing)}")
            self.tables[ENGKV, s] = kern.table(args, device=dev, peers=self.eng_peers)
        self.eng_bufs = _buffers(self.eng_programs, self.arena, dev, self.step)
        return sum(
            t.numel() * t.element_size()
            for k, t in args.items()
            if k not in ("ids", "pos", "hist")
        )

    def use_host_tables(self, host, staging) -> None:
        """The Engram kv launch reads each layer's table shard from ATOM's page-locked host shard (``host``: its
        ``EngramHost``, page-locked at load, a page-locked copy or the mapping registered in place: the same rows
        the hi runtime's ``shard_rows`` names, checked), its pointer tables rebuilt; and ``staging`` (ATOM's
        ``EngramStaging``) skips the step's lookup, all-gather, copies and compressed-id snapshot on the steps the hook
        serves (``EngramStep.served_by_round``, the builder's routing decision: nothing reads its staged rows; the
        begin-step writes the cursor candidates the verdict commits: ``_fold_args``), keeping the cursor advance.
        """
        if not getattr(host, "uva", False) or staging is None:
            raise RuntimeError("V4.1 mono hi: the Engram lookup from ATOM's host shards needs ATOM's UVA Engram lookup "
                               "with overlap staging")  # fmt: skip
        for n, i in self.eng.items():
            # (its rows and its device read checked at install: ``check``)
            tq, ts, (lo, hi), tq_dev, ts_dev = host_shard(
                host, self._eng_consts, i, self.tp, self.rank
            )
            logger.info("Engram layer %d: ATOM's host shard rows [%d, %d), page-locked at %#x / %#x", i, lo, hi,
                        tq.untyped_storage().data_ptr(), ts.untyped_storage().data_ptr())  # fmt: skip
            self._eng_args[f"{n}.tq"], self._eng_args[f"{n}.ts"] = tq_dev, ts_dev
            # kept alive with the hook
            self._eng_host = getattr(self, "_eng_host", []) + [
                tq,
                ts,
            ]
        dev = self.ids.device
        for s in self.widths:
            self.tables[ENGKV, s] = self.eng_programs[s].table(
                self._eng_args, device=dev, peers=self.eng_peers
            )
        log_once(
            logging.INFO,
            "V4.1 mono hi: Engram lookup from ATOM's host shards at widths %s; ATOM's staging skipped on "
            "served steps",
            self.widths,
        )

    def _head_tables(self, model, dev) -> int:
        """The verify head's pointer tables (one a width): ``norm.g`` / ``head.w`` packed from ATOM's own final
        norm and this rank's head rows, the last layer's streams (its R_F and gate; its plain output unless
        chained), the persistent outputs. -> bytes held."""
        norm_w, head_w = model.norm.weight.detach(), model.head.weight.detach()
        R, H = self.h_in0.shape[0], self.h_in0.shape[-1]
        self.head_ids = torch.zeros(R, dtype=torch.int32, device=dev)
        self.head_vals = torch.zeros(R, dtype=torch.float32, device=dev)
        # the fold's R: ATOM collapses it
        self.r_h = torch.zeros(R, HC, H, dtype=torch.bfloat16, device=dev)
        packed = self.H.pack(norm_w, head_w, device=dev)
        # the block draft's MarkovHead reads the same tensor
        self.head_w_packed = packed["head.w"]
        last = self.sets[self.mf[-1] % 2]
        args = {**packed, "r_f": last.r_f, "gate_f": last.gate_f, "ffn_out": last.ffn_out, "head.ids": self.head_ids,
                "head.vals": self.head_vals, "r_h": self.r_h}  # fmt: skip
        # a launch a width (one request, the wide head) or a request ("split"): (width, launch) -> (first row, rows);
        # a band's head: its program's rows (``pwidth``: its ``head.par`` block holds them, the rows past the step's
        # greedy, the seed after them)
        launches = {}
        for s in self.head_programs:
            nb, rq = self.split[s]
            for b in range(nb if self.head_mode[s] == "split" else 1):
                launches[s, b] = (
                    b * rq,
                    rq if self.head_mode[s] == "split" else self.pwidth(s),
                )
        # a launch's block: (T, top_p) a row of its rows as f32 bits, then the seed's two words ([2 rows + 2]: the
        # head program's ``head.par``); every block in one buffer, the c1 width's first (its offset 0), staged from a
        # pinned ring in one copy (``set_head_sampling``)
        seed = HEAD_SEED
        words = np.array(
            [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF], dtype=np.uint32
        ).view(np.int32)
        self.par_blocks, off = {}, 0
        for key, (r0, n) in launches.items():
            self.par_blocks[key] = (off, r0, n)
            off += 2 * n + 2
        # (host, numpy: the per-step check costs ~2 us)
        self.par_last = np.zeros(off, dtype=np.int32)
        for o, _, n in self.par_blocks.values():
            self.par_last[o + 2 * n : o + 2 * n + 2] = words
        self.par_rows = max(r0 + n for _, r0, n in self.par_blocks.values())
        self.head_par = torch.from_numpy(self.par_last.copy()).to(dev)
        self.par_ring = [
            torch.zeros(off, dtype=torch.int32).pin_memory() for _ in range(4)
        ]
        self.par_events = [torch.cuda.Event() for _ in range(4)]
        self.par_next = 0
        # ``set_head_sampling``'s persistent host buffers (filled in place each step, nothing allocated): the rows'
        # (T, top_p), and the next blocks (their seed words kept from ``par_last``)
        self.par_row_buf = np.zeros((self.par_rows, 2), dtype=np.float32)
        self.par_work = self.par_last.copy()
        self.par_work_f = self.par_work.view(np.float32)
        # (the pinned slots' host views)
        self.par_ring_np = [t.numpy() for t in self.par_ring]
        for (s, b), (r0, n) in launches.items():
            kern = self.head_programs[s]
            at = {
                k: args[k][r0:]
                for k in ("r_f", "gate_f", "ffn_out", "head.ids", "head.vals", "r_h")
            }
            o = self.par_blocks[s, b][0]
            at["head.par"] = self.head_par[o : o + 2 * n + 2]
            missing = (
                set(kern.program.args)
                - set(args)
                - set(at)
                - {"_peers", "_rank", "_peer_top"}
            )
            if missing:
                raise KeyError(f"verify head: no pointer for {sorted(missing)}")
            self.tables[HEAD, s, b] = kern.table(
                {**args, **at}, device=dev, peers=self.peers
            )
        self._packed.append(packed)
        return sum(t.numel() * t.element_size() for t in packed.values())

    def set_head_sampling(self, temps=None, top_ps=None, rows=None) -> bool:
        """This step's rows' (temperature, top_p) for the sampling head, with the step's metadata (the builder's
        routing decision, ``round.HiRound.service``): request i's ``temps[i]`` / ``top_ps[i]`` on its ``rows[i]`` rows
        (numpy, the step's requests in order; None: every row greedy; rows past them greedy). Filled in place into
        persistent host buffers; one H2D copy from a pinned slot when they changed, on the current stream (so ahead of
        the forward that reads them); unchanged: nothing (a few µs of host). -> False when a graph capture is
        recording (nothing staged).
        """
        buf = self.par_row_buf
        buf.fill(0.0)  # (past the step's rows: greedy)
        if temps is not None:
            r = 0
            for t, p, n in zip(temps, top_ps, rows):
                n = min(int(n), self.par_rows - r)
                buf[r : r + n, 0], buf[r : r + n, 1] = t, p
                r += n
        # each launch's block: its rows' pairs
        f = self.par_work_f
        for o, r0, n in self.par_blocks.values():
            f[o : o + 2 * n] = buf[r0 : r0 + n].reshape(-1)
        if np.array_equal(self.par_work, self.par_last):
            return True
        # (a capture's dummy step: nothing staged into the graph)
        if torch.cuda.is_current_stream_capturing():
            return False
        i = self.par_next
        self.par_next = (i + 1) % len(self.par_ring)
        self.par_events[i].synchronize()  # the slot's previous copy is done
        self.par_ring_np[i][:] = self.par_work
        self.head_par.copy_(self.par_ring[i], non_blocking=True)
        self.par_events[i].record()
        self.par_last[:] = self.par_work
        return True

    def head_output(self, rows: int):
        """The head output of the forward that just ran on the host (an eager step, or the one a graph capture
        records; ``HiDecodeModel.compute_head``): its in-graph verify's output ids and verdict at width ``rows``
        (``post_verify``), or None when it ran none."""
        if self.pv is None or self.verified != rows:
            return None
        return self.pv.output(rows)

    # ------------------------------------------------------------ routing

    def serves_width(self, rows: int, max_q_len) -> bool:
        """Host-side, TP-uniform: whether a decode of ``rows`` rows at query bucket ``max_q_len`` is a shape the hi
        layers run: a width of the hook whose query bucket is the width's rows a request (one request: the whole step,
        a bs-1 graph; B requests: B x rq rows, a bs-B graph whose requests each verify rq rows). Part of the step's
        routing decision (``round.HiRound.service``)."""
        split = self.split.get(rows)
        return split is not None and int(max_q_len or 0) == split[1]

    def invalidate(self, step=None, dummy: bool = False) -> None:
        """A forward this hook does not serve (a prefill, another width, ATOM's own path) writes the caches of its
        own requests: a request slot whose last served step recorded one of their ATOM slots imports in full next
        (the incremental import); every other slot keeps its record (a prefill of request B between served steps of
        request A: A's next step stays incremental). ``step``: the forward's ``BatchStep`` (its slots); None, or a
        dummy forward: every slot. A device op on the current stream, so a graph captured for such a forward replays
        it (on the slots its replay runs)."""
        slots = getattr(step, "slots", None)
        if dummy or slots is None or slots.numel() == 0:
            self.imp.fill_(-1)
            return
        ml._invalidate_kernel[(self.bmax,)](self.imp, slots, slots.numel(), BLOCK=triton.next_power_of_2(
            max(int(slots.numel()), 2)))  # fmt: skip

    def runs(self, layer_id: int) -> bool:
        p = self.plan.get(layer_id)
        return p is not None and p.hi

    # ------------------------------------------------------------ one step

    def _geo(self, cache):
        w0 = cache.geometry.window(0, cache.num_pages)
        return (
            w0.ring_start,
            w0.slot_rows,
            w0.ring_slots,
            cache.num_slots,
            self.max_ctx - 16,
        )

    def _bind_cache(self, cache, step) -> None:
        """The programs' pointers into ATOM's storage (``_late_args``) and the page geometry (``req``'s), from the
        KV cache and the step's block table, when they are not bound to them yet (the first served step; the
        addresses stay fixed after: the pool, the planes and the published block table are allocated once, so a
        captured graph keeps them). Device writes (capturable). Refuses a layout the programs do not address.
        """
        bt = step.block_tables
        key = (cache.pool.data_ptr(), bt.data_ptr(), bt.stride(0))
        if key == self._bound:
            return
        g = cache.geometry
        if (
            cache.pool.shape[-1] != HEAD_DIM
            or not cache.pool.is_contiguous()
            or g.row_bytes != 2 * HEAD_DIM
        ):
            raise RuntimeError(
                f"pool {tuple(cache.pool.shape)}: not the bf16 [rows, 512] view"
            )
        if bt.dtype != torch.int32 or bt.stride(-1) != 1:
            raise RuntimeError(
                f"block table {bt.dtype} / {bt.stride()}: not int32 rows"
            )
        if g.compress_ring_slots != SLOTS:
            raise RuntimeError(
                f"compress ring of {g.compress_ring_slots} slots, the programs' {SLOTS}"
            )
        addr, rings = {}, set()
        for o, ratio in self.kv_owners.items():
            if g.rows_per_page(ratio) != BLOCK_SIZE // ratio:
                raise RuntimeError(
                    f"owner {o}: {g.rows_per_page(ratio)} rows a page at ratio {ratio}"
                )
            units = cache.index_units[o]
            if units.layout != self.index_layout:
                raise RuntimeError(
                    f"owner {o}: FP4 index plane in the {units.layout} layout, the programs' {self.index_layout}"
                )
            # the FP4 plane's units as ``paged`` addresses them (8 keys a unit: 512 value bytes, 32 scale bytes)
            if (units.values.stride(0), units.scales.stride(0)) != (
                ml.FP4_PAGE * ml.IDX_DIM // 2,
                ml.FP4_PAGE * 4,
            ):
                raise RuntimeError(
                    f"FP4 index plane strides {units.values.stride()} / {units.scales.stride()}"
                )
            addr[o] = {"ckv": cache.pool.data_ptr() + g.main_offset(o) * g.row_bytes,
                       "index_k": units.values.data_ptr(), "index_k.s": units.scales.data_ptr()}  # fmt: skip
            if ratio == 2:
                kv, sc = cache.compress_state(o)
                if (
                    kv.dtype != torch.float32
                    or kv.stride()[1:] != (HEAD_DIM, 1)
                    or kv.stride() != sc.stride()
                ):
                    raise RuntimeError(
                        f"compressor ring {kv.dtype} strides {kv.stride()} / {sc.stride()}"
                    )
                addr[o].update(
                    {"comp.kv_state": kv.data_ptr(), "comp.score_state": sc.data_ptr()}
                )
                rings.add(kv.stride(0))
        if len(rings) > 1:
            raise RuntimeError(
                f"compressor rings of slot strides {sorted(rings)}: the programs take one"
            )
        for w, v in enumerate(
            (bt.stride(0), g.page_bytes // g.row_bytes, rings.pop() if rings else 0, 0)
        ):
            self.req[w].fill_(v)  # (the request words' page geometry)
        for table, idx, name, own in self._late:
            table[idx].fill_(bt.data_ptr() if own is None else addr[own][name])
        self._bound = key
        log_once(
            logging.INFO,
            "V4.1 mono hi: %d pointers of %d layers bound to ATOM's KV pool, FP4 index planes, compress rings "
            "and block table (in place)",
            len(self._late),
            len(self.mf),
        )

    def begin_step(
        self, cache, step, rows: int, hidden, staged=None, input_ids=None
    ) -> None:
        """One launch (``_begin_step_kernel``): the step bump (recorded by a capture), the rings, the request words,
        the embedding rows (``hidden``: ATOM's lookup of the step's tokens, [rows, hidden]) expanded into layer 0's
        input, the Engram history and cursor candidates (``staged``: ATOM's Engram rows of the step, its staging's
        step; ``input_ids``: the step's tokens); then the Engram token-only halves. First the programs' pointers into
        ATOM's storage (``_bind_cache``)."""
        self._bind_cache(cache, step)
        if self.pv is not None and input_ids is not None:
            # the post-verify's drafts (a fixed address under a capture)
            self.step_ids = input_ids.view(-1)
        # every program's launch record, the Engram kv launch's and the head's included (one step, one bump)
        for b in (
            *self.bufs.values(),
            *getattr(self, "eng_bufs", {}).values(),
            *self.head_bufs.values(),
        ):
            b.launched.clear()
        if self.peers is not None:
            self.peers.last_parity = None
        # the step's requests (each its batch slot of the hi-side caches)
        nb = self.split[rows][0]
        eflat, ke = self._engram_segments(cache, step, rows, staged, input_ids)
        rp = self.ring_p
        n = nb * len(self.mf) * rp + rows + ke["NE0"] + ke["NE1"] + ke["EHIST"]
        ml._begin_step_kernel[(n,)](
            cache.pool, step.positions, step.batch_ids, step.slots, self.rings, self.ring_layers, self.pos,
            self.bounds, *self._geo(cache), rows, hidden, hidden.stride(0), self.h_in0,
            self.step, self.req, *eflat, self.imp, NMF=len(self.mf), WIN=WINDOW, DIM=HEAD_DIM,
            RB=self.RB, HIDDEN=hidden.shape[-1], BLOCK=1024, NHC=HC, **ke, INCR=1, RP=rp, NB=nb, BM=self.bmax,
            RQP=triton.next_power_of_2(rows // nb),
        )  # fmt: skip
        if rows in self.band_of:
            self._band_step(rows)
        if self.eng:
            self._engram_launch(rows)

    def _band_step(self, rows: int) -> None:
        """A band's step (module docstring, "Widths"; device ops, captured with the step): the live words (rows,
        spec, batch) after the step counter, and the band's dead request slots' import records cleared. Their
        requests' work runs on whatever the slots hold and writes those slots' rings (only the live rows and
        requests are bounded; none of ATOM's storage: their request words are dead), so a request that later
        continues in such a slot imports its ring in full.
        """
        self.live.stage(rows)
        dead = self.live.dead(rows)
        if dead is not None:
            self.imp[dead.start * ml.IMP_WORDS : dead.stop * ml.IMP_WORDS].fill_(-1)

    def _engram_segments(
        self, cache, step, rows, staged, input_ids
    ) -> tuple[list, dict]:
        """The fused begin-step's Engram segments (``_begin_step_kernel``): its pointer args and constexprs (the
        table lookup: a history program a request, which also writes the request's ids and cursor candidates,
        ``_fold_args``)."""
        dummy = self.pos
        if not self.eng:
            return [dummy, 0, dummy, dummy, 0, dummy, dummy, 0, dummy], {
                "NE0": 0,
                "NE1": 0,
                "EHIST": 0,
            }
        flat = [
            dummy,
            0,
            dummy,
            dummy,
            0,
            dummy,
            cache.cursor,
            cache.cursor.stride(0),
            self.hist,
        ]
        ke = {
            "NE0": 0,
            "NE1": 0,
            "EHIST": self.split[rows][0],
            "hist_stride": self.hist.stride(0),
        }
        return flat, dict(**ke, **self._fold_args(staged, rows, input_ids))

    def _fold_args(self, staged, rows: int, input_ids) -> dict:
        """Each request's Engram history program of the begin-step also copies the request's ids into the kv launch's
        ``ids`` and writes the cursor candidates ATOM's staging snapshot would (``ml._eng_fold``; ATOM's snapshot is
        skipped on served steps: ``use_host_tables``). Fixed addresses: ATOM's ``EngramStep`` of this width (a capture
        records them)."""
        nb, rq = self.split[rows]
        st = getattr(getattr(staged, "staging", None), "step", None)
        if st is None or st.candidates is None:
            raise RuntimeError(
                "V4.1 mono hi: a served step without ATOM's Engram verify staging"
            )
        cand = st.candidates  # [requests, rows, NGRAM]: request b's row l at [b, l]
        if (
            cand.shape[0] < nb
            or cand.shape[1] < rq
            or cand.shape[-1] != 4
            or cand.stride(-1) != 1
            or rq > 16
        ):
            raise RuntimeError(f"V4.1 mono hi: Engram candidates {tuple(cand.shape)} / {cand.stride()}, "
                               f"{nb} x {rq} rows")  # fmt: skip
        ids = input_ids.view(-1)
        return {"fold_ids": ids, "fold_dst": self.ids, "fold_tmap": self._eng_args["tmap"], "fold_dead": st.dead,
                "fold_cand": cand, "fold_cand_stride": cand.stride(1), "fold_cand_req_stride": cand.stride(0),
                "fold_live": st.live, "EFOLD": 1, "RPOW": triton.next_power_of_2(rq)}  # fmt: skip

    def collapse_view(self, rows: int) -> torch.Tensor:
        """Where the end-step writes ATOM's collapse of the step's hidden: the runner's own forward output rows (so
        the graph's output copy is a copy onto itself, which torch skips): [rows, hidden].
        """
        return self.runner.forward_vars["outputs"][:rows]

    def _engram_launch(self, rows: int) -> None:
        """The token-only halves of every Engram layer (one launch, its own peer group), before layer 0."""
        if self.eng_peers is not None:
            self.eng_peers.last_parity = None
        self.eng_programs[rows].launch(self.tables[ENGKV, rows], self.eng_bufs[rows], layer=ENGKV_LAUNCH,
                                       stream=torch.cuda.current_stream())  # fmt: skip
        _end_step(self.eng_peers)  # (one launch a step: the step phase flips)

    def layer(self, i: int, rows: int) -> None:
        """The hi runtime layer i (every layer runs on the hi path: layer i - 1's outputs are its inputs, in place); after the
        last, the verify head."""
        key = self._key(i, rows)
        self.programs[key].launch(
            self.tables[i, rows],
            self.bufs[key],
            layer=i,
            stream=torch.cuda.current_stream(),
        )
        if self.plan.get(i + 1) is None:
            # the verify head (module docstring), launch id i + 1: chained, its fold sums layer i's pushed partials;
            # B requests: B launches of the one-request head at their rows (``head_launch_ids``)
            launches = self.split[rows][0] if self.head_mode[rows] == "split" else 1
            for b, lid in enumerate(head_launch_ids(i, launches)):
                self.head_programs[rows].launch(self.tables[HEAD, rows, b], self.head_bufs[rows], layer=lid,
                                                stream=torch.cuda.current_stream())  # fmt: skip
            # the step's last launch on the group: 41 (odd) flip the step phase
            _end_step(self.peers)

    def _taps(self) -> list:
        """(r_a, ATOM's aux buffer) of the hi layers ATOM's DSpark taps. (Bound here on the first served step,
        before the end-step launch writes them: ``CTX`` reads them right after it.)"""
        if self.aux is None:
            self._ctx_taps()
        if self.aux is None:
            return []
        return [(r_a, self.aux.buffers[self.aux.layer_ids.index(i)]) for i, r_a in self.r_a.items()
                if i in self.aux.layer_ids]  # fmt: skip

    def end_step(self, cache, step, rows: int) -> None:
        """The end-step launch (the cache export, the DSpark taps, the last layer's gates planar and ATOM's
        collapse); then the draft's context write and the post-verify, inside the target graph (module docstring).
        """
        self._end_step(cache, step, rows)
        self._ctx_after_target(cache, step, rows)
        if self.pv is not None and step.max_q_len == self.split[rows][1]:
            # the verdict and commit in the graph (``post_verify``): the forward's head output (``head_output``)
            self.pv.launch(self, cache, step, rows)
            self.verified = rows

    def _end_step(self, cache, step, rows: int) -> None:
        taps = self._taps()
        if len(taps) > ml.MAX_TAPS:
            raise RuntimeError(
                f"V4.1 mono hi: {len(taps)} DSpark taps, the end-step launch takes {ml.MAX_TAPS}"
            )
        tp = [x for t in taps for x in t] + [self.r_a_shared, self.r_a_shared[:, 0]] * (
            ml.MAX_TAPS - len(taps)
        )
        hidden = taps[0][1].shape[-1] if taps else self.h_in0.shape[-1]
        last = self.sets[self.mf[-1] % 2]
        segs = len(self.mf) + len(taps) + int(self.planar_at_end)
        # ATOM's collapse after the planar segment, into the runner's outputs (``HiDecodeModel._hi_forward``): a
        # segment a 1024-wide block of the hidden
        segs += -(-hidden // 1024)
        ke = {
            "col_res": self.r_h,
            "col_out": self.collapse_view(rows),
            "COLLAPSE": 1,
            "enable_fp_fusion": False,
        }
        ml._end_step_kernel[(segs * rows,)](
            cache.pool, step.positions, step.batch_ids, step.slots, self.rings, self.ring_layers,
            *self._geo(cache), rows, *tp, last.gate_f, self.pre, self.post, self.comb, self.imp, NMF=len(self.mf),
            WIN=WINDOW, DIM=HEAD_DIM, HIDDEN=hidden, BLOCK=1024, NHC=HC, NMIX=MIX, **ke, NTAPS=len(taps),
            PLANAR=self.planar_at_end, TRACK=1, NB=self.split[rows][0], BM=self.bmax,
        )  # fmt: skip


def head_launch_ids(last: int, requests: int) -> tuple[int, ...]:
    """The verify head's launch ids at a width of ``requests`` requests, after the last layer ``last``: ``last + 1``
    for the first (one request: the chained head), then ``HEAD_MORE`` (consecutive launches on one peer group
    alternate tag parity, as layer ids do)."""
    if requests > 1 and (last + 1) % 2 == HEAD_MORE[0] % 2:
        raise ValueError(
            f"the verify head after layer {last}: ids {(last + 1, *HEAD_MORE)} do not alternate parity"
        )
    return (last + 1, *HEAD_MORE[: requests - 1])


def check(model, atom_config, engram, draft, tp: int, rank: int) -> None:
    """Every refusal of this TP rank's install that reads what it loaded, raised (``MonoUnsupported``) before anything
    collective and before anything is built or allocated (temporaries aside): the model's geometry and context, the
    layer plan (every layer on the hi path), the target layers' and the draft stages' weights
    (``hi_weights.check_block``: the FP8 linears, the routed experts' layout and padding codes, a rank's own shard), the
    Engram layers' and the context write's linears, the RoPE caches, the verify head's tensors, the draft's stages and
    norms and ATOM's Engram host shards (their rows, read on the device through their device addresses). ``engram``:
    ATOM's Engram runtime (its hash mapping and host shards); ``draft``: ATOM's DSpark draft model.
    ``hi_dispatch.install_hi`` agrees on the verdict over the TP group before anything is built, so every rank builds
    or every rank falls back with nothing built."""
    from .hi.v41 import draft as D
    from .hi.v41 import head as H
    from .hi.v41 import layer as L

    cfg = model.config
    if cfg.hc_mult != HC or cfg.hidden_size != L.HIDDEN:
        raise MonoUnsupported(
            f"hc_mult {cfg.hc_mult} / hidden {cfg.hidden_size} is not the hi runtime's V4.1"
        )
    if int(atom_config.max_model_len) > MAX_CTX:
        raise MonoUnsupported(
            f"max_model_len {atom_config.max_model_len} > the hi programs' context bucket {MAX_CTX} (set "
            "--max-model-len)"
        )
    plan = hi_plan.plan_layers(model.topology, cfg.engram_layer_ids, engram_on=True)
    atom_layers = [p.layer_id for p in plan if not p.hi]
    if atom_layers:
        raise MonoUnsupported(
            f"V4.1 mono hi: layers {atom_layers} cannot run on the hi path (the hook runs every layer)"
        )
    inter = cfg.moe_intermediate_size
    for p in plan:
        block = model.layers[p.layer_id]
        hw.check_block(block, inter, tp)
        if p.engram:
            hw.check_linears(block.engram.wkv)
    for rope in (model.window_rope, model.global_rope, draft.rope):
        hw.rope_args(rope)
    # the verify head: ATOM's final norm and this rank's head rows
    if float(getattr(model.norm, "eps", L.EPS)) != float(L.EPS):
        raise MonoUnsupported(
            f"V4.1 mono hi verify head: final norm eps {model.norm.eps} != the hi runtime's {L.EPS}"
        )
    norm_w, head_w = model.norm.weight.detach(), model.head.weight.detach()
    n = H.VOCAB // tp
    if tuple(head_w.shape) != (n, cfg.hidden_size) or head_w.dtype != torch.bfloat16:
        raise MonoUnsupported(f"V4.1 mono hi verify head: head.weight {tuple(head_w.shape)} {head_w.dtype}, want bf16 "
                              f"[{n}, {cfg.hidden_size}] (this rank's vocab rows)")  # fmt: skip
    if not torch.equal(norm_w.float(), norm_w.bfloat16().float()):
        raise MonoUnsupported(
            "V4.1 mono hi verify head: the final norm weight is not bf16-exact"
        )
    # the DSpark draft: its context write (``bind_draft``, ``_ctx_build``) and its stages (``_draft_build``)
    layers = tuple(draft.context_layers)
    if len(layers) != D.STAGES:
        raise MonoUnsupported(
            f"V4.1 mono hi draft: {len(layers)} draft stages, the hi runtime's CTX has {D.STAGES}"
        )
    eps = {float(a.kv_norm.eps) for a in layers} | {float(draft.mtp[0].main_norm.eps)}
    if eps != {float(D.EPS)}:
        raise MonoUnsupported(
            f"V4.1 mono hi draft: the draft's norm eps {eps} != the hi runtime's {D.EPS}"
        )
    hw.check_linears(draft.mtp[0].main_proj)
    for s in range(D.STAGES):
        hw.check_block(draft.mtp[s], inter, tp)
    # ATOM's Engram host shards, the rows the hi runtime reads (``use_host_tables``)
    eng = [p.layer_id for p in plan if p.engram]
    if eng:
        consts = engram_consts(engram.mapping)
        for i in eng:
            tq, ts, _, tq_dev, ts_dev = host_shard(engram.host, consts, i, tp, rank)
            _check_host_shard(i, tq, ts, tq_dev, ts_dev)


def host_shard(host, consts: dict, layer: int, tp: int, rank: int) -> tuple:
    """ATOM's page-locked host shard of Engram layer ``layer`` (``host``: its ``EngramHost``, page-locked at load, a
    page-locked copy or the mapping registered in place) as the hi runtime reads it -> (codes, scales) uint8
    [rows, 256] / [rows, 8], their rows [lo, hi) and their device addresses. Raises ``MonoUnsupported`` when the rows
    are not those the hi runtime's ``shard_rows`` names for this rank (``consts``: ``engram_consts``).
    """
    from .hi.blocks.gfx950.engram import shard_rows
    from .hi.v41 import engram_io

    weight, scales, lo, hi = host.prefetcher._tables[layer].registered_shard()
    want = tuple(
        int(x) for x in shard_rows(engram_io.hash_words(consts, layer), tp, rank)
    )
    tq, ts = weight.view(torch.uint8), scales.view(torch.uint8)
    if (
        (lo, hi) != want
        or tq.shape[1:] != (256,)
        or ts.shape[1:] != (8,)
        or tq.shape[0] != hi - lo
    ):
        raise MonoUnsupported(f"Engram layer {layer}: ATOM's host shard rows [{lo}, {hi}) {tuple(tq.shape)} / "
                              f"{tuple(ts.shape)}, the hi runtime's {want} x 256 / 8")  # fmt: skip
    # the device's addresses of the page-locked host pages (the host pointer is not one: Triton maps it per launch, a
    # hi pointer table holds what the device dereferences); each shard tensor's storage starts at its page-locked
    # region's base (``host_memory.pin_rows``)
    wbase, sbase = tq.untyped_storage().data_ptr(), ts.untyped_storage().data_ptr()
    return (
        tq,
        ts,
        (lo, hi),
        device_address(tq.data_ptr(), wbase),
        device_address(ts.data_ptr(), sbase),
    )


def install(model, atom_config, engram_mapping, draft) -> HiLayers | None:
    """The hook for this rank (the deployment is one the hi path serves: ``hi_gate.decide``). ``engram_mapping``:
    ATOM's Engram hash mapping (its runtime's), the source of the table lookup's hash constants (``engram_consts``);
    ``draft``: ATOM's DSpark draft model (loaded: the draft's weights are packed from its tensors). Collective (the
    peer groups): every rank calls it at the same point."""
    widths = list(ROWS)
    max_ctx = MAX_CTX
    for p in hi_plan.plan_layers(
        model.topology, model.config.engram_layer_ids, engram_on=True
    ):
        log_once(
            logging.INFO,
            "V4.1 mono hi plan: layer %2d %-30s %s",
            p.layer_id,
            f"{p.kind} r{p.ratio}"
            + (f" cand {p.cand}" if p.cand else "")
            + (" +engram" if p.engram else ""),
            "hi" if p.hi else f"ATOM: {p.why}",
        )
    # the installers run while ModelRunner still has the model's default dtype
    # (bf16) and device set: the hi runtime's packing and tables assume torch's defaults
    from .hi.emit.compile import serving

    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    try:
        # the builds from FlyDSL's disk cache where it holds them (a restart loads them untraced), their trace checks
        # left to the CPU checks of the served programs (``hi.emit.compile.serving``)
        with serving():
            hook = HiLayers(model, atom_config, widths, max_ctx, engram_mapping, draft)
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)
    from atom.models.deepseek_v41.mono.round import post_verify

    # the verdict and the Engram cursor commit inside the target graph (``post_verify``): the forward's head output
    hook.pv = post_verify.PostVerify(hook.widths, hook.split, hook.step.device)
    return hook


def _check_host_shard(
    layer: int, tq, ts, tq_dev: int, ts_dev: int, n: int = 64
) -> None:
    """The install's check (``check``) of Engram layer ``layer``'s host shard: ``n`` sampled rows of ATOM's host shard
    (``tq`` codes / ``ts`` scales) and its first / last rows, read on the device through their device addresses
    (``tq_dev`` / ``ts_dev``: ``_read_rows``, a kernel's loads as the hi runtime's), are the host's bytes. Raises
    ``MonoUnsupported`` otherwise. (The rows are ATOM's, read from the checkpoint at load; a checkpoint re-read here
    would page the tables' files back in beside every rank's resident copy.)"""
    t0 = time.time()
    rows = tq.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0x5EED + layer)
    idx = sorted(
        {0, rows - 1, *torch.randint(0, rows, (n,), generator=g, device="cpu").tolist()}
    )
    for name, host, dev in (("codes", tq, tq_dev), ("scales", ts, ts_dev)):
        got = torch.stack([host[r].to("cpu", copy=True) for r in idx])
        back = _read_rows(dev, host.shape[1], idx).to("cpu")
        if not torch.equal(got, back):
            raise MonoUnsupported(f"Engram layer {layer} {name}: the device read of ATOM's host shard differs from the "
                                  f"host's bytes on rows {idx[:4]}..")  # fmt: skip
    logger.info("V4.1 mono hi: Engram layer %d host shard read on the device == host (%d rows, %.1f s)", layer,
                len(idx), time.time() - t0)  # fmt: skip


class _DeviceAddress:
    """A raw device address as a Triton kernel's pointer argument."""

    def __init__(self, address: int, dtype=torch.uint8):
        self.address, self.dtype = address, dtype

    def data_ptr(self) -> int:
        return self.address


@triton.jit
def _read_rows_kernel(src, rows, out, WIDTH: tl.constexpr):
    """Program i: ``WIDTH`` bytes of row ``rows[i]`` of the table at ``src`` (a device address) into ``out[i]``."""
    i = tl.program_id(0)
    r = tl.load(rows + i).to(tl.int64)
    c = tl.arange(0, WIDTH)
    tl.store(out + i * WIDTH + c, tl.load(src + r * WIDTH + c))


def _read_rows(addr: int, width: int, idx) -> torch.Tensor:
    """uint8 [len(idx), width] read on the device from the uint8 [*, width] table at device address ``addr``."""
    rows = torch.tensor(idx, dtype=torch.int64, device="cuda")
    out = torch.empty(len(idx), width, dtype=torch.uint8, device="cuda")
    _read_rows_kernel[(len(idx),)](_DeviceAddress(addr), rows, out, WIDTH=width)
    torch.cuda.synchronize()
    return out


def engram_consts(mapping) -> dict:
    """the hi runtime's Engram hash constants (the dict ``engram_io.hash_words`` / ``pack_kv`` read: the token map, each
    layer's multipliers / primes / offsets, the pad id, the layer ids, the table rows) from ATOM's own
    ``NgramHashMapping`` (``mapping``: the Engram runtime's, its tokenizer compression, multipliers and head primes,
    the primes' sums checked against the checkpoint's row counts when it was built), so the served path reads no file.
    The primes are n-gram major ([orders, heads]: ``pack_hash``'s column order, ATOM's head order).
    """
    cfg = mapping.config
    layers = [int(x) for x in cfg.layer_ids]

    def stack(per_layer, shape):
        return torch.stack([torch.as_tensor(np.asarray(per_layer[i], dtype=np.int64), device="cpu").view(shape)
                            for i in layers])  # fmt: skip

    orders = len(cfg.ngram_orders)
    table = np.asarray(mapping.tokenizer.lookup_table, dtype=np.int64)
    return {"token_map": torch.as_tensor(table, device="cpu").to(torch.int32),
            "multipliers": stack(mapping.layer_multipliers, (cfg.max_ngram_size,)),
            "primes": stack(mapping.head_vocab_sizes, (orders, cfg.n_heads)),
            "offsets": stack(mapping.head_offsets, (orders * cfg.n_heads,)), "pad_id": int(mapping.pad_id),
            "layer_ids": layers, "num_embeddings": [int(x) for x in cfg.num_embeddings]}  # fmt: skip
