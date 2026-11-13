# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi programs of one TP rank: composed and built (compiled) for the served configuration, no tensors.

``HiPrograms`` is the composition the hi runner (``hi_runner.HiLayers``) serves with, and the one the CPU program checks
compile: the layer plan, the widths and bands, then, in this order, each width's verify head and layer programs (a
width whose programs do not build is not served: its steps keep ATOM's mono decode), the Engram token-only launch, the
DSpark context write and the block draft's plans. Every program is a ``MonoKernel`` (built on construction; a build that
does not fit takes the fallback the methods name).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from atom.models.deepseek_v41.mono import hi_plan
from atom.models.deepseek_v41.mono.hi_bands import plan_widths
from atom.models.deepseek_v41.mono.hi_gate import INDEX_LAYOUTS, MAX_CTX, REQ_ROWS
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")

# the draft's stages 0-2 and its head (their own peer group)
DRAFT_LAUNCH = (
    43,
    44,
    45,
    46,
)
# the verify head's launches for a step's requests 1-3 (B requests: B one-request heads)
HEAD_MORE = (
    47,
    48,
    49,
)
MAX_BATCH = 4  # the most requests a static width serves
# the step widths served (rows): B x 6 rows is B requests verifying 6 rows each (DSpark's 5 drafts and the anchor);
# 1-4 requests on a width's own programs, 5-8 on the batch bands
ROWS = tuple(REQ_ROWS * b for b in range(1, 9))
# past MAX_BATCH requests, a width of B requests runs the smallest band of at least B (one ``Program.dyn_rows`` build
# a band, launched with the step's live requests)
BANDS = (6, 8)
SLOTS = 7  # the ratio-2 sources' compressor state slots: ATOM's compress ring (2 + a verify's drafts), in place


@dataclass
class _DraftLaunch:
    """One launch of the block draft: its kernel (``HiLayers.dr_kerns[key]``), launch id, the requests it covers
    (first, count) and whether it is chained to the previous launch (reads its pushed MoE partials).
    """

    key: tuple
    kern: object
    lid: int
    r0: int
    n: int
    chained: bool
    stage: int  # 0-2, or 3: a head


@dataclass
class _DraftPlan:
    """The block draft of ``n`` requests: the launches in order, or (``per_request``) the one-request plan once a
    request."""

    n: int
    launches: list
    per_request: bool = False

    def describe(self) -> str:
        if self.per_request:
            return f"{self.n} x the one-request set"
        return " + ".join(
            f"{'stage' + str(d.stage) if d.stage < 3 else 'head'}[{d.n}]"
            for d in self.launches
        )


class HiPrograms:
    """The hi programs of one TP rank (module docstring): built from the model's config and attention topology,
    the served widths, the context bucket and the TP size."""

    def __init__(self, cfg, topology, widths, max_ctx: int, tp: int, group=None,
                 index_layout: str = "preshuffle"):  # fmt: skip
        from .hi.plan import program as mf_program
        from .hi.plan.program import Program, Scenario
        from .hi.runtime.kernel import MonoKernel
        from .hi.v41 import draft as D
        from .hi.v41 import layer as L

        self.L, self.tp = L, tp
        # ATOM's FP4 index plane's layout (``config.index_cache_layout``), which the index sources' programs address
        # in place (``hi.device.gfx950.paged``)
        if index_layout not in INDEX_LAYOUTS:
            raise MonoUnsupported(
                f"V4.1 mono hi: FP4 index layout {index_layout!r}, the programs address {', '.join(INDEX_LAYOUTS)}"
            )
        self.index_layout = index_layout
        # the TP group's CPU group: every build fallback below is taken by every rank or none (``_agree``); None: one
        # process (the CPU checks)
        self.group = group
        # every peer group here hosts programs of several layouts (a step's layer kinds and head, the widths, the
        # draft's stages): their parity-1 slots mirrored from the group's top, so no launch's early put lands on a slot
        # the previous launch still polls on a slower rank (hi ``Program.peer_shared``). Set before any program
        # is built.
        mf_program.share_peer_groups(True)
        # every layer on the hi path, the Engram layers included (their gate half in seam A, the token-only halves in a
        # launch before layer 0)
        plan = hi_plan.plan_layers(topology, cfg.engram_layer_ids, engram_on=True)
        self.plan = {p.layer_id: p for p in plan}
        self.mf = [i for i, p in sorted(self.plan.items()) if p.hi]
        atom_layers = sorted(i for i, p in self.plan.items() if not p.hi)
        if atom_layers:
            raise MonoUnsupported(
                f"V4.1 mono hi: layers {atom_layers} cannot run on the hi path (the hook runs every layer)"
            )
        # a width's requests and rows a request: one request up to 16 rows, or B requests of ``REQ_ROWS`` each, B <=
        # MAX_BATCH, past it a band's
        self.req_rows = REQ_ROWS
        self.bands = BANDS
        self.width_plan = plan_widths(
            widths, self.req_rows, MAX_BATCH, self.bands, one_request_max=16
        )
        self.split = {w: (x.requests, x.req_rows) for w, x in self.width_plan.items()}
        self.widths = sorted(self.split)
        # a width past MAX_BATCH requests: its band (``bands``), the smallest of at least its requests
        self.band_of = {
            w: x.band for w, x in self.width_plan.items() if x.band is not None
        }
        # the caches' batch slots: the most requests a step, a band's (its dead requests' slots too)
        self.bmax = max(
            max(b for b, _ in self.split.values()),
            max(self.band_of.values(), default=0),
        )
        # chained (TP, one kernel a layer): in the run of hi layers a layer pushes its MoE partials and the next
        # one's seam A (or the verify head's fold) sums them; the run's programs pin the cross-launch region first
        self.chain = True
        self.max_ctx, self.slots = max_ctx, SLOTS
        self.aux_ids = tuple(
            int(i) for i in getattr(cfg, "dspark_target_layer_ids", ()) or ()
        )
        self.cl = {r: hi_plan.n_max(max_ctx, r) for r in (1, 2)}
        self.eng = {f"eng{i}": i for i in self.mf if self.plan[i].engram}
        # ---- programs: one a (kind, width), shared by the layers of the kind; a width's verify head first
        self.head_programs, self._heads, self.head_mode = {}, {}, {}
        self.sources, self.programs = {}, {}
        for i in self.mf:
            p = self.plan[i]
            if p.kind in ("kv+index source", "index source"):
                self.sources[i] = L.Source(p.ratio, self.cl[p.ratio], kv=p.kind == "kv+index source", index=True,
                                           n_max=self.cl[p.ratio], cand=p.cand,
                                           state_slots=SLOTS if p.ratio == 2 and p.kind == "kv+index source" else 0)  # fmt: skip
        # a width whose programs do not build is not served (its steps take ATOM's own mono decode: ``serves_width``)
        unbuilt = [s for s in self.widths if not self._agree(self._build_width(s))]
        if unbuilt:
            self._drop_widths(unbuilt)
        # a width's programs read / write the cross-launch region at the same offset (a step runs one width)
        for s in self.widths:
            chained = [k for key, k in self.programs.items() if key[1] == s] + [
                self.head_programs.get(s)
            ]
            # (a program that doesn't touch a region has no offset for it)
            offs = {
                n: {k.program.peer.get(n) for k in chained if k is not None} - {None}
                for n in self._chain_peer()
            }
            if any(len(v) > 1 for v in offs.values()):
                raise RuntimeError(
                    f"width {s}: chained programs disagree on the cross-launch peer regions: {offs}"
                )
        # the Engram token-only halves a width (the history's length: the bucket + a verify's rows)
        self.eng_programs = {}
        if self.eng:
            for s in self.widths:
                sc = Scenario(*self.split[s], self.tp)
                self.eng_programs[s] = MonoKernel(
                    Program(f"atom_v41_engkv_r{s}", sc, L.engram_kv_stages(sc, self.eng, max_ctx + 16),
                            sgpr_spill_ok=L.engram_kv_spill_ok(self.tp))
                )  # fmt: skip
        # ---- the DSpark draft: its context-KV write a width (the hi runtime's RingStore at B requests, each row at
        # its request's position and slot), then the block draft's plans by requests
        self.D = D
        self.ctx_programs = {}
        for s in self.widths:
            _, prog = D.context_program(
                f"atom_v41_ctx_r{s}", Scenario(*self.split[s], self.tp), "shard", False
            )
            self.ctx_programs[s] = MonoKernel(prog)
        W = int(getattr(cfg, "dspark_block_size", 5) or 5)
        if self.tp < 2:
            raise ValueError("V4.1 mono hi draft: TP (the stages are chained)")
        self.dr_width = W
        # the most requests a draft serves on the hi path (``draft_run``): the most a target step serves
        self.dr_bmax = self.bmax
        self.dr_kerns = {}
        self.dr_plans = {n: self._draft_plan(n, W) for n in range(1, self.dr_bmax + 1)}
        # a plan's chained launches read the cross-launch regions at one offset
        for (
            n,
            plan,
        ) in self.dr_plans.items():
            # (not the plain heads)
            chained = [d.kern for d in plan.launches if d.stage < 3 or d.chained]
            offs = {
                r: {k.program.peer.get(r) for k in chained} - {None}
                for r in self._chain_peer()
            }
            if any(len(v) > 1 for v in offs.values()):
                raise RuntimeError(f"the draft's chained programs ({n} requests) disagree on the cross-launch peer "
                                   f"regions: {offs}")  # fmt: skip

    def _build_width(self, s: int) -> bool:
        """Width s's verify head (before its layer programs: the last layer's chain role follows it) and layer
        programs; -> whether they all built (a resource or plan refusal of any of them: False, logged).
        """
        from .hi.emit.compile import ResourceError
        from .hi.plan.program import PlanError

        try:
            self.head_programs[s] = self._width_head(s)
            for i in self.mf:
                key = self._key(i, s)
                if key not in self.programs:
                    self.programs[key] = self._program(
                        self.plan[i], s, *self.role(i, s), full_r_a=self.full_r_a(i, s)
                    )
        except (ResourceError, PlanError) as e:
            logger.warning("V4.1 mono hi: width %d does not build (%s: %s): ATOM's mono decode serves it", s,
                           type(e).__name__, str(e)[:160])  # fmt: skip
            return False
        return True

    def _width_head(self, s: int):
        """Width s's verify head: one request, the head at its width; B requests, one wide head at Scenario(B, rows)
        where hi builds it, else B launches of the one-request head at their rows ("split", ``head_launch_ids``)
        after a chain tail; a band, its wide head (a dyn_rows build: no split fallback, ids for 4 heads).
        """
        b, rq = self.split[s]
        if b == 1:
            self.head_mode[s] = "one"
            return self._head_program(s, self.chain, (1, s))
        if s in self.band_of:
            self.head_mode[s] = "wide"
            return self._head_program(
                self.pwidth(s), self.chain, (self.band_of[s], rq), dyn=True
            )
        kern = None
        try:
            kern = self._head_program(s, self.chain, (b, rq))
            self.head_mode[s] = "wide"
        # PlanError / ResourceError: the hi runtime's head at B
        except Exception as e:  # noqa: BLE001 - a build refusal
            logger.info("V4.1 mono hi: the verify head at %d x %d rows does not build (%s: %s): %d "
                        "launches of the %d-row head", b, rq, type(e).__name__, str(e)[:160], b, rq)  # fmt: skip
        if not self._agree(
            kern is not None
        ):  # (another rank's did not build: every rank splits)
            kern = None
        if kern is None:
            self.head_mode[s] = "split"
            kern = self._head_program(rq, False, (1, rq))
        return kern

    def _drop_widths(self, unbuilt: list[int]) -> None:
        """Take the widths whose programs did not build out of the served ones (``split``: the step's routing) and
        their programs no served width shares out of the built ones."""
        for s in unbuilt:
            for table in (
                self.width_plan,
                self.split,
                self.band_of,
                self.head_programs,
                self.head_mode,
            ):
                table.pop(s, None)
        self.widths = sorted(self.split)
        if not self.widths:
            raise MonoUnsupported("V4.1 mono hi: no width's programs build")
        kept = {self.pwidth(s) for s in self.widths}
        self.programs = {k: v for k, v in self.programs.items() if k[1] in kept}
        served = {id(k) for k in self.head_programs.values()}
        self._heads = {k: v for k, v in self._heads.items() if id(v) in served}
        self.bmax = max(
            max(b for b, _ in self.split.values()),
            max(self.band_of.values(), default=0),
        )

    def _agree(self, ok: bool) -> bool:
        """``ok`` on every rank of the TP group (a build fallback every rank takes alike: the ranks' programs must
        match, their launches wait on each other); this process's ``ok`` without a group.
        """
        if self.group is None:
            return ok
        from atom.distributed.group_agreement import group_agree

        return group_agree(ok, self.group)

    def kernels(self) -> dict:
        """Every program built: ``(kind, key) -> MonoKernel`` (kind: layer, head, engkv, ctx, draft)."""
        out = {}
        for kind, table in (("layer", self.programs), ("head", self._heads), ("engkv", self.eng_programs),
                            ("ctx", self.ctx_programs), ("draft", self.dr_kerns)):  # fmt: skip
            out.update({(kind, key): k for key, k in table.items()})
        return out

    def head_chained(self, s: int) -> bool:
        """Whether width s's verify head is chained to the last layer (its fold sums the layer's pushed MoE
        partials): one head launch at the width (one request, or the wide head of B requests); "split" (B one-request
        heads) runs after a chain tail."""
        mode = getattr(self, "head_mode", {}).get(
            s, "one" if self.split[s][0] == 1 else "split"
        )
        return self.chain and mode != "split"

    def role(self, i: int, s: int) -> tuple[bool, bool]:
        """(push_moe, prev_moe) of hi layer i at width s: chained to a hi
        neighbour (module docstring)."""
        if not self.chain:
            return False, False
        return hi_plan.chain_role(self.plan, i, self.head_chained(s))

    def pwidth(self, s: int) -> int:
        """The rows of width s's programs: s, or a band's width (its requests x ``req_rows``: every width of the band
        runs the band's programs, ``band_of``)."""
        b = self.band_of.get(s) if hasattr(self, "band_of") else None
        return s if b is None else b * self.req_rows

    def _key(self, i: int, s: int) -> tuple:
        key = (self.plan[i].program_key, self.pwidth(s), *self.role(i, s))
        # (a tap layer's own program)
        return key + ("full_r_a",) if self.full_r_a(i, s) else key

    def full_r_a(self, i: int, s: int) -> bool:
        """Whether layer i's program at width s writes its R_A whole for a DSpark aux tap: the hi runtime's sharded residual
        (``layer.residual_layout``: RESIDUAL "sharded", or "auto" at s rows) leaves a chain's R_A the rank's columns,
        except in the tap layers' programs (``window_layer_blocks(full_r_a=True)``). Replicated, every R_A is whole
        already: False."""
        if i not in self.aux_ids:
            return False
        return self.L.residual_layout(self.tp, rows=self.pwidth(s)) == "sharded"

    def _chain_peer(self) -> tuple[str, ...]:
        """The chain's cross-launch peer regions, pinned first in every program: the hi runtime's ``layer.CHAIN_PEER`` (the
        MoE partial and, from 6 rows, the owner's gathered sums ``moe.part.dag``)."""
        return tuple(self.L.CHAIN_PEER)

    def _program(
        self,
        p: hi_plan.LayerPlan,
        s: int,
        push: bool = False,
        prev: bool = False,
        full_r_a: bool = False,
    ):
        from .hi.plan.program import Program, Scenario
        from .hi.runtime.kernel import MonoKernel

        nb, rq = self.split[s]
        # a band's program: its requests, the step's live (``begin_step``'s live words)
        dyn = s in self.band_of
        if dyn:
            nb, s = self.band_of[s], self.pwidth(s)
        # B requests of rq rows (one request: Scenario(1, s))
        L, sc = self.L, Scenario(nb, rq, self.tp)
        # the row-count-dependent choices (the 6-row MoE / seam exchanges)
        ch = {"push_moe": push, "prev_moe": prev, "rows": s}
        # a DSpark aux tap's R_A whole under a sharded residual (``full_r_a``)
        if full_r_a:
            ch["full_r_a"] = True
        sparse = self.cl[p.ratio] if p.kind == "consumer" else 0
        src = self.sources.get(p.layer_id)
        # SGPR spills into VGPR lanes (never memory): the hi runtime's rule for the kind at the width's rows and this TP (a
        # sharded residual's window kinds sit at the SGPR budget); the hook's layer programs are one kernel a layer
        spill = L.sgpr_spill_ok(
            sparse,
            src,
            rows=s,
            engram=p.engram,
            sparse_ratio=p.ratio,
            split=False,
            tp=self.tp,
        )
        if p.kind == "window0":
            blocks = L.window_layer_blocks(first=True, tp=self.tp, push_moe=push, rows=s,
                                           **({"full_r_a": True} if full_r_a else {}))  # fmt: skip
        # a non-first window layer: V4.1's layer 1, with its Engram
        elif p.kind == "window":
            blocks = L.window_layer_blocks(
                first=False, tp=self.tp, engram=p.engram, **ch
            )
        elif p.kind == "consumer":
            # the consumer's ratio (its source's entries of the same step read from the bf16 ckv)
            blocks = L.window_layer_blocks(
                first=False, tp=self.tp, sparse=sparse, sparse_ratio=p.ratio, **ch
            )
        else:
            # ratio 2: Source(state_slots=) -> rejection-safe Compressor state (module docstring)
            if self.index_layout != "preshuffle":
                ch["knobs"] = {"INDEX_KLAYOUT": self.index_layout}
                # TP2's MFMA-layout readers in a band spill 2 VGPRs at the block's 16 merge heads, none at 8
                if dyn and self.tp == 2 and p.kind == "index source":
                    ch["knobs"]["SPARSE_MHEADS"] = 8
            blocks = L.window_layer_blocks(first=False, tp=self.tp, source=self.sources[p.layer_id],
                                           engram=p.engram, **ch)  # fmt: skip
        stages = L.moe_route_early([st for b in blocks for st in b.stages(sc)])
        tag = "_".join(str(x) for x in p.program_key).replace(" ", "").replace("+", "")
        tag += (
            ("_push" if push else "")
            + ("_prev" if prev else "")
            + ("_tap" if full_r_a else "")
        )
        tag += "_band" if dyn else ""
        tag += "_mfma" if "knobs" in ch else ""
        dk = {"dyn_rows": True} if dyn else {}
        return MonoKernel(Program(f"atom_v41_{tag}_r{s}", sc, stages, sgpr_spill_ok=spill,
                                  peer_first=self._chain_peer() if self.chain else (), **dk))  # fmt: skip

    def _head_program(
        self, s: int, chain: bool, split: tuple | None = None, dyn: bool = False
    ):
        """The verify head's program at ``s`` rows (module docstring), ``split`` = (requests, rows a request) of its
        scenario (default one request): ``chain``, its fold sums the last layer's pushed MoE partials
        (``hi_plan.chain_role``); its R into ``r_h``. One program a (rows, requests, chain), shared by the widths that run
        it."""
        from .hi.plan.program import Scenario
        from .hi.runtime.kernel import MonoKernel
        from .hi.v41 import head as H

        self.H = H
        nb, rq = split or (1, s)
        if (s, nb, chain, dyn) not in self._heads:
            tag = (f"_b{nb}" if nb > 1 else "") + ("_band" if dyn else "")
            name = f"atom_v41_head{'_chain' if chain else ''}_sample_race_r{s}{tag}"
            # sampling (a row with T > 0: one draw in its top-p nucleus), the rows with top_p >= 1 raced in the dot
            # epilogue (module docstring)
            knobs = {"sample": True, "race": True}
            # (a band's head: launched with the step's live words, as its layers)
            if dyn:
                knobs["dyn"] = True
            self._heads[s, nb, chain, dyn] = MonoKernel(H.head_program(name, Scenario(nb, rq, self.tp), chain=chain,
                                                                       res_out="r_h", **knobs))  # fmt: skip
        return self._heads[s, nb, chain, dyn]

    def _draft_kern(self, key: tuple, make):
        """A draft kernel, built once a key: ``make()`` -> ``Program``."""
        from .hi.runtime.kernel import MonoKernel

        if key not in self.dr_kerns:
            self.dr_kerns[key] = MonoKernel(make())
        return self.dr_kerns[key]

    def _draft_plan(self, n: int, W: int) -> _DraftPlan:
        """The block draft of n requests: one request, stages 0-2 and the head at ``Scenario(1, W)``, chained (launch
        ids ``DRAFT_LAUNCH``); n requests with the hi runtime's stages at ``Scenario(n, W)``: the three stages once, then the
        head chained at n requests where it builds, else stage 2 a chain tail and an odd number of plain heads on
        their rows (ids 46-48: the plan's launch count stays even, so what follows it back to back on the draft's peer
        group alternates parity); where the stages, or the heads, at n requests do not build, the one-request plan once
        a request.
        """
        from .hi.emit.compile import ResourceError
        from .hi.plan.program import PlanError, Program, Scenario

        D, tp = self.D, self.tp

        def stage(s, m, tail=False):
            def make():
                roles = D.STAGE_ROLES
                # stage 2 sums stage 1's partials and stores its own MoE plain (no push): plain heads follow
                if tail:
                    D.STAGE_ROLES = {**roles, s: (roles[s][0], False, roles[s][2])}
                try:
                    return D.stage_program(f"atom_v41_draft{s}{'_tail' if tail else ''}_r{W}{f'_b{m}' if m > 1 else ''}",
                                           s, Scenario(m, W, tp))[1]  # fmt: skip
                finally:
                    D.STAGE_ROLES = roles

            return self._draft_kern(("stage", s, m, tail), make)

        def head(m, chain):
            from .hi.v41 import head as H

            def make():
                name = f"atom_v41_draft_head{'' if chain else '_plain'}_r{W}{f'_b{m}' if m > 1 else ''}"
                sc = Scenario(m, W, tp)
                if chain:
                    return D.head_program(name, sc, res_out="r_h")[1]
                blocks = (
                    H.head_blocks(tp, chain=False, rows=m * W, res_out="r_h")[:2]
                    + D.head_blocks(tp, m * W, "r_h")[2:]
                )
                return Program(name, sc, [st for b in blocks for st in b.stages(sc)])

            return self._draft_kern(("head", m, chain), make)

        one = [
            _DraftLaunch(
                ("stage", s, 1, False), stage(s, 1), DRAFT_LAUNCH[s], 0, 1, s > 0, s
            )
            for s in range(3)
        ]
        one.append(
            _DraftLaunch(
                ("head", 1, True), head(1, True), DRAFT_LAUNCH[3], 0, 1, True, 3
            )
        )
        if n == 1:
            return _DraftPlan(1, one)
        stages = None
        try:
            stages = [
                _DraftLaunch(
                    ("stage", s, n, False), stage(s, n), DRAFT_LAUNCH[s], 0, n, s > 0, s
                )
                for s in range(3)
            ]
        # this the hi runtime's stages at n requests
        except Exception as e:  # noqa: BLE001 - a build refusal
            logger.info("V4.1 mono hi: the draft stages at %d requests do not build (%s: %s): one request at a time",
                        n, type(e).__name__, str(e)[:160])  # fmt: skip
        if not self._agree(
            stages is not None
        ):  # (every rank one request at a time, or none)
            return _DraftPlan(n, one, per_request=True)
        last = None
        try:
            last = _DraftLaunch(
                ("head", n, True), head(n, True), DRAFT_LAUNCH[3], 0, n, True, 3
            )
        except Exception as e:  # noqa: BLE001 - the Markov head at n requests
            logger.info("V4.1 mono hi: the draft head at %d requests does not build (%s: %s): three plain heads "
                        "(or fewer) after a stage-2 tail", n, type(e).__name__, str(e)[:160])  # fmt: skip
        if self._agree(last is not None):
            return _DraftPlan(n, stages + [last])
        stages[2] = _DraftLaunch(
            ("stage", 2, n, True),
            stage(2, n, tail=True),
            DRAFT_LAUNCH[2],
            0,
            n,
            True,
            2,
        )
        # an odd number of heads (ids 46, 47, 48: alternating parity), so the plan has an even number of launches and
        # ends on the parity the one-request plan ends on: the launches that follow it back to back on the draft's peer
        # group (the next draft) start on the other parity
        # (past 4 requests, the batch bands' widths: three heads as even as they come, the largest first)
        sizes = (
            [n]
            if n <= 2
            else (
                [1, 1, 1]
                if n == 3
                else [2, 1, 1] if n == 4 else [-(-(n - i) // 3) for i in range(3)]
            )
        )
        heads, r0 = [], 0
        try:
            for lid, m in zip((DRAFT_LAUNCH[3], *HEAD_MORE[:2]), sizes, strict=False):
                heads.append(
                    _DraftLaunch(
                        ("head", m, False), head(m, False), lid, r0, m, False, 3
                    )
                )
                r0 += m
        # the Markov head at m requests (TP2's two-block head spills past one request)
        except (ResourceError, PlanError) as e:
            logger.info("V4.1 mono hi: the draft's plain heads at %d requests do not build (%s: %s): one request at "
                        "a time", n, type(e).__name__, str(e)[:160])  # fmt: skip
            heads = None
        if not self._agree(heads is not None):
            return _DraftPlan(n, one, per_request=True)
        if r0 != n:
            raise ValueError(
                f"V4.1 mono hi draft: {n} requests: the fallback's heads cover {r0}"
            )
        return _DraftPlan(n, stages + heads)


def served(cfg, topology, tp: int, index_layout: str = "preshuffle") -> HiPrograms:
    """The programs the hook serves with: its widths (``ROWS``) and context bucket (``hi_gate.MAX_CTX``), on ATOM's
    FP4 index plane in ``index_layout``."""
    return HiPrograms(cfg, topology, list(ROWS), MAX_CTX, tp, index_layout=index_layout)
