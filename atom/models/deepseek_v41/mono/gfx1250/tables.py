# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step kernel's pointer tables (``fptab`` / ``tp`` / ``ctxtab``), from the loaded model:

- ``fp``: FPTRS, then ATAB, then one LKEYS row a layer: the 40 target layers, the 3 draft stages, the draft entry's row
  (its a* = draft stage 0's attention front);
- ``tp``: ``tail.TPTRS`` (head, sampling, verify, draft head, Markov);
- ``ctxtab``: ``draft_ctx.CTPTRS`` (the draft context KV), its address in ``fp`` (FPTRS ``ctxtab``).

Every pointer is the address of a tensor this object keeps (``self.keep``). The STEP entries (``buffers.STEP``: step
metadata, input ids, the cache pool, the RoPE rows) get owned, graph-static buffers here (``self.step``) that the runner
fills before each launch; the cache pool itself is bound with ``bind_pool``. The indexer / compressor keys of a layer row
point at a zero buffer until their binding (lflags 0: those layer bodies off)."""

import torch

from atom.models.deepseek_v41.mono.gfx1250 import buffers
from atom.models.deepseek_v41.mono.gfx1250.weights import draft as draft_weights
from atom.models.deepseek_v41.mono.gfx1250.weights import layer as layer_weights
from atom.models.deepseek_v41.mono.gfx1250.weights import planes
from atom.mono.runtime.consensus import MonoUnsupported

NL, NDR = 40, 3
CTX_TAP_AT = (
    NL - 4
)  # lflags bit 4: the layers whose outputs feed the draft context (NL - 4 .. NL - 2)
T = buffers.T_MAX
ILW = buffers.ILW  # the scorer's row width: the widest bound a step may have
BCOLS = buffers.BCOLS  # block-table columns a request (16-token pages)
ENG_W, ENG_N = (
    6144,
    25600,
)  # Engram: embedding width; kv rows (4 x 5120 keys, then the 5120-wide value)


class StepTables:
    def __init__(self, K, model, drafter, scratch, recv, rank, tp_group):
        M, ST, CX = K["step"], K["tail"], K["draft_ctx"]
        dev = next(iter(scratch.values())).device
        self.keep = []
        target = [_find(model, f"layers.{i}") for i in range(NL)]
        stages = [_find(drafter.model, f"mtp.{j}") for j in range(NDR)]
        target_base = _base(model)
        self.target_blocks, self.draft_blocks = target, stages
        self._geom_key = None
        dmodel = drafter.model
        self.dummy = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
        # STEP buffers (filled a step): sized for the widest step
        z32 = lambda n, fill=0: torch.full((n,), fill, dtype=torch.int32, device=dev)
        self.step = {
            "kmeta": z32(T * 4),
            "knsel": z32(T),
            "ksel": z32(T * 512),
            "kvw": z32(T, -1),
            "rpos": torch.zeros(T, dtype=torch.int64, device=dev),
            "kmeta_d": z32(T * 4),
            "kvw_d": z32(T, -1),
            "rpos_d": torch.zeros(T, dtype=torch.int64, device=dev),
            "ksel_d": z32(NDR * T * 512),
            "knsel_d": z32(T),
            "rcos": torch.zeros(T * 32, dtype=torch.float32, device=dev),
            "rsin": torch.zeros(T * 32, dtype=torch.float32, device=dev),
            "dids": z32(buffers.TD_MAX),
            "ctx_kvw": z32(T, -1),
            "ctx_roff": z32(NDR),
            "ctx_rcos": torch.zeros(T * 32, dtype=torch.float32, device=dev),
            "ctx_rsin": torch.zeros(T * 32, dtype=torch.float32, device=dev),
            "tids": z32(T),  # this step's input ids (TARGET_ENTRY embeds them)
            "sparam": _sparam_default(
                T, dev
            ),  # each target row's sampling (sample.SPW words: ``_sparam_default``)
            "dgeom": z32(
                64
            ),  # window, slot_rows, ring_slots, num_slots, -, -, -, -, then the requests' slots (DRAFT_META_DEV)
        }
        # STATE (buffers.STATE): the prob drafter's q of the last kernel step, read by the next step's verification
        self.state = {
            "qlog": torch.zeros(
                8 * 5 * K["markov"].VR, dtype=torch.float32, device=dev
            ),
            "qids": z32(8 * 5, -1),
        }
        self.pool = self.dummy  # cache.pool, bound by bind_pool
        # the indexer (IDX: the FULL / REINDEX layers' scorer and top-512 selection; their selections are the later
        # layers' keys): per-step inputs, one selection buffer an owner layer, scratch shared by the layers
        self.idx = True
        # (integers, and ibat: a non-zero ibat runs the scorer; the rest default to a dummy buffer)
        self.zero_keys = {
            "engw",
            "engws",
            "enggw",
            "engemb",
            "ibat",
            "kring",
            "ccap",
            "wcap",
            "cmain",
            "crpp",
            "cprow",
            "bstride",
            "irpp",
            "ilw",
            "imain",
            "lflags",
            "cstate",
        }
        if self.idx:
            self.step.update(
                ibat=z32(T, -1), ibound1=z32(T), ibound2=z32(T), btab=z32(8 * BCOLS)
            )
            # the selecting layers: FULL (their own top-512) and REINDEX (24..36: they score layer 20's top-2048
            # candidate blocks, icand)
            self.cand = True
            self.selecting = [
                i for i in range(NL) if target[i].attn.indexer is not None
            ]
            self.isel = {i: z32((T + 1) * 512, -1) for i in self.selecting}
            self.irow = {i: z32((T + 1) * 512) for i in self.selecting}
            self.ilog = torch.zeros((T + 1) * ILW, dtype=torch.float32, device=dev)
            self.ibmax = torch.zeros(
                (T + 1) * (ILW // 8), dtype=torch.float32, device=dev
            )
            self.icand = z32((T + 1) * 2048)
            self.idx_w = {}
            for i in self.selecting:
                ix = target[i].attn.indexer
                w = layer_weights.checkpoint_weight(ix.wq_b)
                if tuple(w.shape) != (4096, 1280) or tuple(
                    ix.wq_b.weight_scale.shape
                ) != (128, 40):
                    raise MonoUnsupported(
                        f"layer {i} indexer wq_b {tuple(w.shape)} / {tuple(ix.wq_b.weight_scale.shape)}"
                    )
                self.idx_w[i] = {
                    "iqw": w.contiguous(),
                    "iqws": ix.wq_b.weight_scale.data.view(torch.uint8).contiguous(),
                    "iww": ix.weights_proj.weight.data.to(torch.bfloat16).contiguous(),
                }
            self.keep.append(self.idx_w)
            # the same layers' compressor -- this step's compressed main KV row, index row and ring entries, which the
            # attention and the scorer of later rows of the step read (compressor)
            self.comp = True
            if self.comp:
                self.step.update(
                    cplan1=z32(T * 4, -1),
                    cplan2=z32(T * 4, -1),
                    wplan1=z32(T * 4, -1),
                    wplan2=z32(T * 4, -1),
                    slots=z32(8),
                )
                self.comp_w = {}
                for i in self.selecting:
                    attn = target[i].attn
                    if attn.compressor is None:  # (REINDEX: no compressor of its own)
                        continue
                    cmp_, ix, ratio = attn.compressor, attn.indexer, attn.spec.ratio
                    cw = cmp_.wkv_gate.weight.data
                    if (
                        tuple(cw.shape) != (512 * (2 if ratio == 2 else 1), 5120)
                        or cw.dtype != torch.bfloat16
                    ):
                        raise MonoUnsupported(
                            f"layer {i} compressor wkv_gate {tuple(cw.shape)} {cw.dtype}"
                        )
                    ape = torch.zeros(
                        2, 512, dtype=torch.float32, device=dev
                    )  # (stage_crow reads 2 x 512: ratio 1 too)
                    ape[: cmp_.ape.shape[0]].copy_(cmp_.ape)
                    self.comp_w[i] = {
                        "cwkv": cw.contiguous(),
                        "cnw": cmp_.norm.weight.data.to(torch.bfloat16).contiguous(),
                        "cape": ape,
                        "cwk": ix.wk.weight.data.to(torch.bfloat16).contiguous(),
                        "cknw": ix.k_norm.weight.data.to(torch.bfloat16).contiguous(),
                    }
                self.keep.append(self.comp_w)
        self.specs = [target[i].attn.spec for i in range(NL)]
        # Engram (layers k with an EngramOp): row k - 1 (lflags bit 7) quantizes this step's embedding rows and projects
        # them (kv = emb . wkv^T); layer k's attention front gates the value into R'. The runner copies the staged
        # embedding rows into engemb[k] each step.
        self.engram = {}
        for k in range(1, NL):
            op = getattr(target[k], "engram", None)
            if op is None:
                continue
            w = planes.checkpoint_weight(op.wkv)
            ws = op.wkv.weight_scale.data.view(torch.uint8)
            if (
                tuple(w.shape) != (ENG_N, ENG_W)
                or tuple(ws.shape) != (ENG_N // 32, ENG_W // 32)
                or op.gate_weight is None
                or tuple(op.gate_weight.shape) != (4, 5120)
            ):
                raise MonoUnsupported(
                    f"layer {k} engram wkv {tuple(w.shape)} / {tuple(ws.shape)}"
                )
            self.engram[k] = {
                "engw": w.contiguous(),
                "engws": ws.contiguous(),
                "enggw": op.gate_weight.float().contiguous(),
                "engemb": torch.zeros(T, ENG_W, dtype=torch.bfloat16, device=dev),
            }
        self.keep.append(self.engram)
        # the layer rows' weights
        rows = []
        for i in range(NL):
            nxt = target[i + 1] if i + 1 < NL else None
            rows.append(
                layer_weights.layer_row(
                    target[i], nxt, None if nxt is not None else target_base.norm
                )
            )
        for j in range(NDR):
            nxt = stages[j + 1] if j + 1 < NDR else None
            rows.append(
                layer_weights.layer_row(
                    stages[j], nxt, None if nxt is not None else dmodel.mtp[-1].norm
                )
            )
        entry = layer_weights.attention_front(
            stages[0]
        )  # the draft entry's row: draft stage 0's attention front
        self.keep += rows + [entry]
        # each layer's RoPE cache (ROPE_POS: bf16 [pos][32], row = rpos): the window one for ratio 0, else the global one;
        # the draft's own
        self.ropes = [
            (
                target_base.global_rope
                if target[i].attn.spec.ratio
                else target_base.window_rope
            )
            for i in range(NL)
        ] + [dmodel.rope] * NDR
        # the routed experts, read in place (the production MoE's own tensors)
        mpl = [planes.routed_experts(target[i].ffn.experts) for i in range(NL)] + [
            planes.routed_experts(stages[j].ffn.experts) for j in range(NDR)
        ]
        shp = [planes.pack_shared(target[i].ffn.shared_experts) for i in range(NL)] + [
            planes.pack_shared(stages[j].ffn.shared_experts) for j in range(NDR)
        ]
        self.keep += shp
        dw = draft_weights.bind(target_base, dmodel, tp_group)
        self.keep.append(dw)
        # fp: FPTRS, ATAB
        src = dict(scratch)
        src.setdefault(
            "tpep", torch.zeros(16, dtype=torch.uint8, device=dev)
        )  # the step epoch word (TP_EPOCH)
        self.keep.append(src["tpep"])
        src.update({k: v for k, v in rows[0].items()})
        src.update(shplane=shp[0], emb=dw["emb"], noise=dw["noise"])
        src.update({k: v for k, v in self.step.items() if not k.startswith("ctx_")})
        # the layer-to-layer state: a layer's FFN front reads the gates the previous attention front wrote (its R comes
        # from LDS, RKEEP: layer 0's from the target entry's front)
        src.update(rcos=self.ropes[0].cos_cache, rsin=self.ropes[0].sin_cache)
        src.update(
            pre=scratch["anpre"],
            post=scratch["anpost"],
            comb=scratch["ancomb"],
            R=scratch["aRout"],
        )
        self.recv = recv
        ctx = {
            "aux8": scratch["aux8"],
            "auxs": scratch["auxs"],
            "hpre": scratch["hpre"],
            "hpref": scratch["hpref"],
            "h8": scratch["h8"],
            "hs": scratch["hs"],
            "kvp": scratch["kvp"],
            "mpw": dw["mpw"],
            "mps": dw["mps"],
            "mnw": dw["mnw"],
            "rcos": self.step["ctx_rcos"],
            "rsin": self.step["ctx_rsin"],
            "kvw": self.step["ctx_kvw"],
            "roff": self.step["ctx_roff"],
        }
        for j in range(NDR):
            ctx.update(
                {
                    f"wkv{j}": dw[f"wkv{j}"],
                    f"wks{j}": dw[f"wks{j}"],
                    f"knw{j}": dw[f"knw{j}"],
                }
            )
        self.ctx_src = ctx
        self.ctxtab = torch.zeros(len(CX.CTPTRS), dtype=torch.int64, device=dev)
        src["ctxtab"] = self.ctxtab
        tsrc = dict(scratch)
        tsrc.update(
            hw=dw["hw"],
            mw1=dw["mw1"],
            mw2=dw["mw2"],
            dids=self.step["dids"],
            sparam=self.step["sparam"],
            **self.state,
        )
        miss = [k for k in M.FPTRS if k not in src] + [
            k for k in ST.TPTRS if k not in tsrc
        ]
        miss += [
            k for k in M.ATAB if k not in src and not k.startswith("recv") and k != "kv"
        ]
        miss += [k for k in CX.CTPTRS if k not in ctx and k != "ring"]
        if miss:
            raise MonoUnsupported(f"step tables: no buffer for {miss[:6]}")
        self.src, self.tsrc = src, tsrc
        self.rows = [
            self._row(M, rows[Ly], mpl[Ly], shp[Ly], Ly) for Ly in range(NL + NDR)
        ] + [self._row(M, {**rows[NL], **entry}, mpl[NL], shp[NL], NL + NDR)]
        # the target entry's row (NL + NDR + 1): layer 0's attention front (TARGET_ENTRY)
        tentry = layer_weights.attention_front(target[0])
        self.keep.append(tentry)
        self.rows.append(
            self._row(M, {**rows[0], **tentry}, mpl[0], shp[0], NL + NDR + 1)
        )
        self.M, self.ST, self.CX = M, ST, CX
        self.dev = dev
        self.fptab = torch.zeros(
            len(M.FPTRS) + len(M.ATAB) + len(self.rows) * len(M.LKEYS),
            dtype=torch.int64,
            device=dev,
        )
        self.tp = torch.tensor(
            [tsrc[k].data_ptr() for k in ST.TPTRS], dtype=torch.int64, device=dev
        )
        self._fill()

    def _row(self, M, w, mpl, shp, Ly):
        """One LKEYS row (pointer values): the layer's weights, planes, RoPE rows, selection metadata; lflags / kring and
        the indexer / compressor keys stay 0 / the zero buffer."""
        p = {k: v.data_ptr() for k, v in w.items()}
        rope = self.ropes[min(Ly, NL + NDR - 1) if Ly != NL + NDR + 1 else 0]
        p.update({k: v.data_ptr() for k, v in mpl.items()})
        p.update(
            shplane=shp.data_ptr(),
            rcos=rope.cos_cache.data_ptr(),
            rsin=rope.sin_cache.data_ptr(),
        )
        if Ly < NL or Ly == NL + NDR + 1:
            p.update(
                ksel=self.step["ksel"].data_ptr(),
                knsel=self.step["knsel"].data_ptr(),
                kring=0,
            )
            p["lflags"] = (
                16 if CTX_TAP_AT <= Ly <= CTX_TAP_AT + 2 else 0
            )  # (the entry row: 0)
            if Ly < NL and Ly + 1 in self.engram:  # the row before an Engram layer
                p["lflags"] |= 128
                p.update({k: v.data_ptr() for k, v in self.engram[Ly + 1].items()})
            if self.idx and Ly < NL:
                spec = self.specs[Ly]
                if (
                    spec.ratio and spec.topk_owner in self.irow
                ):  # a layer reading a selection: its owner's rows, its bound
                    p.update(
                        ksel=self.irow[spec.topk_owner].data_ptr(),
                        knsel=self.step[f"ibound{spec.ratio}"].data_ptr(),
                    )
                if (
                    Ly in self.idx_w
                ):  # a selecting layer: its scorer (geometry ints: set_geometry)
                    p["lflags"] |= 1 | 8
                    if spec.produces_candidates and self.cand:
                        p["lflags"] |= 32
                    elif (
                        spec.candidate_source is not None
                    ):  # REINDEX: the candidates' columns, lifted to real rows
                        p["lflags"] |= 64
                    p.update({k: v.data_ptr() for k, v in self.idx_w[Ly].items()})
                    p.update(
                        ibat=self.step["ibat"].data_ptr(),
                        ibound=self.step[f"ibound{spec.ratio}"].data_ptr(),
                        ilog=self.ilog.data_ptr(),
                        ilw=ILW,
                        isel=self.isel[Ly].data_ptr(),
                        irow=self.irow[Ly].data_ptr(),
                        ibmax=self.ibmax.data_ptr(),
                        icand=self.icand.data_ptr(),
                        btab=self.step["btab"].data_ptr(),
                        bstride=BCOLS,
                    )
                    if self.comp and Ly in self.comp_w:
                        r = spec.ratio
                        p["lflags"] |= 2 | (4 if r == 1 else 0)
                        p.update({k: v.data_ptr() for k, v in self.comp_w[Ly].items()})
                        p.update(
                            cplan=self.step[f"cplan{r}"].data_ptr(),
                            wplan=self.step[f"wplan{r}"].data_ptr(),
                            ccap=T,
                            wcap=T,
                            slots=self.step["slots"].data_ptr(),
                        )
        else:
            j = min(Ly - NL, NDR - 1)
            p.update(
                ksel=self.step["ksel_d"].data_ptr() + j * T * 512 * 4,
                knsel=self.step["knsel_d"].data_ptr(),
                kring=0,
                lflags=0,
            )
        return [
            p.get(k, 0 if k in self.zero_keys else self.dummy.data_ptr())
            for k in M.LKEYS
        ]

    def bind_pool(self, pool: torch.Tensor):
        self.pool = pool
        self._fill()

    def set_geometry(self, cache, layer_ids) -> None:
        """The cache's pool, each layer's ring start (the row's ``kring``) and the draft context's ring starts (``roff``);
        refilled only when the pool or its page count changes."""
        key = (cache.pool.data_ptr(), cache.num_pages)
        if key == self._geom_key:
            return
        self._geom_key = key
        ix = self.M.LKEYS.index("kring")
        for Ly, lid in enumerate(layer_ids):
            self.rows[Ly][ix] = int(
                cache.geometry.window(lid, cache.num_pages).ring_start
            )
        if self.idx:
            g = cache.geometry
            L = self.M.LKEYS
            for Ly in self.idx_w:
                spec = self.specs[Ly]
                if g.packed:
                    raise MonoUnsupported(
                        "single-launch step: a packed pool (byte-addressed rows)"
                    )
                vals = {
                    "iplane": cache.index_units[spec.kv_owner].values.data_ptr(),
                    "irpp": g.rows_per_page(spec.ratio),
                    "imain": g.main_offset(spec.kv_owner),
                    "cprow": g.page_bytes // g.row_bytes,
                }
                if (
                    self.comp and Ly in self.comp_w
                ):  # the owner's (the layer's own: FULL) main field, index plane and ring state
                    kv_s, sc_s = cache.compress_state(spec.kv_owner)
                    for t_ in (kv_s, sc_s):
                        if (
                            t_.dtype != torch.float32
                            or t_.stride(0) * 4 != g.state_bytes
                            or tuple(t_.shape[1:]) != (7, 512)
                            or t_.stride(1) != 512
                            or t_.stride(2) != 1
                        ):
                            raise MonoUnsupported(
                                f"compress state {tuple(t_.shape)} {t_.stride()} {t_.dtype}, slot {g.state_bytes} B"
                            )
                    vals.update(
                        cmain=vals["imain"],
                        crpp=vals["irpp"],
                        cplane=vals["iplane"],
                        ckv=kv_s.data_ptr(),
                        cscore=sc_s.data_ptr(),
                        cstate=g.state_bytes,
                    )
                for k, v in vals.items():
                    self.rows[Ly][L.index(k)] = int(v)
        roff = [
            int(cache.geometry.window(lid, cache.num_pages).ring_start)
            for lid in layer_ids[NL : NL + NDR]
        ]
        self.step["ctx_roff"][:NDR].copy_(torch.tensor(roff, dtype=torch.int32))
        self.pool = cache.pool
        self._fill()

    def ptr(self, k) -> int:
        """An ATAB / FPTRS name's pointer (the pool for ``kv``)."""
        return (self.pool if k == "kv" else self.src[k]).data_ptr()

    def _fill(self):
        M, CX = self.M, self.CX
        rv = dict(zip(("recv0", "recv1", "recv2", "recv3"), self.recv))
        head = [self.src[k].data_ptr() for k in M.FPTRS]
        head += [
            rv[k] if k in rv else (self.pool if k == "kv" else self.src[k]).data_ptr()
            for k in M.ATAB
        ]
        vals = head + [v for row in self.rows for v in row]
        self.fptab.copy_(torch.tensor(vals, dtype=torch.int64))
        self.ctxtab.copy_(
            torch.tensor(
                [
                    (self.pool if k == "ring" else self.ctx_src[k]).data_ptr()
                    for k in CX.CTPTRS
                ],
                dtype=torch.int64,
            )
        )


def _sparam_default(rows, dev):
    """[rows][8] 32 B a row (sample.SPW): invT f32, top_p f32, seed i32, greedy i32 | invTd f32, hasq i32,
    qsrc i32, seed_acc i32 -- greedy rows (argmax, no q) until ``runner._sampling`` fills them.
    """
    sp = torch.zeros(rows, 8, dtype=torch.float32, device=dev)
    sp[:, 0], sp[:, 1], sp[:, 4] = 1.0, 1.0, 1.0
    sp.view(torch.int32)[:, 3] = 1
    return sp


def _find(root, suffix):
    hit = [
        m
        for n, m in root.named_modules()
        if n.endswith(suffix) and n.split(".")[-1] == suffix.split(".")[-1]
    ]
    if not hit:
        raise MonoUnsupported(f"no module {suffix}")
    return hit[0]


def _base(model):
    """The V4.1 base model (``embed``, ``head``, ``norm``, ``layers``)."""
    for m in model.modules():
        if (
            hasattr(m, "embed")
            and hasattr(m, "head")
            and hasattr(m, "norm")
            and hasattr(m, "layers")
        ):
            return m
    raise MonoUnsupported("no V4.1 base model (embed / head / norm / layers)")
