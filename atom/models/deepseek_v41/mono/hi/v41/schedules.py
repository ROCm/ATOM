# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Schedule knobs of the DeepSeek-V4.1 layers.

``DEFAULT`` is every knob of ``recipe.window_layer_blocks`` and its placement passes, by name, at the value the
layers run with; ``"auto"`` there means "the schedule decides". ``SCHEDULES[(kind, rows)]`` holds what a layer kind
at a row count runs with where a measurement on MI355X at TP4 decided it (``rows`` None: the row count is not known to
the build). A knob still ``"auto"`` after that takes its value in ``AUTO``; the few rules that depend on the build
itself (the chain's exchange, the TP degree, the merge's cache copies) are the recipe's (``recipe`` module docstring,
"Knobs"). An index source at a context bucket from ``IDX_LONG`` on takes its kind's ``LONG`` (and ``LONG_ROWS``) on
top. A build's own ``knobs`` dict goes over all of it (``knobs(over=)``).

Kinds are the model's layer kinds (``kind_of``): ``window`` (layer 0), ``window+engram`` (layer 1),
``ratio2_kv_source`` (2, 8), ``ratio2_kv_source+engram`` (14), ``ratio1_kv_source`` (20), ``ratio1_index_source``
(24 .. 36), ``ratio2_consumer`` / ``ratio1_consumer`` (the rest; ``consumer``: a consumer built without its ratio).
A layer built as two kernels (the attention kernel, then the FFN kernel) takes the kind ``<kind>/split``.

Every entry's comment gives the measurement that decided it.
"""

from __future__ import annotations

import copy

# ---------------------------------------------------------------------------------------------------------------- knobs

DEFAULT: dict = {
    # attention projections
    "WO_A_COMPUTE": "bf16",  # wo_a's arithmetic: "bf16" (the reference's plain einsum) or "scaled" (the MX MFMA)
    "WO_A_PREWIDEN": "auto",  # bf16 wo_a: weight units a wave widens before its input lands: an int, 0 or "auto"
    "WO_B_PARTIAL": "f32",  # TP: wo_b's exchanged partial, "f32" (D10 vs the TP1 reference) or "bf16"
    "WO_B_SUM": "2shot",  # TP, one kernel: "2shot" (column quarters to their owners), "consumer", "producer"
    "WQB_ROWS": "auto",  # wq_b's output rows a task: None (the plan's), an int, "auto"
    "WKV_KSPLIT": "auto",  # wkv's K slices: None (the plan's), an int, "auto"
    "INDEX_KFMT": "fp4",  # the Indexer's key cache: "fp4" (its kpack stage fills it) or "bf16"
    # the FP4 index plane's layout in ATOM's storage (``config.index_cache_layout``): "preshuffle" or "mfma"
    # (``device.gfx950.paged``); the deployment's, never a (kind, rows) entry
    "INDEX_KLAYOUT": "preshuffle",
    "XA_PROTO": "flag",  # attn_norm -> wq_a / wkv: "tagged" pairs or "flag" (dense words + a flag a norm task)
    "Q_NORM": "stage",  # q_norm: "stage" or "wq_b" (wq_b's NormIn prologue normalizes the qr rows)
    # the seams (HCSeam)
    "OWN_QUADS": "auto",  # seam F's two-shot owner stage: element quads a task, an int or "auto"
    "HC_DIRECT": True,  # seam A: a thread folds a (row, 2 columns) and puts LIN at once
    "HC_LDS_PAD": "auto",  # the seams' R / hc_fn rows at a pitch of KT + 4 words: "a", "f", "af", "" or "auto"
    "SEAM_NORM": False,  # attn_norm fused into seam A's slices (SeamNorm): measured worse
    "HC_EARLY": "auto",  # the seams' previous-gate values issued at the slice task's start
    "HC_ENTRY": "auto",  # a chained seam A's first poll and loads at the stage's entry; the recipe's rule
    "HCF_GATE_AT": "auto",  # seam F's gate stage on the CTAs from this one: None, an int or "auto"
    "R_CHANNEL": True,  # seam F reads seam A's R from a tagged channel (False: the plain arg, stale across XCDs)
    "RESIDUAL": "auto",  # TP, one kernel: R "replicated", "sharded" (column-owned seams) or "auto"
    "R_SHARD_ROWS": 6,  # RESIDUAL "auto": sharded from this many rows on (1 row: the MoE's one-shot push stays)
    "R_SHARD_MAX_ROWS": 24,  # ... up to this many (a step's rows; B = 2-4 requests of 6: 12 / 18 / 24)
    "SHARD_SEAM_A": "start",  # sharded, a chain: seam A's fold + LIN "start" (the next launch) or "end" (this one's)
    "SHARD_SPILL_OK": False,  # sharded: SGPR spills into VGPR lanes allowed (where measured: no loss)
    "SHARD_ENGRAM_A": "replicated",  # sharded, Engram layers: seam A after the Engram "sharded" or "replicated"
    "SHARD_COMP_TILES": None,  # sharded: the Compressor's COMP_TILES instead (None: COMP_TILES)
    "FULL_R_A": False,  # sharded: every layer writes R_A whole (the DSpark aux taps' form, ``full_r_a``)
    "R_GATHER": "late",  # sharded: a whole R's gather "late" (a stage of its own, off the path) or in the "slice"
    "RS_PACK": "auto",  # TP: every two-shot exchange's reduce-scatter partials (wo_b -> seam F, the MoE combine ->
    #                     its owner / the next seam A): "pair" ((fp32, tag), 8 B a value), "f32p" (a lane's tagged
    #                     quad of 3 + a pair, 6 B), "f32q" (3 values + tags a 16 B quad across lanes, 5.5 B; slower) or
    #                     "auto" (f32p from RS_PACK_ROWS rows), the same bits; a step's, never a (kind, rows) entry
    #                     (CHAIN_KNOBS)
    "RS_PACK_ROWS": 12,  # RS_PACK "auto": f32p from this many rows (a step's; 6 rows unconfirmed)
    # norms
    "FFN_NORM": "stage",  # the router's input: "stage" (ffn_norm's bf16 copy), "router" or "auto"
    "NORM_SPLIT": "auto",  # norm stages' tasks a row: "ffn_norm:10;q_norm:5", "" or "auto"
    "FFN_NORM_SPLIT": 4,  # NORM_SPLIT "auto": ffn_norm's tasks a row
    "SRC_NORM_SPLIT": "auto",  # one-kernel source layers: attn_norm's tasks a row
    # the attention
    "RING_FIRST": "auto",  # the attention's window task lands the ring rows before its q poll
    "ATTN_DSPLIT": 2,  # window layers: MLAWindowAttn tasks a (row, tile) by P V dims
    "RING_SLOTS": 128,  # the window ring's slots: 128 (the reference's) or 128 + drafts (rejection-safe)
    "NEW_EARLY": "auto",  # KV sources: the chunk tasks poll the step's emitted entries at task start
    "SPARSE_DSPLIT": "auto",  # the sparse attention's window / chunk tasks by P V dims, 1 or 2
    "SPARSE_SMERGE": "auto",  # MLASparseAttn(smerge=): the merge takes the raw scores
    "SMERGE_PLACE": True,  # ... and its placement: wo_a off the merge's CTAs
    "SPARSE_CKV4": False,  # ... the merge's compressed V from the cache's FP4 copy: False, True, an int or "auto"
    "SPARSE_RING8": False,  # ... the merge's window V from the ring's MXFP8 copy: as SPARSE_CKV4
    "SPARSE_WSPLIT": True,  # ... the window's two chunks as two tasks
    "SPARSE_MHEADS": "auto",  # ... the merge tasks' heads (1, 2, 4, 8, 16; 0: the block's) or "auto"
    "SPARSE_MHEADS_SH": "auto",  # ... and with a sharded residual: those heads instead (0: SPARSE_MHEADS's)
    "SPARSE_LEAN": True,  # ... the chunk / merge tasks' lean index math and v_perm_b32 V^T pairs
    "SPARSE_VTR": "auto",  # ... the merge's V key-major, P V's B by LDS transpose reads (same bits) or "auto"
    "SPARSE_KFIRST": True,  # ... the chunk tasks' gathered keys into LDS before their q poll (same bits)
    "WO_A_HOLD": "auto",  # s_sleep units at wo_a's entry (Stage.hold), an int or "auto"
    # the KV / index sources
    "IDX_WQB_ROWS": "auto",  # the Indexer's wq_b output rows a task: None (the block's), an int or "auto"
    "IDX_AFTER_POOL": "auto",  # the Indexer's first stage on the CTA after comp.pool's last: bool or "auto"
    "KPACK_LANES": "auto",  # the Indexer's kpack converts a key to FP4 over 64 lanes
    "IDX_SUBTILES": "auto",  # the Indexer's 16-position subtiles a wave: an int, or "auto" (the block's rule)
    "IDX_COMPACT": "auto",  # candidate readers from this n_max (> 16384) score only the candidate blocks; 0 off
    "IDX_SHORT": False,  # full-width sources with wide tiles: a narrow score stage for short rows (Indexer short_tiles)
    "IDX_LIVE": False,  # the Indexer's score / TopK tasks only those the step's rows reach (Indexer live)
    "IDX_PERSIST": 0,  # the Indexer's score tasks a (request, row group), each looping over its tiles; -1 the grid's
    #                    share; 0 a task a tile (Indexer persist)
    "IDX_ROWS": 0,  # the Indexer's score rows a task (0: the block's; Indexer rows_per_task)
    "IDX_SEG": 0,  # the Indexer's TopK segment (0: the block's; Indexer seg)
    "IDX_CSEG": 0,  # layer 20's candidate TopK segment (0: the block's; Indexer cseg)
    "IDX_SPLIT": False,  # TP: the Indexer's positions split across the ranks, the TopKs merged (needs IDX_PERSIST)
    "IDX_PUNROLL": 1,  # IDX_PERSIST: tiles a persistent score task loads together (Indexer punroll)
    "IDX_ITEMS": False,  # IDX_PERSIST: its tasks shared among the step's requests by their tiles, not IDX_PERSIST a
    #                      (request, row group) whatever its length (Indexer items)
    "IDX_PCAP": 0,  # IDX_PERSIST + IDX_LIVE: requests up to this many context tokens take the tile tasks (Indexer pcap,
    #               in the layer's compressed positions: / ratio)
    "IDX_PNARROW": False,  # IDX_PCAP: the tile stage at one subtile a wave (Indexer pnarrow)
    "IDX_ROWS_LONG": 0,  # the Indexer's score rows a task for requests past IDX_RCAP (with IDX_PCAP: the persistent
    #                      stage's), the block's rows below (Indexer rows_long); 0 off
    "IDX_RCAP": 0,  # IDX_ROWS_LONG without IDX_PCAP: requests past this many context tokens take it (Indexer rcap,
    #               in the layer's compressed positions: / ratio)
    "IDX_LEAN_EPI": False,  # the Indexer's score epilogue in fewer VALU ops, the same scores (Indexer lean_epi)
    "IDX_TOPK_PACKED": False,  # the selection's TopK: level 0's lists one tagged word an entry (TopK packed)
    "IDX_XHOIST": "auto",  # split TopKs: the final level's peer addresses before its select (TopK xhoist)
    "IDX_CLIST_SPARSE": False,  # compact readers: the candidate list from the row's live words / set bits
    "IDX_CMASK_LDS": "auto",  # layer 20: the candidate bitmap's words built in LDS (an OR an entry)
    "IDX_BMAX_BF16": "auto",  # layer 20: the block maxima a BF16 channel, the candidate TopK in bf16 mode (same keys)
    "IDX_CAND_APART": "auto",  # the candidate source's attention path off the candidate chain's CTAs
    #                            (``recipe.cand_apart``): a bool, 2 (its selection's TopK too) or "auto"
    "IDX_LONG": "auto",  # index sources from this context bucket on (tokens: the Indexer's n_max x ratio): their
    #                      kind's ``LONG`` as their schedule entry (``knobs(tokens=)``); 0 off; "auto": the
    #                      schedule's
    "COMP_FIRST": "auto",  # the Compressor's stages before wq_a / wkv in program order
    "COMP_KSLICES": 4,  # the Compressor's dots' K slices (GatheredDot dense mode)
    "COMP_TILES": "auto",  # ... and 16-column tiles a task, an int or "auto"
    # precision: ATOM's arithmetic (False / True / True, the default: merge rule 4, no op below ATOM's default path;
    # the gate's reference is the ATOM-arithmetic one, keyed by these three) or the official
    # model's (True / False / False, the hw reference's). Numerics choices, never schedule entries
    "COMPRESS_QAT": False,  # the KV sources' compressed-cache entry: the rotated bf16 latent as is (ATOM's bf16 pool;
    #                         Compressor(qat=False)) or FP4 QAT'd (True: the official model's)
    "IDX_SCORE_F32": True,  # the index scores in f32 (q . k, relu(s) w, the head sum, an F32 channel, TopK's f32 mode;
    #                         layer 20's block maxima from them; ATOM's FP4-plane scorer: Indexer(scores_f32=)); on,
    #                         IDX_TOPK_PACKED / IDX_BMAX_BF16 / IDX_LEAN_EPI (bf16 scores only) resolve off
    "ATTN_OUT_F32_ROPE": True,  # the attention output's inverse RoPE quantized (MXFP8) from f32 (ATOM's order;
    #                             MLAWindowAttn / MLASparseAttn(unrope_f32=)); False: bf16-rounded first (official)
    # the router and the MoE
    "ROUTER_TILES": 4,  # the router dot's 16-column tiles a task
    "ROUTER_KSLICES": "auto",  # the router dot's K slices, an int or "auto"
    "ROUTER_SCORES": "fused",  # the route's scores: "wave0", "waves" or "fused"
    "SHARED": "auto",  # the shared expert: "fold" (in MoEExperts, sharded), "linear" or "auto" (fold at TP > 1)
    "SHARED_ORDER": "fold",  # the folded shared expert at TP: "fold" (in the routed partial) or "ref"
    "MOE_LAYOUT": "intra",  # TP: the routed experts' split, "intra" (every expert's width slice) or "expert"
    "MOE_UG_G": "auto",  # MoEExperts ug_g (row groups a routed ug task): None, an int, {rows: g} or "auto"
    "MOE_DOWN_G": "auto",  # ... down_g, as MOE_UG_G
    "MOE_RT_POLL": "picks",  # MoEExperts rt_poll: "all" or "picks"
    "MOE_RT_LEAN": "auto",  # MoEExperts rt_lean: the rt table on fewer barriers (same table), a bool or "auto"
    "MOE_UG_RING": 0,  # MoEExperts ug_ring: a ug task's first units a wave in an LDS ring the previous one fills
    "MOE_DOWN_RING": 0,  # MoEExperts down_ring: a down task's first units a wave copied into an LDS ring before it
    #                     polls mid (they stream during the wait)
    "MOE_SUM": "auto",  # TP: the MoE partial's exchange, "push", "2shot", "own" or "auto"; the recipe's rule
    "MOE_OWN_ROWS": 6,  # MOE_SUM "auto": a chain's exchange is "own" from this many rows on
    "MOE_OWN_QUADS": "auto",  # the MoE owner stage's element quads a task; "auto": 128 for a chain's "own"
    # placement and the launch
    "NUMA_DUAL": 512,  # KiB: dual-copy (Region.home "numa2") tagged regions up to this size; 0: off
    "SPILL_OK": "auto",  # Program.sgpr_spill_ok (SGPR spills into VGPR lanes, never memory): bool or "auto" (the
    #                      recipe's rule: the sources and the consumers with RING_FIRST; ``recipe.sgpr_spill_ok``)
    # the Engram's token-only half (engram_kv_blocks)
    "ENGRAM_LOOK_DST": True,  # TP: its look puts to one rank a task (Engram(look_dst=))
    "ENGRAM_STORE_CHUNKS": 8,  # TP: its store tasks a row (Engram(store_chunks=))
    "ENGRAM_STAGED": False,  # from rows already looked up (plain <name>.rows) instead of the table shard in HBM
}

# A knob still "auto" after the schedule: its value
AUTO: dict = {
    "WQB_ROWS": 48,  # 171 wq_b tasks at TP4, so the narrow stages after it get CTAs of their own
    "WO_A_PREWIDEN": 0,
    "WKV_KSPLIT": 4,
    "OWN_QUADS": 512,
    "HC_LDS_PAD": "af",
    "HC_EARLY": False,
    "SRC_NORM_SPLIT": 2,
    "RING_FIRST": True,
    "NEW_EARLY": False,
    "SPARSE_DSPLIT": 1,
    "SPARSE_SMERGE": True,
    "SPARSE_MHEADS": 0,  # the block's own (4 up to 6 rows, 16 above)
    "SPARSE_MHEADS_SH": 0,
    "SPARSE_VTR": True,
    "IDX_WQB_ROWS": None,
    "KPACK_LANES": False,
    "IDX_SUBTILES": 0,
    "IDX_COMPACT": 0,
    "COMP_FIRST": True,
    "COMP_TILES": 1,
    "ROUTER_KSLICES": 10,
    "MOE_RT_LEAN": False,
    "SHARED_ORDER": "fold",
    "MOE_UG_G": None,  # MoEExperts.groups()' rule
    "MOE_DOWN_G": None,
    "IDX_LONG": 0,
    "IDX_CAND_APART": False,
    "IDX_CMASK_LDS": False,
    "IDX_BMAX_BF16": False,
    "IDX_XHOIST": False,
}


# ------------------------------------------------------------------------------------------------------------- kinds

WINDOW, ENGRAM = "window", "window+engram"
KV2, KV2E, KV1, IDX1 = (
    "ratio2_kv_source",
    "ratio2_kv_source+engram",
    "ratio1_kv_source",
    "ratio1_index_source",
)
C2, C1, CONSUMER = "ratio2_consumer", "ratio1_consumer", "consumer"
KINDS = (WINDOW, ENGRAM, KV2, KV2E, KV1, IDX1, C2, C1, CONSUMER)
SPLIT = "/split"  # suffix: the kind built as two kernels
MAX_ROWS = 24  # the widest step: B = 4 requests of 6 rows (``Scenario(B, 6, tp)``: a schedule entry keys on its rows)
ROWS = (None, *range(1, MAX_ROWS + 1))  # None: a build that does not know its row count
MANY = range(
    6, MAX_ROWS + 1
)  # 6 rows and up: the entries measured at 6-16 rows carry on to the batched widths
BATCHED = (12, 18, 24)  # B = 2, 3, 4 requests of 6 rows

SOURCES = (KV2, KV2E, KV1, IDX1)
KV_SOURCES = (KV2, KV2E, KV1)
KV_INDEX2 = (KV2, KV2E)  # ratio-2 KV + index sources: 2, 8, 14
CONSUMERS = (C2, C1, CONSUMER)


def split(kinds) -> tuple:
    return tuple(k + SPLIT for k in kinds)


ONE = KINDS  # one kernel a layer (the engine's build)
TWO = split(KINDS)  # two kernels a layer
BOTH = ONE + TWO


def kind_of(
    sparse: int = 0,
    source=None,
    engram: bool = False,
    sparse_ratio: int = 0,
    split_: bool = False,
) -> str:
    """The schedule kind of ``recipe.window_layer_blocks``' flags (``source``: a ``recipe.Source``)."""
    if source is not None:
        if source.kv and not source.index:
            raise ValueError(
                "V4.1's KV sources are index sources too (Source(kv=True, index=True))"
            )
        kind = f"ratio{source.ratio}_{'kv' if source.kv else 'index'}_source" + (
            "+engram" if engram else ""
        )
    elif sparse:
        kind = f"ratio{sparse_ratio}_consumer" if sparse_ratio else CONSUMER
    else:
        kind = ENGRAM if engram else WINDOW
    return kind + (SPLIT if split_ else "")


# ---------------------------------------------------------------------------------------------------------- schedules

SCHEDULES: dict[tuple, dict] = {}


def _put(kinds, rows, **knobs) -> None:
    for kind in kinds:
        for r in rows:
            SCHEDULES.setdefault((kind, r), {}).update(knobs)


# wkv's 128 K-slice tasks held CTAs under the Indexer's tasks on the ratio-2 KV + index sources: the plan's split
_put(KV_INDEX2 + split(KV_INDEX2), ROWS, WKV_KSPLIT=None)
# seam F's two-shot owner stage spread over 64 tasks of 16 KB from 6 rows
_put(BOTH, MANY, OWN_QUADS=128)
# the seams' LDS pad, off in the two-kernel window layer 0 (measured worse there)
_put(split([WINDOW]), ROWS, HC_LDS_PAD="")
# the seams' early gate values, one-kernel layers from 6 rows
_put(ONE, MANY, HC_EARLY=True)
# ffn_norm split over 4 tasks a row in one-kernel layers (NORM_SPLIT "auto"); none in two-kernel layers
_put(TWO, ROWS, NORM_SPLIT="")
# attn_norm over 2 tasks a row on the one-kernel sources, 1 on the ratio-1 KV source (layer 20)
_put([KV1], ROWS, SRC_NORM_SPLIT=1)
# the attention lands its ring rows before the q poll on window layers and consumers, not on the sources
_put(SOURCES + split(SOURCES), ROWS, RING_FIRST=False)
# the step's emitted entries polled at the chunk task's start on the KV sources
_put(KV_SOURCES + split(KV_SOURCES), ROWS, NEW_EARLY=True)
# the sparse attention split by P V dims on one-kernel consumers (moot while the merge takes raw scores)
_put(CONSUMERS, ROWS, SPARSE_DSPLIT=2)
# two heads a merge task at 1 row
_put(BOTH, [1], SPARSE_MHEADS=2)
# the Indexer at 48 wq_b rows a task in the two-kernel ratio-2 KV + index sources (with wkv's 4 K slices); its kpack over
# 64 lanes on them
_put(split(KV_INDEX2), ROWS, IDX_WQB_ROWS=48)
_put(KV_INDEX2 + split(KV_INDEX2), ROWS, KPACK_LANES=True)
# the Compressor after wq_a / wkv in two-kernel layers; 4 tiles a dot task on the one-kernel ratio-2 KV sources
_put(TWO, ROWS, COMP_FIRST=False)
_put(KV_INDEX2, ROWS, COMP_TILES=4)
# the router dot in 20 K slices in two-kernel layers (10 in one-kernel layers)
_put(TWO, ROWS, ROUTER_KSLICES=20)
# MoE row groups 8 / 32 at 6 rows where the step's distinct experts made groups()' 4 / 16 slower (one kernel)
_put((ENGRAM, KV1, IDX1) + CONSUMERS, ROWS, MOE_UG_G={6: 8}, MOE_DOWN_G={6: 32})
# the merge's V by LDS transpose reads but on the KV sources from 6 rows (no gain there; 1 row: -0.7 µs)
_put(KV_SOURCES + split(KV_SOURCES), MANY, SPARSE_VTR=False)
# the candidate readers (24 .. 36) score only layer 20's candidate blocks from n_max 24576 (at 20K: +0.5 µs at
# 6 rows, -1.1 at 1 row; 32K: -3.5 / -2.6; 128K: -31.4 / -14.6)
_put((IDX1,) + split((IDX1,)), ROWS, IDX_COMPACT=24576)
# wo_a's weights widened before its input lands (same bits) where measured (1, 6, 16 rows; the single-layer
# rows-less builds); 2-5 rows keep the old code (the DSpark draft's 5-row window stages among them)
_put(BOTH, (None, 1, *range(6, 17)), WO_A_PREWIDEN=8)
# the MoE rt table on fewer barriers (the same table), the same rows
_put(BOTH, (None, 1, *range(6, 17)), MOE_RT_LEAN=True)
# the owners' tasks at 64 quads at 1 and 6 rows (seam F's; the chain MoE owner's from 6 rows): 120 tasks of 8 KB
# at 6 rows; at 16 rows 64 loses (+0.3..+0.6 µs): the 128 above stays from 7 rows; layer 20 (ratio-1 KV source) at 6
# rows not (+0.55 µs over the two entries above there)
_put(ONE, [1], OWN_QUADS=64)
_put(tuple(k for k in ONE if k != KV1), [6], OWN_QUADS=64, MOE_OWN_QUADS=64)
# a sharded residual (RESIDUAL "sharded" / "auto") on the window kinds spills 3-10 SGPRs into VGPR lanes (they sit
# at the budget); measured no loss (layer 0 k1 -2.47 µs at 6 rows, layer 1 chained +0.15 within the controls' spread)
_put((WINDOW, ENGRAM), ROWS, SHARD_SPILL_OK=True)
# the ratio-2 KV + index sources at 18 rows (B = 3) sit at 256 VGPRs replicated (6 Compressor state slots, the
# hook's); sharded they spill one VGPR to memory with COMP_TILES 4: 2 there, as at 24 rows
_put(KV_INDEX2, [18], SHARD_COMP_TILES=2)

# the window layers at the batched widths spill 1-4 SGPRs into VGPR lanes (12 rows: window 4, window + Engram 1;
# none to memory), as the sources and consumers do at every width
_put((WINDOW, ENGRAM) + split((WINDOW, ENGRAM)), BATCHED, SPILL_OK=True)
# ... and the ratio-2 KV + index sources' Compressor dots at 2 tiles a task at 24 rows (4 spill 5 VGPRs there: two
# 16-row chunks' accumulators a tile)
_put(KV_INDEX2, [24], COMP_TILES=2)

# (the ug / down LDS rings at the batched widths are withdrawn: they won only at a one-layer timing footprint; at 16
# weight copies the ug ring is ~0 at B = 2 / 3 requests and loses 0.7 % at 4, and the down ring loses. MOE_UG_RING /
# MOE_DOWN_RING stay knobs, default 0)

# the index sources at the 1M bucket take LONG's knobs (layer 20 -133, the ratio-2 sources -52 µs at ~1M
# tokens; a step -27 at 1K, -16 at 32K, +3 at ~1M against the line without IDX_PCAP / IDX_SUBTILES)
_put(SOURCES + split(SOURCES), ROWS, IDX_LONG=1 << 20)
# IDX_LONG's knobs by kind (by kind, over a long-context serving mix where 70 % of the decode tokens are at
# >= 256K): the ratio-2 sources the whole line (IDX_SUBTILES 2: kv2e B3S6 spills with IDX_PCAP otherwise); layer 20
# without IDX_SUBTILES 2 (+0.4..+4 µs at 354K-1M, 1-2 requests); the readers only IDX_CLIST_SPARSE (the rest of the
# line: ±0.4 µs, IDX_SUBTILES 2 +1..+2.6)
_LINE = {"IDX_LIVE": True, "IDX_PERSIST": -1, "IDX_SPLIT": True, "IDX_PCAP": 65536}
# the split TopKs' final puts addressed before their select (IDX_XHOIST): the ratio-2 sources -1.2..-1.6 µs at
# 128K-1M (0 at 1K), layer 20 -0.8..-1.3 at 1 request, -3.9 at 2 (its candidate chain's puts too)
_LINE["IDX_XHOIST"] = True
# the ratio-2 sources' and the readers' selection lists packed (IDX_TOPK_PACKED; layer 20 ±0.4 then); layer
# 20 too, now that its selection is its path: -1.8..-4.0 µs at 128K-1M, 0 at 1K
_PK = {"IDX_TOPK_PACKED": True}
# layer 20's candidate chain shorter, the same bitmap (its words built in LDS; the block maxima bf16, so the
# candidate TopK in bf16 mode): the chain ends ~15 µs earlier at 354K / ~1M; chained -0.6..-2.9 µs at 1 request
# (128K-1M, 0 at 1K / 512K), -13.3..-13.6 at 2 requests and -9.9 at 3 (354K), where the chain is the layer's path
_CAND = {"IDX_CMASK_LDS": True, "IDX_BMAX_BF16": True}
LONG: dict = {KV2: {**_LINE, "IDX_SUBTILES": 2, **_PK}, KV2E: {**_LINE, "IDX_SUBTILES": 2, **_PK},
              KV1: {**_LINE, **_CAND, **_PK}, IDX1: {"IDX_CLIST_SPARSE": True, **_PK}}  # fmt: skip
# ... and by (kind, rows) on top: layer 20 at 6 rows its attention path off the candidate chain's CTAs (-19.7 /
# -23.5 µs at 354K / ~1M; +4 at 1K; 2 requests -0.6 / +3.7: not there); its selection's TopK tail too (2: a
# further -5.8 / -2.1 at 354K / ~1M, -0.4..-0.6 at 256K / 512K, 0 at 1K / 128K). At 6 rows the ratio-2 KV + index
# source's attn_norm hands wq_a / wkv tagged pairs (layer 2 0.990-0.991 at 1K-1M, chained -0.7 / -1.0 µs at 354K / ~1M;
# at B3S6 it spills a VGPR) and the readers' wq_b takes 32 rows a task (layer 24 0.976-0.986 alone; chained ~0)
LONG_ROWS: dict = {(KV1, 6): {"IDX_CAND_APART": 2}, (KV2, 6): {"XA_PROTO": "tagged"},
                   (IDX1, 6): {"IDX_WQB_ROWS": 32}}  # fmt: skip

# the TP degree the entries were measured at (a program past it allows SGPR spills into VGPR lanes: head, draft)
KEY_TP = 4


def _put_mi355x(kind, rows, **knobs) -> None:
    """An entry measured on MI355X (the served hardware) over the (kind, rows) entry: ``merge_knobs``'d onto it."""
    from ..plan.knobs import merge_knobs

    SCHEDULES[(kind, rows)] = merge_knobs(SCHEDULES.get((kind, rows), {}), knobs)


# the DSpark draft's stages (the window kind at 5 rows: only the draft runs it; reach: draft0 / 1 / 2 r5 change,
# nothing else) take the router's input straight from h1 (FFN_NORM "router"): its four launches 287.14 -> 284.17 us
# (0.9894, 32/32), bitwise
_put_mi355x(WINDOW, 5, FFN_NORM="router")
# the sparse merge at 8 heads a task (96 -> 192 tasks) at the batched widths where the block's 16 above 6 rows
# lost: chained, 2-3 distinct requests of 6 rows, -2.5 / -1.4 µs (layer 3, 12 / 18 rows), -1.8 / -1.8 (layer 21), the
# readers -1.4 at 12; 24 rows +2.1 (two rounds of tasks), the sources 2 / 20 +0.4 / +1.1: the block's there
for _k, _rows in (
    (C2, (12, 18)),
    (C1, (12,)),
    (IDX1, (12,)),
):  # (C1 at 18: the head-fused program spills)
    for _r in _rows:
        _put_mi355x(_k, _r, SPARSE_MHEADS=8)
# ... and the ratio-1 consumers at 18 rows with a sharded residual (replicated, their head-fused program spills):
# layer 21 -2.3 (distinct requests) / -0.9 µs (real routing)
_put_mi355x(C1, 18, SPARSE_MHEADS_SH=8)
# the ratio-1 consumers at 6 rows (a c1 decode step, measured decode-alone) take the router dot in 5 K slices: layer
# 21 61.71 -> 60.28 µs chained (0.9759, 32/32; 1K, real routing, gate PASS). D10:
# the dot's summation order changes; the served paired accuracy check held (142 vs 145 / 164, McNemar p 0.375)
_put_mi355x(C1, 6, ROUTER_KSLICES=5)

# The batch bands: Program.dyn_rows builds of B = 6 / 8 requests of 6 rows (36 / 48 rows)
# serving 5 .. 6 / 7 .. 8 live requests (one 48-row band cost +16 / +12 % chained at b = 5 / 6 over static B5 / B6 on
# layer 3: the second band). Their plan decisions are the widest static width's (24 rows: every (kind, 24) entry,
# hardware-keyed and tp-keyed ones included, IDX_LONG's among them), then the bands' own fits (compile-only at B6S6 /
# B8S6, TP4, 1M bucket): wq_b at the plan's output rows a task (below), and the window kinds' SGPR spills into VGPR
# lanes (3-8 at 36-48 rows, none to memory; as at the batched widths above)
BAND_ROWS = (36, 48)
for _key, _knobs in list(SCHEDULES.items()):
    if _key[1] == MAX_ROWS:
        for _r in BAND_ROWS:
            SCHEDULES[(_key[0], _r)] = copy.deepcopy(_knobs)
# wq_b's 48 output rows a task have no streamed shape past STREAM_ROWS (32) rows: the plan's there (the band; the DSpark
# draft's stages at 7-8 requests of 5 rows, 35-40)
_put(BOTH, range(33, 49), WQB_ROWS=None)
_put((WINDOW, ENGRAM) + split((WINDOW, ENGRAM)), BAND_ROWS, SPILL_OK=True)

# knobs a chain's launches must agree on (one reads the partials the previous one pushed, in their format): a step's,
# by its rows (``recipe.rs_pack``), never in a (kind, rows) entry (``knobs`` refuses one)
CHAIN_KNOBS = ("RS_PACK", "RS_PACK_ROWS")
# the precision knobs: a deployment's numerics contract (the official model's or ATOM's), one value a process
# (live / pinned), never a (kind, rows) entry (``knobs`` refuses one)
PRECISION_KNOBS = ("COMPRESS_QAT", "IDX_SCORE_F32", "ATTN_OUT_F32_ROPE")


def _entry(table: dict, kind: str, rows: int | None) -> dict:
    if (kind, rows) in table or not kind.split(SPLIT)[0].endswith("+engram"):
        return table.get((kind, rows), {})
    base = kind.replace(
        "+engram", ""
    )  # an Engram layer of another kind: its kind's schedule
    return table.get((base, rows), {})


def current() -> dict:
    """``DEFAULT`` (a copy; no kind: the placement passes and blocks built without one)."""
    return dict(DEFAULT)


def knobs(
    kind: str, rows: int | None, over: dict | None = None, tokens: int = 0
) -> dict:
    """Every knob of a ``kind`` layer at ``rows`` (a copy): ``DEFAULT``, the schedule's entry for (kind, rows) and,
    an index source whose context bucket ``tokens`` (its Indexer's n_max x ratio) reaches ``IDX_LONG``, its kind's
    ``LONG``; then ``over``; a knob still "auto" takes ``AUTO``'s value (the recipe resolves the rest: module
    docstring)."""
    entry = copy.deepcopy(_entry(SCHEDULES, kind, rows))
    entry.update(_env_over(kind, rows))
    given = dict(over or {})
    long_from = given.get("IDX_LONG", "auto")
    if isinstance(long_from, str) and long_from == "auto":
        long_from = entry.get("IDX_LONG", AUTO["IDX_LONG"])
    if long_from and tokens >= long_from:
        base = kind.split(SPLIT)[0]
        entry = {**entry, **LONG.get(base, {}), **LONG_ROWS.get((base, rows), {})}
    out = {**DEFAULT, **entry}
    for n, v in given.items():
        if n not in DEFAULT:
            raise KeyError(f"no V4.1 knob {n}")
        if not (
            isinstance(v, str) and v == "auto" and n in entry
        ):  # "auto": the schedule's value
            out[n] = copy.deepcopy(v)
    for n, v in AUTO.items():
        if isinstance(out[n], str) and out[n] == "auto":
            out[n] = v
    return out


def _env_over(kind: str, rows: int | None) -> dict:
    """``ATOM_DSV41_MONO_HI_KNOBS`` (a tuning aid): a JSON dict of {"*" | kind | "kind@rows": {knob: value}}, applied
    over the schedule's entry in that order (most specific last)."""
    import json
    import os

    raw = os.environ.get("ATOM_DSV41_MONO_HI_KNOBS")
    if not raw:
        return {}
    table = json.loads(raw)
    out = {}
    for key in ("*", kind.split(SPLIT)[0], kind, f"{kind}@{rows}"):
        out.update(table.get(key, {}))
    for n in out:
        if n not in DEFAULT:
            raise KeyError(f"ATOM_DSV41_MONO_HI_KNOBS: no V4.1 knob {n}")
    return out


def _check_entries() -> None:
    """No (kind, rows) entry sets a step's knob (``CHAIN_KNOBS``) or a process's numerics (``PRECISION_KNOBS``)."""
    for key, entry in SCHEDULES.items():
        bad = [n for n in (*CHAIN_KNOBS, *PRECISION_KNOBS) if n in entry]
        if bad:
            raise ValueError(
                f"V4.1 schedule entry {key}: {bad} are a step's or a process's, not a kind's"
            )


_check_entries()
