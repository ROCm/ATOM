# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every pointer the step kernel reads, by its table, and who provides it.

The kernel takes four pointer tables (``step``): ``fp`` (FPTRS, the FFN / step buffers, then ATAB, then one
row of LKEYS a layer -- the target's, the draft stages', and the draft entry's row), ``tp`` (``tail.TPTRS``:
head, sampling, verify, draft head, Markov) and the context-KV table (``draft_ctx.CTPTRS``, its address in ``fp``).
Each name is one of:

- ``SCRATCH``: owned by the runner, allocated once at the widest step (``T_MAX`` rows), never read across steps but
  through the kernel itself (``scratch_bytes`` gives their sizes);
- ``PEER``: the TP group's receive buffers (``recv0..3``), the step epoch word (``tpep``) flips them, no host clear;
- ``WEIGHT``: the model's tensors, or a layout the runner derives from them once (planes, head tiles, f32 norms);
- ``STEP``: per-step inputs the runner refreshes before the launch (graph-static addresses): the step metadata
  (``write_step_meta`` / ``write_draft_step_meta``), the input ids / anchors, the cache pool;
- ``STATE``: owned by the runner, kept across steps, never cleared by the host after construction: the draft
  distribution the step's prob drafter kept for the next step's verification (``qlog`` / ``qids``).
"""

SCRATCH, PEER, WEIGHT, STEP, STATE = "scratch", "peer", "weight", "step", "state"

MAX_REQ = 16  # requests a step
T_MAX = MAX_REQ * 6  # target rows: 16 requests x (5 drafts + 1)
TD_MAX = MAX_REQ * 5  # draft rows: 16 requests x 5

# fp table (FPTRS), in order
FPTRS_ROLE = {
    # the FFN front's state between layers
    "R": SCRATCH,
    "y": SCRATCH,
    "pre": SCRATCH,
    "post": SCRATCH,
    "comb": SCRATCH,
    "fnh": WEIGHT,
    "fnl": WEIGHT,
    "hsc": WEIGHT,
    "hbs": WEIGHT,
    "nw": WEIGHT,
    "gate": WEIGHT,
    "gbias": WEIGHT,
    "Rout": SCRATCH,
    "xn": SCRATCH,
    "fx8": SCRATCH,
    "fxs": SCRATCH,
    "part": SCRATCH,
    "plog": SCRATCH,
    "cand": SCRATCH,
    "cands": SCRATCH,
    "npre": SCRATCH,
    "npost": SCRATCH,
    "ncomb": SCRATCH,
    "ids": SCRATCH,
    "wts": SCRATCH,
    "mout": SCRATCH,
    "mid8": SCRATCH,
    "midc": SCRATCH,
    "afnh": WEIGHT,
    "afnl": WEIGHT,
    "ahsc": WEIGHT,
    "ahbs": WEIGHT,
    "anw": WEIGHT,
    "aRout": SCRATCH,
    "axn": SCRATCH,
    "anpre": SCRATCH,
    "anpost": SCRATCH,
    "ancomb": SCRATCH,
    "tp2out": SCRATCH,
    "cyc": SCRATCH,
    "y1": SCRATCH,
    "y2": SCRATCH,
    "shplane": WEIGHT,  # the shared expert, FP8 plane
    "kmeta": STEP,
    "knsel": STEP,
    "ksel": STEP,
    "kvw": STEP,
    "rpos": STEP,
    "iq": SCRATCH,
    "iwp": SCRATCH,
    "iq_out": SCRATCH,
    "iqs_out": SCRATCH,
    "iw_out": SCRATCH,
    "xh": SCRATCH,  # the head's input (FINAL_XH: the last target layer's front writes it)
    "cpart": SCRATCH,
    "ilvl": SCRATCH,
    "aux8": SCRATCH,
    "auxs": SCRATCH,  # the draft context taps (target layers NL-4 .. NL-2)
    "tpep": PEER,
    "kmeta_d": STEP,
    "kvw_d": STEP,
    "rpos_d": STEP,
    "dR": SCRATCH,
    "dy": SCRATCH,
    "dpre": SCRATCH,
    "dpost": SCRATCH,
    "dcomb": SCRATCH,
    "emb": WEIGHT,  # the full bf16 embedding table (the anchor rows of the block state)
    "noise": WEIGHT,  # the DSpark noise token's embedding row
    "x8": SCRATCH,
    "xs": SCRATCH,
    "ctxtab": SCRATCH,  # the context-KV pointer table (filled by the runner)
    "dgeom": STEP,  # the draft metadata's geometry and the requests' slots (DRAFT_META_DEV)
    "tids": STEP,  # this step's input ids (TARGET_ENTRY)
    # Engram (engram): the step's embedding rows as MXFP8, their kv projection, the gate's partial sums
    "eng8": SCRATCH,
    "engs": SCRATCH,
    "engkv": SCRATCH,
    "engpart": SCRATCH,
}

# fp table, after FPTRS (ATAB): the attention's
ATAB_ROLE = {
    "x8": SCRATCH,
    "xs": SCRATCH,
    "wqkv": WEIGHT,
    "wqkv_s": WEIGHT,
    "qnw": WEIGHT,
    "kvnw": WEIGHT,
    "wqb": WEIGHT,
    "wqb_s": WEIGHT,
    "kv": STEP,  # cache.pool
    "sel": SCRATCH,
    "sink": WEIGHT,
    "rcos": STEP,
    "rsin": STEP,
    "woa": WEIGHT,
    "woa_s": WEIGHT,
    "wob": WEIGHT,
    "wob_s": WEIGHT,
    "qkv_part": SCRATCH,
    "qkv": SCRATCH,
    "ss": SCRATCH,
    "qr8": SCRATCH,
    "qrs": SCRATCH,
    "kvn": SCRATCH,
    "q": SCRATCH,
    "apart": SCRATCH,
    "aml": SCRATCH,
    "o8": SCRATCH,
    "os_": SCRATCH,
    "y8": SCRATCH,
    "ys": SCRATCH,
    "ypart": SCRATCH,
    "out": SCRATCH,
    "opart": SCRATCH,
    "tpout": SCRATCH,
    "recv0": PEER,
    "recv1": PEER,
    "recv2": PEER,
    "recv3": PEER,
}

# tp table (tail.TPTRS)
TPTRS_ROLE = {
    "xh": SCRATCH,
    "hw": WEIGHT,  # the LM head's rank shard, in the head kernel's tile order
    "logits": SCRATCH,
    "s0": SCRATCH,
    "s1": SCRATCH,
    "s2": SCRATCH,
    "s3": SCRATCH,
    "sids": SCRATCH,  # sampled ids (verify input)
    "dids": STEP,  # this step's draft tokens (verify input: the previous step's Markov output)
    "outtok": SCRATCH,
    "verdict": SCRATCH,
    "anc": SCRATCH,  # verify outputs -> ``verify_outputs``
    "dh": SCRATCH,
    "dlog": SCRATCH,
    "mw1": WEIGHT,
    "mw2": WEIGHT,  # the Markov sampler's
    "mids": SCRATCH,  # draft tokens out -> ``draft_ids``
    "memb": SCRATCH,
    "mbest": SCRATCH,
    # each target row's sampling [T][8] (invT, top_p, seed, greedy | invTd, hasq, qsrc, seed_acc) -> ``runner._sampling``
    "sparam": STEP,
    "qlog": STATE,  # [8][5][VR] f32: this rank's draft logits of the last kernel step (its prob drafts' q)
    "prow": SCRATCH,
    "qids": STATE,  # [8][5] i32: the drafts qlog was kept for (-1 at start)
}

# context-KV table (draft_ctx.CTPTRS)
CTPTRS_ROLE = {
    "aux8": SCRATCH,
    "auxs": SCRATCH,
    "mpw": WEIGHT,
    "mps": WEIGHT,
    "mnw": WEIGHT,  # main_proj (FP8 + scales), main_norm
    "hpre": SCRATCH,
    "hpref": SCRATCH,  # f32 [T][5120] main_proj sums (ctx v2: zero at allocation, the kernel re-zeroes it)
    "h8": SCRATCH,
    "hs": SCRATCH,
    "wkv0": WEIGHT,
    "wkv1": WEIGHT,
    "wkv2": WEIGHT,
    "wks0": WEIGHT,
    "wks1": WEIGHT,
    "wks2": WEIGHT,
    "knw0": WEIGHT,
    "knw1": WEIGHT,
    "knw2": WEIGHT,  # the draft stages' wkv / kv_norm
    "kvp": SCRATCH,
    "rcos": STEP,
    "rsin": STEP,
    "kvw": STEP,  # the context rows' ring rows (the accepted target positions)
    "ring": STEP,  # cache.pool
    "roff": STEP,  # the draft stages' ring starts (geometry.window(layer).ring_start)
}


# The model's full 1M context at compression 1. Scoring, block maxima,
# selection lists and the staged page table all use this capacity.
ILW = 1048576
BCOLS = ILW // 16  # columns at the smallest supported (16-token) page size


def scratch_bytes(K, nlt: int) -> dict[str, int]:
    """Bytes of every SCRATCH buffer at the widest step; ``K``: the kernel modules (attention, front,
    moe, head, sample_common, step), as the runner imported them; ``nlt``: target + draft layers.
    """
    S, FS, MS, HF = K["attention"], K["front"], K["moe"], K["head"]
    M = K["step"]
    T, H = T_MAX, FS.H
    QV = S.smem_layout(T)["QV"]
    NRD = S.d_rounds(
        T, S.smem_layout(T)["RPC"]
    )  # stage D's rounds at the widest step (each its own partials)
    TPG = (T * 10 + 2) // 3  # TP words a CTA at the widest step
    G = FS.NG * FS.CL * FS.TG
    IP = M.TK
    level_stride = max(
        -(-ILW // IP.SEG) * IP.KSEL,
        -(-(ILW // 8) // IP.SEG) * IP.KBLK,
    )
    return {
        # attention
        "sel": T * 640 * 4,
        "qkv_part": 2 * T * S.NQKV * 4,
        "qkv": T * S.NQKV * 2,
        "ss": nlt * S.SS_LAYER_F32 * 4,
        "qr8": T * S.QL,
        "qrs": T * S.QL // 32,
        "kvn": T * S.KVD * 2,
        "q": T * S.NQ * 2,
        "apart": NRD * 256 * QV * S.HD * 2,
        "aml": NRD * 256 * QV * 2 * 4,
        "o8": T * S.NQ,
        "os_": T * S.NQ // 32,
        "y8": T * S.NWOA,
        "ys": T * S.NWOA // 32,
        "ypart": 256 * T * 32 * 4,
        "out": T * S.HID * 4,
        "opart": 256 * T * 80 * 4,
        "tpout": 256 * TPG * 3 * 4,
        "x8": T * S.HID,
        "xs": T * S.HID // 32 * 4,
        # FFN front, MoE
        "R": T * 4 * H * 2,
        "y": T * H * 2,
        "pre": T * 16,
        "post": T * 16,
        "comb": T * 64,
        "Rout": T * 4 * H * 2,
        "xn": T * H * 2,
        "fx8": T * H,
        "fxs": T * H // 32,
        "part": G * 32 * 4,
        "plog": G * FS.NE * 4,
        "cand": G * FS.TOPK * 8,
        "cands": G * FS.TOPK * 4,
        "npre": T * 16,
        "npost": T * 16,
        "ncomb": T * 64,
        "ids": T * FS.TOPK * 4,
        "wts": T * FS.TOPK * 4,
        "mout": (T * 6 + T + 1) * H * 4,
        "mid8": MS.NEX * MS.MAXTOK * MS.MROW,
        "midc": MS.NEX * MS.MAXTOK * MS.CROW,
        "aRout": T * 4 * H * 2,
        "axn": T * H * 2,
        "anpre": T * 16,
        "anpost": T * 16,
        "ancomb": T * 64,
        "tp2out": 256 * TPG * 3 * 4,
        "cyc": 256 * 32 * 8 * 2,
        "y1": T * H * 2,
        "y2": T * H * 2,
        "iq": T * 4096 * 2,
        "iwp": 32 * T * 32 * 4,
        "iq_out": T * 4096,
        "iqs_out": T * 32 * 4,
        "iw_out": T * 32 * 4,
        "cpart": 8
        * T
        * 1024
        * 4,  # the compressor's wkv_gate K-split partials (compressor.stage_cgemm)
        # the selection tree (M.IDX_LEVELS levels): a level's lists of (value, index) pairs a row, a level-0 list per
        # 8192 index rows of the bound (ILW), k a list -- 2048 for layer 20's candidate blocks (topk "blk": also
        # its dummy-row store, row T, lands there), 512 otherwise
        "ilvl": M.IDX_LEVELS * (T + 1) * level_stride * 8 + 64,
        "aux8": T * 15360,
        "auxs": T * 480,
        "dR": T * 4 * H * 2,
        "dy": T * H * 2,
        "dpre": T * 16,
        "dpost": T * 16,
        "dcomb": T * 64,
        "eng8": T * 6144,
        "engs": T * 192,
        "engkv": T * 25600 * 2,
        "engpart": G * 16 * 4,
        # step tail
        "xh": T * HF.HID * 2,
        "logits": T * HF.VR * 4,
        "s0": T * 5 * 24,  # sample records
        "s1": T * 5 * 256 * 4,
        "s2": T * 5 * 256 * 4,
        "s3": T * 5 * 20,
        "prow": T * 8 * 4,
        "sids": T * 4,
        "outtok": T * 4,
        "verdict": 3 * MAX_REQ * 4,
        "anc": MAX_REQ * 4,
        "dh": T * HF.HID * 2,
        "dlog": T * HF.VR * 4,
        "mids": TD_MAX * 4,
        "memb": TD_MAX * 256 * 2,
        "mbest": TD_MAX * 8,
        # context KV
        "hpre": T * H * 2,
        "hpref": T * H * 4,
        "h8": T * H,
        "hs": T * 160,
        "kvp": 3 * T * 512 * 2,
        # handoff counters, stamps
        "cnt": (nlt * M.LCB // 4 + 64) * 4 + 1024,
        "st": 256 * 32 * 8,
    }


def check_catalog(K) -> list[str]:
    """Names the kernel's tables hold that this catalog does not classify (empty when in sync)."""
    M, ST, CX = K["step"], K["tail"], K["draft_ctx"]
    miss = [f"FPTRS {n}" for n in M.FPTRS if n not in FPTRS_ROLE]
    miss += [f"ATAB {n}" for n in M.ATAB if n not in ATAB_ROLE]
    miss += [f"TPTRS {n}" for n in ST.TPTRS if n not in TPTRS_ROLE]
    miss += [f"CTPTRS {n}" for n in CX.CTPTRS if n not in CTPTRS_ROLE]
    return miss
