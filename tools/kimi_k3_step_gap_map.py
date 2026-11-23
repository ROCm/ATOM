# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The part of a Kimi-K3 DSpark decode step between two target forwards, kernel
by kernel, from a torch profiler trace of one rank (``trace_rank0.json.gz``):
every GPU kernel and copy attributed to the host call that launched it (the
runtime call of the same correlation id, the ``cpu_op`` and the
``record_function`` label around it) and, in ``map``, to the code that emits it.

A between-step region runs from the kernel after a step's last AttnRes to the
next step's first AttnRes (two AttnRes kernels more than 1.5 ms apart). Regions
are told apart by the rows of the step before and after (``_s<rows>`` of the
mono kernels). No GPU; reads the trace only.

    python tools/kimi_k3_step_gap_map.py TRACE regions
    python tools/kimi_k3_step_gap_map.py TRACE seq [index|med]
    python tools/kimi_k3_step_gap_map.py TRACE table ROWS    per-position medians
    python tools/kimi_k3_step_gap_map.py TRACE stat ROWS a,b substrings' spread
    python tools/kimi_k3_step_gap_map.py TRACE htod          the large uploads
    python tools/kimi_k3_step_gap_map.py TRACE map ROWS      the segment table

``map`` rows: segment, what the kernel is, launches a step, kernel time a step
(median over the width's regions; the 0.8 us gap behind each launch is in the
segment's span, not in its kernels), the stream, the emitter. Line numbers are
those of the tree this tool was written against (ws-combo3 d8ebb3a87).
"""

import collections
import gzip
import itertools
import json
import re
import statistics
import sys

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
DSPARK = "atom/spec_decode/dspark_proposer.py"
RUNNER = "atom/model_engine/model_runner.py"
DRAFT = "atom/models/dspark_draft.py"
K3D = "atom/models/kimi_k3_dspark.py"
STEP_TAIL = "STEP TAIL (ATOM_KIMI_MONO_STEP_TAIL takes it)"
PAGED = f"{DSPARK} _build_paged_block_metadata"

# the bookkeeping between the id broadcast and the draft's MLA metadata, in
# launch order: (substring of "kernel | host ops", what, emitter)
BOOK = (
    (
        "gather",
        "next_token_ids = gather(ids, anchor column)",
        f"{RUNNER}:3604 postprocess",
    ),
    (
        "_stage_anchors_kernel",
        "anchor ids and positions into the block's inputs",
        f"{DSPARK}:231 _stage_anchors",
    ),
    ("CUDAFunctor_add", "block_positions = anchor + 1..T", f"{PAGED}:961"),
    ("add", "global_lens = anchor + 1 + T", f"{PAGED}:997"),
    (
        "div",
        "page_idx = block_positions // tokens_per_entry",
        f"{DSPARK}:1188 _block_page_idx",
    ),
    ("lt", "in_table = page_idx < columns", f"{DSPARK}:1188 _block_page_idx"),
    ("clamp_", "page_idx.clamp_", f"{DSPARK}:1188 _block_page_idx"),
    ("remainder", "block_positions % virtual_block", f"{PAGED}:1008"),
    ("div", "local_off = ... // dcp_world_size", f"{PAGED}:1008"),
    ("remainder", "block_positions % dcp_world_size", f"{PAGED}:1013"),
    ("eq", "... == dcp_rank", f"{PAGED}:1013"),
    ("bitwise_and", "in_table & owned", f"{PAGED}:1013"),
    ("gather", "gather(block_tables, page_idx)", f"{PAGED}:1013"),
    ("mul", "... * block_size", f"{PAGED}:1013"),
    ("add", "... + local_off", f"{PAGED}:1013"),
    ("fill_", "where()'s scalar -1", f"{PAGED}:1013"),
    ("where", "where(owned, slot, -1)", f"{PAGED}:1013"),
    ("add", "global_lens + (dcp - 1 - rank)", f"{PAGED}:1025"),
    ("div", "... // dcp_world_size", f"{PAGED}:1025"),
    ("copy_", "ctx_lens.copy_", f"{PAGED}:1025"),
    ("clamp_", "ctx_lens.clamp_", f"{PAGED}:1084"),
    ("copy_", "cumsum's int32 -> int64", f"{PAGED}:1088"),
    ("cumsum", "cumsum(ctx_lens)", f"{PAGED}:1088"),
    ("copy_", "kv_indptr[1:].copy_", f"{PAGED}:1088"),
    ("kv_indices_generate_kernel", "the block's KV page indices", f"{PAGED}:1118"),
)
# the next step's preparation, in launch order
PREPARE = (
    (
        "clone",
        "draft ids out of the graph's pool",
        "atom/spec_decode/draft_graph.py:33 _owned (DraftGraph.run :277)",
    ),
    (
        "scatter_packed",
        "token inputs: packed upload's scatter",
        "atom/utils/packed_h2d.py:93 PackedCopy.submit (token_inputs group)",
    ),
    (
        "_fill_deferred_decode_ids_kernel",
        "this step's input ids from the last verdict",
        f"atom/model_ops/decode_input_ids.py:146 ({RUNNER}:743 prepare_input_ids)",
    ),
    (
        "scatter_packed",
        "decode metadata: packed upload's scatter",
        "atom/utils/packed_h2d.py:93 PackedCopy.submit (decode_metadata group)",
    ),
    (
        "kv_indices_generate_kernel",
        "target MLA layers' KV page indices",
        "the target's attention metadata builder (finish_decode)",
    ),
    (
        "kn_get_mla_metadata",
        "target MLA layers' decode plan",
        "the target's attention metadata builder (get_mla_metadata_v1)",
    ),
    (
        "fill_",
        "KDA state: write cursors",
        "atom/model_ops/attentions/gdn_attn.py (decode metadata)",
    ),
    (
        "_replayssm_commit_kernel",
        "KDA state: commit the accepted tokens",
        "atom/model_ops/attentions/gdn_attn.py:1604 replayssm_commit",
    ),
    ("fill_", "KDA state", "atom/model_ops/attentions/gdn_attn.py (decode metadata)"),
    ("copy_", "KDA state", "atom/model_ops/attentions/gdn_attn.py (decode metadata)"),
    ("fill_", "KDA state", "atom/model_ops/attentions/gdn_attn.py (decode metadata)"),
)
CONTEXT = (
    (
        "CatArray",
        "cat of the 5 aux taps [rows, 35840]",
        f"{DRAFT}:625 concat_taps ({K3D}:660 project_context)",
    ),
    (
        "gemm_a16w16",
        "context_proj BF16 GEMM, [7168, 35840] replicated (514 MB)",
        f"{K3D}:660 project_context -> ReplicatedLinear -> tgemm.mm",
    ),
    ("add_rmsnorm_quant_kernel", "context_norm RMSNorm", f"{K3D}:660 project_context"),
    (
        "dynamic_per_group_scaled_quant",
        "E8M0 quant of the SAME ctx tensor, once a layer",
        f"{K3D}:360 write_context_kv -> fused_qkv_a_proj (LinearBase.forward)",
    ),
    (
        "gemm_a8w8_splitk_reduce",
        "fused_qkv_a_proj split-K reduce, a layer",
        f"{K3D}:360",
    ),
    (
        "gemm_a8w8",
        "fused_qkv_a_proj FP8 GEMM, a layer (2112 rows, 576 kept)",
        f"{K3D}:360",
    ),
    (
        "_fused_mla_ctx_kv_kernel",
        "kv norm + rope + cache store, a layer",
        f"{K3D}:366 -> atom/model_ops/attention_mla.py:3020 write_context_kv_latent",
    ),
)
SEGMENTS = (
    ("tail", "target head (" + STEP_TAIL + ")"),
    ("context", "draft context path"),
    ("verify", "sampling and verify (" + STEP_TAIL + ")"),
    ("book", "bookkeeping: anchors and the block's paged metadata"),
    ("draft_meta", "draft MLA metadata"),
    ("draft_embed", "draft block: embedding"),
    ("draft_layers", "draft block: 5 layers (the draft-layer task)"),
    ("draft_head", "draft block: head and Markov (" + STEP_TAIL + ")"),
    ("prepare", "next step's preparation"),
    ("target_embed", "target embedding"),
    ("layer0", "target layer 0 before its first AttnRes (target forward)"),
)


def load(path):
    with gzip.open(path, "rt") as f:
        evs = json.load(f)["traceEvents"]
    gpu = sorted(
        (e for e in evs if e.get("ph") == "X" and e.get("cat") in GPU_CATS),
        key=lambda e: e["ts"],
    )
    by_tid = collections.defaultdict(lambda: ([], []))
    for e in evs:
        if e.get("ph") != "X":
            continue
        if e.get("cat") in ("cpu_op", "user_annotation"):
            by_tid[e["tid"]][0].append(e)
        elif e.get("cat") == "cuda_runtime":
            by_tid[e["tid"]][1].append(e)
    attr = {}
    for ivs, rts in by_tid.values():
        ivs.sort(key=lambda e: (e["ts"], -e["dur"]))
        rts.sort(key=lambda e: e["ts"])
        stack, i = [], 0
        for r in rts:
            t = r["ts"]
            while i < len(ivs) and ivs[i]["ts"] <= t:
                stack.append(ivs[i])
                i += 1
            stack = [s for s in stack if s["ts"] + s["dur"] >= t]
            corr = r["args"].get("correlation")
            if corr is None or corr in attr:
                continue
            attr[corr] = (
                r["name"],
                [s["name"] for s in stack if s["cat"] == "user_annotation"],
                [
                    s["name"].replace("aten::", "")
                    for s in stack
                    if s["cat"] == "cpu_op"
                ],
                r["ts"],
            )
    marks = [i for i, e in enumerate(gpu) if "_attn_res" in e["name"]]
    regions = [
        (p + 1, n)
        for p, n in itertools.pairwise(marks)
        if gpu[n]["ts"] - gpu[p]["ts"] > 1500
    ]
    return gpu, attr, regions


def width(gpu, idx, step):
    rng = (
        range(idx, max(idx - 60, 0), -1)
        if step < 0
        else range(idx, min(idx + 60, len(gpu)))
    )
    for j in rng:
        m = re.search(r"_s(\d+)(_|$)", gpu[j]["name"])
        if m and "k3_" in gpu[j]["name"]:
            return int(m.group(1))
    return 0


def sname(n):
    n = (
        re.sub(r"^void ", "", n)
        .replace("at::native::", "")
        .replace("(anonymous namespace)::", "")
    )
    n = re.sub(
        r"elementwise_kernel_manual_unroll<(\d+), (\d+), gpu_kernel_impl_nocast<(.*)",
        r"ew<\1,\2> \3",
        n,
    )
    n = re.sub(
        r"elementwise_kernel_manual_unroll<(\d+), (\d+), gpu_kernel_impl<(.*)",
        r"ewc<\1,\2> \3",
        n,
    )
    n = n.replace(
        "_ZN5aiter37dynamic_per_group_scaled_quant_kernel",
        "aiter::dynamic_per_group_scaled_quant",
    )
    return n.replace(
        "_ZN5aiter24add_rmsnorm_quant_kernel", "aiter::add_rmsnorm_quant_kernel"
    )


def info(attr, e):
    rt, ann, ops, rts = attr.get(e["args"].get("correlation"), ("?", [], [], 0.0))
    return rt, ann, ops, rts


def big_upload(e):
    return "HtoD" in e["name"] and e["dur"] > 20


def plain(gpu, regions, rows):
    """The width's regions with one kernel sequence and no large upload."""
    sel = [
        (a, b)
        for a, b in regions
        if width(gpu, a - 1, -1) == rows and width(gpu, b, 1) == rows
    ]
    sel = [(a, b) for a, b in sel if not any(big_upload(gpu[i]) for i in range(a, b))]
    if not sel:
        sys.exit(f"no region of {rows} -> {rows} rows")
    seqs = collections.Counter(
        tuple(gpu[i]["name"] for i in range(a, b)) for a, b in sel
    )
    ref = seqs.most_common(1)[0][0]
    return [(a, b) for a, b in sel if tuple(gpu[i]["name"] for i in range(a, b)) == ref]


def classify(gpu, attr, a, b):
    """(segment, what, emitter) of every event of region [a, b)."""
    names = [sname(gpu[i]["name"]) for i in range(a, b)]
    meta = [info(attr, gpu[i]) for i in range(a, b)]
    graph = [m[0] == "hipGraphLaunch" for m in meta]
    out = []
    seg, book, prep = "tail", 0, 0
    head_at = None
    for k, n in enumerate(names):
        if "ddaAllGather" in n and graph[k]:
            j = k - 1
            if "splitk_reduce" in names[j]:
                j -= 1
            head_at = j - 1
    after_ar = False
    for k, n in enumerate(names):
        e = gpu[a + k]
        _, ann, ops, _ = meta[k]
        an = ann[-1] if ann else ""
        key = f"{n} | {'>'.join(ops[-3:])}"
        copy = e["cat"] == "gpu_memcpy"
        if seg == "tail" and "CatArray" in n:
            seg = "context"
        elif seg == "context" and not an.startswith("draft_kv") and "CatArray" not in n:
            seg = "verify"
        elif seg == "verify" and k and "ncclDevKernel" in names[k - 1]:
            seg = "book"
        elif seg == "book" and "kn_get_mla_metadata" in n:
            seg = "draft_meta"
        elif seg == "draft_meta" and graph[k]:
            seg, after_ar = "draft_embed", False
        elif seg == "draft_embed" and after_ar:
            seg = "draft_layers"
        elif (
            seg in ("draft_layers", "draft_embed")
            and head_at is not None
            and k >= head_at
        ):
            seg = "draft_head"
        elif seg == "draft_head" and not graph[k] and not copy:
            seg = "prepare"
        elif seg == "prepare" and graph[k]:
            seg, after_ar = "target_embed", False
        elif seg == "target_embed" and after_ar:
            seg = "layer0"
        what, emitter = n[:60], ""
        if seg == "tail":
            what = "last copy of the recorded forward" if "copyBuffer" in n else what
        elif seg == "context":
            for sub, w, em in CONTEXT:
                if sub in n:
                    what, emitter = w, em
                    break
        elif seg == "book":
            if copy:
                what, emitter = (
                    f"side copy: {'>'.join(ops[-2:])}",
                    f"{RUNNER} postprocess: ids / spec counts to the output path, or slots.copy_ ({PAGED}:1013)",
                )
            elif book < len(BOOK) and BOOK[book][0] in key:
                what, emitter = BOOK[book][1], BOOK[book][2]
                book += 1
        elif seg == "draft_meta":
            what, emitter = (
                "get_mla_metadata_v1 for the block (one CTA, one wave)",
                f"{PAGED}:1138",
            )
        elif seg == "draft_embed":
            if "FillFunctor" in n:
                what, emitter = (
                    "draft_ids = full(noise id)",
                    f"{DRAFT}:692 block_backbone",
                )
            elif "direct_copy" in n:
                what, emitter = (
                    "draft_ids[:, 0] = anchors",
                    f"{DRAFT}:693 block_backbone",
                )
            elif "_masked_embedding_kernel" in n:
                what, emitter = (
                    "masked embedding (this rank's vocab shard, zeros elsewhere)",
                    f"{DRAFT}:694 -> atom/model_ops/embed_head.py:175",
                )
            else:
                what, emitter = (
                    "embedding TP all-reduce (aiter)",
                    "atom/model_ops/embed_head.py:179",
                )
                after_ar = True
        elif seg == "draft_layers":
            what = "draft layers' kernels"
        elif seg == "draft_head":
            what = "draft head and Markov kernels"
        elif seg == "prepare":
            if copy:
                what, emitter = (
                    f"copy: {'>'.join(ops[-2:])} ({'upload' if 'HtoD' in n else 'device'})",
                    "packed upload arenas / output path",
                )
            elif prep < len(PREPARE) and PREPARE[prep][0] in key:
                what, emitter = PREPARE[prep][1], PREPARE[prep][2]
                prep += 1
        elif seg == "target_embed":
            if "_masked_embedding_kernel" in n:
                what, emitter = (
                    "masked embedding (this rank's vocab shard)",
                    "atom/models/kimi_k3/model.py:2009 -> atom/model_ops/embed_head.py:175",
                )
            else:
                what, emitter = (
                    "embedding TP all-reduce (aiter)",
                    "atom/model_ops/embed_head.py:179",
                )
                after_ar = True
        elif seg == "layer0":
            what = "layer 0's attention kernels"
        elif seg == "verify":
            what = "sampler / verify kernels"
        out.append((seg, what, emitter))
    return out


def main():
    path, mode = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "regions"
    gpu, attr, regions = load(path)
    if mode == "regions":
        for k, (a, b) in enumerate(regions):
            big = [
                (round(gpu[i]["dur"], 1), gpu[i]["args"].get("bytes"))
                for i in range(a, b)
                if big_upload(gpu[i])
            ]
            print(
                f"region {k:3d}: rows {width(gpu, a - 1, -1):3d} -> {width(gpu, b, 1):3d} span {gpu[b]['ts'] - gpu[a]['ts']:7.1f} "
                f"kernels {sum(gpu[i]['dur'] for i in range(a, b)):7.1f} n {b - a:4d} large uploads {big}"
            )
    elif mode == "seq":
        sel = sys.argv[3] if len(sys.argv) > 3 else "med"
        if sel == "med":
            order = sorted(
                range(len(regions)),
                key=lambda k: gpu[regions[k][1]]["ts"] - gpu[regions[k][0]]["ts"],
            )
            k = order[len(order) // 2]
        else:
            k = int(sel)
        a, b = regions[k]
        t0 = gpu[a]["ts"]
        print(
            f"region {k}: rows {width(gpu, a - 1, -1)} -> {width(gpu, b, 1)} span {gpu[b]['ts'] - t0:.1f} us, {b - a} events"
        )
        print(
            "# +start  gap  dur  host-lead  kernel | grid or bytes | runtime call | host ops | labels"
        )
        end = t0
        for i in range(a, b + 1):
            e = gpu[i]
            rt, ann, ops, rts = info(attr, e)
            g, bl = e["args"].get("grid"), e["args"].get("block")
            extra = f"{g[0]}x{g[1]}x{bl[0]}" if g else f"{e['args'].get('bytes')}B"
            print(
                f"+{e['ts'] - t0:7.1f} {e['ts'] - end:6.1f} {e['dur']:6.1f} {e['ts'] - rts:8.1f} {sname(e['name'])[:58]:<58} {extra:<12} "
                f"| {rt[:16]:<16} | {'>'.join(ops[-3:])[:60]:<60} | {' > '.join(x[:44] for x in ann[-3:])}"
            )
            end = e["ts"] + e["dur"]
    elif mode == "table":
        sel = plain(gpu, regions, int(sys.argv[3]))
        n = sel[0][1] - sel[0][0]
        spans = [gpu[b]["ts"] - gpu[a]["ts"] for a, b in sel]
        print(
            f"# rows {sys.argv[3]}: {len(sel)} regions of {n} events; span median {statistics.median(spans):.1f} us"
        )
        print(
            "# idx  +start  dur_p50  gap_before_p50  kernel | grid | launch | host ops | label"
        )
        a0 = sel[0][0]
        for k in range(n + 1):
            durs = [gpu[a + k]["dur"] for a, b in sel]
            offs = [gpu[a + k]["ts"] - gpu[a]["ts"] for a, b in sel]
            gaps = [
                gpu[a + k]["ts"] - gpu[a + k - 1]["ts"] - gpu[a + k - 1]["dur"]
                for a, b in sel
            ]
            e = gpu[a0 + k]
            rt, ann, ops, _ = info(attr, e)
            g, bl = e["args"].get("grid"), e["args"].get("block")
            extra = f"{g[0]}x{g[1]}x{bl[0]}" if g else f"{e['args'].get('bytes')}B"
            kind = {
                "hipGraphLaunch": "G",
                "hipLaunchKernel": "K",
                "hipModuleLaunchKernel": "M",
            }.get(rt, rt[:14])
            print(
                f"{k:3d} +{statistics.median(offs):7.1f} {statistics.median(durs):6.1f} {statistics.median(gaps):6.1f} "
                f"{sname(e['name'])[:60]:<60} {extra:<11} {kind:<3} {'>'.join(ops[-3:])[:44]:<44} {(ann[-1] if ann else '')[:44]}"
            )
    elif mode == "stat":
        sel = plain(gpu, regions, int(sys.argv[3]))
        pats = sys.argv[4].split(",")
        for k in range(sel[0][1] - sel[0][0] + 1):
            nm = gpu[sel[0][0] + k]["name"]
            if not any(p in nm for p in pats):
                continue
            d = sorted(gpu[a + k]["dur"] for a, b in sel)

            def q(f, d=d):
                return d[min(len(d) - 1, int(f * len(d)))]

            streams = {gpu[a + k]["args"].get("stream") for a, b in sel}
            print(
                f"{k:3d} n={len(d):2d} min {d[0]:6.1f} p10 {q(0.1):6.1f} p50 {q(0.5):6.1f} p90 {q(0.9):6.1f} max {d[-1]:6.1f} streams {streams} {sname(nm)[:70]}"
            )
    elif mode == "htod":
        for k, (a, b) in enumerate(regions):
            for i in range(a, b):
                e = gpu[i]
                if not big_upload(e):
                    continue
                rt, ann, ops, rts = info(attr, e)
                t0 = gpu[a]["ts"]
                print(
                    f"region {k} (rows {width(gpu, a - 1, -1)} -> {width(gpu, b, 1)}, span {gpu[b]['ts'] - t0:.1f}): {e['args'].get('bytes')} B, "
                    f"host call {rt} at +{rts - t0:.1f}, copy +{e['ts'] - t0:.1f} .. +{e['ts'] + e['dur'] - t0:.1f}, "
                    f"kernel before it ends +{gpu[i - 1]['ts'] + gpu[i - 1]['dur'] - t0:.1f}, the next starts +{gpu[i + 1]['ts'] - t0:.1f} "
                    f"| {'>'.join(ops[-3:])} | {' > '.join(ann[-2:])}"
                )
    elif mode == "map":
        rows = int(sys.argv[3])
        sel = plain(gpu, regions, rows)
        per = collections.defaultdict(lambda: collections.defaultdict(float))
        cnt = collections.defaultdict(lambda: collections.defaultdict(int))
        seg_span = collections.defaultdict(list)
        seg_sum = collections.defaultdict(list)
        streams = collections.defaultdict(set)
        order = []
        for ri, (a, b) in enumerate(sel):
            cls = classify(gpu, attr, a, b)
            first = {}
            sums = collections.defaultdict(float)
            for k, (seg, what, em) in enumerate(cls):
                e = gpu[a + k]
                key = (seg, what, em)
                if key not in order:
                    order.append(key)
                per[key][ri] += e["dur"]
                cnt[key][ri] += 1
                streams[key].add(e["args"].get("stream"))
                first.setdefault(seg, e["ts"])
                sums[seg] += e["dur"]
            segs = [s for s, _ in SEGMENTS if s in first]
            for s, nxt in zip(segs, segs[1:] + [None]):
                end = gpu[b]["ts"] if nxt is None else first[nxt]
                seg_span[s].append(end - first[s])
                seg_sum[s].append(sums[s])
        spans = [gpu[b]["ts"] - gpu[a]["ts"] for a, b in sel]
        print(
            f"# {rows}-row steps: {len(sel)} regions, {sel[0][1] - sel[0][0]} kernels and copies a region, span median {statistics.median(spans):.1f} us"
        )
        med = statistics.median
        for seg, title in SEGMENTS:
            if seg not in seg_span:
                continue
            print(
                f"\n## {title}: span {med(seg_span[seg]):.1f} us, kernels {med(seg_sum[seg]):.1f} us"
            )
            for key in order:
                if key[0] != seg:
                    continue
                n = med(list(cnt[key].values()))
                us = med(list(per[key].values()))
                st = (
                    "main" if streams[key] == {2} else f"streams {sorted(streams[key])}"
                )
                print(f"  {us:7.1f} us  x{n:<4g} {key[1]:<62} {st:<16} {key[2]}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
