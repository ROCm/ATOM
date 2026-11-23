# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DSpark block's paged metadata in one launch (``step_gap.block_meta``)
against ``DSparkProposer._build_paged_block_metadata`` itself, on one GPU: a
drafter built without the engine (the real class, its real buffer allocation
and its real helpers; only the persistent MLA buffers' sizing, which asks the
engine's process groups, is replaced by the same sizes from aiter). A case
runs the drafter's own method, keeps what it left, puts every buffer back as
it was and runs the mirror on the same buffers (the MLA plan holds addresses,
so the two must write the same storage to be comparable); the drafter's own
method run twice the same way says which buffers it leaves the same at all.

Per case (requests running / scheduled, drafts a request, DCP rank; anchors at
0, on page boundaries, past the table, random): every buffer the method fills
(positions, slots, context lengths, the KV index pointers and indices, the
borrowed page column, the MLA plan's six buffers), the block tables after it,
and every field of the attention metadata, compared whole. Then the launches
of one eager build and the CUDA-graph time of a build, the drafter's own
against the mirror.

    python tools/kimi_k3_step_gap_block_meta_check.py
    env: ROUNDS (40 random cases a shape), GN (8), REPS (9), BENCH (1),
    ALT_PROPOSER (another tree's dspark_proposer.py: its two mirrored methods
    as a second reference, and whether the mirror accepts their digests)
"""

import ast
import collections
import hashlib
import json
import os
import statistics
import tempfile
import textwrap
from types import SimpleNamespace as NS

import torch
from aiter import dtypes

from atom.model_ops.attentions.aiter_mla import (
    _MLA_META_SUPPORTS_MAX_SPLIT,
    _MLA_SPLIT_BUDGET_AUTO,
    get_mla_metadata_info_v1,
)
from atom.models.kimi_k3.step_gap import block_meta as BM
from atom.spec_decode.dspark_proposer import DSparkProposer

ROUNDS = int(os.environ.get("ROUNDS", "40"))
GN = int(os.environ.get("GN", "8"))
REPS = int(os.environ.get("REPS", "9"))
BENCH = os.environ.get("BENCH", "1") == "1"
MAX_BS, BLOCK, COLS, HEADS = 16, 16, 512, 96
KEEP = []
PS_NAMES = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)
BUFFERS = (
    "_blk_positions",
    "_blk_slots",
    "_blk_ctx_lens",
    "_blk_kv_indptr",
    "_blk_kv_indices",
    "_blk_saved_page0",
    "_blk_last_page_lens",
    "_blk_cu_seqlens_q",
)

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device="cpu").manual_seed(31)
FC = NS(
    kv_cache_data={
        "layer_93": NS(k_cache=torch.zeros(4, 1, 576, dtype=dtypes.fp8, device=dev))
    }
)


def drafter(drafts, dcp, rank, cls=DSparkProposer):
    """A drafter as its constructor leaves the block state, without the
    engine."""
    d = cls.__new__(cls)
    d.config = NS(max_num_seqs=MAX_BS, max_model_len=COLS * BLOCK * dcp)
    d.device, d.dtype = dev, torch.bfloat16
    d.mtp_k = drafts
    d.dcp_world_size, d.dcp_rank = dcp, rank
    d.speculative_config = NS(use_dspark_with_draft=lambda: True, draft_kv_dtype="fp8")
    d.model = NS(
        kv_pool_spec=None, block_attention_is_mla=True, first_attention_layer_num=93
    )
    tables = torch.zeros(MAX_BS, COLS, dtype=torch.int32, device=dev)
    d.runner = NS(block_size=BLOCK, forward_vars={"block_tables": NS(gpu=tables)})
    d._init_draft_block_buffers()
    kw = (
        {"max_split_per_batch": _MLA_SPLIT_BUDGET_AUTO}
        if _MLA_META_SUPPORTS_MAX_SPLIT
        else {}
    )
    sizes = get_mla_metadata_info_v1(
        MAX_BS,
        drafts,
        HEADS,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
        **kw,
    )
    d.ps = {
        name: torch.zeros(size, dtype=ty, device=dev)
        for name, (size, ty) in zip(PS_NAMES, sizes)
    }
    d._blk_padded_heads, d._blk_split_kwargs = HEADS, kw
    d._init_block_persistent_buffers = lambda dtype_q, dtype_kv: d.ps
    d.tables = tables
    KEEP.append(d)
    return d


def case(running, scheduled, drafts, dcp, kind):
    """Anchors (the padded tail at 0, as ``_stage_anchors`` leaves it) and a
    block table."""
    virtual = BLOCK * dcp
    last = COLS * virtual  # first position the table does not address
    if kind == "zero":
        anchors = torch.zeros(running, dtype=torch.int64)
    elif kind == "boundary":
        anchors = torch.randint(
            1, COLS - 1, (running,), generator=gen
        ) * virtual - torch.randint(1, drafts + 2, (running,), generator=gen)
    elif kind == "end":
        anchors = last - torch.randint(1, drafts + 3, (running,), generator=gen)
    else:
        anchors = torch.randint(0, last - drafts - 2, (running,), generator=gen)
    anchors[scheduled:] = 0
    tables = torch.randint(1, 1 << 20, (MAX_BS, COLS), generator=gen, dtype=torch.int32)
    return anchors.to(dev), tables.to(dev), int(anchors.max()) + 1 + drafts


def build(d, lean, anchors, scheduled, max_k):
    md = NS(max_seqlen_k=0)
    fn = BM.build_paged_block_metadata if lean else type(d)._build_paged_block_metadata
    fn(d, FC, md, anchors, scheduled_bs=scheduled, max_seqlen_k=max_k)
    return md


def other_proposer(path):
    """``DSparkProposer`` with the two mirrored methods that differ between
    trees taken from another tree's ``dspark_proposer.py`` (``ALT_PROPOSER``),
    and those methods' digests."""
    import atom.spec_decode.dspark_proposer as module

    with open(path) as f:
        source = f.read()
    found, digests = {}, {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_build_paged_block_metadata",
            "_paged_pad_ctx_len",
        ):
            text = ast.get_source_segment(source, node)
            scope = {}
            exec(  # noqa: S102 -- the other tree's two methods, as a reference
                compile(textwrap.dedent("    " + text), f"other:{node.name}", "exec"),
                vars(module),
                scope,
            )
            found[node.name] = scope[node.name]
            digests[node.name] = hashlib.sha256(
                "".join(text.split()).encode()
            ).hexdigest()[:16]
    found["_paged_pad_ctx_len"] = property(found["_paged_pad_ctx_len"])
    return type("OtherProposer", (DSparkProposer,), found), digests


def state(d):
    out = {name: getattr(d, name) for name in BUFFERS}
    out.update({f"ps.{name}": d.ps[name] for name in PS_NAMES})
    out["block_tables"] = d.tables
    return out


def snapshot(d, md=None):
    snap = {name: t.clone() for name, t in state(d).items()}
    if md is not None:
        for name, value in md.__dict__.items():
            snap[f"metadata.{name}"] = (
                value.clone() if isinstance(value, torch.Tensor) else value
            )
    torch.cuda.synchronize()
    return snap


def restore(d, snap):
    for name, t in state(d).items():
        t.copy_(snap[name])


def differences(a, b):
    bad = [f"fields {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
    for name, x in a.items():
        y = b.get(name)
        if isinstance(x, torch.Tensor):
            if not (
                isinstance(y, torch.Tensor)
                and x.shape == y.shape
                and x.dtype == y.dtype
                and torch.equal(x, y)
            ):
                bad.append(name)
        elif x != y:
            bad.append(name)
    return bad


def check_shape(running, scheduled, drafts, dcp, rank, cls=DSparkProposer):
    d = drafter(drafts, dcp, rank, cls)
    own_tree = cls is DSparkProposer
    assert not own_tree or BM.lean_refusal(d) is None
    kinds = ["zero", "boundary", "end"] + ["random"] * ROUNDS
    bad, unstable, slots_dropped = [], set(), 0
    for kind in kinds:
        anchors, tables, max_k = case(running, scheduled, drafts, dcp, kind)
        d.tables.copy_(tables)
        before = snapshot(d)
        own = snapshot(d, build(d, False, anchors.clone(), scheduled, max_k))
        restore(d, before)
        again = snapshot(d, build(d, False, anchors.clone(), scheduled, max_k))
        unstable.update(differences(own, again))
        restore(d, before)
        lean = snapshot(d, build(d, True, anchors.clone(), scheduled, max_k))
        diff = [name for name in differences(own, lean) if name not in unstable]
        if diff:
            bad.append({"kind": kind, "differences": diff[:8]})
        slots_dropped += int((own["_blk_slots"][: running * drafts] < 0).sum())
    bound = True
    if own_tree:
        # through the instance, as the drafter's callers reach it
        installed = drafter(drafts, dcp, rank)
        bound = (
            BM.install(installed)
            and installed._build_paged_block_metadata.__func__
            is BM.build_paged_block_metadata
        )
    res = {
        "proposer": "this tree's" if own_tree else "the other tree's (ALT_PROPOSER)",
        "running": running, "scheduled": scheduled, "drafts": drafts, "dcp": dcp, "rank": rank, "cases": len(kinds),
        "mismatched_cases": len(bad), "first": bad[:2], "own_method_unstable_in": sorted(unstable),
        "compared": len(own) - len(unstable), "slots_dropped": slots_dropped,
        "slots_kept": int((own["_blk_slots"][: running * drafts] >= 0).sum()), "install_binds": bool(bound),
    }  # fmt: skip
    print(json.dumps(res), flush=True)
    return not bad and bound and not unstable, d


def graph_of(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    KEEP.append((g, fn))
    return g


def timed(g):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def launches(fn):
    """The GPU launches (kernels and copies) of one eager call."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    os.unlink(path)
    events = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")
    ]
    names = collections.Counter(e["name"][:48] for e in events)
    return len(events), names.most_common(3)


def bench(running, scheduled, drafts, dcp, rank, d):
    anchors, tables, max_k = case(running, scheduled, drafts, dcp, "random")
    d.tables.copy_(tables)
    KEEP.append(anchors)

    def body(is_lean):
        def go():
            for _ in range(GN):
                build(d, is_lean, anchors, scheduled, max_k)

        return go

    graphs = {"own": graph_of(body(False)), "lean": graph_of(body(True))}
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    out = {"running": running, "scheduled": scheduled, "drafts": drafts}
    for name, is_lean in (("own", False), ("lean", True)):
        count, top = launches(
            lambda is_lean=is_lean: build(d, is_lean, anchors, scheduled, max_k)
        )
        out[name] = {
            "us_per_build": round(statistics.median(times[name]), 2), "min": round(min(times[name]), 2),
            "max": round(max(times[name]), 2), "launches_per_build": count, "most_launched": top,
        }  # fmt: skip
    print(json.dumps(out), flush=True)


ok = True
SHAPES = [
    (1, 1, 7, 8, 0), (2, 2, 7, 8, 5), (3, 3, 7, 8, 7), (8, 8, 7, 8, 0), (8, 5, 7, 8, 3), (16, 16, 7, 8, 1), (16, 9, 7, 8, 6),
    (8, 8, 4, 8, 2), (16, 11, 4, 8, 7), (4, 4, 7, 2, 1), (8, 1, 7, 8, 4),
]  # fmt: skip
for shape in SHAPES:
    good, d = check_shape(*shape)
    ok = ok and good
    if BENCH and shape[:2] in ((1, 1), (8, 8), (8, 5)) and shape[2] == 7:
        bench(*shape, d)
ALT = os.environ.get("ALT_PROPOSER")
if ALT:
    Other, digests = other_proposer(ALT)
    accepted = all(digests[name] in BM.MIRRORED[name] for name in digests)
    print(
        json.dumps(
            {"other_proposer_digests": digests, "accepted_by_the_mirror": accepted}
        ),
        flush=True,
    )
    ok = ok and accepted
    for shape in SHAPES:
        good, _ = check_shape(*shape, cls=Other)
        ok = ok and good
print(
    json.dumps({"all_ok": ok, "shapes": len(SHAPES), "other_proposer": bool(ALT)}),
    flush=True,
)
