# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step tail's host side on one GPU without the engine: the real
``StepTailDecodeModel`` and ``StepTailRunner`` driven as ``ModelRunner`` and
the draft block's epilogue drive them, against production on the same inputs.

Only what one process cannot have is replaced: the TP group (this process is
rank RANK of 8 and every agreement is its own verdict), the peer mapping (the
eight regions are local buffers between guard bands: no symmetric memory is
allocated or mapped), the mono forward under the wrapper, and the engine's
recorded forward (a stand-in; the tool puts the hidden rows where a replay
leaves them). The host plays the other seven ranks as in
``steptail_check.py`` / ``steptail_markov_check.py``.

Run: a deployment's install, the runner's refusals, a served verify step a
width (ids, verdict, regions, counters, the padded batch), the steps the
override declines, the waits that give up (each fails the step), the
draft block's wrapped epilogue eager and in a CUDA graph, and its refusals.
At the end: the lines of ``runner.py`` and ``dispatch.py`` that ran.

    python tools/steptail_runner_check.py
    env: RANK (5), WIDTHS (64x8,8x8,40x5: rows x qlen), BLOCKS (8x7,8x4,2x7:
    requests x drafts), REPLAYS (6)
"""

import logging
import os
import sys
import time
from types import SimpleNamespace as NS

os.environ["ATOM_KIMI_MONO_STEP_TAIL"] = "1"
RAN = {"step_tail/runner.py": set(), "step_tail/dispatch.py": set()}


def _trace(frame, event, arg):
    for name, lines in RAN.items():
        if frame.f_code.co_filename.endswith(name):
            lines.add(frame.f_lineno)

            def local(frame, event, arg, lines=lines):
                if event == "line":
                    lines.add(frame.f_lineno)
                return local

            return local
    return None


sys.settrace(_trace)  # before the two modules are imported

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steptail_check as C
import steptail_markov_check as MC
from aiter.dist import parallel_state

from atom.models.kimi_k3.mono import symm_peers
from atom.models.kimi_k3.mono.dispatch import MonoDecodeModel
from atom.models.kimi_k3.step_tail import dispatch as tail
from atom.models.kimi_k3.step_tail import kernels as K
from atom.models.kimi_k3.step_tail import layout as L
from atom.models.kimi_k3.step_tail import markov_layout as M
from atom.models.kimi_k3.step_tail import runner as R
from atom.mono.runtime import consensus
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils.forward_context import request_bucket

RANK = int(os.environ.get("RANK", "5"))
WIDTHS = [
    tuple(int(v) for v in s.split("x"))
    for s in os.environ.get("WIDTHS", "64x8,8x8,40x5").split(",")
]
BLOCKS = [
    tuple(int(v) for v in s.split("x"))
    for s in os.environ.get("BLOCKS", "8x7,8x4,2x7").split(",")
]
REPLAYS = int(os.environ.get("REPLAYS", "6"))
dev, fail = C.dev, C.fail
WORLD = {"tp": L.TP, "peers": True}
REGIONS = []
FC = NS(spec_decode_metadata=None, attn_metadata=NS(max_seqlen_q=8))


class Seen(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append((record.levelname, record.getMessage()))


SEEN = Seen()
logging.getLogger("atom").addHandler(SEEN)
logging.getLogger("atom").setLevel(logging.INFO)


def local_peers(nbytes, counters, group, tp, device):
    """``map_symm_peers`` for one process: every rank's region a local buffer."""
    if not WORLD["peers"]:
        return None
    if nbytes != L.PEER_BYTES or tp != L.TP:
        raise RuntimeError(f"peer regions of {nbytes} bytes for TP {tp}")
    regions = torch.full((L.TP * C.SPAN,), C.PATTERN, dtype=torch.uint8, device=device)
    base = regions.data_ptr()
    peers = torch.tensor(
        [base + o * C.SPAN + C.GUARD for o in range(L.TP)],
        dtype=torch.int64,
        device=device,
    )
    REGIONS.append(regions)
    C.KEEP.append(regions)
    mine = RANK * C.SPAN + C.GUARD
    return symm_peers.SymmPeers(
        regions[mine : mine + nbytes],
        peers,
        torch.zeros(counters, dtype=torch.int32, device=device),
    )


def one_process():
    """This process as TP rank RANK of ``WORLD["tp"]``."""
    parallel_state.get_tensor_model_parallel_world_size = lambda: WORLD["tp"]
    parallel_state.get_tensor_model_parallel_rank = lambda: RANK
    parallel_state.get_tp_group = lambda: NS(cpu_group=None)
    consensus.group_agree = lambda ok, group: ok
    symm_peers.map_symm_peers = local_peers
    tail.get_forward_context = lambda: FC
    tail.mono_dispatch.get_forward_context = lambda: NS(context=None)
    tail.mono_dispatch.install_mono_decode = (
        lambda model, cfg, drafter, metadata_builder=None: MonoDecodeModel(
            model, drafter, metadata_builder
        )
    )


class Inner(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.lm_head = NS(weight=weight, bias=bias)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        return "the model's own forward"


class Recorded:
    """The engine's recorded forwards (``ReqGraph``)."""

    def __init__(self):
        self.keys, self.replayed = set(), []

    def has(self, key):
        return key in self.keys

    def replay(self, key):
        self.replayed.append(key)


class Drafter:
    lossless = None
    verify_scheduler = None

    def __init__(self, w1, w2, deferred=True):
        self.runner = NS(
            req_graph=Recorded(),
            tokenID_processor=NS(is_deferred_out=deferred),
            forward_vars={
                "outputs": torch.zeros(
                    L.MAX_ROWS, L.HIDDEN, dtype=torch.bfloat16, device=dev
                )
            },
        )
        self.model = NS(
            markov_head=NS(markov_w1=NS(weight=w1), markov_w2=NS(weight=w2)),
            head_and_sample=self.own_head,
        )
        self.kv, self.own = [], []

    def own_head(self, out, anchor_ids, num_draft, *, greedy=False):
        self.own.append((anchor_ids.shape[0], num_draft))
        return "the model's own head", None

    def compute_draft_kv(self, positions, hidden_states, next_token_ids):
        self.kv.append(hidden_states)


def refuse_null(tab, names):
    table = tab.tolist()
    missing = [name for name in names if not table[L.POINTERS.index(name)]]
    if missing:
        raise RuntimeError(f"null table pointers {missing}: not launching")


class VerifySide(C.Fixture):
    """``steptail_check``'s staging of a step on the runner's own buffers."""

    def __init__(self, runner, regions, rows, qlen, hidden, ids):
        self.rows, self.qlen, self.rank, self.tapped = rows, qlen, runner.rank, False
        self.x, self.ids = hidden, ids
        self.cnt, self.tab, self.regions = runner.cnt, runner.tab, regions

    def refuse_null_pointers(self):
        refuse_null(self.tab, ("w", "rec", "cnt", "peers", "outtok", "verdict"))


class DraftSide(MC.Fixture):
    """``steptail_markov_check``'s staging of a block on the runner's buffers."""

    def __init__(self, runner, regions, requests, drafts):
        self.requests, self.drafts, self.rank = requests, drafts, runner.rank
        self.x = torch.zeros(
            requests * drafts, L.HIDDEN, dtype=torch.bfloat16, device=dev
        )
        self.anchors = torch.zeros(requests, dtype=torch.int32, device=dev)
        self.cnt, self.tab, self.regions = runner.cnt, runner.tab, regions
        C.KEEP.append(self)

    def refuse_null_pointers(self):
        refuse_null(
            self.tab, ("w", "w1", "w2", "rec", "cnt", "peers", "drafts", "mbest")
        )


def regions_of(runner):
    mine = RANK * C.SPAN + C.GUARD
    return next(
        r for r in REGIONS if r.data_ptr() + mine == runner.peers.buffer.data_ptr()
    )


def report(what, checks):
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        fail(f"{what}: {bad}")
    return not bad


def refused(what, make):
    try:
        make()
    except MonoUnsupported as why:
        print(f"  refused, {what}: {why}", flush=True)
        return True
    fail(f"{what}: not refused")
    return False


def batch_of(requests, bs=None, **changes):
    fields = {
        "structured_requests": None,
        "required_channel": None,
        "any_logprobs": False,
        "total_seqs_num": bs or requests,
        "temperatures": np.zeros(requests, dtype=np.float32),
    }
    fields.update(changes)
    return NS(**fields)


def offer(m, drafter, side, bs=None):
    """The step as ``ModelRunner.run_model`` offers it to the model."""
    requests = side.rows // side.qlen
    FC.spec_decode_metadata = NS(
        num_spec_steps=side.qlen - 1,
        num_draft_tokens_np=np.full(requests, side.qlen - 1),
        input_ids=side.ids,
    )
    FC.attn_metadata = NS(max_seqlen_q=side.qlen)
    drafter.runner.req_graph.keys.add(request_bucket(requests, side.qlen))
    return m.step_override(side.ids, requests, batch_of(requests, bs))


def verify_case(what, m, drafter, side, w, salt, bs=None):
    rows, qlen = side.rows, side.qlen
    requests = rows // qlen
    bs = bs or requests
    x = C.bf16(rows, L.HIDDEN)
    logits = C.production_logits(w, x)
    ids = C.draft_ids(logits, qlen, salt)
    want_out, want_verdict = C.production_verify(logits, ids, qlen)
    runner = m._tail.runner
    before = runner.cnt.tolist()
    epoch, expected = side.stage(w, x, ids, logits)
    replayed = len(drafter.runner.req_graph.replayed)
    served = offer(m, drafter, side, bs)
    if served is not runner:
        fail(f"{what}: the step was not served")
        return False
    out, verdict = served.verify_outputs(bs)
    drafter.compute_draft_kv("positions", None, "next ids")
    served.output_stream.synchronize()
    torch.cuda.synchronize()
    c = runner.cnt.tolist()
    return report(
        what,
        {
            "ids": torch.equal(out, want_out[:bs]),
            "verdict": torch.equal(verdict, want_verdict[:, :bs]),
            "regions and guards": torch.equal(side.regions, expected),
            "the recorded forward replayed once": drafter.runner.req_graph.replayed[
                replayed:
            ]
            == [request_bucket(requests, qlen)],
            "counters": c[L.CNT_EPOCH] == epoch
            and c[L.CNT_TIMEOUT] == before[L.CNT_TIMEOUT],
            "the draft KV got the served rows": drafter.kv[-1] is not None
            and drafter.kv[-1].data_ptr() == side.x.data_ptr(),
        },
    )


def verify_steps(m, drafter, w):
    passed = cases = 0
    sides = {}
    for rows, qlen in WIDTHS:
        ids = torch.zeros(rows, dtype=torch.int32, device=dev)
        # the warmup forward builds the width's tail beside the mono kernels
        hidden = m._mono_forward(ids, "positions", qlen)
        runner = m._tail.runner
        if (
            hidden != "the mono rows"
            or runner is None
            or (rows, qlen) not in runner.builds
        ):
            fail(f"s{rows} q{qlen}: the warmup forward built no tail")
            continue
        side = VerifySide(
            runner,
            regions_of(runner),
            rows,
            qlen,
            drafter.runner.forward_vars["outputs"][:rows],
            ids,
        )
        sides[(rows, qlen)] = side
        for it in range(2):
            passed += verify_case(f"s{rows} q{qlen} step {it}", m, drafter, side, w, it)
            cases += 1
        if rows // qlen >= 2:  # a padded batch: fewer scheduled requests than rows
            bs = rows // qlen - 1
            passed += verify_case(
                f"s{rows} q{qlen} bs {bs}", m, drafter, side, w, 2, bs=bs
            )
            cases += 1
        print(f"served s{rows} q{qlen}: {passed}/{cases} steps so far", flush=True)
    print(f"served verify steps: {passed}/{cases}", flush=True)
    return sides


def declined_steps(m, drafter, side):
    """Steps the override leaves to the engine: nothing replayed or launched."""
    requests = side.rows // side.qlen
    runner = m._tail.runner
    spec = NS(
        num_spec_steps=side.qlen - 1,
        num_draft_tokens_np=np.full(requests, side.qlen - 1),
        input_ids=side.ids,
    )
    hot = np.zeros(requests, dtype=np.float32)
    hot[0] = 0.7
    ragged = np.full(requests, side.qlen - 1)
    ragged[0] -= 1
    cases = {
        "no batch": (spec, None),
        "logprobs": (spec, batch_of(requests, any_logprobs=True)),
        "structured output": (spec, batch_of(requests, structured_requests=[1])),
        "a required channel": (spec, batch_of(requests, required_channel="x")),
        "a sampled request": (spec, batch_of(requests, temperatures=hot)),
        "no drafts": (None, batch_of(requests)),
        "a ragged verify": (
            NS(**{**vars(spec), "num_draft_tokens_np": ragged}),
            batch_of(requests),
        ),
        "int64 ids": (
            NS(**{**vars(spec), "input_ids": side.ids.long()}),
            batch_of(requests),
        ),
    }
    before = (runner.cnt.tolist(), len(drafter.runner.req_graph.replayed))
    kept = 0
    for what, (spec_now, batch) in cases.items():
        FC.spec_decode_metadata = spec_now
        FC.attn_metadata = NS(max_seqlen_q=side.qlen)
        if m.step_override(side.ids, requests, batch) is None:
            kept += 1
        else:
            fail(f"declined steps: {what} was served")
    # a width with no recorded forward
    FC.spec_decode_metadata = spec
    drafter.runner.req_graph.keys.clear()
    kept += m.step_override(side.ids, requests, batch_of(requests)) is None
    torch.cuda.synchronize()
    after = (runner.cnt.tolist(), len(drafter.runner.req_graph.replayed))
    ok = report("declined steps", {"nothing replayed or launched": before == after})
    print(
        f"declined steps: {kept}/{len(cases) + 1} kept the engine's tail, nothing launched: {ok}; reasons logged: "
        f"{sorted(m._tail_declines)}",
        flush=True,
    )


def fresh(cfg, w, w1, w2, rows, qlen, deferred=True):
    """A deployment of its own (a wait that gives up is the end of its runner),
    warmed at one width."""
    drafter = Drafter(w1, w2, deferred)
    m = tail.install_mono_decode(Inner(C.shard(w, RANK)), cfg, drafter)
    m._mono = NS(runner=NS(forward=lambda ids, positions, qlen: "the mono rows"))
    ids = torch.zeros(rows, dtype=torch.int32, device=dev)
    m._mono_forward(ids, "positions", qlen)
    runner = m._tail.runner
    side = VerifySide(
        runner,
        regions_of(runner),
        rows,
        qlen,
        drafter.runner.forward_vars["outputs"][:rows],
        ids,
    )
    return m, drafter, side


def staged(side, w, salt, omit=None):
    """A step's inputs and its peers' lines in place (``omit``: one withheld)."""
    x = C.bf16(side.rows, L.HIDDEN)
    logits = C.production_logits(w, x)
    ids = C.draft_ids(logits, side.qlen, salt)
    side.stage(w, x, ids, logits, omit=omit)


def raised(act):
    """The RuntimeError ``act`` raises, or None."""
    try:
        act()
    except RuntimeError as why:
        return str(why)
    return None


def give_up_cases(cfg, w, w1, w2):
    """A peer line that never comes, at the serving bound: the step fails
    before the engine could read its ids on the host."""
    rows, qlen = 16, 8
    requests = rows // qlen
    late = ((RANK + 1) % L.TP, 0)
    verify = f"gave up so far (rank {RANK}, last site {K.SITE_SAMPLE})"
    # a model that never served: nothing to settle
    quiet = tail.StepTailDecodeModel(Inner(C.shard(w, RANK)), Drafter(w1, w2))
    still = quiet.forward(torch.zeros(4, dtype=torch.int32, device=dev), "positions")

    # deferred output, the count on the host by the next offer
    m, drafter, side = fresh(cfg, w, w1, w2, rows, qlen)
    staged(side, w, 0, omit=late)
    served = offer(m, drafter, side)
    stale = raised(lambda: served.verify_outputs(requests))
    staged(side, w, 1)
    torch.cuda.synchronize()
    runner = m._tail.runner
    before = (runner.cnt.tolist(), len(drafter.runner.req_graph.replayed))
    at_offer = raised(lambda: offer(m, drafter, side))
    torch.cuda.synchronize()
    after = (runner.cnt.tolist(), len(drafter.runner.req_graph.replayed))
    again = raised(lambda: m.step_override(side.ids, requests, None))
    report(
        "give-up, deferred output",
        {
            "the stale step itself hands its outputs over": stale is None,
            "one wait gave up": before[0][L.CNT_TIMEOUT] == 1,
            "the next offer fails": at_offer is not None and verify in at_offer,
            "before anything is replayed or launched": before == after,
            "and every step after": again == at_offer,
        },
    )
    print(f"give-up, deferred output: the next offer raises {at_offer!r}", flush=True)

    # deferred output, the count still queued: a step left to the engine, a
    # forward of the engine's own and the next step's outputs all wait for it
    waited = {}
    for what in ("a step left to the engine", "an engine forward", "the next outputs"):
        m, drafter, side = fresh(cfg, w, w1, w2, rows, qlen)
        staged(side, w, 0, omit=late)
        served = offer(m, drafter, side)
        served.verify_outputs(requests)
        act = {
            "a step left to the engine": lambda m=m, side=side: m.step_override(
                side.ids, requests, None
            ),
            "an engine forward": lambda m=m, side=side: m.forward(
                side.ids, "positions"
            ),
            "the next outputs": lambda served=served: served.verify_outputs(requests),
        }[what]
        t0 = time.perf_counter()
        why = raised(act)
        waited[what] = (why, time.perf_counter() - t0)
        torch.cuda.synchronize()
    report(
        "give-up, count still queued",
        {
            f"{what} waits for the launch and fails": why is not None
            and verify in why
            and took > 0.5
            for what, (why, took) in waited.items()
        },
    )
    print(
        "give-up, count still queued: "
        + "; ".join(
            f"{what} raised after {took:.1f} s" for what, (_, took) in waited.items()
        ),
        flush=True,
    )

    # output read at once: the step's own outputs fail
    m, drafter, side = fresh(cfg, w, w1, w2, rows, qlen, deferred=False)
    staged(side, w, 0, omit=late)
    served = offer(m, drafter, side)
    own = raised(lambda: served.verify_outputs(requests))
    torch.cuda.synchronize()
    report(
        "give-up, output read at once",
        {"the step's own outputs fail": own is not None and verify in own},
    )
    print(
        f"give-up, output read at once: the step's own outputs raise {own!r}",
        flush=True,
    )

    # a draft block's give-up: in the next verify launch's count, so the step
    # after the one that verified those drafts fails (its ids never reach the host)
    m, drafter, side = fresh(cfg, w, w1, w2, rows, qlen)
    runner = m._tail.runner
    block = DraftSide(runner, regions_of(runner), requests, 7)
    xd, anchors = MC.inputs(requests, 7)
    _, key, gid, _ = MC.production(w, w1, w2, xd, anchors, requests, 7)
    block.stage(w, w1, w2, xd, anchors, key, gid, omit=((RANK + 1) % L.TP, 0, 0))
    drafted = raised(
        lambda: drafter.model.head_and_sample(block.x, block.anchors, 7, greedy=True)
    )
    staged(side, w, 0)
    served = offer(m, drafter, side)
    verified = raised(lambda: served.verify_outputs(requests))
    staged(side, w, 1)
    torch.cuda.synchronize()
    after_block = raised(lambda: offer(m, drafter, side))
    torch.cuda.synchronize()
    report(
        "give-up in a draft block",
        {
            "a model that never served settles nothing": still
            == "the model's own forward",
            "the block and the step verifying its drafts go on": drafted is None
            and verified is None,
            "one wait gave up": runner.cnt.tolist()[L.CNT_TIMEOUT] == 1,
            "the next offer fails": after_block is not None
            and f"last site {K.SITE_MARKOV})" in after_block,
        },
    )
    print(
        f"give-up in a draft block: the offer after the next raises {after_block!r}",
        flush=True,
    )


def draft_case(what, m, side, weights, fire):
    w, w1, w2 = weights
    requests, drafts = side.requests, side.drafts
    xd, anchors = MC.inputs(requests, drafts)
    want, key, gid, _ = MC.production(w, w1, w2, xd, anchors, requests, drafts)
    runner = m._tail.runner
    before = runner.cnt.tolist()
    epoch, expected = side.stage(w, w1, w2, xd, anchors, key, gid)
    ids = fire()
    torch.cuda.synchronize()
    c = runner.cnt.tolist()
    return report(
        what,
        {
            "ids": isinstance(ids, torch.Tensor) and torch.equal(ids, want),
            "regions and guards": torch.equal(side.regions, expected),
            "counters": c[M.CNT_MEPOCH] == epoch
            and c[M.CNT_MFLAG] == epoch * 8 + drafts
            and c[L.CNT_TIMEOUT] == before[L.CNT_TIMEOUT],
        },
    )


def draft_blocks(m, drafter, weights):
    w, w1, w2 = weights
    runner = m._tail.runner
    head = drafter.model.head_and_sample  # the wrapped epilogue
    for requests, drafts in BLOCKS:
        side = DraftSide(runner, regions_of(runner), requests, drafts)
        own = len(drafter.own)

        def eager(side=side, drafts=drafts):
            ids, extra = head(side.x, side.anchors, drafts, greedy=True)
            return ids if extra is None else None

        passed = sum(
            draft_case(f"b{requests} t{drafts} eager {it}", m, side, weights, eager)
            for it in range(2)
        )
        table = runner.tab.tolist()
        bound = report(
            f"b{requests} t{drafts}",
            {
                "the Markov tables bound in place": table[L.POINTERS.index("w1")]
                == w1.data_ptr()
                and table[L.POINTERS.index("w2")] == w2.data_ptr()
                and runner.markov_rank == w1.shape[1],
                "the model's own head never ran": len(drafter.own) == own,
            },
        )
        # recorded with the block pass, as DraftGraph records the epilogue: a capture may run the launch, so its
        # epoch's lines are staged too
        xd, anchors = MC.inputs(requests, drafts)
        _, key, gid, _ = MC.production(w, w1, w2, xd, anchors, requests, drafts)
        side.stage(w, w1, w2, xd, anchors, key, gid)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            held = head(side.x, side.anchors, drafts, greedy=True)[0]
        torch.cuda.synchronize()
        C.KEEP.append((graph, held))
        runner.cnt[L.CNT_TIMEOUT] = 0

        def replay(graph=graph, held=held):
            graph.replay()
            return held

        replayed = sum(
            draft_case(f"b{requests} t{drafts} replay {it}", m, side, weights, replay)
            for it in range(REPLAYS)
        )
        print(
            f"draft block b{requests} t{drafts} through the wrapped epilogue: {passed}/2 eager, "
            f"{replayed}/{REPLAYS} replays, bound: {bound}",
            flush=True,
        )
    # blocks that keep the model's own head: nothing launched
    side = DraftSide(runner, regions_of(runner), *BLOCKS[0])
    drafts = BLOCKS[0][1]
    before = runner.cnt.tolist()
    kept = {
        "int64 anchors": head(side.x, side.anchors.long(), drafts, greedy=True),
        "rows of another dtype": head(
            side.x.float(), side.anchors, drafts, greedy=True
        ),
        "a block past the kernel": head(
            torch.zeros(16 * 7, L.HIDDEN, dtype=torch.bfloat16, device=dev),
            torch.zeros(16, dtype=torch.int32, device=dev),
            7,
            greedy=True,
        ),
        "the same block again": head(
            torch.zeros(16 * 7, L.HIDDEN, dtype=torch.bfloat16, device=dev),
            torch.zeros(16, dtype=torch.int32, device=dev),
            7,
            greedy=True,
        ),
    }
    drafter.model.draft_sampling = object()
    kept["sampled drafting"] = head(side.x, side.anchors, drafts, greedy=False)
    del drafter.model.draft_sampling
    drafter.model.confidence_head = object()
    kept["a confidence head"] = head(side.x, side.anchors, drafts, greedy=True)
    del drafter.model.confidence_head
    torch.cuda.synchronize()
    report(
        "kept draft blocks",
        {
            **{what: got[0] == "the model's own head" for what, got in kept.items()},
            "nothing launched": runner.cnt.tolist() == before,
        },
    )
    print(
        f"draft blocks that keep the model's own head: {len(kept)}; refused shapes "
        f"{sorted(runner._draft_refused)}",
        flush=True,
    )


def dispatch_corners(m, drafter, side, cfg, w, w1, w2):
    """The dispatch model's remaining refusals and installs: nothing launched."""
    requests = side.rows // side.qlen
    runner = m._tail.runner
    before = runner.cnt.tolist()
    spec = NS(
        num_spec_steps=side.qlen - 1,
        num_draft_tokens_np=np.full(requests, side.qlen - 1),
        input_ids=side.ids,
    )
    FC.attn_metadata = NS(max_seqlen_q=side.qlen)
    drafter.runner.req_graph.keys.add(request_bucket(requests, side.qlen))
    kept = {}
    FC.spec_decode_metadata = NS(**{**vars(spec), "num_spec_steps": side.qlen - 2})
    kept["rows that are not the requests' steps"] = m.step_override(
        side.ids, requests, batch_of(requests)
    )
    FC.spec_decode_metadata = spec
    outputs = drafter.runner.forward_vars["outputs"]
    drafter.runner.forward_vars["outputs"] = outputs.float()
    kept["hidden rows of another dtype"] = m.step_override(
        side.ids, requests, batch_of(requests)
    )
    drafter.runner.forward_vars["outputs"] = outputs
    m._tail.enabled = False
    kept["a tail turned off"] = m.step_override(side.ids, requests, batch_of(requests))
    m._tail.enabled = True
    torch.cuda.synchronize()
    # a forward the engine records: nothing is settled inside a capture
    graph, scratch = torch.cuda.CUDAGraph(), torch.zeros(4, device=dev)
    with torch.cuda.graph(graph):
        scratch.add_(1)
        captured = m.forward(side.ids, "positions")
    torch.cuda.synchronize()
    # a drafter whose model has no head of its own; a wrapped head before any runner exists
    bare = Drafter(w1, w2)
    del bare.model.head_and_sample
    tail.StepTailDecodeModel(Inner(C.shard(w, RANK)), bare)
    early = Drafter(w1, w2)
    tail.StepTailDecodeModel(Inner(C.shard(w, RANK)), early)
    first = early.model.head_and_sample(
        torch.zeros(8 * 7, L.HIDDEN, dtype=torch.bfloat16, device=dev),
        torch.zeros(8, dtype=torch.int32, device=dev),
        7,
        greedy=True,
    )
    # installs: the switch off, mono itself off, an aiter without the chip query
    os.environ["ATOM_KIMI_MONO_STEP_TAIL"] = "0"
    off = tail.install_mono_decode(Inner(C.shard(w, RANK)), cfg, Drafter(w1, w2))
    os.environ["ATOM_KIMI_MONO_STEP_TAIL"] = "1"
    installer = tail.mono_dispatch.install_mono_decode
    plain = Inner(C.shard(w, RANK))
    chip = sys.modules.get("aiter.jit.utils.chip_info")
    try:
        tail.mono_dispatch.install_mono_decode = (
            lambda model, cfg, drafter, metadata_builder=None: model
        )
        stays = tail.install_mono_decode(plain, cfg, Drafter(w1, w2)) is plain
        tail.mono_dispatch.install_mono_decode = installer
        sys.modules["aiter.jit.utils.chip_info"] = None
        blind = tail.tail_refusal(cfg, Drafter(w1, w2))
    finally:
        tail.mono_dispatch.install_mono_decode = installer
        if chip is None:
            sys.modules.pop("aiter.jit.utils.chip_info", None)
        else:
            sys.modules["aiter.jit.utils.chip_info"] = chip
    report(
        "dispatch corners",
        {
            **{what: got is None for what, got in kept.items()},
            "nothing launched": runner.cnt.tolist() == before,
            "a forward inside a capture is the model's own": captured
            == "the model's own forward",
            "a model without a head of its own is left alone": not hasattr(
                bare.model, "head_and_sample"
            ),
            "a block before any runner keeps the model's own head": first[0]
            == "the model's own head"
            and early.own == [(8, 7)],
            "the switch off is the mono model": type(off) is MonoDecodeModel,
            "mono off stays as installed": stays,
            "no chip query refuses": blind == "not gfx1250",
        },
    )
    print(
        f"dispatch corners: {len(kept)} more steps kept the engine's tail; installs: switch off -> "
        f"{type(off).__name__}, mono off stays: {stays}, no chip query -> {blind!r}",
        flush=True,
    )


def runner_refusals(inner, drafter, w, w1, w2):
    WORLD["tp"] = 4
    refused("TP 4", lambda: R.StepTailRunner(inner, drafter))
    WORLD["tp"] = L.TP
    WORLD["peers"] = False
    refused("no peer memory", lambda: R.StepTailRunner(inner, drafter))
    WORLD["peers"] = True
    weight = C.shard(w, RANK)
    refused(
        "an LM head with a bias",
        lambda: R.StepTailRunner(Inner(weight, bias=object()), drafter),
    )
    refused(
        "an LM head of another shape",
        lambda: R.StepTailRunner(Inner(weight[:100]), drafter),
    )
    # a runner of its own: widths and blocks it has no build for
    runner = R.StepTailRunner(inner, None)
    scratch = torch.zeros(4, device=dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch.add_(1)
        inside = runner.prepare(48, 8), runner.prepare_draft(3, 7)
    torch.cuda.synchronize()
    untouched = not (
        runner.builds or runner._refused or runner.draft_builds or runner._draft_refused
    )
    odd = Drafter(w1[:, :100], w2[:, :100])
    report(
        "runner refusals",
        {
            "a width the layout refuses": not runner.prepare(9, 8)
            and not runner.prepare(9, 8)
            and (9, 8) in runner._refused,
            "nothing is built or refused inside a capture": inside == (False, False)
            and untouched,
            "no Markov head": not runner.prepare_draft(8, 7)
            and not runner.prepare_draft(8, 7),
            "Markov tables of another rank": not R.StepTailRunner(
                inner, odd
            ).prepare_draft(8, 7),
        },
    )
    print(
        f"runner refusals: widths {runner._refused}, blocks {runner._draft_refused}",
        flush=True,
    )
    # builds that are turned down: a GEMM dispatch the tail does not mirror, a build with scratch, a table
    # entry left unset (nothing is launched: a build runs under compile_only)
    from atom.models.kimi_k3.step_tail import head_dispatch
    from atom.mono.runtime import compile as mono_compile

    fresh = R.StepTailRunner(inner, drafter)
    splits, resources = head_dispatch.head_splits, mono_compile.launcher_resources

    def no_splits(rows):
        raise ValueError("a GEMM dispatch the tail does not mirror")

    try:
        head_dispatch.head_splits = no_splits
        unmirrored = not fresh.prepare(16, 8), not fresh.prepare_draft(4, 4)
        head_dispatch.head_splits = splits
        mono_compile.launcher_resources = lambda launch: [
            {"scratch": 16, "vgpr_spill": 0, "sgpr_spill": 0}
        ]
        hazard = not fresh.prepare(24, 8), not fresh.prepare_draft(2, 4)
    finally:
        head_dispatch.head_splits = splits
        mono_compile.launcher_resources = resources
    fresh.tab[L.POINTERS.index("mbest")] = 0
    unset = not fresh.prepare_draft(3, 4)
    report(
        "turned-down builds",
        {
            "a GEMM dispatch the tail does not mirror": unmirrored == (True, True),
            "a build with scratch": hazard == (True, True),
            "an unset table pointer": unset,
            "nothing kept": not fresh.builds and not fresh.draft_builds,
        },
    )
    print(
        f"turned-down builds: widths {fresh._refused}, blocks {fresh._draft_refused}",
        flush=True,
    )


def executable(path):
    with open(path) as f:
        code = compile(f.read(), path, "exec")
    out, todo = set(), [code]
    while todo:
        c = todo.pop()
        out |= {line for _, _, line in c.co_lines() if line}
        todo += [k for k in c.co_consts if hasattr(k, "co_lines")]
    return out


def main():
    one_process()
    w = C.bf16(L.VOCAB, L.HIDDEN, scale=0.02)
    w1 = C.bf16(L.VOCAB, MC.RK, scale=0.25)
    w2 = C.bf16(L.VOCAB, MC.RK, scale=0.25)
    C.KEEP.append((w, w1, w2))
    inner, drafter = Inner(C.shard(w, RANK)), Drafter(w1, w2)
    cfg = NS(
        tensor_parallel_size=L.TP,
        prefill_context_parallel_size=1,
        speculative_config=NS(
            method="dspark", num_speculative_tokens=7, synthetic_acceptance_rates=None
        ),
    )
    why = tail.tail_refusal(cfg, drafter)
    m = tail.install_mono_decode(inner, cfg, drafter)
    small = NS(**{**vars(cfg), "tensor_parallel_size": 4})
    report(
        "install",
        {
            f"this deployment is covered ({why})": why is None,
            "the installed model serves steps": isinstance(m, tail.StepTailDecodeModel),
            "TP 4 keeps the mono model": type(
                tail.install_mono_decode(
                    Inner(C.shard(w, RANK)), small, Drafter(w1, w2)
                )
            )
            is MonoDecodeModel,
        },
    )
    print(f"install: {type(m).__name__}; refusal {why}", flush=True)
    m._mono = NS(runner=NS(forward=lambda ids, positions, qlen: "the mono rows"))
    sides = verify_steps(m, drafter, w)
    if sides:
        side = next(iter(sides.values()))
        declined_steps(m, drafter, side)
        # an engine forward (a warmup, a prefill): not a served step, and the draft KV gets what the engine passes
        got = m.forward(side.ids, "positions")
        drafter.compute_draft_kv("positions", None, "next ids")
        report(
            "an engine forward",
            {
                "the model's own": got == "the model's own forward",
                "no served rows left": m._tail_served is None
                and drafter.kv[-1] is None,
            },
        )
        draft_blocks(m, drafter, (w, w1, w2))
        dispatch_corners(m, drafter, side, cfg, w, w1, w2)
    runner_refusals(inner, drafter, w, w1, w2)
    give_up_cases(cfg, w, w1, w2)
    sys.settrace(None)
    for name, module in (("step_tail/runner.py", R), ("step_tail/dispatch.py", tail)):
        lines = executable(module.__file__)
        never = sorted(lines - RAN[name])
        print(
            f"{name}: {len(lines) - len(never)} of {len(lines)} executable lines ran; never: {MC.spans(never)}",
            flush=True,
        )
    print("FAILED" if C.FAILED else "PASSED", len(C.FAILED), flush=True)


if __name__ == "__main__":
    main()
