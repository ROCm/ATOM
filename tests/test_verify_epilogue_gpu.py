# SPDX-License-Identifier: MIT
"""ATOM_MTP_FUSED_VERIFY_EPILOGUE: `verify_epilogue` against the eager chain it
replaces after a verify's sampler, on one GPU.

The chain: `rejection_sample` on the sampled target ids, `PresencePenaltyStep.mark`,
the next-anchor gather, `DraftPresencePenalty.begin` and `stage_verify_step0`. Both
run from the same state and must leave everything bit for bit the same: the output
rows, the verdict, the next ids, the presence masks, the draft penalty's buffers,
step 0's staged inputs and the anchor scatter into the target's ids. With
ATOM_MTP_VERIFY_HOST_HANDOFF the pinned slots must hold the output rows and the
status the D2H copies would have carried.
"""

import itertools
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_engine.output_transfer import D2HQueue, HostHandoffRing
from atom.model_ops import verify_epilogue as ve
from atom.model_ops.presence_penalty import (
    DraftPresencePenalty,
    PresencePenaltyState,
    PresencePenaltyStep,
)
from atom.model_ops.rejection_sampler import rejection_sample
from atom.spec_decode.eagle_fused_glue import stage_verify_step0
from atom.spec_decode.status import SpecStatusQueue

DEV = torch.device("cuda")
V = 248320
S = 3
Q = S + 1
H = 2048
MAX_SEQS = 20


def _case(bs, rbs, seed, accept=None, num_draft=None):
    """A verify of ``bs`` requests (``num_draft`` drafts each, default ``S``),
    sampled ids accepting ``accept`` drafts a request, and every buffer the
    chain touches, on fresh state."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nd = torch.full((bs,), S) if num_draft is None else torch.as_tensor(num_draft)
    lens = nd + 1
    T = int(lens.sum())
    starts = torch.cumsum(lens, 0) - lens
    ids = torch.randint(0, V, (4096 + 1,), generator=g, device=DEV, dtype=torch.int32)
    tli = torch.cat([s + torch.arange(n) for s, n in zip(starts, nd)]).to(DEV)
    cu = torch.cumsum(nd, 0).to(torch.int32).to(DEV)
    bli = (starts + nd).to(torch.int64).to(DEV)
    drafts = ids[tli + 1]
    if accept is None:
        accept = torch.randint(0, S + 1, (bs,), generator=g, device=DEV)
    accept = torch.minimum(accept.to(DEV), nd.to(DEV))
    pos = torch.cat([torch.arange(n) for n in nd]).to(DEV)
    acc = accept.repeat_interleave(nd.to(DEV))
    target = torch.randint(0, V, (T - bs,), generator=g, device=DEV, dtype=torch.int32)
    target = torch.where(pos < acc, drafts, target)
    target = torch.where((pos == acc) & (target == drafts), (drafts + 1) % V, target)
    bonus = torch.randint(0, V, (bs,), generator=g, device=DEV, dtype=torch.int32)
    state = PresencePenaltyState(V, MAX_SEQS, 8, DEV)
    state.mask.copy_(
        (torch.rand(state.mask.shape, generator=g, device=DEV) < 1e-3).to(torch.uint8)
    )
    slots = torch.randint(0, state.num_slots, (bs,), generator=g, device=DEV)
    seq_start = starts.to(torch.int32).to(DEV)
    row_seq = torch.repeat_interleave(torch.arange(bs), lens).to(torch.int32).to(DEV)
    step = NS(
        slots=slots.to(torch.int32),
        penalties=torch.rand(bs, generator=g, device=DEV) * 2,
        seq_start=seq_start,
        row_seq=row_seq,
        row_pos=(torch.arange(T, device=DEV) - seq_start.long()[row_seq.long()]).to(
            torch.int32
        ),
    )
    rows = max(MAX_SEQS, rbs)
    dp = NS(
        slots=torch.randint(0, 40, (rows,), generator=g, device=DEV, dtype=torch.int32),
        penalties=torch.rand(rows, generator=g, device=DEV),
        pos=torch.full((rows,), 2, dtype=torch.int32, device=DEV),
        ctx=torch.full((rows * Q,), 5, dtype=torch.int32, device=DEV),
    )
    staged = {
        "input_ids": torch.randint(
            0, V, (rbs, Q), generator=g, device=DEV, dtype=torch.int32
        ),
        "positions": torch.randint(0, 9, (rbs, Q), generator=g, device=DEV),
        "hidden_states": torch.randn(rbs, Q, H, generator=g, device=DEV).to(
            torch.bfloat16
        ),
        "last_token_indices": torch.randint(0, 9, (rbs,), generator=g, device=DEV),
    }
    return NS(
        bs=bs,
        rbs=rbs,
        T=T,
        S=S,
        ids=ids,
        tli=tli,
        cu=cu,
        bli=bli,
        target=target,
        bonus=bonus,
        state=state,
        step=step,
        dp=dp,
        positions=torch.randint(0, 100000, (3, rbs * Q), generator=g, device=DEV),
        hidden=torch.randn(T, H, generator=g, device=DEV).to(torch.bfloat16),
        staged=staged,
        uniform=num_draft is None,
    )


def _bind(c):
    """The case's state as the engine's objects, on copies (each path its own)."""
    ids = c.ids.clone()
    state = PresencePenaltyState.__new__(PresencePenaltyState)
    state.__dict__.update(c.state.__dict__)
    state.mask = c.state.mask.clone()
    step = PresencePenaltyStep(
        state,
        c.step.slots,
        c.step.penalties,
        c.step.seq_start,
        c.step.row_seq,
        c.step.row_pos,
        ids[: c.T],
    )
    step._anchors_marked = True
    dp = DraftPresencePenalty.__new__(DraftPresencePenalty)
    dp.state, dp.max_rows, dp.ctx_width = state, c.dp.slots.numel(), Q
    for k in ("slots", "penalties", "pos", "ctx"):
        setattr(dp, k, getattr(c.dp, k).clone())
    meta = NS(
        input_ids=ids[: c.T],
        num_spec_steps=c.S,
        cu_num_draft_tokens=c.cu,
        target_logits_indices=c.tli,
        bonus_logits_indices=c.bli,
    )
    staged = {k: v.clone() for k, v in c.staged.items()}
    return NS(ids=ids, state=state, step=step, dp=dp, meta=meta, staged=staged)


def _eager(c, b, with_step, with_dp, with_stage):
    out, verdict = rejection_sample(
        b.meta.input_ids,
        c.tli,
        c.S,
        c.cu,
        None,
        None,
        c.bonus,
        c.bli,
        target_token_ids=c.target,
    )
    if with_step:
        b.step.mark(out.view(c.bs, -1))
    nxt = torch.gather(out.view(c.bs, -1), 1, verdict[0].view(-1, 1)).view(c.bs)
    if with_dp:
        b.dp.begin(b.step if with_step else None, c.bs)
    if with_stage:
        stage_verify_step0(
            b.ids[1 : c.T + 1],
            nxt,
            verdict[2],
            c.positions[0],
            c.hidden,
            b.staged,
            scheduled_bs=c.bs,
            running_bs=c.rbs,
            query_width=Q,
            max_bs=max(MAX_SEQS, c.rbs),
        )
    return out, verdict, nxt


def _fused(c, b, with_step, with_dp, with_stage, host=None, uniform=False):
    stage = None
    if with_stage:
        stage = {
            "input_ids": b.ids[1 : c.T + 1],
            "positions": c.positions[0],
            "hidden_states": c.hidden,
            "staged": b.staged,
            "scheduled_bs": c.bs,
            "running_bs": c.rbs,
            "query_width": Q,
        }
    return ve.verify_epilogue(
        c.target,
        c.bonus,
        b.meta,
        max_bs=max(MAX_SEQS, c.rbs),
        presence_step=b.step if with_step else None,
        draft_presence=b.dp if with_dp else None,
        draft_presence_bs=c.bs,
        stage=stage,
        host=host,
        uniform=uniform,
    )


def _snapshot(b, res):
    out, verdict, nxt = res
    snap = {"out": out, "verdict": verdict, "next": nxt, "ids": b.ids}
    snap["mask"] = b.state.mask
    for k in ("slots", "pos", "ctx"):
        snap["dp_" + k] = getattr(b.dp, k)
    snap["dp_penalties"] = b.dp.penalties.view(torch.int32)
    for k, v in b.staged.items():
        snap["staged_" + k] = v.view(torch.int16) if v.dtype == torch.bfloat16 else v
    return snap


def _assert_same(a, b, why):
    for key in a:
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, (why, key)
        assert torch.equal(a[key], b[key]), (why, key)


_PARTS = list(itertools.product([False, True], repeat=3))


@pytest.mark.parametrize("bs", [1, 2, 3, 4, 5, 9, 13, 20])
@pytest.mark.parametrize("pattern", ["random", "all", "none"])
def test_the_epilogue_leaves_what_the_eager_chain_leaves(bs, pattern):
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20]
    rbs = next(s for s in sizes if s >= bs)
    accept = {
        "random": None,
        "all": torch.full((bs,), S),
        "none": torch.zeros(bs, dtype=torch.int64),
    }[pattern]
    for rbs_, (with_step, with_dp, with_stage) in itertools.product(
        sorted({bs, rbs, 20}), _PARTS
    ):
        c = _case(bs, rbs_, seed=bs * 31 + rbs_, accept=accept)
        for uniform in (False, True):
            a, b = _bind(c), _bind(c)
            eager = _snapshot(a, _eager(c, a, with_step, with_dp, with_stage))
            fused = _snapshot(
                b, _fused(c, b, with_step, with_dp, with_stage, uniform=uniform)
            )
            torch.cuda.synchronize()
            _assert_same(eager, fused, (rbs_, with_step, with_dp, with_stage, uniform))


@pytest.mark.parametrize("sbs,rbs", [(103, 128), (150, 160)])
def test_large_batches_stage_their_pad_rows(sbs, rbs):
    c = _case(sbs, rbs, seed=sbs)
    a, b = _bind(c), _bind(c)
    eager = _snapshot(a, _eager(c, a, True, True, True))
    fused = _snapshot(b, _fused(c, b, True, True, True, uniform=True))
    _assert_same(eager, fused, "large")


def test_a_ragged_verify_reads_the_metadata_indices():
    """Requests with fewer drafts (no staging: `stage_verify_step0` needs a
    uniform batch): the verdict, rows and marks from the metadata's indices."""
    c = _case(6, 8, seed=5, num_draft=[3, 1, 0, 2, 3, 3])
    for with_step, with_dp in itertools.product([False, True], repeat=2):
        a, b = _bind(c), _bind(c)
        eager = _snapshot(a, _eager(c, a, with_step, with_dp, False))
        fused = _snapshot(b, _fused(c, b, with_step, with_dp, False))
        _assert_same(eager, fused, (with_step, with_dp))


def test_the_host_handoff_holds_what_the_copies_carried():
    tok_ring, status_ring = HostHandoffRing(8, MAX_SEQS * Q), HostHandoffRing(8, 40)
    status_queue = SpecStatusQueue(MAX_SEQS, torch.cuda.Stream())
    tokens = D2HQueue(MAX_SEQS * Q, torch.cuda.Stream())
    expected = []
    for k, bs in enumerate([4, 1, 9, 20, 3, 5, 2, 7, 13, 6, 8]):
        c = _case(bs, bs, seed=100 + k)
        b = _bind(c)
        if k % 3 == 2:
            # an eager step between hand-offs: received in order
            out, verdict, _ = _eager(c, b, True, False, False)
            status_queue.send(verdict[:2])
            tokens.send(out)
        else:
            host = (tok_ring.claim((bs, Q)), status_ring.claim((2, bs)))
            out, verdict, _ = _fused(c, b, True, False, False, host=host, uniform=True)
            status_queue.send_direct(host[1])
            tokens.send_direct(host[0])
        expected.append((out.cpu().numpy(), verdict[:2].cpu().numpy()))
        if len(expected) > 2:
            # receive lagging two steps, as the runner does at most
            out_np, status_np = expected.pop(0)
            rejected, accepted = status_queue.receive()
            np.testing.assert_array_equal(accepted, status_np[0])
            np.testing.assert_array_equal(rejected, status_np[1])
            np.testing.assert_array_equal(tokens.receive(), out_np)
    status_queue.drain()
    tokens.drain()
    assert not any(tok_ring._busy) and not any(status_ring._busy)


def test_a_slot_is_not_claimed_again_before_its_receive():
    ring = HostHandoffRing(2, 8)
    first = ring.claim((2, 2))
    ring.claim((2, 2))
    with pytest.raises(RuntimeError):
        ring.claim((2, 2))
    first.release()
    ring.claim((2, 2))


def test_warmup_compiles_every_variant_serving_launches():
    def variants():
        return sum(len(e[0]) for e in ve._verify_epilogue_kernel.device_caches.values())

    dp = DraftPresencePenalty(PresencePenaltyState(V, MAX_SEQS, 8, DEV), 20, S, DEV)
    ve.warm_verify_epilogue_kernels(
        max_bs=20,
        num_spec_steps=S,
        vocab=V,
        hidden_size=H,
        hidden_dtype=torch.bfloat16,
        ids_dtype=torch.int32,
        device=DEV,
        draft_presence=dp,
        host_capacity=(MAX_SEQS * Q, 2 * MAX_SEQS),
    )
    before = variants()
    rings = HostHandoffRing(8, MAX_SEQS * Q), HostHandoffRing(8, 2 * MAX_SEQS)
    for bs, rbs, with_step, with_stage in [
        (2, 2, True, True),
        (9, 12, True, True),
        (5, 8, False, True),
        (16, 16, True, False),
        (1, 1, False, False),
    ]:
        c = _case(bs, rbs, seed=bs)
        b = _bind(c)
        b.dp = dp
        host = (rings[0].claim((bs, Q)), rings[1].claim((2, bs)))
        _fused(c, b, with_step, True, with_stage, host=host, uniform=True)
        for slot in host:
            slot.synchronize()
            slot.release()
    torch.cuda.synchronize()
    assert variants() == before


def _runner(c, b, monkeypatch, *, deferred=True, drafter_stage=True, tp=1, pcp=1):
    """`ModelRunner._fused_verify_epilogue` on a stand-in runner over the case."""
    mr = pytest.importorskip("atom.model_engine.model_runner")
    monkeypatch.setattr(mr, "get_tp_group", lambda: NS(world_size=tp))
    monkeypatch.setattr(mr, "get_pcp_world_size", lambda: pcp)
    context = NS(positions=c.positions, scheduled_bs=c.bs, running_bs=c.rbs)
    monkeypatch.setattr(mr, "get_forward_context", lambda: NS(context=context))
    rings = HostHandoffRing(8, MAX_SEQS * Q), HostHandoffRing(8, 2 * MAX_SEQS)

    def targets(ids, positions, hidden):
        assert ids.data_ptr() == b.ids[1:].data_ptr() and ids.numel() == c.T
        stage = None
        if drafter_stage:
            stage = {
                "input_ids": ids,
                "positions": positions[0],
                "hidden_states": hidden,
                "staged": b.staged,
                "scheduled_bs": c.bs,
                "running_bs": c.rbs,
                "query_width": Q,
            }
        return stage, b.dp

    runner = NS(
        rejection_sampler=NS(synthetic_acceptance_rates=None),
        config=NS(
            max_num_seqs=MAX_SEQS,
            compilation_config=NS(cudagraph_capture_sizes=[1, 2, 4, 8, 12, 16, 20]),
        ),
        tokenID_processor=NS(
            is_deferred_out=deferred,
            input_ids=NS(gpu=b.ids),
            claim_verify_handoff=lambda bs, w: (
                rings[0].claim((bs, w)),
                rings[1].claim((2, bs)),
            ),
        ),
        drafter=NS(verify_epilogue_targets=targets),
        _presence_step=b.step,
    )
    runner._verify_epilogue_max_bs = lambda: mr.ModelRunner._verify_epilogue_max_bs(
        runner
    )
    b.meta.num_draft_tokens_np = np.full(c.bs, S, dtype=np.int32)
    batch = NS(
        is_dummy_run=False,
        total_seqs_num=c.bs,
        total_tokens_num=c.T,
        next_token_ids=None,
        return_logprobs=[False] * c.bs,
    )
    return mr, runner, batch


@pytest.mark.parametrize("host", [False, True])
def test_the_runner_epilogue_leaves_what_the_eager_chain_leaves(monkeypatch, host):
    monkeypatch.setenv("ATOM_MTP_VERIFY_HOST_HANDOFF", "1" if host else "0")
    c = _case(5, 8, seed=77)
    a, b = _bind(c), _bind(c)
    eager = _snapshot(a, _eager(c, a, True, True, True))
    mr, runner, batch = _runner(c, b, monkeypatch)
    res = mr.ModelRunner._fused_verify_epilogue(
        runner, batch, b.meta, None, c.target, c.bonus, c.hidden, True
    )
    out, verdict, nxt, slots = res
    _assert_same(eager, _snapshot(b, (out, verdict, nxt)), "runner")
    handoff = runner.drafter.verify_handoff
    assert handoff.staged0 is b.staged and handoff.presence_begun
    assert (slots is not None) == host
    if host:
        for slot in slots:
            slot.synchronize()
        np.testing.assert_array_equal(slots[0].array_view, eager["out"].cpu().numpy())
        np.testing.assert_array_equal(
            slots[1].array_view, eager["verdict"][:2].cpu().numpy()
        )


def test_the_runner_keeps_the_eager_chain_where_the_epilogue_does_not_apply(
    monkeypatch,
):
    c = _case(4, 4, seed=3)
    for why, kwargs, args in [
        ("logits to verify", {}, {"target_logits": torch.zeros(1)}),
        ("no sampled ids", {}, {"target": None}),
        ("unsynced TP tokens", {"tp": 2}, {"synced": False}),
        ("PCP", {"pcp": 2}, {}),
    ]:
        b = _bind(c)
        mr, runner, batch = _runner(c, b, monkeypatch, **kwargs)
        res = mr.ModelRunner._fused_verify_epilogue(
            runner,
            batch,
            b.meta,
            args.get("target_logits"),
            args.get("target", c.target),
            c.bonus,
            c.hidden,
            args.get("synced", True),
        )
        assert res is None, why
    # Not deferred (or the scheduler overrides anchors): verdict and marks only,
    # the drafter stages and resets its penalty itself.
    b = _bind(c)
    mr, runner, batch = _runner(c, b, monkeypatch, deferred=False)
    res = mr.ModelRunner._fused_verify_epilogue(
        runner, batch, b.meta, None, c.target, c.bonus, c.hidden, True
    )
    assert res is not None and res[3] is None
    assert not hasattr(runner.drafter, "verify_handoff")
