# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Speculative decode (DSpark) inside a mixed prefill+decode batch.

A mixed batch is `[prefill rows | decode rows]`. Under spec, a prefill row
verifies as a 0-draft span sampling one logit row, and a decode row as its
usual anchor + drafts; everything downstream of the LM head indexes the batch
by request. See docs/scheduling_kv_cache_guide.md for the design these pin.

Layout used throughout: two prefill rows of 5 and 3 tokens, then two decode
rows of anchor + 3 drafts (mtp_k = 3):

    token rows  [0..4 | 5..7 | 8..11 | 12..15]        16 tokens
    logit rows  [0    | 1    | 2..5  | 6..9  ]        10 rows (LM head gather)
"""

import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module", autouse=True)
def kernel_imports():
    # Keep the real runner, drafter and carrier arithmetic on CPU. The shared
    # import seam replaces only unavailable kernel dependencies and restores
    # the module state afterwards; GPU hosts retain the real AITER modules.
    with atom_without_aiter():
        yield


N_P, N_D, K = 2, 2, 3
PREFILL_LENS = [5, 3]
N_P_TOKENS = sum(PREFILL_LENS)
NUM_SCHEDULED = np.array(PREFILL_LENS + [K + 1] * N_D, dtype=np.int32)


def _cpu_token_processor(monkeypatch, batch, *, previous_rows=0):
    from atom.model_engine import model_runner as module

    def buffer(n):
        host = np.zeros(n, dtype=np.int32)
        return SimpleNamespace(np=host, gpu=torch.from_numpy(host))

    n_p = batch.total_seqs_num_prefill
    lens = np.asarray(batch.num_scheduled_tokens[n_p:], dtype=np.int32)
    cu = np.r_[np.int32(0), np.cumsum(lens, dtype=np.int32)]
    processor = object.__new__(module.tokenIDProcessor)
    processor.runner = SimpleNamespace(
        enforce_eager=True,
        attn_metadata_builder=SimpleNamespace(
            decode_spans=lambda b: (len(lens), lens, cu)
        ),
        forward_vars={"cu_seqlens_q": SimpleNamespace(np=cu, gpu=torch.from_numpy(cu))},
    )
    processor.input_ids = buffer(64)
    processor.decode_src = buffer(16)
    processor.batch_info = buffer(3)
    processor.use_spec = True
    processor.is_deferred_out = True
    processor.is_pipeline_parallel = False
    processor.is_pd_producer = False
    processor.draft_tokens_per_seq = K
    processor.pre_num_decode_token_per_seq = K + 1
    processor.prev_batch = (
        SimpleNamespace(req_ids=list(range(previous_rows))) if previous_rows else None
    )
    processor.prev_token_ids = torch.zeros(previous_rows, dtype=torch.int32)
    processor.draft_token_ids = torch.zeros((previous_rows, K), dtype=torch.int32)
    processor.num_rejected = np.ones(previous_rows, dtype=np.int32)
    processor.num_bonus = np.ones(previous_rows, dtype=np.int32)
    processor.prev_rejected_num = processor.num_rejected.copy()
    processor.prev_bonus_num = processor.num_bonus.copy()
    # This probe checks host status ownership, not the deferred gather kernel.
    monkeypatch.setattr(module, "fill_deferred_decode_ids", lambda *a, **kw: None)
    group = SimpleNamespace(set_counts=lambda counts: None, publish=lambda: None)
    return processor, group


@pytest.mark.parametrize("previous_rows", [0, 1, 9])
def test_cpu_idle_preparation_replaces_previous_rejection_counts(
    monkeypatch, previous_rows
):
    from atom.model_engine.scheduler import ScheduledBatch

    batch = ScheduledBatch({}, [], 0, is_dummy_run=True)
    # Do not drain a device status queue in this host-only ownership probe.
    batch.produces_output = lambda: False
    processor, group = _cpu_token_processor(
        monkeypatch, batch, previous_rows=previous_rows
    )
    if not previous_rows:
        processor.num_rejected = None  # clean()'s initial state
    processor.prepare_input_ids(batch, K + 1, group)
    assert processor.num_rejected is None or processor.num_rejected.size == 0
    assert processor.verify_ctx_lens(np.empty(0, dtype=np.int32)).size == 0


def test_cpu_mixed_preparation_clears_only_prefill_rejections(monkeypatch):
    batch = _mixed_batch(
        total_tokens_num=16,
        total_tokens_num_prefill=8,
        total_tokens_num_decode=8,
        scheduled_tokens=np.arange(16, dtype=np.int32),
        num_rejected=np.array([5, 6, 1, 2], dtype=np.int32),
        num_bonus=np.array([5, 6, 1, 1], dtype=np.int32),
        num_spec_step=K,
        produces_output=lambda: False,
    )
    processor, group = _cpu_token_processor(monkeypatch, batch)
    processor.prepare_input_ids(batch, K + 1, group)
    assert processor.num_rejected.tolist() == [0, 0, 1, 2]
    assert processor.num_bonus.tolist() == [0, 0, 1, 1]


def test_prepare_model_builds_and_passes_mixed_verify_metadata(monkeypatch):
    from atom.model_engine import model_runner as module
    from atom.spec_decode.drafter import Drafter

    batch = _mixed_batch(
        num_spec_step=K,
        is_dummy_run=False,
        next_token_ids=None,
        total_tokens_num=16,
        total_tokens_num_prefill=8,
        total_tokens_num_decode=8,
    )
    mode = SimpleNamespace(
        is_prefill=True,
        running_bs=4,
        scheduled_bs=4,
        running_tokens=16,
        running_tokens_are_unified=True,
        max_seqlen_q=K + 1,
        sync=SimpleNamespace(any_rank_has_prefill=True, ub_tokens_across_dp=None),
        ubatch_slices=None,
        pcp_groups=None,
        decode_ubatch_slices=None,
    )
    monkeypatch.setattr(module.ForwardMode, "decide", lambda **kw: mode)
    captured = {}
    monkeypatch.setattr(module, "set_forward_context", lambda **kw: captured.update(kw))
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(context=captured["context"]),
    )
    runner = _index_runner()
    for var in runner.forward_vars.values():
        var.gpu = torch.from_numpy(var.np)
    runner.forward_vars["batch_info"] = SimpleNamespace(gpu=torch.zeros(3))
    runner.config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_tbo=False,
        enable_tbo_decode=False,
    )
    runner.enforce_eager = True
    runner.capture_sizes_np = np.array([0], dtype=np.int32)
    runner.running_tokens_per_dp_rank = np.array([16], dtype=np.int32)
    runner.tbo_decode_graphs = False
    runner._dspark_apply_q_bucket = lambda b: None
    runner._mono_blocked_step = lambda b: False
    runner._local_tbo_eligibility = lambda b: False
    runner._producer_graph_shapes = lambda: None
    runner._cut_ubatches = lambda m, b: m
    runner.prepare_sample = lambda *a: (torch.ones(4), None, None, True, False)
    for name in ("_verify_spans", "_verifies_drafts", "prepare_inputs"):
        setattr(
            runner, name, types.MethodType(getattr(module.ModelRunner, name), runner)
        )
    runner.tokenID_processor = SimpleNamespace(
        is_pd_producer=False, prepare_input_ids=lambda *a, **kw: torch.arange(16)
    )
    metadata = _mixed_metadata()
    runner.attn_metadata_builder = SimpleNamespace(
        step_needs_eager=lambda b: True,
        prepare_query_metadata=lambda *a: {},
        consumes_device_verdict=lambda *a: False,
        settles_spec_status=lambda *a: False,
        take_spec_status=lambda processor: None,
        build=lambda **kw: (metadata, torch.arange(16)),
        prepare_model_inputs=lambda *a: None,
    )
    group = SimpleNamespace(set_counts=lambda counts: None, check_writable=lambda: None)
    runner.h2d_groups = {"token_inputs": group, "spec_decode": group}
    drafter = SimpleNamespace(mtp_k=K, is_block_drafter=True, runner=runner)
    drafter.prepare_spec_decode_indices = types.MethodType(
        Drafter.prepare_spec_decode_indices, drafter
    )
    drafter.calc_spec_decode_metadata = types.MethodType(
        Drafter.calc_spec_decode_metadata, drafter
    )
    runner.drafter = drafter
    prepare = getattr(
        module.ModelRunner.prepare_model,
        "__wrapped__",
        module.ModelRunner.prepare_model,
    )
    prepare(runner, batch)
    spec = captured["spec_decode_metadata"]
    assert spec is not None
    assert spec.num_draft_tokens_np.tolist() == [0, 0, 3, 3]
    assert spec.target_logits_indices.tolist() == [2, 3, 4, 6, 7, 8]
    assert spec.input_ids.tolist() == [4, 7, 8, 9, 10, 11, 12, 13, 14, 15]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU staging kernel")
@pytest.mark.parametrize("speculative", [False, True])
def test_idle_decode_keeps_padding_and_deferred_status_contract(
    monkeypatch, speculative
):
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )
    from tests.test_h2d_runner_publication import runner_with_buffers

    runner = runner_with_buffers(
        monkeypatch,
        speculative=speculative,
        builder_cls=DeepseekV4AttentionMetadataBuilder,
    )
    runner.enforce_eager = False
    runner.capture_sizes = [4, 2, 1]
    processor = runner.tokenID_processor
    processor.use_spec = speculative
    processor.num_spec_tokens = K if speculative else 0
    processor.prev_batch = SimpleNamespace(req_ids=[10], is_dummy_run=False)
    processor.prev_token_ids = torch.tensor([42], dtype=torch.int32, device="cuda")
    processor.num_rejected = np.array([2], dtype=np.int32)
    processor.num_bonus = np.array([1], dtype=np.int32)
    processor.prev_rejected_num = processor.num_rejected.copy()
    processor.prev_bonus_num = processor.num_bonus.copy()
    processor.pre_num_decode_token_per_seq = 1
    batch = SimpleNamespace(
        is_mixed=False,
        is_dummy_run=True,
        has_prefill=False,
        req_ids=[],
        total_tokens_num=0,
        total_tokens_num_prefill=0,
        total_tokens_num_decode=0,
        total_seqs_num=0,
        total_seqs_num_prefill=0,
        total_seqs_num_decode=0,
        num_spec_step=0,
        scheduled_tokens=np.array([], dtype=np.int32),
        num_scheduled_tokens=np.array([], dtype=np.int32),
        num_rejected=np.array([], dtype=np.int32),
        num_bonus=np.array([], dtype=np.int32),
        produces_output=lambda: False,
    )
    runner._gate_staging_reuse()
    processor.input_ids.gpu.fill_(77)
    group = runner.h2d_groups["token_inputs"]
    group.set_counts(
        runner.attn_metadata_builder.prepare_query_metadata(
            batch.num_scheduled_tokens, 0, 0
        )
    )
    q = K + 1 if speculative else 1
    ids = processor.prepare_input_ids(batch, q, group)
    runner._mark_staging_h2d_enqueued()
    assert ids.numel() == 0
    assert processor.input_ids.gpu[:q].cpu().tolist() == [0] * q
    assert processor.input_ids.gpu[q : q + 1].cpu().item() == 77
    assert processor.carries_over
    assert processor.num_rejected.size == processor.num_bonus.size == 0


@pytest.mark.parametrize("prefill_rows", [0, 1, 4])
@pytest.mark.parametrize("draft_width", [3, 7])
@pytest.mark.parametrize("with_draft", [False, True])
def test_prefill_confidence_cannot_change_decode_budget_or_request_mapping(
    monkeypatch, prefill_rows, draft_width, with_draft
):
    from test_dspark_scheduler import _stepping_scheduler

    from atom.model_engine import model_runner as mr
    from atom.spec_decode import dspark_proposer as proposer

    DSparkProposer = proposer.DSparkProposer

    verify = _stepping_scheduler(monkeypatch)
    decode_confidence = torch.full((4, draft_width), 0.8)
    expected = verify.compute_ell(decode_confidence)
    confidence = torch.cat(
        (
            torch.full((prefill_rows, draft_width), 0.999),
            decode_confidence,
            torch.ones(2, draft_width),
        ),  # graph padding must also be excluded
    )
    scheduled_bs = prefill_rows + 4
    metadata = SimpleNamespace(
        is_mixed=bool(prefill_rows), num_prefill_seqs=prefill_rows, max_seqlen_k=64
    )
    fc = SimpleNamespace(
        context=SimpleNamespace(
            positions=torch.arange(scheduled_bs), running_bs=scheduled_bs + 2
        ),
        attn_metadata=metadata,
    )
    monkeypatch.setattr(mr, "get_forward_context", lambda: fc)
    monkeypatch.setattr(proposer, "get_forward_context", lambda: fc)
    monkeypatch.setattr(mr, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    drafter = DSparkProposer.__new__(DSparkProposer)
    drafter._verify_scheduler = verify
    drafter.mtp_k = draft_width
    drafter._captures_aux = True
    drafter.speculative_config = SimpleNamespace(
        use_dspark_with_draft=lambda: with_draft
    )
    drafter.model = SimpleNamespace(prepare_block=lambda *args: None)
    drafter.runner = SimpleNamespace(
        attn_metadata_builder=None,
        forward_vars={"batch_info": SimpleNamespace(gpu=torch.zeros(3))},
    )
    drafter._block_pass = lambda: SimpleNamespace(
        run=lambda *a, **kw: (torch.zeros(scheduled_bs + 2, draft_width), confidence),
        label=lambda *args: "test",
    )
    drafter._stage_anchors = lambda *args: {"anchor_positions": fc.context.positions}
    drafter._publish_draft_shape = lambda *args, **kwargs: None
    drafter._build_paged_block_metadata = lambda *args, **kwargs: None
    # Exercise both production proposal paths through the runner's recording
    # path; only the GPU block execution and its input staging are replaced.
    drafter.propose = drafter._propose
    ids = [f"prefill-{i}" for i in range(prefill_rows)] + ["A", "B", "C", "D"]
    batch = SimpleNamespace(
        is_mixed=bool(prefill_rows),
        total_seqs_num_prefill=prefill_rows,
        total_seqs_num=scheduled_bs,
        req_ids=ids,
        next_token_ids=None,
    )
    published = []
    runner = SimpleNamespace(
        drafter=drafter,
        _commit_target_step=lambda *a: torch.arange(scheduled_bs),
        attn_metadata_builder=SimpleNamespace(publish_draft_state=published.append),
        tokenID_processor=SimpleNamespace(
            prepare_draft_ids=lambda batch, drafts: drafts,
            await_draft_ids_copy=lambda stream: None,
        ),
    )
    result = mr.ModelRunner.propose_draft_token_ids(
        runner, batch, None, None, torch.zeros(scheduled_bs), None, None
    )
    torch.testing.assert_close(verify._last_ell[-4:], expected, rtol=0, atol=0)
    torch.testing.assert_close(verify._last_ell, expected, rtol=0, atol=0)
    assert verify._ell_pending[-1][2] == ["A", "B", "C", "D"]
    assert result.shape == (scheduled_bs, draft_width)
    assert published == ([metadata] if prefill_rows else [])


@pytest.mark.parametrize("scheduler_enabled", [False, True])
@pytest.mark.parametrize("has_confidence", [False, True])
def test_separate_draft_handles_optional_confidence(scheduler_enabled, has_confidence):
    from atom.spec_decode.dspark_proposer import DSparkProposer

    lengths = []
    confidence = torch.arange(15).reshape(3, 5) if has_confidence else None
    verify = SimpleNamespace(
        compute_ell=lambda rows: rows.sum(dim=1), set_last_ell=lengths.append
    )
    drafter = DSparkProposer.__new__(DSparkProposer)
    drafter._verify_scheduler = verify if scheduler_enabled else None
    drafter.mtp_k = 3
    drafter.model = SimpleNamespace()
    drafter.runner = SimpleNamespace(
        forward_vars={"batch_info": SimpleNamespace(gpu=torch.zeros(3))}
    )
    drafter._publish_draft_shape = lambda *args, **kwargs: None
    drafter._build_paged_block_metadata = lambda *args, **kwargs: None
    metadata = SimpleNamespace(is_mixed=False, max_seqlen_k=64)
    fc = SimpleNamespace(context=SimpleNamespace(running_bs=3, is_prefill=False))
    ids = torch.arange(15).reshape(3, 5)
    block = SimpleNamespace(
        run=lambda *args, **kwargs: (ids, confidence), label=lambda *args: "test"
    )
    result = drafter._propose_with_draft(
        block, fc, metadata, {"anchor_positions": torch.zeros(3)}, scheduled_bs=2
    )
    torch.testing.assert_close(result, ids[:2, :3])
    if not scheduler_enabled:
        assert lengths == []
    elif not has_confidence:
        assert lengths == [None]
    else:
        assert len(lengths) == 1
        torch.testing.assert_close(lengths[0], confidence[:2, :3].sum(dim=1))


def _mixed_batch(**overrides):
    fields = {
        "is_mixed": True,
        "is_dummy_run": False,
        "total_seqs_num_prefill": N_P,
        "total_seqs_num_decode": N_D,
        "total_seqs_num": N_P + N_D,
        "total_tokens_num_prefill": N_P_TOKENS,
        "num_scheduled_tokens": NUM_SCHEDULED.copy(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


# ── verify layout in logit-row space ────────────────────────────────────────


def _spec_runner(*, drafter=True, pd_producer=False):
    """A runner with just what `_verify_spans` asks of it, `_verifies_drafts`."""
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(is_pd_producer=pd_producer)
    )
    if drafter:
        runner.drafter = object()
    runner._verifies_drafts = types.MethodType(ModelRunner._verifies_drafts, runner)
    return runner


def test_verify_spans_put_prefill_rows_at_one_sample_each():
    from atom.model_engine.model_runner import ModelRunner

    lens, cu_end = ModelRunner._verify_spans(_spec_runner(), _mixed_batch())
    assert lens.tolist() == [1, 1, 4, 4]
    assert cu_end.tolist() == [1, 2, 6, 10]  # ends at the 10 logit rows


@pytest.mark.parametrize(
    "phase,verify",
    [
        ("prefill", False),
        ("decode", False),
        ("decode", True),
        ("mixed", False),
        ("mixed", True),
    ],
)
def test_prepare_inputs_uses_the_prepared_spec_indices(monkeypatch, phase, verify):
    from atom.model_engine import model_runner as module

    runner = object.__new__(module.ModelRunner)
    runner.config = SimpleNamespace(speculative_config=object() if verify else None)
    runner.forward_vars = {"batch_info": SimpleNamespace(gpu=torch.zeros(3))}
    runner.running_tokens_per_dp_rank = None
    ids = torch.arange(6)
    metadata = SimpleNamespace(sampled_token_indices=torch.tensor([2, 3, 4, 5]))
    runner.attn_metadata_builder = SimpleNamespace(
        build=lambda **kw: (metadata, ids),
        prepare_model_inputs=lambda *args: None,
    )
    calls = []
    spec_metadata = object()

    def calc_spec_decode_metadata(sampled_ids, indices):
        calls.append((sampled_ids, indices))
        return spec_metadata

    if verify:
        runner.drafter = SimpleNamespace(
            calc_spec_decode_metadata=calc_spec_decode_metadata
        )
    captured = {}
    monkeypatch.setattr(module, "set_forward_context", lambda **kw: captured.update(kw))
    batch = SimpleNamespace(
        is_mixed=phase == "mixed", total_tokens_num=6, is_dummy_run=phase == "prefill"
    )
    mode = SimpleNamespace(
        is_prefill=phase != "decode",
        sync=SimpleNamespace(ub_tokens_across_dp=None),
        running_tokens_are_unified=True,
        running_bs=2,
        running_tokens=6,
        max_seqlen_q=3,
        ubatch_slices=None,
        pcp_groups=None,
        scheduled_bs=2,
    )
    indices = (np.array([0, 2]), 2) if verify else None
    runner.prepare_inputs(batch, ids, mode, spec_decode_indices=indices)

    assert captured["attn_metadata"] is metadata
    assert captured["context"].is_prefill == mode.is_prefill
    assert captured["spec_decode_metadata"] is (spec_metadata if verify else None)
    if verify:
        assert len(calls) == 1 and calls[0][1] is indices
        expected = ids[metadata.sampled_token_indices] if phase == "mixed" else ids
        torch.testing.assert_close(calls[0][0], expected)
    else:
        assert not calls


def test_verify_spans_skip_pure_prefill_and_idle_ranks():
    from atom.model_engine.model_runner import ModelRunner

    runner = _spec_runner()
    assert ModelRunner._verify_spans(runner, _mixed_batch(is_mixed=False)) is None
    assert (
        ModelRunner._verify_spans(
            runner,
            _mixed_batch(
                is_mixed=False,
                is_dummy_run=True,
                total_seqs_num_prefill=0,
                total_seqs_num_decode=0,
                total_seqs_num=0,
            ),
        )
        is None
    )
    assert (
        ModelRunner._verify_spans(_spec_runner(drafter=False), _mixed_batch()) is None
    )


def test_verify_spans_skip_a_pd_producer_mixed_step():
    """A P/D producer keeps its drafter only to write context KV and never
    pads drafts, so a mixed step of its verifies nothing, as its decodes don't."""
    from atom.model_engine.model_runner import ModelRunner

    runner = _spec_runner(pd_producer=True)
    assert ModelRunner._verify_spans(runner, _mixed_batch()) is None


def _index_runner(cap=64):
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(
        arange_np=np.arange(cap, dtype=np.int32),
        forward_vars={
            name: SimpleNamespace(np=np.full(cap, -1, dtype=np.int32))
            for name in (
                "target_logits_indices",
                "cu_num_draft_tokens",
                "bonus_logits_indices",
            )
        },
    )
    runner._get_cumsum_and_arange = types.MethodType(
        ModelRunner._get_cumsum_and_arange, runner
    )
    return runner


def test_spec_indices_and_draft_gather_over_the_mixed_layout():
    """The real index builder, fed the mixed spans, then the draft-id gather."""
    from atom.model_engine.model_runner import ModelRunner
    from atom.spec_decode.drafter import Drafter

    lens, cu_end = ModelRunner._verify_spans(_spec_runner(), _mixed_batch())
    runner = _index_runner()
    drafter = SimpleNamespace(mtp_k=K, runner=runner)
    names = ["target_logits_indices", "cu_num_draft_tokens", "bonus_logits_indices"]
    group = SimpleNamespace(
        counts=[None] * 3, indices={n: i for i, n in enumerate(names)}
    )
    group.set_counts = lambda counts: [
        group.counts.__setitem__(group.indices[n], v) for n, v in counts.items()
    ]
    num_draft, total = Drafter.prepare_spec_decode_indices(drafter, lens, cu_end, group)
    var = runner.forward_vars
    assert num_draft.tolist() == [0, 0, 3, 3]
    assert total == N_D * K
    # Each decode row's first three logit rows are scored; prefill rows none.
    target = var["target_logits_indices"].np[:total]
    assert target.tolist() == [2, 3, 4, 6, 7, 8]
    # One bonus per request: a prefill row's single row, a decode row's last.
    assert var["bonus_logits_indices"].np[: N_P + N_D].tolist() == [0, 1, 5, 9]
    assert var["cu_num_draft_tokens"].np[: N_P + N_D].tolist() == [0, 0, 3, 6]

    # The same carrier mapping selects LM-head rows, draft IDs and verdicts.
    input_ids = torch.arange(N_P_TOKENS + N_D * (K + 1))
    rows = _mixed_metadata().sampled_token_indices
    drafts = input_ids[rows][1:][torch.from_numpy(target).long()]
    assert drafts.tolist() == [9, 10, 11, 13, 14, 15]


# ── verify outcome: the verdict, re-expressed in token rows ─────────────────


def _mixed_metadata():
    from atom.model_ops.attentions.deepseek_v4_attn import (
        AttentionMetaData_DSV4,
        _mixed_carrier,
    )
    from atom.utils.forward_context import AttnState

    prefill = AttentionMetaData_DSV4(
        cu_seqlens_q=torch.tensor([0, 5, 8], dtype=torch.int32),
        state_slot_out=torch.tensor([0, 1]),
        state=AttnState.PREFILL_PREFIX,
    )
    decode = AttentionMetaData_DSV4(
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        state_slot_out=torch.tensor([2, 3]),
        state=AttnState.DECODE,
    )
    return _mixed_carrier(prefill, decode, 2, 2, 8, 8, 4)


@pytest.fixture
def mixed_forward_context():
    from atom.utils import forward_context as fc_mod

    ctx = SimpleNamespace(attn_metadata=_mixed_metadata())
    prev = getattr(fc_mod._forward_context_local, "ctx", None)
    fc_mod._forward_context_local.ctx = ctx
    try:
        yield ctx
    finally:
        fc_mod._forward_context_local.ctx = prev


def _logit_verdict(accepted):
    """What the rejection kernel writes, in LOGIT rows: rejected is
    `K - accepted`, anchor is `bonus_row - num_draft + accepted` (see
    `_finish_request`). Bonus rows and draft counts are the mixed layout's."""
    bonus_rows = [0, 1, 5, 9]
    num_draft = [0, 0, K, K]
    reject = torch.tensor([K - a for a in accepted], dtype=torch.int32)
    anchors = torch.tensor(
        [b - d + a for b, d, a in zip(bonus_rows, num_draft, accepted)],
        dtype=torch.int32,
    )
    return reject, anchors


def test_mixed_verdict_moves_to_token_rows(mixed_forward_context):
    """Prefill rows reject nothing and anchor on their chunk's last token;
    decode rows shift by `n_p_tokens - n_p`. Left at `mtp_k`, a prefill row's
    reject count rolls the request's ctx back K the moment it decodes."""
    from atom.model_engine.model_runner import ModelRunner

    reject, anchors = _logit_verdict([0, 0, 1, 3])
    ModelRunner._settle_mixed_verdict(None, _mixed_batch(), reject, anchors)
    assert reject.tolist() == [0, 0, 2, 0]
    # prefill: last token of each chunk; decode: segment start + accepted.
    assert anchors.tolist() == [4, 7, 9, 15]


def test_unverified_step_anchors_every_row_on_its_last(mixed_forward_context):
    """With no verdict the drafter anchors on each span's last row, which over
    the carrier's whole-batch spans is right for prefill and decode rows alike."""
    from atom.spec_decode.drafter import Drafter

    anchors = Drafter.prepare_inputs(None, N_P + N_D)
    assert anchors.tolist() == [4, 7, 11, 15]


# ── DSpark length shrink preserves the prefill region ───────────────────────


@pytest.mark.parametrize("previous_prefills", [[1, 2], [90]])
def test_mixed_ragged_shrink_preserves_prefill_even_under_full_cudagraphs(
    previous_prefills,
):
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(
        drafter=SimpleNamespace(
            uses_confidence_schedule=True,
            mtp_k=K,
            verify_scheduler=SimpleNamespace(ell_by_req={3: 0, 4: 2}),
        ),
        config=SimpleNamespace(
            dspark=SimpleNamespace(ragged=True, ragged_graph_sizes="4"),
            enable_mixed_prefill_decode=True,
        ),
        tokenID_processor=SimpleNamespace(
            prev_batch=SimpleNamespace(
                req_ids=previous_prefills + [3, 4],
                total_seqs_num_prefill=len(previous_prefills),
                is_dummy_run=False,
            )
        ),
        enforce_eager=False,
        _tr_hybrid_active=lambda: False,
    )
    runner._dspark_apply_ragged = types.MethodType(
        ModelRunner._dspark_apply_ragged, runner
    )
    batch = _mixed_batch(
        req_ids=[1, 2, 3, 4],
        num_spec_step=K,
        num_bonus=np.array([99, 99, 0, 1]),
        scheduled_tokens=np.arange(16),
        total_tokens_num_decode=8,
        total_tokens_num=16,
    )
    assert ModelRunner._dspark_apply_q_bucket(runner, batch) == 3
    assert batch.num_scheduled_tokens.tolist() == [5, 3, 1, 3]
    assert batch.scheduled_tokens.tolist() == list(range(9)) + [12, 13, 14]
    assert batch.total_tokens_num == 12
    assert batch.total_tokens_num_prefill == 8


# ── V4 builder pieces ────────────────────────────────────────────────────────


def test_carrier_spans_cover_the_whole_batch():
    from atom.model_ops.attentions.deepseek_v4_attn import _mixed_carrier_spans

    prefill_meta = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 5, 8], dtype=torch.int32),
        state_slot_out=torch.tensor([11, 12], dtype=torch.int32),
    )
    decode_meta = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        state_slot_out=torch.tensor([21, 22], dtype=torch.int32),
    )
    running_bs = 6  # a DP ladder rung wider than this rank's 4 requests
    cu, slots = _mixed_carrier_spans(
        prefill_meta, decode_meta, N_P, N_D, N_P_TOKENS, running_bs
    )
    assert cu.tolist() == [0, 5, 8, 12, 16, 16, 16]
    assert cu[: N_P + 1].tolist() == prefill_meta.cu_seqlens_q.tolist()
    assert slots.tolist() == [11, 12, 21, 22, 0, 0]


# ── token inputs: the mixed branch of prepare_input_ids, end to end ─────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stages through the GPU")
@pytest.mark.parametrize("previous_dummy", [False, True])
def test_mixed_deferred_inputs_under_spec(monkeypatch, previous_dummy):
    """Every kind of row a mixed spec step can hold, in one batch:

    - 40: a prefill middle chunk that was ALSO in the previous batch -- must not
      be taken for a carried-over decode (why the map runs over decode rows).
    - 10: a decode row carried over: anchor + drafts from last step's device
      buffers, and its rejected/bonus remapped from last step's status.
    - 30: last step's PREFILL row, decoding for the first time: its anchor and
      drafts come from the propose on that prefill row.
    - 50: a decode row not in the previous batch: scheduler-staged drafts.
    """
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )
    from tests.test_h2d_runner_publication import runner_with_buffers

    runner = runner_with_buffers(
        monkeypatch, speculative=True, builder_cls=DeepseekV4AttentionMetadataBuilder
    )
    processor = runner.tokenID_processor
    processor.use_spec = True
    processor.num_spec_tokens = K
    runner._gate_staging_reuse()

    processor.prev_batch = SimpleNamespace(
        req_ids=[] if previous_dummy else [10, 20, 30, 40], is_dummy_run=previous_dummy
    )
    processor.prev_token_ids = torch.tensor(
        [100, 200, 300, 400], dtype=torch.int32, device="cuda"
    )
    processor.draft_token_ids = torch.tensor(
        [[r + 1, r + 2, r + 3] for r in (100, 200, 300, 400)],
        dtype=torch.int32,
        device="cuda",
    )
    processor.pre_num_decode_token_per_seq = K + 1
    processor.prev_rejected_num = np.array([1, 0, 2, 0], dtype=np.int32)
    processor.prev_bonus_num = np.array([2, 3, 1, 3], dtype=np.int32)

    spec = np.zeros((4, K), dtype=np.int32)
    spec[3] = [501, 502, 503]
    batch = SimpleNamespace(
        num_spec_step=K,
        has_prefill=True,
        is_mixed=True,
        is_dummy_run=False,
        req_ids=[40, 10, 30, 50],
        scheduled_tokens=np.array(
            [7, 8, 9] + [-1] * 4 + [-1] * 4 + [500, -1, -1, -1], dtype=np.int32
        ),
        total_tokens_num=15,
        total_tokens_num_prefill=3,
        total_tokens_num_decode=12,
        total_seqs_num_prefill=1,
        total_seqs_num_decode=3,
        total_seqs_num=4,
        num_scheduled_tokens=np.array([3, 4, 4, 4], dtype=np.int32),
        scheduled_spec_decode_tokens=spec,
        num_rejected=np.array([5, 0, 0, 0], dtype=np.int32),
        num_bonus=np.array([5, 0, 0, 0], dtype=np.int32),
        # False: keep the status set above rather than drain the (empty) queue.
        produces_output=lambda: False,
    )
    group = runner.h2d_groups["token_inputs"]
    group.set_counts(
        runner.attn_metadata_builder.prepare_query_metadata(
            batch.num_scheduled_tokens[1:], 3, 12
        )
    )
    # Decode-local spans, from 0, anchor + 3 drafts each.
    assert runner.forward_vars["cu_seqlens_q"].np[:4].tolist() == [0, 4, 8, 12]

    ids = processor.prepare_input_ids(batch, K + 1, group).clone()
    runner._mark_staging_h2d_enqueued()
    runner.h2d_owner.completion.synchronize()
    expected = (
        [7, 8, 9]  # prefill middle chunk, as scheduled
        + [100, 101, 102, 103]  # 10: carried-over decode
        + [300, 301, 302, 303]  # 30: first decode after its prefill row
        + [500, 501, 502, 503]  # 50: scheduler-staged
    )
    if previous_dummy:
        expected = [7, 8, 9] + [-1, 0, 0, 0] * 2 + [500, 501, 502, 503]
    assert ids.cpu().tolist() == expected
    # The whole batch's counts reach the device: MoE routes rows past
    # `scheduled_tokens_gpu` (batch_info[1]) to no expert.
    assert runner.forward_vars["batch_info"].gpu[:2].tolist() == [4, 15]
    # Prefill row: nothing rejected. Carried-over rows: last step's status.
    assert processor.num_rejected.tolist() == (
        [0] * 4 if previous_dummy else [0, 1, 2, 0]
    )
    assert processor.num_bonus.tolist() == ([0] * 4 if previous_dummy else [0, 2, 1, 0])


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("previous_dummy", [False, True])
@pytest.mark.parametrize("lengths", [[4, 4, 4], [1, 3, 4]])
def test_cpu_deferred_input_wiring(monkeypatch, mixed, previous_dummy, lengths):
    from atom.model_engine import model_runner as module

    prefix = [7, 8, 9] if mixed else []
    n_p = int(mixed)
    scheduled = prefix + [-1] * sum(lengths[:-1]) + [500, -1, -1, -1]
    spec = np.zeros((n_p + 3, K), dtype=np.int32)
    spec[-1] = [501, 502, 503]
    batch = SimpleNamespace(
        num_spec_step=K,
        is_mixed=mixed,
        is_dummy_run=False,
        req_ids=([40] if mixed else []) + [10, 30, 50],
        scheduled_tokens=np.array(scheduled, dtype=np.int32),
        total_tokens_num=len(scheduled),
        total_tokens_num_prefill=len(prefix),
        total_tokens_num_decode=sum(lengths),
        total_seqs_num_prefill=n_p,
        total_seqs_num_decode=3,
        total_seqs_num=n_p + 3,
        num_scheduled_tokens=np.array(([3] if mixed else []) + lengths),
        scheduled_spec_decode_tokens=spec,
        num_rejected=np.zeros(n_p + 3, dtype=np.int32),
        num_bonus=np.zeros(n_p + 3, dtype=np.int32),
        produces_output=lambda: False,
    )
    processor, group = _cpu_token_processor(monkeypatch, batch, previous_rows=4)
    processor.prev_batch.req_ids = [] if previous_dummy else [10, 20, 30, 40]
    processor.prev_token_ids = torch.tensor([100, 200, 300, 400], dtype=torch.int32)
    processor.draft_token_ids = torch.tensor(
        [[v + 1, v + 2, v + 3] for v in [100, 200, 300, 400]]
    )
    processor.prev_rejected_num = np.array([1, 0, 2, 0], dtype=np.int32)
    processor.prev_bonus_num = np.array([2, 3, 1, 3], dtype=np.int32)
    processor.runner.enforce_eager = False
    processor.runner.capture_sizes = [8, 4, 2, 1]

    def gather(dest, cu, sources, anchors, drafts, *, max_tokens_per_seq, width):
        bs, fill_to = len(sources), width
        assert max_tokens_per_seq == max(lengths)
        assert bs == 3 and sources[:bs].tolist() == (
            [-1] * 3 if previous_dummy else [0, 2, -1]
        )
        assert dest.data_ptr() == processor.input_ids.gpu[len(prefix) :].data_ptr()
        assert fill_to == (sum(lengths) if mixed else 16)
        assert dest[sum(lengths[:-1]) : sum(lengths)].tolist() == [500, 501, 502, 503]
        for row, source in enumerate(sources[:bs].tolist()):
            if source >= 0:
                values = torch.cat((anchors[source : source + 1], drafts[source]))
                dest[cu[row] : cu[row + 1]] = values[: cu[row + 1] - cu[row]]
        dest[sum(lengths) : fill_to] = 0

    monkeypatch.setattr(module, "fill_deferred_decode_ids", gather)
    actual = processor.prepare_input_ids(batch, K + 1, group).tolist()
    first = (
        [-1] + [0] * (lengths[0] - 1)
        if previous_dummy
        else [100, 101, 102, 103][: lengths[0]]
    )
    second = (
        [-1] + [0] * (lengths[1] - 1)
        if previous_dummy
        else [300, 301, 302, 303][: lengths[1]]
    )
    assert actual == prefix + first + second + [500, 501, 502, 503]
    assert processor.num_rejected.tolist() == [0] * n_p + (
        [0, 0, 0] if previous_dummy else [1, 2, 0]
    )
    assert processor.num_bonus.tolist() == [0] * n_p + (
        [0, 0, 0] if previous_dummy else [2, 1, 0]
    )
