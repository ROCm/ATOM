# SPDX-License-Identifier: MIT
"""What `postprocess` hands the sampler, in rows.

The LM head emits one row per sequence the step FORWARDED. On a prefill that is
`running_bs` -- the ladder-rounded, DP-agreed batch -- while the sampler's
per-request parameters are `scheduled_bs` wide, because they describe requests.
Every step between the two has to happen on the same side of the cut.

Skipped where `model_engine.model_runner` cannot import (it needs aiter at
module load); nothing here touches a GPU.
"""

import types

import numpy as np
import pytest
import torch

# Not `importorskip("aiter")`: the non-GPU runner has `aiter` as a namespace
# package, so that import succeeds and `model_runner` still fails on a symbol.
# Naming the module we actually need catches both, and `exc_type` is what makes
# a non-ModuleNotFoundError ImportError a skip rather than a collection error --
# one of which takes the whole suite down with it. See tests/import_guard.py.
mod = pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="model_runner imports aiter at module load",
    exc_type=ImportError,
)
v4_mod = pytest.importorskip(
    "atom.model_ops.attentions.deepseek_v4_attn",
    reason="DeepSeek-V4 metadata builder imports aiter at module load",
    exc_type=ImportError,
)


def _batch(seqs):
    return types.SimpleNamespace(
        total_seqs_num=seqs,
        total_tokens_num=seqs,
        req_ids=list(range(seqs)),
        return_logprobs=[False] * seqs,
        is_dummy_run=False,
    )


def _runner(seen):
    """A ModelRunner carrying only what the no-spec branch of postprocess reads."""
    runner = object.__new__(mod.ModelRunner)

    def _sampler(logits, temperatures, top_ks, top_ps, all_greedy, **kw):
        seen["logits_rows"] = logits.shape[0]
        seen["temperature_rows"] = temperatures.shape[0]
        return torch.zeros(logits.shape[0], dtype=torch.int)

    runner.sampler = _sampler
    runner.forward_done_event = types.SimpleNamespace(record=lambda: None)
    runner.tokenID_processor = types.SimpleNamespace(
        is_deferred_out=False,
        prev_batch=None,
        default_num_rejected_tokens=torch.zeros(64, dtype=torch.int32),
        prepare_sampled_ids=lambda *a, **k: ({}, {}),
    )
    return runner


def test_the_sampler_gets_as_many_rows_as_it_has_parameters(monkeypatch):
    """A prefill forwards more sequences than it scheduled, and only one of the
    two numbers describes a request.

    `decide` rounds every step onto the capture ladder, so a 3-sequence prefill
    runs 4 rows, and `prepare_prefill` pads `cu_seqlens_q` to match -- which is
    what the LM head slices by. `prepare_sample` sizes `temperatures` at the
    scheduled count and never pads. Sampling before the cut divides
    `[4, V]` by `[3, 1]`, which raises on the top-k/top-p path and reads past
    the parameter buffer on the temperature one.
    """
    seen = {}
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )

    scheduled_bs, running_bs, vocab = 3, 4, 8
    out = _runner(seen).postprocess(
        batch=_batch(scheduled_bs),
        logits=torch.zeros(running_bs, vocab),
        temperatures=torch.ones(scheduled_bs),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    assert seen["logits_rows"] == seen["temperature_rows"] == scheduled_bs
    assert out.num_rejected.tolist() == [0] * scheduled_bs


def test_a_step_that_padded_nothing_is_unaffected(monkeypatch):
    """The cut is a slice, so the common case has to stay a no-op."""
    seen = {}
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )

    bs = 5
    _runner(seen).postprocess(
        batch=_batch(bs),
        logits=torch.zeros(bs, 8),
        temperatures=torch.ones(bs),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    assert seen["logits_rows"] == bs


def test_the_logprob_gather_reads_the_cut_logits(monkeypatch):
    """`log_probs.gather` indexes with the sampled ids, which are per-request.

    Left on the padded logits it is `[running_bs, V]` gathered by `[scheduled_bs,
    1]` -- a second break from the same uncut tensor, and the one a client asking
    for logprobs hits.
    """
    seen = {}
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )

    scheduled_bs, running_bs = 3, 4
    batch = _batch(scheduled_bs)
    batch.return_logprobs = [True] * scheduled_bs
    captured = {}
    runner = _runner(seen)
    runner.tokenID_processor.prepare_sampled_ids = lambda b, ids, ev, lp: (
        captured.update(logprobs=lp) or ({}, {})
    )

    runner.postprocess(
        batch=batch,
        logits=torch.zeros(running_bs, 8),
        temperatures=torch.ones(scheduled_bs),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    assert captured["logprobs"].shape == (scheduled_bs,)


def test_prefill_really_does_run_wider_than_it_scheduled():
    """The premise, from the decider itself -- not a number invented here."""
    mode = mod.ForwardMode.decide(
        batch=types.SimpleNamespace(
            total_tokens_num_prefill=9,
            total_tokens_num=9,
            total_seqs_num=3,
            is_dummy_run=False,
        ),
        dp_size=1,
        dp_group=None,
        enforce_eager=False,
        capture_sizes=np.array([1, 2, 4, 8], dtype=np.int32),
        captured_tokens=None,
        is_block_drafter=False,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=1,
    )

    assert (mode.scheduled_bs, mode.running_bs) == (3, 4)


def test_deferred_output_is_published_before_current_draft_is_enqueued(monkeypatch):
    """Let the scheduler work while this step's MTP proposal is still running.

    Deferred output exposes only generation N-1.  Its sampled IDs, draft IDs,
    and accept/reject status are therefore immutable before generation N's
    proposal starts and can be sent to EngineCore at that boundary.
    """

    order = []
    published = []
    batch = _batch(1)
    batch.total_tokens_num = 1

    class _TokenProcessor:
        is_deferred_out = True
        prev_batch = types.SimpleNamespace(total_seqs_num=1)
        prev_rejected_num = np.array([2], dtype=np.int32)
        prev_bonus_num = np.array([1], dtype=np.int32)
        default_num_rejected_tokens = torch.zeros(1, dtype=torch.int32)
        input_ids = types.SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32))

        def prepare_sampled_ids(self, *_args, **_kwargs):
            self.prev_req_ids = [17]
            return ({17: (41,), -1: 1}, None)

        def take_draft_output(self):
            order.append("take previous draft")
            return np.array([[51, 52, 53]], dtype=np.int32)

        def send_mtp_status_to_cpu_async(self, *_args):
            order.append("stage current status")

    runner = object.__new__(mod.ModelRunner)
    runner.sampler = lambda *_args, **_kwargs: torch.tensor([7], dtype=torch.int32)
    runner.forward_done_event = types.SimpleNamespace(record=lambda: None)
    runner.tokenID_processor = _TokenProcessor()
    runner.drafter = types.SimpleNamespace(mtp_k=3, verify_scheduler=None)
    runner._forward_output_sink = lambda output: (
        order.append("publish previous output"),
        published.append(output),
    )
    runner._dp_metadata_device_sync = True
    runner._forward_output_published_early = False

    def _propose(*_args, **kwargs):
        assert kwargs["stage_deferred_output"] is True
        order.append("enqueue current draft")

    runner.propose_draft_token_ids = _propose

    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )

    result = runner.postprocess(
        batch=batch,
        logits=torch.zeros(1, 8),
        temperatures=torch.ones(1),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    assert order.index("publish previous output") < order.index("enqueue current draft")
    assert published == [result]
    assert result.req_ids == [17]
    assert result.token_ids == [(41,)]
    np.testing.assert_array_equal(result.draft_token_ids, [[51, 52, 53]])
    assert runner._forward_output_published_early is True


def test_dummy_completion_is_published_before_its_draft_is_enqueued(monkeypatch):
    """A dummy releases EngineCore early without publishing model output.

    EngineCore can use that completion to run the next Gloo state sync and
    queue the next worker RPC. The worker remains FIFO, so the next metadata
    H2D/RCCL submission starts only after this dummy's draft has been enqueued.
    """

    order = []
    published = []
    batch = _batch(1)
    batch.is_dummy_run = True

    class _TokenProcessor:
        is_deferred_out = True
        prev_batch = types.SimpleNamespace(total_seqs_num=1)
        prev_rejected_num = np.array([2], dtype=np.int32)
        prev_bonus_num = np.array([1], dtype=np.int32)
        default_num_rejected_tokens = torch.zeros(1, dtype=torch.int32)
        input_ids = types.SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32))

        def prepare_sampled_ids(self, *_args, **_kwargs):
            return ({17: (41,), -1: 1}, None)

        def take_draft_output(self):
            order.append("consume previous draft")
            return np.array([[51, 52, 53]], dtype=np.int32)

        def send_mtp_status_to_cpu_async(self, *_args):
            order.append("stage current status")

    runner = object.__new__(mod.ModelRunner)
    runner.sampler = lambda *_args, **_kwargs: torch.tensor([7], dtype=torch.int32)
    runner.forward_done_event = types.SimpleNamespace(record=lambda: None)
    runner.tokenID_processor = _TokenProcessor()
    runner.drafter = types.SimpleNamespace(mtp_k=3, verify_scheduler=None)
    runner._forward_output_sink = lambda output: (
        order.append("publish dummy completion"),
        published.append(output),
    )
    runner._dp_metadata_device_sync = True
    runner._forward_output_published_early = False

    def _propose(*_args, **kwargs):
        assert kwargs["stage_deferred_output"] is True
        order.append("enqueue dummy draft")

    runner.propose_draft_token_ids = _propose

    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )

    result = runner.postprocess(
        batch=batch,
        logits=torch.zeros(1, 8),
        temperatures=torch.ones(1),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    assert order.index("consume previous draft") < order.index(
        "publish dummy completion"
    )
    assert order.index("publish dummy completion") < order.index("enqueue dummy draft")
    assert published == [True]
    assert result.req_ids == [17]
    assert runner._forward_output_published_early is True


def test_dummy_execution_falls_back_to_normal_completion_without_early_output():
    runner = object.__new__(mod.ModelRunner)
    runner.label = "test runner"
    runner.block_size = 1
    runner.drafter = types.SimpleNamespace(mtp_k=3)

    def _forward(batch):
        assert batch.is_dummy_run
        runner._forward_output_published_early = False

    runner.forward = _forward

    assert runner.dummy_execution() is True


def test_dummy_execution_does_not_duplicate_an_early_completion():
    runner = object.__new__(mod.ModelRunner)
    runner.label = "test runner"
    runner.block_size = 1
    runner.drafter = types.SimpleNamespace(mtp_k=3)

    def _forward(batch):
        assert batch.is_dummy_run
        runner._forward_output_published_early = True

    runner.forward = _forward

    assert runner.dummy_execution() is None


def test_decode_prefix_view_is_reused_and_observes_new_buffer_contents():
    builder = object.__new__(v4_mod.DeepseekV4AttentionMetadataBuilder)
    backing = torch.arange(16).reshape(4, 4)

    first = builder._cached_decode_prefix(backing, 2)
    second = builder._cached_decode_prefix(backing, 2)

    assert first is second
    backing[0, 0] = 99
    assert first[0, 0].item() == 99


def test_decode_prefix_view_cache_distinguishes_storage_offsets():
    builder = object.__new__(v4_mod.DeepseekV4AttentionMetadataBuilder)
    backing = torch.arange(24).reshape(6, 4)
    left = backing[:4]
    right = backing[2:]

    left_prefix = builder._cached_decode_prefix(left, 2)
    right_prefix = builder._cached_decode_prefix(right, 2)

    assert left_prefix.data_ptr() != right_prefix.data_ptr()
    assert left_prefix.tolist() == backing[:2].tolist()
    assert right_prefix.tolist() == backing[2:4].tolist()


def test_cached_decode_plan_keeps_views_but_refreshes_host_values():
    builder = object.__new__(v4_mod.DeepseekV4AttentionMetadataBuilder)
    compress_view = torch.empty(8, 4, dtype=torch.int32)
    write_view = torch.empty(16, 4, dtype=torch.int32)
    old_cu = np.array([0, 1], dtype=np.int32)
    new_cu = np.array([0, 2, 3], dtype=np.int32)
    new_rows = np.arange(12, dtype=np.int32).reshape(3, 4)
    cached_plan = types.SimpleNamespace(
        compress_plan_gpu=compress_view,
        write_plan_gpu=write_view,
        num_compress=1,
        num_write=1,
        cu_compress_cpu=old_cu,
        compress_plan_cpu=None,
    )
    staged_plan = types.SimpleNamespace(
        num_compress=3,
        num_write=7,
        cu_compress_cpu=new_cu,
        compress_plan_cpu=new_rows,
    )

    builder._refresh_staged_compress_plans({4: cached_plan}, {4: staged_plan})

    assert cached_plan.compress_plan_gpu is compress_view
    assert cached_plan.write_plan_gpu is write_view
    assert cached_plan.num_compress == 3
    assert cached_plan.num_write == 7
    assert cached_plan.cu_compress_cpu is new_cu
    assert cached_plan.compress_plan_cpu is new_rows


def test_cached_decode_metadata_shell_isolated_from_mtp_rewrites():
    positions = torch.arange(8, dtype=torch.int32)
    original = v4_mod.AttentionMetaData_DSV4(
        cu_seqlens_q=positions[:3],
        max_seqlen_q=4,
    )
    template = original.__dict__.copy()

    first = v4_mod.DeepseekV4AttentionMetadataBuilder._clone_decode_metadata(template)
    first.max_seqlen_q = 1
    first.slot_mapping = positions[:1]
    second = v4_mod.DeepseekV4AttentionMetadataBuilder._clone_decode_metadata(template)

    assert second is not first
    assert second.max_seqlen_q == 4
    assert second.slot_mapping is None
    assert second.cu_seqlens_q is original.cu_seqlens_q
