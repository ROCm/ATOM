from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# On the module, not on "aiter" -- see tests/test_dspark.py for why a name-only
# guard passes on the non-GPU runner and then aborts collection.
pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="requires AITER to import model_runner",
    exc_type=ImportError,
)

from atom.model_engine.model_runner import (
    ModelRunner,
    _kv_config_has_producer,
    tokenIDProcessor,
)
from atom.model_engine.scheduler import ScheduledBatch, defers_output
from atom.spec_decode.status import SpecStatusQueue
from atom.utils.selector import Family


def _prefill_batch(is_final_chunk: list[bool]) -> ScheduledBatch:
    batch = object.__new__(ScheduledBatch)
    batch.scheduled_tokens = np.arange(4, dtype=np.int32)
    batch.total_tokens_num = 4
    batch.total_tokens_num_prefill = 4
    batch.total_tokens_num_decode = 0
    batch.total_seqs_num = len(is_final_chunk)
    batch.total_seqs_num_prefill = len(is_final_chunk)
    batch.total_seqs_num_decode = 0
    batch.has_prefill = True
    batch.is_final_chunk = is_final_chunk
    return batch


def _batch_info():
    """`tokenIDProcessor.batch_info`: what `prepare_input_ids` writes first."""
    return SimpleNamespace(np=np.zeros(3, dtype=np.int32), copy_to_gpu=mock.Mock())


def _input_ids_group():
    """The runner's `token_inputs`, as far as `prepare_input_ids` sees it."""
    group = SimpleNamespace(
        counts=[None, None, None],
        indices={"input_ids": 0, "decode_src": 1, "batch_info": 2},
        publish=mock.Mock(),
        check_writable=mock.Mock(),
    )

    def set_counts(counts):
        for name, count in counts.items():
            group.counts[group.indices[name]] = count

    group.set_counts = set_counts
    return group


def _batch_with_ids(req_ids: list[int]) -> ScheduledBatch:
    """The one field `get_token_locations` reads, and nothing else."""
    batch = object.__new__(ScheduledBatch)
    batch.req_ids = req_ids
    return batch


def _locations(prev: ScheduledBatch, cur: ScheduledBatch):
    processor = object.__new__(tokenIDProcessor)
    processor.prev_batch = prev
    return tokenIDProcessor.get_token_locations(processor, cur)


def test_a_real_batch_after_a_dummy_takes_the_host_path():
    """A DP-sync dummy schedules no request, so nothing carries over from it:
    its own `postprocess` reports the PREVIOUS batch's tokens, so by the time
    this request is scheduled its anchor is already on the host."""
    dummy = _batch_with_ids([])
    real = _batch_with_ids([7, 9])

    locs = _locations(dummy, real)

    assert locs.deferred_curr.tolist() == []
    assert locs.new_curr.tolist() == [0, 1]


def test_a_real_batch_still_carries_over_from_a_real_batch():
    """The dummy rule must not cost an ordinary decode step its deferred rows."""
    prev = _batch_with_ids([7, 9])
    cur = _batch_with_ids([9, 11])

    locs = _locations(prev, cur)

    assert locs.deferred_curr.tolist() == [0]  # req 9 is position 0 now...
    assert locs.deferred_prev.tolist() == [1]  # ...and was position 1 before
    assert locs.new_curr.tolist() == [1]


def _processor() -> tokenIDProcessor:
    processor = object.__new__(tokenIDProcessor)
    processor.runner = SimpleNamespace(h2d_groups={"input_ids": _input_ids_group()})
    processor.input_ids = SimpleNamespace(
        np=np.zeros(8, dtype=np.int32),
        gpu=np.zeros(8, dtype=np.int32),
        copy_to_gpu=mock.Mock(),
    )
    processor.batch_info = _batch_info()
    processor.draft_tokens_per_seq = 0
    processor.recv_spec_status = mock.Mock(
        return_value=(
            np.array([2], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
    )
    processor.prev_rejected_num = np.array([7], dtype=np.int32)
    processor.prev_bonus_num = np.array([8], dtype=np.int32)
    return processor


@pytest.mark.parametrize(
    ("kv_config", "expected"),
    [
        ({}, False),
        ({"kv_connector": "mooncake", "kv_role": "kv_consumer"}, False),
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, True),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                ],
            },
            True,
        ),
    ],
)
def test_detects_remote_prefill_producer(kv_config, expected):
    assert _kv_config_has_producer(kv_config) is expected


def _processor_for(*, kv_config: dict, family: Family) -> tokenIDProcessor:
    runner = SimpleNamespace(
        config=SimpleNamespace(
            pipeline_parallel_size=1, kv_transfer_config=kv_config, max_num_seqs=8
        ),
        attn_family=family,
        device="cuda",
    )
    with (
        mock.patch("atom.model_engine.model_runner.CpuGpuBuffer"),
        mock.patch("atom.model_engine.model_runner.torch.cuda.Stream"),
        mock.patch("atom.model_engine.model_runner.torch.zeros"),
    ):
        return tokenIDProcessor(runner, max_num_batched_tokens=8)


@pytest.mark.parametrize("pp_size", [1, 2])
def test_the_scheduler_resolves_the_runners_deferral(pp_size):
    """The scheduler has no runner: it reads the family off the config, and
    must land where the runner does for a recurrent-state producer."""
    config = SimpleNamespace(
        pipeline_parallel_size=pp_size,
        kv_transfer_config={"kv_role": "kv_producer"},
        hf_config=object(),
    )
    with (
        mock.patch("atom.model_engine.scheduler.get_hf_text_config"),
        mock.patch(
            "atom.model_engine.scheduler.attn_family", return_value=Family.KIMI_MLA
        ),
    ):
        assert defers_output(config) is False
    config.kv_transfer_config = {}
    assert defers_output(config) is (pp_size == 1)


def test_a_consumer_keeps_deferred_output():
    """The other polarity of the sweep below, which fixes the role to producer.

    Which config shapes read as a producer is `_kv_config_has_producer`'s
    answer, pinned above; what is left for this layer is that the processor
    asks it about `runner.config.kv_transfer_config` at all.
    """
    processor = _processor_for(
        kv_config={"kv_role": "kv_consumer"}, family=Family.KIMI_MLA
    )
    assert processor.is_deferred_out is True


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        (Family.KIMI_MLA, False),
        (Family.QSA_GDN, False),
        (Family.GDN, False),
        (Family.V4, True),
        (Family.MLA, True),
        (Family.MHA, True),
        (Family.CSA2, False),
    ],
)
def test_only_recurrent_state_producers_give_up_deferred_output(family, expected):
    """A paged write lands at T0's own position, so the consumer repeating it
    changes nothing -- V4 and the other paged models stay on the path they
    used before P/D handoff existed. CSA2 (DeepSeek-V4.1) hands off a STATE
    slot whose cursor, compressor rings and Engram history advance per token,
    so like the recurrent families its producer emits T0 from the prefill."""
    processor = _processor_for(kv_config={"kv_role": "kv_producer"}, family=family)
    assert processor.is_deferred_out is expected


@pytest.mark.parametrize(
    ("kv_role", "expected"), [("kv_producer", True), ("kv_consumer", False)]
)
def test_the_processor_records_the_producer_role_whatever_the_state(kv_role, expected):
    """The runner reads it to size its DP dummy and to scope its graph lookup,
    and a paged producer -- which keeps deferred output -- is still one."""
    processor = _processor_for(kv_config={"kv_role": kv_role}, family=Family.V4)
    assert processor.is_pd_producer is expected


def _decode_batch(*, num_spec_step: int) -> ScheduledBatch:
    batch = object.__new__(ScheduledBatch)
    batch.scheduled_tokens = np.array([11, 12, 13], dtype=np.int32)
    batch.total_tokens_num = 3
    batch.total_tokens_num_prefill = 0
    batch.total_tokens_num_decode = 3
    batch.total_seqs_num = 3
    batch.total_seqs_num_prefill = 0
    batch.total_seqs_num_decode = 3
    batch.has_prefill = False
    batch.num_spec_step = num_spec_step
    return batch


def _recurrent_producer_processor() -> tokenIDProcessor:
    """Non-deferred, and `use_spec` still on: the runner keeps the drafter."""
    processor = _processor()
    processor.is_deferred_out = False
    processor.is_pipeline_parallel = False
    processor.use_spec = True
    return processor


def test_a_non_deferred_decode_stages_the_scheduled_ids_as_they_are():
    processor = _recurrent_producer_processor()
    group = processor.runner.h2d_groups["input_ids"]

    tokenIDProcessor.prepare_input_ids(
        processor, _decode_batch(num_spec_step=0), 1, group
    )

    np.testing.assert_array_equal(processor.input_ids.np[:3], [11, 12, 13])
    assert group.counts == [3, None, 3]
    group.publish.assert_called_once()


def test_the_non_deferred_branch_refuses_a_batch_carrying_drafts():
    """Nothing reaches it with drafts: PP verifies none locally and a producer
    never speculates. Staging them here would size rows by `max_seqlen_q`,
    which under DP is the group's maximum for a block drafter, not this
    batch's row width."""
    processor = _recurrent_producer_processor()

    with pytest.raises(AssertionError, match="num_spec_step=3"):
        tokenIDProcessor.prepare_input_ids(
            processor,
            _decode_batch(num_spec_step=3),
            4,
            processor.runner.h2d_groups["input_ids"],
        )


@pytest.mark.parametrize(("is_pd_producer", "expected_q"), [(False, 4), (True, 1)])
def test_a_dp_dummy_runs_at_the_schedulers_query_width(is_pd_producer, expected_q):
    """An idle rank's dummy is unified with its peers' decodes, so it must
    match their height. A producer's peers decode at q=1 whatever its drafter's
    `mtp_k`; a q=mtp_k+1 dummy beside them hands the uniform decode collectives
    two different `running_tokens`."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.drafter = SimpleNamespace(mtp_k=3)
    runner.tokenID_processor = SimpleNamespace(is_pd_producer=is_pd_producer)
    runner.block_size = 16
    runner.label = "runner"
    seen = []
    runner.forward = seen.append

    ModelRunner.dummy_execution(runner)

    (batch,) = seen
    # A step with no request, not a fabricated one: its rows are padding.
    assert not batch.is_dummy_run and not batch.has_prefill
    # It schedules nothing; `decide` reads the query width off `num_spec_step`.
    assert batch.total_seqs_num == batch.total_tokens_num == 0
    assert batch.num_spec_step == expected_q - 1


def _graph_runner(*, is_pd_producer: bool, drafter: bool = True) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.tokenID_processor = SimpleNamespace(is_pd_producer=is_pd_producer)
    if drafter:
        runner.drafter = SimpleNamespace(mtp_k=3)
    runner._tr_hybrid_active = lambda: False
    # mtp_k=3: the request bucket of 8 requests verifying 3 drafts each
    runner.req_graph = {(8, 3)}
    return runner


def test_only_a_producer_with_a_drafter_may_miss_a_recording():
    """Everywhere else a missing request bucket is a capture gap, and handing
    `decide` no shapes keeps it a `KeyError` at replay instead of a silent
    eager step."""
    producer = _graph_runner(is_pd_producer=True)
    assert (8, 3) in ModelRunner._producer_graph_shapes(producer)

    assert (
        ModelRunner._producer_graph_shapes(_graph_runner(is_pd_producer=False)) is None
    )
    assert (
        ModelRunner._producer_graph_shapes(
            _graph_runner(is_pd_producer=True, drafter=False)
        )
        is None
    )

    producer.req_graph = None  # enforce_eager: nothing recorded
    assert ModelRunner._producer_graph_shapes(producer) is None


@pytest.mark.parametrize(
    ("is_pd_producer", "drafter", "is_prefill", "decodes", "expected"),
    [
        (False, True, False, 2, True),
        # The producer's decode surfaces a deferred T0 at q=1: no drafts, so a
        # rejection pass would select over zero target rows.
        (True, True, False, 2, False),
        (False, True, True, 2, False),
        # An idle DP rank's step: no request, nothing drafted to score.
        (False, True, False, 0, False),
        (True, True, False, 0, False),
        (False, False, False, 2, False),
    ],
)
def test_only_a_drafting_decode_verifies(
    is_pd_producer, drafter, is_prefill, decodes, expected
):
    """Both the spec-decode metadata and its packed indices follow this
    predicate, so a producer step neither scores nor publishes drafts."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.tokenID_processor = SimpleNamespace(is_pd_producer=is_pd_producer)
    if drafter:
        runner.drafter = SimpleNamespace(mtp_k=3)
    batch = SimpleNamespace(total_seqs_num_decode=decodes)

    assert ModelRunner._verifies_drafts(runner, batch, is_prefill) is expected


def test_middle_prefills_preserve_status_until_mixed_final_batch():
    processor = _processor()
    group = processor.runner.h2d_groups["input_ids"]

    # Pure middle chunks skip postprocess, so neither the deferred-token queue
    # nor its matching speculative-status queue may advance.
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False]), 1, group)
    tokenIDProcessor.prepare_input_ids(
        processor, _prefill_batch([False, False]), 1, group
    )

    processor.recv_spec_status.assert_not_called()
    np.testing.assert_array_equal(processor.prev_rejected_num, [7])
    np.testing.assert_array_equal(processor.prev_bonus_num, [8])

    # If any request reaches its final chunk, the batch runs postprocess. Its
    # status dequeue must therefore happen exactly once, even though another
    # request in the same batch is still a middle chunk.
    tokenIDProcessor.prepare_input_ids(
        processor, _prefill_batch([False, True]), 1, group
    )

    processor.recv_spec_status.assert_called_once_with()
    np.testing.assert_array_equal(processor.prev_rejected_num, [2])
    np.testing.assert_array_equal(processor.prev_bonus_num, [1])


def _deferred_processor(cu_seqlens_q: np.ndarray, new_curr: list[int]):
    """A deferred-output processor whose next decode admits `new_curr`."""
    bs = len(cu_seqlens_q) - 1
    processor = object.__new__(tokenIDProcessor)
    processor.input_ids = SimpleNamespace(
        np=np.zeros(16, dtype=np.int32),
        gpu=np.zeros(16, dtype=np.int32),
        copy_to_gpu=mock.Mock(),
    )
    processor.decode_src = SimpleNamespace(
        np=np.zeros(bs, dtype=np.int32),
        gpu=np.zeros(bs, dtype=np.int32),
        copy_to_gpu=mock.Mock(return_value=np.zeros(bs, dtype=np.int32)),
    )
    processor.batch_info = _batch_info()
    processor.draft_tokens_per_seq = 3
    input_ids_group = _input_ids_group()
    processor.is_deferred_out = True
    processor.prev_batch = object()
    processor.use_spec = True
    processor.prev_rejected_num = None
    processor.prev_bonus_num = None
    processor.pre_num_decode_token_per_seq = 1
    processor.spec_status_queue = SpecStatusQueue(8, mock.Mock())
    processor.prev_token_ids = np.zeros(bs, dtype=np.int32)
    processor.draft_token_ids = None
    forward_vars = {"cu_seqlens_q": SimpleNamespace(np=cu_seqlens_q, gpu=cu_seqlens_q)}
    processor.runner = SimpleNamespace(
        enforce_eager=True,
        capture_sizes=[],
        forward_vars=forward_vars,
        h2d_groups={"input_ids": input_ids_group},
        # Mirrors AttentionMetadataBuilder.decode_spans: the lengths and the
        # cumsum come off the batch and the published buffer, not off a second
        # copy this path keeps.
        attn_metadata_builder=SimpleNamespace(
            decode_spans=lambda b: (
                b.total_seqs_num_decode,
                b.num_scheduled_tokens[: b.total_seqs_num_decode],
                forward_vars["cu_seqlens_q"].np[: b.total_seqs_num_decode + 1],
            )
        ),
    )
    processor.get_token_locations = mock.Mock(
        return_value=SimpleNamespace(
            deferred_curr=np.array([], dtype=np.int32),
            deferred_prev=np.array([], dtype=np.int32),
            new_curr=np.array(new_curr, dtype=np.int32),
        )
    )
    return processor, input_ids_group


def test_deferred_decode_returns_the_flat_width_it_staged():
    processor, input_ids_group = _deferred_processor(
        np.array([0, 4, 8], dtype=np.int32), [0, 1]
    )

    batch = SimpleNamespace(
        scheduled_tokens=np.arange(8, dtype=np.int32),
        # Two q=4 rows, so the flat width is 8 and the request count is 2.
        total_tokens_num=8,
        total_tokens_num_prefill=0,
        total_tokens_num_decode=8,
        total_seqs_num=2,
        total_seqs_num_prefill=0,
        total_seqs_num_decode=2,
        has_prefill=False,
        num_scheduled_tokens=np.array([4, 4], dtype=np.int32),
        req_ids=[10, 11],
        num_spec_step=3,
        scheduled_spec_decode_tokens=np.arange(6, dtype=np.int32).reshape(2, 3),
        num_rejected=np.zeros(2, dtype=np.int32),
        num_bonus=np.zeros(2, dtype=np.int32),
        produces_output=lambda: False,
    )

    with mock.patch(
        "atom.model_engine.model_runner.fill_deferred_decode_ids"
    ) as fill_ids:
        result = tokenIDProcessor.prepare_input_ids(
            processor, batch, 4, input_ids_group
        )

    assert result.shape == (8,)
    assert input_ids_group.counts == [8, 2, 3]
    # Two requests, eight scheduled tokens, three draft tokens each.
    np.testing.assert_array_equal(processor.batch_info.np, [2, 8, 6])
    fill_ids.assert_called_once()
    # Both requests are new, so their draft columns come from the draft table,
    # not from `scheduled_tokens` (which would leave 1-3 and 5-7 here).
    np.testing.assert_array_equal(processor.input_ids.np[1:4], [0, 1, 2])
    np.testing.assert_array_equal(processor.input_ids.np[5:8], [3, 4, 5])


def test_a_producer_dp_dummy_stages_without_a_draft_table():
    """A producer keeps `use_spec` (its drafter writes context KV) but its DP
    dummy is q=1 with `num_spec_step=0`, and it schedules nothing, so staging
    it reads no draft table: that raised AttributeError on every idle DPA rank
    when the dummy still fabricated a request."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.drafter = SimpleNamespace(mtp_k=3)
    runner.tokenID_processor = SimpleNamespace(is_pd_producer=True)
    runner.block_size = 16
    runner.label = "runner"
    seen = []
    runner.forward = seen.append
    ModelRunner.dummy_execution(runner)
    (batch,) = seen
    assert batch.num_spec_step == 0
    spec = getattr(batch, "scheduled_spec_decode_tokens", None)
    assert spec is None or spec.size == 0

    processor, input_ids_group = _deferred_processor(np.array([0], dtype=np.int32), [])
    with mock.patch("atom.model_engine.model_runner.fill_deferred_decode_ids"):
        result = tokenIDProcessor.prepare_input_ids(
            processor, batch, 1, input_ids_group
        )

    assert result.shape == (0,)
