from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.models.deepseek_v4.mono import dispatch as mono_dispatch
from atom.models.deepseek_v4.mono.cache import V4CacheAdapter
from atom.models.deepseek_v4.mono.contract import (
    BatchContinuation,
    ContinuationRow,
    continues_batch,
)
from atom.models.deepseek_v4.mono.profile import MTP3_TARGET_PROFILE
from atom.models.deepseek_v4.mono.runner import V4MonoDecodeRunner
from atom.models.deepseek_v4.mono.session import V4MonoSession
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils.forward_context import AttnState

_DEFAULT_BLOCKS = tuple(range(65))


def _batch_frontier(width=4, position=8192):
    return BatchContinuation(
        7,
        (width, 1),
        23,
        tuple(
            ContinuationRow(row + 3, row + 11, position, tuple(range(65)))
            for row in range(width)
        ),
    )


def test_batch_continuation_accepts_exact_width_next_step():
    previous = _batch_frontier()
    current = tuple(
        (row.position + 1, row.slot, row.blocks + ((71,) if i == 2 else ()))
        for i, row in enumerate(previous.rows)
    )

    assert continues_batch(previous, (4, 1), current)


def test_batch_continuation_rejects_membership_width_or_table_change():
    previous = _batch_frontier()
    current = tuple((row.position + 1, row.slot, row.blocks) for row in previous.rows)

    assert not continues_batch(previous, (2, 1), current)
    assert not continues_batch(previous, (4, 1), current[:-1])
    assert not continues_batch(
        previous,
        (4, 1),
        ((8193, 99, previous.rows[0].blocks), *current[1:]),
    )
    assert not continues_batch(
        previous,
        (4, 1),
        (
            (8193, previous.rows[0].slot, (*previous.rows[0].blocks[:-1], 99)),
            *current[1:],
        ),
    )


@pytest.mark.parametrize("advance", [1, 2, 3, 4])
def test_mtp_q4_continuation_accepts_any_committed_prefix(advance):
    blocks = tuple(range(33))
    previous = BatchContinuation(
        7,
        (1, 4),
        23,
        (ContinuationRow(3, 11, 8192, blocks),),
    )

    assert continues_batch(previous, (1, 4), ((8192 + advance, 3, blocks),))


@pytest.mark.parametrize("advance", [0, 5])
def test_mtp_q4_continuation_rejects_stale_or_skipped_window(advance):
    blocks = tuple(range(33))
    previous = BatchContinuation(
        7,
        (1, 4),
        23,
        (ContinuationRow(3, 11, 8192, blocks),),
    )

    assert not continues_batch(previous, (1, 4), ((8192 + advance, 3, blocks),))


def test_batch_slot_reset_invalidates_any_member():
    session = object.__new__(V4MonoSession)
    session.slot_generations = {3: 11, 4: 12, 5: 13, 6: 14}
    session.armed = False
    session.batch_continuation = _batch_frontier()

    session.slots_reset((5,))

    assert session.slot_generations[5] == 14
    assert session.batch_continuation is None


def test_graph_recapture_advances_generation_and_invalidates_continuation():
    session = object.__new__(V4MonoSession)
    session.graph_generation = 23
    session.graph_keys = {(1, 4)}
    session.graph_generations = {(1, 4): 23}
    session.batch_continuation = BatchContinuation(
        7,
        (1, 4),
        23,
        (ContinuationRow(3, 11, 8192, tuple(range(33))),),
    )

    session.remember_graph((1, 4), True)

    assert session.graph_generation == 24
    assert session.graph_generations == {(1, 4): 24}
    assert session.batch_continuation is None


def test_graph_removal_advances_generation_and_invalidates_continuation():
    session = object.__new__(V4MonoSession)
    session.graph_generation = 23
    session.graph_keys = {(1, 4)}
    session.graph_generations = {(1, 4): 23}
    session.batch_continuation = BatchContinuation(
        7,
        (1, 4),
        23,
        (ContinuationRow(3, 11, 8192, tuple(range(33))),),
    )

    session.remember_graph((1, 4), False)

    assert session.graph_generation == 24
    assert session.graph_keys == set()
    assert session.graph_generations == {}
    assert session.batch_continuation is None


def test_mtp_continuation_can_cross_one_block_table_frontier():
    old_blocks = tuple(range(64))
    previous = BatchContinuation(
        7,
        (1, 4),
        23,
        (ContinuationRow(3, 11, 8188, old_blocks),),
    )

    assert continues_batch(
        previous,
        (1, 4),
        ((8192, 3, old_blocks + (71,)),),
    )


def test_batch_fast_publication_uses_exact_graph_and_skips_device_validation():
    previous = _batch_frontier()
    current = tuple((row.position + 1, row.slot, row.blocks) for row in previous.rows)
    writes = []
    session = object.__new__(V4MonoSession)
    session.armed = False
    session.publication_generation = 0
    session.publication_slot_generations = ()
    session.capture_publication = False
    session.adapter = SimpleNamespace(
        clear_publication=lambda: None,
        continuation_contract=lambda context: current,
        write=lambda *args, **kwargs: writes.append(kwargs),
    )
    session.runner = object()
    session.decode_attempts = 0
    session.batch_continuation = previous
    session.owner_generation = 7
    session.graph_generations = {(4, 1): 23}
    session.slot_generations = {row.slot: row.slot_generation for row in previous.rows}
    session.weights_ready = True
    session.fast_publications_by_width = {}
    session.graph_is_current = lambda key: key == (4, 1)
    session._arm_publication = lambda *, capture: not capture
    context = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=False,
            is_dummy_run=False,
            running_bs=4,
        ),
        attn_metadata=SimpleNamespace(max_seqlen_q=1),
    )

    assert session.publish(object(), object(), context)
    assert session.decode_attempts == 1
    assert session.fast_publications_by_width == {4: 1}
    assert writes == [
        {
            "validate_static": False,
            "validate_device_values": False,
            "supported": True,
        }
    ]
    assert tuple(row.position for row in session.batch_continuation.rows) == (
        8193,
        8193,
        8193,
        8193,
    )


def test_mtp_q4_fast_publication_tracks_request_frontier_and_four_rows():
    blocks = tuple(range(33))
    previous = BatchContinuation(
        7,
        (1, 4),
        23,
        (ContinuationRow(3, 11, 8192, blocks),),
    )
    current = ((8195, 3, blocks),)
    writes = []
    session = object.__new__(V4MonoSession)
    session.profile = MTP3_TARGET_PROFILE
    session.armed = False
    session.publication_generation = 0
    session.publication_slot_generations = ()
    session.capture_publication = False
    session.adapter = SimpleNamespace(
        clear_publication=lambda: None,
        continuation_contract=lambda context: current,
        write=lambda *args, **kwargs: writes.append(kwargs),
    )
    session.runner = object()
    session.decode_attempts = 0
    session.batch_continuation = previous
    session.owner_generation = 7
    session.graph_generations = {(1, 4): 23}
    session.slot_generations = {3: 11}
    session.weights_ready = True
    session.fast_publications_by_width = {}
    session.graph_is_current = lambda key: key == (1, 4)
    session._arm_publication = lambda *, capture: not capture
    context = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=False,
            is_dummy_run=False,
            running_bs=1,
        ),
        attn_metadata=SimpleNamespace(max_seqlen_q=4),
    )

    assert session.publish(object(), object(), context)
    assert session.fast_publications_by_width == {4: 1}
    assert writes == [
        {
            "validate_static": False,
            "validate_device_values": False,
            "supported": True,
        }
    ]
    assert session.batch_continuation.rows[0].position == 8195


def test_s8_uses_one_exact_two_phase_stack():
    assert V4MonoDecodeRunner._launch_plan(8) == {
        "launch_shape": "two",
        "qkv_batch_limit": 1,
        "qkv_samples_per_pass": 1,
        "qkv_norm_handoff": True,
    }


def test_wider_bucket_plan_prefers_exact_s8():
    adapter = object.__new__(V4CacheAdapter)
    adapter.max_rows = 16
    adapter._buckets = {
        width: tuple(
            SimpleNamespace(width=width, ordinal=ordinal)
            for ordinal in range(2 if width == 8 else 1)
        )
        for width in (1, 2, 4, 8)
    }

    assert [
        (offset, bucket.width, bucket.ordinal) for offset, bucket in adapter._plan(15)
    ] == [(0, 8, 0), (8, 4, 0), (12, 2, 0), (14, 1, 0)]
    assert [
        (offset, bucket.width, bucket.ordinal) for offset, bucket in adapter._plan(16)
    ] == [(0, 8, 0), (8, 8, 1)]


def test_publication_consensus_tensor_encodes_every_effective_field():
    adapter = object.__new__(V4CacheAdapter)
    adapter.max_rows = 4
    adapter.table_cols = 5
    adapter.published_rows = 2
    adapter.published_contract = (
        (17, 8192, 3, (9, 4)),
        (23, 8193, 7, (9, 4, 12)),
    )

    payload = adapter.publication_consensus_tensor()

    assert payload.device.type == "cpu"
    assert payload.dtype == torch.int64
    assert payload.tolist() == [
        2,
        17,
        8192,
        3,
        2,
        9,
        4,
        -1,
        -1,
        -1,
        23,
        8193,
        7,
        3,
        9,
        4,
        12,
        -1,
        -1,
        *([-1] * 18),
    ]


def _mtp_adapter_fixture():
    adapter = object.__new__(V4CacheAdapter)
    adapter.max_rows = 4
    adapter.max_requests = 1
    adapter.tokens_per_seq = 4
    adapter.prepared_widths = (4,)
    adapter.model = SimpleNamespace(embed=SimpleNamespace(num_embeddings=1000))
    adapter.table_cols = 2
    adapter.positions = torch.empty(4, dtype=torch.int32)
    adapter.tables = torch.empty((4, 2), dtype=torch.int32)
    adapter.configs = (SimpleNamespace(max_seq=512),)
    adapter._layer_bases = ()
    forward_vars = {
        "v4_meta_state_slot_out": SimpleNamespace(np=np.array([7], dtype=np.int32)),
        "v4_meta_state_slot_in": SimpleNamespace(np=np.array([7], dtype=np.int32)),
        "positions": SimpleNamespace(np=np.array([255, 256, 257, 258], dtype=np.int32)),
        "block_tables": SimpleNamespace(np=np.array([[2, 3]], dtype=np.int32)),
    }
    adapter.builder = SimpleNamespace(
        num_blocks=8,
        model_runner=SimpleNamespace(forward_vars=forward_vars),
    )
    bucket = SimpleNamespace(
        width=4,
        ordinal=0,
        tokens=torch.empty(4, dtype=torch.int32),
        positions=torch.empty(4, dtype=torch.int32),
        slots=torch.empty(4, dtype=torch.int32),
        tables=torch.empty((4, 2), dtype=torch.int32),
        metadata={},
        layers=(),
    )
    adapter._buckets = {4: (bucket,)}
    context = SimpleNamespace(
        ubatch_slices=None,
        context=SimpleNamespace(
            is_prefill=False,
            scheduled_bs=1,
            scheduled_tokens=4,
            running_bs=1,
        ),
        attn_metadata=SimpleNamespace(
            state=AttnState.DECODE,
            max_seqlen_q=4,
            state_slot_in=torch.tensor([7], dtype=torch.int32),
            state_slot_out=torch.tensor([7], dtype=torch.int32),
            block_tables=torch.tensor([[2, 3]], dtype=torch.int32),
        ),
    )
    return adapter, bucket, context


def test_mtp_q4_admission_maps_four_rows_to_one_request():
    adapter, _, context = _mtp_adapter_fixture()

    assert adapter._actual_shape(context) == (1, 4)
    assert adapter.supported(context)
    assert adapter.continuation_contract(context) == ((255, 7, (2, 3)),)

    context.context.scheduled_bs = 4
    assert adapter._actual_shape(context) == (0, 0)


def test_mtp_q4_admission_requires_consecutive_positions():
    adapter, _, context = _mtp_adapter_fixture()
    adapter.builder.model_runner.forward_vars["positions"].np[2] = 67

    assert not adapter.supported(context)


def test_mtp_q4_publication_repeats_slot_and_table_per_verify_row():
    adapter, bucket, context = _mtp_adapter_fixture()
    tokens = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    positions = torch.tensor([255, 256, 257, 258], dtype=torch.int32)

    adapter.write(
        tokens,
        positions,
        context,
        validate_static=False,
        supported=True,
    )

    assert bucket.slots.tolist() == [7, 7, 7, 7]
    assert bucket.tables.tolist() == [[2, 3], [2, 3], [2, 3], [2, 3]]
    assert adapter.published_contract == (
        (11, 255, 7, (2,)),
        (12, 256, 7, (2, 3)),
        (13, 257, 7, (2, 3)),
        (14, 258, 7, (2, 3)),
    )


def test_mtp_q4_publication_keeps_request_span_after_ring_index_build():
    adapter, bucket, context = _mtp_adapter_fixture()
    cfg = SimpleNamespace(
        compress_ratio=8,
        window=2,
        n_index=0,
        indexed=False,
    )
    adapter.configs = (cfg,)
    adapter.geometry = SimpleNamespace(
        window_params=lambda ratio: SimpleNamespace(
            ring_slots=8,
            ring_stride=2,
            slot_rows=32,
            ring_start=0,
            run_rows=8,
        )
    )
    bucket.metadata = {
        8: (
            torch.empty((2, 4), dtype=torch.int64),
            torch.empty((4, 2), dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
    }

    adapter.write(
        torch.tensor([11, 12, 13, 14], dtype=torch.int32),
        torch.tensor([255, 256, 257, 258], dtype=torch.int32),
        context,
        validate_static=False,
        supported=True,
    )

    assert [row[2] for row in adapter.published_contract] == [7, 7, 7, 7]


def test_live_prefill_bypasses_mono_admission():
    cleared = []
    session = object.__new__(V4MonoSession)
    session.armed = True
    session.publication_generation = 9
    session.publication_slot_generations = ((3, 1),)
    session.capture_publication = True
    session.adapter = SimpleNamespace(clear_publication=lambda: cleared.append(True))
    session.runner = object()
    session.batch_continuation = _batch_frontier(1)
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_dummy_run=False)
    )

    assert not session.publish(None, None, context)
    assert cleared == [True]
    assert session.batch_continuation is None
    assert not session.armed
    assert session.publication_generation == 0
    assert session.publication_slot_generations == ()
    assert not session.capture_publication


def test_close_retires_graph_generation_and_publication(monkeypatch):
    events = []
    session = object.__new__(V4MonoSession)
    session.graph_generation = 23
    session.armed = True
    session.graph_keys = {(1, 4)}
    session.graph_generations = {(1, 4): 23}
    session.publication_generation = 7
    session.publication_slot_generations = ((3, 11),)
    session.capture_publication = True
    session.batch_continuation = BatchContinuation(
        7,
        (1, 4),
        23,
        (ContinuationRow(3, 11, 8192, tuple(range(33))),),
    )
    session.adapter = SimpleNamespace(
        clear_publication=lambda: events.append("publication")
    )
    session.runner = SimpleNamespace(close=lambda: events.append("runner"))
    session.entry_sources = (object(),)
    session.output = object()
    session.token_out = object()
    monkeypatch.setattr(session, "_release_lease", lambda: events.append("lease"))

    session.close()

    assert session.graph_generation == 24
    assert session.graph_keys == set()
    assert session.graph_generations == {}
    assert session.batch_continuation is None
    assert session.runner is None and session.adapter is None
    assert events == ["publication", "runner", "lease"]


@pytest.mark.parametrize("cache_reset", [False, True])
def test_weight_update_stays_native_until_real_cache_rebind(cache_reset):
    session = object.__new__(V4MonoSession)
    called = []
    session.weights_ready = False
    session.requires_cache_rebind = True
    session.refusal = None
    session.prepare = lambda **kwargs: called.append(kwargs)

    session.weights_updated(cache_reset=cache_reset)

    assert session.weights_ready
    assert session.requires_cache_rebind
    assert called == []
    assert "explicit cache/state rebind" in session.refusal


def test_prepare_refuses_weight_generation_without_cache_rebind():
    session = object.__new__(V4MonoSession)
    session.disabled = False
    session.weights_ready = True
    session.requires_cache_rebind = True
    session.refusal = None

    assert not session.prepare()
    assert "explicit cache/state rebind" in session.refusal


def _install_config(policy="auto", *, tp=8):
    return SimpleNamespace(
        mono_decode=policy,
        tensor_parallel_size=tp,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        enable_tbo=False,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
        kv_cache_dtype="fp8",
        index_cache_dtype="fp4",
        enable_prefix_caching=False,
        hf_config=SimpleNamespace(use_index_cache=True, index_topk_freq=4),
        enable_rapidserve=False,
        disagg_is_decode=False,
        compilation_config=SimpleNamespace(
            level=0,
            cudagraph_mode=SimpleNamespace(name="FULL"),
        ),
    )


def _mtp_install_config(policy="auto"):
    config = _install_config(policy)
    config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=3)
    config.hf_config.use_index_cache = False
    config.hf_config.index_topk_freq = 1
    return config


def test_mono_policy_defaults_to_auto_install(monkeypatch):
    installed = object()
    monkeypatch.setattr(mono_dispatch, "V4MonoSession", lambda *args: installed)
    model = SimpleNamespace()

    assert (
        mono_dispatch.install_mono_decode(model, _install_config(), None, object())
        is model
    )
    assert model._v4_mono is installed


def test_mtp3_profile_installs_with_q4_and_share1(monkeypatch):
    installed = object()
    seen = []

    def make_session(*args):
        seen.append(args[-1])
        return installed

    monkeypatch.setattr(mono_dispatch, "V4MonoSession", make_session)
    model = SimpleNamespace()
    drafter = SimpleNamespace(mtp_k=3, is_block_drafter=False)

    assert (
        mono_dispatch.install_mono_decode(
            model, _mtp_install_config(), drafter, object()
        )
        is model
    )
    assert model._v4_mono is installed
    assert seen[0].name == "mtp3_target"
    assert seen[0].tokens_per_seq == 4
    assert seen[0].auto_request_widths == (1,)
    assert seen[0].accepts_graph(2, 4, force=True)
    assert not seen[0].accepts_graph(2, 4, force=False)
    assert not seen[0].greedy_terminal
    assert seen[0].continuation


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: setattr(config.hf_config, "use_index_cache", True),
        lambda config: setattr(config.hf_config, "index_topk_freq", 4),
        lambda config: setattr(config.speculative_config, "num_speculative_tokens", 5),
    ],
)
def test_mtp3_profile_refuses_other_index_or_verify_contracts(mutate):
    model = SimpleNamespace()
    config = _mtp_install_config()
    mutate(config)
    drafter = SimpleNamespace(mtp_k=3, is_block_drafter=False)

    assert mono_dispatch.install_mono_decode(model, config, drafter, object()) is model
    assert not hasattr(model, "_v4_mono")


def test_mono_policy_off_keeps_native(monkeypatch):
    monkeypatch.setattr(
        mono_dispatch,
        "V4MonoSession",
        lambda *args: pytest.fail("off policy must not install Mono"),
    )
    model = SimpleNamespace()

    assert (
        mono_dispatch.install_mono_decode(model, _install_config("off"), None, object())
        is model
    )
    assert not hasattr(model, "_v4_mono")


def test_mono_policy_auto_keeps_unsupported_deployment_native():
    model = SimpleNamespace()

    assert (
        mono_dispatch.install_mono_decode(model, _install_config(tp=4), None, object())
        is model
    )
    assert not hasattr(model, "_v4_mono")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: setattr(config, "enable_prefix_caching", True),
        lambda config: setattr(config.hf_config, "index_topk_freq", 1),
    ],
)
def test_mono_policy_auto_preserves_unqualified_online_features(mutate):
    model = SimpleNamespace()
    config = _install_config()
    mutate(config)

    assert mono_dispatch.install_mono_decode(model, config, None, object()) is model
    assert not hasattr(model, "_v4_mono")


def test_mono_policy_force_rejects_unsupported_deployment():
    with pytest.raises(MonoUnsupported, match="force policy"):
        mono_dispatch.install_mono_decode(
            SimpleNamespace(), _install_config("force", tp=4), None, object()
        )


def test_mono_policy_force_rejects_live_decode_fallback():
    session = object.__new__(V4MonoSession)
    session.policy = "force"
    session.refusal = "unsupported_contract"
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False, is_dummy_run=False)
    )

    with pytest.raises(MonoUnsupported, match="unsupported_contract"):
        session.enforce_selection(context, False)

    session.enforce_selection(context, True)
    session.enforce_selection(
        SimpleNamespace(context=SimpleNamespace(is_prefill=True, is_dummy_run=False)),
        False,
    )


def test_auto_sampling_request_falls_back_before_admission():
    session = object.__new__(V4MonoSession)
    session.policy = "auto"
    session.armed = False
    session.publication_generation = 0
    session.publication_slot_generations = ()
    session.capture_publication = False
    session.adapter = SimpleNamespace(clear_publication=lambda: None)
    session.runner = object()
    session.batch_continuation = None
    session.decode_attempts = 0
    session.fallback_steps = 0
    session.fallback_reasons = {}
    context = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=False,
            is_dummy_run=False,
            running_bs=4,
        )
    )

    assert not session.publish(None, None, context, allow_auto=False)
    assert session.decode_attempts == 1
    assert session.fallback_steps == 1
    assert session.fallback_reasons == {
        "sampling/logprob request is outside the V4 Mono auto profile": 1
    }


def test_auto_unpublished_width_falls_back_before_admission():
    session = object.__new__(V4MonoSession)
    session.policy = "auto"
    session.armed = False
    session.publication_generation = 0
    session.publication_slot_generations = ()
    session.capture_publication = False
    session.adapter = SimpleNamespace(clear_publication=lambda: None)
    session.runner = object()
    session.batch_continuation = None
    session.decode_attempts = 0
    session.fallback_steps = 0
    session.fallback_reasons = {}
    context = SimpleNamespace(
        context=SimpleNamespace(
            is_prefill=False,
            is_dummy_run=False,
            running_bs=16,
        ),
        attn_metadata=SimpleNamespace(max_seqlen_q=1),
    )

    assert not session.publish(None, None, context)
    assert session.decode_attempts == 1
    assert session.fallback_steps == 1
    assert session.fallback_reasons == {
        "graph shape (16, 1) is not enabled for V4 Mono ordinary profile": 1
    }
