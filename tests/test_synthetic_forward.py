# SPDX-License-Identifier: MIT
"""Forced acceptance preserves model IDs unless synthetic forward is enabled."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from atom.spec_decode.synthetic import resolve_synthetic_token_id


def _config(rates=(1.0, 0.5), **kwargs):
    hf = SimpleNamespace(vocab_size=32)
    return SimpleNamespace(
        hf_config=hf,
        speculative_config=SimpleNamespace(
            synthetic_acceptance_rates=rates, draft_model_hf_config=hf
        ),
        **kwargs,
    )


@pytest.fixture
def synthetic_forward(monkeypatch):
    monkeypatch.setenv("ATOM_SPEC_DECODE_SYNTHETIC_FORWARD", "1")


@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_disabled_acceptance_needs_no_model_metadata(monkeypatch, setting):
    monkeypatch.delenv("ATOM_SPEC_DECODE_SYNTHETIC_FORWARD", raising=False)
    if setting is not None:
        monkeypatch.setenv("ATOM_SPEC_DECODE_SYNTHETIC_FORWARD", setting)
    assert resolve_synthetic_token_id(SimpleNamespace(speculative_config=None)) is None
    assert resolve_synthetic_token_id(_config(rates=None)) is None


@pytest.mark.parametrize("setting", [None, "0"])
def test_rejection_only_is_default_and_needs_no_vocab(monkeypatch, setting):
    monkeypatch.delenv("ATOM_SPEC_DECODE_SYNTHETIC_FORWARD", raising=False)
    if setting is not None:
        monkeypatch.setenv("ATOM_SPEC_DECODE_SYNTHETIC_FORWARD", setting)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(synthetic_acceptance_rates=(1.0, 0.5))
    )
    assert resolve_synthetic_token_id(config) is None


def test_fake_id_is_shared_and_avoids_special_and_stop_tokens(synthetic_forward):
    config = _config(eos_token_id=0, stop_token_ids=[1, 5])
    config.hf_config.bos_token_id = 2
    config.generation_config = SimpleNamespace(eos_token_id=[3, 4])
    config.speculative_config.draft_model_hf_config = SimpleNamespace(
        vocab_size=16, pad_token_id=6
    )
    assert resolve_synthetic_token_id(config) == 7
    assert resolve_synthetic_token_id(config) == 7


def test_zero_acceptance_still_enables_synthetic_forward(synthetic_forward):
    assert resolve_synthetic_token_id(_config(rates=(0.0, 0.0))) == 0


def test_fake_id_must_fit_both_vocabularies(synthetic_forward):
    config = _config(stop_token_ids=[0, 1])
    config.speculative_config.draft_model_hf_config = SimpleNamespace(vocab_size=2)
    with pytest.raises(ValueError, match="non-special token"):
        resolve_synthetic_token_id(config)


def _runner_module():
    return pytest.importorskip("atom.model_engine.model_runner", exc_type=ImportError)


@pytest.mark.parametrize("fake_id", [None, 7])
@pytest.mark.parametrize("prefill", [False, True])
def test_target_staging_precedes_metadata_and_includes_graph_padding(
    monkeypatch, fake_id, prefill
):
    mod = _runner_module()
    runner = object.__new__(mod.ModelRunner)
    runner.synthetic_token_id = fake_id
    runner.config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1), enable_tbo=False
    )
    runner.enforce_eager = False
    runner.capture_sizes_np = np.array([4])
    runner._dspark_apply_q_bucket = lambda batch: None
    runner._local_tbo_eligibility = lambda batch: False
    runner._piecewise_cg_active = lambda: False
    runner.prepare_sample = lambda batch: (None, None, None, True, False)
    mode = SimpleNamespace(sync=None, max_seqlen_q=3, running_tokens=12)
    monkeypatch.setattr(mod.ForwardMode, "decide", lambda **kwargs: mode)
    storage = torch.arange(16, dtype=torch.int32)
    original = storage.clone()
    runner.tokenID_processor = SimpleNamespace(
        input_ids=SimpleNamespace(gpu=storage),
        prepare_input_ids=lambda batch, q: storage[:6],
    )
    runner.attn_metadata_builder = SimpleNamespace(publish_cu_seqlens_q=Mock())
    seen = []
    runner.prepare_inputs = lambda batch, ids, **kwargs: seen.append(ids.clone())
    batch = SimpleNamespace(
        total_tokens_num=6, total_seqs_num_prefill=int(prefill), num_spec_step=2
    )

    result = runner.prepare_model(batch)[0]
    if fake_id is not None and not prefill:
        assert result.tolist() == [fake_id] * 6
        assert seen[0].tolist() == [fake_id] * 6
        assert storage[:12].tolist() == [fake_id] * 12
    else:
        torch.testing.assert_close(storage, original)
    torch.testing.assert_close(storage[12:], original[12:])


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("fake_id", [None, 7])
def test_serial_draft_head_feedback_and_export_match(reuse, fake_id):
    mod = pytest.importorskip("atom.spec_decode.eagle_proposer", exc_type=ImportError)
    proposer = object.__new__(mod.EagleProposer)
    proposer.synthetic_token_id = fake_id
    proposer._reuse_step_buffers = reuse

    def sample(hidden, *, out):
        out.fill_(19)
        return out

    proposer.model = SimpleNamespace(compute_draft_ids=Mock(side_effect=sample))
    hidden = torch.randn(3, 8)
    staged_ids = torch.zeros(3, dtype=torch.int32)
    _, ids = proposer._step_head(
        hidden, 3, input_ids=staged_ids, hidden_states=torch.empty_like(hidden)
    )
    assert ids.tolist() == [19 if fake_id is None else fake_id] * 3
    assert proposer.model.compute_draft_ids.call_count == 1
    if reuse:
        assert ids.data_ptr() == staged_ids.data_ptr()


@pytest.mark.parametrize("fake_id", [None, 7])
def test_dspark_block_keeps_model_work_and_exports_fake_tokens(fake_id):
    mod = pytest.importorskip("atom.spec_decode.dspark_proposer", exc_type=ImportError)
    proposer = object.__new__(mod.DSparkProposer)
    proposer.synthetic_token_id = fake_id
    proposer.mtp_k = 3
    confidence = torch.rand(2, 3)
    proposer.model = SimpleNamespace(
        head_and_sample=Mock(return_value=(torch.full((2, 3), 19), confidence))
    )
    ids, actual_confidence = proposer._block_head(
        torch.randn(2, 3, 8), 2, anchor_ids=torch.full((2,), 7)
    )
    assert ids.tolist() == [[19 if fake_id is None else fake_id] * 3] * 2
    assert actual_confidence is confidence
    assert proposer.model.head_and_sample.call_count == 1


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _sample(rates, lengths, *, step=0, device="cuda:0", fake_id=7, draft_id=19):
    from atom.model_ops.rejection_sampler import rejection_sample

    n = len(rates) if rates is not None else max(lengths)
    num_tokens = sum(lengths)
    logits = torch.zeros(num_tokens, 32, device=device)
    logits[:, 19] = 1
    return rejection_sample(
        torch.full((num_tokens,), draft_id, dtype=torch.int32, device=device),
        n,
        torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0).int(),
        None,
        logits,
        torch.full((len(lengths), 1), 23, dtype=torch.int32, device=device),
        synthetic_acceptance_rates=rates,
        synthetic_step=step,
        synthetic_token_id=fake_id,
    )


@GPU
@pytest.mark.parametrize("accepted", [0, 1, 3])
@pytest.mark.parametrize("fake_id", [None, 0, 7])
def test_synthetic_kernel_preserves_counts_and_invalid_tail(accepted, fake_id):
    rates = (1.0,) * accepted + (0.0,) * (3 - accepted)
    lengths = [3, 0, 1, 2]
    ids, counts = _sample(rates, lengths, fake_id=fake_id, draft_id=17)
    expected_counts = [min(accepted, length) for length in lengths]
    assert counts.tolist() == expected_counts
    expected = []
    for length, count in zip(lengths, expected_counts):
        if fake_id is None:
            # Even when forced acceptance disagrees with the model, keep the
            # draft ID (17), target correction (19), or bonus (23).
            valid = [17] * count + [19 if count < length else 23]
        else:
            valid = [fake_id] * (count + 1)
        expected.append(valid + [-1] * (3 - count))
    assert ids.tolist() == expected


@GPU
def test_sampler_defaults_to_rejection_only():
    from atom.model_ops.rejection_sampler import RejectionSampler

    sampler = RejectionSampler(synthetic_acceptance_rates=[1.0, 0.0])
    metadata = SimpleNamespace(
        draft_token_ids=torch.tensor([17, 18], device="cuda", dtype=torch.int32),
        num_spec_steps=2,
        cu_num_draft_tokens=torch.tensor([2], device="cuda", dtype=torch.int32),
    )
    logits = torch.zeros(2, 32, device="cuda")
    logits[:, 19] = 1
    ids, counts = sampler(metadata, logits, torch.tensor([[23]], device="cuda"))
    assert ids.tolist() == [[17, 19, -1]]
    assert counts.tolist() == [1]


@GPU
def test_synthetic_fractional_schedule_is_rank_consistent_and_rng_isolated():
    rates = (1.0, 1.0, 0.78, 0.0)
    lengths = [4] * 8192
    ids, counts = _sample(rates, lengths, step=41)
    assert set(counts.tolist()) == {2, 3}
    assert abs(counts.float().mean().item() + 1 - 3.78) < 0.02
    assert torch.all(ids[ids >= 0] == 7)
    legacy_ids, legacy_counts = _sample(rates, lengths, step=41, fake_id=None)
    torch.testing.assert_close(counts, legacy_counts)
    torch.testing.assert_close(ids == -1, legacy_ids == -1)
    assert torch.all(legacy_ids[legacy_ids >= 0] == 19)
    torch.rand(1234, device="cuda:0")
    ids_again, counts_again = _sample(rates, lengths, step=41)
    torch.testing.assert_close(ids, ids_again)
    torch.testing.assert_close(counts, counts_again)
    _, next_counts = _sample(rates, lengths, step=42)
    assert not torch.equal(counts, next_counts)
    if torch.cuda.device_count() > 1:
        # Triton launches on the current device, as each TP worker does.
        with torch.cuda.device(1):
            other_ids, other_counts = _sample(rates, lengths, step=41, device="cuda:1")
        torch.testing.assert_close(ids.cpu(), other_ids.cpu())
        torch.testing.assert_close(counts.cpu(), other_counts.cpu())


@GPU
def test_disabled_synthetic_mode_keeps_real_draft_and_bonus_ids():
    ids, counts = _sample(None, [3, 1])
    assert ids.tolist() == [[19, 19, 19, 23], [19, 23, -1, -1]]
    assert counts.tolist() == [3, 1]


@GPU
@pytest.mark.parametrize("reuse", [False, True])
def test_serial_draft_fake_feedback_survives_graph_replay(reuse):
    mod = pytest.importorskip("atom.spec_decode.eagle_proposer", exc_type=ImportError)
    proposer = object.__new__(mod.EagleProposer)
    proposer.synthetic_token_id = 7
    proposer._reuse_step_buffers = reuse

    def sample(hidden, *, out):
        out.copy_(hidden.argmax(-1).int())
        return out

    proposer.model = SimpleNamespace(compute_draft_ids=sample)
    hidden = torch.randn(3, 32, device="cuda")
    staged_hidden = torch.empty_like(hidden)
    staged_ids = torch.empty(3, dtype=torch.int32, device="cuda")

    def head():
        return proposer._step_head(
            hidden, 3, input_ids=staged_ids, hidden_states=staged_hidden
        )[1]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        head()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids = head()
    for _ in range(3):
        hidden.normal_()
        staged_ids.fill_(29)
        graph.replay()
        assert ids.tolist() == [7, 7, 7]


@GPU
def test_dspark_fake_block_survives_graph_replay():
    mod = pytest.importorskip("atom.spec_decode.dspark_proposer", exc_type=ImportError)
    proposer = object.__new__(mod.DSparkProposer)
    proposer.synthetic_token_id = 7
    proposer.mtp_k = 3
    proposer.model = SimpleNamespace(
        head_and_sample=lambda out, anchors, n: (out.argmax(-1).int(), None)
    )
    logits = torch.randn(2, 3, 32, device="cuda")
    anchors = torch.full((2,), 7, dtype=torch.int32, device="cuda")

    def head():
        return proposer._block_head(logits, 2, anchor_ids=anchors)[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        head()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids = head()
    for _ in range(3):
        logits.normal_()
        graph.replay()
        assert ids.tolist() == [[7, 7, 7], [7, 7, 7]]
