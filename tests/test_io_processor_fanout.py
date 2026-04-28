# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for InputOutputProcessor.preprocess_fanout (SamplingParams.n>1).

These tests bypass ``__init__`` because the real processor needs a full
engine config, a loaded tokenizer, and mamba-detection helpers. We only
need to exercise ``preprocess_fanout`` / ``preprocess`` here — both of
those touch just ``self.tokenizer.encode``, ``self.block_size``, and a
handful of instance attributes, which we wire up by hand.
"""

import sys
import types
from itertools import count
from unittest.mock import MagicMock

import pytest

# ``atom.model_engine.llm_engine`` transitively imports aiter via CoreManager
# on real installs; here we stub CoreManager so the module is importable
# without AMD/ROCm-only dependencies.
if "atom.model_engine.engine_core_mgr" not in sys.modules:
    _stub = types.ModuleType("atom.model_engine.engine_core_mgr")

    class _StubCoreManager:  # noqa: D401 - placeholder
        def __init__(self, *a, **kw):
            self.added = []

        def add_request(self, reqs):
            self.added.extend(reqs)

    _stub.CoreManager = _StubCoreManager
    sys.modules["atom.model_engine.engine_core_mgr"] = _stub

from atom.model_engine.llm_engine import InputOutputProcessor  # noqa: E402
from atom.model_engine.sequence import Sequence  # noqa: E402
from atom.sampling_params import SamplingParams  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_sequence_counter():
    Sequence.counter = count()
    yield
    Sequence.counter = count()


def _make_processor() -> InputOutputProcessor:
    proc = InputOutputProcessor.__new__(InputOutputProcessor)
    proc.config = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(side_effect=lambda s, **_: list(range(len(s))))
    proc.tokenizer = tokenizer
    proc.block_size = 4
    proc.requests = {}
    proc.mamba_enabled = False
    proc.num_speculative_tokens = 0
    return proc


class TestPreprocessSingle:
    def test_returns_single_sequence(self):
        proc = _make_processor()
        seq = proc.preprocess("hello", SamplingParams(n=1))
        assert isinstance(seq, Sequence)
        assert seq.needs_independent_noise is False
        assert seq.sibling_index == 0
        assert seq.parent_request_id is None

    def test_rejects_n_greater_than_one(self):
        proc = _make_processor()
        with pytest.raises(ValueError, match="preprocess_fanout"):
            proc.preprocess("hello", SamplingParams(n=3, temperature=0.8))


class TestPreprocessFanout:
    def test_n_one_returns_list_of_one(self):
        proc = _make_processor()
        seqs = proc.preprocess_fanout("hello", SamplingParams(n=1))
        assert len(seqs) == 1
        assert seqs[0].needs_independent_noise is False

    def test_n_greater_than_one_fans_out(self):
        proc = _make_processor()
        sp = SamplingParams(n=4, temperature=0.8)
        seqs = proc.preprocess_fanout("hello", sp, parent_request_id="chatcmpl-abc")
        assert len(seqs) == 4
        assert [s.sibling_index for s in seqs] == [0, 1, 2, 3]
        # Every sibling must request independent noise so the sampler
        # produces diverse outputs across siblings.
        assert all(s.needs_independent_noise for s in seqs)
        assert all(s.parent_request_id == "chatcmpl-abc" for s in seqs)

    def test_fanout_seq_ids_are_unique(self):
        proc = _make_processor()
        seqs = proc.preprocess_fanout("hello", SamplingParams(n=3, temperature=0.8))
        assert len({s.id for s in seqs}) == 3

    def test_fanout_registers_every_sibling_in_requests(self):
        proc = _make_processor()
        seqs = proc.preprocess_fanout("hello", SamplingParams(n=3, temperature=0.8))
        for s in seqs:
            assert s.id in proc.requests

    def test_fanout_shares_prompt_tokens(self):
        proc = _make_processor()
        seqs = proc.preprocess_fanout("hello", SamplingParams(n=2, temperature=0.8))
        assert seqs[0].token_ids == seqs[1].token_ids
        # Mutations on one sibling must not leak to the others (Sequence
        # copies the input token list internally).
        seqs[0].append_token(999)
        assert seqs[1].token_ids != seqs[0].token_ids

    def test_fanout_routes_per_sibling_callbacks(self):
        proc = _make_processor()
        callbacks = [MagicMock() for _ in range(3)]
        seqs = proc.preprocess_fanout(
            "hello",
            SamplingParams(n=3, temperature=0.8),
            stream_callbacks=callbacks,
        )
        for i, seq in enumerate(seqs):
            assert seq.stream_callback is callbacks[i]

    def test_fanout_stream_callbacks_length_mismatch_raises(self):
        proc = _make_processor()
        with pytest.raises(ValueError, match="stream_callbacks length"):
            proc.preprocess_fanout(
                "hello",
                SamplingParams(n=3, temperature=0.8),
                stream_callbacks=[MagicMock(), MagicMock()],
            )

    def test_fanout_falls_back_to_scalar_callback(self):
        proc = _make_processor()
        cb = MagicMock()
        seqs = proc.preprocess_fanout(
            "hello",
            SamplingParams(n=2, temperature=0.8),
            stream_callback=cb,
        )
        assert all(s.stream_callback is cb for s in seqs)
