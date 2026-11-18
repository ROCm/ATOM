# SPDX-License-Identifier: MIT
"""The runner publishes decode-local query spans before token assembly."""

from types import SimpleNamespace

import numpy as np
import pytest
from aiter_stub import atom_without_aiter
from test_mixed_decode_view import make_batch
from test_mixed_metadata_publication import _cpu_query_runner


@pytest.fixture(scope="module", autouse=True)
def kernel_imports():
    with atom_without_aiter():
        yield


@pytest.mark.parametrize("spec", [0, 1])
def test_prepare_model_publishes_decode_local_query_spans(monkeypatch, spec):
    from atom.model_engine import model_runner

    batch = make_batch(num_spec_step=spec)
    runner, _, _, _ = _cpu_query_runner(monkeypatch)
    mode = SimpleNamespace(
        sync=SimpleNamespace(any_rank_has_prefill=True),
        is_prefill=True,
        running_bs=batch.total_seqs_num,
        running_tokens=batch.total_tokens_num,
        max_seqlen_q=spec + 1,
        decode_ubatch_slices=None,
    )
    monkeypatch.setattr(model_runner.ForwardMode, "decide", lambda **kwargs: mode)
    runner.config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_tbo=False,
        enable_tbo_decode=False,
    )
    runner._dspark_apply_q_bucket = lambda batch: None
    runner._local_tbo_eligibility = lambda batch: False
    runner._producer_graph_shapes = lambda: None
    runner._cut_ubatches = lambda mode, batch: mode
    runner._verify_spans = lambda batch: None
    runner.prepare_sample = lambda batch, group: (None, None, None, True, False)
    runner.enforce_eager = True
    runner.capture_sizes_np = np.array([], dtype=np.int32)
    runner.running_tokens_per_dp_rank = None
    runner.tbo_decode_graphs = False
    calls = []

    def assemble(batch, width, group, *, defer_spec_status):
        # Token assembly is the first consumer of the real query publication.
        group.publish()
        calls.append("tokens")
        cu = runner.forward_vars["cu_seqlens_q"].gpu[:4].tolist()
        assert cu == ([0, 2, 3, 5] if spec else [0, 1, 2, 3])
        return object()

    def inputs(batch, input_ids, *, forward_mode, spec_decode_indices):
        assert forward_mode is mode
        calls.append("attention")

    runner.tokenID_processor = SimpleNamespace(prepare_input_ids=assemble)
    runner.prepare_inputs = inputs
    runner.h2d_owner.begin()
    model_runner.ModelRunner.prepare_model(runner, batch)
    assert calls == ["tokens", "attention"]
    ids = runner.forward_vars["batch_id_per_q_token"].gpu[
        : batch.total_tokens_num_decode
    ]
    assert ids.tolist() == ([0, 0, 1, 2, 2] if spec else [0, 1, 2])
    runner.h2d_owner.finish()
