# SPDX-License-Identifier: MIT
"""Prove the raw-token oracle catches corruption at each runtime boundary."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from atom.model_engine.engram_runtime import EngramInputPreparer
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from tests.model_ops.test_engram import make_runtime

from .dspark_state_audit import RuntimeStateAudit


@pytest.mark.parametrize(
    "fault, message",
    [
        (None, None),
        ("incoming", "history at"),
        ("embedding", "staged Engram"),
        ("acceptance", "target argmax"),
        ("commit", "committed cursor/history"),
        ("output", "finalized token"),
    ],
)
def test_raw_prefix_audit_detects_boundary_corruption(fault, message):
    host = make_runtime()
    preparer = EngramInputPreparer(host.prefetcher._hash_mapping, host)
    span = RequestSpan(31, 3, 0, 3, 0, (0,))
    metadata = SimpleNamespace(
        step=SimpleNamespace(
            requests=(span,), tentative=True, cu_seqlens_q=torch.tensor([0, 3])
        ),
        cache=SimpleNamespace(cursor=torch.zeros(1, 3, dtype=torch.int64)),
    )

    def commit(metadata, indices):
        # Accept token9 but reject token42: state includes inputs through9,
        # while the sampler may already return bonus10 to the scheduler.
        metadata.cache.cursor[0] = torch.tensor(
            [5, 9, 42] if fault == "commit" else [5, 8, 9]
        )

    runner = SimpleNamespace(
        attn_metadata_builder=SimpleNamespace(
            engram=preparer, commit_speculative_state=commit
        ),
        run_model=lambda *args: None,
    )
    if fault == "embedding":
        original = preparer.prepare

        def corrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            next(iter(result.embeddings.values()))[0, 0, 0] += 123
            return result

        preparer.prepare = corrupted
    audit = RuntimeStateAudit(runner)
    seq = SimpleNamespace(
        id=31,
        prompt_token_ids=[5, 6, 7, 8],
        num_prompt_tokens=4,
        token_ids=[5, 6, 7, 8],
        num_finalized_tokens=4,
        is_finished=True,
        leave_reason="max_tokens",
    )
    audit.bind([seq])
    audit.ledger[31] = [5, 6, 7]

    def execute():
        history = [[6, 99]] if fault == "incoming" else [[6, 7]]
        preparer.prepare((span,), torch.tensor([8, 9, 42]), np.asarray(history))
        audit.top1 = [9, 10, 11]
        index = 2 if fault == "acceptance" else 1
        runner.attn_metadata_builder.commit_speculative_state(
            metadata, torch.tensor([index])
        )
        seq.token_ids += [9, 99 if fault == "output" else 10]
        seq.num_finalized_tokens = 6
        audit.finish()

    try:
        if message is None:
            execute()
            assert audit.records[0]["accepted_draft_counts"] == {1: 1}
            assert audit.records[0]["checks"]["finalized_generated_tokens"] == 2
        else:
            with pytest.raises(AssertionError, match=message):
                execute()
    finally:
        audit.close()
        host.shutdown()


def test_audit_commits_middle_prefill_without_logits(monkeypatch):
    from . import dspark_state_audit

    span = RequestSpan(31, 0, 0, 2, 0, (0,))
    metadata = SimpleNamespace(
        dummy=False, step=SimpleNamespace(tentative=False, requests=(span,))
    )
    monkeypatch.setattr(
        dspark_state_audit,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    audit = RuntimeStateAudit.__new__(RuntimeStateAudit)
    audit._run = lambda *args: (None, "hidden")
    calls = []
    audit.check_commit = lambda metadata, lengths: calls.append(lengths)
    from collections import Counter

    audit.stats = Counter()
    assert audit.run(torch.tensor([5, 6]), None) == (None, "hidden")
    assert calls == [[2]]
    assert audit.stats["middle_prefill_forwards"] == 1


def test_visible_cache_comparison_ignores_rejected_expired_and_draft_rows():
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    from .dspark_runtime_trace import compare_visible_cache

    geometry = V41PoolGeometry(
        3, ((0, 2), (1, 1)), 32, 128, 128, 32, speculative_tokens=5
    )
    cache = PagedAttentionCache(geometry, 12, 2, "cpu")
    shadow = PagedAttentionCache(geometry, 12, 2, "cpu")
    blocks = (5, 2, 8, 1, 0, 3, 4, 7, 6)
    span = RequestSpan(31, 130, 0, 6, 1, blocks)
    window = cache.state.view("window")
    window[0, 1, 131, 0] = 3  # accepted, still visible
    window[0, 1, 3, 0] = 5  # expired at committed end132
    window[0, 1, 132, 0] = 7  # rejected
    window[2, 1, 131, 0] = 9  # draft layer has its own state
    pages = cache.pages.view("main_0")[0]
    rows = pages.shape[1]  # a ratio-2 owner's rows per PAGE, not the PAGE
    for index in [65, 66]:  # first accepted, second rejected
        pages[blocks[index // rows], index % rows, 0] = 11
    result = compare_visible_cache(cache, shadow, span, 2, target_layers=2)
    assert result["window"] == [
        {"layer": 0, "max_error": 3, "positions": [131], "unequal": 1}
    ]
    assert result["global"] == [
        {"field": "main_0", "rows": [65], "unequal": 1, "max_error": 11}
    ]
