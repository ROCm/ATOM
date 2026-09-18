"""The block table has to grow with the allocation, not just with admission.

vLLM calls `update_state_after_alloc` exactly once per admission and hands it
only the blocks allocated by then. Every block allocated afterwards -- each
further chunk of a chunked prefill, and each block decode appends -- is
announced only through `scheduled_cached_reqs.new_block_ids`.

The adapter caps the save frontier at what the block table covers, which is the
right semantics (KV past the table is in no block this connector knows about),
but with a table frozen at admission the cap pins the frontier at the first
prefill chunk forever: `aligned == saved` on every later step, so the save loop
never fires again. On GLM-5.2 with a 16,384-token budget that meant 20k-token
prompts stored exactly 16,384 tokens and nothing else -- half of every long
prefix invisible to the external tier, with every metric reporting success.

Driven against the REAL `DenseOffloadScheduler`, so what is asserted is the
saves it actually emits, not the adapter's view of its own bookkeeping.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry

connector_mod = pytest.importorskip(
    "atom.plugin.vllm.kv_transfer.connector",
    reason="the adapter imports vLLM's connector base",
)
dense_mod = pytest.importorskip(
    "atom.kv_transfer.offload.dense.connector",
    reason="the dense scheduler pulls the offload stack",
)

from atom.kv_transfer.disaggregation.types import ConnectorCompletion
from atom.kv_transfer.offload import config as offcfg

BLOCK, CHUNK, WORLD = 64, 64, 4
BUDGET = 256  # one prefill chunk: four blocks
PROMPT = 1024  # four budgets, so three saves are hidden behind the first


def _adapter(monkeypatch, dcp=1):
    """The adapter wired to a real `DenseOffloadScheduler`.

    Built without `__init__` for the same reason the sibling files do it: a real
    construction needs a VllmConfig and a live LMCache engine.
    """
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _c=None: SimpleNamespace(chunk_size=CHUNK * dcp),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_a: object())
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "kv_both"},
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=dcp,
        tensor_parallel_size=WORLD,
    )
    adapter = object.__new__(connector_mod.AtomLMCacheOffloadConnector)
    adapter._scheduler = dense_mod.DenseOffloadScheduler(config)
    adapter._config = config
    adapter._seqs = SeqViewRegistry()
    adapter._promised_loads = {}
    adapter._deferred_frees = set()
    adapter._deferred_free_at = {}
    adapter._next_save_reconcile_at = 0.0
    adapter._releases_in_flight = set()
    adapter._save_reports = {}
    adapter._load_failure_reports = {}
    adapter._completion_reports = {}
    adapter._world_size = WORLD
    return adapter, adapter._scheduler


def _admit(adapter, req_id="r0", prompt=PROMPT, allocated=BUDGET):
    """Admit a request whose block table covers only its first prefill chunk."""
    request = SimpleNamespace(request_id=req_id, prompt_token_ids=list(range(prompt)))
    blocks = list(range(allocated // BLOCK))
    adapter.update_state_after_alloc(request, (blocks,), 0)
    return request, adapter._seqs.get(req_id), len(blocks)


def _step(adapter, req_id, computed, new_blocks=None, resumed=()):
    """One `schedule()` worth of announcements, as vLLM shapes them."""
    scheduler_output = SimpleNamespace(
        preempted_req_ids=set(),
        scheduled_new_reqs=(),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[req_id],
            num_computed_tokens=[computed],
            new_block_ids=[None if new_blocks is None else (list(new_blocks),)],
            resumed_req_ids=set(resumed),
        ),
    )
    return list(adapter.build_connector_meta(scheduler_output).inner.requests)


def _land(adapter, operation):
    """Report the save landed on every rank, so the next one may be emitted."""
    for _ in range(WORLD):
        adapter.update_connector_output(
            SimpleNamespace(
                finished_sending=set(),
                finished_recving=set(),
                kv_connector_worker_meta=connector_mod.AtomOffloadWorkerMetadata(
                    {str(operation.req_id): 1},
                    {},
                    [
                        ConnectorCompletion(
                            dense_mod.DENSE_PAGE_STORE_CHANNEL, operation, True
                        )
                    ],
                ),
            )
        )


def _saves(requests):
    """`(skip_leading_tokens, end)` of every save in a step's metadata."""
    return [
        (int(r.save_spec.skip_leading_tokens), len(r.token_ids))
        for r in requests
        if getattr(r, "save_spec", None) is not None and r.save_spec.can_save
    ]


def test_a_chunked_prefill_saves_past_its_first_chunk(monkeypatch):
    """The regression itself: chunk 2 onwards must reach the external tier."""
    adapter, _scheduler = _adapter(monkeypatch)
    _request, seq, blocks = _admit(adapter)
    assert len(seq.block_table) == blocks == BUDGET // BLOCK

    stored = []
    for chunk in range(1, PROMPT // BUDGET + 1):
        computed = chunk * BUDGET
        # vLLM allocates the NEXT chunk's blocks in the step that computes this
        # one, so the announcement rides along with the frontier it precedes.
        grown = range(chunk * (BUDGET // BLOCK), (chunk + 1) * (BUDGET // BLOCK))
        requests = _step(adapter, "r0", computed, new_blocks=list(grown))
        stored.extend(_saves(requests))
        for req in requests:
            if getattr(req, "save_operation", None) is not None:
                _land(adapter, req.save_operation)

    assert stored == [(0, 256), (256, 512), (512, 768), (768, 1024)], (
        "without the growth loop the frontier is pinned at the admitted table "
        "and only the first chunk is ever stored"
    )


def test_the_frontier_never_outruns_the_blocks_it_names(monkeypatch):
    """Growth must not undo the cap: a frontier ahead of the table still waits.

    LMCache is handed the block table alongside the token range and fails the
    transfer outright ("needed_blocks=N+1, available_blocks=N") if the range
    names a block that is not there.
    """
    adapter, _scheduler = _adapter(monkeypatch)
    _request, seq, _blocks = _admit(adapter)

    # Frontier claims two chunks; only one chunk's blocks were ever announced.
    requests = _step(adapter, "r0", 2 * BUDGET, new_blocks=None)

    assert seq.num_cached_tokens == BUDGET
    assert _saves(requests) == [(0, BUDGET)]


def test_a_resumed_request_replaces_its_table_instead_of_appending(monkeypatch):
    """Preemption returns the blocks to the pool; the new ids are the whole table.

    Appending here would leave the table naming blocks that belong to another
    request now, and the save loop reads exactly that table.
    """
    adapter, _scheduler = _adapter(monkeypatch)
    _request, seq, _blocks = _admit(adapter)
    _step(adapter, "r0", BUDGET, new_blocks=[4, 5, 6, 7])
    assert seq.block_table == [0, 1, 2, 3, 4, 5, 6, 7]

    fresh = [90, 91, 92, 93, 94, 95, 96, 97]
    _step(adapter, "r0", BUDGET, new_blocks=fresh, resumed=["r0"])

    assert seq.block_table == fresh


def test_the_frontier_is_capped_in_virtual_blocks_under_dcp(monkeypatch):
    """One scheduler block id covers `block_size * dcp` tokens, not `block_size`.

    The cap exists because KV past the block table is in no block this connector
    knows about. Priced in physical blocks it under-reports coverage by the DCP
    factor, so on a dcp=2 deployment every request's save frontier is pinned at
    half of what is actually resident -- the same "half of every long prefix is
    invisible to the tier, and every metric says success" shape this file was
    written for, just triggered by a config rather than by prompt length.

    The whole offload stack already indexes this table in virtual units
    (`chunked_scheduler.virtual_block_size`), so the cap is the one place that
    disagreed.
    """
    dcp = 2
    adapter, scheduler = _adapter(monkeypatch, dcp=dcp)
    assert scheduler.virtual_block_size == BLOCK * dcp

    blocks = 8
    resident = blocks * BLOCK * dcp  # 1024 tokens, all of them computed
    request = SimpleNamespace(
        request_id="r0", prompt_token_ids=list(range(resident + 1))
    )
    adapter.update_state_after_alloc(request, (list(range(blocks)),), 0)

    _step(adapter, "r0", resident)

    seq = adapter._seqs.get("r0")
    assert seq.num_cached_tokens == resident
