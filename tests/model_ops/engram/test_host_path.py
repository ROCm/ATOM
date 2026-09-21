"""Unit tests for the engram host path.

The load-bearing test is `test_head_vocab_sizes_match_checkpoint`: the derived
per-layer row counts have to equal what DeepSeek-V4.1-Flash's config declares,
which only happens if the prime search runs in exactly the reference order.
"""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    """`EngramOp` builds an ATOM layer, and those read the process group.

    `linear` reaches AITER, which a CPU-only runner does not have. Nothing that
    needs this patch can construct there either -- `make_op` skips first -- so
    the absence is a no-op rather than a failure for the host-path tests.
    """
    from types import SimpleNamespace

    try:
        from atom.model_ops import linear
    except ModuleNotFoundError:
        return

    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )


from atom.model_ops.engram.host import (
    EngramHost,
    EngramPrefetchCache,
    EngramPrefetcher,
    EngramRequest,
)
from atom.model_ops.engram.mapping import (
    EngramConfig,
    NgramHashMapping,
    _is_prime,
    _next_prime,
)
from atom.model_ops.engram.tables import HostEmbeddingTable

# The engram block of deepseek-ai/DeepSeek-V4.1-Flash config.json -> text_config.
V41_FLASH = {
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
}


class StubTokenizer:
    """Identity compression over a small vocabulary."""

    def __init__(self, vocab_size: int = 512):
        self.lookup_table = np.arange(vocab_size, dtype=np.int64)

    def __len__(self):
        return len(self.lookup_table)

    def __call__(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        out = arr.copy()
        valid = arr >= 0
        out[valid] = self.lookup_table[arr[valid]]
        return out


def tiny_config(**overrides) -> EngramConfig:
    base = {
        "layer_ids": (0, 2),
        "num_embeddings": (0, 0),  # replaced below
        "max_ngram_size": 3,
        "vocab_size": 1024,
        "n_heads": 2,
        "head_dim": 8,
        "pad_token_id": 0,
        "compressed_vocab_size": 512,
    }
    base.update(overrides)
    cfg = EngramConfig(**base)
    # Derive the true row counts for this toy shape so the checkpoint assertion
    # passes; the real numbers are exercised by the V4.1 test below.
    seen: set[int] = set()
    totals = []
    for _ in cfg.layer_ids:
        total = 0
        for _ in cfg.ngram_orders:
            start = cfg.vocab_size - 1
            for _ in range(cfg.n_heads):
                p = _next_prime(start, seen)
                seen.add(p)
                total += p
                start = p
        totals.append(total)
    return EngramConfig(**{**base, "num_embeddings": tuple(totals)})


def build_mapping(cfg: EngramConfig | None = None) -> NgramHashMapping:
    cfg = cfg or tiny_config()
    return NgramHashMapping(cfg, StubTokenizer(cfg.compressed_vocab_size))


def test_is_prime_matches_reference_small_cases():
    assert [n for n in range(2, 30) if _is_prime(n)] == [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
    ]
    assert not _is_prime(1)
    assert not _is_prime(0)


def test_next_prime_skips_seen():
    seen = {11, 13}
    assert _next_prime(10, seen) == 17


def test_head_vocab_sizes_match_checkpoint():
    """Derived row counts must equal DeepSeek-V4.1-Flash's declared ones."""
    cfg = EngramConfig.from_hf(V41_FLASH)
    mapping = NgramHashMapping(cfg, StubTokenizer(cfg.compressed_vocab_size))
    for layer_id, expected in zip(cfg.layer_ids, cfg.num_embeddings):
        assert int(mapping.head_vocab_sizes[layer_id].sum()) == expected


def test_head_vocab_sizes_are_globally_distinct():
    cfg = EngramConfig.from_hf(V41_FLASH)
    mapping = NgramHashMapping(cfg, StubTokenizer(cfg.compressed_vocab_size))
    every = np.concatenate([mapping.head_vocab_sizes[lid] for lid in cfg.layer_ids])
    assert len(set(every.tolist())) == every.size


def test_wrong_row_count_is_rejected():
    cfg = EngramConfig.from_hf({**V41_FLASH, "engram_num_embeddings": [1, 2]})
    with pytest.raises(ValueError, match="does not match the trained tables"):
        NgramHashMapping(cfg, StubTokenizer(cfg.compressed_vocab_size))


def test_from_hf_returns_none_without_engram():
    assert EngramConfig.from_hf({"hidden_size": 5120}) is None


def test_config_shape_helpers():
    cfg = EngramConfig.from_hf(V41_FLASH)
    assert cfg.ngram_orders == (2, 3, 4)
    assert cfg.num_hash_heads == 24


def test_hash_shape_and_determinism():
    mapping = build_mapping()
    ids = np.array([[5, 9, 13, 21], [1, 2, 3, 4]], dtype=np.int64)
    first = mapping.hash_layer(ids, layer_id=0)
    assert first.shape == (2, 4, mapping.config.num_hash_heads)
    np.testing.assert_array_equal(first, mapping.hash_layer(ids, layer_id=0))


def test_hash_is_in_range_per_head():
    mapping = build_mapping()
    ids = np.random.default_rng(0).integers(0, 512, size=(3, 16), dtype=np.int64)
    hashes = mapping.hash_layer(ids, layer_id=0)
    sizes = mapping.head_vocab_sizes[0]
    assert (hashes >= 0).all()
    assert (hashes < sizes[None, None, :]).all()


def test_layers_hash_differently():
    mapping = build_mapping()
    ids = np.array([[7, 8, 9, 10]], dtype=np.int64)
    assert not np.array_equal(mapping.hash_layer(ids, 0), mapping.hash_layer(ids, 2))


def test_hash_all_layers_agrees_with_per_layer():
    mapping = build_mapping()
    ids = np.array([[3, 4, 5, 6, 7]], dtype=np.int64)
    every = mapping.hash_all_layers(ids)
    for layer_id in mapping.config.layer_ids:
        np.testing.assert_array_equal(
            every[layer_id], mapping.hash_layer(ids, layer_id)
        )


def test_hash_is_causal_in_the_prefix():
    """Changing a later token must not disturb an earlier position's hash."""
    mapping = build_mapping()
    a = np.array([[11, 12, 13, 14]], dtype=np.int64)
    b = a.copy()
    b[0, 3] = 99
    np.testing.assert_array_equal(
        mapping.hash_layer(a, 0)[:, :3], mapping.hash_layer(b, 0)[:, :3]
    )


def test_row_indices_stay_inside_the_table():
    mapping = build_mapping()
    ids = np.random.default_rng(1).integers(0, 512, size=(2, 12), dtype=np.int64)
    rows = mapping.to_row_indices(mapping.hash_layer(ids, 0), 0)
    assert rows.min() >= 0
    assert rows.max() < int(mapping.head_vocab_sizes[0].sum())


def test_row_indices_do_not_overlap_between_heads():
    mapping = build_mapping()
    offsets = mapping.head_offsets[0]
    sizes = mapping.head_vocab_sizes[0]
    assert offsets[0] == 0
    for i in range(1, len(offsets)):
        assert offsets[i] == offsets[i - 1] + sizes[i - 1]


def test_host_table_gather_matches_direct_index():
    table = torch.arange(40 * 4, dtype=torch.float32).reshape(40, 4)
    host = HostEmbeddingTable(table, num_rows=40, head_dim=4)
    idx = np.array([[[0, 39], [7, 7]]], dtype=np.int64)
    out = host.gather(idx)
    assert out.shape == (1, 2, 2, 4)
    torch.testing.assert_close(out[0, 0, 0], table[0])
    torch.testing.assert_close(out[0, 1, 1], table[7])


def test_host_table_rejects_out_of_range():
    host = HostEmbeddingTable(torch.zeros(8, 4), num_rows=8, head_dim=4)
    with pytest.raises(IndexError, match="out of range"):
        host.gather(np.array([[[8]]], dtype=np.int64))


def test_host_table_rejects_wrong_shape():
    with pytest.raises(ValueError, match="rows"):
        HostEmbeddingTable(torch.zeros(3, 4), num_rows=8, head_dim=4)
    with pytest.raises(ValueError, match="wide"):
        HostEmbeddingTable(torch.zeros(8, 5), num_rows=8, head_dim=4)


def request(
    seq_id,
    tokens=(3,),
    *,
    position=0,
    generation=0,
    history=(-1, -1),
    token_mask=None,
):
    return EngramRequest(
        seq_id, generation, position, tuple(tokens), tuple(history), token_mask
    )


def test_cache_evicts_least_recently_used():
    cache = EngramPrefetchCache(capacity=2)
    cache.put(request(1), 0, torch.zeros(1))
    cache.put(request(2), 0, torch.zeros(1))
    cache.put(request(3), 0, torch.zeros(1))
    assert cache.take(request(1), 0) is None
    assert cache.take(request(3), 0) is not None


def test_cache_take_is_destructive():
    cache = EngramPrefetchCache()
    cache.put(request(4), 1, torch.zeros(1))
    assert cache.take(request(4), 1) is not None
    assert cache.take(request(4), 1) is None


def test_cache_drop_forgets_every_layer_of_a_request():
    cache = EngramPrefetchCache()
    cache.put(request(5), 0, torch.zeros(1))
    cache.put(request(5), 2, torch.zeros(1))
    cache.put(request(6), 0, torch.zeros(1))
    cache.drop(5)
    assert cache.take(request(5), 0) is None and cache.take(request(5), 2) is None
    assert cache.take(request(6), 0) is not None


def make_prefetcher() -> EngramPrefetcher:
    mapping = build_mapping()
    tables = {
        layer_id: HostEmbeddingTable(
            torch.arange(
                int(mapping.head_vocab_sizes[layer_id].sum()) * 8, dtype=torch.float32
            ).reshape(-1, 8),
            num_rows=int(mapping.head_vocab_sizes[layer_id].sum()),
            head_dim=8,
        )
        for layer_id in mapping.config.layer_ids
    }
    return EngramPrefetcher(mapping, tables)


def test_prefetch_result_equals_inline_compute():
    pf = make_prefetcher()
    seq_ids = [11, 12]
    ids = np.array([[3, 4, 5], [6, 7, 8]], dtype=np.int64)
    requests = [request(seq, tokens) for seq, tokens in zip(seq_ids, ids)]
    expected = pf.compute(requests)
    assert pf.submit_compute(requests).result(timeout=30) is None
    assert pf.wait(timeout=30)
    for (seq_id, layer_id), value in expected.items():
        torch.testing.assert_close(pf.cache.take(seq_id, layer_id), value)
    pf.shutdown()


def test_prefetch_drop_requests_clears_cache():
    pf = make_prefetcher()
    pf.submit_compute([request(21, (1, 2, 3))]).result(timeout=30)
    pf.drop_requests([21])
    assert len(pf.cache) == 0
    pf.shutdown()


def test_wait_without_submit_is_a_noop():
    pf = make_prefetcher()
    assert pf.wait(timeout=1)
    pf.shutdown()


# --- device-side modules and the staging runtime ---


def make_op(hidden=16, engram_hidden=24, hc=2):
    """Unquantized, so the projection is BF16 -- the dtype `EngramHost` gathers
    its rows into, and therefore what the embeddings below are.

    The op is an ATOM layer and reaches AITER through it, so it is imported
    here rather than at module scope: everything above this line is host
    arithmetic that a CPU-only runner can and should still check.
    """
    pytest.importorskip("aiter", reason="EngramOp builds an AITER-backed layer")
    from atom.model_ops.engram.device.layer import EngramOp

    op = EngramOp(
        layer_id=1, hidden_size=hidden, engram_hidden_size=engram_hidden, hc_mult=hc
    )
    op.process_weights_after_loading()
    return op


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_engram_op_is_token_flat():
    """ATOM residual streams are [num_tokens, hc, dim], not [batch, seq, ...]."""
    op = make_op().cuda()
    out = op(
        torch.randn(5, 2, 16).cuda(),
        torch.randn(5, 24, dtype=torch.bfloat16).cuda(),
    )
    assert out.shape == (5, 2, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_engram_op_accepts_leading_batch_dims():
    op = make_op().cuda()
    out = op(
        torch.randn(2, 3, 2, 16).cuda(),
        torch.randn(2, 3, 24, dtype=torch.bfloat16).cuda(),
    )
    assert out.shape == (2, 3, 2, 16)


def test_engram_op_rejects_wrong_embedding_width():
    with pytest.raises(ValueError, match="expected 24"):
        make_op()(torch.randn(2, 2, 16), torch.randn(2, 25))


def test_engram_op_rejects_wrong_branch_count():
    with pytest.raises(ValueError, match="hc_mult=2"):
        make_op()(torch.randn(2, 3, 16), torch.randn(2, 24))


def test_engram_op_rejects_token_count_mismatch():
    with pytest.raises(ValueError, match="tokens"):
        make_op()(torch.randn(5, 2, 16), torch.randn(4, 24))


def test_engram_op_has_no_short_conv():
    """V4.1 omits the short causal convolution (tech report section 2.4.2)."""
    assert not any("short_conv" in n for n, _ in make_op().named_modules())


def test_wkv_is_fused_keys_then_value():
    """The checkpoint stores hc_mult key projections first, then one value."""
    op = make_op(hidden=16, hc=2)
    assert op.wkv.weight.shape == (3 * 16, 24)
    assert op.key_rows == 2 * 16


def test_load_checkpoint_weights_validates_shapes():
    op = make_op(hidden=16, engram_hidden=24, hc=2)
    good = torch.zeros(3 * 16, 24)
    with pytest.raises(ValueError, match="wkv is"):
        op.load_checkpoint_weights(
            torch.zeros(24, 48), torch.zeros(2, 16), torch.zeros(2, 16)
        )
    with pytest.raises(ValueError, match="k_weight is"):
        op.load_checkpoint_weights(good, torch.zeros(3, 16), torch.zeros(2, 16))
    with pytest.raises(ValueError, match="q_weight is"):
        op.load_checkpoint_weights(good, torch.zeros(2, 16), torch.zeros(2, 8))


def test_load_checkpoint_weights_dequantizes_blocks():
    op = make_op(hidden=16, engram_hidden=32, hc=2)
    rows, cols = 3 * 16, 32
    wkv = torch.ones(rows, cols)
    scale = torch.full((rows // 8, cols // 8), 3.0)
    op.load_checkpoint_weights(
        wkv, torch.ones(2, 16), torch.ones(2, 16), wkv_scale=scale, block=8
    )
    torch.testing.assert_close(
        op.wkv.weight, torch.full((rows, cols), 3.0, dtype=torch.bfloat16)
    )


def test_load_checkpoint_weights_rejects_bad_scale_shape():
    op = make_op(hidden=16, engram_hidden=32, hc=2)
    with pytest.raises(ValueError, match="wkv scale is"):
        op.load_checkpoint_weights(
            torch.ones(48, 32),
            torch.ones(2, 16),
            torch.ones(2, 16),
            wkv_scale=torch.ones(2, 2),
            block=8,
        )


def make_runtime() -> EngramHost:
    pf = make_prefetcher()
    cfg = pf._hash_mapping.config
    return EngramHost(
        pf,
        max_num_tokens=8,
        num_hash_heads=cfg.num_hash_heads,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_runtime_stage_uses_prefetched_rows():
    rt = make_runtime()
    seq_ids = [31, 32]
    tokens = np.array([[5], [6]], dtype=np.int64)
    requests = [request(seq, ids) for seq, ids in zip(seq_ids, tokens)]
    rt.prefetch(requests).result(timeout=30)
    assert rt.stage_embeddings(requests) == 2
    for layer_id in rt.layer_ids:
        assert rt.embeddings(layer_id).shape == (2, rt.embed_width)
    rt.shutdown()


def test_runtime_stage_recomputes_on_prefetch_miss():
    """A missed prefetch changes latency, never the answer."""
    rt = make_runtime()
    seq_ids = [41]
    tokens = np.array([[9]], dtype=np.int64)
    requests = [request(seq, ids) for seq, ids in zip(seq_ids, tokens)]
    rt.prefetch(requests).result(timeout=30)
    rt.stage_embeddings(requests)
    warm = {lid: rt.embeddings(lid).clone() for lid in rt.layer_ids}

    cold = make_runtime()
    cold.stage_embeddings(requests)  # nothing prefetched
    for layer_id in cold.layer_ids:
        torch.testing.assert_close(cold.embeddings(layer_id), warm[layer_id])
    rt.shutdown()
    cold.shutdown()


def test_runtime_rejects_empty_lookup_snapshot():
    with pytest.raises(ValueError, match="needs tokens"):
        request(51, ())


def test_runtime_rejects_more_rows_than_capacity():
    rt = make_runtime()
    ids = list(range(9))
    with pytest.raises(ValueError, match="exceeds staging capacity"):
        rt.stage_embeddings([request(seq) for seq in ids])
    rt.shutdown()


def test_host_table_applies_block_scales():
    """A block-quantized table without its scales is off by orders of magnitude."""
    weight = torch.ones(4, 8)
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0], [1.0, 1.0], [2.0, 2.0]])
    host = HostEmbeddingTable(weight, num_rows=4, head_dim=8, scale=scale)
    assert host.block_size == 4
    out = host.gather(np.array([[[1]]], dtype=np.int64))[0, 0, 0]
    torch.testing.assert_close(out, torch.tensor([4.0] * 4 + [8.0] * 4))


def test_host_table_without_scale_is_unscaled():
    host = HostEmbeddingTable(torch.ones(4, 8), num_rows=4, head_dim=8)
    assert host.block_size == 0
    out = host.gather(np.array([[[1]]], dtype=np.int64))[0, 0, 0]
    torch.testing.assert_close(out, torch.ones(8))


def test_host_table_rejects_mismatched_scale():
    with pytest.raises(ValueError, match="scale has"):
        HostEmbeddingTable(torch.ones(4, 8), 4, 8, scale=torch.ones(3, 2))
    with pytest.raises(ValueError, match="not divisible"):
        HostEmbeddingTable(torch.ones(4, 8), 4, 8, scale=torch.ones(4, 3))


def test_ragged_prefill_stages_all_history_aware_rows_and_padding():
    rt = make_runtime()
    requests = [
        request(71, (5, 6, 7), position=3, history=(3, 4)),
        request(72, (9, 10), position=4, history=(-1, 8), token_mask=(False, True)),
    ]
    rt.prefetch(requests).result(timeout=30)
    assert rt.stage_embeddings(requests, padded_rows=8) == 8
    mapping = rt.prefetcher._hash_mapping
    for layer in rt.layer_ids:
        # Compute hashes over the whole prefix independently of snapshot history.
        expected = []
        for req in requests:
            ids = np.array([req.history + req.token_ids])
            mask = np.array(
                [
                    [True] * len(req.history)
                    + list(req.token_mask or (True,) * len(req.token_ids))
                ]
            )
            hashes = mapping.hash_layer(ids, layer, compress=False, token_mask=mask)[
                :, len(req.history) :
            ]
            rows = mapping.to_row_indices(hashes, layer)
            expected.append(
                rt.prefetcher._tables[layer].gather(rows).reshape(-1, rt.embed_width)
            )
        torch.testing.assert_close(rt.embeddings(layer)[:5], torch.cat(expected))
        assert torch.count_nonzero(rt.embeddings(layer)[5:]) == 0
    rt.shutdown()


def test_cache_identity_covers_history_position_generation_and_tentative_tokens():
    from dataclasses import replace

    cache = EngramPrefetchCache()
    original = request(7)
    for changed in (
        replace(original, history=(1, 2, 3)),
        replace(original, position=1),
        replace(original, generation=1),
        replace(original, token_ids=(4,)),
        replace(original, token_mask=(False,)),
    ):
        cache.put(original, 1, torch.ones(1))
        assert cache.take(changed, 1) is None
        assert cache.take(original, 1) is not None


def test_drop_during_prefetch_cannot_repopulate_recycled_request(monkeypatch):
    import threading

    pf = make_prefetcher()
    entered, release = threading.Event(), threading.Event()
    compute = pf.compute

    def delayed(requests):
        entered.set()
        assert release.wait(30)
        return compute(requests)

    monkeypatch.setattr(pf, "compute", delayed)
    future = pf.submit_compute([request(7)])
    try:
        assert entered.wait(30)
        pf.drop_requests([7])
        release.set()
        future.result(timeout=30)
        assert len(pf.cache) == 0
    finally:
        release.set()
        pf.shutdown()


def test_staging_rejects_padding_that_truncates_or_exceeds_capacity():
    rt = make_runtime()
    for padded in (2, 9):
        with pytest.raises(ValueError, match="staging capacity"):
            rt.stage_embeddings([request(7, (1, 2, 3))], padded_rows=padded)
    rt.shutdown()


class FakeCudart:
    """Page-lock accounting, and a scripted refusal.

    `cudaHostUnregister` removes from the same list registration appends to, so
    giving back a pointer that was never taken raises here rather than being
    counted as a release.
    """

    def __init__(self, refuse_call=None):
        self.refuse_call = refuse_call
        self.registered = []
        self.calls = 0

    def cudaHostRegister(self, ptr, size, flags):
        self.calls += 1
        if self.calls == self.refuse_call:
            return 1
        self.registered.append(ptr)
        return 0

    def cudaHostUnregister(self, ptr):
        self.registered.remove(ptr)
        return 0


def scaled_table() -> HostEmbeddingTable:
    """A table whose values and scales are two separate registrations."""
    scale = torch.ones(4, 2, dtype=torch.float8_e8m0fnu)
    return HostEmbeddingTable(torch.ones(4, 8), num_rows=4, head_dim=8, scale=scale)


def test_refused_scale_registration_gives_back_the_values_it_pinned(monkeypatch):
    """The failing half must not leave the succeeding half locked.

    Values and scales are registered in turn; without the rollback the first
    call's pages stay unswappable for the life of the process, and
    `disable_uva` cannot find them because the range was never recorded.
    """
    cudart = FakeCudart(refuse_call=2)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    table = scaled_table()
    assert table.enable_uva(0, 4) is False
    assert cudart.registered == []
    assert table._uva is None


def test_disable_uva_returns_every_registration(monkeypatch):
    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    table = scaled_table()
    assert table.enable_uva(0, 4) is True
    assert len(cudart.registered) == 2
    table.disable_uva()
    assert cudart.registered == []
    assert table._uva is None
    # Idempotent: a second release has nothing to give back and must not try.
    table.disable_uva()
    assert cudart.registered == []


def test_enabling_the_same_range_twice_registers_once(monkeypatch):
    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    table = scaled_table()
    assert table.enable_uva(0, 4) is True
    assert table.enable_uva(0, 4) is True
    assert cudart.calls == 2


class FakeGroup:
    def __init__(self, world_size, rank_in_group=0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.cpu_group = object()


def patch_tp_group(monkeypatch, group):
    """Serve `from aiter.dist.parallel_state import get_tp_group` a stub.

    Undone at teardown by `monkeypatch.setitem`, which matters: an `aiter` left
    in `sys.modules` makes every later `importorskip("aiter")` succeed.
    """
    import sys
    from types import ModuleType

    parallel_state = ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: group
    for name, module in (
        ("aiter", ModuleType("aiter")),
        ("aiter.dist", ModuleType("aiter.dist")),
        ("aiter.dist.parallel_state", parallel_state),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_uva_is_the_groups_decision_not_this_ranks(monkeypatch):
    """A rank outvoted after registering falls back AND releases.

    The device lookup ends in an all-gather. A rank that kept the UVA path
    while a peer took the host path would enter a collective the peer never
    joins, so one refusal has to turn the whole group around.
    """
    from atom.model_ops.engram import host as engram_host

    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    patch_tp_group(monkeypatch, FakeGroup(world_size=2))

    def refused(tensor, op=None, group=None):
        tensor.fill_(0)

    monkeypatch.setattr(engram_host.dist, "all_reduce", refused)
    rt = make_runtime()
    assert (
        rt._enable_uva(rt.prefetcher, rt.prefetcher._hash_mapping.config.num_hash_heads)
        is False
    )
    assert cudart.registered == []
    assert rt._pinned_tables == []
    rt.shutdown()


def test_an_empty_shard_registers_nothing(monkeypatch):
    """Heads that do not divide leave a tail rank with none of them.

    That rank is as unable to take the UVA path as one whose registration was
    refused, and for the collective downstream the two are the same event.
    """
    from atom.model_ops.engram import host as engram_host

    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    # Every peer registered, so this rank's own answer is the only thing that
    # can turn the group around.
    monkeypatch.setattr(engram_host.dist, "all_reduce", lambda *a, **k: None)
    rt = make_runtime()
    heads = rt.prefetcher._hash_mapping.config.num_hash_heads
    patch_tp_group(
        monkeypatch, FakeGroup(world_size=heads - 1, rank_in_group=heads - 2)
    )
    assert rt._enable_uva(rt.prefetcher, heads) is False
    assert rt.local_heads == 0
    assert cudart.registered == []
    rt.shutdown()


def test_uva_survives_a_group_that_agrees(monkeypatch):
    from atom.model_ops.engram import host as engram_host

    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    patch_tp_group(monkeypatch, FakeGroup(world_size=2))
    monkeypatch.setattr(engram_host.dist, "all_reduce", lambda *a, **k: None)
    rt = make_runtime()
    heads = rt.prefetcher._hash_mapping.config.num_hash_heads
    assert rt._enable_uva(rt.prefetcher, heads) is True
    assert len(cudart.registered) == len(rt._pinned_tables)
    # Shutdown is the other place these pages can leak: the mapping they belong
    # to is closed right after it.
    rt.shutdown()
    assert cudart.registered == []
