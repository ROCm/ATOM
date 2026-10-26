"""Unit tests for the engram host path.

The load-bearing test is `test_head_vocab_sizes_match_checkpoint`: the derived
per-layer row counts have to equal what DeepSeek-V4.1-Flash's config declares,
which only happens if the prime search runs in exactly the reference order.
"""

from concurrent.futures import ThreadPoolExecutor

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


@pytest.fixture(autouse=True)
def _no_real_host_allocator(monkeypatch):
    """Keep every test off the real `hipHostMalloc`: the registration tests
    script `cudaHostRegister`, and the copy tests install `FakeHostAllocator`
    themselves."""
    from atom.model_ops.engram import host_memory

    monkeypatch.setattr(host_memory, "_host_allocator", lambda: None)


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


class FakeHostAllocator:
    """Page-locked allocations backed by ctypes buffers kept alive here."""

    def __init__(self, refuse_call=None):
        self.refuse_call = refuse_call
        self.live = {}
        self.calls = 0

    def alloc(self, nbytes):
        import ctypes

        self.calls += 1
        if self.calls == self.refuse_call:
            return None
        buf = ctypes.create_string_buffer(nbytes)
        self.live[ctypes.addressof(buf)] = buf
        return ctypes.addressof(buf)

    def free(self, ptr):
        del self.live[ptr]


def test_copied_shard_holds_exactly_the_rows_the_kernel_addresses(monkeypatch):
    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    weight = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
    scale = torch.full((6, 2), 2.0).to(torch.float8_e8m0fnu)
    table = HostEmbeddingTable(weight, num_rows=6, head_dim=8, scale=scale)
    assert table.enable_uva(2, 5) is True
    got_w, got_s, start, end = table.registered_shard()
    assert (start, end) == (2, 5)
    assert torch.equal(got_w, weight[2:5])
    assert torch.equal(got_s, scale[2:5].view(torch.uint8))
    # Copies, not views of the checkpoint mapping.
    assert got_w.data_ptr() != weight[2:5].data_ptr()
    assert len(allocator.live) == 2
    del got_w, got_s
    table.disable_uva()
    assert allocator.live == {} and table._uva is None
    table.disable_uva()


def test_a_held_shard_tensor_keeps_its_memory_until_dropped(monkeypatch):
    """The tensors `registered_shard` hands out own the page-locked memory:
    `disable_uva` cannot free memory a caller can still read."""
    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    table = scaled_table()
    assert table.enable_uva(0, 4) is True
    weight = table.registered_shard()[0]
    table.disable_uva()
    # The scales had no other holder; the values are still referenced.
    assert list(allocator.live) == [weight.data_ptr()]
    assert torch.equal(weight, torch.ones(4, 8))
    del weight
    assert allocator.live == {}


def test_a_table_dropped_without_disable_frees_its_shard(monkeypatch):
    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    table = scaled_table()
    assert table.enable_uva(0, 4) is True
    assert len(allocator.live) == 2
    del table
    assert allocator.live == {}


def test_a_held_registration_stays_registered_until_dropped(monkeypatch):
    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    table = scaled_table()
    assert table.enable_uva(0, 4) is True
    weight = table.registered_shard()[0]
    table.disable_uva()
    assert len(cudart.registered) == 1
    del weight
    assert cudart.registered == []


def test_enabling_a_different_range_while_one_is_held_raises(monkeypatch):
    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    table = scaled_table()
    assert table.enable_uva(0, 4) is True
    with pytest.raises(RuntimeError, match="disable_uva"):
        table.enable_uva(1, 3)
    assert table.registered_shard()[2:] == (0, 4)
    table.disable_uva()
    assert table.enable_uva(1, 3) is True
    assert table.registered_shard()[2:] == (1, 3)
    assert allocator.calls == 4


def test_a_release_during_graph_capture_is_skipped(monkeypatch):
    """Freeing synchronizes the device, which a capture forbids."""
    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    table = scaled_table()
    assert table.enable_uva(0, 4) is True
    monkeypatch.setattr(host_memory, "_capturing", lambda: True)
    table.disable_uva()
    assert len(allocator.live) == 2
    # Retained, not merely skipped: nothing frees it later either.
    import gc

    gc.collect()
    assert len(allocator.live) == 2
    assert len(host_memory._RETAINED) >= 2
    host_memory._RETAINED.clear()


def test_a_registration_released_during_capture_keeps_its_backing(monkeypatch):
    """In place, the region is the table's own storage: it must outlive any
    graph captured over it, so the lease keeps the storage too."""
    import gc

    from atom.model_ops.engram import host_memory

    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    weight = torch.ones(4, 8)
    address = weight.data_ptr()
    table = HostEmbeddingTable(weight, num_rows=4, head_dim=8)
    assert table.enable_uva(0, 4) is True
    monkeypatch.setattr(host_memory, "_capturing", lambda: True)
    del table, weight
    gc.collect()
    assert len(cudart.registered) == 1
    kept = [lease._keep for lease in host_memory._RETAINED]
    assert any(k is not None and k.data_ptr() == address for k in kept)
    host_memory._RETAINED.clear()


def test_a_copy_leaves_an_anonymous_source_intact(monkeypatch):
    """Dropping copied source pages is for checkpoint (file) mappings only: on
    anonymous memory it would discard the table."""
    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    monkeypatch.setattr(host_memory, "_BYTES_PER_THREAD", 1 << 14)
    monkeypatch.setattr(host_memory.os, "sched_getaffinity", lambda pid: set(range(4)))
    weight = torch.randn(4096, 64)
    expected = weight.clone()
    assert host_memory._source_dropper(weight) is None
    table = HostEmbeddingTable(weight, num_rows=4096, head_dim=64)
    assert table.enable_uva(0, 4096) is True
    assert torch.equal(weight, expected)


def _resident_pages(tensor):
    """Pages of `tensor` mapped in this process (`/proc/self/pagemap`)."""
    import os

    import numpy as np

    lo, hi = tensor.data_ptr() // 4096, -(-(tensor.data_ptr() + tensor.nbytes) // 4096)
    fd = os.open("/proc/self/pagemap", os.O_RDONLY)
    try:
        e = np.frombuffer(os.pread(fd, 8 * (hi - lo), 8 * lo), dtype="<u8")
    finally:
        os.close(fd)
    return int(np.count_nonzero(e >> np.uint64(63)))


def test_a_copied_file_mapping_is_dropped_and_still_reads_back(monkeypatch, tmp_path):
    import numpy as np

    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    data = np.random.default_rng(0).standard_normal((4096, 64)).astype(np.float32)
    path = tmp_path / "table.bin"
    data.tofile(path)
    mapped = torch.from_numpy(np.memmap(path, np.float32, "c", shape=data.shape))
    dropped = []
    real = host_memory._source_dropper

    def watched(src):
        drop = real(src)
        assert drop is not None  # a file mapping

        def record(offset, nbytes):
            dropped.append((offset, nbytes))
            drop(offset, nbytes)

        record.close = drop.close
        return record

    monkeypatch.setattr(host_memory, "_source_dropper", watched)
    # Small chunks, several threads: exercise the chunked copy and drops.
    monkeypatch.setattr(host_memory, "_BYTES_PER_THREAD", 1 << 14)
    monkeypatch.setattr(host_memory, "_CHUNK_BYTES", 1 << 14)
    monkeypatch.setattr(host_memory.os, "sched_getaffinity", lambda pid: set(range(4)))
    table = HostEmbeddingTable(mapped, num_rows=4096, head_dim=64)
    assert table.enable_uva(0, 4096) is True
    assert len(dropped) > 2 and sum(n for _, n in dropped) == data.nbytes
    # Every chunk dropped; fault-around while copying a later chunk can map a
    # few clean pages of an earlier one back in (page cache, harmless).
    assert _resident_pages(mapped) < data.nbytes // 4096 // 8
    assert torch.equal(table.registered_shard()[0], torch.from_numpy(data))
    assert torch.equal(mapped, torch.from_numpy(data))  # faulted back in


def test_the_allocator_disables_huge_pages_only_around_the_allocation():
    """hipHostMalloc on huge pages compacts memory once the page cache has
    fragmented it; the allocation runs with THP off and restores the
    process's own setting."""
    from types import SimpleNamespace

    from atom.model_ops.engram import host_memory as hm

    calls, state = [], {"thp_off": 0}

    def prctl(op, arg, *rest):
        calls.append((op, arg))
        if op == hm._PR_GET_THP_DISABLE:
            return state["thp_off"]
        state["thp_off"] = arg
        return 0

    def malloc(ptr, nbytes, flags):
        calls.append(("malloc", state["thp_off"]))
        ptr._obj.value = 4096
        return 0

    # Plain functions: the allocator sets `argtypes` on them, as on ctypes'.
    libc = SimpleNamespace(prctl=prctl)
    hip = SimpleNamespace(hipHostMalloc=malloc, hipHostFree=lambda ptr: 0)
    alloc = hm._HipHostAllocator(hip, libc)
    assert alloc.alloc(1 << 20) == 4096
    assert ("malloc", 1) in calls and state["thp_off"] == 0
    state["thp_off"] = 1  # a process that disabled THP itself keeps it off
    calls.clear()
    alloc.alloc(1 << 20)
    assert state["thp_off"] == 1 and (hm._PR_SET_THP_DISABLE, 0) not in calls


def test_a_written_private_mapping_is_not_dropped(monkeypatch, tmp_path):
    """A copy-on-write mapping that was modified holds the changes in
    anonymous pages; dropping them would put the file's bytes back under the
    table, so the host fallback would read different values."""
    import numpy as np

    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    data = np.zeros((4096, 64), np.float32)
    path = tmp_path / "table.bin"
    data.tofile(path)
    mapped = np.memmap(path, np.float32, "c", shape=data.shape)
    mapped[100:4000] = 7.0  # private copies, not in the file
    weight = torch.from_numpy(mapped)
    table = HostEmbeddingTable(weight, num_rows=4096, head_dim=64)
    assert table.enable_uva(0, 4096) is True
    assert float(weight[2000, 0]) == 7.0
    assert float(weight[99, 0]) == 0.0
    assert torch.equal(table.registered_shard()[0], weight)


def test_a_mapping_split_into_clean_and_written_parts_is_not_dropped(
    monkeypatch, tmp_path
):
    """`mprotect` splits one file mapping in two; only the second was written
    (and the first is now read-only). Written pages are kept wherever they
    are; clean ones are still dropped."""
    import ctypes

    import numpy as np

    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    rows, width = 4096, 64  # 1 MiB of float32, page-aligned halves
    path = tmp_path / "table.bin"
    np.zeros((rows, width), np.float32).tofile(path)
    mapped = np.memmap(path, np.float32, "c", shape=(rows, width))
    mapped[rows // 2 + 10 :] = 7.0  # written: anonymous pages, second half
    weight = torch.from_numpy(mapped)
    libc = ctypes.CDLL(None)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    half = rows * width * 4 // 2
    assert libc.mprotect(weight.data_ptr(), half, 1) == 0  # PROT_READ: a split
    first = weight[: rows // 2]
    first.sum()  # fault the clean half in
    monkeypatch.setattr(host_memory, "_BYTES_PER_THREAD", 1 << 14)
    monkeypatch.setattr(host_memory, "_CHUNK_BYTES", 1 << 14)
    monkeypatch.setattr(host_memory.os, "sched_getaffinity", lambda pid: set(range(4)))
    table = HostEmbeddingTable(weight, num_rows=rows, head_dim=width)
    assert table.enable_uva(0, rows) is True
    assert float(weight[rows - 1, 0]) == 7.0
    assert float(weight[rows // 2 + 10, 0]) == 7.0
    assert _resident_pages(first) < rows * width * 4 // 2 // 4096 // 8
    assert torch.equal(table.registered_shard()[0], weight)
    del table
    assert libc.mprotect(weight.data_ptr(), half, 3) == 0


def test_one_copy_worker_still_drops_the_source_chunk_by_chunk(monkeypatch, tmp_path):
    import numpy as np

    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    monkeypatch.setattr(host_memory.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(host_memory, "_CHUNK_BYTES", 1 << 14)
    data = np.random.default_rng(1).standard_normal((4096, 64)).astype(np.float32)
    path = tmp_path / "table.bin"
    data.tofile(path)
    mapped = torch.from_numpy(np.memmap(path, np.float32, "c", shape=data.shape))
    dropped = []
    real = host_memory._source_dropper

    def watched(src):
        drop = real(src)

        def record(o, n):
            dropped.append((o, n))
            drop(o, n)

        record.close = drop.close
        return record

    monkeypatch.setattr(host_memory, "_source_dropper", watched)
    table = HostEmbeddingTable(mapped, num_rows=4096, head_dim=64)
    assert table.enable_uva(0, 4096) is True
    assert len(dropped) == data.nbytes // (1 << 14)  # page-aligned map
    assert _resident_pages(mapped) < data.nbytes // 4096 // 8
    assert torch.equal(table.registered_shard()[0], torch.from_numpy(data))


def test_a_failing_chunk_waits_for_the_others_before_the_dropper_closes(
    monkeypatch,
):
    """One chunk raising must not let the caller close the dropper (and its
    pagemap descriptor) while other chunks are still dropping."""
    import threading
    import time

    from atom.model_ops.engram import host_memory as hm

    monkeypatch.setattr(hm, "_BYTES_PER_THREAD", 1 << 14)
    monkeypatch.setattr(hm, "_CHUNK_BYTES", 1 << 14)
    monkeypatch.setattr(hm.os, "sched_getaffinity", lambda pid: set(range(4)))
    src = torch.zeros(1 << 18, dtype=torch.uint8)
    dst = torch.empty_like(src)
    closed, late, lock = [], [], threading.Lock()

    def drop(offset, nbytes):
        if offset == 0:
            raise RuntimeError("chunk failed")
        time.sleep(0.01)
        with lock:
            if closed:
                late.append(offset)

    with ThreadPoolExecutor(4) as pool:
        with pytest.raises(RuntimeError, match="chunk failed"):
            hm._copy_into(dst, src, pool, drop)
        closed.append(True)  # what `_copy` does next
    assert not late


def test_a_failed_worker_start_drains_queued_chunks(monkeypatch):
    """`submit` that queues its chunk and then fails to start a thread returns
    no future; the queued chunk must still be done or cancelled before the
    copy returns, or it would write `dst` and read the dropper afterwards."""
    import threading
    import time

    from atom.model_ops.engram import host_memory as hm

    monkeypatch.setattr(hm, "_BYTES_PER_THREAD", 1 << 14)
    monkeypatch.setattr(hm, "_CHUNK_BYTES", 1 << 14)
    monkeypatch.setattr(hm.os, "sched_getaffinity", lambda pid: set(range(4)))
    src = torch.ones(1 << 18, dtype=torch.uint8)
    dst = torch.zeros_like(src)
    running, after = [], []
    done = threading.Event()

    def drop(offset, nbytes):
        running.append(offset)
        time.sleep(0.005)
        if done.is_set():
            after.append(offset)

    pool = ThreadPoolExecutor(1)
    real_submit = pool.submit
    calls = [0]

    def submit(fn, *args):
        calls[0] += 1
        future = real_submit(fn, *args)  # queued
        if calls[0] == 3:
            raise RuntimeError("can't start new thread")
        return future

    pool.submit = submit
    with pytest.raises(RuntimeError, match="new thread"):
        hm._copy_into(dst, src, pool, drop)
    done.set()
    time.sleep(0.05)
    assert running and not after  # queued chunks ran or were cancelled, not later


def test_closing_a_dropper_waits_for_a_drop_in_progress(tmp_path, monkeypatch):
    import threading
    import time

    import numpy as np

    from atom.model_ops.engram import host_memory as hm

    path = tmp_path / "t.bin"
    np.zeros(1 << 16, np.float32).tofile(path)
    src = torch.from_numpy(np.memmap(path, np.float32, "c", shape=(1 << 16,)))
    src.sum()
    real_pread, closed_fds, reads = hm.os.pread, [], []
    real_close = hm.os.close

    def slow_pread(fd, n, off):
        time.sleep(0.05)
        reads.append(fd in closed_fds)
        return real_pread(fd, n, off)

    def close(fd):
        closed_fds.append(fd)
        real_close(fd)

    monkeypatch.setattr(hm.os, "pread", slow_pread)
    monkeypatch.setattr(hm.os, "close", close)
    drop = hm._source_dropper(src)
    worker = threading.Thread(target=drop, args=(0, src.nbytes))
    worker.start()
    time.sleep(0.01)
    drop.close()
    worker.join()
    assert reads and not any(reads)


def test_a_closed_dropper_does_nothing(tmp_path):
    import numpy as np

    from atom.model_ops.engram import host_memory as hm

    path = tmp_path / "t.bin"
    np.zeros(1 << 16, np.float32).tofile(path)
    src = torch.from_numpy(np.memmap(path, np.float32, "c", shape=(1 << 16,)))
    drop = hm._source_dropper(src)
    drop.close()
    drop.close()  # idempotent
    drop(0, src.nbytes)  # no read of a reused descriptor


def test_chunks_cut_at_page_boundaries():
    import itertools

    from atom.model_ops.engram.host_memory import _PAGE, _chunks

    for addr in (0, 100, _PAGE - 1, 5 * _PAGE + 7):
        for nbytes in (1, _PAGE, 10 * _PAGE + 3, 1 << 20):
            pieces = _chunks(addr, nbytes, 3 * _PAGE)
            assert pieces[0][0] == 0 and sum(n for _, n in pieces) == nbytes
            assert all(a + n == b for (a, n), (b, _) in itertools.pairwise(pieces))
            assert all((addr + o) % _PAGE == 0 for o, _ in pieces[1:])


def test_copy_threads_scale_with_size_and_respect_cpu_affinity(monkeypatch):
    from atom.model_ops.engram import host_memory as hm

    monkeypatch.setattr(hm.os, "sched_getaffinity", lambda pid: set(range(64)))
    # The documented policy: one copy_ below 512 MiB, then 8..16 threads.
    assert hm.copy_threads((128 << 20) - 1) == 1
    assert hm.copy_threads(256 << 20) == 1  # two threads lose to one copy_
    assert hm.copy_threads((512 << 20) - 1) == 1
    assert hm.copy_threads(512 << 20) == 8
    assert hm.copy_threads(1 << 30) == 8
    assert hm.copy_threads(3 << 30) == 16
    assert hm.copy_threads(1 << 40) == hm._MAX_COPY_THREADS
    monkeypatch.setattr(hm.os, "sched_getaffinity", lambda pid: {0, 1, 2})
    assert hm.copy_threads(1 << 40) == 3


def test_a_threaded_copy_is_exact(monkeypatch):
    from atom.model_ops.engram import host_memory as hm

    allocator = FakeHostAllocator()
    monkeypatch.setattr(hm, "_host_allocator", lambda: allocator)
    monkeypatch.setattr(hm, "_BYTES_PER_THREAD", 1000)
    monkeypatch.setattr(hm.os, "sched_getaffinity", lambda pid: set(range(8)))
    weight = torch.randn(64, 128)
    table = HostEmbeddingTable(weight, num_rows=64, head_dim=128)
    assert table.enable_uva(3, 61) is True
    assert torch.equal(table.registered_shard()[0], weight[3:61])


def test_refused_scale_copy_frees_the_values_it_took(monkeypatch):
    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator(refuse_call=2)
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)
    table = scaled_table()
    assert table.enable_uva(0, 4) is False
    assert allocator.live == {} and table._uva is None


def test_a_copy_that_fails_frees_what_it_took(monkeypatch):
    """No thread to copy on (or any copy error): every allocation goes back
    and the table reports no shard, so the group can fall back."""
    from atom.model_ops.engram import host_memory

    allocator = FakeHostAllocator()
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: allocator)

    raised = []

    def no_thread(dst, src, pool, drop=None):
        raised.append(True)
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(host_memory, "_copy_into", no_thread)
    table = scaled_table()
    assert table.enable_uva(0, 4) is False
    assert allocator.live == {} and table._uva is None
    assert raised  # the scripted failure, not an argument mismatch
    table.disable_uva()


def test_without_a_host_allocator_the_mapping_is_registered_in_place(monkeypatch):
    from atom.model_ops.engram import host_memory

    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    monkeypatch.setattr(host_memory, "_host_allocator", lambda: None)
    table = scaled_table()
    assert table.enable_uva(0, 4) is True
    assert len(cudart.registered) == 2
    weight, _, _, _ = table.registered_shard()
    assert weight.data_ptr() == table._tensor.data_ptr()


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


def test_a_table_that_raises_votes_no_instead_of_leaving_the_vote(monkeypatch):
    """An exception from one table's `enable_uva` (a table already holding
    another range, a bad range) must reach the group vote as a refusal: a
    rank that raised before the all-reduce would leave its peers in it."""
    from atom.model_ops.engram import host as engram_host

    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    patch_tp_group(monkeypatch, FakeGroup(world_size=2))
    votes = []

    def vote(tensor, op=None, group=None):
        votes.append(int(tensor))

    monkeypatch.setattr(engram_host.dist, "all_reduce", vote)
    rt = make_runtime()
    tables = list(rt.prefetcher._tables.values())
    tables[-1].enable_uva(0, 1)  # holds another range: its enable_uva raises
    heads = rt.prefetcher._hash_mapping.config.num_hash_heads
    assert rt._enable_uva(rt.prefetcher, heads) is False
    assert votes == [0]
    assert rt._pinned_tables == []
    tables[-1].disable_uva()
    assert cudart.registered == []
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
