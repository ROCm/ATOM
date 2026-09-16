# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the engram host path. The load-bearing check is in
`test_primes_and_head_vocab_sizes`: the derived per-layer row counts must equal
DeepSeek-V4.1-Flash's declared ones, which needs the exact reference prime order.
"""

import numpy as np
import pytest
import torch

from atom.model_ops.engram import (
    EngramConfig,
    EngramHost,
    EngramPrefetchCache,
    EngramPrefetcher,
    HostEmbeddingTable,
    NgramHashMapping,
    _fp8_storage_dtype,
    _is_prime,
    _next_prime,
    config_declares_engram,
    decode_block_scale,
    decode_fp8,
    engram_text_config,
)
from atom.model_ops.engram_layer import EngramOp

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
        out[arr >= 0] = self.lookup_table[arr[arr >= 0]]
        return out


def tiny_config(**overrides) -> EngramConfig:
    base = {
        "layer_ids": (0, 2), "num_embeddings": (0, 0), "max_ngram_size": 3,
        "vocab_size": 1024, "n_heads": 2, "head_dim": 8, "pad_token_id": 0,
        "compressed_vocab_size": 512,
    }  # fmt: skip
    base.update(overrides)
    cfg = EngramConfig(**base)
    # Derive true per-layer row counts for this toy shape (V4.1 numbers below).
    seen, totals = set(), []
    for _ in cfg.layer_ids:
        total = 0
        for _ in cfg.ngram_orders:
            start = cfg.vocab_size - 1
            for _ in range(cfg.n_heads):
                start = _next_prime(start, seen)
                seen.add(start)
                total += start
        totals.append(total)
    return EngramConfig(**{**base, "num_embeddings": tuple(totals)})


def build_mapping(cfg: EngramConfig | None = None) -> NgramHashMapping:
    cfg = cfg or tiny_config()
    return NgramHashMapping(cfg, StubTokenizer(cfg.compressed_vocab_size))


def make_prefetcher(max_concurrent_seqs: int = 4096) -> EngramPrefetcher:
    mapping = build_mapping()
    tables = {
        lid: HostEmbeddingTable(
            torch.arange(int(mapping.head_vocab_sizes[lid].sum()) * 8, dtype=torch.float32).reshape(-1, 8),
            num_rows=int(mapping.head_vocab_sizes[lid].sum()),
            head_dim=8,
        )
        for lid in mapping.config.layer_ids
    }  # fmt: skip
    return EngramPrefetcher(mapping, tables, max_concurrent_seqs=max_concurrent_seqs)


def make_runtime() -> EngramHost:
    pf = make_prefetcher()
    cfg = pf._hash_mapping.config
    return EngramHost(pf, 8, cfg.num_hash_heads, 8, torch.device("cpu"))


def make_op(hidden=16, engram_hidden=24, hc=2) -> EngramOp:
    return EngramOp(1, hidden, engram_hidden, hc)


def test_primes_and_head_vocab_sizes():
    assert [n for n in range(2, 12) if _is_prime(n)] == [2, 3, 5, 7, 11]
    assert not _is_prime(1) and not _is_prime(0)
    assert _next_prime(10, {11, 13}) == 17

    cfg = EngramConfig.from_hf(V41_FLASH)
    mapping = NgramHashMapping(cfg, StubTokenizer(cfg.compressed_vocab_size))
    # Derived per-layer row counts must equal the checkpoint's declared ones.
    for lid, expected in zip(cfg.layer_ids, cfg.num_embeddings):
        assert int(mapping.head_vocab_sizes[lid].sum()) == expected
    # Every head's prime-sized bucket is globally distinct.
    every = np.concatenate([mapping.head_vocab_sizes[lid] for lid in cfg.layer_ids])
    assert len(set(every.tolist())) == every.size
    # A config whose row counts disagree with the derived sizes is rejected.
    bad = EngramConfig.from_hf({**V41_FLASH, "engram_num_embeddings": [1, 2]})
    with pytest.raises(ValueError, match="does not match the trained tables"):
        NgramHashMapping(bad, StubTokenizer(bad.compressed_vocab_size))


def test_config_parsing_and_detection():
    assert EngramConfig.from_hf({"hidden_size": 5120}) is None
    cfg = EngramConfig.from_hf(V41_FLASH)
    assert cfg.ngram_orders == (2, 3, 4) and cfg.num_hash_heads == 24

    class _Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    # engram_text_config returns the container that declares engram (root or
    # nested, dict or object), so callers read engram_* off the right one. The
    # object-with-dict-text_config shape must return the dict (else the scheduler
    # would read engram_max_ngram_size off the outer object and raise).
    nested = {"engram_layer_ids": [1, 14], "engram_max_ngram_size": 4}
    assert engram_text_config({"text_config": nested}) is nested
    assert engram_text_config(_Obj(text_config=nested)) is nested
    root = {"engram_layer_ids": [1], "text_config": {"a": 4}}
    assert engram_text_config(root) is root  # root wins when it declares
    obj = _Obj(engram_layer_ids=[1])
    assert engram_text_config(obj) is obj
    assert engram_text_config({"text_config": {"hidden_size": 4}}) is None
    assert engram_text_config(_Obj(hidden_size=4)) is None
    assert config_declares_engram({"engram_layer_ids": [1, 14]})
    assert not config_declares_engram({"hidden_size": 4})

    # Parallel arrays consumed pairwise by zip; a length mismatch fails closed.
    base = {"max_ngram_size": 4, "vocab_size": 100, "n_heads": 2, "head_dim": 8, "pad_token_id": 0, "compressed_vocab_size": 50}  # fmt: skip
    with pytest.raises(ValueError, match="exactly one table size"):
        EngramConfig(layer_ids=(1, 14), num_embeddings=(1000,), **base)
    with pytest.raises(ValueError, match="declares no layers"):
        EngramConfig(layer_ids=(), num_embeddings=(), **base)
    EngramConfig(layer_ids=(1, 14), num_embeddings=(1000, 2000), **base)


def test_hash_properties():
    # Exact hash values are covered by the scalar-reference test; here just the
    # shape, layer independence, causal prefix, and table-layout invariants.
    mapping = build_mapping()
    ids = np.array([[5, 9, 13, 21], [1, 2, 3, 4]], dtype=np.int64)
    h = mapping.hash_layer(ids, layer_id=0)
    assert h.shape == (2, 4, mapping.config.num_hash_heads)
    assert not np.array_equal(h, mapping.hash_layer(ids, 2))  # layers differ
    every = mapping.hash_all_layers(ids)
    for lid in mapping.config.layer_ids:
        np.testing.assert_array_equal(every[lid], mapping.hash_layer(ids, lid))
    b = ids.copy()
    b[0, 3] = 99  # a later token cannot change an earlier position's hash
    np.testing.assert_array_equal(h[:, :3], mapping.hash_layer(b, 0)[:, :3])
    # Row indices stay inside the table; heads occupy contiguous, disjoint slices.
    sizes, offsets = mapping.head_vocab_sizes[0], mapping.head_offsets[0]
    rows = mapping.to_row_indices(h, 0)
    assert rows.min() >= 0 and rows.max() < int(sizes.sum()) and offsets[0] == 0
    assert all(
        offsets[i] == offsets[i - 1] + sizes[i - 1] for i in range(1, len(offsets))
    )


def test_hash_layer_matches_scalar_reference():
    """`hash_layer` + `to_row_indices` must match an independent scalar reimpl
    (same constants, different code), catching vectorization/head/left-pad bugs."""
    mapping = build_mapping()
    layer_id = mapping.config.layer_ids[0]
    ids = np.array([[5, 7, 9, 2, 5]], dtype=np.int64)  # T=5, short enough to left-pad
    compressed = mapping.tokenizer(ids)
    prod = mapping.to_row_indices(
        mapping.hash_layer(ids, layer_id, compress=True), layer_id
    )
    mult = mapping.layer_multipliers[layer_id]
    sizes = mapping.head_vocab_sizes[layer_id]
    offsets = mapping.head_offsets[layer_id]
    pad, n_heads = int(mapping.pad_id), mapping.config.n_heads
    B, T = compressed.shape
    expected = np.empty((B, T, mapping.config.num_hash_heads), dtype=np.int64)
    for b in range(B):
        for t in range(T):
            head = 0
            for order_idx, n in enumerate(mapping.config.ngram_orders):
                mix = int(compressed[b, t]) * int(mult[0])
                for k in range(1, n):
                    prev = int(compressed[b, t - k]) if t - k >= 0 else pad
                    mix ^= prev * int(mult[k])
                for j in range(n_heads):
                    hidx = order_idx * n_heads + j
                    expected[b, t, head] = mix % int(sizes[hidx]) + int(offsets[hidx])
                    head += 1
    np.testing.assert_array_equal(prod, expected)


def test_host_table_gather_and_fp8():
    table = torch.arange(40 * 4, dtype=torch.float32).reshape(40, 4)
    host = HostEmbeddingTable(table, num_rows=40, head_dim=4)
    out = host.gather(np.array([[[0, 39], [7, 7]]], dtype=np.int64))
    assert out.shape == (1, 2, 2, 4)
    torch.testing.assert_close(out[0, 0, 0], table[0])
    torch.testing.assert_close(out[0, 1, 1], table[7])
    with pytest.raises(IndexError, match="out of range"):  # not clamped
        host.gather(np.array([[[40]]], dtype=np.int64))
    with pytest.raises(ValueError, match="rows"):
        HostEmbeddingTable(torch.zeros(3, 4), num_rows=8, head_dim=4)
    with pytest.raises(ValueError, match="wide"):
        HostEmbeddingTable(torch.zeros(8, 5), num_rows=8, head_dim=4)

    # Block scale applied per block; without it, values are orders of magnitude off.
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0], [1.0, 1.0], [2.0, 2.0]])
    scaled = HostEmbeddingTable(torch.ones(4, 8), num_rows=4, head_dim=8, scale=scale)
    assert scaled.block_size == 4
    torch.testing.assert_close(
        scaled.gather(np.array([[[1]]], dtype=np.int64))[0, 0, 0],
        torch.tensor([4.0] * 4 + [8.0] * 4),
    )
    plain = HostEmbeddingTable(torch.ones(4, 8), num_rows=4, head_dim=8)
    assert plain.block_size == 0
    torch.testing.assert_close(
        plain.gather(np.array([[[1]]], dtype=np.int64))[0, 0, 0], torch.ones(8)
    )
    with pytest.raises(ValueError, match="scale has"):
        HostEmbeddingTable(torch.ones(4, 8), 4, 8, scale=torch.ones(3, 2))
    with pytest.raises(ValueError, match="not divisible"):
        HostEmbeddingTable(torch.ones(4, 8), 4, 8, scale=torch.ones(4, 3))

    # gather flattens indices first, so a scaled gather over [B, heads] keeps scale
    # as [rows, n_blocks] (E8M0 bytes 127,128 -> scales 1,2).
    s8 = HostEmbeddingTable(
        torch.ones(8, 4), 8, 4, scale=torch.tensor([[127, 128]] * 8, dtype=torch.uint8)
    )
    md = s8.gather(np.array([[0, 3], [5, 7]], dtype=np.int64))
    assert md.shape == (2, 2, 4)
    assert all(
        torch.equal(md[b, h], torch.tensor([1.0, 1.0, 2.0, 2.0]))
        for b in range(2)
        for h in range(2)
    )
    # A raw-uint8 table is reinterpreted with the target's own E4M3 variant (FN or
    # FNUZ), never read as the integer 60; derive the expected value from that same
    # variant so the check holds on both platforms.
    fp8 = _fp8_storage_dtype()
    bytes3 = torch.tensor([0x3C, 0x00, 0x40], dtype=torch.uint8)
    assert decode_fp8(bytes3, torch.float32)[0].item() != 60.0  # reinterpreted
    torch.testing.assert_close(
        decode_fp8(bytes3, torch.float32), bytes3.view(fp8).float()
    )
    raw = HostEmbeddingTable(torch.full((4, 2), 0x3C, dtype=torch.uint8), 4, 2)
    expected = torch.full((2, 2), bytes3.view(fp8).float()[0].item())
    torch.testing.assert_close(raw.gather(np.array([0, 3])), expected)
    # Native float8 passes straight through .to().
    native = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
    torch.testing.assert_close(
        decode_fp8(native, torch.float32), torch.tensor([1.0, 2.0])
    )

    # E8M0 scale decode: biased exponents, code 0 the exact-zero sentinel.
    got = decode_block_scale(
        torch.tensor([127, 128, 126, 130, 0], dtype=torch.uint8), torch.float32
    )
    torch.testing.assert_close(got, torch.tensor([1.0, 2.0, 0.5, 8.0, 0.0]))
    assert got.dtype == torch.float32
    f = torch.tensor([1.5, 2.5])
    torch.testing.assert_close(decode_block_scale(f, torch.float32), f)  # passthrough


def test_cache_evict_take_drop():
    cache = EngramPrefetchCache(capacity=2)
    for s in (1, 2, 3):
        cache.put(s, 0, torch.zeros(1))
    assert cache.take(1, 0) is None and cache.take(3, 0) is not None  # LRU evicts 1
    c = EngramPrefetchCache()
    c.put(4, 1, torch.zeros(1))
    assert c.take(4, 1) is not None and c.take(4, 1) is None  # take is destructive
    for s, layer in ((5, 0), (5, 2), (6, 0)):
        c.put(s, layer, torch.zeros(1))
    c.drop(5)  # drop forgets every layer of a request
    assert c.take(5, 0) is None and c.take(5, 2) is None and c.take(6, 0) is not None


def test_compressed_tokenizer_cache_key_fast_vs_slow():
    """Fast tokenizer keys the disk cache on its serialization; a slow one yields
    None so `_load_or_build` rebuilds rather than reuse a vocab-only key."""
    from atom.model_ops.engram import CompressedTokenizer

    ct = object.__new__(CompressedTokenizer)  # bypass __init__ (which builds)

    class _Slow:
        def get_vocab(self):
            return {"a": 0}

    ct._tokenizer = _Slow()
    assert ct._cache_key() is None

    class _Fast:
        class backend_tokenizer:
            @staticmethod
            def to_str():
                return "SERIALIZED-TOKENIZER"

    ct._tokenizer = _Fast()
    key = ct._cache_key()
    assert isinstance(key, str) and len(key) == 16


def test_prefetcher():
    # Cache keyed by (seq, layer): capacity scales by layer count, window by seqs.
    pf = make_prefetcher(max_concurrent_seqs=3)
    assert len(pf.layer_ids) >= 2
    assert pf.cache._capacity == 3 * len(pf.layer_ids) and pf._window_capacity == 3
    pf.shutdown()

    # Async prefetch result == inline compute over the same window.
    pf = make_prefetcher()
    pf.seed_context(11, [3, 4])
    pf.seed_context(12, [6, 7])
    expected = pf.compute([11, 12], np.array([[3, 4, 5], [6, 7, 8]], dtype=np.int64))
    fut = pf.submit_compute([11, 12], np.array([[5], [8]], dtype=np.int64))
    assert fut.result(timeout=30) is None and pf.wait(timeout=30)
    for (seq_id, layer_id), value in expected.items():
        torch.testing.assert_close(pf.cache.take(seq_id, layer_id), value)
    pf.shutdown()

    # drop clears the cache; wait without a submit is a no-op.
    pf = make_prefetcher()
    pf.submit_compute([21], np.array([[1, 2, 3]], dtype=np.int64)).result(timeout=30)
    pf.drop_requests([21])
    assert len(pf.cache) == 0 and pf.wait(timeout=1)
    pf.shutdown()


def test_compute_prefill():
    pf = make_prefetcher()
    empty = np.array([], dtype=np.int64)
    # Each prefill position == the decode-window compute for the same n-gram.
    pre = pf.compute_prefill([np.array([3, 4])], [np.array([5, 6, 7])])
    for pos, window in enumerate([[3, 4, 5], [4, 5, 6], [5, 6, 7]]):
        exp = pf.compute([0], np.array([window], dtype=np.int64))
        for lid in pf.layer_ids:
            torch.testing.assert_close(pre[lid][0][pos], exp[(0, lid)].reshape(-1))
    # Chunked with the n-1 context carried across the boundary == unchunked.
    whole = pf.compute_prefill([empty], [np.array([1, 2, 3, 4, 5, 6])])
    p1 = pf.compute_prefill([empty], [np.array([1, 2, 3])])
    p2 = pf.compute_prefill([np.array([2, 3])], [np.array([4, 5, 6])])
    for lid in pf.layer_ids:
        torch.testing.assert_close(
            torch.cat([p1[lid][0], p2[lid][0]], 0), whole[lid][0]
        )
    # First chunk (empty context) left-pads, so position 0 is a 1-gram.
    first = pf.compute_prefill([empty], [np.array([9, 8])])
    e0 = pf.compute([0], np.array([[9]], dtype=np.int64))
    for lid in pf.layer_ids:
        torch.testing.assert_close(first[lid][0][0], e0[(0, lid)].reshape(-1))
    pf.shutdown()


def test_stage_prefill():
    rt = make_runtime()
    chunks = [np.array([1, 2, 3]), np.array([4, 5])]
    ctxs = [np.array([], dtype=np.int64), np.array([6, 7])]  # 72 is a continuation
    pre = rt.prefetcher.compute_prefill(ctxs, chunks)
    assert rt.stage_prefill([71, 72], chunks, ctxs, final_mask=[True, True]) == 5
    for lid in rt.layer_ids:
        emb = rt.embeddings(lid)
        assert emb.shape == (5, rt.embed_width)
        torch.testing.assert_close(emb[0:3], pre[lid][0][:, : rt.embed_width])
        torch.testing.assert_close(emb[3:5], pre[lid][1][:, : rt.embed_width])
    # Final chunk seeds the decode window with the prompt's trailing n-1 tokens.
    assert list(rt.prefetcher._window[71]) == [2, 3]
    assert list(rt.prefetcher._window[72]) == [4, 5]
    with pytest.raises(ValueError, match="exceed staging capacity"):
        rt.stage_prefill([81], [np.arange(9)], [np.array([], dtype=np.int64)])
    rt.shutdown()


def test_engram_op_and_checkpoint():
    op = make_op()
    # ATOM residual streams are [num_tokens, hc, dim], with optional leading dims.
    assert op(torch.randn(5, 2, 16), torch.randn(5, 24)).shape == (5, 2, 16)
    assert op(torch.randn(2, 3, 2, 16), torch.randn(2, 3, 24)).shape == (2, 3, 2, 16)
    with pytest.raises(ValueError, match="expected 24"):
        op(torch.randn(2, 2, 16), torch.randn(2, 25))
    with pytest.raises(ValueError, match="hc_mult=2"):
        op(torch.randn(2, 3, 16), torch.randn(2, 24))
    with pytest.raises(ValueError, match="tokens"):
        op(torch.randn(5, 2, 16), torch.randn(4, 24))
    # V4.1 omits the short conv; wkv is hc key projections then one value.
    assert not any("short_conv" in n for n, _ in op.named_modules())
    assert op.wkv.weight.shape == (3 * 16, 24) and op.key_rows == 2 * 16
    # The gate is computed in fp32; the contribution is cast to the residual dtype.
    opb = make_op().to(torch.bfloat16)
    out_b = opb(
        torch.randn(3, 2, 16, dtype=torch.bfloat16),
        torch.randn(3, 24, dtype=torch.bfloat16),
    )
    assert out_b.dtype == torch.bfloat16

    good = torch.zeros(3 * 16, 24)
    with pytest.raises(ValueError, match="wkv is"):
        op.load_checkpoint_weights(
            torch.zeros(24, 48), torch.zeros(2, 16), torch.zeros(2, 16)
        )
    with pytest.raises(ValueError, match="k_weight is"):
        op.load_checkpoint_weights(good, torch.zeros(3, 16), torch.zeros(2, 16))
    with pytest.raises(ValueError, match="q_weight is"):
        op.load_checkpoint_weights(good, torch.zeros(2, 16), torch.zeros(2, 8))
    # Block dequant: scale 3.0 over all-ones weights -> 3.0; bad scale shape rejected.
    op2 = make_op(hidden=16, engram_hidden=32, hc=2)
    ones = (torch.ones(48, 32), torch.ones(2, 16), torch.ones(2, 16))
    op2.load_checkpoint_weights(*ones, wkv_scale=torch.full((6, 4), 3.0), block=8)
    torch.testing.assert_close(op2.wkv.weight, torch.full((48, 32), 3.0))
    with pytest.raises(ValueError, match="wkv scale is"):
        op2.load_checkpoint_weights(*ones, wkv_scale=torch.ones(2, 2), block=8)


def test_runtime_staging_miss_and_drop():
    rt = make_runtime()
    tokens = np.array([[5], [6]], dtype=np.int64)
    rt.prefetch_next([31, 32], tokens)
    assert rt.stage_embeddings([31, 32], tokens) == 2
    for lid in rt.layer_ids:
        assert rt.embeddings(lid).shape == (2, rt.embed_width)
    rt.prefetch_next([31, 32], tokens)  # padded_rows below batch must not drop rows
    assert rt.stage_embeddings([31, 32], tokens, padded_rows=1) == 2
    with pytest.raises(RuntimeError, match="no token ids were supplied"):
        rt.stage_embeddings([51], None)
    with pytest.raises(ValueError, match="exceeds staging capacity"):
        rt.stage_embeddings(list(range(9)), np.zeros((9, 1), dtype=np.int64))
    rt.shutdown()

    # A miss changes latency, never the answer (cold == warm).
    rt = make_runtime()
    rt.prefetch_next([41], np.array([[9]], dtype=np.int64))
    rt.stage_embeddings([41], np.array([[9]], dtype=np.int64))
    warm = {lid: rt.embeddings(lid).clone() for lid in rt.layer_ids}
    cold = make_runtime()
    cold.stage_embeddings([41], np.array([[9]], dtype=np.int64))  # nothing prefetched
    for lid in cold.layer_ids:
        torch.testing.assert_close(cold.embeddings(lid), warm[lid])
    rt.shutdown()
    cold.shutdown()

    # A carried-over miss recomputes from its window, ignoring the placeholder; and
    # drop_requests clears both the cache and the window.
    rt = make_runtime()
    rt.seed_context(61, [2, 3])
    rt.prefetch_next([61], np.array([[4]], dtype=np.int64))
    rt.stage_embeddings([61], np.array([[4]], dtype=np.int64))
    warm = {lid: rt.embeddings(lid).clone() for lid in rt.layer_ids}
    rt.stage_embeddings([61], np.array([[999]], dtype=np.int64))  # placeholder ignored
    for lid in rt.layer_ids:
        torch.testing.assert_close(rt.embeddings(lid), warm[lid])
    # A fresh prefetch repopulates cache + window; drop_requests clears both.
    rt.prefetch_next([61], np.array([[5]], dtype=np.int64))
    rt.prefetcher.wait(timeout=30)
    assert 61 in rt.prefetcher._window
    assert rt.prefetcher.cache.contains(61, rt.layer_ids[0])
    rt.drop_requests([61])
    assert 61 not in rt.prefetcher._window
    assert not rt.prefetcher.cache.contains(61, rt.layer_ids[0])
    rt.shutdown()
