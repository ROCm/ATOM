# SPDX-License-Identifier: MIT
"""QSA FP8 K/V and FP8 / MXFP4 compressed index-key caches."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")

from atom.model_ops.qwen4_exp.ops import qsa

gpu = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _config(kv, index=None):
    return SimpleNamespace(kv_cache_dtype=kv, index_cache_dtype=index)


@pytest.mark.parametrize(
    ("kv", "index", "expected"),
    [
        ("bf16", None, ("bf16", "bf16")),
        ("auto", None, ("bf16", "bf16")),
        ("fp8", None, ("fp8", "fp8")),
        ("fp8", "bf16", ("fp8", "bf16")),
        ("fp8", "fp4", ("fp8", "fp4")),
        ("bf16", "fp8", ("bf16", "fp8")),
    ],
)
def test_cache_formats_follow_the_kv_dtype(kv, index, expected):
    assert qsa.qsa_cache_formats(_config(kv, index)) == expected


@pytest.mark.parametrize(("kv", "index"), [("int8", None), ("fp4", None)])
def test_cache_formats_reject_unserved(kv, index):
    with pytest.raises(ValueError, match="Qwen QSA serves"):
        qsa.qsa_cache_formats(_config(kv, index))


def _builder(kv, index, draft_rows, block=16, rope=False):
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    builder = object.__new__(Qwen4ExpMetadataBuilder)
    hf = SimpleNamespace(head_dim=256, indexer_head_dim=128, indexer_compress_ratio=4)
    config = SimpleNamespace(kv_cache_dtype=kv, index_cache_dtype=index, hf_config=hf)
    builder.model_runner = SimpleNamespace(config=config, _get_num_kv_heads=lambda: 2)
    builder.block_size = block
    builder.cache_rope_positions = rope
    counts = {"qsa": 12, "qsa_draft": draft_rows}
    builder.row_counts = lambda: {k: v for k, v in counts.items() if v}
    return builder


def test_bf16_fields_are_unchanged():
    fields = _builder("bf16", None, 0)._qsa_fields()
    assert [(f.name, f.layers, f.shape, f.dtype) for f in fields] == [
        ("k", 12, (16, 2, 256), torch.bfloat16),
        ("v", 12, (16, 2, 256), torch.bfloat16),
        ("index_raw", 12, (16, 1, 128), torch.bfloat16),
        ("index_compressed", 12, (4, 1, 128), torch.bfloat16),
    ]


def test_quantized_fields_and_bf16_draft_rows():
    builder = _builder("fp8", "fp4", 1, block=64)
    fields = {f.name: f for f in builder._qsa_fields()}
    fp8 = qsa.FP8_DTYPE
    expected = {
        "k": (12, (64, 2, 256), fp8),
        "v": (12, (64, 2, 256), fp8),
        "k_scale": (12, (64, 2), torch.float32),
        "v_scale": (12, (64, 2), torch.float32),
        "index_raw": (12, (64, 1, 128), torch.bfloat16),
        "index_compressed": (12, (16, 1, 64), torch.uint8),
        "index_compressed_scale": (12, (16, 1, 4), torch.uint8),
        "k_draft": (1, (64, 2, 256), torch.bfloat16),
        "v_draft": (1, (64, 2, 256), torch.bfloat16),
        "index_raw_draft": (1, (64, 1, 128), torch.bfloat16),
        "index_compressed_draft": (1, (16, 1, 128), torch.bfloat16),
    }
    assert {n: (f.layers, f.shape, f.dtype) for n, f in fields.items()} == expected


def test_fp8_pool_fits_more_blocks():
    bf16 = _builder("bf16", None, 0, rope=True)
    bf16.row_counts = lambda: {"qsa": 13}
    # Block bytes of Qwen3.8-Flash-Next's 12 target + 1 draft QSA layers, as
    # served (multimodal, so mRoPE positions are cached).
    assert bf16._page_bytes() == 497664
    assert _builder("fp8", "fp8", 1, rope=True)._page_bytes() == 298240
    assert _builder("fp8", "bf16", 1, 64, rope=True)._page_bytes() == 1216000


def _store(fmt, values, page_rows):
    rows, heads, dim = values.shape
    pages = -(-rows // page_rows)
    vdim, vdtype, tail, sdtype = qsa.qsa_cache_row_layout(fmt, dim)
    cache = torch.zeros(pages, page_rows, heads, vdim, device="cuda", dtype=vdtype)
    if tail is None:
        cache.view(-1, heads, dim)[:rows] = values
        return cache, None
    scale = torch.zeros(pages, page_rows, heads, *tail, device="cuda", dtype=sdtype)
    slots = torch.arange(rows, device="cuda")
    qsa.qsa_store_quantized_rows(cache, scale, slots, values)
    return cache, scale


def _dequant(fmt, cache, scale):
    if scale is None:
        return cache.float()
    if fmt == "fp8":
        return cache.float() * scale.unsqueeze(-1)
    grid = torch.tensor(E2M1 + tuple(-x for x in E2M1), device="cuda")
    codes = cache.long()
    values = torch.stack([grid[codes & 15], grid[codes >> 4]], dim=-1).flatten(-2)
    exponent = scale.float().repeat_interleave(qsa.FP4_GROUP, dim=-1)
    return values * torch.exp2(exponent - 127)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@gpu
def test_fp8_writer_never_saturates():
    torch.manual_seed(0)
    rows = 2048
    magnitude = torch.logspace(-4, 4, rows, device="cuda")[:, None, None]
    key = (torch.randn(rows, 2, 256, device="cuda") * magnitude).bfloat16()
    value = (torch.randn(rows, 2, 256, device="cuda") * magnitude).bfloat16()
    caches = [torch.zeros(rows // 16, 16, 2, 256, device="cuda", dtype=qsa.FP8_DTYPE)]
    caches.append(torch.zeros_like(caches[0]))
    scales = [torch.zeros(rows // 16, 16, 2, device="cuda") for _ in range(2)]
    slots = torch.arange(rows, device="cuda")
    slots[5] = -1
    qsa.qsa_store_quantized_rows(
        caches[0], scales[0], slots, key, caches[1], scales[1], value
    )
    for cache, scale, source in zip(caches, scales, (key, value)):
        restored = _dequant("fp8", cache, scale).view(rows, 2, 256)
        assert torch.isfinite(restored).all()
        per_row = (restored - source.float()).norm(dim=-1) / source.float().norm(dim=-1)
        assert per_row[slots >= 0].max() < 0.06
        assert not restored[5].any()


@gpu
def test_fp4_writer_is_exact_on_the_grid_and_close_off_it():
    torch.manual_seed(1)
    rows = 512
    grid = torch.tensor(E2M1 + tuple(-x for x in E2M1), device="cuda")
    # Each 32-dim group holds +-6 so the shared exponent is the group's own.
    codes = torch.randint(0, 16, (rows, 1, 128), device="cuda")
    exact = grid[codes]
    exact[..., :: qsa.FP4_GROUP] = 6.0
    exact *= torch.exp2(
        torch.randint(-20, 20, (rows, 1, 4), device="cuda")
    ).repeat_interleave(qsa.FP4_GROUP, -1)
    cache, scale = _store("fp4", exact.bfloat16(), 4)
    torch.testing.assert_close(
        _dequant("fp4", cache, scale).view(-1, 1, 128)[:rows], exact, rtol=0, atol=0
    )
    noisy = torch.randn(rows, 1, 128, device="cuda").bfloat16()
    cache, scale = _store("fp4", noisy, 4)
    restored = _dequant("fp4", cache, scale).view(-1, 1, 128)[:rows]
    per_row = (restored - noisy.float()).norm(dim=-1) / noisy.float().norm(dim=-1)
    assert per_row.mean() < 0.15


def _index_case(requests, context, rows_per_request, page_rows, seed=2):
    g = torch.Generator(device="cuda").manual_seed(seed)
    groups = context // 4
    pages_per_request = -(-groups // page_rows)
    pages = requests * pages_per_request + 3
    order = torch.randperm(pages, device="cuda", generator=g).to(torch.int32)
    table = order[: requests * pages_per_request].view(requests, pages_per_request)
    keys = torch.randn(pages * page_rows, 1, 128, device="cuda", generator=g)
    q = torch.randn(requests * rows_per_request, 4, 128, device="cuda", generator=g)
    t2r = torch.arange(requests, device="cuda", dtype=torch.int32)
    t2r = t2r.repeat_interleave(rows_per_request)
    positions = torch.arange(rows_per_request, device="cuda").repeat(requests)
    positions = (positions + context - rows_per_request).to(torch.int64)
    lens = torch.full((requests,), context, device="cuda", dtype=torch.int32)
    return keys.bfloat16(), q.bfloat16(), table, t2r, positions, lens


@gpu
@pytest.mark.parametrize("fmt", ["fp8", "fp4"])
@pytest.mark.parametrize(
    ("rows_per_request", "stacked", "page_rows"),
    [(4, "1", 4), (4, "0", 4), (4, "1", 16), (256, "1", 4), (256, "1", 16)],
)
def test_quantized_index_scores_match_dequantized_keys(
    monkeypatch, fmt, rows_per_request, stacked, page_rows
):
    """Every scorer (stacked decode, CUDA-core decode, prefill MFMA) scores a
    quantized page as FP32 math on its dequantized keys would."""
    monkeypatch.setenv("ATOM_QSA_STACKED_DECODE_SCORER", stacked)
    requests = 9 if rows_per_request == 4 else 2
    keys, q, table, t2r, positions, lens = _index_case(
        requests, 8192, rows_per_request, page_rows
    )
    cache, scale = _store(fmt, keys, page_rows)
    out, visible = qsa.qsa_paged_mqa_logits(
        q, cache, table, t2r, positions, lens, 4, compressed_k_scale=scale
    )
    restored = _dequant(fmt, cache, scale).view(-1, page_rows, 128)
    groups = torch.arange(out.shape[1], device="cuda")
    ref = torch.empty_like(out)
    for request in range(table.shape[0]):
        rows = t2r == request
        k = restored[table[request, groups // page_rows].long(), groups % page_rows]
        scores = torch.relu(q[rows].float() @ k.T).sum(1) / 128**0.5
        ref[rows] = scores
    expected = torch.minimum((positions + 1) // 4, lens[t2r.long()] // 4)
    assert torch.equal(visible, expected.to(visible.dtype))
    seen = groups[None, :] < visible[:, None]
    torch.testing.assert_close(out[seen], ref[seen], rtol=1e-4, atol=1e-4)


@gpu
@pytest.mark.parametrize(("tokens", "requests"), [(36, 9), (1024, 1)])
def test_fp8_sparse_gqa_matches_dequantized_kv(tokens, requests):
    g = torch.Generator(device="cuda").manual_seed(3)
    pages_per_request, page_rows = 64, 16
    pages = requests * pages_per_request + 2
    order = torch.randperm(pages, device="cuda", generator=g).to(torch.int32)
    table = order[: requests * pages_per_request].view(requests, pages_per_request)
    shape = (pages, page_rows, 2, 256)
    key = torch.randn(shape, device="cuda", generator=g).bfloat16()
    value = (torch.randn(shape, device="cuda", generator=g) * 3).bfloat16()
    q = torch.randn(tokens, 24, 256, device="cuda", generator=g).bfloat16()
    context = pages_per_request * page_rows
    selected = torch.randint(0, context, (tokens, 515), device="cuda", generator=g)
    selected = selected.to(torch.int32)
    selected[:, -3:] = -1
    t2r = (torch.arange(tokens, device="cuda") * requests // tokens).to(torch.int32)

    caches = [torch.empty(shape, device="cuda", dtype=qsa.FP8_DTYPE) for _ in "kv"]
    scales = [torch.empty(shape[:3], device="cuda") for _ in "kv"]
    flat = pages * page_rows
    qsa.qsa_store_quantized_rows(
        caches[0],
        scales[0],
        torch.arange(flat, device="cuda"),
        key.view(flat, 2, 256),
        caches[1],
        scales[1],
        value.view(flat, 2, 256),
    )
    decode = requests if tokens == 36 else None
    out = qsa.qsa_sparse_paged_gqa(
        q,
        caches[0],
        caches[1],
        selected,
        table,
        t2r,
        1 / 16,
        num_decode_requests=decode,
        k_scale=scales[0],
        v_scale=scales[1],
    )
    ref = qsa.qsa_sparse_paged_gqa(
        q,
        _dequant("fp8", caches[0], scales[0]).bfloat16(),
        _dequant("fp8", caches[1], scales[1]).bfloat16(),
        selected,
        table,
        t2r,
        1 / 16,
        num_decode_requests=decode,
    )
    assert torch.isfinite(out).all()
    assert _rel(out, ref) < 1e-2
