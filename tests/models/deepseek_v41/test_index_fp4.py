# SPDX-License-Identifier: MIT
"""The FP4 index plane: what its writer stores and what its scorer selects."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)

DIM, HEADS, PAGE = 128, 32, 8


def _norm(monkeypatch, eps=1e-20):
    """The indexer's `k_norm` as the model builds it, bf16 weight."""
    from atom.model_ops import layernorm

    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    norm = layernorm.RMSNorm(DIM, eps)
    norm.weight = torch.nn.Parameter(
        (1 + 0.3 * torch.randn(DIM)).bfloat16().cuda(), requires_grad=False
    )
    return norm


def _rope():
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding

    return RotaryEmbedding(
        64, 1 << 16, base=160000.0, original_length=65536, factor=16.0,
        beta_fast=32, beta_slow=1,
    ).cuda()  # fmt: skip


def _pool(pages, per_page):
    """One owner's two FP4 planes, `[pages, rows, bytes]`, and their units."""
    from atom.model_ops.deepseek_v41.index_plane import IndexUnits

    values = torch.zeros(pages, per_page, DIM // 2, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(pages, per_page, DIM // 32, dtype=torch.uint8, device="cuda")
    units = IndexUnits(
        values.view(-1, PAGE, DIM // 2), scales.view(-1, PAGE, DIM // 32)
    )
    return values, scales, units


def _write(keys, table, per_page, ratio, rows, units, norm, rope, batch=0):
    """Compression boundaries of request `batch` whose index rows are `rows`
    (position rows x ratio + ratio - 1, rotated at rows x ratio), through the
    FP4 writer; every third plan row a sentinel. -> the slots they should land
    at, -1 for a sentinel."""
    from atom.model_ops.deepseek_v41.index_write import write_index_rows_fp4

    plan = torch.zeros(keys.shape[0], 4, dtype=torch.int32, device="cuda")
    plan[:, 1] = batch
    plan[::3, 1] = -1
    plan[:, 2] = rows * ratio + ratio - 1
    write_index_rows_fp4(
        keys, units, plan, table, per_page, ratio=ratio, norm=norm, rope=rope
    )
    slots = table[batch, (rows // per_page).long()].long() * per_page + rows % per_page
    return torch.where(plan[:, 1] >= 0, slots, -1)


@pytest.mark.parametrize("ratio", [1, 2])
def test_fp4_writer_stores_the_quantized_key_in_the_rowgroup_layout(monkeypatch, ratio):
    """Bytes equal to the model's own key chain (RMSNorm, RoPE, `quantize_fp4`)
    packed by the scorer's `pack_kv_cache` at the slot the plan and PAGE table
    give; a sentinel row writes nothing."""
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
        pack_kv_cache,
    )

    from atom.model_ops.blockscale import quantize_fp4

    torch.manual_seed(0)
    pages, per_page = 4, 64
    norm, rope = _norm(monkeypatch), _rope()
    values, scales, units = _pool(pages, per_page)
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda")
    rows = pages * per_page
    keys = (3 * torch.randn(rows, DIM, device="cuda")).bfloat16()
    index_rows = torch.randperm(rows, device="cuda").int()
    slots = _write(keys, table, per_page, ratio, index_rows, units, norm, rope)
    live = slots >= 0

    key = rope(norm(keys.clone())[None], index_rows.long() * ratio)[0]
    ref_v, ref_s = quantize_fp4(key, group_size=32, scale_dtype=torch.float8_e8m0fnu)
    natural_v = torch.zeros(rows, DIM // 2, dtype=torch.uint8, device="cuda")
    natural_s = torch.zeros(rows, DIM // 32, dtype=torch.uint8, device="cuda")
    natural_v[slots[live]] = ref_v.view(torch.uint8)[live]
    natural_s[slots[live]] = ref_s.view(torch.uint8)[live]
    packed_v, packed_s = pack_kv_cache(natural_v, natural_s, PAGE)
    assert torch.equal(values.flatten(), packed_v.flatten())
    assert torch.equal(scales.flatten(), packed_s.flatten())


@pytest.mark.parametrize("far", [False, True])
def test_fp4_scorer_selects_the_official_top_k(monkeypatch, far):
    """The FP4 plane's scoring (`quantize_query_fp4` with the query's RoPE,
    `score_topk_quantized`) picks the rows the official FP4 arithmetic ranks
    highest (q after its bf16 RoPE and k through `quantize_fp4`, fp32 logits
    = sum_h w_h relu(q_h . k) x weights_scale), a row's own visibility only;
    ids may differ only where two logits tie within the summation's rounding.
    ``far``: the pages sit past 4 GiB of the values plane, out of a 32-bit
    offset's reach."""
    from atom.model_ops.blockscale import quantize_fp4
    from atom.model_ops.deepseek_v41 import paged_scoring as scoring
    from atom.model_ops.deepseek_v41.unit_table import unit_table

    torch.manual_seed(1)
    pages, per_page, topk = 4, 128, 64
    # pages past 4.5 GiB of values when far
    skip = (9 << 29) // (per_page * DIM // 2) if far else 0
    norm, rope = _norm(monkeypatch), _rope()
    _, _, units = _pool(skip + pages, per_page)
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda") + skip
    width = pages * per_page
    keys = (3 * torch.randn(width, DIM, device="cuda")).bfloat16()
    positions = torch.arange(width, device="cuda", dtype=torch.int32)
    slots = _write(keys, table, per_page, 1, positions, units, norm, rope)
    written = torch.zeros(width, dtype=torch.bool, device="cuda")
    written[positions[slots >= 0].long()] = True

    rows = 7
    tiles = unit_table(
        table, torch.zeros(rows, dtype=torch.int32, device="cuda"), per_page // PAGE
    )
    query = torch.randn(rows, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    q_pos = torch.randint(0, 1 << 16, (rows,), device="cuda")
    weights = torch.rand(rows, HEADS, dtype=torch.bfloat16, device="cuda")
    visible = torch.tensor([0, 1, 33, 64, 129, 511, 512], device="cuda").int()
    weights_scale = (HEADS * DIM) ** -0.5
    selected, _ = scoring.score_topk_quantized(
        scoring.quantize_query_fp4(query, rope, q_pos), weights, units, tiles,
        visible, topk=topk, weight_scale=weights_scale,
    )  # fmt: skip

    key = rope(norm(keys.clone())[None], positions.long())[0]
    kd = quantize_fp4(key, group_size=32, dequantize=True).float()
    kd[~written] = 0  # a sentinel's row was never written: zero bytes, zero key
    rotated = rope(query.clone()[None], q_pos)[0]
    qd = quantize_fp4(rotated, group_size=32, dequantize=True).float()
    logits = torch.einsum("rhd,kd->rkh", qd, kd).relu()
    logits = (logits * weights.float()[:, None, :]).sum(-1) * weights_scale
    for r in range(rows):
        seen = int(visible[r])
        got = selected[r][selected[r] >= 0]
        assert got.numel() == min(seen, topk)
        assert (got < seen).all() and torch.equal(got, got.sort().values)
        if seen <= topk:
            assert torch.equal(got, torch.arange(seen, device="cuda", dtype=got.dtype))
            continue
        row = logits[r, :seen]
        threshold = row.topk(topk).values[-1]
        tol = 1e-4 * row.abs().max()
        reference = set(row.topk(topk).indices.tolist())
        for i in set(got.tolist()) ^ reference:
            assert abs(row[i] - threshold) <= tol, (r, i, row[i], threshold)


@pytest.mark.parametrize("band", [None, 3])
@pytest.mark.parametrize("lengths", [(3, 2), (20, 1)])
def test_fp4_rows_by_sequence_select_what_rows_alone_do(monkeypatch, band, lengths):
    """A request's rows as one sequence (`ragged`: its rows share each key
    load) select exactly what scoring every row alone does: ragged requests (a
    decode step's few rows, or a prefill's more than one M tile holds), an
    empty padding request and padding rows; ``band`` cuts the rows into bands
    that split requests."""
    from atom.model_ops.deepseek_v41 import paged_scoring as scoring
    from atom.model_ops.deepseek_v41.unit_table import unit_table
    from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata

    if band is not None:
        monkeypatch.setattr(scoring, "plane_rows", lambda width: band)
    torch.manual_seed(3)
    pages, per_page = 8, 128
    _, _, units = _pool(pages, per_page)
    for t in (units.values, units.scales):
        t.copy_(torch.randint(0, 256, t.shape, dtype=torch.uint8, device="cuda"))
    units.scales.clamp_(120, 132)
    tables = torch.tensor(
        [[2, 0, 3, 1], [5, 7, 4, 6], [0, 0, 0, 0]], dtype=torch.int32, device="cuda"
    )
    # the two requests, an empty padding request, then 2 padding rows; a
    # request's last row sees its whole context, each row before it one less
    contexts = (302, 512)
    starts = torch.tensor(
        [0, lengths[0], sum(lengths), sum(lengths)], dtype=torch.int32, device="cuda"
    )
    owners = [b for b, n in enumerate(lengths) for _ in range(n)] + [-1, -1]
    seen = [c - n + 1 + r for c, n in zip(contexts, lengths) for r in range(n)]
    owners = torch.tensor(owners, dtype=torch.int32, device="cuda")
    tiles = unit_table(tables, owners, per_page // PAGE)
    visible = torch.tensor(seen + [0, 0], dtype=torch.int32, device="cuda")
    rows = visible.numel()
    query = scoring.quantize_query_fp4(
        torch.randn(rows, HEADS, DIM, device="cuda").bfloat16(), _rope(),
        torch.randint(0, 1 << 16, (rows,), device="cuda"),
    )  # fmt: skip
    weights = torch.rand(rows, HEADS, device="cuda").bfloat16()

    def select(ragged):
        return scoring.score_topk_quantized(
            query, weights, units, None if ragged else tiles, visible, topk=64,
            weight_scale=(HEADS * DIM) ** -0.5, ragged=ragged,
        )[0]  # fmt: skip

    # a request's rows one sequence, off its PAGE table of per_page // PAGE pages
    ragged = Fp4MqaRaggedMetadata(starts, max(lengths), tables, per_page // PAGE)
    assert torch.equal(select(ragged), select(None))


@pytest.mark.parametrize("ratio", [1, 2])
def test_only_a_full_layers_fp4_rows_score_by_sequence(ratio):
    """`Indexer._ragged`: a FULL layer's FP4 rows, prefill and decode alike,
    its PAGE holding `rows_per_page / index_block_rows` plane pages (the
    expansion `unit_tiles` makes); None for a candidate-bounded layer or FP8."""
    from types import SimpleNamespace

    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )
    from atom.models.deepseek_v41.attention import Indexer

    def ragged(source=None, fp4=True):
        fp4_plane = {"index_block_rows": PAGE, "index_fp4": True} if fp4 else {}
        geometry = V41PoolGeometry(1, ((0, ratio),), 256, 128, 512, DIM, **fp4_plane)
        indexer = Indexer.__new__(Indexer)
        object.__setattr__(
            indexer, "spec", SimpleNamespace(ratio=ratio, candidate_source=source)
        )
        step = SimpleNamespace(cu_seqlens_q="starts", max_q_len=6, block_tables="table")
        return indexer._ragged(SimpleNamespace(geometry=geometry), step)

    assert ragged() == ("starts", 6, "table", 256 // ratio // PAGE)
    assert ragged(source=20) is None
    assert ragged(fp4=False) is None


def test_fp4_scorer_builds_one_kernel_for_every_band_length():
    """Bands of different lengths share one row-group build: its batch is a
    runtime argument, as Triton's is. A build per batch size was a JIT stall
    per new prefill length, on whichever rank met it first, every other rank
    waiting at the next all-reduce."""
    rope = _rope()
    from aiter.ops.flydsl.kernels.mqa_logits import pa_mqa_logits_fp4_rowgroup as rg

    from atom.model_ops.deepseek_v41 import paged_scoring as scoring
    from atom.model_ops.deepseek_v41.unit_table import unit_table

    torch.manual_seed(2)
    pages, per_page = 4, 128
    _, _, units = _pool(pages, per_page)
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda")
    rg.compile_pa_mqa_logits_fp4_rowgroup.cache_clear()
    for rows in (7, 13, 40, 130):
        tiles = unit_table(
            table, torch.zeros(rows, dtype=torch.int32, device="cuda"), per_page // PAGE
        )
        visible = torch.randint(100, pages * per_page, (rows,), device="cuda").int()
        query = scoring.quantize_query_fp4(
            torch.randn(rows, HEADS, DIM, device="cuda").bfloat16(), rope,
            torch.randint(0, 1 << 16, (rows,), device="cuda"),
        )  # fmt: skip
        scoring.score_topk_quantized(
            query, torch.rand(rows, HEADS, device="cuda").bfloat16(), units, tiles,
            visible, topk=64, weight_scale=(HEADS * DIM) ** -0.5,
        )  # fmt: skip
    assert rg.compile_pa_mqa_logits_fp4_rowgroup.cache_info().misses == 1
