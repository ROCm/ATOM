# SPDX-License-Identifier: MIT
"""New OPUS API: band mapping, schedule ownership and packed output."""

from functools import lru_cache

import pytest
import torch

rows_module = pytest.importorskip("atom.model_ops.indexer.rows", exc_type=ImportError)


@pytest.mark.parametrize(
    "heads,max_qlen,pages_per_block",
    [
        (64, 1, 1),
        (64, 4, 1),
        (64, 6, 1),
        (64, 8, 1),
        (64, 20, 1),
        (32, 3, 1),
        (32, 3, 16),
        (32, 20, 32),
    ],
)
def test_opus_bands_keep_sequence_mapping_and_schedule_identity(
    monkeypatch, heads, max_qlen, pages_per_block
):
    calls = []

    class Config:
        block_table_width = 2

        def schedule_buffers(self):
            return object()

    @lru_cache
    def config(device, **bounds):
        assert bounds["heads"] == heads
        assert bounds["max_qlen"] == max_qlen
        assert bounds["pages_per_block"] == pages_per_block
        assert bounds["page_size"] == (8 if heads == 32 else 64)
        return Config()

    def schedule(cfg, starts, ends, *, buffers):
        calls.append((cfg, starts, ends, buffers))
        return object()

    monkeypatch.setattr(rows_module, "mqa_logits_mxfp4_config", config)
    monkeypatch.setattr(rows_module, "mqa_logits_mxfp4_schedule", schedule)
    sequences = rows_module.Fp4MqaRaggedMetadata(
        torch.tensor([0, 2, 5], dtype=torch.int32),
        max_qlen,
        torch.tensor([[3], [7]], dtype=torch.int32),
        pages_per_block,
    )
    ends = torch.tensor([7, 8, 3, 4, 5, 0], dtype=torch.int32)
    storages = rows_module.Fp4LaunchStorages("cpu", 6, 2)

    def prepare(ubatch):
        return rows_module.fp4_rows(
            ends,
            64,
            None,
            sequences,
            heads=heads,
            page_size=8 if heads == 32 else 64,
            layout=rows_module.NATURAL,
            bands=((0, 3), (3, 6)),
            storage=storages[ubatch],
        ).fp4_bands

    first, second, other = prepare(""), prepare(""), prepare("ubatch1")
    assert [b["query_start_loc"].tolist() for b in second] == [[0, 2, 3], [0, 0, 2]]
    assert [b["row_ends"].tolist() for b in second] == [[7, 8, 3], [4, 5, 0]]
    assert all(b["block_tables"].tolist() == [[3, 0], [7, 0]] for b in second)
    assert (
        first[1]["query_start_loc"].data_ptr()
        == second[1]["query_start_loc"].data_ptr()
    )
    assert (
        first[1]["query_start_loc"].data_ptr() != other[1]["query_start_loc"].data_ptr()
    )
    if heads == 64 and max_qlen <= 8:
        assert len(calls) == 6
        assert calls[0][3] is calls[2][3]
        assert calls[0][3] is not calls[4][3]
        for launch, call in zip(second, calls[2:4]):
            assert launch["cfg"] is call[0]
            assert launch["query_start_loc"] is call[1]
            assert launch["row_ends"] is call[2]
    else:
        assert not calls
        assert all(b["schedule"] is None for b in second)


def test_v41_candidate_band_forwards_packed_offsets(monkeypatch):
    from atom.model_ops.indexer import score as scoring

    seen = {}
    monkeypatch.setattr(
        scoring,
        "mqa_logits_mxfp4",
        lambda *args, **kwargs: seen.update(args=args, **kwargs),
    )
    values = torch.empty(3, 1, 32, 64, dtype=torch.uint8)
    scales = torch.empty(3, 1, 32, 4, dtype=torch.uint8)
    weights = torch.empty(3, 32, dtype=torch.bfloat16)
    cache = rows_module.IndexCache(
        torch.empty(8, 8, 64), torch.empty(8, 8, 4), 8, rows_module.NATURAL
    )
    cfg = object()
    launch = {
        "cfg": cfg,
        "schedule": None,
        "query_start_loc": torch.arange(4, dtype=torch.int32),
        "row_ends": torch.tensor([8, 16, 24], dtype=torch.int32),
        "block_tables": torch.zeros(3, 8, dtype=torch.int32),
    }
    rows = rows_module.IndexRows(
        launch["row_ends"],
        64,
        ((0, 3),),
        rows_module.DcpSplit.of(3),
        fp4_bands=(launch,),
    )
    out = torch.empty(48)
    offsets = torch.tensor([0, 8, 24], dtype=torch.int64)
    scoring._fp4_band_logits(
        values, scales, weights, cache, out, offsets, 0.125, 0, rows
    )
    assert seen["out"] is out and seen["out_offsets"] is offsets
    assert seen["cfg"] is cfg and seen["weight_scale"] == 0.125
    assert seen["args"][0].shape == (3, 32, 64)
    assert seen["args"][1].shape == (3, 32, 4)
    assert seen["args"][4] is weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("qlen", [1, 4, 6, 8, 9])
def test_gfx950_query_layout_logits_and_graph_replay(qlen):
    from aiter.jit.utils.chip_info import get_gfx

    from atom.model_ops.indexer.query import quantize_query
    from atom.model_ops.indexer.score import _fp4_band_logits, score_topk

    if get_gfx() != "gfx950":
        pytest.skip("gfx950 OPUS layout")
    torch.manual_seed(901)
    dev, count = "cuda", 2 * qlen + 1
    q = torch.randn(count, 64, 128, device=dev).bfloat16()
    angles = torch.randn(count, 32)
    cos, sin = angles.cos().bfloat16().to(dev), angles.sin().bfloat16().to(dev)
    positions = torch.arange(count, device=dev)
    natural = quantize_query(
        q, cos, sin, positions, 64, fp4=True, layout=rows_module.NATURAL
    )
    query = quantize_query(
        q, cos, sin, positions, 64, fp4=True, layout=rows_module.MFMA
    )
    torch.testing.assert_close(query[0], natural[0], rtol=0, atol=0)

    def permute_scales(scale):
        return (
            scale.reshape(-1, 2, 32, 2, 2)
            .permute(0, 4, 2, 3, 1)
            .reshape(-1, 2, 32, 4)
            .contiguous()
        )

    torch.testing.assert_close(
        query[1].squeeze(1), permute_scales(natural[1]), rtol=0, atol=0
    )
    kv = torch.randint(0, 256, (4, 64, 64), dtype=torch.uint8, device=dev)
    ks = torch.randint(122, 129, (4, 64, 4), dtype=torch.uint8, device=dev)
    cache = rows_module.IndexCache(
        kv.reshape(4, 64, 4, 16).permute(0, 2, 1, 3).contiguous(),
        permute_scales(ks),
        64,
        rows_module.MFMA,
    )
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device=dev)
    starts = torch.tensor([0, qlen, 2 * qlen], dtype=torch.int32, device=dev)
    ends = torch.tensor(
        [*range(65 - qlen, 65), *range(129 - qlen, 129), 0],
        dtype=torch.int32,
        device=dev,
    )
    weights = torch.randn(count, 64, dtype=torch.bfloat16, device=dev)
    storage = rows_module.Fp4LaunchStorage(dev, count, 2)
    logits = torch.full((count, 128), float("nan"), device=dev)
    selected = torch.empty(count, 16, dtype=torch.int32, device=dev)

    def forward():
        sequences = rows_module.Fp4MqaRaggedMetadata(starts, qlen, table)
        rows = rows_module.fp4_rows(
            ends,
            128,
            None,
            sequences,
            heads=64,
            page_size=64,
            layout=rows_module.MFMA,
            storage=storage,
            bands=((0, qlen + 1), (qlen + 1, count)),
        )
        for i, (first, end) in enumerate(rows.bands):
            _fp4_band_logits(
                query[0][first:end],
                query[1][first:end],
                weights[first:end],
                cache,
                logits[first:end],
                None,
                0.125,
                i,
                rows,
            )
        score_topk(
            query, weights, cache, rows, topk=16, weight_scale=0.125, out=selected
        )

    def dequant(values, scales):
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev
        )
        codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2).long()
        return lut[codes] * torch.exp2(scales.float() - 127).repeat_interleave(
            32, dim=-1
        )

    qd = dequant(natural[0].squeeze(1), natural[1].squeeze(1))
    kd = dequant(kv, ks)[table.long()].reshape(2, 128, 128)
    owners = torch.tensor([0] * qlen + [1] * (qlen + 1), device=dev)
    expected = (
        torch.einsum("rhd,rkd->rhk", qd, kd[owners]).relu()
        * weights.float().unsqueeze(-1)
    ).sum(1) * 0.125

    def check():
        live = torch.arange(128, device=dev)[None, :] < ends[:, None]
        torch.testing.assert_close(logits[live], expected[live], atol=2e-3, rtol=2e-4)
        assert torch.isnan(logits[~live]).all()
        assert (selected[-1] == -1).all()
        for r in range(count - 1):
            reference = expected[r, : int(ends[r])]
            threshold = reference.topk(16).values[-1]
            got = selected[r]
            assert (got >= 0).all() and (got < ends[r]).all()
            assert (reference[got.long()] >= threshold - 2e-3).all()

    for _ in range(3):
        forward()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    ends[:-1].sub_(1)
    logits.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check()
