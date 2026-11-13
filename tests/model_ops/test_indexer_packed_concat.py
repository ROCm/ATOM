# SPDX-License-Identifier: MIT
"""FP8 prefill: host planning, packed scoring and request/global top-k ids."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mla = pytest.importorskip("atom.model_ops.attentions.aiter_mla", exc_type=ImportError)
from atom.model_ops.indexer import score
from atom.model_ops.indexer.rows import ConcatKeys, RowSplit, concat_rows
from atom.model_ops.indexer.workspace import band_offsets
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("request_local", [False, True])
@pytest.mark.parametrize("world", [1, 2, 16])
@pytest.mark.parametrize("mode", ["plain", "pcp", "ubatch"])
@pytest.mark.parametrize("stable", [False, True])
def test_builder_packed_concat_matches_dense(
    monkeypatch, request_local, world, mode, stable
):
    """Real kernels consume the builder's plan, including padding and empty ranks."""
    if not score.FP8_MQA_LOGITS_TAKES_PACKED:
        pytest.skip("AITER fp8_mqa_logits(out_offsets=) is required")
    torch.manual_seed(3043)
    device, heads, dim, topk = "cuda", 32, 128, 64
    prefix = "ubatch_0_" if mode == "ubatch" else ""
    full_visible = np.array([510, 511, 512, 513, 767, 768, 769], np.int32)
    cu_q, cu_k = np.array([0, 4, 7], np.int32), np.array([0, 513, 1282], np.int32)
    starts = [0] * 4 + [513] * 3
    pcp = mode == "pcp"
    if pcp:
        monkeypatch.setattr(mla, "get_pcp_world_size", lambda: 2)
        monkeypatch.setattr(mla, "get_pcp_rank", lambda: 1)
        visible = np.array([511, 513, 768, 0], np.int32)
        starts = [0, 0, 513, 0]
    else:
        visible = full_visible
    count = len(visible)
    starts_gpu = torch.tensor(starts, device=device, dtype=torch.int32)
    visible_gpu = torch.tensor(visible, device=device, dtype=torch.int32)
    q = torch.randn(count, heads, dim, device=device, dtype=torch.float32).to(
        score.dtypes.fp8
    )
    k = torch.randn(1282, dim, device=device, dtype=torch.float32).to(score.dtypes.fp8)
    scales = torch.rand(1282, 1, device=device, dtype=torch.float32) + 0.5
    weights = torch.rand(count, heads, device=device, dtype=torch.float32)
    dense = score.fp8_mqa_logits(
        q, k, scales, weights, starts_gpu, starts_gpu + visible_gpu
    )
    expected = torch.empty(count, topk, device=device, dtype=torch.int32)
    score.top_k_per_row_prefill(
        dense,
        starts_gpu,
        starts_gpu + visible_gpu,
        expected,
        None,
        count,
        dense.stride(0),
        1,
        k=topk,
        stable=stable,
    )
    if request_local:
        expected = torch.where(expected < 0, expected, expected - starts_gpu[:, None])
    workspace = LogitsWorkspace(1536, device)
    # Guards also cover stores beyond the workspace's declared capacity.
    storage = torch.full(
        (workspace.capacity + 34,), float("nan"), device=device, dtype=torch.float32
    )
    workspace._logits = storage[17:-17]
    actual = torch.full_like(expected, -99)
    observed, covered = [], []
    for rank in range(world):
        buffers = {
            prefix
            + name: SimpleNamespace(
                np=np.zeros(16, np.int32),
                gpu=torch.empty(16, device=device, dtype=torch.int32),
            )
            for name in ("index_row_ends", "index_logits_offsets")
        }
        buffers[prefix + "cu_seqlens_q"] = SimpleNamespace(np=cu_q)
        buffers[prefix + "cu_seqlens_k"] = SimpleNamespace(np=cu_k)
        builder = mla.AiterMLAMetadataBuilder.__new__(mla.AiterMLAMetadataBuilder)
        builder.model_runner = SimpleNamespace(forward_vars=buffers)
        builder.is_sparse, builder._indexer_fp4, builder.index_kpool = True, False, 1
        builder.dcp_rank, builder.dcp_world_size = rank, world
        builder.index_logits_workspace = workspace
        builder.index_prefill_key_workspace = None
        counts = {}
        staged = builder._stage_prefill_index_rows(
            2,
            2,
            full_visible,
            pcp,
            True,
            SimpleNamespace(set_counts=counts.update),
            prefix,
        )
        for name, size in counts.items():
            buffers[name].gpu[:size].copy_(
                torch.tensor(buffers[name].np[:size], device=device)
            )
        meta = SimpleNamespace(
            has_cached=True,
            block_tables=None,
            cu_seqlens_q=torch.tensor(cu_q, device=device),
            cu_seqlens_k=torch.tensor(cu_k, device=device),
            cu_seqlen_ks=starts_gpu,
        )
        builder._finish_prefill_index_rows(meta, 2, staged, 1282, prefix, pcp)
        rows = meta.index_rows
        assert (
            rows.packed_offsets.data_ptr()
            == buffers[prefix + "index_logits_offsets"].gpu.data_ptr()
        )
        assert rows.width == max(visible)
        covered.extend(i for first, end in rows.bands for i in range(first, end))

        def observe(logits, span, rows=rows):
            assert (
                logits.ndim == 1 and logits.data_ptr() == workspace._logits.data_ptr()
            )
            offsets = rows.packed_offsets[span].tolist()
            for row, offset in zip(range(span.start, span.stop), offsets):
                length = int(visible[row])
                torch.testing.assert_close(
                    logits[offset : offset + length],
                    dense[row, starts[row] : starts[row] + length],
                    rtol=0,
                    atol=0,
                )
            observed.append(span)

        score._score_bands(
            q,
            weights,
            None,
            rows,
            topk,
            1.0,
            stable,
            observe,
            actual,
            (k, scales),
            request_local,
        )
    assert covered == list(range(count))
    assert len(observed) > 1
    if stable:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        valid = actual >= 0
        assert torch.equal(valid.sum(1), visible_gpu.clamp_max(topk).to(torch.int64))
        columns = actual if request_local else actual - starts_gpu[:, None]
        assert ((columns[valid] >= 0) & (columns < visible_gpu[:, None])[valid]).all()
        ordered = actual.sort(1).values
        assert ((ordered[:, 1:] < 0) | (ordered[:, 1:] != ordered[:, :-1])).all()

        def selected_scores(indices):
            positions = indices + starts_gpu[:, None] if request_local else indices
            values = dense.gather(1, positions.clamp_min(0).long())
            return values.masked_fill(indices < 0, -float("inf")).sort(1).values

        torch.testing.assert_close(
            selected_scores(actual), selected_scores(expected), rtol=0, atol=0
        )
    assert torch.isnan(storage[:17]).all() and torch.isnan(storage[-17:]).all()


def test_old_aiter_keeps_rectangular_metadata(monkeypatch):
    monkeypatch.setattr(mla, "FP8_MQA_LOGITS_TAKES_PACKED", False)
    builder = mla.AiterMLAMetadataBuilder.__new__(mla.AiterMLAMetadataBuilder)
    builder.is_sparse, builder._indexer_fp4, builder.index_kpool = True, False, 1
    mirror = SimpleNamespace(np=np.zeros(4, np.int32))
    builder.model_runner = SimpleNamespace(forward_vars={"index_row_ends": mirror})
    counts = {}
    staged = builder._stage_prefill_index_rows(
        1,
        1,
        np.array([1, 2, 3], np.int32),
        False,
        False,
        SimpleNamespace(set_counts=counts.update),
        "",
    )
    assert staged.packed_bands is None
    assert counts == {"index_row_ends": 3}
    assert mirror.np.tolist() == [1, 2, 3, 0]


@pytest.fixture(scope="module")
def dcp_process_group():
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    yield rank, device
    dist.destroy_process_group()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="run with torchrun --nproc-per-node=2",
)
@pytest.mark.parametrize("topk", [64, 2048])
def test_packed_concat_dcp_collective(monkeypatch, topk, dcp_process_group):
    """Two ranks gather key shards and exchange packed-scored row selections."""
    if not score.FP8_MQA_LOGITS_TAKES_PACKED:
        pytest.skip("AITER fp8_mqa_logits(out_offsets=) is required")
    import torch.distributed as dist

    rank, device = dcp_process_group
    try:
        monkeypatch.setattr(
            score,
            "get_dcp_group",
            lambda: SimpleNamespace(device_group=dist.group.WORLD),
        )
        torch.manual_seed(3043)
        count, keys, heads, dim = 13, 8192, 32, 128
        q = torch.randn(count, heads, dim, device=device, dtype=torch.float32).to(
            score.dtypes.fp8
        )
        full_k = torch.randn(keys, dim, device=device, dtype=torch.float32).to(
            score.dtypes.fp8
        )
        weights = torch.rand(count, heads, device=device, dtype=torch.float32)
        gathered = torch.empty(keys, dim, device=device, dtype=torch.uint8)
        dist.all_gather_into_tensor(
            gathered, full_k[rank * 4096 : (rank + 1) * 4096].view(torch.uint8)
        )
        scales = torch.ones(keys, 1, device=device, dtype=torch.float32)
        starts = torch.tensor([0] * 6 + [4096] * 7, device=device, dtype=torch.int32)
        lengths = np.array(
            [1, 63, 2047, 2048, 2049, 4096, 1, 64, 2047, 2048, 2049, 4095, 4096],
            np.int32,
        )
        visible = torch.tensor(lengths, device=device)
        dense = score.fp8_mqa_logits(
            q, full_k, scales, weights, starts, starts + visible
        )
        expected = torch.empty(count, topk, device=device, dtype=torch.int32)
        score.top_k_per_row_prefill(
            dense,
            starts,
            starts + visible,
            expected,
            None,
            count,
            dense.stride(0),
            1,
            k=topk,
            stable=True,
        )
        expected = torch.where(expected < 0, expected, expected - starts[:, None])
        split = RowSplit.of(count, rank, 2)
        workspace = LogitsWorkspace(8192, device)
        host_offsets = np.zeros(count, np.int32)
        local = band_offsets(
            lengths[split.first : split.end],
            workspace.capacity,
            host_offsets[split.first : split.end],
        )
        rows = concat_rows(
            visible,
            ConcatKeys(None, None, keys, starts),
            split,
            workspace,
            requests=(np.array([0, 6, 13]), np.array([0, 4096, 8192])),
            bands=tuple(
                (first + split.first, end + split.first) for first, end in local
            ),
            packed_offsets=torch.tensor(host_offsets, device=device),
            width=4096,
        )
        actual = score.score_topk(
            q,
            weights,
            None,
            rows,
            topk=topk,
            stable=True,
            keys=(gathered.view(score.dtypes.fp8), scales),
        )
        torch.testing.assert_close(actual[:count], expected, rtol=0, atol=0)
    finally:
        dist.barrier()
