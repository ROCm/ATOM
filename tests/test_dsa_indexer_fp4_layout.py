# SPDX-License-Identifier: MIT
"""CPU byte contracts for the production DCP FP4 metadata and staging path."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def staging():
    root = Path(__file__).parents[1] / "atom/model_ops"
    namespace = {
        "torch": torch,
        "np": np,
        "MFMA": "mfma",
        "PRESHUFFLE": "preshuffle",
        "FP4_KV_BLOCK_SIZE": 64,
        "_MFMA_M": 16,
        "upload_numpy": lambda values, device: torch.from_numpy(values.copy()).to(
            device
        ),
    }
    for filename, names in (
        ("indexer/layout.py", {"fp4_scale_byte"}),
        ("sparse_indexer_fp4.py", {"fp4_index_scale_rows"}),
        (
            "attentions/aiter_mla.py",
            {"_build_dcp_indexer_fp4_prefill_meta", "_build_dcp_indexer_prefill_meta"},
        ),
        ("indexer/indexer_forward.py", {"_dcp_stage_indexer_fp4_prefill"}),
    ):
        path = root / filename
        functions = [
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in functions} == names
        exec(  # noqa: S102 -- runs production host bodies without GPU imports
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    return namespace


@pytest.mark.parametrize("layout", ["mfma", "preshuffle"])
@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("interleave", [1, 16])
def test_builder_and_staging_preserve_bytes_across_batches(
    monkeypatch, staging, layout, world, interleave
):
    generator = torch.Generator().manual_seed(91)
    data = [
        torch.randint(256, (12, 64, 64), dtype=torch.uint8, generator=generator)
        for _ in range(world)
    ]
    scales = [
        torch.randint(256, (12, 64, 4), dtype=torch.uint8, generator=generator)
        for _ in range(world)
    ]
    swizzle = (torch.arange(64) % 16) * 4 + torch.arange(64) // 16
    cache = data[0].reshape(12, 64, 4, 16).permute(0, 2, 1, 3).contiguous()
    if layout == "mfma":
        scale_cache = (
            scales[0]
            .reshape(12, 2, 32, 2, 2)
            .permute(0, 4, 2, 3, 1)
            .contiguous()
            .reshape(12, 2, 32, 4)
        )
    else:
        cache = cache.unsqueeze(1)
        scale_cache = torch.empty(12, 1, 4, 64, dtype=torch.uint8)
        scale_cache[:, 0, :, swizzle] = scales[0].transpose(1, 2)
    builder = SimpleNamespace(
        model_runner=SimpleNamespace(block_size=64), device="cpu", block_table_cols=4
    )
    meta = SimpleNamespace()
    previous = None
    for lengths in (np.array([139, 17, 0]), np.array([0, 0, 0]), np.array([65, 1, 34])):
        lpad = -(-lengths // (world * interleave)) * interleave
        cu_pad, g_cu = np.r_[0, lpad.cumsum()], np.r_[0, lengths.cumsum()]
        table = np.array([[7, 3, 0, 0], [11, 2, 9, 0], [5, 1, 0, 0]], dtype=np.int32)
        staging["_build_dcp_indexer_fp4_prefill_meta"](
            builder, meta, 3, lpad, cu_pad, g_cu, lengths, table, layout
        )
        if layout == "mfma":
            assert meta.dcp_indexer_fp4_read_scale_at is not previous
            previous = meta.dcp_indexer_fp4_read_scale_at
            for side in ("read", "stage"):
                for part in ("page", "row", "scale_at"):
                    assert (
                        getattr(meta, f"dcp_indexer_fp4_{side}_{part}").dtype
                        == torch.int64
                    )
        read = torch.as_tensor(
            np.concatenate(
                [
                    table[b, np.arange(n) // 64] * 64 + np.arange(n) % 64
                    for b, n in enumerate(lpad)
                ]
            )
        )
        peer_data = [plane.reshape(-1, 64)[read] for plane in data]
        peer_scales = [plane.reshape(-1, 4)[read] for plane in scales]
        peers = (
            [torch.cat((d, s), 1) for d, s in zip(peer_data, peer_scales)]
            if layout == "mfma"
            else None
        )
        order, destinations, wanted_data, wanted_scales = [], [], [], []
        first_page = 0
        for b, length in enumerate(lengths):
            for pos in range(length):
                rank = (pos // interleave) % world
                local = (pos // (world * interleave)) * interleave + pos % interleave
                slot = int(table[b, local // 64]) * 64 + local % 64
                order.append(rank * int(cu_pad[-1]) + int(cu_pad[b]) + local)
                destinations.append(first_page * 64 + pos)
                wanted_data.append(data[rank].reshape(-1, 64)[slot])
                wanted_scales.append(scales[rank].reshape(-1, 4)[slot])
            first_page += -(-length // 64)
        meta.dcp_indexer_gather_index = torch.tensor(order, dtype=torch.int64)
        calls = []

        def gather(
            values,
            dim,
            peers=peers,
            peer_data=peer_data,
            peer_scales=peer_scales,
            calls=calls,
        ):
            assert values.dtype == torch.uint8 and dim == 0
            if layout == "mfma":
                inputs = peers
            elif not calls:
                inputs = [p.reshape(-1, 1, 4, 16) for p in peer_data]
            else:
                inputs = [p.reshape(-1, 1, 4) for p in peer_scales]
            assert torch.equal(values, inputs[0])
            calls.append(values)
            return torch.cat(inputs, dim=0)

        staging["get_dcp_group"] = lambda: SimpleNamespace(all_gather=gather)
        with monkeypatch.context() as guard:
            guard.setitem(
                staging,
                "fp4_scale_byte",
                lambda *a: pytest.fail("layer rebuilt scale indices"),
            )
            result, result_scale = staging["_dcp_stage_indexer_fp4_prefill"](
                cache, scale_cache, meta, layout
            )
        assert len(calls) == (1 if layout == "mfma" else 2)
        if layout == "preshuffle":
            result = result[:, 0]
            result_scale = result_scale[:, 0, :, swizzle].transpose(1, 2)
        else:
            result_scale = (
                result_scale.reshape(first_page, 2, 32, 2, 2)
                .permute(0, 4, 2, 3, 1)
                .contiguous()
                .reshape(first_page, 64, 4)
            )
        result = result.permute(0, 2, 1, 3).reshape(first_page, 64, 64)
        expected_data = torch.zeros(first_page * 64, 64, dtype=torch.uint8)
        expected_scale = torch.zeros(first_page * 64, 4, dtype=torch.uint8)
        if destinations:
            expected_data[destinations] = torch.stack(wanted_data)
            expected_scale[destinations] = torch.stack(wanted_scales)
        assert result.dtype == result_scale.dtype == torch.uint8
        assert torch.equal(result.reshape(-1, 64), expected_data)
        assert torch.equal(result_scale.reshape(-1, 4), expected_scale)


@pytest.mark.parametrize("layout", ["mfma", "preshuffle"])
def test_prefill_builder_passes_the_active_layout(staging, layout):
    seen = []
    builder = SimpleNamespace(
        model_runner=SimpleNamespace(
            config=SimpleNamespace(index_cache_layout=layout),
            forward_vars={"prefill_block_tables": SimpleNamespace(np=np.zeros((1, 1)))},
        ),
        _indexer_fp4=True,
        _dcp_shard_lengths=lambda *a: (
            np.array([0, 1]),
            np.array([1]),
            np.array([1]),
            np.array([0, 1]),
        ),
        _dcp_gathers=lambda *a: [(1, object(), object())],
        _build_dcp_indexer_fp4_prefill_meta=lambda *args: seen.append(args),
    )
    staging["_build_dcp_indexer_prefill_meta"](
        builder, SimpleNamespace(), 1, "prefill_"
    )
    assert seen[0][-1] == layout
