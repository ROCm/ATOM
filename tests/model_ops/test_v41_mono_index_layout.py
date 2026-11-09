# SPDX-License-Identifier: MIT
"""V4.1 Mono reads the same FP4 keys from either physical cache layout."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm required")


def _key_probe(mfma):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import range_constexpr

    from atom.models.deepseek_v41.mono.kernels.index_score_fp4 import STAGE_ARGS, _keys
    from atom.mono.device.ops import ld_i32, rsrc

    @flyc.kernel
    def probe(
        values: fx.Int64,
        scales: fx.Int64,
        pages: fx.Int64,
        out: fx.Int64,
        out_scale: fx.Int64,
    ):
        lane, block = fx.thread_idx.x, fx.block_idx.x
        args = {name: fx.Int64(0) for name in STAGE_ARGS}
        args.update(plane=values, pscale=scales)
        page = ld_i32(pages, block * 8 + lane % 16 // 2)
        kv, ks = _keys({"args": args, "lane": lane, "index_mfma": mfma}, page, False)
        for t in range_constexpr(4):
            bo.buffer_store(kv[t], rsrc(out), ((block * 64 + lane) * 4 + t) * 4)
        bo.buffer_store(ks, rsrc(out_scale), block * 64 + lane)

    @flyc.jit
    def launch(
        values: fx.Int64,
        scales: fx.Int64,
        pages: fx.Int64,
        out: fx.Int64,
        out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        probe(values, scales, pages, out, out_scale).launch(
            grid=(4, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


@pytest.mark.parametrize("mfma", [False, True], ids=["flydsl", "opus"])
def test_mono_key_reader_matches_natural_bytes_and_graph(mfma):
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
        pack_kv_cache,
    )
    from flydsl.expr import Stream

    if get_gfx() != "gfx950":
        pytest.skip("gfx950 layout")
    torch.manual_seed(6952)
    values = torch.randint(0, 256, (256, 64), dtype=torch.uint8, device="cuda")
    scales = torch.randint(0, 256, (256, 4), dtype=torch.uint8, device="cuda")
    if mfma:
        kv = values.view(4, 64, 4, 16).permute(0, 2, 1, 3).contiguous()
        ks = scales.view(4, 2, 32, 2, 2).permute(0, 4, 2, 3, 1).contiguous()
    else:
        kv, ks = pack_kv_cache(values, scales, 8)
    pages = torch.randperm(32, dtype=torch.int32, device="cuda").view(4, 8)
    out = torch.empty(4, 64, 4, 4, dtype=torch.int32, device="cuda")
    out_scale = torch.empty(4, 64, dtype=torch.int32, device="cuda")
    launch = _key_probe(mfma)

    def run():
        launch(
            kv.data_ptr(),
            ks.data_ptr(),
            pages.data_ptr(),
            out.data_ptr(),
            out_scale.data_ptr(),
            Stream(torch.cuda.current_stream()),
        )

    def check():
        lane = torch.arange(64, device="cuda")
        rows = pages[:, lane % 16 // 2].long() * 8 + (lane % 2)[None] * 4
        rows = rows[:, :, None] + torch.arange(4, device="cuda")
        chunks = lane[None, :, None] // 16
        expected = (
            values.view(256, 4, 4, 4)[rows, chunks].contiguous().view(torch.int32)
        )
        torch.testing.assert_close(out, expected.squeeze(-1), rtol=0, atol=0)
        expected_scale = scales[rows, chunks].contiguous().view(torch.int32).squeeze(-1)
        torch.testing.assert_close(out_scale, expected_scale, rtol=0, atol=0)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    pages.copy_(pages.flip((0, 1)))
    graph.replay()
    check()
