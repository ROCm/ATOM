# SPDX-License-Identifier: MIT
"""GPU regression for MegaMoE tail-row ownership."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rows", [1, 65, 8192])
def test_tail_copy_owns_rank_slice_and_replays_changed_input(fp8, rows):
    from atom.model_ops.sp_mega_buffers import copy_tail

    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    source = torch.empty((rows * 4, 6144), dtype=dtype, device="cuda")
    source.view(torch.uint8).random_(0, 256)
    scales = torch.rand(rows * 4, device="cuda")
    out = torch.empty((rows + 14, 6144), dtype=dtype, device="cuda")
    out.view(torch.uint8).fill_(42)
    out_scale = torch.full((rows + 14, 1), -1.0, device="cuda")
    own = slice(2 * rows, 3 * rows)

    def run():
        copy_tail(
            source[own],
            scales[own] if fp8 else None,
            out[7:-7],
            out_scale[7:-7],
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    source.view(torch.uint8).random_(0, 256)
    scales.add_(1)
    graph.replay()
    torch.cuda.synchronize()
    expected = source[own].view(torch.uint8).clone()
    torch.testing.assert_close(out[7:-7].view(torch.uint8), expected, atol=0, rtol=0)
    assert (out[:7].view(torch.uint8) == 42).all()
    assert (out[-7:].view(torch.uint8) == 42).all()
    if fp8:
        torch.testing.assert_close(out_scale[7:-7, 0], scales[own], atol=0, rtol=0)
    source.view(torch.uint8).zero_()
    torch.testing.assert_close(out[7:-7].view(torch.uint8), expected, atol=0, rtol=0)
