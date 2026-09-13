# SPDX-License-Identifier: MIT
"""Graph replay refreshes row inputs and isolates padded rows and buckets."""

import pytest
import torch

from atom.models.deepseek_v41.execution import DenseGraphExecutor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_dense_graph_replay_refreshes_inputs_and_padding():
    executor = DenseGraphExecutor()
    torch.manual_seed(41)
    weight = torch.randn(32, 32, device="cuda")

    def stage(residual, mix):
        output = ((residual.float() * mix.unsqueeze(-1)).sum(-2) @ weight).bfloat16()
        return output, mix + 1

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for bucket in [4, 1, 2]:
            x = torch.randn(1, bucket, 4, 32, device="cuda", dtype=torch.bfloat16)
            mix = torch.randn(1, bucket, 4, device="cuda")
            actual = executor.run(stage, x, mix, bucket=bucket, capture=True)
            for a, b in zip(actual, stage(x, mix)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for bucket, length in [(4, 3), (1, 1), (4, 2), (2, 1), (4, 4)]:
            x = torch.randn(1, length, 4, 32, device="cuda", dtype=torch.bfloat16)
            mix = torch.randn(1, length, 4, device="cuda")
            actual = executor.run(stage, x, mix, bucket=bucket)
            for a, b in zip(actual, stage(x, mix)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)
    assert len(executor.entries) == 3 and executor.replays == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_uncaptured_bucket_falls_back_without_recording_request_work():
    executor = DenseGraphExecutor()
    x = torch.ones(1, 3, 32, device="cuda")
    (actual,) = executor.run(lambda value: (value + 2,), x, bucket=4)
    torch.testing.assert_close(actual, x + 2, rtol=0, atol=0)
    assert executor.entries == {} and executor.replays == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_text_capture_cannot_discard_later_image_mask():
    executor = DenseGraphExecutor()

    def stage(x, mask):
        return (x + 0 if mask is None else x.masked_fill(mask[..., None], -7),)

    x = torch.ones(1, 1, 32, device="cuda")
    mask = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        executor.run(stage, x, None, bucket=1, capture=True)
        (actual,) = executor.run(stage, x, mask, bucket=1)
        torch.testing.assert_close(actual, torch.full_like(x, -7), rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)
    assert executor.replays == 1
