# SPDX-License-Identifier: MIT
"""Rank ordering, padded BF16 bits and fallback layouts for prefill Q gather."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a GPU for the fused reorder kernel", allow_module_level=True)

from atom.model_ops.dcp_ops import dcp_all_gather_prefill_query_heads


class RecordedPeers:
    """Supply peer tensors; real DCP2/4 collectives have separate native tests."""

    device_communicator = None

    def __init__(self, peers):
        self.peers = peers
        self.world_size = len(peers)

    def all_gather(self, x, dim):
        return torch.cat([peer.reshape(x.shape) for peer in self.peers], dim=dim)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "world,tokens,heads,dim,padded,layout",
    [
        (2, 1, 12, 576, 32, "contiguous"),
        (4, 33, 24, 576, 128, "contiguous"),
        (4, 3, 12, 575, 64, "contiguous"),
        (4, 7, 16, 576, 64, "contiguous"),
        (2, 3, 12, 576, 32, "token_stride"),
        (4, 3, 24, 576, 128, "feature_stride"),
    ],
)
def test_rank_order_padding_and_changed_input_graph(
    dtype, world, tokens, heads, dim, padded, layout
):
    device = torch.device("cuda", torch.cuda.current_device())
    bit_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    peers = []
    for _ in range(world):
        shape = (
            tokens * (2 if layout == "token_stride" else 1),
            heads,
            dim * (2 if layout == "feature_stride" else 1),
        )
        tensor = torch.empty(shape, device=device, dtype=dtype)
        if layout == "token_stride":
            tensor = tensor[::2]
        if layout == "feature_stride":
            tensor = tensor[..., ::2]
        peers.append(tensor)
    group = RecordedPeers(peers)

    def update(phase):
        cpu_peers = []
        for rank, peer in enumerate(peers):
            values = (
                (
                    torch.arange(peer.numel(), dtype=torch.int32) * 17
                    + rank * 101
                    + phase * 8191
                )
                .to(bit_dtype)
                .reshape(peer.shape)
            )
            # Raw float32 patterns also explicitly include signed zero and NaNs.
            if dtype == torch.float32:
                values.view(-1)[:4] = torch.tensor(
                    [0, -2147483648, 2139095040, 2143289361], dtype=torch.int32
                )
            peer.copy_(values.view(dtype))
            cpu_peers.append(values)
        expected = torch.zeros(tokens, padded, dim, dtype=bit_dtype)
        expected[:, : world * heads] = torch.cat(cpu_peers, dim=1)
        return expected

    def run():
        return dcp_all_gather_prefill_query_heads(group, peers[0], padded)

    expected = update(0)
    actual = run()
    assert actual.shape == (tokens, padded, dim)
    assert actual.is_contiguous() and actual.dtype == dtype
    assert torch.equal(actual.cpu().view(bit_dtype), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = run()
    torch.cuda.current_stream().wait_stream(stream)
    for phase in (1, 2, 3):
        expected = update(phase)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual.cpu().view(bit_dtype), expected)
    graph.reset()
