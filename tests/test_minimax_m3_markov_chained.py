# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 draft's chained sharded Markov block
(`dspark._markov_stage1_chained`, stage 2, the pair exchange) against the
whole-vocab `dspark_markov_argmax` position by position, the TP4 exchange
emulated on one GPU: the same ids."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("bs", [1, 3, 16, 64])
def test_chained_matches_whole_vocab(bs):
    import triton

    from atom.model_ops.dspark_markov_argmax import (
        BLOCK_K,
        BLOCK_V,
        _dspark_markov_argmax_stage2,
        argmax_gather,
        dspark_markov_argmax,
        markov_launch_geometry,
    )
    from atom.models.minimax_m3.dspark import _markov_stage1_chained

    torch.manual_seed(bs)
    dev = torch.device("cuda")
    W, VS, R, T = 4, 50016, 256, 7
    V = W * VS
    w1 = (torch.randn(V, R, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(V, R, device=dev) * 0.05).to(torch.bfloat16)
    base = (torch.randn(bs, T, V, device=dev) * 2).to(torch.bfloat16)
    anchors = torch.randint(0, V, (bs,), device=dev, dtype=torch.int32)

    ref = torch.empty(bs, T + 1, dtype=torch.int32, device=dev)
    ref[:, 0] = anchors
    for k in range(T):
        dspark_markov_argmax(base[:, k], ref[:, k], w1, w2, ref[:, k + 1])

    out = torch.empty(bs, T, dtype=torch.int32, device=dev)
    geometry = markov_launch_geometry(bs, VS)
    tiles = geometry.v_tiles
    gathered = torch.zeros(W * bs, 2, device=dev)
    for k in range(T):
        packs = []
        for r in range(W):
            shard = base[:, k, r * VS : (r + 1) * VS]
            part_val = torch.empty((tiles, bs), dtype=torch.float32, device=dev)
            part_idx = torch.empty((tiles, bs), dtype=torch.int32, device=dev)
            packed = torch.empty((bs, 2), dtype=torch.float32, device=dev)
            _markov_stage1_chained[(tiles, geometry.row_tiles)](
                shard, anchors, gathered, out[:, max(k - 1, 0)], w1, w2, part_val, part_idx,
                bs, VS, R, shard.stride(0), anchors.stride(0), out.stride(0), w1.stride(0),
                w2.stride(0), r * VS, FIRST=k == 0, W=W, BLOCK_ROW=geometry.block_row,
                BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K, num_warps=geometry.num_warps, num_stages=2,
            )  # fmt: skip
            _dspark_markov_argmax_stage2[(bs,)](
                part_val, part_idx, part_idx, bs, tiles, 0, packed,
                BLOCK_TILE=triton.next_power_of_2(tiles), PACKED=True, num_warps=4, num_stages=1,
            )  # fmt: skip
            packs.append(packed)
        gathered = torch.cat(packs, dim=0)
    argmax_gather(gathered.view(W, bs, 2), out[:, T - 1])
    torch.testing.assert_close(out, ref[:, 1:], atol=0, rtol=0)


@pytest.mark.parametrize("bs", [1, 3, 16, 64])
def test_atomic_matches_whole_vocab(bs):
    import triton

    from atom.model_ops.dspark_markov_argmax import BLOCK_K, BLOCK_V, dspark_markov_argmax, markov_launch_geometry
    from atom.models.minimax_m3.dspark import _markov_last_atomic, _markov_step_atomic

    torch.manual_seed(100 + bs)
    dev = torch.device("cuda")
    W, VS, R, T = 4, 50016, 256, 7
    V = W * VS
    w1 = (torch.randn(V, R, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(V, R, device=dev) * 0.05).to(torch.bfloat16)
    base = (torch.randn(bs, T, V, device=dev) * 2).to(torch.bfloat16)
    # a tie across two ranks' shards at position 0 of row 0: the lower id wins
    base[0, 0, 5] = base[0, 0, 2 * VS + 5] = 80.0
    anchors = torch.randint(0, V, (bs,), device=dev, dtype=torch.int32)
    ref = torch.empty(bs, T + 1, dtype=torch.int32, device=dev)
    ref[:, 0] = anchors
    for k in range(T):
        dspark_markov_argmax(base[:, k], ref[:, k], w1, w2, ref[:, k + 1])

    for rounds in range(2):  # the buffers come back to their reset value
        packed = [torch.full((T, bs), -(2**63), dtype=torch.int64, device=dev) for _ in range(W)]
        if rounds:
            packed = prev_packed
        out = torch.empty(bs, T, dtype=torch.int32, device=dev)
        geometry = markov_launch_geometry(bs, VS)
        gathered = None
        for k in range(T):
            for r in range(W):
                shard = base[:, k, r * VS : (r + 1) * VS]
                _markov_step_atomic[(geometry.v_tiles, geometry.row_tiles)](
                    shard, anchors, packed[r][k] if gathered is None else gathered,
                    packed[r][max(k - 1, 0)], out[:, max(k - 1, 0)], w1, w2, packed[r][k],
                    bs, VS, R, shard.stride(0), anchors.stride(0), out.stride(0), w1.stride(0),
                    w2.stride(0), r * VS, FIRST=k == 0, W=W, BLOCK_ROW=geometry.block_row,
                    BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K, num_warps=geometry.num_warps, num_stages=2,
                )  # fmt: skip
            gathered = torch.cat([packed[r][k] for r in range(W)]).clone()
        for r in range(W):
            _markov_last_atomic[(1,)](
                gathered, packed[r][T - 1], out[:, T - 1], bs, out.stride(0), W=W,
                BLOCK_ROW=triton.next_power_of_2(bs), num_warps=1,
            )  # fmt: skip
        torch.testing.assert_close(out, ref[:, 1:], atol=0, rtol=0)
        for r in range(W):
            assert int(packed[r].max()) == -(2**63), "packed bests not reset"
        prev_packed = packed
