# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest


def test_pool_candidates_mix_history_with_fresh_rows():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from atom.model_ops.glm5_next import kpool
    from atom.model_ops.glm5_next.speculative import build_speculative_pool_candidates

    torch.manual_seed(19)
    pool = 4
    dim = 128
    history = torch.randn(2, 2, pool, dim, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(5, dim, device="cuda", dtype=torch.bfloat16)
    gates = torch.randn_like(keys)
    positions = torch.tensor([10, 11, 20, 21, 22], device="cuda")
    cu = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    ape = torch.randn(pool, dim, device="cuda")

    got, requests = build_speculative_pool_candidates(
        history, keys, gates, positions, cu, slots, ape, pool
    )

    expected_keys = []
    expected_gates = []
    for row, (request, start) in enumerate(zip([0, 0, 1, 1, 1], [0, 0, 2, 2, 2])):
        start_position = int(positions[start])
        query_position = int(positions[row])
        row_keys = []
        row_gates = []
        for position in range(query_position - pool + 1, query_position + 1):
            if position < start_position:
                row_keys.append(history[request, 0, position % pool])
                row_gates.append(history[request, 1, position % pool])
            else:
                fresh_row = start + position - start_position
                row_keys.append(keys[fresh_row])
                row_gates.append(gates[fresh_row])
        expected_keys.append(torch.stack(row_keys))
        expected_gates.append(torch.stack(row_gates))
    expected = kpool.pool_compress_ref(
        torch.stack(expected_keys), torch.stack(expected_gates), ape
    )
    expected = kpool.hadamard128_ref(expected.to(torch.bfloat16).float()).to(
        torch.bfloat16
    )

    assert requests.tolist() == [0, 0, 1, 1, 1]
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


def test_history_update_preserves_unwritten_residues():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from atom.model_ops.glm5_next.speculative import update_speculative_kpool_history

    pool = 4
    dim = 128
    history = (
        torch.arange(4 * 2 * pool * dim, device="cuda", dtype=torch.float32)
        .reshape(4, 2, pool, dim)
        .to(torch.bfloat16)
    )
    before = history.clone()
    keys = torch.full((5, dim), 101, device="cuda", dtype=torch.bfloat16)
    gates = torch.full((5, dim), 202, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([10, 11, 20, 21, 22], device="cuda")
    cu = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    slots_in = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    slots_out = torch.tensor([2, 3], device="cuda", dtype=torch.int32)

    update_speculative_kpool_history(
        history, keys, gates, positions, cu, slots_in, slots_out
    )
    torch.cuda.synchronize()

    # Request 0 writes absolute positions 10/11 (ring residues 2/3), copying
    # residues 0/1 from its source slot. Request 1 writes 20/21/22 (0/1/2).
    torch.testing.assert_close(history[2, :, :2], before[0, :, :2])
    assert torch.all(history[2, 0, 2:] == 101)
    assert torch.all(history[2, 1, 2:] == 202)
    assert torch.all(history[3, 0, :3] == 101)
    assert torch.all(history[3, 1, :3] == 202)
    torch.testing.assert_close(history[3, :, 3], before[1, :, 3])


def test_token_indices_map_to_request_cache_slots():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from atom.model_ops.glm5_next.speculative import map_token_indices_to_slots

    token_indices = torch.tensor(
        [[0, 15, 16, 31], [0, 1, 16, 32]],
        device="cuda",
        dtype=torch.int32,
    )
    request_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    block_tables = torch.tensor(
        [[10, 20], [30, 40]],
        device="cuda",
        dtype=torch.int32,
    )
    output_indptr = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
    output = torch.empty(8, device="cuda", dtype=torch.int32)

    map_token_indices_to_slots(
        token_indices,
        request_indices,
        block_tables,
        output_indptr,
        output,
        block_size=16,
    )
    torch.cuda.synchronize()

    assert output.tolist() == [160, 175, 320, 335, 480, 481, 640, 0]
