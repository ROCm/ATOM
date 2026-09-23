# SPDX-License-Identifier: MIT
"""`engram_row_indices` against the published arithmetic, bit for bit.

A row index is discrete: an implementation that agrees to within anything at
all still reads a different embedding row, so the only useful tolerance is
zero. The cases below are the ones where the two implementations could diverge
without the shapes disagreeing -- a lookback that crosses out of the chunk into
the cursor, a DEAD marker that has to latch, a masked (image) token, a request
whose chunk is shorter than the lookback, and the largest compressed id, which
is where `token * multiplier` comes within 22771 of the sign bit.

Needs triton and a GPU, so CI never runs it (see
`test_hash_bounds.py` for the part that does). Run it on the box and
put the result in the commit message.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

from atom.model_ops.engram.device.hashing import (
    EngramHashTables,
    engram_compress,
    engram_cursor_rows,
    engram_cursor_rows_reference,
    engram_row_indices,
    engram_row_indices_reference,
)
from atom.model_ops.engram.mapping import EngramConfig
from tests.model_ops.engram.test_hash_bounds import (
    V41_FLASH,
    build,
    tiny_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to run the kernel"
)


def case(mapping, lengths, *, seed, dead=(), masked=(), extreme=False):
    """A ragged batch plus its per-request history, on the host."""
    rng = np.random.default_rng(seed)
    vocab = mapping.tokenizer_vocab_size
    width = mapping.config.max_ngram_size - 1
    tokens = [rng.integers(0, vocab, size=n, dtype=np.int64) for n in lengths]
    if extreme:
        tokens[0][:] = vocab - 1
    histories = [rng.integers(0, vocab, size=width, dtype=np.int64) for _ in lengths]
    for request, column in dead:
        histories[request][column] = -1
    masks = None
    if masked:
        masks = [np.ones(n, dtype=bool) for n in lengths]
        for request, position in masked:
            masks[request][position] = False
    return tokens, histories, masks


def run(mapping, tokens, histories, masks, device, *, permute=False, strided=False):
    """Kernel rows, with the history plane laid out the way production has it."""
    tables = EngramHashTables.from_mapping(mapping, device)
    lengths = [len(t) for t in tokens]
    width = mapping.config.max_ngram_size - 1
    flat = torch.as_tensor(np.concatenate(tokens), dtype=torch.int32, device=device)
    batch_ids = torch.as_tensor(
        np.repeat(np.arange(len(lengths)), lengths).astype(np.int32), device=device
    )
    cu = torch.as_tensor(
        np.concatenate(([0], np.cumsum(lengths))).astype(np.int32), device=device
    )
    index = np.arange(len(lengths))
    if permute:
        index = index[::-1].copy()
    plane = np.zeros((len(lengths), width), dtype=np.int64)
    plane[index] = np.stack(histories)
    if strided:
        # The committed cursor is `[slots, 1 + width]` with the position in
        # column 0, so production hands this function a slice, not a tensor.
        wide = torch.zeros(len(lengths), width + 1, dtype=torch.int64, device=device)
        wide[:, 1:] = torch.as_tensor(plane, device=device)
        history = wide[:, 1:]
    else:
        history = torch.as_tensor(plane, device=device)
    return engram_row_indices(
        tables,
        mapping.config.layer_ids[0],
        compress(tables, flat, masks, device),
        batch_ids,
        cu,
        history,
        torch.as_tensor(index.astype(np.int32), device=device),
        torch.empty(
            len(flat), mapping.config.num_hash_heads, dtype=torch.int64, device=device
        ),
    )


def compress(tables, flat, masks, device):
    """`masks` are the reference's keep sense; the kernel takes its complement."""
    dead = (
        None
        if masks is None
        else torch.as_tensor(~np.concatenate(masks), device=device)
    )
    return engram_compress(tables, flat, dead)


@pytest.fixture(params=["v41-flash", "tiny"])
def mapping(request):
    return build(
        EngramConfig.from_hf(V41_FLASH)
        if request.param == "v41-flash"
        else tiny_config()
    )


@pytest.mark.parametrize(
    "name,lengths,kwargs",
    [
        ("decode", [1, 1, 1], {}),
        ("verify", [6, 6, 6], {}),
        ("ragged", [1, 6, 3, 2], {}),
        ("prefill", [64], {}),
        ("shorter than the lookback", [1, 2], {}),
        ("dead history", [4, 4], {"dead": [(0, 0), (1, -2)]}),
        ("dead at the newest", [4], {"dead": [(0, -1)]}),
        ("masked token", [5, 5], {"masked": [(0, 0), (1, 3)]}),
        ("largest compressed id", [4, 4], {"extreme": True}),
    ],
)
def test_kernel_matches_reference(mapping, name, lengths, kwargs):
    tokens, histories, masks = case(mapping, lengths, seed=len(name), **kwargs)
    rows = run(mapping, tokens, histories, masks, torch.device("cuda"))
    expected = engram_row_indices_reference(
        mapping, mapping.config.layer_ids[0], tokens, histories, masks
    )
    np.testing.assert_array_equal(rows.cpu().numpy(), expected)


def test_every_layer_uses_its_own_multipliers(mapping):
    """Two layers must not produce the same rows for the same tokens."""
    device = torch.device("cuda")
    tokens, histories, _ = case(mapping, [5, 5], seed=7)
    tables = EngramHashTables.from_mapping(mapping, device)
    produced = []
    for layer_id in mapping.config.layer_ids:
        rows = run(mapping, tokens, histories, None, device)
        expected = engram_row_indices_reference(mapping, layer_id, tokens, histories)
        produced.append(expected)
        if layer_id == mapping.config.layer_ids[0]:
            np.testing.assert_array_equal(rows.cpu().numpy(), expected)
    assert not np.array_equal(produced[0], produced[1])
    assert set(tables.layers) == set(mapping.config.layer_ids)


def test_history_plane_may_be_a_strided_slice(mapping):
    """Production passes `cursor[:, 1:]`; the kernel must read the same ids."""
    device = torch.device("cuda")
    tokens, histories, _ = case(mapping, [3, 4], seed=11)
    dense = run(mapping, tokens, histories, None, device)
    sliced = run(mapping, tokens, histories, None, device, strided=True)
    torch.testing.assert_close(dense, sliced, rtol=0, atol=0)


def test_history_row_is_taken_from_the_index_not_the_batch(mapping):
    """A request reads the slot it owns, not the position it sits at."""
    device = torch.device("cuda")
    tokens, histories, _ = case(mapping, [3, 4], seed=13)
    rows = run(mapping, tokens, histories, None, device, permute=True)
    expected = engram_row_indices_reference(
        mapping, mapping.config.layer_ids[0], tokens, histories
    )
    np.testing.assert_array_equal(rows.cpu().numpy(), expected)


def cursor_inputs(mapping, tokens, histories, positions, device, masks=None):
    """The device tensors both cursor modes take, plus the cursor plane itself."""
    lengths = [len(t) for t in tokens]
    width = mapping.config.max_ngram_size - 1
    flat = torch.as_tensor(np.concatenate(tokens), dtype=torch.int32, device=device)
    cu = torch.as_tensor(
        np.concatenate(([0], np.cumsum(lengths))).astype(np.int32), device=device
    )
    ramp = np.concatenate(
        [positions[i] + np.arange(n) for i, n in enumerate(lengths)]
    ).astype(np.int64)
    cursor = torch.zeros(len(lengths), width + 1, dtype=torch.int64, device=device)
    cursor[:, 0] = torch.as_tensor(np.asarray(positions), device=device)
    cursor[:, 1:] = torch.as_tensor(np.stack(histories), device=device)
    tables = EngramHashTables.from_mapping(mapping, device)
    compressed = compress(tables, flat, masks, device)
    return compressed, cu, torch.as_tensor(ramp, device=device), cursor


@pytest.mark.parametrize(
    "lengths,positions",
    [([1, 1], [9, 40]), ([6, 6], [0, 512]), ([1, 6, 3], [7, 8, 9]), ([12], [3])],
)
def test_candidate_cursors_match_reference(mapping, lengths, positions):
    """Every prefix a verify step could accept, against `advance_history`."""
    device = torch.device("cuda")
    tokens, histories, _ = case(mapping, lengths, seed=sum(lengths))
    tables = EngramHashTables.from_mapping(mapping, device)
    flat, cu, ramp, cursor = cursor_inputs(
        mapping, tokens, histories, positions, device
    )
    width = mapping.config.max_ngram_size - 1
    plane = torch.zeros(
        len(lengths), max(lengths), width + 1, dtype=torch.int64, device=device
    )
    engram_cursor_rows(
        tables,
        flat,
        cu,
        ramp,
        cursor[:, 1:],
        torch.arange(len(lengths), dtype=torch.int32, device=device),
        plane,
        candidates=max(lengths),
    )
    expected = engram_cursor_rows_reference(mapping, tokens, histories, positions)
    for index, rows in enumerate(expected):
        np.testing.assert_array_equal(
            plane[index, : len(rows)].cpu().numpy(), rows, err_msg=f"request {index}"
        )


@pytest.mark.parametrize("lengths", [[1, 2], [3, 1], [5, 5], [9]])
def test_committed_cursor_is_written_over_its_own_history(mapping, lengths):
    """`candidates=0` writes the cursor whose history it is reading.

    `out[1]` and `history[0]` are the same word, and for any prefix shorter
    than the history the windows overlap, so a kernel that stored before it
    had loaded the row would feed itself.
    """
    device = torch.device("cuda")
    positions = [11 * (i + 1) for i in range(len(lengths))]
    tokens, histories, _ = case(mapping, lengths, seed=len(lengths) + 100)
    tables = EngramHashTables.from_mapping(mapping, device)
    flat, cu, ramp, cursor = cursor_inputs(
        mapping, tokens, histories, positions, device
    )
    engram_cursor_rows(
        tables,
        flat,
        cu,
        ramp,
        cursor[:, 1:],
        torch.arange(len(lengths), dtype=torch.int32, device=device),
        cursor,
    )
    expected = engram_cursor_rows_reference(mapping, tokens, histories, positions)
    for index, rows in enumerate(expected):
        np.testing.assert_array_equal(
            cursor[index].cpu().numpy(), rows[-1], err_msg=f"request {index}"
        )


def test_committed_cursor_follows_the_slot_and_the_mask(mapping):
    """Rows land by slot, and a masked token is DEAD in the history."""
    device = torch.device("cuda")
    lengths, positions = [2, 3], [4, 5]
    tokens, histories, masks = case(mapping, lengths, seed=21, masked=[(0, 0), (1, 1)])
    tables = EngramHashTables.from_mapping(mapping, device)
    flat, cu, ramp, cursor = cursor_inputs(
        mapping, tokens, histories, positions, device, masks
    )
    slots = torch.tensor([1, 0], dtype=torch.int32, device=device)
    plane = torch.zeros_like(cursor)
    plane[:, 1:] = cursor[:, 1:][slots.long()]
    engram_cursor_rows(tables, flat, cu, ramp, plane[:, 1:], slots, plane)
    expected = engram_cursor_rows_reference(
        mapping, tokens, histories, positions, masks
    )
    for index, rows in enumerate(expected):
        np.testing.assert_array_equal(
            plane[int(slots[index])].cpu().numpy(), rows[-1], err_msg=f"request {index}"
        )


def test_non_contiguous_history_row_is_refused(mapping):
    device = torch.device("cuda")
    tables = EngramHashTables.from_mapping(mapping, device)
    width = mapping.config.max_ngram_size - 1
    plane = torch.zeros(2, width, 2, dtype=torch.int64, device=device)[..., 0]
    with pytest.raises(ValueError, match="contiguous"):
        engram_row_indices(
            tables,
            mapping.config.layer_ids[0],
            torch.zeros(2, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
            torch.tensor([0, 2], dtype=torch.int32, device=device),
            plane,
            torch.zeros(2, dtype=torch.int32, device=device),
            torch.zeros(2, tables.heads, dtype=torch.int64, device=device),
        )
