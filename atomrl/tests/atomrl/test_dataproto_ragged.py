# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DataProto.ragged must follow rows through every reindexing operation.

R3 ships per-row MoE routing this way, and a silent misalignment there would not
crash - it would inject the wrong token's expert choices and look like a
convergence bug. CPU-only. Run: python -m atomrl.tests.test_dataproto_ragged
"""

import torch
from atomrl.core.protocol import DataProto


def _proto(n=6):
    """Row i carries tensor value i and a ragged payload of length i+1 filled with i."""
    return DataProto(
        tensors={"x": torch.arange(n).unsqueeze(1).float()},
        ragged={"routing": [[i] * (i + 1) for i in range(n)]},
        meta={"unrelated": "kept"},
    )


def _rows(p):
    """Recover the original row id from both columns; they must agree."""
    xs = [int(v) for v in p.tensors["x"].squeeze(1).tolist()]
    rs = [r[0] for r in p.ragged["routing"]]
    assert xs == rs, f"ragged desynced from tensors: {xs} vs {rs}"
    return xs


def test_select_idxs_list_and_slice_and_bool():
    p = _proto()
    assert _rows(p.select_idxs([4, 0, 2])) == [4, 0, 2]
    assert _rows(p.select_idxs(slice(1, 4))) == [1, 2, 3]
    mask = torch.tensor([True, False, True, False, False, True])
    assert _rows(p.select_idxs(mask)) == [0, 2, 5]
    # lengths must travel with the rows, not just the leading value
    assert [len(r) for r in p.select_idxs([4, 0]).ragged["routing"]] == [5, 1]


def test_split_and_minibatches():
    p = _proto()
    assert [_rows(c) for c in p.split(3)] == [[0, 1], [2, 3], [4, 5]]
    assert [_rows(c) for c in p.mini_batches(4)] == [[0, 1, 2, 3], [4, 5]]


def test_concat_and_merge_roundtrip():
    p = _proto()
    chunks = p.split(3)
    assert _rows(DataProto.concat(chunks)) == [0, 1, 2, 3, 4, 5]
    assert _rows(DataProto.merge(chunks)) == [0, 1, 2, 3, 4, 5]


def test_reorder_in_place():
    p = _proto()
    p.reorder([3, 1, 0, 2, 5, 4])
    assert _rows(p) == [3, 1, 0, 2, 5, 4]


def test_repeat_interleave_and_tile():
    p = _proto(3)
    assert _rows(p.repeat(2, interleave=True)) == [0, 0, 1, 1, 2, 2]
    assert _rows(p.repeat(2, interleave=False)) == [0, 1, 2, 0, 1, 2]
    assert _rows(p.sample_level_repeat([1, 0, 3])) == [0, 2, 2, 2]


def test_pad_to_divisor_and_unpad():
    p = _proto(5)
    padded, n = p.pad_to_divisor(4)
    assert n == 3 and _rows(padded) == [0, 1, 2, 3, 4, 0, 1, 2]
    assert _rows(padded.unpad(3)) == [0, 1, 2, 3, 4]


def test_select_keys_and_to_preserve_ragged():
    p = _proto()
    assert _rows(p.select(["x"])) == list(range(6))
    assert _rows(p.to("cpu")) == list(range(6))


def test_update_merges_ragged():
    p, q = _proto(3), _proto(3)
    q.ragged = {"other": [["z"]] * 3}
    p.update(q)
    assert set(p.ragged) == {"routing", "other"}


def test_check_consistency_catches_desync():
    p = _proto(4)
    p.ragged["routing"] = p.ragged["routing"][:2]
    try:
        p.check_consistency()
    except ValueError as exc:
        assert "ragged" in str(exc)
    else:
        raise AssertionError("check_consistency did not catch a short ragged column")


def test_absent_ragged_is_a_noop():
    p = DataProto(tensors={"x": torch.arange(4).unsqueeze(1)})
    assert p.select_idxs([2, 1]).ragged == {}
    assert DataProto.concat(p.split(2)).batch_size == 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all DataProto.ragged tests passed")
