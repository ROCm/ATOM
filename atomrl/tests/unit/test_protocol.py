# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import pytest
import torch
from atomrl.core.protocol import DataProto


def test_create_empty() -> None:
    p = DataProto()
    assert p.batch_size == 0
    assert p.keys() == []
    assert p.meta == {}


def test_create_with_tensors() -> None:
    p = DataProto(tensors={"x": torch.arange(6).view(2, 3)}, meta={"k": 1})
    assert p.batch_size == 2
    assert "x" in p
    assert torch.equal(p["x"], torch.arange(6).view(2, 3))
    assert p.meta["k"] == 1


def test_batch_size() -> None:
    p = DataProto(tensors={"a": torch.zeros(5, 2), "b": torch.ones(5)})
    assert len(p) == 5
    assert p.batch_size == 5


def test_split_and_merge() -> None:
    p = DataProto(
        tensors={"x": torch.arange(12).view(4, 3).float()},
        meta={"m": True},
    )
    chunks = p.split(3)
    assert len(chunks) == 3
    sizes = [c.batch_size for c in chunks]
    assert sum(sizes) == 4
    merged = DataProto.merge(chunks)
    assert merged.batch_size == 4
    assert torch.allclose(merged["x"], p["x"])
    assert merged.meta == p.meta


def test_to_device_cpu_only() -> None:
    p = DataProto(tensors={"x": torch.tensor([1.0], device="cpu")})
    q = p.to("cpu")
    assert q["x"].device.type == "cpu"
    assert torch.equal(q["x"], p["x"])


def test_select_keys() -> None:
    p = DataProto(
        tensors={"a": torch.zeros(2), "b": torch.ones(2), "c": torch.full((2,), 2.0)}
    )
    s = p.select(["b", "a"])
    assert set(s.keys()) == {"a", "b"}
    assert "c" not in s


def test_update() -> None:
    p = DataProto(tensors={"a": torch.zeros(2)}, meta={"x": 1})
    q = DataProto(tensors={"b": torch.ones(2)}, meta={"y": 2})
    p.update(q)
    assert set(p.keys()) == {"a", "b"}
    assert p.meta["x"] == 1 and p.meta["y"] == 2


def test_mini_batches() -> None:
    p = DataProto(tensors={"t": torch.arange(10).view(10, 1)})
    batches = list(p.mini_batches(3))
    assert len(batches) == 4
    assert [b.batch_size for b in batches] == [3, 3, 3, 1]
    cat = torch.cat([b["t"] for b in batches], dim=0)
    assert torch.equal(cat.squeeze(-1), torch.arange(10))


def test_router_distributions() -> None:
    p = DataProto()
    p.add_router_distributions(0, torch.randn(2, 4))
    p.add_router_distributions(2, torch.randn(2, 4))
    dists = p.get_router_distributions()
    assert set(dists.keys()) == {0, 2}
    assert dists[0].shape == (2, 4)
    assert p.has_router_distributions()


def test_merge_preserves_data() -> None:
    a = DataProto(tensors={"x": torch.tensor([[1.0], [2.0]])}, meta={"tag": "a"})
    b = DataProto(tensors={"x": torch.tensor([[3.0]])}, meta={"tag": "a"})
    m = DataProto.merge([a, b])
    assert torch.allclose(m["x"], torch.tensor([[1.0], [2.0], [3.0]]))
    assert m.meta["tag"] == "a"


def test_merge_key_mismatch_raises() -> None:
    a = DataProto(tensors={"x": torch.tensor([[1.0]]), "y": torch.tensor([[2.0]])})
    b = DataProto(tensors={"x": torch.tensor([[3.0]])})
    with pytest.raises(ValueError, match="keys mismatch"):
        DataProto.merge([a, b])


def test_select_idxs_with_int_slice_and_list() -> None:
    p = DataProto(tensors={"x": torch.arange(12).view(6, 2)})
    one = p.select_idxs(2)
    sl = p.select_idxs(slice(1, 5, 2))
    lst = p.select_idxs([0, 5, 3])
    assert one.batch_size == 1
    assert torch.equal(one["x"], torch.tensor([[4, 5]]))
    assert torch.equal(sl["x"], torch.tensor([[2, 3], [6, 7]]))
    assert torch.equal(lst["x"], torch.tensor([[0, 1], [10, 11], [6, 7]]))


def test_getitem_supports_row_selection() -> None:
    p = DataProto(tensors={"x": torch.arange(10).view(5, 2)})
    rows = p[1:4]
    mask = p[torch.tensor([True, False, True, False, False])]
    assert isinstance(rows, DataProto)
    assert rows.batch_size == 3
    assert torch.equal(mask["x"], torch.tensor([[0, 1], [4, 5]]))


def test_concat_vs_merge_padding_behavior() -> None:
    a = DataProto(tensors={"x": torch.tensor([[1, 2, 3]])})
    b = DataProto(tensors={"x": torch.tensor([[4, 5, 6]])})
    c = DataProto.concat([a, b])
    assert c.batch_size == 2
    assert torch.equal(c["x"], torch.tensor([[1, 2, 3], [4, 5, 6]]))

    p = DataProto(tensors={"x": torch.tensor([[1, 2, 3]])})
    q = DataProto(tensors={"x": torch.tensor([[4, 5]])})
    m = DataProto.merge([p, q])
    assert m["x"].shape == (2, 3)
    assert torch.equal(m["x"][1], torch.tensor([4, 5, 0]))


def test_reorder_in_place() -> None:
    p = DataProto(tensors={"x": torch.tensor([[1], [2], [3]])})
    p.reorder([2, 0, 1])
    assert torch.equal(p["x"], torch.tensor([[3], [1], [2]]))


def test_pad_to_divisor_and_unpad() -> None:
    p = DataProto(tensors={"x": torch.arange(10).view(5, 2)})
    padded, pad_size = p.pad_to_divisor(4)
    restored = padded.unpad(pad_size)
    assert pad_size == 3
    assert padded.batch_size == 8
    assert restored.batch_size == p.batch_size
    assert torch.equal(restored["x"], p["x"])


def test_repeat_and_sample_level_repeat() -> None:
    p = DataProto(tensors={"x": torch.tensor([[1], [2], [3]])})
    interleave = p.repeat(2, interleave=True)
    stacked = p.repeat(2, interleave=False)
    per_sample = p.sample_level_repeat([1, 3, 0])
    assert torch.equal(interleave["x"], torch.tensor([[1], [1], [2], [2], [3], [3]]))
    assert torch.equal(stacked["x"], torch.tensor([[1], [2], [3], [1], [2], [3]]))
    assert torch.equal(per_sample["x"], torch.tensor([[1], [2], [2], [2]]))


def test_check_consistency_raises_on_mismatched_dim0() -> None:
    p = DataProto(tensors={"a": torch.zeros(2, 1), "b": torch.zeros(3, 1)})
    with pytest.raises(ValueError, match="inconsistent batch dimensions"):
        p.check_consistency()
