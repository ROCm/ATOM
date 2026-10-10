# SPDX-License-Identifier: MIT
"""Run the production slot copy on CPU tensors, with no AITER/GPU imports."""

from dataclasses import replace

import pytest
import torch

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)
from atom.model_ops.attentions.pool_layout.slot_checkpoint import (
    copy_kda_checkpoint_slots,
)


def tensors(num_spec=0):
    conv = torch.empty((3, 5, 3 + num_spec, 7), dtype=torch.bfloat16)
    state = torch.empty((3, 5, 2, 4, 6), dtype=torch.float16)
    for plane in (conv, state):
        raw = plane.view(torch.uint8)
        raw.copy_(torch.arange(raw.numel()).reshape(raw.shape).to(torch.uint8))
    nbytes = sum(x[:, 0].numel() * x.element_size() for x in (conv, state))
    spec = PagedStateCheckpointSpec(64, nbytes, "kda-cpu", image_bytes=nbytes)
    return conv, state, spec


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("num_spec", [0, 3])
def test_snapshot_survives_producer_overwrite_and_restores_two_readers(rank, num_spec):
    conv, state, spec = tensors(num_spec)
    for plane in (conv, state):
        plane[:, 0].view(torch.uint8).fill_(rank + 17)
    expected = [plane[:, 0].clone().view(torch.uint8) for plane in (conv, state)]
    before = [plane.clone() for plane in (conv, state)]
    store = CheckpointStoreOp(0, (), spec.image_bytes, spec.layout_id, dst_slot=2)
    copy_kda_checkpoint_slots(conv, state, spec, (store,), ())
    for plane, old in zip((conv, state), before):
        for untouched in (0, 1, 3, 4):
            assert torch.equal(
                plane[:, untouched].view(torch.uint8),
                old[:, untouched].view(torch.uint8),
            )
        plane[:, 0].zero_()  # next forward overwrites the producer's live state
    restores = tuple(
        CheckpointRestoreOp(dst, (), spec.image_bytes, spec.layout_id, src_slot=2)
        for dst in (1, 3)
    )
    copy_kda_checkpoint_slots(conv, state, spec, (), restores)
    for plane, snapshot in zip((conv, state), expected):
        for slot in (1, 2, 3):
            assert torch.equal(plane[:, slot].view(torch.uint8), snapshot)
        assert torch.count_nonzero(plane[:, 0]) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"dst_slot": 5},
        {"src_slot": -1},
        {"dst_slot": 0},
        {"unit_ids": (0,)},
        {"total_bytes": 1},
        {"layout_id": "wrong"},
    ],
)
def test_invalid_operation_fails_before_any_destination_is_written(change):
    conv, state, spec = tensors()
    before = [x.clone() for x in (conv, state)]
    op = CheckpointStoreOp(0, (), spec.image_bytes, spec.layout_id, dst_slot=2)
    with pytest.raises(RuntimeError):
        copy_kda_checkpoint_slots(conv, state, spec, (replace(op, **change),), ())
    for plane, old in zip((conv, state), before):
        assert torch.equal(plane.view(torch.uint8), old.view(torch.uint8))


def test_duplicate_destinations_and_cross_operation_aliases_are_rejected():
    conv, state, spec = tensors()
    op = CheckpointStoreOp(0, (), spec.image_bytes, spec.layout_id, dst_slot=2)
    for other in (replace(op, src_slot=1), replace(op, src_slot=2, dst_slot=3)):
        with pytest.raises(RuntimeError, match="alias"):
            copy_kda_checkpoint_slots(conv, state, spec, (op, other), ())


def test_wrong_tensor_geometry_is_rejected():
    conv, state, spec = tensors()
    op = CheckpointStoreOp(0, (), spec.image_bytes, spec.layout_id, dst_slot=2)
    with pytest.raises(RuntimeError, match="image size"):
        copy_kda_checkpoint_slots(conv[:2], state, spec, (op,), ())
    with pytest.raises(RuntimeError, match="slot counts"):
        copy_kda_checkpoint_slots(conv, state[:, :4], spec, (op,), ())
