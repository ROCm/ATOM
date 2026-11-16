# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which LM heads a decode graph may record at TP > 1: only one whose logits
all-gather takes the custom kernel (the RCCL fallback cannot be captured).
Nothing here touches a GPU."""

import types

import pytest

embed_head = pytest.importorskip(
    "atom.model_ops.embed_head",
    reason="embed_head imports aiter at module load",
    exc_type=ImportError,
)


def head(tp_size=4, vocab_per_rank=129280 // 4, element_size=2):
    return types.SimpleNamespace(
        tp_size=tp_size,
        num_embeddings_per_partition=vocab_per_rank,
        weight=types.SimpleNamespace(element_size=lambda: element_size),
    )


@pytest.fixture
def tp(monkeypatch):
    ca = types.SimpleNamespace(disabled=False, _has_gather_scatter_kernel=lambda: True)
    group = types.SimpleNamespace(device_communicator=types.SimpleNamespace(ca_comm=ca))
    monkeypatch.setattr(embed_head, "get_tp_group", lambda: group)
    monkeypatch.setattr(embed_head, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(embed_head, "sp_is_enabled", lambda: False)
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    return group


def recordable(h):
    return embed_head.ParallelLMHead.gather_recordable(h)


def test_custom_gather_at_tp4_is_recordable(tp):
    assert recordable(head())


def test_tp1_has_no_gather(tp):
    tp.device_communicator.ca_comm = None
    assert recordable(head(tp_size=1))


@pytest.mark.parametrize(
    "break_it",
    [
        lambda tp, mp: setattr(tp.device_communicator, "ca_comm", None),
        lambda tp, mp: setattr(tp.device_communicator.ca_comm, "disabled", True),
        lambda tp, mp: setattr(
            tp.device_communicator.ca_comm, "_has_gather_scatter_kernel", lambda: False
        ),
        lambda tp, mp: mp.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "0"),
        lambda tp, mp: mp.setattr(embed_head, "sp_is_enabled", lambda: True),
        lambda tp, mp: mp.setattr(embed_head, "is_plugin_mode", lambda: True),
    ],
)
def test_rccl_fallback_keeps_the_head_eager(tp, monkeypatch, break_it):
    break_it(tp, monkeypatch)
    assert not recordable(head())


def test_rows_the_custom_kernel_cannot_take_stay_eager(tp):
    # 16-byte rows only (`GroupCoordinator.all_gather`): 4 bf16 columns are 8.
    assert not recordable(head(vocab_per_rank=32324))
    assert recordable(head(vocab_per_rank=32328))
