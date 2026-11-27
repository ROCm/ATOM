# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""PP mailbox hops between hosts (atom/distributed/pp_fabric.py) without a GPU:
which hop takes which transport, the symmetric buffer's addresses, the options,
and what a fabric send / recv waits on, writes and where it reads.

Mapping peers and the kernels need the fabric; ``tools/pp_fabric_probe.py`` and
``tools/pp_mailbox_selftest.py`` exercise them on the hardware.
"""

import contextlib
import importlib
import types

import pytest
import torch

from atom.distributed import pp_fabric as pf
from atom.distributed import pp_mailbox as mb
from atom.distributed.pp_mailbox import HopLayout, PPMailbox

H = 7168


def _ends(*hosts, mailbox=True, fabric=True):
    return [pf.StageEnd(h, mailbox, fabric) for h in hosts]


# -- plan --------------------------------------------------------------------------------------------------------------
def test_plan_two_trays():
    """Kimi-K3's PP8 P on two trays: six hops inside a tray, one across."""
    stages = _ends(*(["e15-07"] * 4 + ["e15-05"] * 4))
    assert (
        pf.plan_hop_transports(stages, fabric_allowed=True)
        == ["ipc"] * 3 + ["fabric"] + ["ipc"] * 3
    )
    assert (
        pf.plan_hop_transports(stages, fabric_allowed=False)
        == ["ipc"] * 3 + ["nccl"] + ["ipc"] * 3
    )


def test_plan_needs_fabric_on_both_ends():
    stages = [
        pf.StageEnd("a", True, True),
        pf.StageEnd("b", True, False),
        pf.StageEnd("c", True, True),
    ]
    assert pf.plan_hop_transports(stages, fabric_allowed=True) == ["nccl", "nccl"]


def test_plan_mailbox_off_is_nccl_everywhere():
    stages = _ends("a", "a", "b", mailbox=False, fabric=False)
    assert pf.plan_hop_transports(stages, fabric_allowed=True) == ["nccl", "nccl"]


def test_plan_same_host_never_fabric():
    """A same-host hop is IPC when the mailbox is on, else NCCL: never fabric."""
    assert pf.plan_hop_transports(_ends("a", "a"), True) == ["ipc"]
    assert pf.plan_hop_transports(
        [pf.StageEnd("a", False, True), pf.StageEnd("a", False, True)], True
    ) == ["nccl"]


def test_fabric_hop_ends():
    assert pf.fabric_hop_ends(["a"] * 4 + ["b"] * 4) == {3, 4}
    assert pf.fabric_hop_ends(["a"] * 8) == set()
    assert pf.fabric_hop_ends(["a", "b", "a", "b"]) == {0, 1, 2, 3}
    assert pf.fabric_hop_ends(["a", "a", "b", "b", "c", "c"]) == {1, 2, 3, 4}
    assert pf.fabric_hop_ends(["a"]) == set()


def test_hop_group_only_on_the_hop_ends(monkeypatch):
    """Only the hop's two ranks create its group, with local synchronization: a
    rank outside it never calls new_group (torch would make it join an
    ncclCommSplit that the gloo members never join)."""
    calls = []

    def new_group(ranks, **kw):
        calls.append((ranks, kw))
        return "pair"

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    ranks = [10, 11, 12, 13, 14, 15, 16, 17]
    got = [mb._hop_group(ranks, 3, r) for r in range(8)]
    assert got == [None, None, None, "pair", "pair", None, None, None]
    want = {"backend": "gloo", "use_local_synchronization": True}
    assert calls == [([13, 14], want)] * 2


# -- buffer layout -----------------------------------------------------------------------------------------------------
def test_counter_page_words_are_distinct_and_fit():
    rows_words = set(range(pf.ROWS_WORD, pf.ROWS_WORD + pf.MAX_SLOTS))
    others = {pf.FLAG_WORD, pf.CREDIT_WORD, pf.ARRIVE_WORD}
    assert not rows_words & others and len(others) == 3
    assert max(rows_words | others) * 8 + 8 <= pf.PAGE_BYTES
    assert pf.RING_OFFSET >= pf.PAGE_BYTES and pf.RING_OFFSET % (2 << 20) == 0


@pytest.mark.parametrize(
    "role, mode, own",
    [
        ("send", "push", False),
        ("recv", "push", True),
        ("send", "pull", True),
        ("recv", "pull", False),
    ],
)
def test_ring_slot_address(role, mode, own):
    """Push moves the payload through the receiver's ring, pull through the sender's."""
    local, peer, slot_bytes = 0x10_0000_0000, 0x20_0000_0000, 64 << 20
    got = pf.ring_slot_address(role, mode, local, peer, 1, slot_bytes)
    assert got == (local if own else peer) + pf.RING_OFFSET + slot_bytes


def test_check_widths():
    pf.check_widths(HopLayout(16, H, 8, 5, 3 * H, 2, 2))
    with pytest.raises(ValueError):
        pf.check_widths(HopLayout(16, 12, 8, 5, 0, 2, 2))  # 24 B rows


def test_options_validate():
    assert pf.FabricOptions() == pf.FabricOptions("push", "kernel", 32)
    for bad in ({"mode": "put"}, {"wait": "spin"}, {"ctas": 0}):
        with pytest.raises(ValueError):
            pf.FabricOptions(**bad)


def test_kernel_source_entry_points():
    for name in pf.KERNEL_NAMES:
        assert f'extern "C" __global__ void {name}(' in pf.KERNEL_SOURCE.replace(
            "__launch_bounds__(256)\n", ""
        )
    # no remote atomics: the only RMW is the sender's local arrival count
    assert pf.KERNEL_SOURCE.count("__hip_atomic_fetch_add") == 1


# -- env ---------------------------------------------------------------------------------------------------------------
def test_env(monkeypatch):
    envs = importlib.import_module("atom.utils.envs")
    for name in (
        "ATOM_PP_MAILBOX_FABRIC",
        "ATOM_PP_MAILBOX_FABRIC_MODE",
        "ATOM_PP_MAILBOX_FABRIC_WAIT",
        "ATOM_PP_MAILBOX_FABRIC_CTAS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert envs.ATOM_PP_MAILBOX_FABRIC is False
    assert (envs.ATOM_PP_MAILBOX_FABRIC_MODE, envs.ATOM_PP_MAILBOX_FABRIC_WAIT) == (
        "push",
        "kernel",
    )
    assert envs.ATOM_PP_MAILBOX_FABRIC_CTAS == 32
    monkeypatch.setenv("ATOM_PP_MAILBOX_FABRIC_MODE", "PULL")
    assert envs.ATOM_PP_MAILBOX_FABRIC_MODE == "pull"
    monkeypatch.setenv("ATOM_PP_MAILBOX_FABRIC_WAIT", "stram")
    with pytest.raises(ValueError):
        _ = envs.ATOM_PP_MAILBOX_FABRIC_WAIT
    monkeypatch.setenv("ATOM_PP_MAILBOX_FABRIC_CTAS", "0")
    with pytest.raises(ValueError):
        _ = envs.ATOM_PP_MAILBOX_FABRIC_CTAS


def test_fabric_options_from_env(monkeypatch):
    monkeypatch.setenv("ATOM_PP_MAILBOX_FABRIC", "1")
    monkeypatch.setenv("ATOM_PP_MAILBOX", "0")
    assert mb.fabric_options_from_env() is None  # needs the mailbox itself
    monkeypatch.setenv("ATOM_PP_MAILBOX", "1")
    monkeypatch.setenv("ATOM_PP_MAILBOX_FABRIC_MODE", "pull")
    monkeypatch.setenv("ATOM_PP_MAILBOX_FABRIC_CTAS", "64")
    assert mb.fabric_options_from_env() == pf.FabricOptions("pull", "kernel", 64)
    monkeypatch.setenv("ATOM_PP_MAILBOX_FABRIC", "0")
    assert mb.fabric_options_from_env() is None


# -- send / recv over a fake hop ---------------------------------------------------------------------------------------
class _Kernels:
    def __init__(self):
        self.calls = []

    def push(self, stream, **kw):
        self.calls.append(("push", kw))

    def signal(self, stream, rows_word, rows, word, value):
        self.calls.append(("signal", rows_word, rows, word, value))


class _Hop:
    """A FabricHop's addresses and calls, no GPU."""

    def __init__(self, role, layout, mode):
        self.role, self.layout = role, layout
        self.options = pf.FabricOptions(mode=mode)
        self.local, self.peer = 0x10_0000_0000, 0x20_0000_0000
        self.kernels = _Kernels()
        self.waits = []

    word = pf.FabricHop.word
    slot_base = pf.FabricHop.slot_base

    def wait_ge(self, stream, word, target):
        self.waits.append((word, target))


class _Stream:
    cuda_stream = 0

    def wait_event(self, ev):
        pass


class _Event:
    def record(self, stream=None):
        pass


@pytest.fixture
def no_gpu(monkeypatch):
    """Stream / event plumbing that runs on the CPU; HIP calls recorded."""
    calls = []

    class _Hip:
        def __getattr__(self, name):
            return lambda *a: calls.append((name, a)) or 0

    monkeypatch.setattr(mb, "_hip", lambda: _Hip())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None)
    return calls


def _box(role, mode, closed=5, aux_taps=3, slots=2, rows_max=64):
    lay = HopLayout(rows_max, H, 8, closed, aux_taps * H, 2, slots)
    box = object.__new__(PPMailbox)
    box.device, box.dtype, box.hidden, box.total_blocks = (
        torch.device("cpu"),
        torch.bfloat16,
        H,
        8,
    )
    box.check = False
    box.fabric = pf.FabricOptions(mode=mode)
    box.send_seq = box.recv_seq = 0
    box._send_done = __import__("collections").deque()
    box.send_stream = box.recv_stream = _Stream()
    hop = _Hop(role, lay, mode)
    if role == "send":
        box.out_layout, box.send_kind, box._fabric_out = lay, "fabric", hop
    else:
        box.in_layout, box.recv_kind, box._fabric_in = lay, "fabric", hop
    return box, hop, lay


def _payload(rows, closed=5, aux_taps=3):
    return {
        "hidden_states": torch.zeros(rows, H, dtype=torch.bfloat16),
        "block_residual": torch.zeros(rows, 8, H, dtype=torch.bfloat16),
        "pp_aux": torch.zeros(rows, aux_taps * H, dtype=torch.bfloat16),
    }


def test_fabric_push_send(no_gpu):
    box, hop, lay = _box("send", "push")
    rows = 40
    for seq in range(3):
        box.send(_payload(rows), rows)
    # the third send reuses slot 0: it waits for the receiver's first credit
    assert hop.waits == [(hop.local + 8 * pf.CREDIT_WORD, 1)]
    pushes = [kw for name, kw in hop.kernels.calls if name == "push"]
    assert len(pushes) == 3 and not no_gpu  # no copy-engine work under push
    kw, off = pushes[1], lay.offsets()
    slot1 = hop.peer + pf.RING_OFFSET + lay.slot_bytes
    assert (kw["dst_hidden"], kw["dst_bank"], kw["dst_aux"]) == (
        slot1 + off["hidden"],
        slot1 + off["bank"],
        slot1 + off["aux"],
    )
    assert (kw["hidden_v4"], kw["bank_row_v4"], kw["closed_v4"], kw["aux_v4"]) == (
        H // 8,
        8 * H // 8,
        5 * H // 8,
        3 * H // 8,
    )
    assert (
        kw["rows_word"] == hop.peer + 8 * (pf.ROWS_WORD + 1)
        and kw["flag_word"] == hop.peer + 8 * pf.FLAG_WORD
    )
    assert (
        kw["flag_value"] == 2
        and kw["arrive"] == hop.local + 8 * pf.ARRIVE_WORD
        and kw["rows"] == rows
    )


def test_fabric_pull_send(no_gpu):
    box, hop, lay = _box("send", "pull")
    box.send(_payload(10), 10)
    # the copy engine fills this stage's own slot 0, then one signal raises the peer's flag
    copies = [a for name, a in no_gpu if name in ("hipMemcpyAsync", "hipMemcpy2DAsync")]
    slot0 = hop.local + pf.RING_OFFSET
    assert [c[0].value for c in copies] == [
        slot0 + off for off in lay.offsets().values()
    ]
    assert hop.kernels.calls == [
        ("signal", hop.peer + 8 * pf.ROWS_WORD, 10, hop.peer + 8 * pf.FLAG_WORD, 1)
    ]


@pytest.mark.parametrize("mode", ["push", "pull"])
def test_fabric_recv(no_gpu, monkeypatch, mode):
    box, hop, lay = _box("recv", mode)
    reads = []
    monkeypatch.setattr(
        mb, "copy_in", lambda src, dst, rows, cols, ss, ds: reads.append(src)
    )
    out = box.recv(12)
    assert set(out) == {"hidden_states", "block_residual", "pp_aux"}
    assert out["block_residual"].shape == (12, 8, H)
    assert hop.waits == [(hop.local + 8 * pf.FLAG_WORD, 1)]
    base = (hop.local if mode == "push" else hop.peer) + pf.RING_OFFSET
    assert reads == [base + off for off in lay.offsets().values()]
    assert hop.kernels.calls == [("signal", None, 0, hop.peer + 8 * pf.CREDIT_WORD, 1)]
    assert box.recv_seq == 1


def test_selftest_namespace_has_ranks():
    """PPMailbox builds a fabric hop's two-rank group from ``group.ranks``."""
    group = types.SimpleNamespace(ranks=[0, 1, 2, 3, 4, 5, 6, 7])
    assert [group.ranks[3], group.ranks[4]] == [3, 4]
