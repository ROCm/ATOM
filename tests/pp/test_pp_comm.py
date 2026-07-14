# SPDX-License-Identifier: MIT
# CPP A·P1a — pp_comm wrapper logic (mock-based, GPU-free).
#
# Validates ATOM-side proxy pack/unpack and routing. The underlying
# send_tensor_dict/recv_tensor_dict transport is aiter's and tested there;
# here we pin the key selection, destination/source routing, and the
# IntermediateTensors round-trip contract.

import sys
import types
from unittest.mock import MagicMock

import torch

# Stub aiter.ops.communication.set_custom_all_reduce so importing pp_comm does
# not require a GPU build. aiter.dist.parallel_state is real (import-only here).
_comm_stub = types.ModuleType("aiter.ops.communication")
_comm_stub.set_custom_all_reduce = lambda *a, **k: None
sys.modules.setdefault("aiter.ops.communication", _comm_stub)

from atom.distributed import pp_comm  # noqa: E402
from atom.models.utils import IntermediateTensors  # noqa: E402


def _fake_pp_group(monkeypatch, rank_in_group=0, world_size=2):
    grp = MagicMock()
    grp.rank_in_group = rank_in_group
    grp.world_size = world_size
    sent = {}

    def send_tensor_dict(tensors, dst, all_gather_group=None):
        sent["tensors"] = tensors
        sent["dst"] = dst
        sent["all_gather_group"] = all_gather_group

    grp.send_tensor_dict.side_effect = send_tensor_dict
    monkeypatch.setattr(pp_comm, "get_pp_group", lambda: grp)
    # Default: single-rank TP so pp_send_allgather_group() returns None and the
    # send/recv paths behave like plain full-tensor transfers. Tests that want
    # send-allgather stub get_tp_group with world_size > 1 explicitly.
    tp = MagicMock()
    tp.world_size = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    return grp, sent


def test_send_packs_only_proxy_keys_to_group_relative_next(monkeypatch):
    # stage 0 of a 2-stage group → group-relative dst = 1
    grp, sent = _fake_pp_group(monkeypatch, rank_in_group=0, world_size=2)
    it = IntermediateTensors(
        {
            "hidden_states": torch.ones(3, 4),
            "residual": torch.zeros(3, 4),
            "extra": torch.arange(5),  # must be dropped
        }
    )
    pp_comm.send_intermediate_tensors(it)

    assert set(sent["tensors"].keys()) == {"hidden_states", "residual"}
    assert sent["dst"] == 1
    assert torch.equal(sent["tensors"]["hidden_states"], torch.ones(3, 4))


def test_send_tolerates_missing_residual(monkeypatch):
    grp, sent = _fake_pp_group(monkeypatch)
    it = IntermediateTensors({"hidden_states": torch.ones(2, 2)})
    pp_comm.send_intermediate_tensors(it)
    assert set(sent["tensors"].keys()) == {"hidden_states"}


def test_recv_wraps_from_group_relative_prev(monkeypatch):
    # stage 1 of a 2-stage group → group-relative src = 0
    grp, _ = _fake_pp_group(monkeypatch, rank_in_group=1, world_size=2)
    payload = {"hidden_states": torch.ones(2, 2), "residual": torch.zeros(2, 2)}
    grp.recv_tensor_dict.return_value = payload

    out = pp_comm.recv_intermediate_tensors()

    grp.recv_tensor_dict.assert_called_once_with(src=0, all_gather_group=None)
    assert isinstance(out, IntermediateTensors)
    assert torch.equal(out["hidden_states"], torch.ones(2, 2))
    assert torch.equal(out["residual"], torch.zeros(2, 2))


def test_roundtrip_preserves_proxy_tensors(monkeypatch):
    """send then recv (looping the payload back) preserves proxy tensors."""
    grp, sent = _fake_pp_group(monkeypatch)
    it = IntermediateTensors(
        {"hidden_states": torch.randn(4, 8), "residual": torch.randn(4, 8)}
    )
    pp_comm.send_intermediate_tensors(it)
    grp.recv_tensor_dict.return_value = sent["tensors"]

    got = pp_comm.recv_intermediate_tensors()
    for k in ("hidden_states", "residual"):
        assert torch.equal(got[k], it[k])


def test_allgather_group_none_when_disabled(monkeypatch):
    tp = MagicMock()
    tp.world_size = 4
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", False)
    assert pp_comm.pp_send_allgather_group() is None


def test_allgather_group_none_when_tp_size_one(monkeypatch):
    tp = MagicMock()
    tp.world_size = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    assert pp_comm.pp_send_allgather_group() is None


def test_allgather_group_selected_when_enabled(monkeypatch):
    tp = MagicMock()
    tp.world_size = 4
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    assert pp_comm.pp_send_allgather_group() is tp


def test_recv_threads_allgather_group(monkeypatch):
    grp, _ = _fake_pp_group(monkeypatch, rank_in_group=1, world_size=2)
    tp = MagicMock()
    tp.world_size = 4
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    grp.recv_tensor_dict.return_value = {
        "hidden_states": torch.ones(2, 2),
        "residual": torch.zeros(2, 2),
    }

    pp_comm.recv_intermediate_tensors()

    grp.recv_tensor_dict.assert_called_once_with(src=0, all_gather_group=tp)


def _fake_async_pp_group(monkeypatch, rank_in_group=0, world_size=2):
    grp = MagicMock()
    grp.rank_in_group = rank_in_group
    grp.world_size = world_size
    grp.ranks = list(range(world_size))
    grp.cpu_group = MagicMock()
    grp.device_group = MagicMock()
    monkeypatch.setattr(pp_comm, "get_pp_group", lambda: grp)
    return grp


def _capture_isend(monkeypatch):
    """Record the payload numel of every isend the send path posts."""
    sent = []

    def isend(buf, dst, group):
        sent.append(buf.numel())
        return MagicMock()

    monkeypatch.setattr(pp_comm.torch.distributed, "isend", isend)
    return sent


def test_async_send_slices_payload_when_allgather_on(monkeypatch):
    _fake_async_pp_group(monkeypatch, rank_in_group=0, world_size=2)
    tp = MagicMock()
    tp.world_size = 4
    tp.rank_in_group = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    sent = _capture_isend(monkeypatch)

    # [4, 8] -> 32 elems, 32 % 4 == 0 -> each rank sends a 8-elem slice.
    it = IntermediateTensors(
        {"hidden_states": torch.randn(4, 8), "residual": torch.randn(4, 8)}
    )
    pp_comm.async_send_intermediate_tensors(it)

    # First two isends are metadata (size + pickled object); the rest are the
    # two sliced proxy tensors, each 32/4 == 8 elements.
    tensor_payloads = sent[2:]
    assert tensor_payloads == [8, 8]


def test_async_send_full_tensor_when_allgather_off(monkeypatch):
    _fake_async_pp_group(monkeypatch, rank_in_group=0, world_size=2)
    tp = MagicMock()
    tp.world_size = 4
    tp.rank_in_group = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", False)
    sent = _capture_isend(monkeypatch)

    it = IntermediateTensors(
        {"hidden_states": torch.randn(4, 8), "residual": torch.randn(4, 8)}
    )
    pp_comm.async_send_intermediate_tensors(it)

    # No slicing: each proxy tensor is sent whole (32 elements).
    assert sent[2:] == [32, 32]


def test_async_send_full_tensor_when_not_divisible(monkeypatch):
    _fake_async_pp_group(monkeypatch, rank_in_group=0, world_size=2)
    tp = MagicMock()
    tp.world_size = 4
    tp.rank_in_group = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    sent = _capture_isend(monkeypatch)

    # 5*7 == 35, 35 % 4 != 0 -> guard fails on both sides -> full tensor sent.
    it = IntermediateTensors({"hidden_states": torch.randn(5, 7)})
    pp_comm.async_send_intermediate_tensors(it)

    assert sent[2:] == [35]
