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

    def send_tensor_dict(tensors, dst):
        sent["tensors"] = tensors
        sent["dst"] = dst

    grp.send_tensor_dict.side_effect = send_tensor_dict
    monkeypatch.setattr(pp_comm, "get_pp_group", lambda: grp)
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

    grp.recv_tensor_dict.assert_called_once_with(src=0)
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
