# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Which servers take the IPC PP boundary (`atom.distributed.pp_ipc`), and the
head's in-flight default that follows it."""

import sys
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.distributed import pp_ipc
    from atom.model_engine.pp_engine_core import pp_max_inflight

_M3 = SimpleNamespace(
    architectures=["MiniMaxM3SparseForConditionalGeneration"],
    text_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"]),
)
_OTHER = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"], model_type="x")
_PRODUCER = {"kv_connector": "mooncake", "kv_role": "kv_producer"}


def _config(hf=_M3, kv=_PRODUCER, pp=4, tp=1, pcp=1, dcp=1, sp=1, dp=1):
    return SimpleNamespace(
        hf_config=hf,
        kv_transfer_config=kv,
        pipeline_parallel_size=pp,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        sequence_parallel_size=sp,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("ATOM_PP_IPC", "ATOM_PP_MAX_INFLIGHT"):
        monkeypatch.delenv(name, raising=False)
    _arch(monkeypatch, "gfx950")


def _arch(monkeypatch, arch):
    chip_info = SimpleNamespace(get_gfx_runtime=lambda: arch)
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", chip_info)


@pytest.mark.parametrize(
    "config, on",
    [
        (_config(), True),
        (_config(pp=2), True),
        (_config(kv={"kv_connector": "moriio", "kv_role": "kv_producer"}), True),
        (_config(kv={}), False),
        (_config(kv={"kv_connector": "mooncake", "kv_role": "kv_consumer"}), False),
        (_config(hf=_OTHER), False),
        (_config(pp=1), False),
        (_config(tp=2), False),
        (_config(pcp=2), False),
        (_config(dcp=2), False),
        (_config(sp=2), False),
        (_config(dp=2), False),
    ],
)
def test_only_an_m3_producer_with_one_rank_a_stage_takes_ipc(config, on):
    assert pp_ipc.pp_ipc_enabled(config) is on


def test_a_decoding_m3_server_keeps_nccl_and_the_stage_count_in_flight():
    config = _config(kv={})
    assert pp_max_inflight(4, pp_ipc.pp_ipc_enabled(config)) == 4


def test_other_arches_and_the_off_switch_keep_nccl(monkeypatch):
    _arch(monkeypatch, "gfx942")
    assert not pp_ipc.pp_ipc_enabled(_config())
    _arch(monkeypatch, "gfx950")
    monkeypatch.setenv("ATOM_PP_IPC", "0")
    assert not pp_ipc.pp_ipc_enabled(_config())


@pytest.mark.parametrize(
    "env, ipc, cap", [(None, False, 4), (None, True, 8), ("6", True, 6)]
)
def test_in_flight_doubles_over_ipc_unless_set(monkeypatch, env, ipc, cap):
    if env is not None:
        monkeypatch.setenv("ATOM_PP_MAX_INFLIGHT", env)
    assert pp_max_inflight(4, ipc) == cap
