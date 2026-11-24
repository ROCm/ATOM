# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest
from atomrl.engine.inference import vllm_ray_server
from atomrl.engine.inference.vllm_ray_server import (
    VLLMRayServer,
    VLLMReplicaManager,
)

CAPABILITY = {
    "protocol_version": 2,
    "module_path": "atomrl.engine.inference.vllm_colocate_worker_ext",
    "online_quant_reload": True,
    "prequantized_stream": True,
}


def test_vllm_runtime_env_propagates_msccl_disable(monkeypatch):
    monkeypatch.setenv("RCCL_MSCCL_ENABLE", "0")
    env_vars = {}

    assert hasattr(vllm_ray_server, "_copy_vllm_runtime_env")
    vllm_ray_server._copy_vllm_runtime_env(env_vars)

    assert env_vars["RCCL_MSCCL_ENABLE"] == "0"


def test_vllm_runtime_env_propagates_aiter_moe_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")
    env_vars = {}

    vllm_ray_server._copy_vllm_runtime_env(env_vars)

    assert env_vars["VLLM_ROCM_USE_AITER_MOE"] == "0"


def test_vllm_runtime_env_propagates_full_aiter_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    env_vars = {}

    vllm_ray_server._copy_vllm_runtime_env(env_vars)

    assert env_vars["VLLM_ROCM_USE_AITER"] == "0"


def test_vllm_runtime_env_propagates_aiter_triton_gemm_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_GEMM", "0")
    env_vars = {}

    vllm_ray_server._copy_vllm_runtime_env(env_vars)

    assert env_vars["VLLM_ROCM_USE_AITER_TRITON_GEMM"] == "0"


def test_vllm_runtime_env_propagates_all_gather_lifecycle_diagnostic(monkeypatch):
    monkeypatch.setenv("ATOMRL_DIAG_ALL_GATHER", "1")
    monkeypatch.setenv("ATOMRL_DIAG_ALL_GATHER_NUMEL", "517120")
    env_vars = {}

    vllm_ray_server._copy_vllm_runtime_env(env_vars)

    assert env_vars["ATOMRL_DIAG_ALL_GATHER"] == "1"
    assert env_vars["ATOMRL_DIAG_ALL_GATHER_NUMEL"] == "517120"


def test_vllm_runtime_env_propagates_weight_integrity_diagnostic(monkeypatch):
    monkeypatch.setenv("ATOMRL_WEIGHT_SYNC_INTEGRITY", "1")
    env_vars = {}

    vllm_ray_server._copy_vllm_runtime_env(env_vars)

    assert env_vars["ATOMRL_WEIGHT_SYNC_INTEGRITY"] == "1"


class RemoteMethod:
    def __init__(self, result=True):
        self.result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def install_fake_ray(monkeypatch):
    fake_ray = ModuleType("ray")

    def get(refs):
        if isinstance(refs, BaseException):
            raise refs
        return refs

    fake_ray.get = get
    monkeypatch.setitem(sys.modules, "ray", fake_ray)


def make_manager(capabilities):
    manager = VLLMReplicaManager(
        SimpleNamespace(num_workers=len(capabilities)),
        "model",
        {"tensor_parallel_size": len(capabilities)},
    )
    server = SimpleNamespace(
        get_rdma_capabilities=RemoteMethod(capabilities),
        receive_weights_rdma=RemoteMethod("receive-ref"),
        destroy_rdma_weight_group=RemoteMethod(True),
    )
    manager.servers = [server]
    manager.rdma_group_name = "group-1"
    return manager, server


def test_server_collects_rdma_capabilities_from_all_tp_workers():
    calls = []

    class Engine:
        async def collective_rpc(self, method, args=(), kwargs=None):
            calls.append((method, args, kwargs))
            return [CAPABILITY, CAPABILITY]

    server = object.__new__(VLLMRayServer)
    server.engine = Engine()

    capabilities = asyncio.run(server.get_rdma_capabilities())

    assert capabilities == [CAPABILITY, CAPABILITY]
    assert calls == [("get_rdma_capabilities", (), {})]


def test_manager_validates_once_per_rdma_group_before_receive_refs(monkeypatch):
    install_fake_ray(monkeypatch)
    manager, server = make_manager([CAPABILITY, CAPABILITY])

    first = manager.start_receive_weights_rdma(
        version=1,
        verify_full_load=True,
        prequantized_fp8=True,
    )
    second = manager.start_receive_weights_rdma(
        version=2,
        verify_full_load=False,
        prequantized_fp8=False,
        online_staging="cpu",
    )

    assert first == ["receive-ref"]
    assert second == ["receive-ref"]
    assert len(server.get_rdma_capabilities.calls) == 1
    assert server.receive_weights_rdma.calls == [
        (
            (),
            {
                "group_name": "group-1",
                "version": 1,
                "verify_full_load": True,
                "prequantized_fp8": True,
                "online_staging": "gpu",
            },
        ),
        (
            (),
            {
                "group_name": "group-1",
                "version": 2,
                "verify_full_load": False,
                "prequantized_fp8": False,
                "online_staging": "cpu",
            },
        ),
    ]


def test_server_forwards_online_staging_to_workers():
    calls = []

    class Engine:
        async def collective_rpc(self, method, args=(), kwargs=None):
            calls.append((method, kwargs))
            return [{"weights": 1.0}]

        async def reset_prefix_cache(self):
            return None

    server = object.__new__(VLLMRayServer)
    server.engine = Engine()

    asyncio.run(server.receive_weights_rdma("group-1", 3, online_staging="cpu"))

    assert calls == [
        (
            "receive_weights_rdma",
            {
                "group_name": "group-1",
                "version": 3,
                "verify_full_load": True,
                "prequantized_fp8": False,
                "online_staging": "cpu",
            },
        )
    ]


@pytest.mark.parametrize(
    ("capability", "error"),
    [
        ({**CAPABILITY, "protocol_version": 1}, "protocol_version"),
        ({**CAPABILITY, "online_quant_reload": False}, "online_quant_reload"),
        ({**CAPABILITY, "prequantized_stream": False}, "prequantized_stream"),
    ],
)
def test_manager_rejects_incompatible_worker_before_receive_refs(
    monkeypatch,
    capability,
    error,
):
    install_fake_ray(monkeypatch)
    manager, server = make_manager([capability])

    with pytest.raises(
        RuntimeError,
        match=rf"server=0 worker=0.*{error}",
    ):
        manager.start_receive_weights_rdma(
            version=1,
            verify_full_load=True,
        )

    assert server.receive_weights_rdma.calls == []


def test_manager_reports_missing_capability_rpc_before_receive_refs(monkeypatch):
    install_fake_ray(monkeypatch)
    manager, server = make_manager([CAPABILITY])
    del server.get_rdma_capabilities

    with pytest.raises(
        RuntimeError,
        match=r"server=0 workers=0\.\.0.*get_rdma_capabilities",
    ):
        manager.start_receive_weights_rdma(
            version=1,
            verify_full_load=True,
        )

    assert server.receive_weights_rdma.calls == []


def test_destroyed_rdma_group_requires_a_new_handshake(monkeypatch):
    install_fake_ray(monkeypatch)
    manager, server = make_manager([CAPABILITY])

    manager.start_receive_weights_rdma(version=1, verify_full_load=True)
    manager.destroy_rdma_weight_group()
    manager.rdma_group_name = "group-2"
    manager.start_receive_weights_rdma(version=2, verify_full_load=True)

    assert len(server.get_rdma_capabilities.calls) == 2
