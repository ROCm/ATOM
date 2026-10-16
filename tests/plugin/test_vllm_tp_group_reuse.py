# SPDX-License-Identifier: MIT
"""CPU lifecycle contract for plugin-owned AITER collectives on borrowed vLLM PGs."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.plugin.vllm.tp_group_reuse import init_aiter_dist_from_vllm


@pytest.fixture
def tp_env(monkeypatch):
    events = []
    resources = []
    options = SimpleNamespace(ca_disabled=False, setup_error=None, owner_error=None)
    gpu_guard = Mock(side_effect=AssertionError("CPU lifecycle test touched a GPU"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", gpu_guard)
    monkeypatch.setattr(torch.cuda, "synchronize", gpu_guard)

    class Resource:
        def __init__(self, name, **kwargs):
            self.name = name
            self.disabled = name == "ca" and options.ca_disabled
            self.close_error = None
            self.close_calls = 0
            resources.append(self)

        def close(self):
            self.close_calls += 1
            events.append(self.name + ".close")
            if self.close_error is not None:
                raise self.close_error

        def register_input_buffer(self, signal):
            assert signal.device.type == "cpu"
            events.append("signal.register")
            if options.setup_error is not None:
                raise options.setup_error

    class DeviceCommunicatorBase:
        def __init__(self, cpu_group, device, device_group, unique_name):
            self.cpu_group = cpu_group
            self.device_group = device_group
            self.device = device
            self.world_size = 2
            self.unique_name = unique_name

    class CudaCommunicator(DeviceCommunicatorBase):
        def destroy(self):
            # Match /app/aiter-test, not a host AITER tree: PyNccl is only
            # dereferenced; _owns_comms gates qr/ca close and fields are cleared
            # only after the corresponding close returns successfully.
            if self.pynccl_comm is not None:
                self.pynccl_comm = None
            if self.qr_comm is not None:
                if self._owns_comms:
                    self.qr_comm.close()
                self.qr_comm = None
            if self.ca_comm is not None:
                if self._owns_comms:
                    self.ca_comm.close()
                self.ca_comm = None
            if self._all2all_manager is not None:
                self._all2all_manager.destroy()
                self._all2all_manager = None

    class GroupCoordinator:
        def destroy(self):
            # Real AITER parent expects these ownership fields even though the
            # adapter skips its constructor. Borrowed groups must survive.
            if hasattr(self, "device_group"):
                if self._owns_device_group:
                    self.device_group.destroy()
                del self.device_group
            if hasattr(self, "cpu_group"):
                if self._owns_cpu_group:
                    self.cpu_group.destroy()
                del self.cpu_group
            if self.device_communicator is not None:
                if self._owns_device_communicator:
                    self.device_communicator.destroy()
                self.device_communicator = None
            if self.mq_broadcaster is not None:
                self.mq_broadcaster.close()
                self.mq_broadcaster = None

    pynccl = Resource("pynccl")

    def process_group(name):
        return SimpleNamespace(destroy=Mock(side_effect=lambda: events.append(name)))

    cpu_group = process_group("cpu_pg.destroy")
    device_group = process_group("device_pg.destroy")

    class Owner:
        rank = local_rank = rank_in_group = 0
        world_size = 2
        device = torch.device("cpu")

        def __init__(self):
            self.ranks = [0, 1]
            self.cpu_group = cpu_group
            self.device_group = device_group
            self.device_communicator = SimpleNamespace(pynccl_comm=pynccl)
            self.destroy_calls = 0

        def destroy(self):
            events.append("owner.destroy")
            self.destroy_calls += 1
            if options.owner_error is not None:
                raise options.owner_error
            # vLLM tears down its communicator before its process groups.
            if self.device_communicator is not None:
                self.device_communicator.pynccl_comm.close()
                self.device_communicator = None
            for name in ("device_group", "cpu_group"):
                if hasattr(self, name):
                    getattr(self, name).destroy()
                    delattr(self, name)
            return "owner-result"

    owner = Owner()
    modules = {}
    for name in (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "aiter",
        "aiter.dist",
        "aiter.dist.parallel_state",
        "aiter.dist.device_communicators",
        "aiter.dist.device_communicators.base_device_communicator",
        "aiter.dist.device_communicators.communicator_cuda",
        "aiter.dist.device_communicators.custom_all_reduce",
        "aiter.dist.device_communicators.quick_all_reduce",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(modules[parent], attr, module)

    vllm_ps = modules["vllm.distributed.parallel_state"]
    aiter_ps = modules["aiter.dist.parallel_state"]
    vllm_ps.get_tp_group = lambda: owner
    pp, dp, ep = object(), object(), object()
    vllm_ps.get_pp_group = lambda: pp
    vllm_ps.get_dp_group = lambda: dp
    vllm_ps._EP = ep
    originals = {name: object() for name in ("_TP", "_PP", "_DP", "_EP")}
    for name, value in originals.items():
        setattr(aiter_ps, name, value)
    aiter_ps._ENABLE_CUSTOM_ALL_REDUCE = True
    aiter_ps.GroupCoordinator = GroupCoordinator
    aiter_ps._register_group = Mock()
    aiter_ps.set_custom_all_reduce = Mock()
    prefix = "aiter.dist.device_communicators."
    modules[prefix + "base_device_communicator"].DeviceCommunicatorBase = (
        DeviceCommunicatorBase
    )
    modules[prefix + "communicator_cuda"].CudaCommunicator = CudaCommunicator
    modules[prefix + "custom_all_reduce"].CustomAllreduce = lambda **kw: Resource(
        "ca", **kw
    )
    modules[prefix + "quick_all_reduce"].QuickAllReduce = lambda **kw: Resource(
        "qr", **kw
    )

    yield SimpleNamespace(
        owner=owner,
        aiter=aiter_ps,
        options=options,
        originals=originals,
        events=events,
        resources=resources,
        owned=lambda: [r for r in resources if r.name in ("ca", "qr")],
        pynccl=pynccl,
        cpu_group=cpu_group,
        device_group=device_group,
    )
    gpu_guard.assert_not_called()


def test_owned_collectives_close_before_vllm_owner(tp_env):
    e = tp_env
    assert init_aiter_dist_from_vllm(2) is True
    assert e.owner.destroy() == "owner-result"

    assert all(r.close_calls == 1 for r in e.owned())
    owner_index = e.events.index("owner.destroy")
    assert all(e.events.index(r.name + ".close") < owner_index for r in e.owned())
    assert e.pynccl.close_calls == 1
    assert e.events.index("pynccl.close") > owner_index
    e.cpu_group.destroy.assert_called_once_with()
    e.device_group.destroy.assert_called_once_with()
    assert e.events.index("device_pg.destroy") > owner_index
    assert e.events.index("cpu_pg.destroy") > owner_index
    assert all(getattr(e.aiter, name) is None for name in e.originals)


def test_repeated_init_and_destroy_do_not_duplicate_owned_resources(tp_env):
    e = tp_env
    assert init_aiter_dist_from_vllm(2) is True
    adapter, hook = e.aiter._TP, e.owner.destroy
    owned = e.owned()
    assert init_aiter_dist_from_vllm(2) is True
    assert e.aiter._TP is adapter
    assert e.owner.destroy is hook
    assert e.owned() == owned
    assert e.events.count("signal.register") == 1

    e.owner.destroy()
    e.owner.destroy()
    assert all(r.close_calls == 1 for r in owned)
    assert e.pynccl.close_calls == 1


@pytest.mark.parametrize("replaced", [("_TP",), ("_PP", "_DP", "_EP")])
def test_destroy_clears_only_identity_matching_globals(tp_env, replaced):
    e = tp_env
    assert init_aiter_dist_from_vllm(2) is True
    replacements = {name: object() for name in replaced}
    for name, value in replacements.items():
        setattr(e.aiter, name, value)

    e.owner.destroy()

    assert all(r.close_calls == 1 for r in e.owned())
    for name in e.originals:
        assert getattr(e.aiter, name) is replacements.get(name)


@pytest.mark.parametrize("failure", ["setup", "ca_disabled", "ca_missing"])
def test_failed_initialization_releases_temporaries_without_publishing(tp_env, failure):
    e = tp_env
    original_destroy = e.owner.destroy
    if failure == "setup":
        e.options.setup_error = RuntimeError("signal registration failed")
    elif failure == "ca_disabled":
        e.options.ca_disabled = True
    else:
        e.aiter._ENABLE_CUSTOM_ALL_REDUCE = False

    assert init_aiter_dist_from_vllm(2) is False

    assert e.owned()
    assert all(r.close_calls == 1 for r in e.owned())
    assert e.owner.destroy == original_destroy
    assert e.owner.destroy_calls == 0
    assert e.pynccl.close_calls == 0
    e.cpu_group.destroy.assert_not_called()
    e.device_group.destroy.assert_not_called()
    e.aiter.set_custom_all_reduce.assert_not_called()
    for name, value in e.originals.items():
        assert getattr(e.aiter, name) is value


def test_original_owner_error_propagates_after_plugin_cleanup(tp_env):
    e = tp_env
    assert init_aiter_dist_from_vllm(2) is True
    error = RuntimeError("original owner failed")
    e.options.owner_error = error

    with pytest.raises(RuntimeError) as caught:
        e.owner.destroy()

    assert caught.value is error
    assert all(r.close_calls == 1 for r in e.owned())
    assert e.owner.destroy_calls == 1
    assert all(getattr(e.aiter, name) is None for name in e.originals)


def test_cleanup_failure_is_fail_closed_and_retryable(tp_env):
    e = tp_env
    assert init_aiter_dist_from_vllm(2) is True
    adapter = e.aiter._TP
    published = {name: getattr(e.aiter, name) for name in e.originals}
    ca = next(r for r in e.owned() if r.name == "ca")
    qr = next(r for r in e.owned() if r.name == "qr")
    error = RuntimeError("ca close failed")
    ca.close_error = error

    with pytest.raises(RuntimeError) as caught:
        e.owner.destroy()

    assert caught.value is error
    assert e.owner.destroy_calls == 0
    assert e.pynccl.close_calls == 0
    e.cpu_group.destroy.assert_not_called()
    e.device_group.destroy.assert_not_called()
    assert adapter.device_communicator.ca_comm is ca
    assert all(getattr(e.aiter, name) is value for name, value in published.items())

    qr_calls = qr.close_calls
    ca.close_error = None
    e.owner.destroy()
    assert ca.close_calls == 2
    assert qr.close_calls == 1
    if qr_calls:
        assert qr.close_calls == qr_calls
    assert e.owner.destroy_calls == 1
    assert all(getattr(e.aiter, name) is None for name in published)


def test_failed_setup_cleanup_error_is_not_swallowed(tp_env):
    e = tp_env
    setup_error = RuntimeError("signal registration failed")
    close_error = RuntimeError("temporary cleanup failed")
    e.options.setup_error = setup_error
    # Inject at the real signal registration boundary, after allocation.
    ca_module = sys.modules["aiter.dist.device_communicators.custom_all_reduce"]
    construct = ca_module.CustomAllreduce

    def failing_resource(**kwargs):
        resource = construct(**kwargs)
        resource.close_error = close_error
        return resource

    ca_module.CustomAllreduce = failing_resource

    with pytest.raises(RuntimeError) as caught:
        init_aiter_dist_from_vllm(2)

    assert caught.value is close_error
    assert e.owner.destroy_calls == 0
    assert e.pynccl.close_calls == 0
    e.cpu_group.destroy.assert_not_called()
    e.device_group.destroy.assert_not_called()
    for name, value in e.originals.items():
        assert getattr(e.aiter, name) is value
