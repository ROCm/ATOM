from types import SimpleNamespace


def test_aiter_device_communicator_reuses_vllm_pynccl(monkeypatch):
    from aiter.dist.device_communicators import base_device_communicator
    from aiter.dist.device_communicators import custom_all_reduce
    from aiter.dist.device_communicators import quick_all_reduce
    from atom.plugin.vllm.tp_group_reuse import _create_aiter_device_communicator

    existing_pynccl = object()
    vllm_tp = SimpleNamespace(
        cpu_group=object(),
        device=object(),
        device_group=object(),
        device_communicator=SimpleNamespace(pynccl_comm=existing_pynccl),
    )

    def fake_base_init(self, cpu_group, device, device_group, unique_name):
        self.cpu_group = cpu_group
        self.device = device
        self.device_group = device_group
        self.unique_name = unique_name
        self.world_size = 8

    class FakeCustomAllreduce:
        disabled = False

        def __init__(self, group, device):
            self.group = group
            self.device = device

    class FakeQuickAllReduce:
        def __init__(self, group, device):
            self.group = group
            self.device = device

    monkeypatch.setattr(
        base_device_communicator.DeviceCommunicatorBase,
        "__init__",
        fake_base_init,
    )
    monkeypatch.setattr(custom_all_reduce, "CustomAllreduce", FakeCustomAllreduce)
    monkeypatch.setattr(quick_all_reduce, "QuickAllReduce", FakeQuickAllReduce)

    communicator = _create_aiter_device_communicator(vllm_tp)

    assert communicator.pynccl_comm is existing_pynccl
    assert isinstance(communicator.ca_comm, FakeCustomAllreduce)
    assert isinstance(communicator.qr_comm, FakeQuickAllReduce)
