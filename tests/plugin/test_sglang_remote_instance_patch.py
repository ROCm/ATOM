# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""R-Fork receiver patch for ATOM models on SGLang."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

# The patch and the MoE base need SGLang and AITER; skip without them.
pytest.importorskip("sglang")
pytest.importorskip("aiter")

from atom.model_ops.moe import FusedMoEMethodBase
from atom.plugin import prepare as plugin_prepare
from atom.plugin.sglang.patches import remote_instance_patch as rip

MODEL_CONFIG = SimpleNamespace(dtype=torch.bfloat16)
DEVICE_CONFIG = SimpleNamespace(device="cpu")


class _MoEQuantMethod(FusedMoEMethodBase):
    def __init__(self, events):
        self.events = events

    def create_weights(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass

    def get_fused_moe_quant_config(self, *args, **kwargs):
        pass

    def init_prepare_finalize(self, module):
        self.events.append("moe_prepare_finalize")

    def process_weights_after_loading(self, module):
        self.events.append("moe_quant")


@pytest.fixture
def events():
    # Building an SGLang plugin model sets the framework, as on a disk load.
    plugin_prepare._set_framework_backbone("sglang")
    yield []
    plugin_prepare._set_framework_backbone("atom")


def _prepare(model):
    rip.prepare_atom_model_for_receive(model, MODEL_CONFIG, DEVICE_CONFIG)


def _atom_like_model(events):
    class Fp8Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.empty(4, 2, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.weight_scale = torch.nn.Parameter(torch.full((1,), float("nan")))

        def process_weights_after_loading(self):
            # Same check as the Qwen4Exp FP8 PLE table.
            assert self.weight_scale.item() > 0
            # Runs under the model dtype, like a disk load.
            assert torch.get_default_dtype() == MODEL_CONFIG.dtype
            events.append("fp8_embedding_hook")

    class MoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = _MoEQuantMethod(events)

    model = torch.nn.Sequential(Fp8Embedding(), MoE())
    model.remote_instance_load_supported = True
    return model


class _Loader:
    def __init__(self):
        self.calls = []

    def load_model_from_remote_instance_by_nccl(
        self, model, client, model_config, device_config
    ):
        self.calls.append("receive")
        return model


def test_prepare_replays_disk_load_order(events):
    _prepare(_atom_like_model(events))
    # moe_quant once: ATOM's step skips it under SGLang, postprocess_weights runs it.
    assert events == ["fp8_embedding_hook", "moe_prepare_finalize", "moe_quant"]


def test_prepare_refuses_meta_params():
    with pytest.raises(NotImplementedError, match="meta"):
        _prepare(torch.nn.Linear(2, 2, device="meta"))


class _RebindsWeight(torch.nn.Module):
    """Post-load pass that re-binds its weight as a plain tensor (MXFP4 Triton)."""

    remote_instance_load_supported = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2))

    def process_weights_after_loading(self):
        weight = self.weight.data.clone()
        del self.weight
        self.weight = weight


class _EmptiesWeight(_RebindsWeight):
    """Post-load pass that moves its weight elsewhere (MegaMoE)."""

    def process_weights_after_loading(self):
        self.weight.data = torch.empty(0)


class _DropsWeight(_RebindsWeight):
    """Post-load pass that drops a parameter (dynamic online quant input_scale)."""

    def process_weights_after_loading(self):
        self.weight = None


@pytest.mark.parametrize("module_cls", [_RebindsWeight, _EmptiesWeight])
def test_prepare_refuses_weights_moved_out_of_params(events, module_cls):
    with pytest.raises(NotImplementedError, match="weight"):
        _prepare(module_cls())


def test_prepare_allows_dropped_params(events):
    # `layer.x = None` drops the parameter on the seed too: nothing to copy.
    _prepare(_DropsWeight())


class _QuantizesToUnsupported(_RebindsWeight):
    """Post-load pass that ends in an unsupported format (e.g. online MXFP4)."""

    def process_weights_after_loading(self):
        self.remote_instance_load_supported = False


def test_prepare_rechecks_support_after_the_pass(events):
    with pytest.raises(NotImplementedError, match="not supported"):
        _prepare(_QuantizesToUnsupported())


def test_receive_prepares_atom_models_only(monkeypatch):
    loader = _Loader()
    receive = rip._wrap_nccl_receive(_Loader.load_model_from_remote_instance_by_nccl)
    monkeypatch.setattr(
        rip, "prepare_atom_model_for_receive", lambda *a: loader.calls.append("prepare")
    )
    monkeypatch.setattr(rip, "_release_seed", lambda *a: loader.calls.append("release"))
    monkeypatch.setattr(rip, "_all_ranks_ok", lambda ok: ok)  # one receiver rank

    def atom_model(supported):
        attrs = {"__module__": "atom.plugin.x"}
        if supported:
            attrs["remote_instance_load_supported"] = True
        return type("Wrapper", (torch.nn.Module,), attrs)()

    receive(loader, atom_model(True), None, MODEL_CONFIG, DEVICE_CONFIG)
    assert loader.calls == ["prepare", "receive"]

    loader.calls.clear()
    with pytest.raises(NotImplementedError, match="Wrapper"):
        receive(loader, atom_model(False), None, MODEL_CONFIG, DEVICE_CONFIG)
    assert loader.calls == ["release"]  # the seed is freed, nothing is copied

    # Any other failure while preparing (e.g. OOM) frees the seed too.
    loader.calls.clear()

    def oom(*args):
        raise torch.OutOfMemoryError("prepare")

    monkeypatch.setattr(rip, "prepare_atom_model_for_receive", oom)
    with pytest.raises(torch.OutOfMemoryError):
        receive(loader, atom_model(True), None, MODEL_CONFIG, DEVICE_CONFIG)
    assert loader.calls == ["release"]
    monkeypatch.setattr(
        rip, "prepare_atom_model_for_receive", lambda *a: loader.calls.append("prepare")
    )

    loader.calls.clear()

    receive(loader, torch.nn.Linear(1, 1), None, MODEL_CONFIG, DEVICE_CONFIG)
    assert loader.calls == ["receive"]


def test_receive_follows_a_failed_peer_rank(monkeypatch):
    # This rank prepared fine, but another receiver rank failed: release the
    # seed together instead of starting a transfer the seed would hang in.
    loader = _Loader()
    receive = rip._wrap_nccl_receive(_Loader.load_model_from_remote_instance_by_nccl)
    monkeypatch.setattr(
        rip, "prepare_atom_model_for_receive", lambda *a: loader.calls.append("prepare")
    )
    monkeypatch.setattr(rip, "_release_seed", lambda *a: loader.calls.append("release"))
    monkeypatch.setattr(rip, "_all_ranks_ok", lambda ok: False)
    model = type(
        "Wrapper",
        (torch.nn.Module,),
        {"__module__": "atom.plugin.x", "remote_instance_load_supported": True},
    )()
    with pytest.raises(RuntimeError, match="another receiver rank"):
        receive(loader, model, None, MODEL_CONFIG, DEVICE_CONFIG)
    assert loader.calls == ["prepare", "release"]


@pytest.mark.parametrize(
    "world_size, peers_ok, local_ok, expected",
    [
        (1, None, True, True),
        (1, None, False, False),
        (2, True, True, True),
        (2, False, True, False),  # a peer failed
    ],
)
def test_all_ranks_ok(monkeypatch, world_size, peers_ok, local_ok, expected):
    group = SimpleNamespace(world_size=world_size, cpu_group="cpu")
    monkeypatch.setattr(rip, "get_tp_group", lambda: group)

    def all_reduce(flag, op, group):
        assert op == rip.torch.distributed.ReduceOp.MIN and group == "cpu"
        flag.fill_(min(int(flag.item()), int(peers_ok)))

    monkeypatch.setattr(rip.torch.distributed, "all_reduce", all_reduce)
    assert rip._all_ranks_ok(local_ok) is expected


@pytest.mark.parametrize("joined", [True, False])
def test_release_seed_joins_and_drops_the_send_group(monkeypatch, joined):
    calls = []

    class Client:
        _model_update_group = "group"

        def build_group(self, **kwargs):
            calls.append(("build_group", kwargs["gpu_id"], kwargs["tp_rank"]))
            return joined, "msg"

    monkeypatch.setattr(
        rip.torch.distributed.distributed_c10d,
        "destroy_process_group",
        lambda group: calls.append(("destroy", group)),
    )
    loader = SimpleNamespace(load_config=SimpleNamespace(tp_rank=1))
    rip._release_seed(loader, Client(), SimpleNamespace(gpu_id=3))
    expected = [("build_group", 3, 1)] + ([("destroy", "group")] if joined else [])
    assert calls == expected


def test_seed_drops_stale_send_groups(monkeypatch):
    calls = []
    clock = [1000.0]

    @dataclass(slots=True)  # like SGLang's WeightExporter: no instance __dict__
    class Exporter:
        tp_rank: int = 1
        _weights_send_group: dict = field(default_factory=dict)

        def init_weights_send_group_for_remote_instance(
            self, master_address, ports, group_rank, world_size, group_name
        ):
            calls.append(("init", group_name))
            self._weights_send_group[f"{group_name}_62101_1"] = group_name
            return True, "ok"

    monkeypatch.setattr(rip.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rip, "_send_group_born", {})
    monkeypatch.setattr(
        rip.torch.distributed.distributed_c10d,
        "destroy_process_group",
        lambda group: calls.append(("destroy", group)),
    )
    init = rip._wrap_seed_init_group(
        Exporter.init_weights_send_group_for_remote_instance
    )
    exporter = Exporter()
    for name in ("old", "young"):  # two refused receivers, a minute apart
        init(exporter, "127.0.0.1", "62100,62101", 0, 2, name)
        clock[0] += 60
    calls.clear()

    # Same name: dropped at once, so the retry does not hit a duplicate name.
    init(exporter, "127.0.0.1", "62100,62101", 0, 2, "young")
    assert calls == [("destroy", "young"), ("init", "young")]

    # Past the group timeout: "old" was given up by its receiver; "young" is
    # recent and may still be waiting for its send, so it stays.
    calls.clear()
    clock[0] += rip._GROUP_TIMEOUT_S - 30
    init(exporter, "127.0.0.1", "62100,62101", 0, 2, "new")
    assert calls == [("destroy", "old"), ("init", "new")]
    assert set(exporter._weights_send_group) == {"young_62101_1", "new_62101_1"}


def test_apply_wraps_once(monkeypatch):
    class Loader(_Loader):
        pass

    class Exporter:
        def init_weights_send_group_for_remote_instance(self, *args):
            pass

    monkeypatch.setattr(rip, "RemoteInstanceModelLoader", Loader)
    monkeypatch.setattr(rip, "WeightExporter", Exporter)
    rip.apply_remote_instance_patch()
    wrapped = Loader.load_model_from_remote_instance_by_nccl
    seed_init = Exporter.init_weights_send_group_for_remote_instance
    rip.apply_remote_instance_patch()
    assert Loader.load_model_from_remote_instance_by_nccl is wrapped
    assert Exporter.init_weights_send_group_for_remote_instance is seed_init
    assert wrapped._atom_rfork
    assert seed_init.__wrapped__ is not seed_init  # the seed side is wrapped too


@pytest.mark.parametrize(
    "quant_type, quant_dtype, weight_dtype, expected",
    [
        ("per_Token", torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, True),  # PTPC
        ("No", torch.bfloat16, torch.bfloat16, False),  # BF16: untested
        ("per_Tensor", torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, False),
        ("per_Token", torch.bfloat16, torch.bfloat16, False),
        ("per_Token", torch.float8_e4m3fnuz, torch.float4_e2m1fn_x2, False),  # MXFP4
    ],
)
def test_qwen4_exp_opts_in_for_ptpc_fp8_only(
    quant_type, quant_dtype, weight_dtype, expected
):
    # Imports the SGLang Qwen4Exp wrapper; kept local to this test.
    from atom.plugin.sglang.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    supported = Qwen4ExpForConditionalGeneration.remote_instance_load_supported.fget
    model = torch.nn.Linear(2, 2)
    model.weight.data = torch.empty(2, 2, dtype=weight_dtype)
    quant = SimpleNamespace(
        quant_type=SimpleNamespace(name=quant_type), quant_dtype=quant_dtype
    )
    model.atom_config = SimpleNamespace(
        quant_config=SimpleNamespace(global_quant_config=quant)
    )
    assert supported(model) is expected
