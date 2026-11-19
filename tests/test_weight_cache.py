# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Weight cache logic that needs no daemon: code fingerprints, the before/after
load diff, export/import (CPU tensors stand in for device ones), the key filters, and
daemon management with fake processes."""

import dataclasses
import json
import os
import pathlib
import pickle
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types
import weakref
from types import SimpleNamespace

import pytest
import torch
from conftest import gpu_stack_import
from torch import nn

import atom.model_loader.weight_cache.__main__ as cli
from atom.model_loader.weight_cache import (
    export,
    key,
    preload,
    transport,
    values,
)
from atom.utils import envs

# --------------------------------------------------------------------------
# Key
# --------------------------------------------------------------------------


def test_checkpoint_rewritten_in_place_changes_the_key(tmp_path):
    def shard(body: bytes):
        header = b'{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}'
        (tmp_path / "model.safetensors").write_bytes(
            len(header).to_bytes(8, "little") + header + body
        )

    (tmp_path / "config.json").write_text("{}")
    shard(b"\0" * 8 + b"x" * (2 << 20))
    a = key._checkpoint_digest(str(tmp_path))
    assert key._checkpoint_digest(str(tmp_path)) == a
    time.sleep(0.01)
    shard(b"\1" * 8 + b"x" * (2 << 20))  # same names, shapes and size: new weights
    assert key._checkpoint_digest(str(tmp_path)) != a


def _key_config(tmp_path, **settings):
    """A toy launch's config with ``settings`` on it."""
    (tmp_path / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(to_dict=lambda: {"hidden_size": 8}),
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        speculative_config=None,
        tensor_parallel_size=1,
        tp_world_size=1,
        pcp_group_size=1,
        enable_dp_attention=False,
        enable_expert_parallel=False,
        **settings,
    )


def _key(tmp_path, monkeypatch, **settings):
    return key.compute_key(_key_config(tmp_path, **settings), 0)


def test_the_key_holds_the_listed_settings_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ATOM_MOE_GU_ITLV", raising=False)
    base = _key(tmp_path, monkeypatch, moe_backend="standard", max_model_len=4096)
    # A serving option is not on the list: the same weights.
    same = _key(tmp_path, monkeypatch, moe_backend="standard", max_model_len=8192)
    assert same == base
    diffs = key._diff_key(base, _key(tmp_path, monkeypatch, moe_backend="mega"))
    assert diffs == ["options.moe_backend: daemon=standard client=mega"]
    monkeypatch.setenv("ATOM_MOE_GU_ITLV", "1")
    monkeypatch.setenv("ATOM_LOADER_NUM_THREADS", "8")  # how it loads, not what
    diffs = key._diff_key(base, _key(tmp_path, monkeypatch, moe_backend="standard"))
    assert diffs == ["env.ATOM_MOE_GU_ITLV: daemon=None client=1"]


def test_a_setting_of_a_sub_config_is_in_the_key(tmp_path, monkeypatch):
    # Project-before-merge: loading then keeps a tensor the other way lacks.
    off, on = (
        _key(
            tmp_path,
            monkeypatch,
            dcp_config=SimpleNamespace(enable_project_before_merge=pbm),
        )
        for pbm in (False, True)
    )
    name = "options.dcp_config.enable_project_before_merge"
    assert key._diff_key(off, on) == [f"{name}: daemon=False client=True"]


def test_the_daemons_key_names_the_checkpoint_as_loading_left_it(tmp_path, monkeypatch):
    daemon = gpu_stack_import("atom.model_loader.weight_cache.daemon")
    config = _key_config(tmp_path)

    def load(config, build, load):  # the first load of a hub model downloads it
        (tmp_path / "model.safetensors").write_bytes(b"\0" * 16)
        return SimpleNamespace(model=nn.Linear(1, 1))

    monkeypatch.setattr(daemon, "_load_and_export", load)
    loader = daemon._DaemonLoad(0)
    loader(config, None, torch.device("cpu"))
    assert loader.key == key.compute_key(config, 0)  # as a server sees it now


@pytest.mark.parametrize(
    "name",
    ["ATOM_USE_TRITON_MOE_DECODE", "ATOM_USE_UNIFIED_ATTN", "ATOM_FORCE_ATTN_TRITON"],
)
def test_a_backend_switch_loading_goes_by_is_in_the_key(tmp_path, monkeypatch, name):
    # They decide the layout experts are kept in, and the dtype of a sink.
    monkeypatch.setenv(name, "1")
    on = _key(tmp_path, monkeypatch)
    monkeypatch.setenv(name, "0")
    off = _key(tmp_path, monkeypatch)
    assert key._diff_key(on, off) == [f"env.{name}: daemon=1 client=0"]


def test_the_quantization_config_is_in_the_key(tmp_path, monkeypatch):
    quant = [SimpleNamespace(compute_hash=lambda h=h: h) for h in ("fp8", "online")]
    a, b = (_key(tmp_path, monkeypatch, quant_config=q) for q in quant)
    assert key._diff_key(a, b) == ["quant: daemon=fp8 client=online"]


# --------------------------------------------------------------------------
# Values, digests, key filters
# --------------------------------------------------------------------------


def module_level_function():
    return 1


def _guard(fn):
    def wrapper(*args, **kwargs):  # named after the decorator, like aiter's guards
        return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn
    return wrapper


@_guard
def guarded_function():
    return 2


def test_pack_unpack_roundtrip():
    ex = export._Exporter()
    value = {
        "a": [torch.arange(3), 2, ("x", torch.ones(2))],
        "fn": module_level_function,
        "guarded": guarded_function,
    }
    # As sent: the wrapper itself would not pickle.
    packed = pickle.loads(pickle.dumps(export._pack(value, ex)))
    make, _ = export._maker({"storages": {}, "host": ex.host}, torch.device("cpu"))
    out = export._unpack(packed, make)
    assert torch.equal(out["a"][0], torch.arange(3)) and out["a"][1] == 2
    assert isinstance(out["a"][2], tuple) and torch.equal(out["a"][2][1], torch.ones(2))
    assert out["fn"] is module_level_function and out["guarded"] is guarded_function
    with pytest.raises(values._Unshippable):
        export._pack(lambda: 0, ex)


def test_values_that_differ_do_not_render_alike():
    assert values._norm(1 + 2j) != values._norm(3 + 4j)

    def boom(v):
        raise RuntimeError("cannot be fingerprinted")

    # Not known to be unchanged: counts as changed, on this side and the other.
    assert values.safe_fp(boom, 1) != values.safe_fp(boom, 1)


def test_digest_sees_reorder():
    t = torch.arange(1000, dtype=torch.float32)
    u = t.clone()
    u[[3, 7]] = u[[7, 3]]
    assert values._digest(t) == values._digest(t.clone()) != values._digest(u)


def test_digest_sees_data_moved_between_chunks(monkeypatch):
    monkeypatch.setattr(values, "_DIGEST_CHUNK", 8)
    monkeypatch.setattr(values, "_DIGEST_WEIGHTS", {})
    t = torch.arange(16, dtype=torch.int64)
    u = t.clone()
    u[[1, 9]] = u[[9, 1]]  # the same position in two chunks
    assert values._digests([t, u])[0] != values._digests([u])[0]


def test_norm_uses_compute_hash():
    class Q:
        def __init__(self, h):
            self.h = h

        def compute_hash(self):
            return self.h

    assert values._norm(Q("a")) != values._norm(Q("b"))


# --------------------------------------------------------------------------
# Before/after diff, export and import (CPU)
# --------------------------------------------------------------------------


class QuantState:
    def __init__(self):
        self.mode = "raw"
        self.scale = torch.ones(2)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        self.register_buffer("rope", torch.arange(8.0))
        self.register_buffer("scratch", torch.zeros(3))
        self.flag = False
        self.q = QuantState()
        self.child = nn.Linear(2, 2)

    def fake_load(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w.data.copy_(torch.randn(4, 4, generator=g))
        self.register_parameter(
            "w_shuffled", nn.Parameter(torch.randn(8, generator=g), requires_grad=False)
        )
        self.scratch[1] = 5.0  # in place
        self.flag = True
        self.absorbed = torch.randn(2, 2, generator=g)
        self.lst = [torch.ones(2), 3]
        self.q.mode = "shuffled"
        self.q.scale = torch.full((2,), 2.0)
        del self.child._parameters["bias"]


@pytest.fixture
def toy_packages(monkeypatch):
    monkeypatch.setattr(
        values, "_DIFF_PACKAGES", values._DIFF_PACKAGES + (QuantState.__module__,)
    )


def _export(model, load):
    built, before = key.structure(model), export.snapshot(model)
    load(model)
    exp = export.build_export(model, before, export.snapshot(model))
    exp["storages"] = export.share_storages(exp.pop("views"))
    return built, exp


def test_export_import_reproduces_the_load(toy_packages):
    daemon = Toy()
    built, exp = _export(daemon, Toy.fake_load)
    assert exp["unshippable"] == []
    assert "scratch" in exp["plan"][""]["buffers"]  # written in place
    with export.params_on_meta():
        client = Toy()
    assert key._diff_key(built, key.structure(client)) == []
    export.apply_export(client, exp, torch.device("cpu"))
    assert export._leftover_meta(client) == []
    for name in ("w", "w_shuffled", "scratch", "absorbed", "rope"):
        assert torch.equal(getattr(client, name), getattr(daemon, name)), name
    assert (
        client.flag
        and client.q.mode == "shuffled"
        and torch.equal(client.q.scale, daemon.q.scale)
    )
    assert torch.equal(client.lst[0], torch.ones(2)) and client.lst[1] == 3
    assert "bias" not in client.child._parameters
    assert client.child.weight.requires_grad and not client.w.requires_grad


def test_none_and_tensors_loading_sets_on_parameters_travel(toy_packages):
    def load(m):
        m.child.bias = None  # a registered parameter dropped by loading
        m.w.scale = torch.full((2,), 3.0)  # a tensor hung on a parameter

    _, exp = _export(Toy(), load)
    assert exp["unshippable"] == []
    with export.params_on_meta():
        client = Toy()
    export.apply_export(client, exp, torch.device("cpu"))
    assert "bias" in client.child._parameters and client.child.bias is None
    assert torch.equal(client.w.scale, torch.full((2,), 3.0))


def test_a_parameter_slot_loading_registers_empty_travels():
    def load(m):  # e.g. the bias of experts quantized while loading
        m.register_parameter("extra", None)

    _, exp = _export(nn.Linear(2, 2), load)
    with export.params_on_meta():
        client = nn.Linear(2, 2)
    export.apply_export(client, exp, torch.device("cpu"))
    assert "extra" in client._parameters and client.extra is None


def test_leaving_modules_out_reports_what_went_with_them():
    item = {"params": {}, "buffers": {}, "attrs": {"mode": ("V", 1)}, "delete": []}
    exp = {
        "plan": {"embed": item, "embed::cfg": item, "w": item},
        "views": {},
        "host": {},
    }
    # An object below a module that is left out may be held elsewhere too.
    assert export.leave_out(exp, {"embed"}) == ["embed::cfg"]
    assert set(exp["plan"]) == {"w"}


def test_a_parameter_registered_empty_is_filled_by_loading():
    def load(m):  # e.g. the scale of a layer quantized while loading
        m.bias = nn.Parameter(torch.full((2,), 5.0), requires_grad=False)

    _, exp = _export(nn.Linear(2, 2, bias=False), load)
    with export.params_on_meta():
        client = nn.Linear(2, 2, bias=False)
    export.apply_export(client, exp, torch.device("cpu"))
    assert torch.equal(client.bias, torch.full((2,), 5.0))


def test_an_object_at_a_freed_ones_address_is_not_taken_for_it(monkeypatch):
    class Kernel:  # outside the walked packages: told apart by its address
        pass

    # As if each Kernel were allocated where the one before it was freed.
    fp = export._fp
    monkeypatch.setattr(
        export,
        "_fp",
        lambda v, *a, **k: (
            ("O", 1, "Kernel") if isinstance(v, Kernel) else fp(v, *a, **k)
        ),
    )

    def load(m):
        m.kernel = Kernel()

    model = nn.Linear(2, 2, bias=False)
    model.kernel = Kernel()
    _, exp = _export(model, load)
    assert any("kernel" in u for u in exp["unshippable"])
    _, exp = _export(model, lambda m: None)  # the same one is left alone
    assert exp["unshippable"] == [] and "kernel" not in exp["plan"][""]["attrs"]


def test_a_snapshot_does_not_keep_the_parameters_loading_replaces():
    model = nn.Linear(2, 2, bias=False)
    old = weakref.ref(model.weight)
    before = export.snapshot(model)
    model.weight = nn.Parameter(torch.ones(2, 2), requires_grad=False)
    assert old() is None  # freed, so its address may name the new one
    exp = export.build_export(model, before, export.snapshot(model))
    assert exp["plan"][""]["params"]["weight"]["replaced"]


@dataclasses.dataclass(frozen=True)
class Spec:
    kind: str = "raw"
    block: tuple = (1, 32)


def test_an_immutable_record_loading_replaces_travels_as_a_value(toy_packages):
    def build():
        m = Toy()
        m.spec = Spec()
        return m

    def load(m):  # e.g. the spec of a layer quantized while loading
        m.spec = Spec("fp8", (128, 128))

    _, exp = _export(build(), load)
    assert exp["unshippable"] == []
    with export.params_on_meta():
        client = build()
    shared = client.spec  # other layers may hold the same record
    export.apply_export(client, exp, torch.device("cpu"))
    assert client.spec == Spec("fp8", (128, 128)) and shared == Spec()


class Method:
    def __init__(self, state):
        self.state, self.kind, self.scale = state, "fp8", torch.ones(2)


def test_an_object_loading_creates_travels_with_what_it_refers_to(toy_packages):
    def load(m):  # e.g. the method of experts quantized while loading
        m.method = Method(m.q)  # refers to an object every client has
        m.hooks = {"loader": m.fake_load}  # and to a method of the model

    _, exp = _export(Toy(), load)
    assert exp["unshippable"] == []
    exp["plan"] = pickle.loads(pickle.dumps(exp["plan"]))  # as sent
    with export.params_on_meta():
        client = Toy()
    export.apply_export(client, exp, torch.device("cpu"))
    assert type(client.method) is Method and client.method.state is client.q
    assert client.method.kind == "fp8" and torch.equal(
        client.method.scale, torch.ones(2)
    )
    assert client.hooks["loader"].__self__ is client


def test_an_object_loading_creates_stays_one_object(toy_packages):
    def load(m):
        m.method = Method(m.q)
        m.also = [m.method]  # held twice

    _, exp = _export(Toy(), load)
    exp["plan"] = pickle.loads(pickle.dumps(exp["plan"]))
    with export.params_on_meta():
        client = Toy()
    export.apply_export(client, exp, torch.device("cpu"))
    assert client.also[0] is client.method


def test_what_was_not_sent_is_not_referred_to(toy_packages):
    def load(m):
        m.method = Method(m.q)
        m.w.hooks = [m.method, lambda: 0]  # dropped: a closure cannot travel
        m.also = m.method

    _, exp = _export(Toy(), load)
    assert exp["unshippable"] == []
    with export.params_on_meta():
        client = Toy()
    export.apply_export(client, exp, torch.device("cpu"))
    assert client.also is client.method and not hasattr(client.w, "hooks")


def test_an_object_of_the_config_is_not_copied(toy_packages):
    config, model = QuantState(), Toy()  # the config: held outside the model
    before = export.snapshot(model, config)
    model.settings = config  # loading hangs it on the model
    exp = export.build_export(model, before, export.snapshot(model))
    assert any(u.startswith("<root>.settings: ") for u in exp["unshippable"])


def test_an_object_moved_onto_anothers_path_is_not_copied(toy_packages):
    def build():
        m = Toy()
        m.child.other = QuantState()
        return m

    def load(m):
        m.child.other.mode = "x"
        m.q = m.child.other  # now reached where another object was

    _, exp = _export(build(), load)
    assert any(u.startswith("<root>.q: ") for u in exp["unshippable"])


def test_an_object_held_where_no_path_leads_is_not_copied(toy_packages):
    def build():
        m = Toy()
        m.holders = [QuantState()]
        return m

    def load(m):
        m.picked = m.holders[0]  # a client's own is another object

    _, exp = _export(build(), load)
    assert any(u.startswith("<root>.picked: ") for u in exp["unshippable"])


def test_prepare_finalize_is_skipped_on_methods_loading_creates():
    FusedMoEMethodBase = gpu_stack_import("atom.model_ops.moe").FusedMoEMethodBase

    class Method(FusedMoEMethodBase):
        def init_prepare_finalize(self, layer):
            raise AssertionError("ran in the daemon")

    class Left(Method):
        pass

    class Right(Method):
        pass

    class Swapped(Left, Right):  # reached through both parents
        def init_prepare_finalize(self, layer):
            raise AssertionError("ran in the daemon")

    with export._skip_prepare_finalize():
        Swapped.init_prepare_finalize(None, None)
    with pytest.raises(AssertionError):  # and put back afterwards
        Swapped.init_prepare_finalize(None, None)


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
        self.b.weight = self.a.weight


@pytest.mark.parametrize("untie", [False, True])
def test_tied_parameters_stay_tied_unless_loading_untied_them(untie):
    def load(m):
        m.a.weight.data.fill_(1.0)
        if untie:  # e.g. one side quantized
            m.b.weight = nn.Parameter(torch.full((2, 2), 7.0), requires_grad=False)

    daemon = Tied()
    _, exp = _export(daemon, load)
    with export.params_on_meta():
        client = Tied()
    export.apply_export(client, exp, torch.device("cpu"))
    assert torch.equal(client.a.weight, daemon.a.weight)
    assert torch.equal(client.b.weight, daemon.b.weight)
    assert (client.a.weight is client.b.weight) is not untie


def test_host_tensors_keep_their_aliasing(toy_packages):
    def load(m):
        m.absorbed = torch.arange(4.0)
        m.tail = m.absorbed[2:]

    _, exp = _export(Toy(), load)
    with export.params_on_meta():
        client = Toy()
    export.apply_export(client, exp, torch.device("cpu"))
    client.absorbed[3] = -1.0
    assert client.tail.tolist() == [2.0, -1.0]


def test_config_written_during_load_is_refused():
    config = SimpleNamespace(hf_config=SimpleNamespace(rope_theta=1.0), block_size=16)
    before = export.process_snapshot(config)
    config.hf_config.rope_theta = 2.0
    exp = export.process_export([before, export.process_snapshot(config)])
    assert any(u.startswith("hf_config:rope_theta") for u in exp["unshippable"])


def test_unshareable_state_is_reported(toy_packages):
    class Kernel:
        pass

    def load(m):
        m.kernel = Kernel()
        del m.child

    _, exp = _export(Toy(), load)
    assert any("kernel" in u for u in exp["unshippable"])
    assert any("child: submodule removed" in u for u in exp["unshippable"])


def test_change_out_of_the_walks_reach_is_reported(toy_packages):
    def build():
        m = Toy()
        m.holders = [QuantState()]  # in a container: the walk does not export it
        return m

    def load(m):
        m.holders[0].mode = "shuffled"

    _, exp = _export(build(), load)
    assert any("holders[0]::mode: changed during load" in u for u in exp["unshippable"])


def test_change_to_an_object_now_reached_another_way_is_reported(toy_packages):
    def build():
        m = Toy()
        m.child.shared = m.q  # one object, held twice
        return m

    def load(m):
        m.q.mode = "shuffled"
        m.q = None  # the walk now reaches it through the other holder

    _, exp = _export(build(), load)
    assert any(
        u.startswith("child::shared: changed during load") for u in exp["unshippable"]
    )


def test_a_table_of_the_daemons_addresses_is_refused(toy_packages):
    def load(m):
        m.absorbed = torch.arange(4.0)
        m.bases = torch.tensor([m.absorbed.data_ptr()])  # where this process has it

    _, exp = _export(Toy(), load)
    assert any("holds addresses" in u for u in exp["unshippable"])


def test_submodule_replaced_by_another_class_is_reported(toy_packages):
    def load(m):
        m.child = nn.Identity()  # same path, another forward

    _, exp = _export(Toy(), load)
    assert any(u.startswith("child: replaced by a ") for u in exp["unshippable"])


def test_process_state_loading_changes_travels_or_is_refused(monkeypatch):
    mod = types.ModuleType("atom._wc_test_state")

    class Settings:
        LAYOUT = 1

    class Holder:
        mode = "raw"

    for cls in (Settings, Holder):
        cls.__module__ = mod.__name__
    mod.Settings, mod.Holder, mod.FLAG, mod.TABLE = Settings, Holder, False, {"a": 1}
    mod.holder = Holder()
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.delenv("ATOM_TEST_LOAD_SET", raising=False)
    before = export.process_snapshot()
    mod.FLAG, Settings.LAYOUT = True, 16  # what loading decided
    mod.TABLE["b"] = 2
    os.environ["ATOM_TEST_LOAD_SET"] = "1"
    mod.kernel = lambda: 0  # cannot be shared
    mod.registry = Holder()  # nor an object: a client may hold its own elsewhere
    mod.holder.mode = "set"  # an object's attribute: out of reach
    new = types.ModuleType("atom._wc_test_imported")  # imported by the load:
    new.make = lambda: 0  # importing it again builds it again
    monkeypatch.setitem(sys.modules, new.__name__, new)
    exp = export.process_export([before, export.process_snapshot()])
    changed = {(kind, ".".join(rest)) for kind, _, rest, _ in exp["changes"]}
    assert changed == {
        ("global", "FLAG"),
        ("global", "TABLE"),
        ("class", "Settings.LAYOUT"),
    }
    assert exp["env"] == {"ATOM_TEST_LOAD_SET": "1"}
    assert any(u.startswith(f"{mod.__name__}:kernel") for u in exp["unshippable"])
    assert any(u.startswith(f"{mod.__name__}:registry") for u in exp["unshippable"])
    assert any(u.startswith(f"{mod.__name__}:holder.mode") for u in exp["unshippable"])
    # A client starts from the state before loading.
    mod.FLAG, mod.TABLE, Settings.LAYOUT = False, {"a": 1}, 1
    held = mod.TABLE  # as a module that imported the name holds it
    del os.environ["ATOM_TEST_LOAD_SET"]
    exp["storages"] = export.share_storages(exp.pop("views"))
    export.apply_process(exp, torch.device("cpu"))
    assert mod.FLAG and held == {"a": 1, "b": 2} and Settings.LAYOUT == 16
    assert os.environ["ATOM_TEST_LOAD_SET"] == "1"


def test_module_tree_change_is_seen():
    built = key.structure(Toy())
    with export.params_on_meta():
        client = Toy()
    client.extra = nn.Linear(2, 2)  # a constructor now builds another submodule
    assert key._diff_key(built, key.structure(client))


def test_params_on_meta_keeps_the_constructor_object():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            p = nn.Parameter(torch.empty(3))
            self.register_parameter("p", p)
            p.loader = "set after registration"  # like set_weight_attrs
            self.held = p

    with export.params_on_meta():
        m = M()
    assert m.p.is_meta and m.p is m.held and m.p.loader == "set after registration"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_zero_byte_cuda_buffer_is_rebuilt_locally():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("e", torch.zeros(0, 5, device="cuda"))

    daemon = M()
    before = export.snapshot(daemon)
    daemon.e = torch.zeros(0, 5, device="cuda")  # loading replaced it
    exp = export.build_export(daemon, before, export.snapshot(daemon))
    exp["storages"] = export.share_storages(exp.pop("views"))
    assert exp["plan"][""]["buffers"]["e"].get("empty")  # no allocation to share
    client = M()
    export.apply_export(client, exp, torch.device("cuda"))
    assert client.e.shape == (0, 5) and client.e.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_compact_moves_weights_out_of_segments_with_gaps():
    daemon = gpu_stack_import("atom.model_loader.weight_cache.daemon")

    torch.cuda.empty_cache()
    torch.empty(
        100 << 20, dtype=torch.uint8, device="cuda"
    )  # freed; its segment stays cached
    kept = torch.arange(30 << 18, dtype=torch.float32, device="cuda")  # lands in it
    small = torch.ones(1000, device="cuda")
    views = {
        i: torch.empty(0, dtype=torch.uint8, device="cuda").set_(
            t.untyped_storage(), 0, (t.untyped_storage().nbytes(),), (1,)
        )
        for i, t in enumerate((kept, small))
    }
    expected = {k: v.cpu() for k, v in views.items()}
    pool = daemon._compact(views, "test")
    assert all(torch.equal(views[k].cpu(), expected[k]) for k in views)
    assert kept.untyped_storage().nbytes() == 0  # moved out of the 70 MiB gap
    assert small.untyped_storage().nbytes() > 0  # no gap worth moving it for
    del views, pool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_daemon_leaves_out_what_attaching_replaces():
    """A drafter whose attach_to_target swaps its embedding for the target's:
    the daemon holds no copy of the drafter's own."""
    daemon = gpu_stack_import("atom.model_loader.weight_cache.daemon")

    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 32, device="cuda")
            self.w = nn.Parameter(torch.zeros(32, device="cuda"), requires_grad=False)

    class Drafter:
        def __init__(self):
            self.model = Draft()

        def attach_to_target(self, target, loaded):
            del self.model.embed
            self.model.embed = target.embed

    d, target = Drafter(), Draft()
    before = export.snapshot(d.model)
    d.model.w.fill_(1.0)  # its load
    exp = export.build_export(d.model, before, export.snapshot(d.model))
    load = SimpleNamespace(drafter=d, model=d.model, loaded=set(), export=exp)
    assert daemon._leave_out_shared(load, target) == []
    assert set(exp["plan"]) == {""}
    assert list(exp["views"]) == [d.model.w.untyped_storage().data_ptr()]


# --------------------------------------------------------------------------
# ATOM_PRELOAD daemon management
# --------------------------------------------------------------------------


@pytest.fixture
def cache_dir(monkeypatch):
    # Under /tmp: a socket path holds 107 bytes.
    d = pathlib.Path(tempfile.mkdtemp(prefix="wc", dir="/tmp")) / "wc"
    monkeypatch.setattr(transport, "_DIR", str(d))
    for k in ("ATOM_PRELOAD", preload._LAUNCH_ENV):
        monkeypatch.setenv(k, "")  # so what a test leaves there is undone too
        monkeypatch.delenv(k)
    yield d
    shutil.rmtree(d.parent, ignore_errors=True)


def _wait_for(cond, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)


def test_process_helpers():
    assert (
        preload._pid_running(os.getpid())
        and not preload._pid_running(0)
        and not preload._pid_running(None)
    )
    assert preload._ancestors(os.getpid())[0] == os.getppid()


def test_record_launch_saves_the_environment_from_exec(cache_dir, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    monkeypatch.setenv("WC_TEST_SET_AT_RUNTIME", "1")  # like an import setting a var
    preload.record_launch()
    path = preload._launch_file()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600  # holds the environment
    with open(path) as f:
        launch = json.load(f)
    assert launch["argv"][0] == sys.executable and launch["cwd"] == os.getcwd()
    assert "PATH" in launch["env"] and "WC_TEST_SET_AT_RUNTIME" not in launch["env"]


def test_spawned_daemon_is_detached_and_started_once(cache_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    code = (
        "import os, time\n"
        "open(os.environ['WC_OUT'], 'w').write(os.environ['ATOM_PRELOAD'])\n"
        "time.sleep(60)\n"
    )
    os.makedirs(cache_dir, mode=0o700)
    launch = cache_dir / "launch_1_1.json"
    env = {**os.environ, "WC_OUT": str(out), "ATOM_PRELOAD": "1"}
    launch.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", code],
                "cwd": str(tmp_path),
                "env": env,
                "cpus": sorted(os.sched_getaffinity(0)),
            }
        )
    )
    monkeypatch.setenv(preload._LAUNCH_ENV, "1_1")
    first = preload._spawn_daemon()
    try:
        assert preload._spawn_daemon() == first  # the other ranks get the same daemon
        _wait_for(lambda: out.exists() and out.read_text())
        assert out.read_text() == "daemon"  # the same launch, as a daemon
        assert preload._pid_running(first["pid"])
        assert os.getpid() not in preload._ancestors(
            first["pid"]
        )  # survives a tree kill
        assert os.getsid(first["pid"]) != os.getsid(0)  # and a Ctrl-C to our group
    finally:
        os.kill(first["pid"], signal.SIGKILL)


_FAKE_MAIN = """
import os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
def stop(*_):
    child.terminate()
    child.wait()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(0.1)
"""

_FAKE_RUNNER = """
import os, signal, sys, time
from atom.model_loader.weight_cache.transport import _DaemonLock
lock = _DaemonLock(
    sys.argv[1], name="TP0", model="toy-model", daemon_pid=os.getppid(), idle_timeout_s=3600
)
def stop(*_):
    lock.release()  # like a runner: the lock goes early in the shutdown
    time.sleep(2)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(0.1)
"""


def test_status_and_stop_a_daemon_launch(cache_dir, tmp_path, capsys):
    """A fake daemon launch: a main process whose child holds a GPU's lock,
    the way a runner does."""
    (tmp_path / "main.py").write_text(_FAKE_MAIN)
    (tmp_path / "runner.py").write_text(_FAKE_RUNNER)
    path = os.path.join(transport.socket_dir(), "gpu_test.sock")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    main = subprocess.Popen(
        [sys.executable, str(tmp_path / "main.py"), str(tmp_path / "runner.py"), path],
        env=env,
        start_new_session=True,
    )
    try:
        _wait_for(
            lambda: transport.daemon_alive(path)
            and (transport.daemon_state(path) or {}).get("pid")
        )
        st = transport.daemon_state(path)
        assert (
            st["daemon_pid"] == main.pid
            and st["state"] == "loading"
            and st["model"] == "toy-model"
        )
        assert cli.main(["status"]) == 0
        shown = capsys.readouterr().out
        assert f"daemon pid {main.pid}: toy-model" in shown and "TP0: loading" in shown
        assert cli.main(["stop"]) == 0
        assert f"stopped daemon pid {main.pid}" in capsys.readouterr().out
        # Returns once the runner has exited, not when it let go of the lock.
        assert not transport.daemon_alive(path) and not preload._pid_running(st["pid"])
        assert main.wait(10) == 0
        assert cli.main(["status"]) == 0
        assert "no weight cache daemon running" in capsys.readouterr().out
    finally:
        if main.poll() is None:
            os.killpg(main.pid, signal.SIGKILL)  # the main process and its runner


def test_idle_timeout_counts_connected_servers(cache_dir):
    path = os.path.join(transport.socket_dir(), "gpu_idle.sock")
    idle, all_serving = threading.Event(), threading.Event()
    server = transport._Server(path, {"key": 1}, {}, "t")
    server.watch_idle(1, idle.set, all_serving)
    try:
        # The clock starts once every rank of the daemon serves.
        assert not idle.wait(1.6)
        all_serving.set()
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(path)
        transport._send(conn, {"cmd": "get", "who": "test"})
        assert transport._recv(conn)["key"] == 1
        assert not idle.wait(1.6)  # a connected server keeps the daemon
        assert transport._ping(path)["clients"] == 1
        conn.close()
        assert idle.wait(3)
        assert transport._ping(path)["clients"] == 0
        # Stopping now: a server that arrives late is turned away.
        late = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        late.connect(path)
        transport._send(late, {"cmd": "get", "who": "late"})
        assert "shutting down" in transport._recv(late)["error"]
        late.close()
    finally:
        server.close()


def _get(path):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(path)
    transport._send(conn, {"cmd": "get", "who": "test"})
    return conn, transport._recv(conn)


def test_write_into_shared_weights_is_caught_after_the_writer_left(cache_dir):
    path = os.path.join(transport.socket_dir(), "gpu_check.sock")
    weights = torch.zeros(64, dtype=torch.uint8)
    server = transport._Server(path, {"key": 1}, {1: weights}, "t")
    try:
        assert not [p for p in os.listdir(transport.socket_dir()) if p.endswith(".tmp")]
        conn, payload = _get(path)
        assert "error" not in payload
        weights[5] = 7  # the connected server writes into the shared weights
        conn.close()
        _wait_for(lambda: server.poisoned, timeout=10)  # checked once it left
        _, payload = _get(path)
        assert "changed since export" in payload["error"]
    finally:
        server.close()


def test_a_daemon_about_to_be_stopped_takes_no_server(cache_dir):
    path = os.path.join(transport.socket_dir(), "gpu_use.sock")
    assert preload._hold_unused(path, wait_s=0.2) == (None, None)  # not serving
    server = transport._Server(path, {"key": 1}, {}, "t")
    try:
        conn, _ = _get(path)
        hold, why = preload._hold_unused(path, wait_s=0.2)
        assert hold is None and "1 other running server" in why
        conn.close()
        hold, why = preload._hold_unused(path, wait_s=5)  # once it has gone
        assert why is None
        # Held: a server that arrives before the stop is turned away...
        late = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        late.connect(path)
        transport._send(late, {"cmd": "get", "who": "late"})
        assert "shutting down" in transport._recv(late)["error"]
        late.close()
        # ...and so is another launch that wants to stop it.
        assert "being stopped" in preload._hold_unused(path, wait_s=0.2)[1]
        hold.close()  # let go without stopping it: it serves again
        _wait_for(lambda: not server.held, 5)
        conn, payload = _get(path)
        assert payload["key"] == 1
        conn.close()
    finally:
        server.close()


def test_a_hold_whose_holder_went_before_the_reply_is_let_go(cache_dir, monkeypatch):
    path = os.path.join(transport.socket_dir(), "gpu_gone.sock")
    server = transport._Server(path, {"key": 1}, {}, "t")
    send = transport._send

    def reply_fails(conn, obj):
        if isinstance(obj, dict) and obj.get("held"):
            raise BrokenPipeError("the holder is gone")
        send(conn, obj)

    monkeypatch.setattr(transport, "_send", reply_fails)
    try:
        holder = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        holder.connect(path)
        send(holder, {"cmd": "hold"})
        assert holder.recv(16) == b""  # the daemon gave the request up
        holder.close()
        assert not server.held
        conn, payload = _get(path)  # so it serves, and can be held again
        assert payload["key"] == 1
        conn.close()
    finally:
        server.close()


def test_daemon_lays_out_dp_ranks_as_the_engines_do():
    mgr = gpu_stack_import("atom.model_engine.engine_core_mgr")
    engine_config, layout_engines = mgr.engine_config, mgr.layout_engines

    pc = SimpleNamespace(
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_rank=0,
        data_parallel_rank_local=None,
        pipeline_parallel_rank=0,
        is_multinode_dp=False,
    )
    config = SimpleNamespace(
        parallel_config=pc,
        tensor_parallel_size=2,
        enable_dp_attention=True,
        pipeline_parallel_size=1,
        fake_eplb=False,
    )
    # DP attention: every TP rank is a DP rank of its own.
    assert layout_engines(config) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert (pc.data_parallel_size, config.tensor_parallel_size) == (4, 1)
    rank3 = engine_config(config, 3, 3).parallel_config
    assert (rank3.data_parallel_rank, rank3.data_parallel_rank_local) == (3, 3)
    assert pc.data_parallel_rank == 0  # a copy


# --------------------------------------------------------------------------
# The client's decisions (one rank; the daemon-facing calls replaced)
# --------------------------------------------------------------------------


_NO_PID = (1 << 22) + 1  # above any pid_max: a fake that can never be signalled


@pytest.fixture
def client():
    """The client module; importing it needs the GPU stack (aiter)."""
    return gpu_stack_import("atom.model_loader.weight_cache.client")


class _Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2), requires_grad=False)
        # Like attention/MoE layers: registered globally by name.
        config.compilation_config.static_forward_context["layers.0"] = self


def _config():
    ctx = {"other": 1}
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=ctx),
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
    )


def _from_cache(client, config=None):
    return client.load_from_cache(
        config or _config(), _Model, torch.device("cpu"), rank=0
    )


class _Conn:
    def recv(self, n):
        return b""

    def close(self):
        pass


@pytest.fixture
def loop(client, monkeypatch):
    """The client with the daemon replaced: ``probes`` lists what each probe
    finds, (daemon there, why it cannot be used)."""
    calls = SimpleNamespace(probes=[], spawned=0, stopped=0, built={})
    payload = {
        "structure": {},
        "export": {"plan": {}, "storages": {}},
        "process": {"changes": [], "env": {}, "storages": {}},
        "draft": {
            "structure": {},
            "export": {"plan": {}, "storages": {}},
            "process": {"changes": [], "env": {}, "storages": {}},
            "loaded": ["mtp.0.w"],
        },
    }

    def probe(me, started, held, own_key):
        present, reasons = calls.probes.pop(0)
        if not present:
            return None, None, [], client._NONE
        status = client._UNUSABLE if reasons else client._USABLE
        return (None if reasons else _Conn()), payload, reasons, status

    def spawn():
        calls.spawned += 1
        return {"pid": _NO_PID, "log": "daemon.log"}

    def stop(path):
        calls.stopped += 1

    for name, fn in {
        "_probe": probe,
        "_wait_together": lambda me, started: False,
        # The fake pids must never be signalled.
        "_pid_running": lambda pid, created=None: False,
        "_spawn_daemon": spawn,
        "stop_daemon": stop,
        "socket_path": lambda device: "/nonexistent/gpu.sock",
        "_started_by_launch": lambda: None,
        "launch_recorded": lambda: True,
        "_hold_unused": lambda path: (None, None),
        "exported_structure": lambda built, export: calls.built,
        "apply_export": lambda model, export, device: 0,
        "apply_process": lambda export, device: 0,
        "_leftover_meta": lambda model: [],
        "_run_prepare_finalize": lambda model: 0,
    }.items():
        monkeypatch.setattr(client, name, fn)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(client, "_draft", None)
    return calls


def test_starts_a_daemon_and_maps_it(loop, client):
    loop.probes = [(False, []), (True, [])]
    assert isinstance(_from_cache(client), _Model)
    assert loop.spawned == 1


def test_outside_the_api_server_it_loads_from_disk(loop, client, monkeypatch):
    # No saved launch to start daemons with: any daemon there is left alone.
    monkeypatch.setattr(client, "launch_recorded", lambda: False)
    loop.probes = [(True, ["env X: daemon=1 client=2"])]
    assert _from_cache(client) is None
    assert loop.stopped == loop.spawned == 0 and len(loop.probes) == 1


class _Drafter:
    def __init__(self, config):
        self.model = _Model(config)

    def attach_to_target(self, target, loaded):
        self.attached = (target, loaded)


def test_drafter_comes_from_the_daemon_the_model_came_from(loop, client, monkeypatch):
    monkeypatch.setattr(client, "construct_drafter", lambda c, d, r: _Drafter(c))
    cpu = torch.device("cpu")
    loop.probes = [(True, [])]
    config = _config()
    target = _from_cache(client, config)
    drafter = client.load_drafter_from_cache(config, cpu, None, target, rank=0)
    assert drafter.attached == (target, {"mtp.0.w"})
    # Taken once: a later drafter has nothing to map (a disk load follows).
    assert client.load_drafter_from_cache(config, cpu, None, target, rank=0) is None
    # One built differently loads from disk, after unregistering its meta build.
    loop.probes = [(True, [])]
    target = _from_cache(client, config)
    loop.built = {"mtp.0": "another layer"}
    del config.compilation_config.static_forward_context["layers.0"]
    assert client.load_drafter_from_cache(config, cpu, None, target, rank=0) is None
    assert config.compilation_config.static_forward_context == {"other": 1}


def test_drafter_modules_left_out_come_from_attaching(loop, client, monkeypatch):
    made = []
    monkeypatch.setattr(
        client,
        "construct_drafter",
        lambda c, d, r: made.append(_Drafter(c)) or made[-1],
    )
    # The daemon left the drafter's embedding out: on meta until attached.
    monkeypatch.setattr(
        client,
        "_leftover_meta",
        lambda m: (
            ["embed.weight"] if made and not hasattr(made[-1], "attached") else []
        ),
    )
    loop.probes = [(True, [])]
    config = _config()
    target = _from_cache(client, config)
    drafter = client.load_drafter_from_cache(
        config, torch.device("cpu"), None, target, rank=0
    )
    assert drafter is made[-1] and drafter.attached[0] is target


def test_a_daemon_that_vanished_is_not_no_daemon(client, tmp_path):
    # Its memory may still be in use: a disk load could run out of it.
    me = client._Rank("TP0", str(tmp_path / "gpu.sock"), None, None)
    assert client._probe(me, None, False, dict)[3] == client._NONE
    _, _, reasons, status = client._probe(me, None, True, dict)
    assert status == client._UNUSABLE and "exited before serving" in reasons[0]


def test_no_rank_stops_a_daemon_another_rank_must_keep(client, monkeypatch):
    monkeypatch.setattr(client, "daemon_alive", lambda path: False)
    hold = SimpleNamespace(closed=False)
    hold.close = lambda: setattr(hold, "closed", True)
    monkeypatch.setattr(
        client, "_hold_unused", lambda path: (hold, None)
    )  # fine here...
    monkeypatch.setattr(
        client, "group_agree", lambda ok, group: False
    )  # ...not elsewhere
    monkeypatch.setattr(
        client, "_stop_and_reclaim", lambda *a: pytest.fail("stopped before agreeing")
    )
    with pytest.raises(RuntimeError):
        client._stop_together(client._Rank("TP0", "/x.sock", None, None), [], False)
    assert hold.closed  # let go: the daemon it keeps serves again


def test_a_gpu_no_daemon_had_is_not_waited_for(client, monkeypatch):
    # A daemon of another layout: the GPUs it never held have nothing to free.
    monkeypatch.setattr(client, "stop_daemon", lambda path: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 2))
    t0 = time.monotonic()
    client._stop_and_reclaim("/x.sock", torch.device("cuda", 0), False)
    assert time.monotonic() - t0 < 1


def test_a_daemon_an_aborted_launch_left_goes_with_ones_own(client, monkeypatch):
    # It took the GPUs before this launch's own daemon could: not a failure.
    stopped = []
    monkeypatch.setattr(client, "_hold_unused", lambda path: (None, None))
    monkeypatch.setattr(client, "_stop_and_reclaim", lambda *a: stopped.append(a[0]))
    monkeypatch.setattr(client, "_pid_running", lambda pid, created=None: False)
    client._stop_together(
        client._Rank("TP0", "/x.sock", None, None),
        [],
        False,
        started={"pid": _NO_PID, "log": "l"},
    )
    assert stopped == ["/x.sock"]


def test_drafter_that_cannot_attach_loads_from_disk(loop, client, monkeypatch):
    class Broken(_Drafter):
        def attach_to_target(self, target, loaded):
            raise RuntimeError("no such module")

    monkeypatch.setattr(client, "construct_drafter", lambda c, d, r: Broken(c))
    loop.probes = [(True, [])]
    config = _config()
    target = _from_cache(client, config)
    del config.compilation_config.static_forward_context["layers.0"]
    cpu = torch.device("cpu")
    assert client.load_drafter_from_cache(config, cpu, None, target, rank=0) is None
    assert config.compilation_config.static_forward_context == {"other": 1}


def test_a_daemon_that_is_its_containers_init_is_not_stopped(monkeypatch):
    monkeypatch.setattr(preload, "daemon_alive", lambda path: True)
    monkeypatch.setattr(
        preload, "daemon_state", lambda path: {"pid": os.getpid(), "daemon_pid": 1}
    )
    monkeypatch.setattr(preload, "_ancestors", lambda pid: [1])
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("signalled PID 1"))
    with pytest.raises(RuntimeError, match="PID 1"):
        preload.stop_daemon("/x.sock")


def test_preload_takes_1_or_daemon_only(monkeypatch):
    for value, mode in (
        ("", "off"),
        ("0", "off"),
        ("1", "server"),
        ("daemon", "daemon"),
    ):
        monkeypatch.setenv("ATOM_PRELOAD", value)
        assert envs.ATOM_PRELOAD == mode
    monkeypatch.setenv("ATOM_PRELOAD", "deamon")
    with pytest.raises(ValueError, match="ATOM_PRELOAD"):
        _ = envs.ATOM_PRELOAD


def test_socket_dir_must_be_private(monkeypatch):
    d = tempfile.mkdtemp(dir="/tmp")
    try:
        os.chmod(d, 0o755)
        monkeypatch.setattr(transport, "_DIR", d)
        with pytest.raises(PermissionError):
            transport.socket_dir()
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o755  # left as it was
        os.chmod(d, 0o700)
        assert transport.socket_dir() == d
    finally:
        os.rmdir(d)


def test_restarts_a_daemon_it_cannot_use_once(loop, client):
    loop.probes = [(True, ["atom/model_loader/loader.py:load_model"]), (True, [])]
    assert isinstance(_from_cache(client), _Model)
    assert loop.stopped == 1 and loop.spawned == 1


def test_a_daemon_on_some_gpus_only_is_replaced(loop, client, monkeypatch):
    # Another rank of the launch finds no daemon on its GPU (another layout).
    lo = iter([client._NONE])
    monkeypatch.setattr(client, "group_min", lambda status, group: next(lo, status))
    loop.probes = [(True, []), (True, [])]
    assert isinstance(_from_cache(client), _Model)
    assert loop.stopped == 1 and loop.spawned == 1


def test_loads_from_disk_when_its_own_daemon_does_not_fit(loop, client, monkeypatch):
    monkeypatch.setattr(
        client, "_started_by_launch", lambda: {"pid": _NO_PID, "log": "d.log"}
    )
    loop.probes = [(True, [])]
    loop.built = {"layers.0": "another layer"}
    config = _config()
    assert _from_cache(client, config) is None  # a disk load follows
    assert loop.stopped == 1 and loop.spawned == 0
    # The meta-built model is unregistered, so the disk load can build it again.
    assert config.compilation_config.static_forward_context == {"other": 1}


def test_loads_from_disk_when_the_model_cannot_be_built_on_meta(
    loop, client, monkeypatch
):
    class Broken(nn.Module):
        def __init__(self, config):
            super().__init__()
            config.compilation_config.static_forward_context["layers.0"] = self
            raise RuntimeError("reads a parameter's value")

    monkeypatch.setattr(
        client, "_started_by_launch", lambda: {"pid": _NO_PID, "log": "d.log"}
    )
    loop.probes = [(True, [])]
    config = _config()
    assert client.load_from_cache(config, Broken, torch.device("cpu"), rank=0) is None
    assert config.compilation_config.static_forward_context == {"other": 1}


def test_loads_from_disk_when_no_daemon_can_start(loop, client, monkeypatch):
    def no_spawn():
        raise values._Miss("launcher exited with 1")

    monkeypatch.setattr(client, "_spawn_daemon", no_spawn)
    loop.probes = [(False, [])]
    assert _from_cache(client) is None


def test_leaves_a_daemon_other_servers_use(loop, client, monkeypatch):
    monkeypatch.setattr(
        client,
        "_hold_unused",
        lambda path: (None, "1 other running server(s) use it"),
    )
    loop.probes = [(True, ["atom/model_loader/loader.py:load_model"])]
    with pytest.raises(RuntimeError, match="other running server"):
        _from_cache(client)
    assert loop.stopped == loop.spawned == 0
