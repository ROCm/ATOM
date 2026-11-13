# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Weight cache logic that needs no daemon: code fingerprints, the before/after
load diff, export/import (CPU tensors travel by value), the key filters, and
daemon management with fake processes."""

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
from types import SimpleNamespace

import psutil
import pytest
import torch
from torch import nn

import atom.model_loader.weight_cache.__main__ as cli
from atom.model_loader.weight_cache import (
    client,
    export,
    key,
    preload,
    transport,
    values,
)
from atom.model_loader.weight_cache import recorder as ir

# --------------------------------------------------------------------------
# Code fingerprint
# --------------------------------------------------------------------------

SRC = '''
"""Module docstring."""
SCALE = 16


def load(x):
    """Docstring."""
    return x * SCALE


def forward(x):
    return x + 1


class Q:
    k = 1

    @property
    def p(self):
        return 1

    @p.setter
    def p(self, v):
        self._p = v
'''


def _hasher(root):
    h = key._SourceHasher()
    h.roots = {"pkg": str(root)}
    return h


def _write(path, text):
    path.write_text(textwrap.dedent(text))


@pytest.fixture
def no_source_stats(monkeypatch):
    # The launch "starts" after every file the test writes.
    monkeypatch.setenv(key._LAUNCH_T0_ENV, str(time.time() + 3600))


def test_function_hash_follows_its_body_only(tmp_path, no_source_stats):
    f = tmp_path / "mod.py"
    _write(f, SRC)
    load0 = _hasher(tmp_path).digest("pkg", "mod.py", "load")

    _write(f, SRC.replace("return x + 1", "return x + 2"))  # forward-only edit
    assert _hasher(tmp_path).digest("pkg", "mod.py", "load") == load0

    _write(f, "# comment\n\n" + SRC.replace('"""Docstring."""', '"""Other words."""'))
    assert (
        _hasher(tmp_path).digest("pkg", "mod.py", "load") == load0
    )  # layout/docs only

    _write(f, SRC.replace("return x * SCALE", "return x * SCALE * 2"))
    assert _hasher(tmp_path).digest("pkg", "mod.py", "load") != load0


def test_all_same_named_defs_are_hashed(tmp_path, no_source_stats):
    f = tmp_path / "mod.py"
    _write(f, SRC)
    p0 = _hasher(tmp_path).digest("pkg", "mod.py", "Q.p")
    _write(f, SRC.replace("self._p = v", "self._p = -v"))  # the setter, not the getter
    assert _hasher(tmp_path).digest("pkg", "mod.py", "Q.p") != p0


def test_module_and_class_bodies_leave_function_bodies_out(tmp_path, no_source_stats):
    # A module or class body that ran keys what it defines, not the bodies of
    # its functions (those that ran are keyed on their own).
    f = tmp_path / "mod.py"
    _write(f, SRC)
    h = _hasher(tmp_path)
    mod0, q0 = h.digest("pkg", "mod.py", "<module>"), h.digest("pkg", "mod.py", "Q")
    _write(f, SRC.replace("x + 1", "x + 2").replace("self._p = v", "self._p = -v"))
    h = _hasher(tmp_path)
    assert h.digest("pkg", "mod.py", "<module>") == mod0
    assert h.digest("pkg", "mod.py", "Q") == q0
    _write(f, SRC.replace("SCALE = 16", "SCALE = 32").replace("k = 1", "k = 2"))
    h = _hasher(tmp_path)
    assert h.digest("pkg", "mod.py", "<module>") != mod0
    assert h.digest("pkg", "mod.py", "Q") != q0


def test_source_edited_after_start_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "mod.py"
    _write(f, SRC)
    time.sleep(0.05)
    monkeypatch.setenv(key._LAUNCH_T0_ENV, str(time.time()))
    assert _hasher(tmp_path).digest("pkg", "mod.py", "load")
    time.sleep(0.05)
    _write(f, SRC + "\nEXTRA = 1\n")
    os.utime(f, (0, 0))  # an old mtime does not hide it: ctime moved
    with pytest.raises(key._SourceChanged):
        _hasher(tmp_path).digest("pkg", "mod.py", "load")


def test_same_file_bytes_skip_the_ast(tmp_path, no_source_stats, monkeypatch):
    f = tmp_path / "mod.py"
    _write(f, SRC)
    monkeypatch.setattr(key, "_package_roots", lambda: {"pkg": str(tmp_path)})
    h = key._SourceHasher()
    fp = {
        "entries": [("fn", "pkg", "mod.py", "load", h.digest("pkg", "mod.py", "load"))],
        "files": {"pkg/mod.py": h.file_hash("pkg", "mod.py")},
    }
    parsed = []
    tree = key._SourceHasher._tree
    monkeypatch.setattr(
        key._SourceHasher,
        "_tree",
        lambda self, pkg, rel: parsed.append(rel) or tree(self, pkg, rel),
    )
    assert key.verify_code(fp) == [] and parsed == []  # same bytes: not parsed
    _write(f, SRC.replace("return x + 1", "return x + 2"))  # forward-only edit
    assert key.verify_code(fp) == [] and parsed == ["mod.py"]
    _write(f, SRC.replace("return x * SCALE", "return x * SCALE * 2"))
    assert key.verify_code(fp) == ["pkg/mod.py:load"]


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


def test_outer_qualname():
    def outer():
        def inner():
            return 1

        return inner

    assert key._outer_qualname(outer().__code__) == "test_outer_qualname"
    assert key._outer_qualname((lambda: 0).__code__) == "test_outer_qualname"


# --------------------------------------------------------------------------
# Values, digests, key filters
# --------------------------------------------------------------------------


def module_level_function():
    return 1


def test_pack_unpack_roundtrip():
    ex = export._Exporter()
    value = {
        "a": [torch.arange(3), 2, ("x", torch.ones(2))],
        "fn": module_level_function,
    }
    packed = export._pack(value, ex)
    out = export._unpack(packed, lambda meta: meta["value"])
    assert torch.equal(out["a"][0], torch.arange(3)) and out["a"][1] == 2
    assert isinstance(out["a"][2], tuple) and torch.equal(out["a"][2][1], torch.ones(2))
    assert out["fn"] is module_level_function
    with pytest.raises(values._Unshippable):
        export._pack(lambda: 0, ex)


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
    mod.Settings, mod.FLAG, mod.TABLE = Settings, False, {"a": 1}
    mod.holder = Holder()
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.delenv("ATOM_TEST_LOAD_SET", raising=False)
    before = export.process_snapshot()
    mod.FLAG, Settings.LAYOUT = True, 16  # what loading decided
    mod.TABLE["b"] = 2
    os.environ["ATOM_TEST_LOAD_SET"] = "1"
    mod.kernel = lambda: 0  # cannot be shared
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
    assert any(u.startswith(f"{mod.__name__}:holder.mode") for u in exp["unshippable"])
    # A client starts from the state before loading.
    mod.FLAG, mod.TABLE, Settings.LAYOUT = False, {"a": 1}, 1
    del os.environ["ATOM_TEST_LOAD_SET"]
    exp["storages"] = export.share_storages(exp.pop("views"))
    export.apply_process(exp, torch.device("cpu"))
    assert mod.FLAG and mod.TABLE == {"a": 1, "b": 2} and Settings.LAYOUT == 16
    assert os.environ["ATOM_TEST_LOAD_SET"] == "1"


def test_module_tree_change_is_seen():
    built = key.structure(Toy())
    with export.params_on_meta():
        client = Toy()
    client.extra = nn.Linear(2, 2)  # a constructor now builds another submodule
    assert key._diff_key(built, key.structure(client))


# --------------------------------------------------------------------------
# Input recorder: what loading reads becomes the key
# --------------------------------------------------------------------------


class Layer(nn.Module):
    LAYOUT = 16  # a class-level layout constant post-processing reads

    def __init__(self, interleave=True, max_len=1024):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.interleave = interleave  # construction setting loading reads
        self.max_len = max_len  # construction setting loading ignores
        self.register_buffer("index", torch.arange(4))

    def process(self):
        scale = self.LAYOUT if self.interleave else 1
        self.w.data.copy_(self.index.float() * scale)


@pytest.fixture
def layer_packages(monkeypatch):
    monkeypatch.setattr(
        values, "_DIFF_PACKAGES", values._DIFF_PACKAGES + (Layer.__module__,)
    )


def _record_load(model, load):
    rec = ir.InputRecorder()
    rec.enter_load(model)
    try:
        load(model)
    finally:
        rec.finish_load(set())
    return rec.key_reads()


def test_recorder_keeps_what_loading_reads(layer_packages):
    reads = _record_load(Layer(), Layer.process)
    assert set(reads["attrs"]) == {"\tinterleave", "\t[buffer]index"}  # not max_len
    assert list(reads["classattrs"]) == [
        f"{Layer.__module__}:{Layer.__qualname__}\tLAYOUT"
    ]
    # A client differing only where loading did not look still matches...
    assert ir.diff_reads(reads, ("attrs",), model=Layer(max_len=8192)) == []
    # ...but not where it did.
    assert ir.diff_reads(reads, ("attrs",), model=Layer(interleave=False))


def test_attribute_written_then_read_keys_its_construction_value(layer_packages):
    # V4's post-processing sets wo_a.quant_type to No, then reads it: the key
    # must hold the value a client constructs, not the one loading wrote.
    def load(m):
        m.interleave = False
        if not m.interleave:
            m.w.data.zero_()

    reads = _record_load(Layer(), load)
    assert reads["attrs"]["\tinterleave"] == ("V", "true")
    assert ir.diff_reads(reads, ("attrs",), model=Layer()) == []


def test_config_read_only_while_constructing_is_not_keyed():
    class Cfg:
        def __init__(self):
            self.max_len, self.hf_config = 1, None

    cfg = Cfg()
    rec = ir.InputRecorder()
    rec.enter_construct(cfg)
    try:
        assert cfg.max_len == 1  # constructing reads it...
        cfg.max_len = 2  # ...and rewrites it; loading never reads it
        rec.enter_load(nn.Module())
    finally:
        rec.finish_load(set())
    assert rec.key_reads()["config"] == {}


def test_recorder_keys_env_read_during_load_only(monkeypatch):
    monkeypatch.setattr(ir, "KEY_ENV_PREFIXES", ir.KEY_ENV_PREFIXES + ("tests/",))
    monkeypatch.setenv("ATOM_TEST_LOAD_FLAG", "1")
    monkeypatch.setenv("ATOM_TEST_SERVE_FLAG", "1")
    rec = ir.InputRecorder.start()
    try:
        os.environ.get(
            "ATOM_TEST_SERVE_FLAG"
        )  # e.g. a constructor picking a run-time kernel
        rec.enter_load(nn.Module())
        os.environ.get("ATOM_TEST_LOAD_FLAG")
        os.getenv("ATOM_TEST_UNSET_FLAG")  # unset counts too: setting it later matters
    finally:
        rec.finish_load(set())
    env = rec.key_reads()["env"]
    assert env == {"ATOM_TEST_LOAD_FLAG": "1", "ATOM_TEST_UNSET_FLAG": None}
    assert os._Environ.__getitem__ is not ir._env_getitem  # unhooked
    monkeypatch.setenv("ATOM_TEST_UNSET_FLAG", "x")
    reads = {"env": env, "config": {}, "classattrs": {}, "globals": {}, "files": {}}
    assert ir.diff_reads(reads, ir.STATIC_READS) == [
        "env ATOM_TEST_UNSET_FLAG: daemon=None client=x"
    ]


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
    from atom.model_loader.weight_cache import daemon

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
    from atom.model_loader.weight_cache import daemon

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
    load = SimpleNamespace(drafter=d, model=d.model, loaded=set())
    assert daemon._leave_out_shared(load, target, exp) == []
    assert set(exp["plan"]) == {""}
    assert list(exp["views"]) == [d.model.w.untyped_storage().data_ptr()]


# --------------------------------------------------------------------------
# ATOM_PRELOAD daemon management
# --------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "wc"
    monkeypatch.setenv("ATOM_WEIGHT_CACHE_DIR", str(d))
    for k in (
        "ATOM_WEIGHT_CACHE",
        "ATOM_PRELOAD",
        preload._LAUNCH_ENV,
        key._LAUNCH_T0_ENV,
        transport._DAEMON_PID_ENV,
        transport._DAEMON_LOG_ENV,
    ):
        monkeypatch.delenv(k, raising=False)
    return d


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
    path = os.environ[preload._LAUNCH_ENV]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600  # holds the environment
    with open(path) as f:
        launch = json.load(f)
    assert launch["argv"][0] == sys.executable and launch["cwd"] == os.getcwd()
    assert "PATH" in launch["env"] and "WC_TEST_SET_AT_RUNTIME" not in launch["env"]


def test_record_launch_in_a_daemon_names_its_main_process(cache_dir, monkeypatch):
    monkeypatch.setenv("ATOM_WEIGHT_CACHE", "daemon")
    before = time.time()
    preload.record_launch()
    assert os.environ[transport._DAEMON_PID_ENV] == str(os.getpid())
    # The launch time is taken now, not from the process start (psutil's can
    # be a second early, so a file edited just before the launch would count
    # as edited after it).
    assert before <= float(os.environ[key._LAUNCH_T0_ENV]) <= time.time()


def test_process_start_is_to_the_tick():
    start = key._process_start()
    assert start <= time.time()
    assert abs(start - psutil.Process().create_time()) < 1.5


def test_spawned_daemon_is_detached_and_started_once(cache_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    code = (
        "import os, time\n"
        "open(os.environ['WC_OUT'], 'w').write(os.environ['ATOM_WEIGHT_CACHE'] + ' ' + os.environ.get('ATOM_PRELOAD', '-'))\n"
        "time.sleep(60)\n"
    )
    os.makedirs(cache_dir, mode=0o700)
    launch = cache_dir / "launch_1_1.json"
    env = {**os.environ, "WC_OUT": str(out), "ATOM_PRELOAD": "1"}
    launch.write_text(
        json.dumps(
            {"argv": [sys.executable, "-c", code], "cwd": str(tmp_path), "env": env}
        )
    )
    monkeypatch.setenv(preload._LAUNCH_ENV, str(launch))
    first = preload._spawn_daemon()
    try:
        assert preload._spawn_daemon() == first  # the other ranks get the same daemon
        _wait_for(lambda: out.exists() and out.read_text())
        assert out.read_text() == "daemon -"  # daemon mode, ATOM_PRELOAD dropped
        assert preload._pid_running(first["pid"])
        assert os.getpid() not in preload._ancestors(
            first["pid"]
        )  # survives a tree kill
        assert os.getsid(first["pid"]) != os.getsid(0)  # and a Ctrl-C to our group
    finally:
        os.kill(first["pid"], signal.SIGKILL)


_FAKE_MAIN = """
import os, signal, subprocess, sys, time
os.environ["ATOM_WEIGHT_CACHE_DAEMON_PID"] = str(os.getpid())
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
import signal, sys, time
from atom.model_loader.weight_cache.transport import _DaemonLock
lock = _DaemonLock(sys.argv[1], rank=0, model="toy-model", idle_timeout_s=0)
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
        assert (
            f"daemon pid {main.pid}: toy-model" in shown and "rank 0: loading" in shown
        )
        assert cli.main(["stop"]) == 0
        assert f"stopped daemon pid {main.pid}" in capsys.readouterr().out
        # Returns once the runner has exited, not when it let go of the lock.
        assert not transport.daemon_alive(path) and not preload._pid_running(st["pid"])
        assert main.wait(10) == 0
        assert cli.main(["status"]) == 0
        assert "no weight cache daemon running" in capsys.readouterr().out
    finally:
        if main.poll() is None:
            main.kill()


def test_idle_timeout_counts_connected_servers(cache_dir):
    path = os.path.join(transport.socket_dir(), "gpu_idle.sock")
    idle = threading.Event()
    server = transport._Server(
        path,
        {"summary": {"pid": 1, "bytes": 0}},
        {},
        "t",
        idle_timeout_s=1,
        on_idle=idle.set,
    )
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(path)
        transport._send(conn, {"cmd": "get", "who": "test"})
        assert transport._recv(conn)["summary"]["pid"] == 1
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
    server = transport._Server(
        path, {"summary": {"pid": 1, "bytes": 64}}, {1: weights}, "t"
    )
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


def test_in_use_by_others(cache_dir):
    path = os.path.join(transport.socket_dir(), "gpu_use.sock")
    assert preload._in_use_by_others(path, wait_s=0.2) is None  # not serving
    server = transport._Server(path, {"summary": {"pid": 1, "bytes": 0}}, {}, "t")
    try:
        conn, _ = _get(path)
        assert "1 other running server" in preload._in_use_by_others(path, wait_s=0.2)
        conn.close()
        _wait_for(lambda: preload._in_use_by_others(path, wait_s=0.2) is None, 5)
    finally:
        server.close()


def test_start_early_only_without_any_daemon(cache_dir, monkeypatch):
    os.makedirs(cache_dir, mode=0o700)
    launch = cache_dir / "launch_1_1.json"
    launch.write_text("{}")
    monkeypatch.setenv(preload._LAUNCH_ENV, str(launch))
    calls = []

    def spawn():
        calls.append(1)
        info = {"pid": 4242, "log": "x.log"}
        (cache_dir / "launch_1_1.json.daemon").write_text(json.dumps(info))
        return info

    monkeypatch.setattr(preload, "_spawn_daemon", spawn)
    lock = transport._DaemonLock(os.path.join(str(cache_dir), "gpu_held.sock"))
    preload.start_early()  # a daemon holds a GPU: the runners decide
    assert calls == [] and preload._started_by_launch() is None
    lock.release()
    preload.start_early()
    assert calls == [1] and preload._started_by_launch()["pid"] == 4242


def test_daemon_lays_out_dp_ranks_as_the_engines_do():
    from atom.model_engine.engine_core_mgr import engine_config, layout_engines

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


def _from_daemon(config=None):
    return client._from_daemon(config or _config(), _Model, torch.device("cpu"), 0)


class _Conn:
    def recv(self, n):
        return b""

    def close(self):
        pass


@pytest.fixture
def loop(monkeypatch):
    """The client with the daemon replaced: ``probes`` lists what each probe
    finds, (daemon there, why it cannot be used)."""
    calls = SimpleNamespace(probes=[], spawned=0, stopped=0, attr_diffs=[])
    payload = {
        "summary": {"pid": 7, "daemon_pid": 7},
        "structure": {},
        "export": {"plan": {}, "storages": {}},
        "process": {"changes": [], "env": {}, "storages": {}},
        "draft": {
            "structure": {},
            "export": {"plan": {}, "storages": {}},
            "process": {"changes": [], "env": {}, "storages": {}},
            "reads": {},
            "code": {},
            "loaded": ["mtp.0.w"],
        },
        "reads": {},
    }

    def probe(path, label, started, held, own_key):
        present, reasons = calls.probes.pop(0)
        if not present:
            return None, None, [], client._NONE
        status = client._UNUSABLE if reasons else client._USABLE
        return (None if reasons else _Conn()), payload, reasons, status

    def spawn():
        calls.spawned += 1
        return {"pid": 1, "log": "daemon.log"}

    def stop(path):
        calls.stopped += 1

    for name, fn in {
        "_probe": probe,
        "_wait_together": lambda path, label, started, group: None,
        "_pid_running": lambda pid: False,  # the fake pids must never be signalled
        "verify_code": lambda code: [],
        "_spawn_daemon": spawn,
        "stop_daemon": stop,
        "socket_path": lambda device: "/nonexistent/gpu.sock",
        "_started_by_launch": lambda: None,
        "_in_use_by_others": lambda path: None,
        "exported_structure": lambda built, export: {},
        "diff_reads": lambda reads, kinds, **kw: list(calls.attr_diffs),
        "apply_export": lambda model, export, device: 0,
        "apply_process": lambda export, device: 0,
        "_leftover_meta": lambda model: [],
        "_run_prepare_finalize": lambda model: 0,
    }.items():
        monkeypatch.setattr(client, name, fn)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(client, "_draft", None)
    monkeypatch.delenv("ATOM_PRELOAD", raising=False)
    monkeypatch.delenv("ATOM_WEIGHT_CACHE", raising=False)
    return calls


def test_client_mode_without_a_daemon_loads_from_disk(loop, monkeypatch):
    monkeypatch.setattr(client, "load_from_disk", lambda *a: "from disk")
    loop.probes = [(False, [])]
    loaded = client.load_from_cache(
        _config(), _Model, torch.device("cpu"), rank=0, tag="TP0"
    )
    assert loaded == "from disk"


def test_client_mode_refuses_a_daemon_it_cannot_use(loop):
    loop.probes = [(True, ["env X: daemon=1 client=2"])]
    with pytest.raises(RuntimeError, match="env X"):
        _from_daemon()
    assert loop.stopped == loop.spawned == 0


def test_preload_starts_a_daemon_and_maps_it(loop, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    loop.probes = [(False, []), (True, [])]
    assert isinstance(_from_daemon(), _Model)
    assert loop.spawned == 1


class _Drafter:
    def __init__(self, config):
        self.model = _Model(config)

    def attach_to_target(self, target, loaded):
        self.attached = (target, loaded)


def test_drafter_comes_from_the_daemon_the_model_came_from(loop, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    monkeypatch.setattr(client, "construct_drafter", lambda c, d, r: _Drafter(c))
    monkeypatch.setattr(client, "load_drafter_from_disk", lambda *a: "from disk")
    cpu = torch.device("cpu")
    loop.probes = [(True, [])]
    config = _config()
    target = _from_daemon(config)
    drafter = client.load_drafter_from_cache(config, cpu, None, target, rank=0)
    assert drafter.attached == (target, {"mtp.0.w"})
    # Taken once: a later drafter has nothing to map.
    assert (
        client.load_drafter_from_cache(config, cpu, None, target, rank=0) == "from disk"
    )
    # One built differently loads from disk, after unregistering its meta build.
    loop.probes = [(True, [])]
    target = _from_daemon(config)
    loop.attr_diffs = ["mtp.0.interleave: daemon=True client=False"]
    del config.compilation_config.static_forward_context["layers.0"]
    assert (
        client.load_drafter_from_cache(config, cpu, None, target, rank=0) == "from disk"
    )
    assert config.compilation_config.static_forward_context == {"other": 1}


def test_drafter_modules_left_out_come_from_attaching(loop, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    made = []
    monkeypatch.setattr(
        client,
        "construct_drafter",
        lambda c, d, r: made.append(_Drafter(c)) or made[-1],
    )
    monkeypatch.setattr(client, "load_drafter_from_disk", lambda *a: "from disk")
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
    target = _from_daemon(config)
    drafter = client.load_drafter_from_cache(
        config, torch.device("cpu"), None, target, rank=0
    )
    assert drafter is made[-1] and drafter.attached[0] is target


def test_a_server_whose_sources_changed_leaves_the_daemon_alone(loop, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    stale = ["key.py changed after this server started; restart it"]
    monkeypatch.setattr(client, "_probe", lambda *a: (None, None, stale, client._STALE))
    with pytest.raises(RuntimeError, match="restart it"):
        _from_daemon()
    assert loop.stopped == loop.spawned == 0


def test_a_daemon_that_vanished_is_not_no_daemon(tmp_path):
    # Its memory may still be in use: a disk load could run out of it.
    path = str(tmp_path / "gpu.sock")
    assert client._probe(path, "TP0", None, False, dict)[3] == client._NONE
    _, _, reasons, status = client._probe(path, "TP0", None, True, dict)
    assert status == client._UNUSABLE and "exited before serving" in reasons[0]


def test_give_up_leaves_another_launchs_daemon(monkeypatch):
    monkeypatch.setattr(client, "daemon_alive", lambda path: True)
    monkeypatch.setattr(client, "daemon_state", lambda path: {"daemon_pid": 99})
    monkeypatch.setattr(
        client, "_stop_and_reclaim", lambda *a: pytest.fail("stopped another's daemon")
    )
    with pytest.raises(RuntimeError, match="another launch's daemon"):
        client._give_up(
            "TP0", "/x.sock", [], {"pid": 7, "log": "l"}, None, None, None, None
        )


def test_socket_dir_must_be_private(monkeypatch):
    d = tempfile.mkdtemp(dir="/tmp")
    try:
        os.chmod(d, 0o755)
        monkeypatch.setenv("ATOM_WEIGHT_CACHE_DIR", d)
        with pytest.raises(PermissionError):
            transport.socket_dir()
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o755  # left as it was
        os.chmod(d, 0o700)
        assert transport.socket_dir() == d
    finally:
        os.rmdir(d)


def test_preload_restarts_a_daemon_it_cannot_use_once(loop, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    loop.probes = [(True, ["atom/model_loader/loader.py:load_model"]), (True, [])]
    assert isinstance(_from_daemon(), _Model)
    assert loop.stopped == 1 and loop.spawned == 1


def test_preload_loads_from_disk_when_its_own_daemon_does_not_fit(loop, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    monkeypatch.setattr(
        client, "_started_by_launch", lambda: {"pid": 1, "log": "d.log"}
    )
    loop.probes = [(True, [])]
    loop.attr_diffs = ["layers.0.interleave: daemon=True client=False"]
    config = _config()
    assert _from_daemon(config) is None  # a disk load follows
    assert loop.stopped == 1 and loop.spawned == 0
    # The meta-built model is unregistered, so the disk load can build it again.
    assert config.compilation_config.static_forward_context == {"other": 1}


def test_preload_loads_from_disk_when_no_daemon_can_start(loop, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")

    def no_spawn():
        raise values._Miss("this engine was not started by the API server")

    monkeypatch.setattr(client, "_spawn_daemon", no_spawn)
    loop.probes = [(False, [])]
    assert _from_daemon() is None


def test_preload_leaves_a_daemon_other_servers_use(loop, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    monkeypatch.setattr(
        client, "_in_use_by_others", lambda path: "1 other running server(s) use it"
    )
    loop.probes = [(True, ["atom/model_loader/loader.py:load_model"])]
    with pytest.raises(RuntimeError, match="other running server"):
        _from_daemon()
    assert loop.stopped == loop.spawned == 0
