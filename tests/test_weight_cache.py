# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Weight cache logic that needs no daemon: code fingerprints, the before/after
load diff, export/import (CPU tensors travel by value), and the key filters."""

import os
import textwrap

import pytest
import torch
from torch import nn

import atom.model_loader.weight_cache as wc

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
    h = wc._SourceHasher()
    h.roots = {"pkg": str(root)}
    return h


def _write(path, text):
    path.write_text(textwrap.dedent(text))


@pytest.fixture
def no_source_stats(monkeypatch):
    monkeypatch.setattr(wc, "_SOURCE_STATS", None)


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


def test_source_edited_after_start_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "mod.py"
    _write(f, SRC)
    st = os.stat(f)
    monkeypatch.setattr(
        wc, "_SOURCE_STATS", {os.path.realpath(f): (st.st_mtime_ns, st.st_size)}
    )
    assert _hasher(tmp_path).digest("pkg", "mod.py", "load")
    _write(f, SRC + "\nEXTRA = 1\n")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(wc._SourceChanged):
        _hasher(tmp_path).digest("pkg", "mod.py", "load")


def test_outer_qualname():
    def outer():
        def inner():
            return 1

        return inner

    assert wc._outer_qualname(outer().__code__) == "test_outer_qualname"
    assert wc._outer_qualname((lambda: 0).__code__) == "test_outer_qualname"


# --------------------------------------------------------------------------
# Values, digests, key filters
# --------------------------------------------------------------------------


def module_level_function():
    return 1


def test_pack_unpack_roundtrip():
    ex = wc._Exporter()
    value = {
        "a": [torch.arange(3), 2, ("x", torch.ones(2))],
        "fn": module_level_function,
    }
    packed = wc._pack(value, ex)
    out = wc._unpack(packed, lambda meta: meta["value"])
    assert torch.equal(out["a"][0], torch.arange(3)) and out["a"][1] == 2
    assert isinstance(out["a"][2], tuple) and torch.equal(out["a"][2][1], torch.ones(2))
    assert out["fn"] is module_level_function
    with pytest.raises(wc._Unshippable):
        wc._pack(lambda: 0, ex)


def test_digest_sees_reorder():
    t = torch.arange(1000, dtype=torch.float32)
    u = t.clone()
    u[[3, 7]] = u[[7, 3]]
    assert wc._digest(t) == wc._digest(t.clone()) != wc._digest(u)


def test_norm_uses_compute_hash():
    class Q:
        def __init__(self, h):
            self.h = h

        def compute_hash(self):
            return self.h

    assert wc._norm(Q("a")) != wc._norm(Q("b"))


# --------------------------------------------------------------------------
# Before/after diff, export and import (CPU)
# --------------------------------------------------------------------------


class QuantState:
    def __init__(self):
        self.mode = "raw"
        self.scale = torch.ones(2)


class Toy(nn.Module):
    def __init__(self, big=0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        self.register_buffer("rope", torch.arange(8.0))
        self.register_buffer("scratch", torch.zeros(big or 3))
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
        wc, "_DIFF_PACKAGES", wc._DIFF_PACKAGES + (QuantState.__module__,)
    )


def _export(model, load):
    built, before = wc.structure(model), wc.snapshot(model)
    load(model)
    return built, wc.build_export(model, before, wc.snapshot(model))


def test_export_import_reproduces_the_load(toy_packages):
    daemon = Toy()
    built, export = _export(daemon, Toy.fake_load)
    assert export["unshippable"] == []
    with wc.params_on_meta():
        client = Toy()
    assert wc._diff_structure(built, wc.structure(client)) == []
    wc.apply_export(client, export, torch.device("cpu"))
    assert wc._leftover_meta(client) == []
    for name in ("w", "w_shuffled", "scratch", "absorbed", "rope"):
        assert torch.equal(getattr(client, name), getattr(daemon, name)), name
    assert (
        client.flag
        and client.q.mode == "shuffled"
        and torch.equal(client.q.scale, daemon.q.scale)
    )
    assert torch.equal(client.lst[0], torch.ones(2)) and client.lst[1] == 3
    assert "bias" not in client.child._parameters


def test_large_in_place_buffer_write_is_shipped(toy_packages):
    daemon = Toy(big=20 << 20)  # 80 MiB of float32: no size cutoff any more
    _, export = _export(daemon, Toy.fake_load)
    assert "scratch" in export["plan"][""]["buffers"]


def test_unshareable_state_is_reported(toy_packages):
    class Kernel:
        pass

    def load(m):
        m.kernel = Kernel()
        del m.child

    _, export = _export(Toy(), load)
    assert any("kernel" in u for u in export["unshippable"])
    assert any("child: submodule removed" in u for u in export["unshippable"])


def test_module_tree_change_is_seen():
    built = wc.structure(Toy())
    with wc.params_on_meta():
        client = Toy()
    client.extra = nn.Linear(2, 2)  # a constructor now builds another submodule
    assert wc._diff_structure(built, wc.structure(client))


# --------------------------------------------------------------------------
# Input recorder: what loading reads becomes the key
# --------------------------------------------------------------------------

import atom.model_loader.input_recorder as ir


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


def _record_load(model, load):
    rec = ir.InputRecorder()
    rec.enter_load(model)
    try:
        load(model)
    finally:
        rec.finish_load(set())
    return rec.key_reads()


def test_recorder_keeps_what_loading_reads(toy_packages, monkeypatch):
    monkeypatch.setattr(wc, "_DIFF_PACKAGES", wc._DIFF_PACKAGES + (Layer.__module__,))
    reads = _record_load(Layer(), Layer.process)
    assert set(reads["attrs"]) == {"\tinterleave", "\t[buffer]index"}  # not max_len
    assert list(reads["classattrs"]) == [
        f"{Layer.__module__}:{Layer.__qualname__}\tLAYOUT"
    ]
    # A client differing only where loading did not look still matches...
    assert ir.diff_attr_reads(reads, Layer(max_len=8192)) == []
    # ...but not where it did.
    assert ir.diff_attr_reads(reads, Layer(interleave=False))


def test_attribute_written_then_read_keys_its_construction_value(monkeypatch):
    # V4's post-processing sets wo_a.quant_type to No, then reads it: the key
    # must hold the value a client constructs, not the one loading wrote.
    monkeypatch.setattr(wc, "_DIFF_PACKAGES", wc._DIFF_PACKAGES + (Layer.__module__,))

    def load(m):
        m.interleave = False
        if not m.interleave:
            m.w.data.zero_()

    reads = _record_load(Layer(), load)
    assert reads["attrs"]["\tinterleave"] == ("V", "true")
    assert ir.diff_attr_reads(reads, Layer()) == []


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
    assert ir.diff_static_reads(reads, None) == [
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

    with wc.params_on_meta():
        m = M()
    assert m.p.is_meta and m.p is m.held and m.p.loader == "set after registration"


# --- ATOM_PRELOAD daemon management ------------------------------------------

import json
import signal
import socket
import stat
import subprocess
import sys
import threading
import time


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "wc"
    monkeypatch.setenv("ATOM_WEIGHT_CACHE_DIR", str(d))
    for k in (
        "ATOM_WEIGHT_CACHE",
        "ATOM_PRELOAD",
        wc._LAUNCH_ENV,
        wc._DAEMON_PID_ENV,
        wc._DAEMON_LOG_ENV,
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
        wc._pid_running(os.getpid())
        and not wc._pid_running(0)
        and not wc._pid_running(None)
    )
    assert wc._ancestors(os.getpid())[0] == os.getppid()


def test_record_launch_saves_the_environment_from_exec(cache_dir, monkeypatch):
    monkeypatch.setenv("ATOM_PRELOAD", "1")
    monkeypatch.setenv("WC_TEST_SET_AT_RUNTIME", "1")  # like an import setting a var
    wc.record_launch()
    path = os.environ[wc._LAUNCH_ENV]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600  # holds the environment
    with open(path) as f:
        launch = json.load(f)
    assert launch["argv"][0] == sys.executable and launch["cwd"] == os.getcwd()
    assert "PATH" in launch["env"] and "WC_TEST_SET_AT_RUNTIME" not in launch["env"]


def test_record_launch_in_a_daemon_names_its_main_process(cache_dir, monkeypatch):
    monkeypatch.setenv("ATOM_WEIGHT_CACHE", "daemon")
    wc.record_launch()
    assert os.environ[wc._DAEMON_PID_ENV] == str(os.getpid())


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
    monkeypatch.setenv(wc._LAUNCH_ENV, str(launch))
    first = wc._spawn_daemon()
    try:
        assert wc._spawn_daemon() == first  # the other ranks get the same daemon
        _wait_for(lambda: out.exists() and out.read_text())
        assert out.read_text() == "daemon -"  # daemon mode, ATOM_PRELOAD dropped
        assert wc._pid_running(first["pid"])
        assert os.getpid() not in wc._ancestors(first["pid"])  # survives a tree kill
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
import atom.model_loader.weight_cache as wc
lock = wc._DaemonLock(sys.argv[1], rank=0, model="toy-model", idle_timeout_s=0)
def stop(*_):
    lock.release()
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
    path = os.path.join(wc.socket_dir(), "gpu_test.sock")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    main = subprocess.Popen(
        [sys.executable, str(tmp_path / "main.py"), str(tmp_path / "runner.py"), path],
        env=env,
    )
    try:
        _wait_for(
            lambda: wc.daemon_alive(path) and (wc.daemon_state(path) or {}).get("pid")
        )
        st = wc.daemon_state(path)
        assert (
            st["daemon_pid"] == main.pid
            and st["state"] == "loading"
            and st["model"] == "toy-model"
        )
        assert wc.main(["status"]) == 0
        shown = capsys.readouterr().out
        assert (
            f"daemon pid {main.pid}: toy-model" in shown and "rank 0: loading" in shown
        )
        assert wc.main(["stop"]) == 0
        assert f"stopped daemon pid {main.pid}" in capsys.readouterr().out
        assert not wc.daemon_alive(path) and main.wait(10) == 0
        assert wc.main(["status"]) == 0
        assert "no weight cache daemon running" in capsys.readouterr().out
    finally:
        if main.poll() is None:
            main.kill()


def test_idle_timeout_counts_connected_servers(cache_dir):
    path = os.path.join(wc.socket_dir(), "gpu_idle.sock")
    idle = threading.Event()
    server = wc._Server(
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
        wc._send(conn, {"cmd": "get", "who": "test"})
        assert wc._recv(conn)["summary"]["pid"] == 1
        assert not idle.wait(1.6)  # a connected server keeps the daemon
        assert wc._ping(path)["clients"] == 1
        conn.close()
        assert idle.wait(3)
        assert wc._ping(path)["clients"] == 0
    finally:
        server.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_zero_byte_cuda_tensor_is_rebuilt_locally():
    meta = wc._Exporter().tensor(torch.empty(0, 5, device="cuda"))
    assert meta.get("empty")
    t = torch.empty_strided(
        meta["shape"], meta["stride"], dtype=meta["dtype"], device="cuda"
    )
    assert t.shape == (0, 5)
