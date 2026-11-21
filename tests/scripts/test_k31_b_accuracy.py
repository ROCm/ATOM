import base64
import hashlib
import importlib.util
import io
import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts"))
spec = importlib.util.spec_from_file_location(
    "k31_b_accuracy", ROOT / ".github/scripts/k31_b_accuracy.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
CID = "a" * 64
ACTUAL_ALLOCATION = (
    "--group-add 992 --device /dev/dri/renderD128 --device /dev/dri/renderD136 "
    "--device /dev/dri/renderD144 --device /dev/dri/renderD152 "
    "--device /dev/dri/renderD160 --device /dev/dri/renderD168 "
    "--device /dev/dri/renderD176 --device /dev/dri/renderD184 --cpuset-cpus 0-127\n"
)


@pytest.fixture(autouse=True)
def permitted_cpu_observation(monkeypatch):
    monkeypatch.setattr(
        ci.os, "sched_getaffinity", lambda pid: set(range(256)), raising=False
    )


def write(path, value):
    path.write_text(json.dumps(value))


def old_allocation(revision="d0fa6a7fb5e9b4da3f559fab153aea0567fd7e1f"):
    path = ROOT / ".github/scripts/k31_b_accuracy.py"
    namespace = {"__file__": str(path), "__name__": "before_allocation_repair"}
    source = subprocess.check_output(
        [
            "git",
            "show",
            revision + ":.github/scripts/k31_b_accuracy.py",
        ],
        cwd=ROOT,
    )
    exec(  # noqa: S102 -- frozen regression source
        compile(source, revision, "exec"), namespace
    )
    return namespace["allocation"]


def mock_device_stat(monkeypatch, inspect):
    original = ci.os.stat
    monkeypatch.setattr(
        ci.os,
        "stat",
        lambda path, *a, **kw: (
            inspect(str(path))
            if str(path).startswith("/dev/")
            else original(path, *a, **kw)
        ),
    )


def test_numeric_producer_group_is_a_supplementary_request(tmp_path, monkeypatch):
    monkeypatch.setattr(ci.os, "major", lambda value: 226, raising=False)
    monkeypatch.setattr(ci.os, "minor", lambda value: value, raising=False)
    inspect = lambda path: SimpleNamespace(
        st_mode=stat.S_IFCHR | 0o660, st_uid=0, st_gid=993, st_rdev=128
    )
    text = "--group-add 110 " + " ".join(
        f"--device /dev/dri/renderD{i}" for i in range(128, 185, 8)
    )
    with pytest.raises(ValueError, match="unexpected allocation argument"):
        old_allocation()(text, inspect)
    with pytest.raises(ValueError, match="declared group not observed"):
        old_allocation("a78232615ee3f57b4b149ea9f8ebf471fb4ff06e")(text, inspect)
    assert len(ci.allocation(text, inspect)["nodes"]) == 9
    assert ci.allocation(text, inspect)["declared_groups"] == [110]
    for bad in (
        text.replace("110", "root"),
        text.replace("110", str(2**32 - 1)),
        text + " --group-add 110",
        text.replace("--group-add 110", "--group-add=110"),
    ):
        with pytest.raises(ValueError):
            ci.allocation(bad, inspect)
    source, output = tmp_path / "allocation", tmp_path / "receipt.json"
    source.write_text(text)
    mock_device_stat(monkeypatch, inspect)
    old_capture = old_allocation(
        "a78232615ee3f57b4b149ea9f8ebf471fb4ff06e"
    ).__globals__["captured_allocation"]
    with pytest.raises(ValueError, match="declared group not observed"):
        old_capture(source, tmp_path / "old.json")
    assert "devices" not in json.loads((tmp_path / "old.json").read_text())
    ci.captured_allocation(source, output)
    report = json.loads(output.read_text())
    assert report["observed_device_groups"] == [993] and report["declared_groups"] == [
        110
    ]
    assert all(
        row["mode"] == "0o660" and row["gid"] == 993 for row in report["devices"]
    )


def test_parser_failure_retains_exact_input_before_tokenizing(tmp_path, monkeypatch):
    source, output = tmp_path / "allocation", tmp_path / "receipt.json"
    raw = b"--unsupported value\n"
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="unexpected allocation argument"):
        old_allocation()(source.read_text())
    assert not output.exists()
    split = ci.shlex.split

    def witnessed(text):
        assert base64.b64decode(json.loads(output.read_text())["raw_base64"]) == raw
        return split(text)

    monkeypatch.setattr(ci.shlex, "split", witnessed)
    with pytest.raises(ValueError, match="unexpected allocation argument"):
        ci.captured_allocation(source, output)
    report = json.loads(output.read_text())
    assert report["tokens"] == ["--unsupported", "value"]
    assert report["phase"] == "parse" and report["error"]["type"] == "ValueError"
    assert report["captured_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "raw,phase",
    [(b'"unfinished', "tokenize"), (b"\xff", "decode"), (b"x" * 16385, "read")],
)
def test_capture_is_bounded_and_retains_early_refusal(
    tmp_path, monkeypatch, raw, phase
):
    source, output = tmp_path / "allocation", tmp_path / "receipt.json"
    opened = Path.open
    sizes = []

    class Bounded(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            assert size == 16385
            return super().read(size)

    monkeypatch.setattr(
        Path,
        "open",
        lambda path, *a, **k: Bounded(raw) if path == source else opened(path, *a, **k),
    )
    with pytest.raises(ValueError):
        ci.captured_allocation(source, output)
    report = json.loads(output.read_text())
    assert sizes == [16385] and report["phase"] == phase
    assert base64.b64decode(report["raw_base64"]) == raw[:16384]
    assert report["truncated"] == (len(raw) > 16384) and "tokens" not in report


@pytest.mark.parametrize("failure", ["type", "stat"])
def test_runner_device_observation_precedes_refusal(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(ci.os, "major", lambda value: 226, raising=False)
    monkeypatch.setattr(ci.os, "minor", lambda value: value, raising=False)

    def inspect(path):
        if failure == "stat":
            raise PermissionError("fixture stat refused")
        return SimpleNamespace(
            st_mode=stat.S_IFREG | 0o640, st_uid=7, st_gid=993, st_rdev=0
        )

    mock_device_stat(monkeypatch, inspect)
    source, output = tmp_path / "allocation", tmp_path / "receipt.json"
    source.write_text(ACTUAL_ALLOCATION)
    with pytest.raises((PermissionError, ValueError)):
        ci.captured_allocation(source, output)
    report = json.loads(output.read_text())
    assert report["observation_location"] == "runner_process"
    assert report["devices"][0]["path"] == "/dev/kfd"
    if failure == "type":
        assert (
            report["devices"][0]["gid"] == 993
            and report["devices"][0]["mode"] == "0o640"
        )
        assert report["devices"][0]["type"] == stat.S_IFREG
    else:
        assert report["devices"][0]["error"]["type"] == "PermissionError"


@pytest.fixture
def access_fixture(tmp_path, monkeypatch):
    paths = ["/dev/kfd", *(f"/dev/dri/renderD{i}" for i in range(128, 185, 8))]
    monkeypatch.setattr(ci.os, "major", lambda value: value[0], raising=False)
    monkeypatch.setattr(ci.os, "minor", lambda value: value[1], raising=False)
    for name, value in {"getuid": 1001, "geteuid": 1001, "getegid": 1001}.items():
        monkeypatch.setattr(ci.os, name, lambda value=value: value, raising=False)
    monkeypatch.setattr(ci.os, "getgroups", lambda: [110, 993, 1001], raising=False)
    monkeypatch.setattr(ci.os, "O_CLOEXEC", 0x80000, raising=False)
    monkeypatch.setattr(ci, "EVIDENCE", tmp_path)

    def inspect(path):
        number = (
            (235, 0) if path == "/dev/kfd" else (226, int(path.rsplit("renderD", 1)[1]))
        )
        return SimpleNamespace(
            st_mode=stat.S_IFCHR | 0o660, st_uid=0, st_gid=110, st_rdev=number
        )

    mock_device_stat(monkeypatch, inspect)
    nodes = [
        {
            "path": path,
            "gid": 993,
            "major": inspect(path).st_rdev[0],
            "minor": inspect(path).st_rdev[1],
        }
        for path in paths
    ]
    admission = {
        "cid": CID,
        "allocation": {"nodes": nodes, "declared_groups": [110]},
        "docker": {"HostConfig": {"GroupAdd": ["110", "993"]}},
    }
    backing = tmp_path / "fd-backing"
    backing.write_bytes(b"")
    opened, closed, calls, state = {}, [], [], {"deny": False, "bad_fstat": False}
    real_open, real_fstat, real_close = ci.os.open, ci.os.fstat, ci.os.close

    def open_device(path, flags, *a, **kw):
        if path not in paths:
            return real_open(path, flags, *a, **kw)
        calls.append((path, flags))
        assert flags == ci.os.O_RDWR | ci.os.O_CLOEXEC
        if state["deny"]:
            raise PermissionError(13, "fixture open refused", path)
        fd = real_open(backing, ci.os.O_RDWR)
        opened[fd] = path
        return fd

    def fstat_device(fd):
        if fd not in opened:
            return real_fstat(fd)
        node = inspect(opened[fd])
        if state["bad_fstat"]:
            node.st_mode = stat.S_IFREG
        return node

    def close_device(fd):
        if fd in opened:
            closed.append(opened.pop(fd))
        real_close(fd)

    monkeypatch.setattr(ci.os, "open", open_device)
    monkeypatch.setattr(ci.os, "fstat", fstat_device)
    monkeypatch.setattr(ci.os, "close", close_device)
    try:
        yield admission, opened, closed, calls, state
    finally:
        for fd in list(opened):
            real_close(fd)


def test_unprivileged_access_uses_container_observations(access_fixture):
    admission, opened, closed, calls, _ = access_fixture
    report = ci.device_access(admission)
    assert report["passed"] and report["observation_location"] == "workload_container"
    assert report["euid"] == 1001 and len(closed) == len(calls) == 9 and not opened
    assert {row["gid"] for row in report["devices"]} == {110}
    assert all(row["read_write_open_closed"] for row in report["devices"])


@pytest.mark.parametrize(
    "failure", ["root", "missing_live", "extra_live", "docker", "deny", "fstat"]
)
def test_unprivileged_access_refuses_and_closes(
    access_fixture, tmp_path, monkeypatch, failure
):
    admission, opened, closed, calls, state = access_fixture
    if failure == "root":
        monkeypatch.setattr(ci.os, "geteuid", lambda: 0)
    elif failure in ("missing_live", "extra_live"):
        monkeypatch.setattr(
            ci.os,
            "getgroups",
            lambda: [993] if failure == "missing_live" else [110, 993, 555],
        )
    elif failure == "docker":
        admission["docker"]["HostConfig"]["GroupAdd"].append("555")
    else:
        state["deny" if failure == "deny" else "bad_fstat"] = True
    with pytest.raises((PermissionError, ValueError)):
        ci.device_access(admission)
    report = json.loads(next(tmp_path.glob("device-access-*.json")).read_text())
    assert not report["passed"] and len(report["devices"]) == 9 and "error" in report
    assert not opened
    assert len(closed) == (1 if failure == "fstat" else 0)
    assert len(calls) == (1 if failure in ("deny", "fstat") else 0)


@pytest.fixture
def isolated_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "EVIDENCE", tmp_path)
    credentials = {"uid": 1001, "euid": 1001, "gid": 1001, "groups": [110, 992]}
    for name, value in {"getuid": 1001, "geteuid": 1001, "getegid": 1001}.items():
        monkeypatch.setattr(ci.os, name, lambda value=value: value, raising=False)
    monkeypatch.setattr(
        ci.os, "getgroups", lambda: credentials["groups"], raising=False
    )
    admission = {"cid": CID}
    write(tmp_path / "admission.json", admission)
    worker = tmp_path / "worker.py"
    worker.write_text("""import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[2])
config = json.loads((root / "worker.json").read_text())
fd = os.open(root / "probe-fd", os.O_CREAT | os.O_RDWR, 0o600)
try:
    if config["mode"] == "timeout":
        time.sleep(10)
    if config["mode"] == "error":
        raise SystemExit(7)
finally:
    os.close(fd)
report = dict(config["credentials"], cid=config["cid"], passed=True,
              pid=os.getpid(), parent_pid=os.getppid())
if config["mode"] == "credentials":
    report["euid"] = 0
(root / f"device-access-from-{os.getppid()}.json").write_text(json.dumps(report))
""")
    events, processes = [], []
    state = {"mode": "success"}
    real_run, real_popen = ci.subprocess.run, ci.subprocess.Popen

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def run(argv, **kwargs):
        events.append("spawn")
        assert argv == [sys.executable, ci.__file__, "access-probe", str(tmp_path)]
        assert (
            kwargs["timeout"] == 30
            and not {"env", "user", "group", "extra_groups", "pass_fds"} & kwargs.keys()
        )
        write(
            tmp_path / "worker.json",
            {"mode": state["mode"], "credentials": credentials, "cid": CID},
        )
        kwargs["timeout"] = 1 if state["mode"] == "timeout" else 3
        try:
            return real_run([argv[0], str(worker), *argv[2:]], **kwargs)
        finally:
            assert processes and all(p.returncode is not None for p in processes)
            events.append("waited")

    def telemetry(path):
        phase = "pre" if path.name.startswith("pre-access") else "post"
        events.append(phase)
        if phase == "post":
            assert all(p.returncode is not None for p in processes)
        if state["mode"] == phase + "_busy":
            raise ValueError("GPU processes remain")

    monkeypatch.setattr(ci.subprocess, "Popen", popen)
    monkeypatch.setattr(ci.subprocess, "run", run)
    monkeypatch.setattr(ci, "telemetry", telemetry)
    monkeypatch.setattr(
        ci, "device_access", lambda *a, **kw: pytest.fail("parent opened devices")
    )
    return tmp_path, admission, state, events, processes


@pytest.mark.parametrize(
    "mode", ["success", "error", "timeout", "credentials", "pre_busy", "post_busy"]
)
def test_probe_waits_before_strict_postcheck(isolated_probe, mode):
    root, admission, state, events, processes = isolated_probe
    state["mode"] = mode
    if mode == "success":
        assert ci.isolated_device_access(root, admission)["passed"]
    else:
        with pytest.raises((ValueError, subprocess.TimeoutExpired)):
            ci.isolated_device_access(root, admission)
    assert events == (
        ["pre"] if mode == "pre_busy" else ["pre", "spawn", "waited", "post"]
    )
    assert all(p.returncode is not None for p in processes)
    if mode != "pre_busy":
        record = json.loads(
            next(root.glob("device-access-from-*-exit.json")).read_text()
        )
        assert record["timeout_seconds"] == 30
        if mode in ("error", "timeout", "credentials"):
            assert "error" in record
        if sys.platform != "win32":
            for process in processes:
                with pytest.raises(ChildProcessError):
                    ci.os.waitpid(process.pid, ci.os.WNOHANG)
        # Windows cannot remove this file while the child's CRT descriptor is open.
        assert (root / "probe-fd").is_file()
        (root / "probe-fd").unlink(missing_ok=True)


def test_child_entry_uses_existing_access_check(tmp_path, monkeypatch):
    write(tmp_path / "admission.json", {"cid": CID})
    monkeypatch.setattr(ci, "EVIDENCE", tmp_path)
    calls = []
    monkeypatch.setattr(
        ci, "device_access", lambda admission, output: calls.append((admission, output))
    )
    ci.device_access_child(tmp_path)
    assert calls == [
        ({"cid": CID}, tmp_path / f"device-access-from-{ci.os.getppid()}.json")
    ]


def test_zero_usage_unknown_pid_still_refuses(tmp_path, monkeypatch):
    nodes = [{"pci": f"0000:{i:02x}:00.0"} for i in range(8)]
    output = {
        "--showpids": "PID PROCESS NAME GPU(s) VRAM USED\n1092302 UNKNOWN 0 0",
        "--showuse": "{}",
        "--showdriverversion": "fixture",
        "--showbus": json.dumps(
            {f"card{i}": {"PCI Bus": n["pci"]} for i, n in enumerate(nodes)}
        ),
    }
    monkeypatch.setattr(
        ci,
        "command",
        lambda argv: {
            "argv": argv,
            "returncode": 0,
            "stdout": output[argv[1]],
            "stderr": "",
        },
    )
    monkeypatch.setattr(ci, "read", lambda path: {"nodes": nodes})
    monkeypatch.setattr(
        ci.os, "uname", lambda: SimpleNamespace(release="fixture"), raising=False
    )
    with pytest.raises(ValueError, match="GPU processes remain"):
        ci.telemetry(tmp_path / "telemetry.json")


def test_allocated_nodes_and_ambiguous_inputs(monkeypatch):
    monkeypatch.setattr(ci.os, "major", lambda value: 226, raising=False)
    monkeypatch.setattr(ci.os, "minor", lambda value: value, raising=False)
    inspect = lambda path: SimpleNamespace(
        st_mode=stat.S_IFCHR, st_gid=109, st_rdev=128
    )
    text = " ".join(f"--device /dev/dri/renderD{i}" for i in range(128, 136))
    assert len(ci.allocation(text, inspect)["nodes"]) == 9
    for invalid in (
        "",
        text + " --privileged",
        text.replace("renderD135", "renderD128"),
        text.replace("/dev/dri/renderD128", "/dev/dri"),
        text + " ; echo bad",
    ):
        with pytest.raises(ValueError):
            ci.allocation(invalid, inspect)
    with pytest.raises(ValueError, match="character device"):
        ci.allocation(text, lambda path: SimpleNamespace(st_mode=stat.S_IFREG))


def test_actual_cpu_constraint_old_refuses_new_binds(tmp_path, monkeypatch):
    assert (
        hashlib.sha256(ACTUAL_ALLOCATION.encode()).hexdigest()
        == "d69d45c2ff00ef48a9fd9e36f61079aca9514776dcfec682bed19bb7031faadb"
    )
    monkeypatch.setattr(ci.os, "major", lambda value: 226, raising=False)
    monkeypatch.setattr(ci.os, "minor", lambda value: value, raising=False)
    inspect = lambda path: SimpleNamespace(
        st_mode=stat.S_IFCHR | 0o660, st_uid=0, st_gid=992, st_rdev=128
    )
    with pytest.raises(ValueError, match="unexpected allocation argument"):
        old_allocation("39ee6af2b66796ee30277a4c557e87d9a9338846")(
            ACTUAL_ALLOCATION, inspect
        )
    allocated = ci.allocation(ACTUAL_ALLOCATION, inspect)
    assert allocated["cpu_affinity"] == list(range(128))
    assert allocated["permitted_cpu_affinity"] == list(range(256))
    assert allocated["declared_cpu_set"] == "0-127" and len(allocated["nodes"]) == 9
    mock_device_stat(monkeypatch, inspect)
    source, output = tmp_path / "allocation", tmp_path / "receipt.json"
    source.write_bytes(ACTUAL_ALLOCATION.encode())
    ci.captured_allocation(source, output)
    receipt = json.loads(output.read_text())
    assert receipt["cpu_affinity"] == list(range(128))
    assert receipt["permitted_cpu_affinity"] == list(range(256))
    assert base64.b64decode(receipt["raw_base64"]) == ACTUAL_ALLOCATION.encode()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "all",
        "-1",
        "2-1",
        "0-999999999999",
        "0,",
        "0,,1",
        "0-3,3",
        "0,0",
        "0;true",
        "0-127 --cpuset-cpus 0-127",
    ],
)
def test_bad_cpu_constraints_refuse(value):
    with pytest.raises(ValueError):
        ci.allocation(ACTUAL_ALLOCATION.replace("0-127", value))


def test_cpu_constraint_does_not_intersect_or_widen(tmp_path, monkeypatch):
    assert ci.allocated_cpus("0-1,4", {0, 1, 4, 5}) == [0, 1, 4]
    assert ci.allocated_cpus(None, {2, 7}) == [2, 7]
    with pytest.raises(ValueError, match="unpermitted"):
        ci.allocated_cpus("0-4", {0, 1, 3, 4})
    with pytest.raises(ValueError, match="no permitted"):
        ci.allocated_cpus("0", set())
    monkeypatch.setattr(ci.os, "sched_getaffinity", lambda pid: set(range(64)))
    with pytest.raises(ValueError, match="outside permitted"):
        ci.allocation(ACTUAL_ALLOCATION)
    source, output = tmp_path / "allocation", tmp_path / "receipt.json"
    source.write_bytes(ACTUAL_ALLOCATION.encode())
    with pytest.raises(ValueError, match="outside permitted"):
        ci.captured_allocation(source, output)
    report = json.loads(output.read_text())
    assert report["permitted_cpu_affinity"] == list(range(64))
    assert report["phase"] == "parse" and "error" in report


@pytest.mark.parametrize(
    "host_set,live,permitted",
    [
        ("0-2", {0, 1}, {0, 1, 2}),
        ("0-1", {0, 1, 2}, {0, 1, 2}),
        ("0-1", {0}, {0, 1}),
        ("0-1", {0, 1}, {0}),
    ],
)
def test_cpu_admission_rejects_unbound_container(
    tmp_path, monkeypatch, host_set, live, permitted
):
    policy = ci.read(ci.POLICY)
    write(
        tmp_path / "admission.json",
        {
            "kind": "ci-container-v1",
            "leg": "B-accuracy",
            "image_manifest": policy["image"],
            "cid": CID,
            "docker": {
                "Id": CID,
                "Image": policy["config"],
                "HostConfig": {"CpusetCpus": host_set},
            },
            "allocation": {
                "cpu_affinity": [0, 1],
                "permitted_cpu_affinity": sorted(permitted),
            },
        },
    )
    monkeypatch.setattr(ci.os, "sched_getaffinity", lambda pid: live)
    with pytest.raises(ValueError, match="CPU"):
        ci.verify_admission(tmp_path)


@pytest.mark.parametrize("kind", ["dataset", "model", "postflight"])
def test_containers_use_only_the_bound_cpu_set(tmp_path, monkeypatch, kind):
    (tmp_path / "output").mkdir()
    write(
        tmp_path / "allocation.json",
        {
            "nodes": [{"path": "/dev/kfd", "gid": 993}],
            "declared_groups": [110],
            "cpu_affinity": [1, 3],
            "permitted_cpu_affinity": list(range(8)),
        },
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    calls = []

    def refused_create(argv, seconds=30):
        calls.append(argv)
        return {"returncode": 1, "stdout": "", "stderr": "fixture stops before Docker"}

    monkeypatch.setattr(ci, "command", refused_create)
    ci.container(kind, tmp_path)
    assert [arg for arg in calls[0] if arg.startswith("--cpuset-cpus")] == [
        "--cpuset-cpus=1,3"
    ]
    assert {arg for arg in calls[0] if arg.startswith("--group-add=")} == (
        set() if kind == "dataset" else {"--group-add=110", "--group-add=993"}
    )


@pytest.mark.parametrize(
    "remove_rc,query_rc,present",
    [(0, 0, False), (1, 0, False), (0, 1, False), (0, 0, True)],
)
def test_cleanup_requires_exact_cid_and_observed_absence(remove_rc, query_rc, present):
    calls = []

    def call(argv):
        calls.append(argv)
        rc = remove_rc if argv[1] == "rm" else query_rc if argv[1] == "ps" else 0
        return {
            "argv": argv,
            "returncode": rc,
            "stdout": CID if present and argv[1] == "ps" else "",
            "stderr": "",
        }

    result = ci.cleanup(CID, call)
    assert result["passed"] == (remove_rc == query_rc == 0 and not present)
    assert calls[:2] == [
        ["docker", "stop", "--time", "10", CID],
        ["docker", "rm", "--force", CID],
    ]
    with pytest.raises(ValueError, match="CID"):
        ci.cleanup("container-name", call)


def test_timeout_retains_failure_and_cleans_saved_cid(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    write(
        tmp_path / "allocation.json",
        {"nodes": [], "declared_groups": [], "cpu_affinity": []},
    )
    policy = tmp_path / "policy.json"
    write(
        policy,
        {"image": "pinned", "config": "configuration", "seconds": {"postflight": 1}},
    )
    monkeypatch.setattr(ci, "POLICY", policy)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")

    def command(argv, seconds=30):
        if argv[1] == "create":
            Path(argv[argv.index("--cidfile") + 1]).write_text(CID)
            return {"returncode": 0, "stdout": CID, "stderr": ""}
        return {
            "returncode": 0,
            "stdout": json.dumps(
                [
                    {
                        "Image": "configuration",
                        "State": {"Running": True, "OOMKilled": False, "ExitCode": 0},
                    }
                ]
            ),
            "stderr": "",
        }

    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(
        ci, "cleanup", lambda cid: {"cid": cid, "passed": True, "absent": True}
    )
    monkeypatch.setattr(
        ci.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(a[0], 1)),
    )
    result = ci.container("postflight", tmp_path)
    assert result["status"] == "failed" and result["attach_exit"] == 124
    assert result["cleanup"]["cid"] == CID
    assert (
        json.loads((tmp_path / "output/postflight-container.json").read_text())
        == result
    )


def test_typed_admission_refuses_legacy_before_native_checks(tmp_path):
    write(tmp_path / "admission.json", {"kind": "local-g24", "leg": "B-accuracy"})
    with pytest.raises(ValueError, match="admission type"):
        ci.verify_admission(tmp_path)


def test_missing_client_fails_before_stock_install_path(tmp_path, monkeypatch):
    current = importlib.util.spec_from_file_location(
        "k31_phase0_assets", ROOT / ".github/scripts/k31_phase0_assets.py"
    )
    assets = importlib.util.module_from_spec(current)
    current.loader.exec_module(assets)
    monkeypatch.setitem(sys.modules, "k31_phase0_assets", assets)
    monkeypatch.setattr(ci.os, "environ", {"PATH": "empty"})
    monkeypatch.setattr(ci.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="refuse install"):
        ci.entry(tmp_path)


def test_real_checker_patch_preserves_full_evaluation(tmp_path):
    for name in ("check_results.py", "verify_inputs.py"):
        source = subprocess.check_output(
            [
                "git",
                "show",
                f"3d7a07a470555840eb686f0b8f0375efc07d58fa:seed/3577/{name}",
            ],
            cwd=ROOT,
        )
        (tmp_path / name).write_bytes(source)
    old = (tmp_path / "check_results.py").read_text()
    subprocess.run(
        ["git", "apply", str(ROOT / ".github/validation/3577-ci-binding.patch")],
        cwd=tmp_path,
        check=True,
    )
    new = (tmp_path / "check_results.py").read_text()
    assert (
        old[old.index('assert read("verified-inputs.json")') :]
        == new[new.index('assert read("verified-inputs.json")') :]
    )
    evidence = tmp_path / "B-accuracy"
    evidence.mkdir()
    state = {"Running": False, "OOMKilled": False, "ExitCode": 0}
    cleanup = {
        "cid": CID,
        "passed": True,
        "absent": True,
        "remove": {"returncode": 0, "argv": ["docker", "rm", "--force", CID]},
        "inventory": {"returncode": 0, "stdout": ""},
    }
    for name, value in {
        "entry-exit.json": {"entry_exit": 0},
        "lm-eval-exit.json": {"exit_code": 0},
        "verified-inputs.json": {"source": "fixed"},
        "postflight.json": {"passed": True},
        "compiled-trace-request.json": {"requested": 0},
        "compiled-trace-outcome.json": {"status": "not_requested"},
        "container-lifecycle.json": dict(
            kind="ci-container-lifecycle-v1",
            **{
                k: {
                    "kind": k,
                    "status": "passed",
                    "attach_exit": 0,
                    "state": state,
                    "cleanup": cleanup,
                    "final_inspect": {"stdout": json.dumps([{"Id": CID}])},
                }
                for k in ("model", "postflight")
            },
        ),
    }.items():
        write(evidence / name, value)
    results = evidence / "accuracy_test_results"
    results.mkdir()
    data = {
        "n-samples": {"gsm8k": {"effective": 1319}},
        "n-shot": {"gsm8k": 3},
        "config": {"limit": None},
        "results": {"gsm8k": {"exact_match,flexible-extract": 0.96}},
        "atom_ci_metadata": {"mtp_acceptance_rate": 25},
    }
    write(results / "result.json", data)
    samples = results / "samples_gsm8k.jsonl"
    samples.write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": i,
                    "doc": {"question": "question"},
                    "arguments": ["question"],
                    "resps": [["answer"]],
                }
            )
            for i in range(1319)
        )
    )
    run = lambda: subprocess.run(
        [sys.executable, str(tmp_path / "check_results.py"), str(evidence)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run().returncode == 0
    legacy = tmp_path / "legacy_checker.py"
    legacy.write_text(old)
    red = subprocess.run(
        [sys.executable, str(legacy), str(evidence)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert red.returncode != 0 and "supervisor-exit.json" in red.stderr
    data["results"]["gsm8k"]["exact_match,flexible-extract"] = 0.95
    write(results / "result.json", data)
    assert run().returncode != 0
    data["results"]["gsm8k"]["exact_match,flexible-extract"] = 0.96
    write(results / "result.json", data)
    samples.write_text(samples.read_text() + "\n" + samples.read_text().splitlines()[0])
    assert run().returncode != 0


def test_workflow_selects_one_case_and_retains_failure_artifacts():
    workflow = yaml.load(
        (ROOT / ".github/workflows/atom-test.yaml").read_text(), Loader=yaml.BaseLoader
    )
    job = workflow["jobs"]["k31-b-accuracy"]
    assert "inputs.k31_b_accuracy" in job["if"]
    assert job["steps"][0]["if"] == "${{ inputs.k31_phase0 }}"
    assert "!inputs.k31_b_accuracy" in workflow["concurrency"]["cancel-in-progress"]
    assert job["steps"][-2]["if"] == job["steps"][-1]["if"] == "always()"
    assert "work" not in job["steps"][-1]["with"].get("exclude", "")
