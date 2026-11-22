"""Temporary, fixed B-accuracy route. Docker owns each dedicated container."""

import argparse
import base64
import ctypes
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from atomesh.numa_exec import parse_cpulist
from k31_phase0_assets import measured_hash
from k31_phase0_inputs import checked_files, digest, require, verify

CONTROL = Path(__file__).resolve().parents[2]
POLICY = CONTROL / ".github/validation/k31-b-accuracy.json"
EVIDENCE = Path("/app/atom-3577/evidence/B-accuracy")
NATIVE_SOURCE = Path("/app/aiter-test/aiter/jit")
OPUS_SOURCE = Path("/job/native-source/native/module_fmha_fwd_bf16_opus.so")
CACHE_ENV = {
    "HOME": "/root",
    "TMPDIR": "/tmp",
    # Library cache labels only; the workload's numeric UID/GID remain authoritative.
    "LOGNAME": "k31-b-accuracy",
    "USER": "k31-b-accuracy",
    "XDG_CACHE_HOME": "/root/.cache",
    "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
    "TRITON_CACHE_DIR": "/root/.cache/triton",
    "TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions",
    "AITER_JIT_DIR": "/root/.aiter/jit",
    "FLYDSL_RUNTIME_CACHE_DIR": "/root/.cache/flydsl",
}


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    return digest(Path(path).read_bytes())


def save(path, value):
    print("K31_CI_REPORT " + json.dumps(value), flush=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def command(argv, seconds=30):
    try:
        p = subprocess.run(
            argv, capture_output=True, text=True, timeout=seconds, check=False
        )
        return {
            "argv": argv,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "argv": argv,
            "returncode": 124,
            "stdout": "",
            "stderr": "command timeout",
        }
    except OSError as error:
        return {"argv": argv, "returncode": 127, "stdout": "", "stderr": repr(error)}


def allocated_cpus(value, permitted):
    require(permitted, "no permitted CPUs observed")
    if value is None:
        return sorted(permitted)
    require(
        re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value),
        "malformed allocated CPU set",
    )
    count = 0
    for part in value.split(","):
        first, _, last = part.partition("-")
        low, high = int(first), int(last or first)
        require(
            low <= high <= max(permitted),
            "allocated CPU range outside permitted bounds",
        )
        count += high - low + 1
    cpus = parse_cpulist(value)
    require(
        len(cpus) == count and cpus <= permitted,
        "repeated or unpermitted allocated CPUs",
    )
    return sorted(cpus)


def allocation(text, inspect_path=os.stat, permitted_cpus=None):
    tokens, paths = shlex.split(text), []
    permitted = set(
        os.sched_getaffinity(0) if permitted_cpus is None else permitted_cpus
    )
    declared_cpus = None
    if len(tokens) >= 2 and tokens[-2] == "--cpuset-cpus":
        declared_cpus, tokens = tokens[-1], tokens[:-2]
    cpus = allocated_cpus(declared_cpus, permitted)
    groups = []
    if tokens[:1] == ["--group-add"]:
        require(
            len(tokens) > 1 and re.fullmatch(r"[0-9]+", tokens[1]),
            "non-numeric allocated group",
        )
        groups, tokens = [int(tokens[1])], tokens[2:]
        require(groups[0] < 2**32 - 1, "allocated group outside Linux GID range")
    while tokens:
        flag = tokens.pop(0)
        value = (
            tokens.pop(0)
            if flag == "--device" and tokens
            else flag.removeprefix("--device=")
        )
        require(
            flag == "--device" or flag.startswith("--device="),
            "unexpected allocation argument",
        )
        require(
            re.fullmatch(r"/dev/dri/renderD[0-9]+", value),
            "ambiguous render allocation",
        )
        paths.append(value)
    require(len(paths) == len(set(paths)) == 8, "need eight allocated render nodes")
    result = []
    for path in ["/dev/kfd", *paths]:
        node = inspect_path(path)
        require(stat.S_ISCHR(node.st_mode), "allocated node is not a character device")
        result.append(
            {
                "path": path,
                "gid": node.st_gid,
                "major": os.major(node.st_rdev),
                "minor": os.minor(node.st_rdev),
            }
        )
    return {
        "nodes": result,
        "declared_groups": groups,
        "declared_cpu_set": declared_cpus,
        "permitted_cpu_affinity": sorted(permitted),
        "cpu_affinity": cpus,
    }


def observe_device(path, report, output):
    row = {"path": path}
    report["devices"].append(row)
    try:
        node = os.stat(path)
        row.update(
            type=stat.S_IFMT(node.st_mode),
            mode=oct(stat.S_IMODE(node.st_mode)),
            uid=node.st_uid,
            gid=node.st_gid,
            major=os.major(node.st_rdev),
            minor=os.minor(node.st_rdev),
        )
        return node
    except OSError as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        save(output, report)


def captured_allocation(path, output):
    limit = 16384
    report = {
        "kind": "ci-allocation-input-v1",
        "path": str(path),
        "phase": "read",
        "observation_location": "runner_process",
        "devices": [],
    }
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        raw = data[:limit]
        report.update(
            captured_bytes=len(raw),
            captured_sha256=digest(raw),
            raw_base64=base64.b64encode(raw).decode("ascii"),
            truncated=len(data) > limit,
        )
        save(output, report)
        require(len(data) <= limit, "allocation input exceeds 16384-byte capture limit")
        report["phase"] = "decode"
        text = raw.decode("utf-8")
        report["phase"] = "tokenize"
        report["tokens"] = shlex.split(text)
        save(output, report)
        report["phase"] = "observe_cpu"
        report["permitted_cpu_affinity"] = sorted(os.sched_getaffinity(0))
        save(output, report)
        report["phase"] = "parse"
        result = allocation(
            text,
            inspect_path=lambda device: observe_device(device, report, output),
            permitted_cpus=report["permitted_cpu_affinity"],
        )
        report["declared_groups"] = result["declared_groups"]
        report["observed_device_groups"] = sorted(
            {node["gid"] for node in result["nodes"]}
        )
        report["declared_cpu_set"] = result["declared_cpu_set"]
        report["cpu_affinity"] = result["cpu_affinity"]
        report["phase"] = "accepted"
        return result
    except (OSError, ValueError) as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        save(output, report)


def prepare(work):
    policy = read(POLICY)
    wanted = native_pins()  # Refuse before allocation, pull or native input reads.
    native = work / "native-source"
    external = policy["bf16_opus_input"]
    require(
        re.fullmatch("[0-9a-f]{40}", external["ref"])
        and native.resolve() == work.resolve() / "native-source"
        and (native / "native").resolve() == native.resolve() / "native",
        "native input ref or containment differs",
    )
    git = ["git", "-C", str(native)]
    head = command(git + ["rev-parse", "HEAD"])
    auth = command(
        git
        + [
            "config",
            "--local",
            "--name-only",
            "--get-regexp",
            r"extraheader|credential|include|pushurl|insteadof",
        ]
    )
    origin = command(git + ["remote", "get-url", "origin"])
    require(
        head["returncode"] == 0
        and head["stdout"].strip() == external["ref"]
        and auth["returncode"] == 1
        and origin["returncode"] == 0
        and origin["stdout"].strip()
        in {"https://github.com/ROCm/ATOM", "https://github.com/ROCm/ATOM.git"},
        "native checkout identity or credential cleanup differs",
    )
    require(external["sha256"] == wanted[OPUS_SOURCE.name], "native input pin differs")
    payload = checked_files(
        native / "native", {"files": {OPUS_SOURCE.name: external["sha256"]}}
    )
    require(
        len(payload[OPUS_SOURCE.name]) == external["bytes"], "native input size differs"
    )
    (native / "native" / OPUS_SOURCE.name).chmod(0o444)
    (native / "native").chmod(0o555)
    verify(
        work / "inputs/seed",
        CONTROL / ".github/validation/k31-phase0.json",
        work / "inputs/input-manifest.json",
        policy["input_manifest"],
    )
    require(
        checksum(work / "phase0/receipt.json") == policy["phase0_receipt"],
        "Phase0 receipt differs",
    )
    (work / "output").mkdir(mode=0o700)
    (work / "output/work").mkdir()
    (work / "data").mkdir()
    shutil.copytree(work / "inputs/seed/3577", work / "input")
    subprocess.run(
        ["git", "apply", str(CONTROL / ".github/validation/3577-ci-binding.patch")],
        cwd=work / "input",
        check=True,
    )
    bindings = {
        "ci_policy": {
            "path": "/control/.github/validation/k31-b-accuracy.json",
            "sha256": checksum(POLICY),
        },
        "ci_adapter": {
            "path": "/control/.github/scripts/k31_b_accuracy.py",
            "sha256": checksum(__file__),
        },
        "prior_phase0": {
            "path": "/job/phase0/receipt.json",
            "sha256": policy["phase0_receipt"],
        },
        "native_input": {
            "path": str(OPUS_SOURCE),
            "sha256": external["sha256"],
            "bytes": external["bytes"],
            "source_ref": external["ref"],
        },
    }
    save(work / "input/preuse-bindings.json", bindings)
    save(
        work / "input/prepared-manifest.json",
        {p.name: checksum(p) for p in sorted((work / "input").iterdir())},
    )
    source = work / "source"
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        == policy["source"],
        "source differs",
    )
    subprocess.run(
        ["git", "apply", str(work / "input/B-validation.patch")], cwd=source, check=True
    )
    allocated = captured_allocation(
        Path("/etc/podinfo/gha-render-devices"), work / "output/allocation-input.json"
    )
    nodes = allocated["nodes"]
    for node in nodes[1:]:
        device = Path("/sys/class/drm") / Path(node["path"]).name / "device"
        node["pci"] = device.resolve(strict=True).name
        require(
            re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", node["pci"]),
            "render PCI mapping unavailable",
        )
        require(
            (device / "vendor").read_text().strip() == "0x1002",
            "render device is not AMD",
        )
    save(
        work / "allocation.json",
        {
            "nodes": nodes,
            "declared_groups": allocated["declared_groups"],
            "input_receipt_sha256": checksum(work / "output/allocation-input.json"),
            "requested_runner": policy["runner"],
            "job": {
                k: os.environ.get(k)
                for k in (
                    "GITHUB_RUN_ID",
                    "GITHUB_RUN_ATTEMPT",
                    "GITHUB_JOB",
                    "GITHUB_SHA",
                    "RUNNER_NAME",
                )
            },
            "cpu_affinity": allocated["cpu_affinity"],
            "permitted_cpu_affinity": allocated["permitted_cpu_affinity"],
            "declared_cpu_set": allocated["declared_cpu_set"],
        },
    )


def stage_dataset(work):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    policy = read(POLICY)
    raw = work / "data/gsm8k/main"
    raw.mkdir(parents=True)
    for split, wanted in policy["parquets"].items():
        name = f"{split}-00000-of-00001.parquet"
        cached = hf_hub_download(
            "openai/gsm8k",
            "main/" + name,
            repo_type="dataset",
            revision=policy["dataset_revision"],
        )
        require(checksum(cached) == wanted, "pinned parquet differs")
        shutil.copyfile(cached, raw / name)
    data = load_dataset("openai/gsm8k", "main", revision=policy["dataset_revision"])
    require(
        len(data["train"]) == 7473 and len(data["test"]) == 1319,
        "dataset row counts differ",
    )
    save(
        EVIDENCE / "dataset.json",
        {
            "revision": policy["dataset_revision"],
            "raw": policy["parquets"],
            "cache": {
                str(p.relative_to(work / "data")): checksum(p)
                for p in (work / "data").rglob("*")
                if p.is_file()
            },
        },
    )


def telemetry(output):
    pids = command(["rocm-smi", "--showpids"])
    memory = command(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"])
    driver = command(["rocm-smi", "--showdriverversion"])
    identities = command(["rocm-smi", "--showbus", "--showuniqueid", "--json"])
    report = {
        "kind": "device-telemetry-v1",
        "resource_metrics": "observed-only",
        "pids": pids,
        "memory": memory,
        "driver": driver,
        "identities": identities,
        "kernel": os.uname().release,
    }
    save(output, report)
    require(
        pids["returncode"]
        == memory["returncode"]
        == driver["returncode"]
        == identities["returncode"]
        == 0,
        "device telemetry unavailable",
    )
    require(
        {c["PCI Bus"].lower() for c in json.loads(identities["stdout"]).values()}
        == {d["pci"] for d in read("/job/allocation.json")["nodes"] if "pci" in d},
        "telemetry allocation differs",
    )
    require("No KFD PIDs currently running" in pids["stdout"], "GPU processes remain")
    cards = json.loads(memory["stdout"])
    require(len(cards) == 8, "ambiguous telemetry device count")
    for card in cards.values():
        # Accuracy observes resource use; native/model allocation checks availability.
        use = float(str(card["GPU use (%)"]))
        used = int(str(card["VRAM Total Used Memory (B)"]))
        total = int(str(card["VRAM Total Memory (B)"]))
        require(
            math.isfinite(use) and 0 <= use <= 100 and 0 <= used <= total and total > 0,
            "invalid telemetry resource values",
        )
    report["checks_passed"] = True
    save(output, report)


def device_map(output):
    import torch

    require(
        torch.version.hip and torch.cuda.device_count() == 8, "need eight HIP devices"
    )
    devices = []
    for ordinal in range(8):
        text = str(torch.cuda.get_device_properties(ordinal))
        require("gfx950" in text, "GPU architecture differs")
        pci = [
            int(re.search(r"\b" + key + r"=(\d+)", text).group(1))
            for key in ("pci_domain_id", "pci_bus_id", "pci_device_id")
        ]
        devices.append(
            {
                "ordinal": ordinal,
                "pci": f"{pci[0]:04x}:{pci[1]:02x}:{pci[2]:02x}.0",
                "uuid": re.search(r"\buuid=([0-9a-fA-F-]+)", text).group(1).lower(),
                "properties": text,
            }
        )
    require(
        len({d["pci"] for d in devices}) == len({d["uuid"] for d in devices}) == 8,
        "device identities are not unique",
    )
    require(
        {d["pci"] for d in devices}
        == {d["pci"] for d in read("/job/allocation.json")["nodes"] if "pci" in d},
        "HIP allocation differs",
    )
    report = {
        "devices": devices,
        "hip": torch.version.hip,
        "scope": "visible devices; not an observed model-rank map",
    }
    save(output, report)
    report["native_modules"] = native_loads()
    report["native_checks_passed"] = True
    save(output, report)


def device_groups(allocated):
    return set(allocated["declared_groups"]) | {
        node["gid"] for node in allocated["nodes"]
    }


def device_access(admission, output=None):
    output = output or EVIDENCE / f"device-access-{os.getpid()}.json"
    report = {
        "kind": "ci-device-access-v1",
        "observation_location": "workload_container",
        "cid": admission["cid"],
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "gid": os.getegid(),
        "groups": os.getgroups(),
        "devices": [],
        "passed": False,
    }
    try:
        save(output, report)
        allocated = admission["allocation"]
        nodes = allocated["nodes"]
        paths = [node["path"] for node in nodes]
        require(
            len(paths) == len(set(paths)) == 9
            and paths[0] == "/dev/kfd"
            and all(
                re.fullmatch(r"/dev/dri/renderD[0-9]+", path) for path in paths[1:]
            ),
            "device access allowlist differs",
        )
        for path in paths:
            observe_device(path, report, output)
        requested = device_groups(allocated)
        require(
            set(admission["docker"]["HostConfig"]["GroupAdd"])
            == {str(gid) for gid in requested},
            "Docker supplementary groups differ",
        )
        require(
            report["uid"] != 0 and report["euid"] != 0,
            "device access requires the non-root workload UID",
        )
        require(
            set(report["groups"]) | {report["gid"]} == requested | {report["gid"]},
            "live supplementary groups differ",
        )
        for expected, row in zip(nodes, report["devices"]):
            require(
                row["type"] == stat.S_IFCHR
                and (row["major"], row["minor"])
                == (expected["major"], expected["minor"]),
                "container device identity differs",
            )
            report["checking_device"] = row["path"]
            fd = os.open(row["path"], os.O_RDWR | os.O_CLOEXEC)
            try:
                opened = os.fstat(fd)
                require(
                    stat.S_ISCHR(opened.st_mode)
                    and (os.major(opened.st_rdev), os.minor(opened.st_rdev))
                    == (row["major"], row["minor"]),
                    "opened device identity differs",
                )
            finally:
                os.close(fd)
            row["read_write_open_closed"] = True
            save(output, report)
        report["passed"] = True
    except (OSError, ValueError) as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        save(output, report)
    return report


def device_access_child(root):
    return device_access(
        read(root / "admission.json"),
        EVIDENCE / f"device-access-from-{os.getppid()}.json",
    )


def isolated_device_access(root, admission):
    parent = os.getpid()
    stem = EVIDENCE / f"device-access-from-{parent}"
    output = stem.with_suffix(".json")
    require(not output.exists(), "device probe receipt already exists")
    telemetry(EVIDENCE / f"pre-access-{parent}.json")
    result = {
        "kind": "ci-access-probe-exit-v1",
        "cid": admission["cid"],
        "parent_pid": parent,
        "timeout_seconds": 30,
    }
    try:
        with stem.with_suffix(".log").open("x") as log:
            child = subprocess.run(
                [sys.executable, __file__, "access-probe", str(root)],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        result["returncode"] = child.returncode
        require(child.returncode == 0, "device access child failed")
        access = read(output)
        require(
            access["passed"]
            and access["cid"] == admission["cid"]
            and access["parent_pid"] == parent
            and access["pid"] != parent,
            "device probe receipt differs",
        )
        require(
            (access["uid"], access["euid"], access["gid"])
            == (os.getuid(), os.geteuid(), os.getegid())
            and set(access["groups"]) == set(os.getgroups()),
            "device probe credentials differ",
        )
        return access
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        save(EVIDENCE / f"device-access-from-{parent}-exit.json", result)
        telemetry(EVIDENCE / f"post-access-{parent}.json")


def native_pins():
    pins = {
        Path(path).name: value
        for path, value in read(CONTROL / ".github/validation/k31-phase0.json")[
            "native_files"
        ].items()
    }
    pins["module_fmha_fwd_bf16_opus.so"] = read(POLICY)["bf16_opus_sha256"]
    for name, value in pins.items():
        require(
            isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value),
            f"required native module has no qualified hash: {name}",
        )
    return pins


def native_view(create=False):
    wanted = native_pins()
    policy = read(POLICY)
    git = ["git", "-C", str(NATIVE_SOURCE.parents[1])]
    head = command(git + ["rev-parse", "HEAD"])
    clean = command(git + ["diff", "--exit-code", "--quiet", "HEAD"])
    require(
        head["returncode"] == clean["returncode"] == 0
        and head["stdout"].strip() == policy["aiter"],
        "native source differs",
    )
    source = sorted(NATIVE_SOURCE.glob("*.so"))
    external = policy["bf16_opus_input"]
    require(OPUS_SOURCE.name not in {p.name for p in source}, "duplicate Opus source")
    require(
        OPUS_SOURCE.is_file()
        and OPUS_SOURCE.resolve() == OPUS_SOURCE
        and OPUS_SOURCE.stat().st_size == external["bytes"]
        and external["sha256"] == wanted[OPUS_SOURCE.name]
        and os.statvfs(OPUS_SOURCE).f_flag & os.ST_RDONLY,
        "external Opus source differs or is writable",
    )
    source.append(OPUS_SOURCE)
    require(set(wanted) <= {p.name for p in source}, "required native module missing")
    modules = {}
    for path in source:
        require(
            re.fullmatch(r"[A-Za-z0-9_]+\.so", path.name) and path.resolve() == path,
            "ambiguous native source path",
        )
        modules[path.name] = {"source": str(path), **measured_hash(path)}
    require(
        all(modules[name]["sha256"] == value for name, value in wanted.items()),
        "required native hash differs",
    )
    view = Path(CACHE_ENV["AITER_JIT_DIR"])
    require(
        view.resolve() == view and view.is_relative_to(Path.home()),
        "native view escapes HOME",
    )
    receipt = EVIDENCE / "native-view.json"
    if create:
        require(
            not view.exists() and not receipt.exists(),
            "native view already initialized",
        )
        view.mkdir(mode=0o700, parents=True)
        (view / "build").mkdir(mode=0o555)
        (view / "build").chmod(0o555)
        for path in source:
            (view / path.name).symlink_to(path)
        view.chmod(0o555)
        save(
            receipt,
            {
                "source": str(NATIVE_SOURCE),
                "view": str(view),
                "aiter": policy["aiter"],
                "modules": modules,
            },
        )
    saved = read(receipt)
    require(
        saved
        == {
            "source": str(NATIVE_SOURCE),
            "view": str(view),
            "aiter": policy["aiter"],
            "modules": modules,
        },
        "native view receipt differs",
    )
    require(
        {p.name for p in view.iterdir()} == set(modules) | {"build"}
        and not list((view / "build").iterdir())
        and all(
            not path.is_symlink() and not path.stat().st_mode & 0o222
            for path in (view, view / "build")
        ),
        "native view or build directory is mutable",
    )
    for path in source:
        alias = view / path.name
        require(
            alias.is_symlink() and alias.resolve() == path and alias.samefile(path),
            "native alias differs",
        )
    return saved


def native_loads():
    snapshot = native_view()
    import aiter
    from aiter.jit import core
    from flydsl.utils import env as flydsl_env

    require(
        Path(aiter.__file__).resolve().parent == NATIVE_SOURCE.parent
        and Path(core.__file__).resolve() == NATIVE_SOURCE / "core.py",
        "AITER package origin differs",
    )
    require(
        os.environ.get("AITER_JIT_DIR") == core.get_user_jit_dir() == snapshot["view"],
        "native JIT selection differs",
    )
    architecture = core.get_gfx_runtime()
    require(
        architecture == "gfx950",
        "native architecture unavailable or different",
    )
    require(
        flydsl_env.runtime.cache_dir == CACHE_ENV["FLYDSL_RUNTIME_CACHE_DIR"]
        and not flydsl_env.runtime.run_only,
        "FlyDSL private normal-JIT environment differs",
    )
    loaded = {}
    for name in native_pins():
        path = Path(snapshot["view"]) / name
        module = core.get_module(Path(name).stem)
        selected = Path(core.build_ctypes_module(Path(name).stem))
        require(
            Path(module.__file__).samefile(path)
            and selected == path
            and selected.samefile(snapshot["modules"][name]["source"]),
            "native loader selected a different module",
        )
        ctypes.CDLL(str(selected))
        loaded[name] = {
            "python_origin": module.__file__,
            "ctypes_path": str(selected),
            "runtime_architecture": architecture,
            "source_sha256": snapshot["modules"][name]["sha256"],
        }
    return loaded


def verify_runtime_cache():
    require(
        all(os.environ.get(key) == value for key, value in CACHE_ENV.items())
        and "FLYDSL_RUNTIME_RUN_ONLY" not in os.environ,
        "runtime cache environment differs",
    )
    for key in CACHE_ENV:
        if key in {"LOGNAME", "USER", "AITER_JIT_DIR"}:
            continue
        path = Path(os.environ[key])
        require(
            path.resolve() == path
            and (key == "TMPDIR" or path.is_relative_to(Path.home())),
            "runtime cache escapes private storage",
        )
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(
            not os.statvfs(path).f_flag & (os.ST_RDONLY | os.ST_NOEXEC),
            "runtime cache filesystem is not writable/executable",
        )
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b"k31")
            probe.seek(0)
            require(probe.read(4) == b"k31", "runtime cache write/read failed")
    return {key: os.environ[key] for key in CACHE_ENV}


def verify_admission(root, initialize_native=False):
    admission, policy = read(root / "admission.json"), read(POLICY)
    require(
        admission["kind"] == "ci-container-v1" and admission["leg"] == "B-accuracy",
        "wrong admission type",
    )
    require(
        admission["image_manifest"] == policy["image"]
        and admission["docker"]["Image"] == policy["config"],
        "image identity differs",
    )
    info = admission["docker"]
    require(
        info["Id"] == admission["cid"] and re.fullmatch("[0-9a-f]{64}", info["Id"]),
        "container identity differs",
    )
    host = info["HostConfig"]
    cpus = set(admission["allocation"]["cpu_affinity"])
    require(
        cpus and cpus <= set(admission["allocation"]["permitted_cpu_affinity"]),
        "CPU admission exceeds observed permission",
    )
    require(
        parse_cpulist(host["CpusetCpus"]) == cpus and os.sched_getaffinity(0) == cpus,
        "container CPU affinity differs",
    )
    require(
        host["CapDrop"] == ["ALL"]
        and not host["CapAdd"]
        and any(value.startswith("no-new-privileges") for value in host["SecurityOpt"]),
        "container privileges differ",
    )
    require(
        host["ReadonlyRootfs"]
        and not host["Privileged"]
        and host["PidMode"] == ""
        and host["IpcMode"] == "private"
        and host["NetworkMode"] == "none"
        and not host["PortBindings"],
        "container isolation differs",
    )
    require(
        info["Config"]["User"] == f"{os.geteuid()}:{os.getegid()}",
        "container/output user differs",
    )
    require(
        {d["PathOnHost"] for d in host["Devices"]}
        == {d["path"] for d in admission["allocation"]["nodes"]},
        "container devices differ",
    )
    require(
        all(
            d["PathOnHost"] == d["PathInContainer"] and d["CgroupPermissions"] == "rw"
            for d in host["Devices"]
        ),
        "device mapping or permissions differ",
    )
    for path in (
        Path("/"),
        root / "source",
        root / "input",
        root / "data",
        Path("/models"),
    ):
        require(os.statvfs(path).f_flag & os.ST_RDONLY, f"writable input: {path}")
    for destination in ("/root", "/tmp"):
        mount = next(
            line
            for line in Path("/proc/self/mountinfo").read_text().splitlines()
            if line.split()[4] == destination
        )
        options = mount.split()[5].split(",")
        require(
            " - tmpfs " in mount and "rw" in options and "noexec" not in options,
            f"{destination} is not writable/executable private tmpfs",
        )
    require(
        not any(
            line.split()[4].startswith("/root/")
            for line in Path("/proc/self/mountinfo").read_text().splitlines()
        ),
        "nested HOME mount",
    )
    require(
        Path.home() == Path("/root")
        and (Path.home() / ".cache/atom").resolve().is_relative_to(Path.home()),
        "cache escapes private HOME",
    )
    manifest = root / "input/prepared-manifest.json"
    require(
        checksum(manifest) == admission["prepared_sha256"], "prepared manifest differs"
    )
    for name, wanted in read(manifest).items():
        require(checksum(root / "input" / name) == wanted, "prepared input differs")
    cache_environment = verify_runtime_cache()
    native_view(create=initialize_native)
    access = isolated_device_access(root, admission)
    admission["observation"] = {
        "namespaces": {
            name: os.readlink(f"/proc/self/ns/{name}")
            for name in ("pid", "ipc", "mnt", "net")
        },
        "mountinfo": Path("/proc/self/mountinfo").read_text(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "uid": os.geteuid(),
        "gid": os.getegid(),
        "groups": os.getgroups(),
        "cache_environment": cache_environment,
        "device_access": access,
    }
    return admission


def entry(work):
    from k31_phase0_assets import (
        check_model,
        check_runtime,
        emit_report,
        payload_backings,
        reserve_output,
    )

    inherited = {
        k: os.environ[k]
        for k in ("PATH", "LD_LIBRARY_PATH", "ROCM_PATH", "LANG", "LC_ALL")
        if k in os.environ
    }
    os.environ.clear()
    os.environ.update(
        inherited,
        **CACHE_ENV,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        HF_HOME="/root/.cache/huggingface",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="safe.directory",
        GIT_CONFIG_VALUE_0="/app/aiter-test",
        LEG="B-accuracy",
        TRACE_COMPILED="0",
        AITER_LOG_LEVEL="WARNING",
        NCCL_SOCKET_IFNAME="lo",
        MORI_SOCKET_IFNAME="lo",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MAX_JOBS="1",
        GPU_PREFLIGHT_KILL_DOCKER="0",
        GPU_PREFLIGHT_KILL_KFD="0",
    )
    require(
        shutil.which("lm_eval") == "/opt/venv/bin/lm_eval",
        "pinned lm_eval executable unavailable; refuse install",
    )
    verify_admission(Path("/app/atom-3577"), initialize_native=True)
    shutil.copytree(work / "data/hf", Path(os.environ["HF_HOME"]))
    report = {
        "kind": "ci-current-assets-v1",
        "models": [],
        "runtime": {},
        "status": "failed",
    }
    stream = None
    try:
        stream = reserve_output(EVIDENCE / "current-assets.json", report)
        views = read(work / "inputs/seed/C/payload/composition-manifest.json")["views"]
        blocked = payload_backings(
            next(v for v in views if v["role"] == "target"), Path("/models")
        )
        check_runtime(
            read(work / "inputs/seed/C/payload/pins.json"),
            read(CONTROL / ".github/validation/k31-phase0.json"),
            report["runtime"],
            blocked,
        )
        for view in views:
            result = {}
            report["models"].append(result)
            check_model(view, Path("/models"), result, blocked)
        telemetry(EVIDENCE / "preflight.json")
        subprocess.run(
            ["bash", str(CONTROL / ".github/scripts/gpu_preflight_check.sh")],
            check=True,
            timeout=120,
        )
        subprocess.run(
            [sys.executable, __file__, "map", str(EVIDENCE / "device-map.json")],
            check=True,
            timeout=60,
        )
        report["passed"] = True
        report["status"] = "passed"
    finally:
        emit_report(stream, report)
    os.execvpe(
        "bash", ["bash", "/app/atom-3577/input/run_3577_readonly.sh"], os.environ
    )


def cleanup(cid, call=command):
    require(re.fullmatch("[0-9a-f]{64}", cid), "invalid cleanup CID")
    stop = call(["docker", "stop", "--time", "10", cid])
    remove = call(["docker", "rm", "--force", cid])
    inventory = call(["docker", "ps", "--all", "--quiet", "--no-trunc"])
    absent = (
        inventory["returncode"] == 0 and cid not in inventory["stdout"].splitlines()
    )
    return {
        "cid": cid,
        "stop": stop,
        "remove": remove,
        "inventory": inventory,
        "absent": absent,
        "passed": remove["returncode"] == 0 and absent,
    }


def container(kind, work):
    policy, allocated = read(POLICY), read(work / "allocation.json")
    output, cid = work / "output", None
    report = {"kind": kind, "started_epoch": time.time(), "status": "failed"}
    owner = f"{output.stat().st_uid}:{output.stat().st_gid}"
    name = (
        f"k31-b-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-{kind}"
    )
    cid_path = output / f"{kind}.cid"
    tmpfs_access = "rw,exec,nosuid,nodev" if kind == "model" else "rw,nosuid,nodev"
    flags = [
        "--name",
        name,
        "--cidfile",
        str(cid_path),
        "--pull=never",
        "--read-only",
        "--user",
        owner,
        "--cpuset-cpus=" + ",".join(str(cpu) for cpu in allocated["cpu_affinity"]),
        "--ipc=private",
        "--network",
        "bridge" if kind == "dataset" else "none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        f"/root:{tmpfs_access},size=16g,mode=0700,uid={output.stat().st_uid},gid={output.stat().st_gid}",
        "--tmpfs",
        f"/tmp:{tmpfs_access},size=16g,mode=1777",
        "--shm-size=16g",
        "--mount",
        f"type=bind,src={CONTROL},dst=/control,readonly",
        "--mount",
        f"type=bind,src={work},dst=/job,readonly",
        "--mount",
        f"type=bind,src={output},dst={EVIDENCE}",
        "--entrypoint",
        "/opt/venv/bin/python",
    ]
    if kind == "dataset":
        flags += [
            "--cpus=4",
            "--memory=4g",
            "-e",
            "HF_HOME=/job/data/hf",
            "-e",
            "HF_HUB_OFFLINE=0",
            "-e",
            "HF_DATASETS_OFFLINE=0",
            "-e",
            "HF_ENDPOINT=https://huggingface.co",
            "--mount",
            f"type=bind,src={work / 'data'},dst=/job/data",
        ]
    else:
        flags += [f"--device={d['path']}:{d['path']}:rw" for d in allocated["nodes"]]
        flags += [f"--group-add={gid}" for gid in sorted(device_groups(allocated))]
    if kind == "model":
        for source, target in [
            (work / "source", "source"),
            (work / "input", "input"),
            (work / "data", "data"),
        ]:
            flags += [
                "--mount",
                f"type=bind,src={source},dst=/app/atom-3577/{target},readonly",
            ]
        (work / "admission.json").touch(exist_ok=False)
        flags += [
            "--mount",
            f"type=bind,src={work / 'admission.json'},dst=/app/atom-3577/admission.json,readonly",
            "--mount",
            "type=bind,src=/models,dst=/models,readonly",
            "--ulimit=memlock=-1:-1",
            "--mount",
            f"type=bind,src={work / 'source/scripts'},dst={EVIDENCE}/work/scripts,readonly",
        ]
    mode = {"dataset": "dataset", "model": "entry", "postflight": "telemetry"}[kind]
    argument = str(EVIDENCE / "postflight.json") if kind == "postflight" else "/job"
    try:
        created = command(
            [
                "docker",
                "create",
                *flags,
                policy["image"],
                "-B",
                "/control/.github/scripts/k31_b_accuracy.py",
                mode,
                argument,
            ]
        )
        report["create"] = created
        if cid_path.is_file():
            candidate = cid_path.read_text().strip()
            require(
                re.fullmatch("[0-9a-f]{64}", candidate), "missing exact container ID"
            )
            cid = candidate
        require(
            created["returncode"] == 0 and cid == created["stdout"].strip(),
            "container creation failed",
        )
        inspected = command(["docker", "inspect", cid])
        require(inspected["returncode"] == 0, "container inspection failed")
        info = json.loads(inspected["stdout"])[0]
        require(info["Image"] == policy["config"], "image configuration differs")
        if kind == "model":
            admission = {
                "kind": "ci-container-v1",
                "leg": "B-accuracy",
                "cid": cid,
                "docker": info,
                "allocation": allocated,
                "image_manifest": policy["image"],
                "prepared_sha256": checksum(work / "input/prepared-manifest.json"),
            }
            admission.update(
                {
                    name + "_sha256": value["sha256"]
                    for name, value in read(work / "input/preuse-bindings.json").items()
                }
            )
            save(work / "admission.json", admission)
        with (output / f"{kind}.log").open("xb") as log:
            try:
                report["attach_exit"] = subprocess.run(
                    ["docker", "start", "--attach", cid],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=policy["seconds"][kind],
                    check=False,
                ).returncode
            except subprocess.TimeoutExpired:
                report["attach_exit"] = 124
        final = command(["docker", "inspect", cid])
        report["final_inspect"] = final
        require(final["returncode"] == 0, "final container state unavailable")
        state = json.loads(final["stdout"])[0]["State"]
        report["state"] = state
        require(
            report["attach_exit"] == 0
            and not state["Running"]
            and not state["OOMKilled"]
            and state["ExitCode"] == 0,
            "container execution failed",
        )
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 -- retain failure before CID cleanup
        report["error"] = repr(error)
    finally:
        if cid:
            report["cleanup"] = cleanup(cid)
            if not report["cleanup"]["passed"]:
                report["status"] = "failed"
        report["finished_epoch"] = time.time()
        save(output / f"{kind}-container.json", report)
    return report


def execute(work):
    dataset = container("dataset", work)
    require(dataset["status"] == "passed", "dataset preparation failed")
    model = container("model", work)
    postflight = (
        container("postflight", work)
        if model.get("cleanup", {}).get("passed")
        else {
            "kind": "postflight",
            "status": "unobserved",
            "reason": "model container cleanup unresolved",
        }
    )
    save(
        work / "output/container-lifecycle.json",
        {"kind": "ci-container-lifecycle-v1", "model": model, "postflight": postflight},
    )
    subprocess.run(
        [sys.executable, str(work / "input/check_results.py"), str(work / "output")],
        check=True,
    )


def recover(work):
    failures = []
    for kind in ("dataset", "model", "postflight"):
        path = work / "output" / f"{kind}.cid"
        record = work / "output" / f"{kind}-container.json"
        try:
            previous = read(record)
        except (OSError, ValueError):
            previous = {}
        if path.is_file() and not previous.get("cleanup", {}).get("passed"):
            result = cleanup(path.read_text().strip())
            save(work / "output" / f"{kind}-recovery.json", result)
            if not result["passed"]:
                failures.append(kind)
    require(not failures, f"CI operator attention: cleanup unresolved for {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "execute",
            "recover",
            "dataset",
            "entry",
            "map",
            "telemetry",
            "access-probe",
        ),
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "execute":

        def stopped(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stopped)
        signal.signal(signal.SIGINT, stopped)
    {
        "prepare": prepare,
        "execute": execute,
        "dataset": stage_dataset,
        "entry": entry,
        "map": device_map,
        "telemetry": telemetry,
        "access-probe": device_access_child,
        "recover": recover,
    }[args.mode](args.path.resolve())
