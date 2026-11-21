"""Temporary, fixed B-accuracy route. Docker owns each dedicated container."""

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from atomesh.numa_exec import parse_cpulist
from k31_phase0_inputs import digest, require, verify

CONTROL = Path(__file__).resolve().parents[2]
POLICY = CONTROL / ".github/validation/k31-b-accuracy.json"
EVIDENCE = Path("/app/atom-3577/evidence/B-accuracy")


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
    require(
        set(groups) <= {node["gid"] for node in result},
        "declared group not observed on allocated devices",
    )
    return {
        "nodes": result,
        "declared_cpu_set": declared_cpus,
        "permitted_cpu_affinity": sorted(permitted),
        "cpu_affinity": cpus,
    }


def captured_allocation(path, output):
    limit = 16384
    report = {"kind": "ci-allocation-input-v1", "path": str(path), "phase": "read"}
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
        result = allocation(text, permitted_cpus=report["permitted_cpu_affinity"])
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
    require(
        all(
            float(c["GPU use (%)"]) == 0
            and int(c["VRAM Total Used Memory (B)"]) < 512 * 1024**2
            and int(c["VRAM Total Memory (B)"]) - int(c["VRAM Total Used Memory (B)"])
            >= 3 * 1024**3
            for c in cards.values()
        ),
        "allocated GPUs are not idle",
    )
    report["passed"] = True
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
    save(
        output,
        {
            "devices": devices,
            "hip": torch.version.hip,
            "scope": "visible devices; not an observed model-rank map",
        },
    )


def verify_admission(root):
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
    require(
        {d["gid"] for d in admission["allocation"]["nodes"]}
        <= set(os.getgroups()) | {os.getegid()},
        "device groups missing",
    )
    for path in (
        Path("/"),
        root / "source",
        root / "input",
        root / "data",
        Path("/models"),
    ):
        require(os.statvfs(path).f_flag & os.ST_RDONLY, f"writable input: {path}")
    mount = next(
        line
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if line.split()[4] == "/root"
    )
    require(
        " - tmpfs " in mount and "rw" in mount.split()[5].split(","),
        "HOME is not private tmpfs",
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
        HOME="/root",
        TMPDIR="/tmp",
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
    verify_admission(Path("/app/atom-3577"))
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
        f"/root:rw,nosuid,nodev,size=16g,mode=0700,uid={output.stat().st_uid},gid={output.stat().st_gid}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=16g,mode=1777",
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
        flags += [
            f"--group-add={gid}"
            for gid in sorted({d["gid"] for d in allocated["nodes"]})
        ]
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
        "recover": recover,
    }[args.mode](args.path.resolve())
