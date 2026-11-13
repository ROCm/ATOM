# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What this machine has: GPUs, who is using them, and how ATOM can run here.

Everything is read from sysfs/procfs so it works without rocm-smi, amd-smi,
torch or a container. GPU ``index`` is the HIP device index -- the order of
GPU nodes in the KFD topology (verified against torch's pci_bus_id on
MI355X) -- which is what ``HIP_VISIBLE_DEVICES`` expects and is NOT
necessarily rocm-smi's numbering.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")
KFD_PROC = Path("/sys/class/kfd/kfd/proc")
DRM = Path("/sys/class/drm")

# An idle MI355X reports ~283 MiB; anything well above that is a tenant.
FREE_VRAM_MIB = 1024

# (arch, VRAM GiB lower bound) -> product, used when amd-smi is unavailable.
_PRODUCT_HINTS = (
    ("gfx950", 200, "AMD Instinct MI355X/MI350X"),
    ("gfx942", 230, "AMD Instinct MI325X"),
    ("gfx942", 150, "AMD Instinct MI300X/MI308X"),
    ("gfx942", 0, "AMD Instinct MI300A"),
    ("gfx90a", 0, "AMD Instinct MI250X/MI210"),
    ("gfx1250", 0, "AMD Instinct MI455X (gfx1250)"),
    ("gfx1201", 0, "AMD Radeon RX 9070 / AI PRO R9700"),
    ("gfx1151", 0, "AMD Ryzen AI Max (Strix Halo)"),
    ("gfx1100", 0, "AMD Radeon RX 7900 / PRO W7900"),
)


@dataclass
class GpuProcess:
    pid: int
    user: str
    command: str
    vram_mib: int


@dataclass
class Gpu:
    index: int
    arch: str
    product: str
    pci_bus: str
    render_node: str
    vram_total_mib: int
    vram_used_mib: int
    busy_percent: int | None
    processes: list[GpuProcess] = field(default_factory=list)
    kfd_gpu_id: int = 0

    @property
    def free(self) -> bool:
        return self.vram_used_mib < FREE_VRAM_MIB and not self.processes


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return default


def _props(node: Path) -> dict[str, int]:
    out = {}
    for line in _read(node / "properties").splitlines():
        key, _, value = line.partition(" ")
        if value.strip().lstrip("-").isdigit():
            out[key] = int(value)
    return out


def gfx_name(target_version: int) -> str:
    """KFD ``gfx_target_version`` (e.g. 90500, 120001, 90010) -> gfx950 / gfx1201 / gfx90a."""
    major, rest = divmod(target_version, 10000)
    minor, stepping = divmod(rest, 100)
    return f"gfx{major}{minor}{stepping:x}"


def _product_hint(arch: str, vram_gib: float) -> str:
    for hint_arch, min_gib, name in _PRODUCT_HINTS:
        if arch == hint_arch and vram_gib >= min_gib:
            return name
    return f"AMD GPU ({arch})"


def _amd_smi_names() -> dict[str, str]:
    """PCI BDF (lower case) -> market name, when amd-smi is installed."""
    if not shutil.which("amd-smi"):
        return {}
    try:
        res = subprocess.run(
            ["amd-smi", "static", "--asic", "--bus", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        data = json.loads(res.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    if isinstance(data, dict):
        data = data.get("gpu_data", [])
    names = {}
    for entry in data if isinstance(data, list) else []:
        bdf = str((entry.get("bus") or {}).get("bdf", "")).lower()
        name = (entry.get("asic") or {}).get("market_name")
        if bdf and name:
            names[bdf] = name
    return names


def _process_owner(pid: int) -> tuple[str, str]:
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
        user = pwd.getpwuid(uid).pw_name
    except (OSError, KeyError):
        user = "?"
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        cmd = raw.replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = ""
    return user, cmd[:160]


def _gpu_processes() -> dict[int, list[GpuProcess]]:
    """kfd gpu_id -> processes holding VRAM on it (needs no privileges)."""
    by_gpu: dict[int, list[GpuProcess]] = {}
    if not KFD_PROC.is_dir():
        return by_gpu
    for proc_dir in KFD_PROC.iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        owner = None
        for vram_file in proc_dir.glob("vram_*"):
            try:
                used = int(_read(vram_file, "0"))
            except ValueError:
                continue
            if used <= 0:
                continue
            if owner is None:
                owner = _process_owner(pid)
            gpu_id = int(vram_file.name.split("_", 1)[1])
            by_gpu.setdefault(gpu_id, []).append(
                GpuProcess(pid, owner[0], owner[1], used >> 20)
            )
    return by_gpu


def list_gpus(with_names: bool = True) -> list[Gpu]:
    if not KFD_TOPOLOGY.is_dir():
        return []
    nodes = []
    for node in KFD_TOPOLOGY.iterdir():
        if node.name.isdigit():
            props = _props(node)
            if props.get("gfx_target_version", 0) and props.get("simd_count", 0):
                nodes.append((int(node.name), node, props))
    nodes.sort()
    names = _amd_smi_names() if with_names else {}
    procs = _gpu_processes()
    gpus = []
    for index, (_, node, props) in enumerate(nodes):
        loc = props.get("location_id", 0)
        bdf = (
            f"{props.get('domain', 0):04x}:{(loc >> 8) & 0xFF:02x}:"
            f"{(loc >> 3) & 0x1F:02x}.{loc & 0x7}"
        )
        minor = props.get("drm_render_minor", -1)
        dev = DRM / f"renderD{minor}" / "device"
        total = int(_read(dev / "mem_info_vram_total", "0") or 0) >> 20
        used = int(_read(dev / "mem_info_vram_used", "0") or 0) >> 20
        busy = _read(dev / "gpu_busy_percent")
        arch = gfx_name(props["gfx_target_version"])
        gpu_id = int(_read(node / "gpu_id", "0") or 0)
        gpus.append(
            Gpu(
                index=index,
                arch=arch,
                product=names.get(bdf) or _product_hint(arch, total / 1024),
                pci_bus=bdf,
                render_node=f"/dev/dri/renderD{minor}",
                vram_total_mib=total,
                vram_used_mib=used,
                busy_percent=int(busy) if busy.isdigit() else None,
                processes=procs.get(gpu_id, []),
                kfd_gpu_id=gpu_id,
            )
        )
    return gpus


def running_atom_servers() -> list[dict]:
    """ATOM OpenAI servers visible in this PID namespace, anyone's."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in argv if a]
        if not any(a.startswith("atom.entrypoints.openai_server") for a in args):
            continue
        user, _ = _process_owner(int(proc.name))

        def flag(*names: str, args=args) -> str | None:
            for i, a in enumerate(args):
                for name in names:
                    if a == name and i + 1 < len(args):
                        return args[i + 1]
                    if a.startswith(name + "="):
                        return a.split("=", 1)[1]
            return None

        found.append(
            {
                "pid": int(proc.name),
                "user": user,
                "model": flag("--model"),
                "port": int(flag("--server-port") or 8000),
                "tp": flag("-tp", "--tensor-parallel-size"),
            }
        )
    # A server's launcher and its engine process share the command line;
    # report one entry per port (the lowest pid, i.e. the launcher).
    by_port: dict[int, dict] = {}
    for s in sorted(found, key=lambda x: x["pid"]):
        by_port.setdefault(s["port"], s)
    return list(by_port.values())


def _docker_info() -> dict:
    info: dict = {"available": bool(shutil.which("docker"))}
    if not info["available"]:
        return info
    try:
        res = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        info["server_version"] = res.stdout.strip() or None
        info["usable"] = res.returncode == 0
        if res.returncode != 0:
            info["error"] = res.stderr.strip()[-300:]
            return info
        res = subprocess.run(
            [
                "docker",
                "images",
                "--format",
                "{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        images = []
        for line in res.stdout.splitlines():
            ref, created, size = (line.split("\t") + ["", ""])[:3]
            if re.match(r"(docker\.io/)?rocm/atom", ref):
                images.append({"image": ref, "created": created[:19], "size": size})
        images.sort(key=lambda x: x["created"], reverse=True)
        info["atom_images"] = images[:8]
    except (OSError, subprocess.SubprocessError) as e:
        info["usable"] = False
        info["error"] = str(e)
    return info


def atom_runnable_here() -> bool:
    """Can this interpreter launch ``atom.entrypoints.openai_server``?"""
    return all(importlib.util.find_spec(m) for m in ("torch", "aiter"))


def _meminfo_gib() -> float | None:
    for line in _read(Path("/proc/meminfo")).splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 2**20, 1)
    return None


def inspect_system(include_docker: bool = True) -> dict:
    gpus = list_gpus()
    try:
        atom_version = importlib.metadata.version("atom")
    except importlib.metadata.PackageNotFoundError:
        atom_version = None
    in_container = Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
    report: dict = {
        "host": os.uname().nodename,
        "in_container": in_container,
        "rocm_version": _read(Path("/opt/rocm/.info/version")) or None,
        "python": sys.version.split()[0],
        "atom_version": atom_version,
        "atom_runnable_here": atom_runnable_here(),
        "cpus": os.cpu_count(),
        "numa_nodes": len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))),
        "host_memory_gib": _meminfo_gib(),
        "gpu_count": len(gpus),
        "free_gpus": [g.index for g in gpus if g.free],
        "gpus": [
            {
                **{
                    k: v
                    for k, v in asdict(g).items()
                    if k not in ("processes", "kfd_gpu_id")
                },
                "free": g.free,
                "processes": [asdict(p) for p in g.processes],
            }
            for g in gpus
        ],
        "atom_servers_running": running_atom_servers(),
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
    }
    if include_docker:
        report["docker"] = _docker_info()
    if not gpus:
        report["warning"] = (
            "No AMD GPU visible through /sys/class/kfd. Inside a container, "
            "start it with --device=/dev/kfd --device=/dev/dri."
        )
    return report
