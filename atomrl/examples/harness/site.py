# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Where the examples run: the hosts, the data, and how to reach each host.

Without a site file the runner uses this machine alone, with ``DATA_ROOT`` from
the environment. A site file (``--site``, ``$ATOMRL_SITE`` or
``examples/site.yaml``) describes more; ``examples/site.example.yaml`` is the
annotated template.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .manifest import EXAMPLES_DIR, ManifestError

RC_MARK = "__ATOMRL_RC="
DEFAULT_EXEC = ("ssh", "-o", "BatchMode=yes", "{host}", "bash -c {qcmd}")
_SITE_KEYS = {
    "data_root", "docker", "container", "hosts", "exec", "host_prelude", "output_filter",
    "rdma", "env", "mounts", "images", "results_dir", "idle_vram_gb",
}  # fmt: skip
_HOST_KEYS = {"name", "address", "ip"}
_RDMA_KEYS = {"interface", "hca", "gid_index", "env"}


@dataclasses.dataclass(frozen=True)
class Result:
    rc: int
    out: str
    err: str = ""

    @property
    def ok(self) -> bool:
        return self.rc == 0


@dataclasses.dataclass(frozen=True)
class Rdma:
    interface: str
    hca: str
    gid_index: int
    env: dict[str, str]

    def nccl_env(self) -> dict[str, str]:
        env = {
            "NCCL_SOCKET_IFNAME": self.interface,
            "NCCL_IB_HCA": self.hca,
            "NCCL_IB_GID_INDEX": str(self.gid_index),
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "3",
        }
        env.update(self.env)
        return env


@dataclasses.dataclass
class Host:
    name: str
    address: str
    ip: str | None = None
    exec_template: tuple[str, ...] = DEFAULT_EXEC
    prelude: str = ""
    output_filter: tuple[str, ...] = ()

    @property
    def is_local(self) -> bool:
        return self.address == "local"

    def argv(self, cmd: str) -> list[str]:
        if self.is_local:
            return ["bash", "-c", cmd]
        # An EXIT trap reports the status however the command ends, `exit` included,
        # on a line of its own even when the output does not end with a newline.
        wrapped = f"trap 'printf \"\\n{RC_MARK}%s\\n\" $?' EXIT\n{cmd}"
        values = {"host": self.address, "cmd": wrapped, "qcmd": shlex.quote(wrapped)}
        return [part.format(**values) for part in self.exec_template]

    def run(self, cmd: str, timeout: float | None = None) -> Result:
        if self.prelude:
            cmd = f"{self.prelude}\n{cmd}"
        try:
            proc = subprocess.run(
                self.argv(cmd),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            return Result(124, out, f"timed out after {timeout}s on {self.name}")
        out, err = proc.stdout, proc.stderr
        if self.output_filter:
            pats = [re.compile(p) for p in self.output_filter]
            out, err = _drop_lines(out, pats), _drop_lines(err, pats)
        if self.is_local:
            return Result(proc.returncode, out, err)
        return _split_rc(out, err, proc.returncode, self.name)


def _drop_lines(text: str, pats: list[re.Pattern]) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(ln for ln in lines if not any(p.search(ln) for p in pats))


def _split_rc(out: str, err: str, transport_rc: int, name: str) -> Result:
    idx = out.rfind("\n" + RC_MARK)
    if idx >= 0:
        tail = out[idx + 1 + len(RC_MARK) :].split()
        if tail and tail[0].lstrip("-").isdigit():
            return Result(int(tail[0]), out[:idx], err)
    detail = f"no exit status came back from {name} (transport exit {transport_rc})"
    return Result(transport_rc or 255, out, f"{err}\n{detail}".strip())


@dataclasses.dataclass
class Site:
    data_root: str | None
    docker: str = "docker"
    container: str = "atomrl-ex"
    hosts: list[Host] = dataclasses.field(default_factory=list)
    rdma: Rdma | None = None
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    mounts: tuple[str, ...] = ()
    images: dict[str, Any] = dataclasses.field(default_factory=dict)
    results_dir: str | None = None
    idle_vram_gb: float = 2.0
    source: Path | None = None

    @property
    def head(self) -> Host:
        return self.hosts[0]


def _host(i: int, data: dict, template: tuple[str, ...], prelude: str, filt) -> Host:
    extra = set(data) - _HOST_KEYS
    if extra:
        raise ManifestError(f"site hosts[{i}]: unknown key(s) {sorted(extra)}")
    address = str(data.get("address", "local"))
    return Host(
        name=str(data.get("name", address)),
        address=address,
        ip=str(data["ip"]) if data.get("ip") else None,
        exec_template=template,
        prelude=prelude,
        output_filter=filt,
    )


def load_site(path: str | Path | None = None) -> Site:
    """Load the site file, or describe this machine alone if there is none."""
    candidates = [path, os.environ.get("ATOMRL_SITE"), EXAMPLES_DIR / "site.yaml"]
    source = next((Path(p) for p in candidates if p and Path(p).is_file()), None)
    if path and source is None:
        raise ManifestError(f"site file not found: {path}")
    data: dict[str, Any] = {}
    if source is not None:
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        extra = set(data) - _SITE_KEYS
        if extra:
            raise ManifestError(f"{source}: unknown key(s) {sorted(extra)}")
    template = tuple(str(p) for p in data.get("exec") or DEFAULT_EXEC)
    prelude = str(data.get("host_prelude") or "").strip()
    filt = tuple(str(p) for p in data.get("output_filter") or ())
    hosts = [
        _host(i, h, template, prelude, filt)
        for i, h in enumerate(data.get("hosts") or [{}])
    ]
    rdma = data.get("rdma")
    if rdma is not None:
        extra = set(rdma) - _RDMA_KEYS
        if extra:
            raise ManifestError(f"site rdma: unknown key(s) {sorted(extra)}")
        rdma = Rdma(
            interface=str(rdma["interface"]),
            hca=str(rdma["hca"]),
            gid_index=int(rdma.get("gid_index", 3)),
            env={str(k): str(v) for k, v in (rdma.get("env") or {}).items()},
        )
    return Site(
        data_root=data.get("data_root") or os.environ.get("DATA_ROOT"),
        docker=str(data.get("docker") or os.environ.get("DOCKER") or "docker"),
        container=str(
            data.get("container") or os.environ.get("ATOMRL_CONTAINER") or "atomrl-ex"
        ),
        hosts=hosts,
        rdma=rdma,
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        mounts=tuple(str(m) for m in data.get("mounts") or ()),
        images=dict(data.get("images") or {}),
        results_dir=data.get("results_dir"),
        idle_vram_gb=float(data.get("idle_vram_gb", 2.0)),
        source=source,
    )
