# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Run one example on its hosts: container, preflight, launch, follow, judge."""

from __future__ import annotations

import dataclasses
import os
import re
import secrets
import shlex
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import checks
from .manifest import (
    ATOMRL_DIR,
    Catalog,
    Checks,
    Example,
    Image,
    ManifestError,
    Variant,
)
from .site import Host, Result, Site

COMPILE_CACHES = (
    "/tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root"
)
ATOM_RECEIVER = "atom/rollout/rdma_weight_receiver.py"
HIGHLIGHT = re.compile(
    r"RLTrainer\.setup|filter_groups round|callbacks.*\bstep=\d+ |Traceback|OutOfMemory|CUDA error"
    r"|HSA_STATUS|=+ .*(passed|failed).* in [\d.]+s|Ray runtime started"
)
READ_CAP = 8 << 20
FETCH_LIMIT = 64 << 20


class Skip(Exception):
    """The example cannot run here (hardware, data or capability missing)."""


class RunError(Exception):
    """The environment is not in a state to run the example."""


@dataclasses.dataclass
class Options:
    full: bool = False
    steps: int | None = None
    overrides: tuple[str, ...] = ()
    check: bool = True
    dry_run: bool = False
    prepare: bool = False
    force: bool = False
    keep_cache: bool = False
    keep_container: bool = False
    verbose: bool = False
    atomrl_src: str = str(ATOMRL_DIR)
    atom_src: str | None = None
    images: dict[str, str] = dataclasses.field(default_factory=dict)
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    poll_s: float = 10.0


@dataclasses.dataclass
class Outcome:
    example: Example
    status: str  # PASS | FAIL | SKIP | ERROR | XFAIL | XPASS
    reason: str = ""
    variant: str = "smoke"
    steps: int | None = None
    started: float = 0.0
    duration_s: float = 0.0
    exit_code: int | None = None
    verdict: checks.Verdict | None = None
    log: str | None = None
    files: dict[str, str] = dataclasses.field(default_factory=dict)
    hosts: list[str] = dataclasses.field(default_factory=list)
    image: str = ""
    image_id: str = ""
    atom_commit: str = ""


def _q(s: str) -> str:
    return shlex.quote(s)


def _clean(text: str) -> str:
    return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


class _Ctx(dict):
    def __missing__(self, key: str) -> Any:
        raise ManifestError(f"unknown placeholder {{{key}}}")


@dataclasses.dataclass(frozen=True)
class _HostView:
    name: str
    ip: str | None


class Runner:
    def __init__(
        self,
        catalog: Catalog,
        site: Site,
        opts: Options,
        results_dir: Path | None,
        echo: Callable[[str], None] = print,
    ) -> None:
        self.catalog = catalog
        self.site = site
        self.opts = opts
        self.results_dir = results_dir
        self.echo = echo

    # ------------------------------------------------------------------ helpers
    def image(self, key: str) -> Image:
        """The image for ``key``: the catalog's, then the site's, then the CLI's.

        A string replaces the reference; ``@other`` swaps in the whole definition
        of image ``other`` (reference, mounts and environment); a mapping (site
        file only) replaces individual fields.
        """
        image = self.catalog.images[key]
        for value in (self.site.images.get(key), self.opts.images.get(key)):
            if isinstance(value, str) and value.startswith("@"):
                other = self.catalog.images.get(value[1:])
                if other is None:
                    raise ManifestError(
                        f"--image {key}={value}: no image {value[1:]!r}"
                    )
                image = dataclasses.replace(other, key=key)
            elif isinstance(value, str):
                image = dataclasses.replace(image, ref=value)
            elif isinstance(value, dict):
                image = dataclasses.replace(image, **value)
        return image

    def container(self, image: Image) -> str:
        return f"{self.site.container}-{image.key}"

    def docker(
        self, host: Host, args: list[str], timeout: float | None = 300
    ) -> Result:
        return host.run(f"{self.site.docker} {shlex.join(args)}", timeout=timeout)

    def _exec(
        self,
        host: Host,
        name: str,
        script: str,
        env: dict[str, str] | None = None,
        detach: bool = False,
        timeout: float | None = 300,
    ) -> Result:
        args = ["exec"] + (["-d"] if detach else [])
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        return self.docker(host, args + [name, "bash", "-lc", script], timeout=timeout)

    def _hosts(self, ex: Example) -> list[Host]:
        if len(self.site.hosts) < ex.nodes:
            raise Skip(f"needs {ex.nodes} hosts, the site has {len(self.site.hosts)}")
        hosts = self.site.hosts[: ex.nodes]
        if ex.nodes > 1:
            missing = [h.name for h in hosts if not h.ip]
            if missing:
                raise RunError(f"multi-node example: site hosts {missing} have no ip")
            if self.site.rdma is None:
                raise RunError("multi-node example: the site file has no rdma section")
        return hosts

    def context(
        self, ex: Example, variant: Variant, image: Image, log: str, steps
    ) -> _Ctx:
        hosts = [_HostView(h.name, h.ip) for h in self.site.hosts]
        ctx = _Ctx(
            atomrl=image.atomrl_mount,
            examples=f"{image.atomrl_mount}/examples",
            ex_dir=f"{image.atomrl_mount}/examples/{ex.family}".rstrip("/."),
            data_root=self.site.data_root,
            log=log,
            steps="" if steps is None else str(steps),
            config=variant.config or "",
            nodes=ex.nodes,
            gpus_per_node=ex.gpus_per_node,
            hosts=hosts,
            head_ip=hosts[0].ip if hosts else None,
            rdma=self.site.rdma,
            id=ex.id,
        )
        overrides = list(variant.overrides) + list(self.opts.overrides)
        ctx["overrides"] = " ".join(o.format_map(ctx) for o in overrides)
        return ctx

    def env(
        self, ex: Example, variant: Variant, image: Image, ctx: _Ctx, steps
    ) -> dict:
        env = dict(image.env)
        env["DATA_ROOT"] = str(self.site.data_root)
        env.update(self.site.env)
        for k, v in variant.env.items():
            env[k] = v.format_map(ctx)
        env.update(self.opts.env)
        env["LOG"] = ctx["log"]
        if steps is not None:
            env["STEPS"] = str(steps)
        if ctx["overrides"]:
            env["EXTRA_OVERRIDE"] = ctx["overrides"]
        if ex.nodes > 1:
            env.update(self.site.rdma.nccl_env())
            env["GPUS_PER_NODE"] = str(ex.gpus_per_node)
        if os.environ.get("WANDB_API_KEY"):
            env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
        return env

    def overrides(self, ex: Example, variant: Variant) -> Variant:
        extra = list(variant.overrides)
        joined = " ".join(extra + list(self.opts.overrides))
        # Long-run configs enable wandb; without a key the run would die after
        # setup, so switch it off up front.
        if (
            self.opts.full
            and not os.environ.get("WANDB_API_KEY")
            and "logger.wandb_enabled" not in joined
        ):
            extra.append("logger.wandb_enabled=false")
        return dataclasses.replace(variant, overrides=tuple(extra))

    def create_args(self, image: Image, name: str, infiniband: bool) -> list[str]:
        dr = str(self.site.data_root)
        args = [
            "run", "-d", "--name", name, "--network=host", "--ipc=host",
            "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video",
            "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined", "--shm-size", "64G",
        ]  # fmt: skip
        if infiniband:
            args += [
                "--device=/dev/infiniband",
                "--cap-add=IPC_LOCK",
                "--ulimit",
                "memlock=-1",
            ]
        args += ["-v", f"{dr}:{dr}", "-e", f"DATA_ROOT={dr}"]
        for m in self.site.mounts:
            args += ["-v", f"{m}:{m}"]
        args += ["-v", f"{self.opts.atomrl_src}:{image.atomrl_mount}"]
        if self.opts.atom_src:
            if not image.atom_mount:
                raise RunError(f"image {image.key} has no atom_mount for --atom-src")
            args += ["-v", f"{self.opts.atom_src}:{image.atom_mount}"]
        return args + list(image.run_args) + [image.ref, "sleep", "infinity"]

    def expected_mounts(self, image: Image) -> dict[str, str]:
        dr = str(self.site.data_root)
        mounts = {dr: dr, image.atomrl_mount: self.opts.atomrl_src}
        mounts.update({m: m for m in self.site.mounts})
        if self.opts.atom_src and image.atom_mount:
            mounts[image.atom_mount] = self.opts.atom_src
        return {os.path.normpath(k): os.path.normpath(v) for k, v in mounts.items()}

    # ---------------------------------------------------------------- preflight
    def prepare_host(self, host: Host, image: Image, ex: Example) -> bool:
        """Get our container on ``host`` fresh and idle. Returns infiniband presence."""
        name = self.container(image)
        r = self.docker(host, ["image", "inspect", "--format", "{{.Id}}", image.ref])
        if not r.ok:
            hint = f" or build it: {image.build}" if image.build else ""
            raise Skip(
                f"image {image.ref} is not on {host.name}: `docker pull {image.ref}`{hint}"
            )
        r = self.docker(
            host,
            ["ps", "-a", "--filter", f"ancestor={image.ref}", "--format", "{{.Names}}"],
        )
        others = [n for n in r.out.split() if n != name]
        if others:
            if not self.opts.force:
                raise RunError(
                    f"{host.name}: other containers on {image.ref} may hold GPU memory: "
                    f"{' '.join(others)} (remove them, or pass --force to remove them)"
                )
            self.docker(host, ["rm", "-f", *others])
        infiniband = host.run("test -e /dev/infiniband").ok
        fmt = (
            "{{.Config.Image}}\n{{range .Mounts}}{{.Destination}}={{.Source}}\n{{end}}"
        )
        r = self.docker(host, ["container", "inspect", "--format", fmt, name])
        if r.ok:
            lines = r.out.strip().splitlines()
            pairs = (ln.partition("=") for ln in lines[1:] if "=" in ln)
            mounts = {os.path.normpath(a): os.path.normpath(b) for a, _, b in pairs}
            want = self.expected_mounts(image)
            stale = lines[:1] != [image.ref] or any(
                mounts.get(k) != v for k, v in want.items()
            )
            stale = stale or (
                image.atom_mount is not None
                and not self.opts.atom_src
                and os.path.normpath(image.atom_mount) in mounts
            )
            if stale:
                self.echo(
                    f"   {host.name}: container {name} has other mounts or image; recreating"
                )
                self.docker(host, ["rm", "-f", name])
                r = Result(1, "")
            else:
                # Ray workers keep ~85 GB per card after a clean run and are
                # invisible from inside; only a restart gives the memory back.
                rr = self.docker(host, ["restart", name], timeout=180)
                if not rr.ok:
                    raise RunError(
                        f"{host.name}: docker restart {name} failed: {rr.err.strip()}"
                    )
        if not r.ok:
            rr = self.docker(
                host, self.create_args(image, name, infiniband), timeout=600
            )
            if not rr.ok:
                raise RunError(
                    f"{host.name}: docker run failed: {rr.err.strip()[-500:]}"
                )
            self.echo(f"   {host.name}: container {name} created")
        time.sleep(5)
        self._idle_check(host, name)
        self._compile_cache(host, name, ex)
        return infiniband

    def _idle_check(self, host: Host, name: str) -> None:
        r = self._exec(
            host,
            name,
            'rocm-smi --showmeminfo vram 2>/dev/null | grep -i "Total Used Memory"',
        )
        used = [int(m) for m in re.findall(r"(\d+)\s*$", r.out, flags=re.MULTILINE)]
        if not used:
            self.echo(f"   {host.name}: WARNING could not read VRAM usage")
            return
        limit = self.site.idle_vram_gb * 1e9
        busy = [u for u in used if u > limit]
        self.echo(
            f"   {host.name}: {len(busy)}/{len(used)} GPUs busy, peak {max(used) / 1e9:.1f} GB"
        )
        if busy and not self.opts.force:
            raise RunError(
                f"{host.name}: {len(busy)} GPU(s) above {self.site.idle_vram_gb:g} GB in use "
                "(another job or tenant); pass --force to run anyway"
            )

    def _compile_cache(self, host: Host, name: str, ex: Example) -> None:
        # Cached ATOM graphs are keyed on nothing the runtime checks: reusing
        # graphs compiled for another precision or model crashes in AOTAutograd
        # during warmup rather than missing the cache.
        if not ex.compile_cache_key:
            return
        # Per host: the data root is often shared, the containers' caches are not.
        state = f"{self.site.data_root}/logs/.atomrl-compile-key-{host.name}-{name}"
        prev = host.run(f"cat {_q(state)} 2>/dev/null").out.strip()
        if prev and prev != ex.compile_cache_key and not self.opts.keep_cache:
            self.echo(
                f"   {host.name}: compile key {prev!r} -> {ex.compile_cache_key!r}; clearing"
            )
            self._exec(host, name, f"rm -rf {COMPILE_CACHES}")
        self._exec(
            host,
            name,
            f"mkdir -p {_q(os.path.dirname(state))} && echo {_q(ex.compile_cache_key)} > {_q(state)}",
        )

    def check_needs(
        self, host: Host, name: str, image: Image, ex: Example, ib: bool
    ) -> None:
        if "rdma" in ex.needs and not ib:
            raise Skip(f"{host.name} has no /dev/infiniband")
        if "atom_rdma" in ex.needs:
            path = f"{image.atom_mount}/{ATOM_RECEIVER}"
            if not self._exec(host, name, f"test -f {_q(path)}").ok:
                raise Skip(
                    "needs an ATOM with the RDMA weight receiver (ROCm/ATOM#3058); "
                    "the image's ATOM has none -- pass --atom-src <ATOM checkout>"
                )

    def missing_data(self, hosts: list[Host], ex: Example) -> list[str]:
        missing = []
        for host in hosts:
            for rel in ex.requires:
                path = rel if rel.startswith("/") else f"{self.site.data_root}/{rel}"
                if not host.run(f"test -e {_q(path)}").ok:
                    missing.append(f"{host.name}:{rel}")
        return missing

    def run_prepare(self, host: Host, image: Image, ex: Example) -> None:
        if not ex.prepare:
            raise Skip(f"{ex.id} has no prepare step")
        ctx = self.context(ex, ex.smoke, image, "", None)
        env = {"DATA_ROOT": str(self.site.data_root), **image.env, **self.site.env}
        self.echo(f"   {host.name}: preparing data: {ex.prepare.format_map(ctx)}")
        r = self._exec(
            host,
            self.container(image),
            ex.prepare.format_map(ctx),
            env,
            timeout=6 * 3600,
        )
        tail = "\n".join(_clean(r.out + r.err).strip().splitlines()[-15:])
        self.echo("\n".join(f"     {ln}" for ln in tail.splitlines()))
        if not r.ok:
            raise RunError(f"prepare step failed with exit {r.rc}")

    # ---------------------------------------------------------------- main path
    def run(self, ex: Example) -> Outcome:
        variant = self.overrides(ex, ex.variant(self.opts.full))
        steps = self.opts.steps or variant.steps
        out = Outcome(
            example=ex,
            status="ERROR",
            variant="full" if self.opts.full else "smoke",
            steps=steps,
            started=time.time(),
        )
        try:
            self._run(ex, variant, steps, out)
        except Skip as exc:
            out.status, out.reason = "SKIP", str(exc)
        except (RunError, ManifestError) as exc:
            out.status, out.reason = "ERROR", str(exc)
        out.duration_s = time.time() - out.started
        if ex.known_issue and out.status in ("FAIL", "ERROR"):
            out.status, out.reason = (
                "XFAIL",
                f"known issue: {ex.known_issue} | {out.reason}",
            )
        elif ex.known_issue and out.status == "PASS":
            out.status, out.reason = (
                "XPASS",
                "passed despite its known_issue; remove it",
            )
        return out

    def _run(self, ex: Example, variant: Variant, steps, out: Outcome) -> None:
        image = self.image(ex.image)
        out.image = image.ref
        hosts = self._hosts(ex)
        out.hosts = [h.name for h in hosts]
        if "rdma" in ex.needs and self.site.rdma is None:
            raise Skip("needs the site file's rdma section (NIC, HCA, GID index)")
        if not self.site.data_root:
            raise RunError("no data root: set DATA_ROOT or data_root in the site file")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log = f"{self.site.data_root}/logs/atomrl-{ex.id}-{stamp}.log"
        outer = log[: -len(".log")] + ".launcher.log"
        out.log = log
        ctx = self.context(ex, variant, image, log, steps)
        env = self.env(ex, variant, image, ctx, steps)
        command = ex.command.format_map(ctx)
        name = self.container(image)
        if ex.log_mode == "stdout":
            inner = f'{command} > {_q(log)} 2>&1; echo "{checks.DONE_MARK}$? ===" >> {_q(log)}'
        else:
            mark = f'echo "{checks.DONE_MARK}$rc ==="'
            inner = f"{command} > {_q(outer)} 2>&1; rc=$?; {mark} >> {_q(outer)}; {mark} >> {_q(log)}"

        if self.opts.dry_run:
            self._print_dry_run(ex, image, hosts, env, command, inner, log)
            out.status, out.reason = "SKIP", "dry run"
            return

        self.echo(f"== {ex.id}: {ex.title}")
        self.echo(
            f"   {out.variant} config={variant.config or '-'} steps={steps or '-'} "
            f"hosts={','.join(out.hosts)} image={image.ref}"
        )
        infiniband = {}
        for host in hosts:
            infiniband[host.name] = self.prepare_host(host, image, ex)
        for host in hosts:
            self.check_needs(host, name, image, ex, infiniband[host.name])
        missing = self.missing_data(hosts, ex)
        if missing and self.opts.prepare and ex.prepare:
            self.run_prepare(hosts[0], image, ex)
            missing = self.missing_data(hosts, ex)
        if missing:
            raise Skip(
                f"missing under {self.site.data_root}: {', '.join(missing)} "
                f"(`run.sh prepare {ex.id}` or --prepare)"
            )
        head = hosts[0]
        out.image_id = self.docker(
            head, ["image", "inspect", "--format", "{{.Id}}", image.ref]
        ).out.strip()[:19]
        out.atom_commit = self._exec(head, name, 'echo "${ATOM_COMMIT:-}"').out.strip()

        if ex.nodes > 1:
            # Ray >= 2.5x keeps a per-container auth token; a worker in another
            # container cannot join with its own. One token for the whole cluster.
            env = {
                **env,
                "RAY_AUTH_MODE": "token",
                "RAY_AUTH_TOKEN": secrets.token_hex(32),
            }
            self._start_ray(ex, hosts, name, env, command)
            env = {**env, "RAY_ROLE": "driver", "NODE_IP": head.ip, "HEAD_IP": head.ip}

        self._exec(
            head,
            name,
            f"mkdir -p {_q(os.path.dirname(log))} && : > {_q(log)} && : > {_q(outer)}",
        )
        self.echo(f"   log: {log}")
        r = self._exec(head, name, inner, env, detach=True)
        if not r.ok:
            raise RunError(f"docker exec failed: {r.err.strip()}")
        problem = self._follow(head, log, outer, variant)
        text = self._fetch(head, log, ex.id, "trainer.log", out, variant.checks)
        if ex.log_mode == "file":
            self._fetch(head, outer, ex.id, "launcher.log", out)
        out.exit_code = checks.exit_code(text)
        if problem:
            out.status, out.reason = "ERROR", problem
        elif not self.opts.check:
            out.status = "PASS" if out.exit_code == 0 else "FAIL"
            out.reason = "" if out.exit_code == 0 else f"exit code {out.exit_code}"
        else:
            out.verdict = checks.judge(text, variant.checks, steps)
            out.status = out.verdict.status
            out.reason = "; ".join(out.verdict.reasons)
        if not self.opts.keep_container:
            for host in hosts:
                self.docker(host, ["stop", "-t", "10", name], timeout=120)

    def _start_ray(
        self, ex: Example, hosts: list[Host], name: str, env: dict, command: str
    ) -> None:
        head = hosts[0]
        for i, host in enumerate(hosts):
            role = "head" if i == 0 else "worker"
            role_env = {**env, "RAY_ROLE": role, "NODE_IP": host.ip, "HEAD_IP": head.ip}
            r = self._exec(host, name, command, role_env, timeout=600)
            started = [ln for ln in r.out.splitlines() if "Ray runtime started" in ln]
            self.echo(
                f"   {host.name}: ray {role} {'started' if started else 'FAILED'}"
            )
            if not r.ok:
                tail = "\n".join(_clean(r.out + r.err).strip().splitlines()[-10:])
                raise RunError(
                    f"ray {role} on {host.name} failed (exit {r.rc}):\n{tail}"
                )
        probe = (
            'python3 -c \'import ray; ray.init(address="auto", logging_level="ERROR"); '
            'print(int(ray.cluster_resources().get("GPU", 0)), '
            'len([n for n in ray.nodes() if n["Alive"]]))\''
        )
        want = f"{ex.nodes * ex.gpus_per_node} {ex.nodes}"
        for _ in range(12):
            got = (
                self._exec(head, name, probe, env, timeout=120).out.strip().splitlines()
            )
            if got and got[-1].strip() == want:
                self.echo(f"   ray cluster: {want.split()[0]} GPUs on {ex.nodes} nodes")
                return
            time.sleep(5)
        raise RunError(
            f"ray cluster did not come up: want '{want}' (GPUs nodes), got {got}"
        )

    def _follow(self, host: Host, log: str, outer: str, variant: Variant) -> str | None:
        offset, partial = 0, b""
        sizes = (0, 0)
        start = last_change = time.time()
        failures = 0
        while True:
            script = (
                f'for f in {_q(log)} {_q(outer)}; do stat -c %s "$f" 2>/dev/null || echo 0; done; '
                f"grep -q {_q(checks.DONE_MARK)} {_q(log)} 2>/dev/null && echo DONE || echo RUNNING; "
                f"tail -c +{offset + 1} {_q(log)} 2>/dev/null | head -c {READ_CAP}"
            )
            r = host.run(script, timeout=180)
            lines = r.out.split("\n", 3)
            if not r.ok or len(lines) < 3:
                failures += 1
                if failures > 10:
                    return f"lost contact with {host.name} while following the log"
                time.sleep(self.opts.poll_s)
                continue
            failures = 0
            now_sizes = (int(lines[0] or 0), int(lines[1] or 0))
            done = lines[2].strip() == "DONE"
            chunk = (lines[3] if len(lines) > 3 else "").encode(
                "utf-8", "surrogateescape"
            )
            chunk = chunk[: max(0, now_sizes[0] - offset)]
            offset += len(chunk)
            data = partial + chunk
            complete, _, partial = data.rpartition(b"\n")
            for ln in complete.decode("utf-8", "replace").splitlines():
                if self.opts.verbose or HIGHLIGHT.search(ln):
                    self.echo("   | " + ln[:200])
            if now_sizes != sizes:
                sizes, last_change = now_sizes, time.time()
            if done:
                return None
            if time.time() - last_change > variant.stall_min * 60:
                return f"no new log output for {variant.stall_min} min; see {log}"
            if time.time() - start > variant.timeout_min * 60:
                return f"timed out after {variant.timeout_min} min; see {log}"
            time.sleep(self.opts.poll_s)

    def _fetch(
        self,
        host: Host,
        path: str,
        ex_id: str,
        fname: str,
        out: Outcome,
        checks_: Checks | None = None,
    ) -> str:
        """Copy a log into the results; return the text to judge.

        Logs past FETCH_LIMIT (an NCCL_DEBUG=INFO run reaches 700 MB) are not
        pulled through the host transport whole: only the lines the checks read,
        plus the tail, come back, and the copy says where the full log is.
        """
        size = host.run(f"stat -c %s {_q(path)} 2>/dev/null || echo 0").out.strip()
        size = int(size) if size.isdigit() else 0
        if size <= FETCH_LIMIT or checks_ is None:
            text = host.run(f"cat {_q(path)} 2>/dev/null", timeout=900).out
            saved = text
        else:
            pats = [
                re.escape(checks.DONE_MARK),
                r"bucket done - updated=",
                r" in [0-9.]+s",
            ]
            pats += [re.escape(checks_.marker)] + [re.escape(f) for f in checks_.forbid]
            text = host.run(
                f"grep -aE {_q('|'.join(pats))} {_q(path)}", timeout=900
            ).out
            tail = host.run(f"tail -n 300 {_q(path)}", timeout=300).out
            saved = (
                f"# {size} bytes, excerpt: the lines the checks read, then the last 300 lines.\n"
                f"# full log: {host.name}:{path}\n{text}\n# ---- tail ----\n{tail}"
            )
        if self.results_dir is not None:
            dest = self.results_dir / ex_id / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(saved.encode("utf-8", "surrogateescape"))
            out.files[fname] = dest.relative_to(self.results_dir).as_posix()
        return text

    def _print_dry_run(self, ex, image, hosts, env, command, inner, log) -> None:
        name = self.container(image)
        d = self.site.docker
        self.echo(f"# {ex.id}: {ex.title}")
        self.echo(f"# on {', '.join(h.name for h in hosts)}; trainer log {log}")
        self.echo("# 1. container (once per host; restart it before every run):")
        self.echo(f"{d} {shlex.join(self.create_args(image, name, ex.nodes > 1))}")
        envflags = " ".join(f"-e {_q(f'{k}={v}')}" for k, v in env.items())
        if ex.nodes > 1:
            self.echo(
                "# 2. Ray, one token for all hosts: export RAY_AUTH_TOKEN=$(openssl rand -hex 32)"
            )
            self.echo("#    head first, then every other host (NODE_IP = that host):")
            envflags += ' -e RAY_AUTH_MODE=token -e RAY_AUTH_TOKEN="$RAY_AUTH_TOKEN"'
            for i, h in enumerate(hosts):
                role = "head" if i == 0 else "worker"
                self.echo(
                    f"{d} exec {envflags} -e RAY_ROLE={role} -e NODE_IP={h.ip} "
                    f"-e HEAD_IP={hosts[0].ip} {name} bash -lc {_q(command)}"
                )
            envflags += (
                f" -e RAY_ROLE=driver -e NODE_IP={hosts[0].ip} -e HEAD_IP={hosts[0].ip}"
            )
        self.echo("# run:")
        self.echo(f"{d} exec -d {envflags} {name} bash -lc {_q(inner)}")
        self.echo(f"tail -f {log}")


def git_rev(path: str) -> str:
    try:
        rev = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short=9", "HEAD"],
            capture_output=True, text=True, check=False, timeout=30,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=False, timeout=60,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return f"{rev}{'-dirty' if dirty else ''}" if rev else "not a git checkout"
