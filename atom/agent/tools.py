# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The agent's hands: every action the LLM can take, with JSON schemas.

Tools marked ``mutating`` change the machine (start/stop servers, run
commands, switch the agent's brain) and go through the approval callback
first; everything else only reads. The same registry backs the chat loop and
the MCP server.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from atom.agent import deploy, hardware, jobs, planner, probe, shell_policy, skills
from atom.agent import image as image_mod
from atom.agent.config import AgentConfig, repo_root
from atom.agent.knowledge import load_knowledge
from atom.agent.leases import LeaseConflict, LeaseManager
from atom.agent.models import find_models, inspect_model
from atom.agent.policy import AuditLog, Policy

MAX_RESULT_CHARS = 9000
# Commands run without asking: they only read state.
READ_ONLY_COMMANDS = {
    "rocm-smi", "amd-smi", "rocminfo", "ls", "cat", "head", "tail", "df",
    "du", "free", "nproc", "lscpu", "numactl", "uptime", "hostname", "uname",
    "pip", "env", "printenv", "which", "ps", "ss", "grep", "wc", "stat",
    "date", "id", "whoami", "lsblk", "ip",
}  # fmt: skip
READ_ONLY_SUBCOMMANDS = {
    "docker": {"ps", "images", "logs", "inspect", "version", "info", "stats"},
    "pip": {"list", "show", "freeze"},
    "git": {"log", "status", "show", "diff", "branch", "rev-parse", "describe"},
}
_SHELL_META = re.compile(r"[;&|`<>]|\$\(|\n")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., object]
    mutating: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolContext:
    cfg: AgentConfig
    # Asks the user: (tool name, args, one-line summary) -> approved?
    # Consulted only when the policy says "ask".
    approve: Callable[[str, dict, str], bool] = lambda *a: False
    policy: Policy = field(default_factory=Policy)
    session: str = ""
    # customer: deployment-side skills only; developer: all of ATOM's/AITER's.
    profile: str = "customer"
    # Live progress lines for long tools (waiting for a server).
    progress: Callable[[str], None] = lambda line: None
    plans: dict[str, dict] = field(default_factory=dict)
    # Set by switch_brain; the chat loop picks it up after the turn.
    pending_brain: dict | None = None

    def __post_init__(self) -> None:
        self.store = deploy.DeploymentStore(self.cfg.state_dir)
        self.leases = LeaseManager(self.cfg.lease_dir)
        self.audit = AuditLog(self.cfg.state_dir, self.session)
        self.jobs = jobs.JobStore(self.cfg.state_dir)
        self._skills: dict | None = None
        self.skill_source = ""
        self._kb = None
        self.kb_source = ""
        self._images: dict[str, image_mod.ImageInfo | None] = {}

    def authorize(
        self, tool: str, args: dict, summary: str, risk: str, detail: str = ""
    ) -> dict | None:
        """None if the action may run; else the refusal to hand the brain."""
        decision = self.policy.decide(risk, detail)
        if decision.action == "ask":
            allowed = bool(self.approve(tool, args, summary))
            by = "user"
        else:
            allowed = decision.action == "allow"
            by = "policy"
            if allowed:
                self.progress(f"⏵ {summary} ({decision.reason})")
        self.audit.write(
            "approval",
            tool=tool,
            risk=risk,
            summary=summary,
            allowed=allowed,
            by=by,
            reason=decision.reason,
        )
        if allowed:
            return None
        if by == "policy":
            return {"declined_by_policy": decision.reason}
        return {"declined_by_user": True}

    def image_info(self, image: str | None) -> image_mod.ImageInfo | None:
        if not image:
            return None
        if image not in self._images:
            self._images[image] = image_mod.probe(image, self.cfg.state_dir)
        return self._images[image]

    def default_image_info(self) -> image_mod.ImageInfo | None:
        if deploy.choose_runtime(self.cfg) != "docker":
            return None
        return self.image_info(deploy.choose_image(self.cfg))

    @property
    def kb(self):
        """Docs of the ATOM that will run: the image's for docker, this
        install's for local; the checkout only as a labelled fallback."""
        if self._kb is None:
            info = self.default_image_info()
            if info and (info.atom_dir / "recipes").is_dir():
                self._kb = load_knowledge(info.atom_dir)
                self.kb_source = f"recipes/docs from {info.describe()}"
            else:
                self._kb = load_knowledge(self.cfg.docs_root)
                self.kb_source = f"recipes/docs from {self.cfg.docs_root}"
        return self._kb

    def flag_table(self, image: str | None) -> tuple[dict | None, str]:
        """Server flags of the ATOM that will run: the image's, or this one's."""
        info = self.image_info(image)
        if info:
            table = info.flag_table()
            if table:
                return table, info.describe()
        return None, "this checkout's ATOM"

    def known_env(self, image: str | None) -> set[str] | None:
        info = self.image_info(image)
        if info:
            return info.env_names or None
        return image_mod.local_env_names() or None

    def provenance(self, image: str | None) -> dict:
        info = self.image_info(image)
        return info.provenance() if info else image_mod.local_provenance()

    @property
    def all_skills(self) -> dict:
        """Every skill the running ATOM/AITER ship, plus the user's own."""
        if self._skills is None:
            sources: list[tuple[str, Path]] = []
            info = self.default_image_info()
            if info:
                sources += [
                    ("atom", info.atom_dir / ".claude"),
                    ("aiter", info.aiter_dir / ".claude"),
                ]
                self.skill_source = f"skills from {info.describe()}"
            else:
                root = repo_root()
                if root:
                    sources.append(("atom", root / ".claude"))
                aiter_pkg = image_mod._package_dir("aiter")
                if aiter_pkg:
                    sources.append(("aiter", aiter_pkg.parent / ".claude"))
                self.skill_source = "skills from this install"
            for d in self.cfg.skill_dirs:
                sources.append(("user", Path(d)))
            self._skills = skills.load_catalog(sources)
        return self._skills

    @property
    def skills(self) -> dict:
        return skills.visible(self.all_skills, self.profile)

    def runtime_cwd(self, dep: deploy.Deployment | None) -> str:
        """Where a skill's "bash scripts/..." runs: the ATOM repo root."""
        if dep is not None and dep.runtime == "docker":
            info = self.image_info(dep.image)
            return info.atom_root if info else "/app/ATOM"
        root = repo_root()
        return str(root) if root else str(Path.cwd())

    def deployment(self, key: str | None) -> deploy.Deployment:
        if key is None:
            active = deploy.live(self.store) or self.store.active()
            if len(active) == 1:
                return active[0]
            raise ValueError(
                "which deployment? active: " + ", ".join(d.id for d in active)
                if active
                else "no deployments yet"
            )
        dep = self.store.get(key)
        if dep is None:
            raise ValueError(f"no deployment {key!r} (ids look like d1, or a port)")
        return dep


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


S_STR = {"type": "string"}
S_INT = {"type": "integer"}
S_DEP = {"type": "string", "description": "Deployment id (d1, d2, ...) or its port."}


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------


def t_inspect_system(ctx: ToolContext) -> dict:
    report = hardware.inspect_system()
    ours = {d.id: d.gpus for d in deploy.live(ctx.store)}
    our_leases = {d.lease for d in ctx.store.active() if d.lease}
    # VRAM alone misses a server that is still starting and another agent's
    # job that has not allocated yet; leases say who has claimed what.
    leases = ctx.leases.held()
    report["leases"] = [
        {
            "user": x.user,
            "label": x.label,
            "gpus": x.gpus,
            "port": x.port,
            "ours": x.id in our_leases,
        }
        for x in leases
    ]
    for g in report["gpus"]:
        g["used_by_agent_deployment"] = next(
            (i for i, gpus in ours.items() if g["index"] in gpus), None
        )
        holder = next((x for x in leases if g["index"] in x.gpus), None)
        g["leased_by"] = (
            f"{holder.user}: {holder.label or holder.id}" if holder else None
        )
        if holder:
            g["free"] = False
    report["free_gpus"] = [g["index"] for g in report["gpus"] if g["free"]]
    report["default_runtime"] = deploy.choose_runtime(ctx.cfg)
    info = ctx.default_image_info()
    if info:
        report["default_image"] = info.provenance()
    ctx.kb  # noqa: B018 -- resolve which docs the agent answers from
    report["knowledge_source"] = ctx.kb_source
    return report


def t_find_models(ctx: ToolContext, query: str | None = None) -> dict:
    return find_models(query, roots=ctx.cfg.model_roots)


def t_inspect_model(ctx: ToolContext, model: str) -> dict:
    gpus = hardware.list_gpus(with_names=False)
    return inspect_model(
        model, gpus[0].vram_total_mib if gpus else None, ctx.cfg.model_roots
    )


def t_search_docs(ctx: ToolContext, query: str, scope: str = "all", k: int = 5) -> dict:
    prefix = {"recipes": "recipes/", "docs": "docs/", "debug": ".claude/"}.get(scope)
    hits = ctx.kb.search(query, k=min(k, 10), path_prefix=prefix)
    return (
        {"query": query, "results": hits}
        if hits
        else {"query": query, "results": [], "hint": "try other words"}
    )


def t_read_doc(ctx: ToolContext, path: str, section: str | None = None) -> dict:
    return ctx.kb.read(path, section)


def t_list_recipes(ctx: ToolContext) -> dict:
    return {"recipes": ctx.kb.list_recipes()}


def t_explain_flags(ctx: ToolContext, flags: list[str]) -> dict:
    return planner.explain_flags(flags)


def t_plan_deployment(
    ctx: ToolContext,
    model: str,
    goal: str | None = None,
    gpus: list[int] | None = None,
    tp: int | None = None,
    port: int | None = None,
    recipe: str | None = None,
    section: str | None = None,
    max_model_len: int | None = None,
    runtime: str | None = None,
) -> dict:
    active = deploy.live(ctx.store)
    leased_gpus, leased_ports = ctx.leases.reserved()
    chosen_runtime = deploy.choose_runtime(ctx.cfg, runtime)
    image = deploy.choose_image(ctx.cfg) if chosen_runtime == "docker" else None
    table, flag_source = ctx.flag_table(image)
    plan = planner.make_plan(
        ctx.kb,
        model,
        gpus=gpus,
        tp=tp,
        goal=goal,
        port=port,
        recipe=recipe,
        section=section,
        max_model_len=max_model_len,
        reserved_gpus={g for d in active for g in d.gpus} | leased_gpus,
        taken_ports={d.port for d in active} | leased_ports,
        model_roots=ctx.cfg.model_roots,
        flag_table=table,
        flag_source=flag_source,
        known_env=ctx.known_env(image),
    )
    if not plan.get("ok"):
        return plan
    plan["runtime"] = chosen_runtime
    if image:
        plan["image"] = image
    plan["provenance"] = ctx.provenance(image)
    plan_id = f"p{len(ctx.plans) + 1}"
    ctx.plans[plan_id] = plan
    preview = deploy.Deployment(
        id="<id>", model=plan["model"], runtime=plan["runtime"], gpus=plan["gpus"],
        port=plan["port"], env=plan["env"], args=plan["args"], log="", created=0,
        image=plan.get("image"), container="atom-agent-<id>",
    )  # fmt: skip
    plan["command"] = deploy.shell_command(preview)
    return {"plan_id": plan_id, **plan}


def t_launch_server(
    ctx: ToolContext,
    plan_id: str,
    add_args: list[str] | None = None,
    remove_flags: list[str] | None = None,
    set_env: dict | None = None,
    unset_env: list[str] | None = None,
    role: str = "service",
    force: bool = False,
) -> dict:
    plan = ctx.plans.get(plan_id)
    if plan is None:
        return {"error": f"unknown plan {plan_id!r}; call plan_deployment first"}
    args = list(plan["args"])
    for flag in remove_flags or []:
        args = planner.remove_arg(args, flag)
    add_args = list(add_args or [])
    # An added flag replaces the plan's value instead of duplicating it.
    for tok in add_args:
        if tok.startswith("--") or tok == "-tp":
            args = planner.remove_arg(args, tok.split("=", 1)[0])
    args += add_args
    env = {k: str(v) for k, v in {**plan["env"], **(set_env or {})}.items()}
    for k in unset_env or []:
        env.pop(k, None)
    table, flag_source = ctx.flag_table(plan.get("image"))
    problems = planner.validate_args(args, table)
    if problems and not force:
        return {
            "error": f"flag check against {flag_source} failed; fix or pass force=true",
            "problems": problems,
        }
    tp = planner.int_arg(args, "-tp", "--tensor-parallel-size", default=1)
    if tp != len(plan["gpus"]):
        return {
            "error": f"TP{tp} but the plan reserved {len(plan['gpus'])} GPUs; re-plan"
        }
    port = planner.int_arg(args, "--server-port", default=plan["port"])
    # Pre-check so the user is not asked to approve an impossible launch; the
    # atomic check is the lease taken inside deploy.launch (races with other
    # agents end there, not here).
    leased_gpus, leased_ports = ctx.leases.reserved()
    busy_why = deploy._gpu_busy_checker()
    conflicts = {
        i: "leased by another deployment" if i in leased_gpus else busy_why(i)
        for i in plan["gpus"]
    }
    conflicts = {i: why for i, why in conflicts.items() if why}
    hard = {i: w for i, w in conflicts.items() if w.startswith("leased")}
    if hard or (conflicts and not force):
        return {
            "error": f"GPUs {sorted(conflicts)} are not free",
            "held_by": conflicts,
            "hint": "re-plan for free GPUs; force=true only skips the sysfs check "
            "(stale memory), never another agent's lease",
        }
    if port in leased_ports or (not force and not planner.port_free(port)):
        return {"error": f"port {port} is taken; re-plan"}
    summary = f"start {Path(plan['model']).name} on GPUs {plan['gpus']} (TP{tp}) port {port} via {plan['runtime']}"
    denied = ctx.authorize(
        "launch_server",
        {"plan_id": plan_id, "force": force},
        summary + (" -- FORCED past the GPU/port check" if force else ""),
        risk="launch_forced" if force else "launch",
    )
    if denied:
        return denied
    try:
        dep = deploy.launch(
            ctx.cfg,
            ctx.store,
            model=plan["model"],
            gpus=plan["gpus"],
            port=port,
            args=args,
            env=env,
            runtime=plan["runtime"],
            image=plan.get("image"),
            role=role,
            source=plan.get("source"),
            leases=ctx.leases,
            force=force,
            provenance=plan.get("provenance"),
        )
    except LeaseConflict as e:
        return {"error": f"lost a race for the GPUs/port: {e}", "held_by": e.holders}
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    return {
        "deployment": dep.id,
        "runtime": dep.runtime,
        "container": dep.container,
        "pid": dep.pid,
        "gpus": dep.gpus,
        "port": dep.port,
        "log": dep.log,
        "expected_startup_minutes": plan.get("expected_startup_minutes"),
        "next": "call wait_until_ready",
    }


def t_wait_until_ready(
    ctx: ToolContext, deployment: str | None = None, timeout_min: float = 20
) -> dict:
    dep = ctx.deployment(deployment)
    t0 = time.monotonic()
    last = {"key": None, "at": -1e9}

    def report(st: dict) -> None:
        # Phase changes at once; counters (shards, graph sizes) every 15 s.
        now = time.monotonic()
        key = (st["state"], st["phase"])
        if key == last["key"] and now - last["at"] < 15:
            return
        last.update(key=key, at=now)
        msg = f"[{int(now - t0):>4}s] {dep.id}: {st['state']} -- {st['phase']}"
        if st.get("progress"):
            msg += f" ({st['progress']})"
        ctx.progress(msg)

    st = deploy.wait_ready(dep, timeout_s=timeout_min * 60, on_progress=report)
    if st["state"] == "ready":
        st["next"] = "call smoke_test"
    elif st["state"] in ("failed", "failing"):
        st["next"] = (
            "explain the findings; propose a fix (re-plan) -- do not relaunch unchanged"
        )
    return st


def t_deployment_status(ctx: ToolContext, deployment: str | None = None) -> dict:
    if deployment is None:
        deps = ctx.store.all()[-12:]
        return {
            "deployments": [
                {
                    k: v
                    for k, v in deploy.status(d).items()
                    if k
                    in (
                        "id",
                        "role",
                        "state",
                        "model",
                        "served_model",
                        "base_url",
                        "gpus",
                        "phase",
                        "uptime_s",
                    )
                }
                for d in deps
                if not d.stopped or time.time() - d.created < 86400
            ]
        }
    return deploy.status(ctx.deployment(deployment))


def t_read_server_log(
    ctx: ToolContext,
    deployment: str | None = None,
    lines: int = 60,
    grep: str | None = None,
) -> dict:
    from atom.agent.diagnose import clean

    dep = ctx.deployment(deployment)
    text = clean(deploy.read_log_tail(dep))
    rows = text.splitlines()
    if grep:
        try:
            rx = re.compile(grep, re.IGNORECASE)
        except re.error as e:
            return {"error": f"bad regex: {e}"}
        rows = [r for r in rows if rx.search(r)]
    rows = [r[:300] for r in rows if r.strip()][-min(lines, 300) :]
    return {"deployment": dep.id, "log": dep.log, "lines": rows}


def t_diagnose(ctx: ToolContext, deployment: str | None = None) -> dict:
    dep = ctx.deployment(deployment)
    st = deploy.status(dep)
    docs = []
    for f in st.get("findings", [])[:2]:
        ref = f.get("doc") or ""
        if "#" in ref:
            path, section = ref.split("#", 1)
            d = ctx.kb.read(path, section, max_chars=2500)
            if "text" in d:
                docs.append(
                    {
                        "for": f["key"],
                        "path": path,
                        "section": section,
                        "text": d["text"],
                    }
                )
    if not st.get("findings") and st["state"] in ("failed", "failing"):
        query = st.get("last_exception") or " ".join(st.get("last_lines", [])[-3:])
        docs = ctx.kb.search(query, k=3)
    return {"status": st, "guidance": docs}


def t_smoke_test(
    ctx: ToolContext,
    deployment: str | None = None,
    prompt: str | None = None,
    check_tool_calling: bool = True,
) -> dict:
    dep = ctx.deployment(deployment)
    kw = {"check_tool_calling": check_tool_calling}
    if prompt:
        kw["prompt"] = prompt
    return probe.smoke_test(f"{dep.base_url}/v1", **kw)


def t_benchmark(
    ctx: ToolContext,
    deployment: str | None = None,
    input_len: int = 1024,
    output_len: int = 256,
    concurrency: int = 8,
    num_requests: int | None = None,
) -> dict:
    dep = ctx.deployment(deployment)
    ctx.progress(
        f"benchmark {dep.id}: ISL~{input_len} OSL {output_len} c{concurrency} ..."
    )
    return probe.benchmark(
        f"{dep.base_url}/v1", input_len, output_len, concurrency, num_requests
    )


def t_stop_server(ctx: ToolContext, deployment: str) -> dict:
    dep = ctx.deployment(deployment)
    if dep.stopped:
        return {"id": dep.id, "already_stopped": True}
    denied = ctx.authorize(
        "stop_server",
        {"deployment": dep.id},
        f"stop {dep.id} ({Path(dep.model).name}, GPUs {dep.gpus})",
        risk="stop_own",
    )
    if denied:
        return denied
    return deploy.stop(ctx.store, dep, leases=ctx.leases)


def t_export_deployment(ctx: ToolContext, deployment: str | None = None) -> dict:
    dep = ctx.deployment(deployment)
    out = ctx.cfg.state_dir / "exports" / dep.id
    out.mkdir(parents=True, exist_ok=True)
    launch = out / "launch.sh"
    launch.write_text(
        "#!/bin/bash\n# Generated by atom-chat\nset -euo pipefail\n"
        + deploy.shell_command(dep)
        + "\n"
    )
    launch.chmod(0o755)
    compose = out / "docker-compose.yaml"
    model_path = planner.get_arg(dep.args, "--model") or dep.model
    mount = deploy.model_mount(model_path)
    env_lines = "".join(
        f"      {k}: {json.dumps(v)}\n"
        for k, v in {
            **dep.env,
            "HIP_VISIBLE_DEVICES": ",".join(map(str, dep.gpus)),
        }.items()
    )
    compose.write_text(
        "services:\n"
        "  atom:\n"
        f"    image: {dep.image or ctx.cfg.image or 'rocm/atom:latest'}\n"
        "    network_mode: host\n    ipc: host\n    shm_size: 16g\n"
        "    devices: [/dev/kfd, /dev/dri]\n    group_add: [video]\n"
        "    cap_add: [SYS_PTRACE]\n    security_opt: [seccomp=unconfined]\n"
        "    ulimits: {memlock: -1, stack: 67108864}\n"
        f"    volumes: [\"{mount}:{mount}:ro\"]\n"
        "    environment:\n" + env_lines +
        "    command: " + json.dumps(["python3", "-u", "-m", deploy.SERVER_MODULE, *dep.args]) + "\n"
        "    restart: unless-stopped\n"
    )  # fmt: skip
    return {"files": [str(launch), str(compose)], "command": deploy.shell_command(dep)}


def is_read_only_command(command: str) -> bool:
    """True for commands that only read state on this host.

    Simple commands are checked against READ_ONLY_COMMANDS (incl. docker
    ps/logs/...); compound ones (pipes, chains, loops) must be read-only in
    every part by the shell policy, with no writes allowed at all.
    """
    if _SHELL_META.search(command):
        return shell_policy.classify(command, scratch=()) == "workload"
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    head = Path(argv[0]).name
    sub = argv[1] if len(argv) > 1 else ""
    return (
        head in READ_ONLY_COMMANDS and head not in READ_ONLY_SUBCOMMANDS
    ) or sub in READ_ONLY_SUBCOMMANDS.get(head, ())


def t_run_command(ctx: ToolContext, command: str, timeout_s: int = 60) -> dict:
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return {"error": f"cannot parse: {e}"}
    if not argv:
        return {"error": "empty command"}
    if not is_read_only_command(command):
        denied = ctx.authorize(
            "run_command",
            {"command": command},
            f"run: {command}",
            risk="command",
            detail=command,
        )
        if denied:
            return denied
    try:
        res = subprocess.run(
            command if _SHELL_META.search(command) else argv,
            shell=bool(_SHELL_META.search(command)),
            capture_output=True,
            text=True,
            timeout=min(timeout_s, 1800),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout_s}s"}
    except OSError as e:
        return {"error": str(e)}
    return {
        "exit_code": res.returncode,
        "stdout": res.stdout[-6000:],
        "stderr": res.stderr[-2000:],
    }


def t_switch_brain(ctx: ToolContext, deployment: str) -> dict:
    dep = ctx.deployment(deployment)
    st = deploy.status(dep)
    if st["state"] != "ready":
        return {"error": f"{dep.id} is {st['state']}, not ready"}
    denied = ctx.authorize(
        "switch_brain",
        {"deployment": dep.id},
        f"continue this chat on {st['served_model']} ({dep.id})",
        risk="switch_brain",
    )
    if denied:
        return denied
    ctx.pending_brain = {
        "base_url": f"{dep.base_url}/v1",
        "model": st["served_model"],
        "label": dep.id,
    }
    return {"ok": True, "note": "the next turn runs on the new model"}


# --------------------------------------------------------------------------
# Skills and execution in the runtime
# --------------------------------------------------------------------------

# forbidden | workload | exec: see atom/agent/shell_policy.py.
classify_exec = shell_policy.classify


def t_list_skills(ctx: ToolContext) -> dict:
    hidden = len(ctx.all_skills) - len(ctx.skills)
    return {
        "profile": ctx.profile,
        "source": ctx.skill_source,
        "skills": [
            {"name": k, "kind": v.kind, "description": v.description[:400]}
            for k, v in sorted(ctx.skills.items())
        ],
        "hidden_for_profile": hidden,
    }


def t_load_skill(ctx: ToolContext, name: str) -> dict:
    sk = skills.resolve(ctx.skills, name)
    if sk is None:
        if skills.resolve(ctx.all_skills, name):
            return {
                "error": f"{name} is a developer skill; this agent runs with "
                f"--profile {ctx.profile}"
            }
        return {"error": f"no skill {name!r}", "available": sorted(ctx.skills)}
    _, body = skills.parse_frontmatter(sk.path.read_text(errors="replace"))
    return {
        "name": sk.name,
        "kind": sk.kind,
        "version": sk.meta.get("version"),
        "source": ctx.skill_source,
        "truncated": len(body) > skills.MAX_BODY_CHARS,
        "body": body[: skills.MAX_BODY_CHARS],
        "files": sk.files(),
        "how_to_run": skills.ADAPTER_NOTES,
    }


def t_read_skill_file(ctx: ToolContext, name: str, path: str) -> dict:
    sk = skills.resolve(ctx.skills, name)
    if sk is None:
        return {"error": f"no visible skill {name!r}"}
    target = (sk.root / path).resolve()
    if sk.root.resolve() not in target.parents or not target.is_file():
        return {"error": f"{path!r} is not a file of {sk.name}", "files": sk.files()}
    text = target.read_text(errors="replace")
    return {
        "name": sk.name,
        "path": path,
        "truncated": len(text) > 30_000,
        "text": text[:30_000],
    }


def _exec_target(ctx: ToolContext, deployment: str | None):
    """The deployment whose runtime a command runs in (None = local install)."""
    dep = None
    if deployment is not None or deploy.live(ctx.store):
        dep = ctx.deployment(deployment)
        if not deploy.is_alive(dep):
            raise ValueError(f"{dep.id} is not running; nothing to execute in")
    elif deploy.choose_runtime(ctx.cfg) == "docker":
        raise ValueError(
            "no running deployment to execute in; launch one first "
            "(commands run inside its container)"
        )
    return dep


def t_exec_in_runtime(
    ctx: ToolContext,
    command: str,
    deployment: str | None = None,
    background: bool = False,
    timeout_s: int = 120,
    env: dict | None = None,
) -> dict:
    command = command.strip()
    if command.endswith("&") and not command.endswith("&&"):
        command, background = command[:-1].rstrip(), True
    kind = classify_exec(command)
    if kind == "forbidden":
        return {
            "error": "refused: this starts/stops servers or resets GPUs behind the "
            "lease system",
            "use_instead": "launch_server (add_args / set_env), wait_until_ready, "
            "stop_server",
        }
    dep = _exec_target(ctx, deployment)
    where = f"{dep.id} ({dep.container or 'local'})" if dep else "the local install"
    denied = ctx.authorize(
        "exec_in_runtime",
        {"command": command, "deployment": dep.id if dep else None},
        f"run in {where}: {command[:160]}",
        risk="exec_workload" if kind == "workload" else "exec",
        detail=command,
    )
    if denied:
        return denied
    env = {k: str(v) for k, v in (env or {}).items()}
    cwd = ctx.runtime_cwd(dep)
    if background:
        job = jobs.start(ctx.jobs, dep, command, cwd=cwd, env=env, label=command[:80])
        return {
            "job": job.id,
            "state": "running",
            "cwd": cwd,
            "log": job.log,
            "next": "wait_job (or job_log to peek)",
        }
    return jobs.run_foreground(
        ctx.jobs, dep, command, cwd=cwd, env=env, timeout_s=min(timeout_s, 600)
    )


def _job(ctx: ToolContext, job_id: str) -> jobs.Job:
    job = ctx.jobs.get(job_id)
    if job is None:
        raise ValueError(f"no job {job_id!r}")
    return job


def t_job_status(ctx: ToolContext, job: str | None = None) -> dict:
    if job is None:
        return {
            "jobs": [
                {k: v for k, v in jobs.status(j, 1).items() if k != "tail"}
                for j in ctx.jobs.all()[-10:]
            ]
        }
    return jobs.status(_job(ctx, job))


def t_wait_job(ctx: ToolContext, job: str, timeout_min: float = 30) -> dict:
    return jobs.wait(
        _job(ctx, job), timeout_s=timeout_min * 60, on_progress=ctx.progress
    )


def t_job_log(
    ctx: ToolContext, job: str, lines: int = 60, grep: str | None = None
) -> dict:
    j = _job(ctx, job)
    try:
        rows = jobs.tail(j, min(lines, 400), grep)
    except re.error as e:
        return {"error": f"bad regex: {e}"}
    return {"job": j.id, "log": j.log, "lines": rows}


def t_cancel_job(ctx: ToolContext, job: str) -> dict:
    j = _job(ctx, job)
    denied = ctx.authorize(
        "cancel_job",
        {"job": j.id},
        f"cancel {j.id}: {j.command[:100]}",
        risk="stop_own",
    )
    if denied:
        return denied
    return jobs.cancel(ctx.jobs, j)


def t_copy_from_runtime(
    ctx: ToolContext, path: str, deployment: str | None = None
) -> dict:
    dep = _exec_target(ctx, deployment)
    dest = (
        ctx.cfg.state_dir
        / "artifacts"
        / (dep.id if dep else "local")
        / time.strftime("%Y%m%d-%H%M%S")
    )
    return jobs.copy_out(dep, path, dest, ctx.runtime_cwd(dep))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def _array(items: dict = S_STR) -> dict:
    return {"type": "array", "items": items}


def _str(description: str, **kw) -> dict:
    return {"type": "string", "description": description, **kw}


# (name, description, parameters, function, mutating)
_TOOL_SPECS = (
    (
        "inspect_system",
        (
            "GPUs (HIP index, arch, VRAM, who is using each), running ATOM "
            "servers, docker and ATOM images, ROCm/ATOM versions. Call once at "
            "the start."
        ),
        _obj({}),
        t_inspect_system,
        False,
    ),
    (
        "find_models",
        (
            "Search local disks and Hugging Face caches for checkpoints. query: "
            "words that must all appear in the name (e.g. 'deepseek v4.1'); "
            "omit to list everything."
        ),
        _obj({"query": S_STR}),
        t_find_models,
        False,
    ),
    (
        "inspect_model",
        (
            "Read a checkpoint's config: architecture, ATOM support, size, "
            "quantization, context length, and TP sizing against this machine."
        ),
        _obj({"model": _str("Local path, HF repo id, or directory name.")}, ["model"]),
        t_inspect_model,
        False,
    ),
    (
        "search_docs",
        (
            "Search ATOM's recipes, guides and debug playbooks. "
            "scope: recipes | docs | debug | all."
        ),
        _obj(
            {
                "query": S_STR,
                "scope": _str(
                    "where to search", enum=["all", "recipes", "docs", "debug"]
                ),
                "k": S_INT,
            },
            ["query"],
        ),
        t_search_docs,
        False,
    ),
    (
        "read_doc",
        "Read a doc (path from search_docs), optionally one section by heading.",
        _obj({"path": S_STR, "section": S_STR}, ["path"]),
        t_read_doc,
        False,
    ),
    ("list_recipes", "All model recipes with titles.", _obj({}), t_list_recipes, False),
    (
        "explain_flags",
        (
            "Help text, default and choices of ATOM server flags (read from the "
            "server's argparse). Use before suggesting any flag."
        ),
        _obj({"flags": _array()}, ["flags"]),
        t_explain_flags,
        False,
    ),
    (
        "plan_deployment",
        (
            "Build a launch plan: picks the best recipe command for this model, "
            "GPU and goal, chooses free GPUs, TP and port, validates flags. "
            "Returns plan_id, the exact command, provenance and warnings. "
            "Starts nothing."
        ),
        _obj(
            {
                "model": S_STR,
                "goal": _str("e.g. 'low latency', 'max throughput', 'agentic coding'"),
                "gpus": {
                    **_array(S_INT),
                    "description": "HIP indices; omit for free ones",
                },
                "tp": S_INT,
                "port": S_INT,
                "recipe": _str("force a recipe path"),
                "section": _str("force a recipe section (substring)"),
                "max_model_len": S_INT,
                "runtime": _str("where to run the server", enum=["local", "docker"]),
            },
            ["model"],
        ),
        t_plan_deployment,
        False,
    ),
    (
        "launch_server",
        (
            "Start the server from a plan (the user is asked to approve). "
            "Optional tweaks: add_args (e.g. ['--max-num-seqs','64']) replace "
            "same-named flags; remove_flags; set_env / unset_env."
        ),
        _obj(
            {
                "plan_id": S_STR,
                "add_args": _array(),
                "remove_flags": _array(),
                "set_env": {"type": "object", "additionalProperties": S_STR},
                "unset_env": _array(),
                "role": _str("what the server is for", enum=["service", "brain"]),
                "force": {
                    "type": "boolean",
                    "description": "skip GPU/port/flag checks; only if the user insists",
                },
            },
            ["plan_id"],
        ),
        t_launch_server,
        True,
    ),
    (
        "wait_until_ready",
        (
            "Block until a deployment serves, fails, or times out, streaming "
            "its startup phase to the user. Large models take 5-15 min."
        ),
        _obj({"deployment": S_DEP, "timeout_min": {"type": "number"}}),
        t_wait_until_ready,
        False,
    ),
    (
        "deployment_status",
        (
            "State, phase, GPU memory and detected problems of one deployment, "
            "or a list of all when omitted."
        ),
        _obj({"deployment": S_DEP}),
        t_deployment_status,
        False,
    ),
    (
        "read_server_log",
        "Tail of a deployment's server log, optionally filtered by a regex.",
        _obj({"deployment": S_DEP, "lines": S_INT, "grep": S_STR}),
        t_read_server_log,
        False,
    ),
    (
        "diagnose",
        (
            "Explain why a deployment failed: detected failure signatures, last "
            "exception, and the matching debug-guide sections."
        ),
        _obj({"deployment": S_DEP}),
        t_diagnose,
        False,
    ),
    (
        "smoke_test",
        (
            "Send a real chat request (TTFT, decode speed, reply) and check "
            "that tool calling parses."
        ),
        _obj(
            {
                "deployment": S_DEP,
                "prompt": S_STR,
                "check_tool_calling": {"type": "boolean"},
            }
        ),
        t_smoke_test,
        False,
    ),
    (
        "benchmark",
        (
            "Quick load test against a deployment: TTFT/TPOT percentiles and "
            "token throughput."
        ),
        _obj(
            {
                "deployment": S_DEP,
                "input_len": S_INT,
                "output_len": S_INT,
                "concurrency": S_INT,
                "num_requests": S_INT,
            }
        ),
        t_benchmark,
        False,
    ),
    (
        "stop_server",
        (
            "Stop a deployment this agent started (the user is asked to "
            "approve). Never touches other servers."
        ),
        _obj({"deployment": S_DEP}, ["deployment"]),
        t_stop_server,
        True,
    ),
    (
        "export_deployment",
        "Write a reproducible launch.sh and docker-compose.yaml for a deployment.",
        _obj({"deployment": S_DEP}),
        t_export_deployment,
        False,
    ),
    (
        "run_command",
        (
            "Run a shell command. Read-only commands (rocm-smi, amd-smi, ls, "
            "cat, df, docker ps/logs, pip list, ...) run directly; anything "
            "else asks the user first."
        ),
        _obj({"command": S_STR, "timeout_s": S_INT}, ["command"]),
        t_run_command,
        True,
    ),
)
_SKILL_SPECS = (
    (
        "list_skills",
        "Procedures ATOM's and AITER's engineers wrote down (benchmark, "
        "accuracy, traces, debugging, ...), read from the image that runs.",
        _obj({}),
        t_list_skills,
        False,
    ),
    (
        "load_skill",
        (
            "Read a skill before following it (name like 'atom:capture-trace'). "
            "Returns its steps, its files, and how to map its shell steps to "
            "these tools."
        ),
        _obj({"name": S_STR}, ["name"]),
        t_load_skill,
        False,
    ),
    (
        "read_skill_file",
        "Read a reference or script that ships with a skill.",
        _obj({"name": S_STR, "path": S_STR}, ["name", "path"]),
        t_read_skill_file,
        False,
    ),
    (
        "exec_in_runtime",
        (
            "Run a shell command where ATOM runs: inside a deployment's container "
            "(cwd = the ATOM repo, e.g. /app/ATOM). For skill steps such as "
            "'bash scripts/run_benchmark.sh ...'. background=true for anything "
            "over a minute, then wait_job. Server start/stop scripts are refused."
        ),
        _obj(
            {
                "command": S_STR,
                "deployment": S_DEP,
                "background": {"type": "boolean"},
                "timeout_s": S_INT,
                "env": {"type": "object", "additionalProperties": S_STR},
            },
            ["command"],
        ),
        t_exec_in_runtime,
        True,
    ),
    (
        "job_status",
        "State, exit code and log tail of a background job; all recent jobs if omitted.",
        _obj({"job": S_STR}),
        t_job_status,
        False,
    ),
    (
        "wait_job",
        "Block until a background job finishes (streams progress); returns its log tail.",
        _obj({"job": S_STR, "timeout_min": {"type": "number"}}, ["job"]),
        t_wait_job,
        False,
    ),
    (
        "job_log",
        "Tail of a job's output, optionally filtered by a regex.",
        _obj({"job": S_STR, "lines": S_INT, "grep": S_STR}, ["job"]),
        t_job_log,
        False,
    ),
    (
        "cancel_job",
        "Stop a background job (its whole process tree in the container).",
        _obj({"job": S_STR}, ["job"]),
        t_cancel_job,
        True,
    ),
    (
        "copy_from_runtime",
        (
            "Copy a file or directory (e.g. a trace dir, a result JSON) out of "
            "the deployment's container to this host; returns the local paths."
        ),
        _obj({"path": S_STR, "deployment": S_DEP}, ["path"]),
        t_copy_from_runtime,
        False,
    ),
)
_SWITCH_BRAIN_SPEC = (
    "switch_brain",
    (
        "Continue this conversation on a model this agent deployed (it must "
        "be ready and support tool calling)."
    ),
    _obj({"deployment": S_DEP}, ["deployment"]),
    t_switch_brain,
    True,
)


def build_tools(include_brain_switch: bool = True) -> list[Tool]:
    specs = (
        _TOOL_SPECS
        + _SKILL_SPECS
        + ((_SWITCH_BRAIN_SPEC,) if include_brain_switch else ())
    )
    return [Tool(*spec) for spec in specs]


class ToolBox:
    def __init__(self, ctx: ToolContext, include_brain_switch: bool = True):
        self.ctx = ctx
        self.tools = {t.name: t for t in build_tools(include_brain_switch)}

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: dict) -> dict:
        t0 = time.monotonic()
        result = self._call(name, arguments)
        self.ctx.audit.write(
            "tool",
            tool=name,
            args={k: _clip(v) for k, v in arguments.items()},
            ok=not (result.get("error") or result.get("declined_by_user")),
            error=_clip(result.get("error")),
            declined=result.get("declined_by_user") or result.get("declined_by_policy"),
            deployment=result.get("deployment") or result.get("id"),
            seconds=round(time.monotonic() - t0, 2),
        )
        return result

    def _call(self, name: str, arguments: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}", "tools": sorted(self.tools)}
        try:
            result = tool.func(self.ctx, **arguments)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001 -- the LLM must see every failure
            return {
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:],
            }
        return result if isinstance(result, dict) else {"result": result}


def _clip(value, n: int = 300):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= n else text[:n] + "..."


def render_result(result: dict, limit: int = MAX_RESULT_CHARS) -> str:
    text = json.dumps(result, default=str, ensure_ascii=False)
    if len(text) > limit:
        text = text[:limit] + f'... [truncated {len(text) - limit} chars]"'
    return text
