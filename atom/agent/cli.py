# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``atom``: talk to ATOM to deploy models.

atom                           # chat; finds a brain or offers to start one
atom --brain-url http://host:8000/v1
atom --bootstrap /models/DeepSeek-V4.1-Flash --bootstrap-gpus 0,1,2,3
atom -p "serve Qwen3-8B on one free GPU" --yes   # one-shot, scriptable
atom --guided                  # step-by-step wizard, no LLM needed
atom status | stop d1 | mcp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from atom.agent import deploy, hardware
from atom.agent.brain import Brain, connect, discover
from atom.agent.config import DEFAULT_MODEL_ROOTS, DEFAULT_STATE_DIR, AgentConfig
from atom.agent.models import find_models
from atom.agent.policy import MODES, Policy
from atom.agent.prompts import build_system_prompt, machine_brief
from atom.agent.session import ChatSession, Hooks, SessionLog
from atom.agent.tools import ToolBox, ToolContext

# Architectures ATOM can parse tool calls for (atom/entrypoints/openai/tool_parser).
TOOL_CAPABLE_ARCHS = {
    "DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM", "GlmMoeDsaForCausalLM",
    "Glm4MoeForCausalLM", "Glm5NextForConditionalGeneration",
    "KimiK25ForConditionalGeneration", "KimiK3ForConditionalGeneration",
    "MiniMaxM2ForCausalLM", "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration", "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM", "Qwen4ExpForConditionalGeneration",
}  # fmt: skip


class Style:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.on else s

    def dim(self, s):
        return self._w("2", s)

    def bold(self, s):
        return self._w("1", s)

    def red(self, s):
        return self._w("31", s)

    def green(self, s):
        return self._w("32", s)

    def yellow(self, s):
        return self._w("33", s)

    def cyan(self, s):
        return self._w("36", s)


def _short(value, n=60) -> str:
    s = value if isinstance(value, str) else json.dumps(value)
    return s if len(s) <= n else s[: n - 1] + "…"


def summarize_result(name: str, r: dict) -> str:
    """One line a person can read; the brain gets the full JSON."""
    if r.get("declined_by_user"):
        return "declined"
    if r.get("error"):
        return "error: " + _short(r["error"], 160)
    if name == "inspect_system":
        g = r.get("gpus") or []
        head = f"{len(g)} x {g[0]['product']} ({g[0]['arch']})" if g else "no GPUs"
        return f"{head}; free {r.get('free_gpus')}; runtime {r.get('default_runtime')}"
    if name == "plan_deployment":
        src = r.get("source") or {}
        where = (
            f"{src.get('recipe')} > {src.get('section', '').split(' > ')[-1]}"
            if src.get("recipe")
            else "generic defaults"
        )
        w = f"; {len(r.get('warnings', []))} warning(s)" if r.get("warnings") else ""
        ver = (r.get("provenance") or {}).get("atom")
        ver = f" [ATOM {ver.split()[0]}]" if ver else ""
        return f"{r.get('plan_id')}: TP{r.get('tp')} on GPUs {r.get('gpus')} :{r.get('port')} via {r.get('runtime')}{ver} <- {where}{w}"
    if name == "launch_server":
        return f"started {r.get('deployment')} ({r.get('container') or 'pid ' + str(r.get('pid'))}); log {r.get('log')}"
    if name in ("wait_until_ready", "deployment_status") and "state" in r:
        extra = (
            f" at {r.get('base_url')} as {r.get('served_model')!r}"
            if r["state"] == "ready"
            else f" ({r.get('phase')})"
        )
        probs = r.get("findings") or []
        if probs and r["state"] != "ready":
            extra += f" -- {probs[0]['title']}"
        return f"{r['id']}: {r['state']}{extra}"
    if name == "smoke_test":
        c = r.get("chat") or {}
        tc = r.get("tool_calling") or {}
        tools = (
            ""
            if not tc
            else (
                " ; tool calling ok"
                if tc.get("works")
                else " ; tool calling NOT working"
            )
        )
        return f"{'ok' if r.get('ok') else 'FAILED'}; TTFT {c.get('ttft_ms')} ms; {c.get('decode_tok_per_s')} tok/s{tools}"
    if name == "benchmark":
        return f"{r.get('requests_ok')} ok / {r.get('requests_failed')} failed; TTFT p50 {r.get('ttft_ms', {}).get('p50')} ms; TPOT p50 {r.get('tpot_ms', {}).get('p50')} ms; {r.get('total_tok_per_s')} total tok/s"
    if name == "stop_server":
        return f"stopped {r.get('id')}" + (
            f"; GPUs still busy {r['gpus_still_busy']}"
            if r.get("gpus_still_busy")
            else ""
        )
    if name == "find_models":
        bad = f", {r['incomplete']} incomplete" if r.get("incomplete") else ""
        return f"{r.get('total_found', len(r.get('models', [])))} model(s){bad}"
    if name == "search_docs":
        return "; ".join(f"{h['path']}" for h in r.get("results", [])[:3]) or "no hits"
    if name == "inspect_model":
        if not r.get("found"):
            return "not found"
        part = (
            ""
            if r.get("complete", True)
            else f" INCOMPLETE {r.get('shards_present')}/{r.get('shards_total')} shards"
        )
        return (
            f"{r.get('architecture')} {r.get('weights_gib')} GiB "
            f"supported={r.get('atom_supported')}{part}"
        )
    if name == "diagnose":
        f = (r.get("status") or {}).get("findings") or []
        return f[0]["title"] if f else "no known signature"
    if name == "export_deployment":
        return ", ".join(r.get("files", []))
    if name == "run_command":
        return f"exit {r.get('exit_code')}"
    if name == "load_skill":
        return f"{r.get('name')} ({r.get('kind')}, v{r.get('version') or '?'})"
    if name in ("exec_in_runtime", "wait_job", "job_status") and r.get("job"):
        code = r.get("exit_code")
        last = (r.get("tail") or [""])[-1][:100]
        return (
            f"{r['job']}: {r.get('state')}"
            + (f" (exit {code})" if code is not None else "")
            + (f" -- {last}" if last else "")
        )
    if name == "copy_from_runtime":
        return f"{r.get('file_count')} file(s) -> {r.get('copied_to')}"
    return "done"


class Terminal:
    """Rendering + approvals for the interactive chat."""

    def __init__(self, show_thinking: bool, interactive: bool):
        self.st = Style(sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
        # Set once the ToolContext exists; "[a]lways" raises its mode.
        self.policy: Policy | None = None
        self.show_thinking = show_thinking
        self.interactive = interactive
        self._in_text = False
        self._held_ws = ""
        self._thinking_noted = False

    def _end_text(self):
        if self._in_text:
            print()
            self._in_text = False

    def content(self, s: str):
        # Models stream blank lines around tool calls; hold whitespace back
        # until more text follows so turns do not end in empty lines.
        if not self._in_text:
            s = s.lstrip()
            if not s:
                return
            print(self.st.cyan("atom ▸ "), end="")
            self._in_text = True
            self._held_ws = ""
        text = self._held_ws + s
        body = text.rstrip()
        self._held_ws = text[len(body) :]
        print(body, end="", flush=True)

    def reasoning(self, s: str):
        if self.show_thinking:
            print(self.st.dim(s), end="", flush=True)
        elif not self._thinking_noted:
            print(self.st.dim("  (thinking…)"), flush=True)
            self._thinking_noted = True

    def turn_end(self):
        self._end_text()
        self._thinking_noted = False

    def tool_start(self, name: str, args: dict):
        self._end_text()
        shown = " ".join(f"{k}={_short(v, 48)}" for k, v in args.items())
        print(self.st.dim(f"  ⚙ {name} {shown}".rstrip()), flush=True)

    def tool_end(self, name: str, result: dict):
        line = summarize_result(name, result)
        bad = result.get("error") or result.get("declined_by_user")
        mark = self.st.red("  ✗ ") if bad else self.st.green("  ✓ ")
        print(mark + line, flush=True)
        for w in result.get("warnings", []) or []:
            print(self.st.yellow(f"    ! {w}"))

    def notice(self, line: str):
        self._end_text()
        print(self.st.yellow(f"  ! {line}"), flush=True)

    def progress(self, line: str):
        self._end_text()
        print(self.st.dim(f"    │ {line}"), flush=True)

    def approve(self, tool: str, args: dict, summary: str) -> bool:
        self._end_text()
        if not self.interactive:
            print(
                self.st.yellow(
                    f"  ⏸ {summary} -- needs approval; rerun with "
                    "--mode auto-safe or --yes"
                )
            )
            return False
        try:
            ans = (
                input(self.st.yellow(f"  ? {summary}  [y]es / [n]o / [a]lways: "))
                .strip()
                .lower()
            )
        except EOFError:
            return False
        if ans in ("a", "always"):
            if self.policy:
                self.policy.mode = "auto"
            return True
        return ans in ("y", "yes")


# --------------------------------------------------------------------------
# Brain selection
# --------------------------------------------------------------------------


def find_brain(args, ctx: ToolContext) -> Brain | None:
    extra = json.loads(args.brain_extra) if args.brain_extra else {}
    if args.brain_url:
        brain = connect(
            args.brain_url,
            args.brain_model,
            args.brain_key,
            api=args.brain_api,
            extra=extra,
            label="--brain-url",
        )
        if brain is None:
            sys.exit(f"error: no model answered at {args.brain_url} ({args.brain_api})")
    else:
        brain = discover_ranked(ctx, args.brain_key)
        if brain is None:
            return None
        brain.extra = extra
    for url in args.fallback_brain_url:
        fb = connect(url, None, args.brain_key, api=args.brain_api, label="fallback")
        if fb is not None:
            brain.fallbacks.append(fb)
        else:
            print(f"warning: fallback brain {url} does not answer; skipped")
    return brain


def _tool_capable(model_id: str) -> bool | None:
    """Can ATOM parse this served model's tool calls? None if unknown."""
    cfg_file = Path(model_id) / "config.json"
    if not cfg_file.is_file():
        return None
    try:
        arch = (json.loads(cfg_file.read_text()).get("architectures") or [None])[0]
    except (OSError, ValueError):
        return None
    return arch in TOOL_CAPABLE_ARCHS


def discover_ranked(ctx: ToolContext, api_key: str | None) -> Brain | None:
    """Environment first; then this agent's brain deployments; then local
    servers whose model can call tools; anything else last. Taking the first
    server that answers could make a 4-layer test checkpoint the brain."""
    if os.environ.get("ATOM_CHAT_BASE_URL") or os.environ.get("OPENAI_BASE_URL"):
        return discover([], api_key)
    brains = [d.port for d in ctx.store.active() if d.role == "brain"]
    # The agent's own service deployments are the user's workloads (maybe a
    # truncated test checkpoint): a brain only if nothing else answers.
    services = [d.port for d in ctx.store.active() if d.role != "brain"]
    others = [s["port"] for s in hardware.running_atom_servers()] + [8000, 30000]
    others = [p for p in others if p not in services]
    fallback = None
    for port in dict.fromkeys(brains + others + services):
        brain = connect(
            f"http://127.0.0.1:{port}", None, api_key, label=f"local :{port}"
        )
        if brain is None:
            continue
        if port in brains or (port not in services and _tool_capable(brain.model)):
            return brain
        if fallback is None:
            fallback = brain  # usable if nothing better answers
    return fallback


def bootstrap_brain(args, ctx: ToolContext, term: Terminal) -> Brain | None:
    """Deploy a model with ATOM and use it as the agent's own brain."""
    gpus = (
        [int(x) for x in args.bootstrap_gpus.split(",")]
        if args.bootstrap_gpus
        else None
    )
    print(term.st.bold(f"Bootstrapping a brain: {args.bootstrap}"))
    plan = ctx_tool(
        ctx,
        "plan_deployment",
        model=args.bootstrap,
        gpus=gpus,
        goal="agentic tool calling",
    )
    if not plan.get("plan_id"):
        print(term.st.red(f"cannot plan the brain: {plan.get('error')}"))
        return None
    print(term.st.dim(textwrap.indent(plan["command"], "    ")))
    res = ctx_tool(ctx, "launch_server", plan_id=plan["plan_id"], role="brain")
    if not res.get("deployment"):
        print(term.st.red(f"cannot launch the brain: {res}"))
        return None
    st = ctx_tool(ctx, "wait_until_ready", deployment=res["deployment"], timeout_min=30)
    if st.get("state") != "ready":
        print(
            term.st.red(
                f"brain {res['deployment']} is {st.get('state')}: {st.get('findings')}"
            )
        )
        return None
    return connect(
        st["base_url"], st["served_model"], label=f"bootstrapped {res['deployment']}"
    )


def ctx_tool(ctx: ToolContext, name: str, **kw) -> dict:
    return ToolBox(ctx).call(name, kw)


def offer_bootstrap(ctx: ToolContext, term: Terminal) -> str | None:
    """No brain anywhere: suggest local models ATOM can serve with tool calls."""
    scan = find_models(None, roots=ctx.cfg.model_roots, limit=None)
    capable = [m for m in scan["models"] if m["architecture"] in TOOL_CAPABLE_ARCHS]
    # Partial downloads would fail minutes into loading: never offer them.
    found = [m for m in capable if m["complete"]]
    partial = [m for m in capable if not m["complete"]]
    print(term.st.yellow("No LLM endpoint found for the agent's brain."))
    print("Options:\n  1) give an OpenAI-compatible URL (--brain-url)")
    if found:
        print("  2) let ATOM serve one of these local models as the brain:")
        for i, m in enumerate(found[:9]):
            print(
                f"       {chr(97 + i)}) {m['name']}  ({m['weights_gib']} GiB)  {m['path']}"
            )
    if partial:
        print(
            term.st.dim(
                f"     (not offered: {len(partial)} incomplete download(s), e.g. "
                f"{partial[0]['name']} with {partial[0]['shards']} shards)"
            )
        )
    if scan["truncated"]:
        print(
            term.st.dim(
                "     (disk scan hit its time limit; --model-root DIR narrows it)"
            )
        )
    print("  3) guided mode without an LLM (--guided)")
    if not term.interactive:
        return None
    ans = input("choice (URL, letter, or 3): ").strip()
    if ans.startswith("http"):
        return ans
    if len(ans) == 1 and ans.isalpha() and found and ord(ans) - 97 < len(found):
        return "bootstrap:" + found[ord(ans) - 97]["path"]
    if ans == "3":
        return "guided"
    return None


# --------------------------------------------------------------------------
# Guided mode (no LLM)
# --------------------------------------------------------------------------


def guided(ctx: ToolContext, term: Terminal) -> None:
    box = ToolBox(ctx)
    st = term.st
    report = box.call("inspect_system", {})
    print(
        st.bold("ATOM guided deployment")
        + "  "
        + machine_brief(report).replace("\n", "\n  ")
    )
    models = [
        m
        for m in box.call("find_models", {})["models"]
        if m["atom_supported"] and m["complete"]
    ]
    for i, m in enumerate(models[:20]):
        print(f"  {i + 1:>2}) {m['name']}  ({m['weights_gib']} GiB)")
    choice = input("model (number or path): ").strip()
    model = (
        models[int(choice) - 1]["path"]
        if choice.isdigit() and 0 < int(choice) <= len(models)
        else choice
    )
    goal = (
        input("goal [latency / throughput / agentic / enter to skip]: ").strip() or None
    )
    plan = box.call("plan_deployment", {"model": model, "goal": goal})
    if not plan.get("plan_id"):
        print(st.red(f"cannot plan: {plan.get('error')}"))
        return
    print(st.bold("\nPlan") + f" ({summarize_result('plan_deployment', plan)})")
    print(textwrap.indent(plan["command"], "    "))
    for line in plan.get("warnings", []) + plan.get("notes", []):
        print(st.yellow(f"  ! {line}"))
    res = box.call("launch_server", {"plan_id": plan["plan_id"]})
    if not res.get("deployment"):
        print(st.red(summarize_result("launch_server", res)))
        return
    status = box.call("wait_until_ready", {"deployment": res["deployment"]})
    print(summarize_result("wait_until_ready", status))
    if status.get("state") != "ready":
        diag = box.call("diagnose", {"deployment": res["deployment"]})
        for f in diag["status"].get("findings", []):
            print(st.red(f"  {f['title']}: {f.get('evidence', '')}"))
            print(f"    fix: {f.get('fix', '')}")
        return
    smoke = box.call("smoke_test", {"deployment": res["deployment"]})
    print(summarize_result("smoke_test", smoke))
    print(f"\nEndpoint: {status['base_url']}  model: {status['served_model']}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="atom",
        description="Chat with ATOM to deploy and operate LLM servers on AMD GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "command",
        nargs="?",
        choices=["chat", "status", "stop", "sessions", "mcp"],
        default="chat",
    )
    p.add_argument("target", nargs="?", help="deployment id for `stop`")
    p.add_argument("-p", "--prompt", help="send one message, print the answer, exit")
    p.add_argument("--brain-url", help="OpenAI-compatible base URL for the agent's LLM")
    p.add_argument(
        "--brain-model", help="model name at --brain-url (default: first listed)"
    )
    p.add_argument(
        "--brain-key",
        default=os.environ.get("ATOM_CHAT_API_KEY"),
        help="API key for --brain-url",
    )
    p.add_argument(
        "--brain-api",
        choices=["openai", "anthropic"],
        default=os.environ.get("ATOM_CHAT_API", "openai"),
        help="dialect of --brain-url: OpenAI chat completions (default) or "
        "Anthropic Messages (ATOM serves both)",
    )
    p.add_argument(
        "--fallback-brain-url",
        action="append",
        default=[],
        metavar="URL",
        help="brain to continue on if the main one stops answering (repeatable)",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="last",
        metavar="SESSION",
        help="continue a previous chat (default: the most recent)",
    )
    p.add_argument(
        "--brain-extra",
        help='JSON merged into every brain request, e.g. \'{"chat_template_kwargs":{"thinking":false}}\'',
    )
    p.add_argument(
        "--bootstrap",
        metavar="MODEL",
        help="serve MODEL with ATOM first and use it as the brain",
    )
    p.add_argument(
        "--bootstrap-gpus", help="HIP indices for the bootstrap brain, e.g. 0,1,2,3"
    )
    p.add_argument(
        "--guided", action="store_true", help="step-by-step deployment without an LLM"
    )
    p.add_argument("--runtime", choices=["auto", "local", "docker"], default="auto")
    p.add_argument(
        "--image",
        help="ATOM docker image (default: newest local rocm/atom*, else rocm/atom:latest)",
    )
    p.add_argument(
        "--model-root",
        action="append",
        default=[],
        help="extra directory to search for models",
    )
    p.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    p.add_argument(
        "--profile",
        choices=["customer", "developer"],
        default=os.environ.get("ATOM_CHAT_PROFILE", "customer"),
        help="customer: deployment skills (benchmark, accuracy, trace, debug); "
        "developer: every ATOM/AITER skill, incl. kernel authoring and PR review",
    )
    p.add_argument(
        "--skills-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="extra skills, laid out like a repo's .claude/ (skills/*/SKILL.md)",
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        default=os.environ.get("ATOM_CHAT_MODE", "ask"),
        help="read-only: change nothing; ask: confirm every action (default); "
        "auto-safe: act on leased/owned resources, ask for risky ones; auto: never ask",
    )
    p.add_argument("-y", "--yes", action="store_true", help="same as --mode auto")
    p.add_argument(
        "--allow-command",
        action="append",
        default=[],
        metavar="GLOB",
        help="shell commands auto-safe may run without asking (fnmatch), repeatable",
    )
    p.add_argument(
        "--show-thinking", action="store_true", help="print the brain's reasoning"
    )
    return p  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = AgentConfig(
        state_dir=Path(args.state_dir),
        model_roots=tuple(args.model_root) + DEFAULT_MODEL_ROOTS,
        runtime=args.runtime,
        image=args.image,
        skill_dirs=tuple(args.skills_dir),
    )
    if args.command == "mcp":
        from atom.agent.mcp_server import serve

        # The MCP client owns approvals; --mode read-only still binds.
        serve(
            cfg,
            mode="read-only" if args.mode == "read-only" else "auto",
            profile=args.profile,
        )
        return
    if args.command == "sessions":
        for sess in SessionLog.list(cfg.state_dir, limit=20):
            print(
                f"{sess['id']}  {sess['updated']}  {sess['turns']:>3} turns  "
                f"{(sess['first_message'] or '').strip()[:60]!r}"
            )
        return
    interactive = sys.stdin.isatty() and not args.prompt
    term = Terminal(args.show_thinking, interactive)
    policy = Policy("auto" if args.yes else args.mode, tuple(args.allow_command))
    term.policy = policy
    history: list[dict] | None = None
    if args.resume:
        sid = SessionLog.resolve(cfg.state_dir, args.resume)
        if sid is None:
            sys.exit(f"no unique session matches {args.resume!r} (see `atom sessions`)")
        _, history = SessionLog.load(cfg.state_dir, sid)
        log = SessionLog(cfg.state_dir, sid)
    else:
        log = SessionLog(cfg.state_dir)
    chatting = args.command == "chat" and not args.guided
    ctx = ToolContext(
        cfg,
        approve=term.approve,
        progress=term.progress,
        policy=policy,
        session=(
            log.id
            if chatting
            else f"cli-{args.command if not args.guided else 'guided'}"
        ),
        profile=args.profile,
    )
    deploy.reconcile(ctx.store, ctx.leases)

    if args.command == "status":
        print(json.dumps(ToolBox(ctx).call("deployment_status", {}), indent=1))
        return
    if args.command == "stop":
        if not args.target:
            sys.exit("usage: atom stop <deployment-id>")
        print(
            json.dumps(
                ToolBox(ctx).call("stop_server", {"deployment": args.target}), indent=1
            )
        )
        return
    if args.guided:
        guided(ctx, term)
        return

    brain = (
        bootstrap_brain(args, ctx, term) if args.bootstrap else find_brain(args, ctx)
    )
    if brain is None and not args.bootstrap:
        choice = offer_bootstrap(ctx, term)
        if choice == "guided":
            guided(ctx, term)
            return
        if choice and choice.startswith("bootstrap:"):
            args.bootstrap = choice.split(":", 1)[1]
            brain = bootstrap_brain(args, ctx, term)
        elif choice:
            brain = connect(choice, None, args.brain_key)
    if brain is None:
        sys.exit("no brain available (see --brain-url, --bootstrap, --guided)")

    brain.on_retry = term.notice
    for fb in brain.fallbacks:
        fb.on_retry = term.notice
    toolbox = ToolBox(ctx)
    report = toolbox.call("inspect_system", {})
    log.meta(brain=brain.describe(), mode=policy.mode, resumed=bool(history))
    session = ChatSession(
        brain,
        toolbox,
        build_system_prompt(ctx, report),
        Hooks(
            on_content=term.content,
            on_reasoning=term.reasoning,
            on_tool_start=term.tool_start,
            on_tool_end=term.tool_end,
            on_turn_end=term.turn_end,
            on_notice=term.notice,
        ),
        log=log,
        history=history,
    )
    st = term.st
    if args.prompt:
        answer = session.send(args.prompt)
        term.turn_end()
        if answer.startswith("[brain error]"):
            print(st.red(answer))
            sys.exit(1)
        return

    print(st.bold("ATOM Chat") + st.dim(f" -- brain: {brain.describe()}"))
    resumed = f"resumed {log.id} ({len(history)} messages)" if history else log.id
    print(
        st.dim(
            f"session {resumed}; mode {policy.mode}; profile {ctx.profile}, "
            f"{len(ctx.skills)} skills ({ctx.skill_source})"
        )
    )
    print(st.dim(machine_brief(report)))
    print(
        st.dim("Describe what you want to serve. /help for commands, Ctrl-D to exit.\n")
    )
    try:
        import readline  # noqa: F401 -- line editing and history for input()
    except ImportError:
        pass
    while True:
        try:
            text = input(st.bold("you ▸ ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not sys.stdin.isatty():
            print(text)  # piped input: keep the transcript readable
        if not text:
            continue
        if text.startswith("/"):
            if handle_slash(text, session, ctx, term):
                break
            continue
        try:
            answer = session.send(text)
        except KeyboardInterrupt:
            term.turn_end()
            print(st.yellow("(interrupted)"))
            continue
        term.turn_end()
        if answer.startswith("[brain error]"):
            print(st.red(answer))
        print()


def handle_slash(
    text: str, session: ChatSession, ctx: ToolContext, term: Terminal
) -> bool:
    cmd, _, arg = text.partition(" ")
    box = session.toolbox
    if cmd in ("/quit", "/exit"):
        return True
    if cmd == "/help":
        print(
            "/status [id]   deployments and their state\n"
            "/stop <id>     stop a deployment\n"
            "/log <id>      tail a deployment's log\n"
            "/brain [id|url] show or switch the agent's LLM\n"
            "/skills        skills the agent can follow (profile-dependent)\n"
            "/mode [m]      show or set: read-only | ask | auto-safe | auto\n"
            "/yes           same as /mode auto\n"
            "/reset         forget the conversation\n"
            "/quit"
        )
    elif cmd == "/status":
        print(
            json.dumps(
                box.call("deployment_status", {"deployment": arg or None}), indent=1
            )[:4000]
        )
    elif cmd == "/stop" and arg:
        term.tool_end("stop_server", box.call("stop_server", {"deployment": arg}))
    elif cmd == "/log":
        for line in box.call(
            "read_server_log", {"deployment": arg or None, "lines": 30}
        ).get("lines", []):
            print(term.st.dim(line))
    elif cmd == "/brain":
        if not arg:
            print(session.brain.describe())
        elif arg.startswith("http"):
            new = connect(arg, None, None)
            if new:
                session.switch_brain(new)
            print(session.brain.describe())
        else:
            res = box.call("switch_brain", {"deployment": arg})
            pending = ctx.pending_brain
            if pending:
                ctx.pending_brain = None
                session.switch_brain(
                    Brain(pending["base_url"], pending["model"], label=pending["label"])
                )
            print(res if res.get("error") else session.brain.describe())
    elif cmd == "/skills":
        for name, sk in sorted(ctx.skills.items()):
            print(f"{name:34s} {sk.description[:90]}")
    elif cmd == "/yes":
        ctx.policy.mode = "auto"
        print("mode: auto (nothing asks from now on)")
    elif cmd == "/mode":
        if arg in MODES:
            ctx.policy.mode = arg
        print(f"mode: {ctx.policy.mode}  (choices: {', '.join(MODES)})")
    elif cmd == "/reset":
        session.reset()
        print("conversation cleared")
    else:
        print("unknown command; /help")
    return False
