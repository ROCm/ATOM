# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Turn "this model, these GPUs, this goal" into a launch spec.

The spec starts from the recipe command that best matches the model, GPU
architecture and goal, then is adapted to the machine (model path, TP, port,
GPUs) and checked against the server's real flag table, which is read
statically from the argparse calls so no torch import is needed.
"""

from __future__ import annotations

import ast
import difflib
import math
import re
import socket
from functools import lru_cache
from pathlib import Path

from atom.agent import hardware
from atom.agent.knowledge import KnowledgeBase, extract_launch_commands, tokenize
from atom.agent.models import inspect_model

ATOM_PKG = Path(__file__).resolve().parents[1]
FLAG_SOURCES = ("model_engine/arg_utils.py", "entrypoints/openai/api_server.py")

# Recipe flags that make outputs or numbers unrepresentative of a real
# deployment, with the reason given to the user when they are dropped.
BENCHMARK_ONLY_FLAGS = {
    "--spec-decode-acceptance-length": (
        "forces a fixed speculative acceptance length -- throughput becomes "
        "synthetic and output quality is not representative; benchmark-only"
    ),
    "--spec-decode-acceptance-rate": "same forced-acceptance knob as a rate",
}
# Recipe-local shell variables that are not engine settings.
NON_ENGINE_ENV = {"MODEL_PATH", "MODEL", "CONC", "PORT", "TP", "CAPTURE_SIZES"}
# Signs that a launch block belongs to a multi-node or PD deployment.
MULTI_NODE_FLAGS = (
    "--kv-transfer-config",
    "--kv_transfer_config",
    "--data-parallel-address",
    "--nnodes",
    "--node-rank",
)
# Words in a goal that say something about which recipe variant fits. Free
# text like "test checkpoint, one GPU, KV cache" must not steer the choice
# through incidental matches with recipe headings.
GOAL_WORDS = {
    "latency", "fast", "interactive", "ttft", "tpot", "throughput", "cost",
    "cheap", "efficient", "efficiency", "agentic", "agent", "coding", "tool",
    "long", "context", "mtp", "speculative", "dspark", "eagle", "eagle3",
    "fp4", "mxfp4", "fp8", "ep", "dp", "tbo", "max", "multimodal", "vision",
    "reasoning", "thinking", "offload", "lmcache",
}  # fmt: skip
ARCH_WORDS = {
    "gfx950": ("mi355x", "mi350x", "mi355", "mi350", "gfx950"),
    "gfx942": ("mi300x", "mi325x", "mi308x", "mi300", "gfx942"),
    "gfx1250": ("mi455x", "mi450", "gfx1250"),
    "gfx1201": ("gfx1201", "9070", "r9700"),
    "gfx1151": ("gfx1151", "strix"),
}


# --------------------------------------------------------------------------
# Server flag table
# --------------------------------------------------------------------------


def _norm(flag: str) -> str:
    """FlexibleArgumentParser accepts --kv_cache_dtype and --kv-cache-dtype."""
    return flag.replace("_", "-") if flag.startswith("--") else flag


def parse_flag_table(sources: list[str]) -> dict[str, dict]:
    """Every ``add_argument`` option in the given Python sources."""
    flags: dict[str, dict] = {}
    for text in sources:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            names = [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
            if not names or not names[0].startswith("-"):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            action = kw.get("action")
            if isinstance(action, ast.Constant):
                action = action.value
            elif isinstance(action, ast.Attribute):  # argparse.BooleanOptionalAction
                action = action.attr
            elif isinstance(action, ast.Name):
                action = action.id
            nargs = kw.get("nargs")
            if action == "BooleanOptionalAction":
                names = names + ["--no-" + n[2:] for n in names if n.startswith("--")]
            entry = {
                "names": names,
                "takes_value": action
                not in (
                    "store_true",
                    "store_false",
                    "store_const",
                    "count",
                    "help",
                    "BooleanOptionalAction",
                )
                and not (isinstance(nargs, ast.Constant) and nargs.value == 0),
                "optional_value": isinstance(nargs, ast.Constant)
                and nargs.value == "?",
                "help": "",
                "choices": None,
                "default": None,
            }
            try:
                if "help" in kw:
                    entry["help"] = " ".join(str(ast.literal_eval(kw["help"])).split())
            except (ValueError, TypeError, SyntaxError):
                entry["help"] = ast.unparse(kw["help"])[:400]
            for key in ("choices", "default"):
                if key in kw:
                    try:
                        entry[key] = ast.literal_eval(kw[key])
                    except (ValueError, TypeError, SyntaxError):
                        entry[key] = ast.unparse(kw[key])[:80]
            for n in names:
                flags[_norm(n)] = entry
    return flags


@lru_cache(maxsize=1)
def server_flags() -> dict[str, dict]:
    """The flag table of the ATOM this package belongs to."""
    texts = []
    for rel in FLAG_SOURCES:
        try:
            texts.append((ATOM_PKG / rel).read_text())
        except OSError:
            continue
    return parse_flag_table(texts)


def validate_args(args: list[str], table: dict | None = None) -> list[str]:
    """Problems argparse would reject, without starting a server."""
    table = table or server_flags()
    if not table:
        return []
    problems = []
    i = 0
    while i < len(args):
        tok = args[i]
        if not tok.startswith("-") or re.fullmatch(r"-\d+(\.\d+)?", tok):
            problems.append(f"stray value {tok!r} (no flag before it)")
            i += 1
            continue
        name, eq, inline = tok.partition("=")
        entry = table.get(_norm(name))
        if entry is None:
            close = difflib.get_close_matches(_norm(name), list(table), n=2)
            hint = f"; did you mean {' or '.join(close)}?" if close else ""
            problems.append(f"unknown flag {name}{hint}")
            i += 1
            continue
        value = inline if eq else None
        if entry["takes_value"] and not eq:
            nxt = args[i + 1] if i + 1 < len(args) else None
            if nxt is not None and not (nxt.startswith("--") or nxt == "-tp"):
                value = nxt
                i += 1
            elif not entry["optional_value"]:
                problems.append(f"{name} needs a value")
        choices = entry.get("choices")
        if (
            value is not None
            and isinstance(choices, (list, tuple))
            and value not in [str(c) for c in choices]
        ):
            problems.append(f"{name}={value} not in {list(choices)}")
        i += 1
    return problems


def explain_flags(names: list[str]) -> dict:
    table = server_flags()
    out = {}
    for n in names:
        n = n if n.startswith("-") else "--" + n
        entry = table.get(_norm(n))
        if entry is None:
            out[n] = {
                "error": "unknown flag",
                "close": difflib.get_close_matches(_norm(n), list(table), n=3),
            }
        else:
            out[n] = {k: entry[k] for k in ("names", "help", "default", "choices")}
    return out


# --------------------------------------------------------------------------
# Arg list helpers
# --------------------------------------------------------------------------


def int_arg(args: list[str], *names: str, default: int | None = None) -> int | None:
    """An integer flag value; a recipe placeholder that did not expand -> default."""
    value = get_arg(args, *names)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def get_arg(args: list[str], *names: str) -> str | None:
    norm = {_norm(n) for n in names}
    for i, a in enumerate(args):
        flag, eq, val = a.partition("=")
        if _norm(flag) in norm:
            return val if eq else (args[i + 1] if i + 1 < len(args) else None)
    return None


def remove_arg(args: list[str], *names: str) -> list[str]:
    norm = {_norm(n) for n in names}
    table = server_flags()
    out, i = [], 0
    while i < len(args):
        flag, eq, _ = args[i].partition("=")
        if _norm(flag) in norm:
            entry = table.get(_norm(flag))
            takes = entry["takes_value"] if entry else True
            has_next = i + 1 < len(args) and not args[i + 1].startswith("--")
            i += 2 if (takes and not eq and has_next) else 1
            continue
        out.append(args[i])
        i += 1
    return out


def set_arg(args: list[str], name: str, value: str | None, aliases=()) -> list[str]:
    args = remove_arg(args, name, *aliases)
    return args + ([name] if value is None else [name, str(value)])


# --------------------------------------------------------------------------
# Recipe selection
# --------------------------------------------------------------------------


def _name_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower()))


def recipe_env(text: str) -> dict[str, str]:
    """Exports from env-only blocks under an "env"/"environment" heading."""
    env: dict[str, str] = {}
    pattern = r"^(#{1,4})\s+([^\n]*)$|^```[^\n]*\n(.*?)^```"
    heading = ""
    for m in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
        if m.group(1):
            heading = m.group(2).lower()
            continue
        if "env" not in heading:
            continue
        lines = [
            ln.strip()
            for ln in (m.group(3) or "").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if lines and all(ln.startswith("export ") for ln in lines):
            for ln in lines:
                k, _, v = ln[len("export ") :].partition("=")
                env[k.strip()] = v.split("#")[0].strip().strip("'\"")
    return env


def score_command(
    cmd: dict,
    model_tokens: set[str],
    arch: str | None,
    goal: str | None,
    free: int,
    tp_fixed: bool = False,
) -> tuple[float, list[str]]:
    why = []
    recipe_model = get_arg(cmd["args"], "--model") or ""
    rtoks = _name_tokens(Path(recipe_model).name) - {"ai"}
    if not rtoks:
        rtoks = _name_tokens(Path(cmd["recipe"]).stem)
    overlap = rtoks & model_tokens
    score = 10.0 * len(overlap) / max(len(rtoks), 1)
    cmd["exact_model"] = bool(rtoks) and overlap == rtoks
    # A version the checkpoint does not carry (V4 recipe for a V4.1 model).
    versions = {t for t in rtoks if re.fullmatch(r"v?\d+(\.\d+)+|v\d+", t)}
    if versions - model_tokens:
        score -= 4
    why.append(f"model match {sorted(overlap)} of {sorted(rtoks)}")
    context = " ".join([cmd["recipe"], cmd["section"]]).lower()
    env_arch = " ".join(cmd["env"].values()).lower()
    if arch:
        ours = ARCH_WORDS.get(arch, (arch,))
        others = [w for a, ws in ARCH_WORDS.items() if a != arch for w in ws]
        if any(w in context for w in ours):
            score += 3
            why.append(f"written for {arch}")
        elif any(w in context or w in env_arch for w in others):
            score -= 6
            why.append("written for a different GPU")
    for flag in MULTI_NODE_FLAGS:
        if get_arg(cmd["args"], flag) is not None or flag in cmd["args"]:
            score -= 20
            why.append("multi-node / PD command")
            break
    tp = int_arg(cmd["args"], "-tp", "--tensor-parallel-size", default=1)
    if tp > free:
        # Fatal only when the recipe's TP will be used as is; a sibling's
        # recipe, or a user-chosen TP, gets resized anyway.
        score -= 8 if cmd["exact_model"] and not tp_fixed else 1
        why.append(f"recipe uses {tp} GPUs, {free} available")
    goal_tokens = set(tokenize(goal or "")) & GOAL_WORDS
    hits = goal_tokens & set(tokenize(context))
    if hits:
        score += 2 * len(hits)
        why.append(f"goal words {sorted(hits)}")
    if re.search(r"experiment|debug|profil|repro", context) and not hits:
        score -= 2
    # Same recipe, different TP: latency wants more ranks, throughput per GPU
    # wants fewer (and more replicas).
    if goal_tokens & {"latency", "fast", "interactive", "ttft", "tpot"}:
        score += 0.5 * math.log2(tp)
    elif goal_tokens & {"throughput", "cost", "cheap", "efficient", "efficiency"}:
        score -= 0.5 * math.log2(tp)
    ep = any(
        f in cmd["args"] for f in ("--enable-expert-parallel", "--enable-dp-attention")
    )
    if ep and not goal_tokens & {"throughput", "ep", "dp", "max"}:
        score -= 1.5
    return score, why


def candidate_commands(kb: KnowledgeBase, model_name: str, model_type: str | None):
    """Launch commands from recipes that are plausibly about this model."""
    query = f"{model_name} {model_type or ''}"
    paths = []
    for hit in kb.search(query, k=25, path_prefix="recipes/"):
        if hit["path"] not in paths and not hit["path"].startswith(
            ("recipes/atom_vllm", "recipes/atom_sglang", "recipes/atom_verl")
        ):
            paths.append(hit["path"])
    out = []
    for p in paths[:8]:
        text = kb.files[p]
        shared_env = recipe_env(text)
        for cmd in extract_launch_commands(text, p):
            if not cmd["uses_docker"]:
                cmd["env"] = {**shared_env, **cmd["env"]}
                out.append(cmd)
    return out


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def port_free(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(preferred: int | None, taken: set[int]) -> int:
    start = preferred or 8000
    for port in range(start, start + 200):
        if port not in taken and port_free(port):
            return port
    return start


def pick_gpus(gpus: list, count: int, reserved: set[int]) -> list[int] | None:
    free = [g.index for g in gpus if g.free and g.index not in reserved]
    if len(free) < count:
        return None
    # Keep a TP group on one NUMA half when possible: on 8-GPU boxes HIP
    # 0-3 / 4-7 usually share a socket, so prefer an aligned contiguous run.
    for start in range(0, len(gpus), count):
        block = list(range(start, start + count))
        if all(i in free for i in block):
            return block
    return free[:count]


def make_plan(
    kb: KnowledgeBase,
    model: str,
    *,
    gpus: list[int] | None = None,
    tp: int | None = None,
    goal: str | None = None,
    port: int | None = None,
    recipe: str | None = None,
    section: str | None = None,
    max_model_len: int | None = None,
    reserved_gpus: set[int] = frozenset(),
    taken_ports: set[int] = frozenset(),
    model_roots: tuple[str, ...] = (),
    flag_table: dict | None = None,
    flag_source: str = "this checkout",
    known_env: set[str] | None = None,
) -> dict:
    gpu_list = hardware.list_gpus(with_names=False)
    vram = gpu_list[0].vram_total_mib if gpu_list else None
    arch = gpu_list[0].arch if gpu_list else None
    info = inspect_model(model, vram, model_roots)
    if not info.get("found"):
        return {"ok": False, "error": f"model {model!r} not found", **info}
    if not info.get("atom_supported"):
        return {"ok": False, "error": info.get("warning"), "model_info": info}
    model_path = info["path"]
    name = Path(model_path).name
    if "/snapshots/" in model_path:  # HF cache: name the repo, not the hash
        name = Path(model_path).parents[1].name.split("--", 1)[-1]
    model_tokens = _name_tokens(name) | _name_tokens(info.get("model_type") or "")

    free_count = len([g for g in gpu_list if g.free and g.index not in reserved_gpus])
    want = len(gpus) if gpus else free_count
    scored = []
    for cmd in candidate_commands(kb, name, info.get("model_type")):
        s, why = score_command(
            cmd, model_tokens, arch, goal, want, tp_fixed=bool(tp or gpus)
        )
        scored.append((s, why, cmd))
    scored.sort(key=lambda x: -x[0])
    if recipe or section:
        chosen = [
            x
            for x in scored
            if (not recipe or recipe in x[2]["recipe"])
            and (not section or section.lower() in x[2]["section"].lower())
        ]
        if not chosen:
            return {
                "ok": False,
                "error": f"no launch command in recipe={recipe!r} section={section!r}",
                "available": [
                    {"recipe": c["recipe"], "section": c["section"]}
                    for _, _, c in scored[:12]
                ],
            }
        scored = chosen + [x for x in scored if x not in chosen]
    best = scored[0] if scored and (scored[0][0] >= 5 or recipe or section) else None

    warnings: list[str] = []
    notes: list[str] = []
    env: dict[str, str] = {}
    if best:
        _, why, cmd = best
        args = list(cmd["args"])
        env = {k: v for k, v in cmd["env"].items() if k not in NON_ENGINE_ENV}
        source = {"recipe": cmd["recipe"], "section": cmd["section"], "why": why}
    else:
        args = ["--kv_cache_dtype", "fp8"]
        source = {"recipe": None, "why": ["no matching recipe; generic ATOM defaults"]}
        warnings.append(
            "No recipe matches this model -- using generic defaults. Expect to "
            "tune flags; check docs/model_support_guide.md."
        )
    env.pop("HIP_VISIBLE_DEVICES", None)  # chosen per deployment below
    env.setdefault("AITER_LOG_LEVEL", "WARNING")

    recipe_tp = int_arg(args, "-tp", "--tensor-parallel-size")
    sizing = info.get("sizing") or {}
    if best and not best[2].get("exact_model") and recipe_tp:
        # A sibling's recipe (V4-Pro's for V4-Flash) carries the sibling's TP.
        notes.append(
            f"recipe is for {get_arg(best[2]['args'], '--model')}; TP sized "
            "from this checkpoint instead of the recipe's TP"
            f"{recipe_tp}. Raise TP for lower latency or more KV room."
        )
        recipe_tp = None
    tp = tp or (len(gpus) if gpus else None) or recipe_tp or sizing.get("min_tp") or 1
    if not recipe_tp and sizing.get("min_tp") and tp < sizing["min_tp"]:
        warnings.append(
            f"TP{tp} may not fit: the estimate says TP{sizing['min_tp']} "
            f"({info.get('weights_gib')} GiB of weights)."
        )
    if gpus:
        if len(gpus) != tp:
            return {"ok": False, "error": f"{len(gpus)} GPUs given for TP{tp}"}
        busy = [
            g.index
            for g in gpu_list
            if g.index in gpus and (not g.free or g.index in reserved_gpus)
        ]
        if busy:
            warnings.append(f"GPUs {busy} are not free (see inspect_system).")
        chosen_gpus = list(gpus)
    else:
        chosen_gpus = pick_gpus(gpu_list, tp, set(reserved_gpus))
        if chosen_gpus is None:
            return {
                "ok": False,
                "error": f"TP{tp} needs {tp} free GPUs; only {free_count} free",
                "free_gpus": [g.index for g in gpu_list if g.free],
            }

    for flag, reason in BENCHMARK_ONLY_FLAGS.items():
        if get_arg(args, flag) is not None:
            args = remove_arg(args, flag)
            notes.append(f"dropped {flag} from the recipe: {reason}")
    args = remove_arg(args, "--model", "--server-port", "-tp", "--tensor-parallel-size")
    chosen_port = pick_port(port, set(taken_ports))
    args = [
        "--model",
        model_path,
        "-tp",
        str(tp),
        "--server-port",
        str(chosen_port),
    ] + args
    if info.get("needs_trust_remote_code") and "--trust-remote-code" not in args:
        args.append("--trust-remote-code")
    if max_model_len:
        args = set_arg(args, "--max-model-len", max_model_len)
    if port and chosen_port != port:
        warnings.append(f"port {port} is busy; using {chosen_port}")

    problems = validate_args(args, flag_table)
    if problems:
        warnings.append(f"flag check against {flag_source}: " + "; ".join(problems))
    stray = [
        k
        for k in env
        if known_env and k.startswith(("ATOM_", "AITER_")) and k not in known_env
    ]
    if stray:
        warnings.append(
            f"env vars not read anywhere in {flag_source}: {', '.join(stray)} "
            "(a typo, or newer than this ATOM/AITER)"
        )
    startup_min = round(2 + (info.get("weights_gib") or 0) / tp / 40, 1)
    return {
        "ok": True,
        "model": model_path,
        "model_info": {
            k: info.get(k)
            for k in (
                "architecture",
                "model_type",
                "weights_gib",
                "quantization",
                "max_context",
                "moe_experts",
            )
        },
        "gpu_arch": arch,
        "gpus": chosen_gpus,
        "tp": tp,
        "port": chosen_port,
        "env": env,
        "args": args,
        "source": source,
        "alternatives": [
            {"recipe": c["recipe"], "section": c["section"], "score": round(s, 1)}
            for s, _, c in scored[1:6]
            if s >= 3
        ],
        "sizing": sizing.get("per_tp", {}).get(tp),
        "expected_startup_minutes": startup_min,
        "flags_checked_against": flag_source,
        "warnings": warnings,
        "notes": notes,
    }
