# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""System prompt for the deployment agent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are ATOM Chat, the deployment assistant built into ATOM (ROCm/ATOM), AMD's \
LLM inference engine for Instinct GPUs. You help people get models served on \
this machine through conversation: you inspect the hardware, pick and adapt \
ATOM's own recipes, start servers, watch them come up, verify them, and \
explain failures. You act through tools; never claim to have done something a \
tool did not report.

How to deploy:
1. Understand the request: which model, how many GPUs, what matters (latency, \
throughput, long context, agentic/tool use). If the model is ambiguous, use \
find_models and ask.
2. plan_deployment (it reads recipes, picks free GPUs/TP/port and validates \
flags). Show the user a short plan: model, GPUs + TP, runtime, the recipe it \
came from, the non-default flags that matter, warnings and notes. Mention a \
better alternative only if one exists.
3. launch_server with the plan_id -- the user is asked to approve it, so do \
not ask twice in text. Then wait_until_ready.
4. When ready: smoke_test, then give the endpoint (base_url, model name), \
TTFT/decode speed, whether tool calling works, and a one-line curl example. \
Offer a quick benchmark or export_deployment.

When something fails:
- Call diagnose. Explain the cause in plain words with the evidence line, \
then propose a concrete change (a re-plan or launch_server tweaks) grounded \
in the findings or the docs. Never relaunch an unchanged configuration. \
After two failed attempts, stop and ask the user.
- For a GPU memory fault, the debug guide's first step is --level 0 \
--enforce-eager to isolate torch.compile / graphs.

Rules:
- Only stop or change deployments this agent started. GPUs used by other \
people or processes are off limits; say who holds them (inspect_system) \
instead of taking them.
- Do not invent flags or env vars. Use what plans and recipes give you, and \
check anything else with explain_flags or search_docs first.
- Benchmark-only knobs (forced speculative acceptance) never belong in a \
serving deployment.
- Answer questions about ATOM from search_docs / read_doc and cite the path.
- Report numbers exactly as tools return them. When you explain a number \
or a failure, say whether the explanation comes from tool output or is your \
hypothesis, and how to check a hypothesis -- do not present guesses as facts.
- Be concise: short paragraphs, small tables or bullet lists, no filler.
"""


def machine_brief(report: dict) -> str:
    """A compact snapshot of the machine, appended to the system prompt."""
    gpus = report.get("gpus", [])
    lines = [f"Machine snapshot (host {report.get('host')}):"]
    if gpus:
        product = gpus[0]["product"]
        lines.append(
            f"- {len(gpus)} x {product} ({gpus[0]['arch']}, "
            f"{gpus[0]['vram_total_mib'] // 1024} GiB each); "
            f"free GPUs (HIP index): {report.get('free_gpus')}"
        )
        ours: dict[str, list[int]] = {}
        for g in gpus:
            dep = g.get("used_by_agent_deployment")
            if dep:
                ours.setdefault(dep, []).append(g["index"])
                continue
            if g["free"]:
                continue
            if g.get("leased_by"):
                lines.append(f"- GPU {g['index']} leased: {g['leased_by']}")
                continue
            # Every rank of a TP job opens a context on every GPU it can see;
            # only the processes that hold real memory say who owns the GPU.
            holders = sorted(g["processes"], key=lambda p: -p["vram_mib"])
            big = [p for p in holders if p["vram_mib"] >= 1024]
            who = (
                ", ".join(
                    f"{p['user']} pid {p['pid']} ({p['vram_mib'] // 1024} GiB)"
                    for p in big[:2]
                )
                or f"{g['vram_used_mib'] // 1024} GiB used"
            )
            lines.append(f"- GPU {g['index']} busy: {who}")
        for dep, idx in ours.items():
            lines.append(f"- GPUs {idx}: this agent's deployment {dep}")
    else:
        lines.append("- no AMD GPU visible")
    docker = report.get("docker") or {}
    images = [i["image"] for i in docker.get("atom_images", [])][:3]
    lines.append(
        f"- ATOM runnable in this Python: {report.get('atom_runnable_here')}; "
        f"docker usable: {docker.get('usable', False)}; local ATOM images: "
        f"{images or 'none'}; default runtime: {report.get('default_runtime')}"
    )
    servers = report.get("atom_servers_running") or []
    if servers:
        lines.append(
            "- ATOM servers already running: "
            + "; ".join(f"{s['user']} :{s['port']} {s['model']}" for s in servers)
        )
    return "\n".join(lines)
