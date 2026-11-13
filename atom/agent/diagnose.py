# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Read an ATOM server log: how far startup got, and what went wrong.

The failure signatures and their fixes are condensed from
.claude/commands/debug-guide.md, scripts/wait_server_ready.sh and the
recipes; ``doc`` points at the section with the full story.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Startup milestones in the order a healthy server passes them.
PHASES = (
    (r"Loading tokenizer", "loading tokenizer"),
    (r"Initializing engine", "initializing engine"),
    (r"Creating EngineCore|ModelRunner rank=", "starting GPU workers"),
    (r"Loading safetensors shards|weight load phases", "loading weights"),
    (r"Model load done", "weights loaded"),
    (r"Dynamo bytecode transform|Compiling a graph|torch\.compile", "compiling"),
    (r"Model warmup done", "warmup done"),
    (r"Memory budget:|Binding KV cache", "allocating KV cache"),
    (r"Capturing", "capturing CUDA graphs"),
    (
        r"Server started successfully|Application startup complete|Uvicorn running on",
        "serving",
    ),
)
_PROGRESS = re.compile(
    r"(Loading safetensors shards|Capturing[^:]*):?[^\r\n]*?(\d+)/(\d+)"
)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class Signature:
    key: str
    pattern: str
    title: str
    fix: str
    doc: str
    fatal: bool = True


SIGNATURES = (
    Signature(
        "bad_flag",
        r"openai_server\.py: error:|error: unrecognized arguments|error: argument ",
        "The server rejected its command line",
        "A flag is misspelled, needs a value, or has an invalid choice. Check "
        "it with explain_flags and relaunch.",
        "atom/model_engine/arg_utils.py",
    ),
    Signature(
        "port_in_use",
        r"Address already in use|address already in use|Errno 98",
        "The port is already taken",
        "Another server holds the port. Pick a free --server-port.",
        "",
    ),
    Signature(
        "model_not_found",
        r"config\.json.*(No such file|does not exist|not found)|"
        r"Repository Not Found|HFValidationError|LocalEntryNotFoundError|"
        r"We couldn't connect to 'https://huggingface\.co'|"
        r"Incorrect path_or_model_id",
        "The model files are not visible to the server",
        "Check the path. In docker the model directory must be mounted (the "
        "agent mounts it read-only at the same path); with HF_HUB_OFFLINE=1 "
        "a repo id must already be in the HF cache.",
        "",
    ),
    Signature(
        "unsupported_arch",
        r"KeyError: '\w+For(CausalLM|ConditionalGeneration)'|"
        r"Model architectures? .* (is|are) not supported",
        "ATOM has no implementation for this architecture",
        "Only architectures in atom/models/registry.py are served natively. "
        "The vLLM/SGLang plugin backends (recipes/atom_vllm, recipes/atom_sglang) "
        "may cover it.",
        "docs/model_support_guide.md",
    ),
    Signature(
        "trust_remote_code",
        r"trust_remote_code=True|requires you to execute the configuration file",
        "The checkpoint ships custom code",
        "Add --trust-remote-code.",
        "",
    ),
    Signature(
        "kv_cache_alloc",
        r"Failed to allocate kv cache|No available memory for the cache blocks|"
        r"num_blocks\W+0\b|available_for_kv=-|exceeds available KV budget|"
        r"KV budget \(-",
        "No room left for the KV cache",
        "Weights + activations + graphs fill the GPUs. Use more GPUs (higher "
        "-tp), --kv_cache_dtype fp8, or lower --max-num-seqs / "
        "--max-num-batched-tokens / --max-model-len. Raising "
        "--gpu-memory-utilization (default 0.9) only helps if no one else "
        "shares the GPUs.",
        ".claude/commands/debug-guide.md#KV Cache / Block Manager",
    ),
    Signature(
        "gpu_oom",
        r"OutOfMemoryError|out of memory|hipErrorOutOfMemory|HIP out of memory",
        "GPU out of memory",
        "At startup: lower --max-num-seqs or --max-num-batched-tokens, use "
        "fp8 KV, or raise TP. Make sure no other process holds memory on the "
        "chosen GPUs (inspect_system shows tenants).",
        ".claude/commands/debug-guide.md#GPU / CUDA",
    ),
    Signature(
        "host_oom",
        r"OOMKilled|Killed process|Cannot allocate memory|"
        r"page-locked .* failed|hipErrorHostMemoryAlreadyRegistered",
        "Host memory exhausted",
        "Some models pin large host buffers (DeepSeek-V4.1 Engram page-locks "
        "~46 GiB per rank). Free host RAM, run fewer replicas, or set "
        "ATOM_DISABLE_MMAP=true as the recipes do.",
        "",
    ),
    Signature(
        "gpu_fault",
        r"Memory access fault|HSA_STATUS_ERROR|hipErrorIllegalAddress|"
        r"GPU core dump|illegal memory access",
        "GPU memory fault",
        "First isolate: relaunch with --level 0 --enforce-eager (disables "
        "torch.compile and graphs). If it persists it is a kernel bug -- note "
        "the AITER version and reproduce under scripts/run_debug_agent.sh. "
        "Delete gpucore.* files: one fault can dump tens of GB per rank.",
        ".claude/commands/debug-guide.md#GPU / CUDA",
    ),
    Signature(
        "no_gpu",
        r"No HIP GPUs are available|hipErrorNoDevice|no ROCm-capable device|"
        r"torch\.cuda\.is_available\(\) (is|returned) False",
        "The server cannot see a GPU",
        "In docker pass --device=/dev/kfd --device=/dev/dri --group-add video; "
        "check HIP_VISIBLE_DEVICES uses HIP indices (inspect_system 'index').",
        "docs/installation.rst",
    ),
    Signature(
        "compile_failed",
        r"InductorError|BackendCompilerFailed|torch\._dynamo\.exc|"
        r"failed to compile the model",
        "torch.compile failed",
        "Clear the compile cache (rm -rf ~/.cache/atom/*) and retry; if it "
        "still fails, run with --level 0 to skip torch.compile.",
        ".claude/commands/debug-guide.md#Compilation",
    ),
    Signature(
        "triton_cluster_dims",
        r"cluster_dims",
        "Triton/Inductor version mismatch (cluster_dims)",
        "Inductor's compile-time autotune hit ROCm Triton metadata without "
        "cluster_dims. Use the torch/triton shipped in the ATOM image, or run "
        "with --level 0.",
        "atom/utils/compiler_inferface.py",
    ),
    Signature(
        "collective_timeout",
        r"Watchdog caught collective operation timeout|NCCL (error|timeout)|"
        r"RCCL.*(error|timeout)|ProcessGroupNCCL.*(timeout|Timeout)",
        "A collective (RCCL) timed out",
        "One rank diverged or died. Look for the first error in the log above "
        "this line; set NCCL_DEBUG=INFO to see per-rank detail.",
        ".claude/commands/debug-guide.md#NCCL / Distributed",
    ),
    Signature(
        "missing_module",
        r"ModuleNotFoundError: No module named '([\w.]+)'",
        "A Python package is missing",
        "The runtime environment lacks a dependency -- use the ATOM docker "
        "image (runtime=docker) rather than a bare Python environment.",
        "",
    ),
    Signature(
        "worker_died",
        r"SHUTDOWN signal|proc died|EXECUTOR_FAILED|EngineCore.*(died|exited)",
        "An engine worker died",
        "This is the symptom; the cause is the first error above it in the log.",
        "",
    ),
    Signature(
        "traceback",
        r"Traceback \(most recent call last\)",
        "A Python traceback was logged",
        "Not always fatal (torch logs tracebacks at WARNING during warmup). "
        "Read the exception line that follows it.",
        "",
        fatal=False,
    ),
)


def clean(text: str) -> str:
    """Drop ANSI codes and collapse tqdm's \\r redraws to their last frame."""
    text = _ANSI.sub("", text)
    return "\n".join(line.split("\r")[-1] for line in text.split("\n"))


def analyze_log(text: str, max_findings: int = 4) -> dict:
    raw = text
    text = clean(text)
    phase, phase_line = "not started", None
    for line_no, line in enumerate(text.splitlines()):
        for pattern, name in PHASES:
            if re.search(pattern, line):
                # A phase never moves backwards (shards reload for drafters).
                order = [n for _, n in PHASES]
                if phase == "not started" or order.index(name) >= order.index(phase):
                    phase, phase_line = name, line_no
    progress = None
    if phase in ("loading weights", "capturing CUDA graphs"):
        # Only the current phase's counter; an old "6/6" says nothing now.
        want = "Loading" if phase == "loading weights" else "Capturing"
        for m in _PROGRESS.finditer(raw[-20000:]):
            if m.group(1).startswith(want):
                progress = f"{m.group(1).strip()} {m.group(2)}/{m.group(3)}"
    findings = []
    seen = set()
    for sig in SIGNATURES:
        for m in re.finditer(sig.pattern, text):
            if sig.key in seen:
                break
            seen.add(sig.key)
            start = text.rfind("\n", 0, m.start()) + 1
            end = text.find("\n", m.end())
            evidence = text[start : end if end != -1 else None].strip()[:300]
            findings.append(
                {
                    "key": sig.key,
                    "title": sig.title,
                    "evidence": evidence,
                    "fix": sig.fix,
                    "doc": sig.doc,
                    "fatal": sig.fatal,
                }
            )
    # Causes first, then "worker died" (the effect), then non-fatal noise.
    fatal_first = sorted(
        findings, key=lambda f: (not f["fatal"], f["key"] == "worker_died")
    )
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {
        "phase": phase,
        "phase_line": phase_line,
        "progress": progress,
        "findings": fatal_first[:max_findings],
        "last_lines": [ln[:240] for ln in lines[-12:]],
    }


def last_exception(text: str) -> str | None:
    """The final exception line of the last traceback, if any."""
    text = clean(text)
    at = text.rfind("Traceback (most recent call last)")
    if at == -1:
        return None
    for line in text[at:].splitlines()[1:]:
        if re.match(r"^[\w.]+(Error|Exception|Interrupt)\b|^\w+Error:", line.strip()):
            return line.strip()[:400]
    return None
