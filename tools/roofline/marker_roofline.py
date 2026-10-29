#!/usr/bin/env python3
"""Operator roofline for ATOM, driven by record_function markers.

Why markers. ATOM already labels every module with a `record_function` whose name
carries the operator's identity, its layer index, and -- for GEMMs -- the problem
shape and the input/weight/output dtypes:

    layers.18.attn.indexer.wq_b[M=1,N=8192,K=1536,a=torch.float8_e4m3fn,...]

The profiler emits the same label on the GPU side as a `gpu_user_annotation` with
a real duration. So one trace carries everything a roofline needs -- identity,
shape, dtype, measured time -- with no name-matching heuristics, no landmark
anchors, and no hand-maintained shape tables. That is the whole design.

Two tiers, and the data splits itself:

    marker has a shape  ->  priced against the roofline (2*M*N*K etc.)
    marker has no shape ->  gap tier: measured time is still counted, no bound

Scope. This is a thermometer, not a microscope: magnitudes and bound categories
are meaningful, absolute numbers are not. Ceilings are theoretical peaks with no
derate, so `gap = measured / t_roofline >= 1` by construction and a large-GEMM
gap of 1.1-1.3x is normal, not a finding.

CUDA-graph note. Decode replays a captured graph, and markers do not fire during
replay -- so a run trace yields prefill operators only. Decode markers live in the
per-bs capture trace under `capture_traces/`; pricing them needs that file's
kernel-launch events to bridge marker -> kernel. Not implemented here yet.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import dataclasses
import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import formulas
import yaml

# A short label for grouping and display. The FULL kernel name is kept alongside
# it: truncating at store time is lossy in a file people grep, and two kernels can
# share a prefix while differing only in their template arguments.
KERNEL_TRIM = re.compile(r"<.*|\(.*")

MARKER_RE = re.compile(r"^layers\.(\d+)\.(.+?)(?:\[(.*)\])?$")
PHASE_RE = re.compile(r"^(prefill|decode)\[")
KV_RE = re.compile(r"(\w+)=([\w.]+)")

# Bytes per element. Keys are the torch dtype spellings that appear in markers.
DTYPE_BYTES = {
    "torch.float8_e4m3fn": 1.0,
    "torch.float8_e5m2": 1.0,
    "torch.bfloat16": 2.0,
    "torch.float16": 2.0,
    "torch.float32": 4.0,
    "torch.uint8": 1.0,
    "torch.int8": 1.0,
}
# Which matrix peak a dtype is executed against.
DTYPE_PEAK = {
    "torch.float8_e4m3fn": "matrix_fp8",
    "torch.float8_e5m2": "matrix_fp8",
    "torch.bfloat16": "matrix_bf16",
    "torch.float16": "matrix_fp16",
    "torch.float32": "matrix_fp32",
}
# Lower is wider. Used to pick the peak when a and w disagree: the wider input
# governs the MFMA mode, which is the conservative (lower-peak) choice.
PEAK_ORDER = ["matrix_fp32", "matrix_bf16", "matrix_fp16", "matrix_fp8", "matrix_fp4"]

# A marker names a MODULE, not a kernel. `attn.wo_b` on TP4 prefill contains a
# quantize, the GEMM, and the tensor-parallel all-reduce -- and the all-reduce is
# 74% of it. Comparing the module's wall time against a pure-GEMM roofline puts
# the wrong number in the denominator, so kernels are classified and only the
# matmul time is measured against the matmul ceiling. Everything else is reported
# beside it instead of being hidden inside it.
# Order matters and gemm MUST come before quant: ck_tile names its fused kernel
# `QuantGemmMultiD...`, which is a GEMM that quantizes inline, not a quantizer.
# Getting this backwards charged ffn.shared_experts.w2 only its 0.39 ms Tensile
# tail instead of its 3.76 ms main kernel and reported 300% of the fp8 peak.
# "other" reads as "unattributed", which it is not -- every kernel here has a
# marker. Name the classes for what they do so a reader is not left guessing what
# the largest slice of decode actually contains.
KERNEL_CLASS = [
    # cross_device_reduce is ATOM's custom all-reduce; missing it here reported
    # decode communication as 1.9% when it is 7.7%.
    (
        "comm",
        re.compile(
            r"nccl|rccl|all_?reduce|reduce_?scatter|all_?gather|cross_device_reduce|custom_ar",
            re.IGNORECASE,
        ),
    ),
    # mfma_* are the MoE expert matmuls -- a matmul kernel that never says "gemm".
    (
        "gemm",
        re.compile(
            r"gemm|Cijk_|hgemm|_mm_|matmul|xdl_cshuffle|^mfma_|_mfma_", re.IGNORECASE
        ),
    ),
    (
        "attn",
        re.compile(
            r"attn|attention|mqa_logits|flash|paged|^pa_|_pa_|compress|indexer|radix"
            r"|^mla_|::mla_|_fwd_kernel|csa_|topk",
            re.IGNORECASE,
        ),
    ),
    (
        "moe",
        re.compile(
            r"moe_sort|moe_sorting|mxfp4_moe|act_and_mul|topk_softplus", re.IGNORECASE
        ),
    ),
    # norm before quant on purpose: a fused `add_rmsnorm_quant` is a norm that
    # quantises on the way out, not a quantiser. Pure `*_scaled_quant` has no norm
    # token and still lands in quant.
    ("norm", re.compile(r"rmsnorm|layernorm|_norm_|norm_rope|rope", re.IGNORECASE)),
    ("quant", re.compile(r"quant", re.IGNORECASE)),
    ("mhc", re.compile(r"mhc_", re.IGNORECASE)),
    (
        "copy",
        re.compile(
            r"fillBuffer|memcpy|clone|copy|as_strided|elementwise|FillFunctor",
            re.IGNORECASE,
        ),
    ),
]


# Two Itanium forms, both length-prefixed. `_ZN<len><ns><len><name>E` is the
# nested one; `_Z<len><name>` is a plain free function, which the nested-only
# pattern skipped entirely -- leaving the largest recoverable operator in prefill
# displayed as `_Z33pa_prefill_16mx1_16nx4_fp8_kernelI25pa_...`.
_MANGLED = re.compile(r"^_ZN?((?:\d+[A-Za-z_][A-Za-z0-9_]*)+)")


def pretty_kernel(name: str) -> str:
    """A readable identifier: the real name, minus the argument noise.

    Three kinds of clutter hide the same short name. Template arguments and the
    signature tail are simply cut. Itanium-mangled names (`_ZN5aiter37foo...`) are
    length-prefixed, so the components can be recovered exactly without a demangler
    -- `aiter::dynamic_per_group_scaled_quant_kernel` instead of 90 characters of
    encoded types. Anything unrecognised is returned untouched rather than guessed
    at; the untrimmed original is always kept beside this in `kernel_full`.
    """
    match = _MANGLED.match(name)
    if match:
        rest = match.group(1)
        parts, cursor = [], 0
        while cursor < len(rest):
            digits = 0
            while cursor + digits < len(rest) and rest[cursor + digits].isdigit():
                digits += 1
            if not digits:
                break
            length = int(rest[cursor : cursor + digits])
            cursor += digits
            parts.append(rest[cursor : cursor + length])
            cursor += length
        if parts:
            return "::".join(parts)
    trimmed = KERNEL_TRIM.sub("", name).strip()
    # Drop a leading return type; it is never how anyone refers to the kernel.
    for prefix in ("void ", "int ", "float ", "bool "):
        if trimmed.startswith(prefix):
            return trimmed[len(prefix) :]
    return trimmed


def classify_kernel(name: str) -> str:
    """First match wins; see KERNEL_CLASS for why gemm outranks quant."""
    for label, pattern in KERNEL_CLASS:
        if pattern.search(name):
            return label
    return "other"


@dataclass
class Op:
    """One **kernel** identity, aggregated over every layer it fires in.

    The atomic unit is a kernel launch, not a marker. A marker names a module and a
    module can contain up to nine kernels here -- `attn.wo_b` is a quantise, a GEMM
    and an all-reduce -- so collapsing to the marker both hides the all-reduce and
    forces a special case where only part of a row is compared to the ceiling. One
    row per kernel removes that: the GEMM row carries the roofline, the nccl row is
    simply communication, and nothing has to be sliced back out.

    The marker's shape describes the module's matmul, so it prices the gemm-class
    kernel only; the quantise and copy kernels beside it are gap tier.
    """

    phase: str
    path: str
    kernel: str = ""  # short label, for grouping and display
    kernel_full: str = ""  # untruncated, kept for grep and disambiguation
    kclass: str = ""
    shape: dict[str, str] = field(default_factory=dict)
    total_us: float = 0.0  # annotation span; kept only for diagnostics
    count: int = 0
    layers: set[int] = field(default_factory=set)
    class_us: dict[str, float] = field(default_factory=dict)  # busy time per class
    ts_by_layer: dict[int, float] = field(default_factory=dict)  # earliest ts per layer
    us_by_layer: dict[int, float] = field(default_factory=dict)  # busy time per layer
    m_values: list[int] = field(default_factory=list)
    # Every firing's duration, kept so a row can report a p50 beside its total.
    # A total answers "how much time did this cost"; it cannot answer "is that
    # number one firing in disguise", and an operator whose mean is wrecked by a
    # single stall reads as a slow operator rather than as one bad firing.
    fire_us: list[float] = field(default_factory=list)
    family_us: dict[str, float] = field(default_factory=dict)
    family_count: dict[str, int] = field(default_factory=dict)  # firings per family
    family_gemm: dict[str, float] = field(default_factory=dict)
    families: set[str] = field(default_factory=set)
    flops: float = 0.0
    bytes: float = 0.0
    peak_key: str = ""
    order: int = 0  # merged sequence, for the combined view
    order_by_family: dict[str, int] = field(default_factory=dict)
    is_layer_boundary: bool = False  # the all-reduce a layer opens on


def load_events(path: str) -> list[dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return json.load(handle).get("traceEvents", [])


def parse_marker(name: str) -> tuple[int, str, dict[str, str]] | None:
    """`layers.18.attn.wq_b[M=1,N=2,...]` -> (18, 'attn.wq_b', {'M': '1', ...})."""
    match = MARKER_RE.match(name)
    if not match:
        return None
    shape = dict(KV_RE.findall(match.group(3))) if match.group(3) else {}
    return int(match.group(1)), match.group(2), shape


def build_phase_index(events: list[dict[str, Any]]) -> tuple[list[float], list[Any]]:
    """Sorted (start, (end, phase)) spans, for locating a marker's phase by time."""
    spans = []
    for event in events:
        if event.get("cat") != "gpu_user_annotation":
            continue
        name = str(event.get("name", ""))
        match = PHASE_RE.match(name)
        if not match:
            continue
        start = float(event.get("ts", 0.0))
        spans.append((start, start + float(event.get("dur", 0.0)), match.group(1)))
    spans.sort()
    return [s[0] for s in spans], spans


def phase_at(starts: list[float], spans: list[Any], ts: float) -> str:
    """Innermost enclosing phase span, or 'unknown' if the marker sits outside."""
    index = bisect.bisect_right(starts, ts) - 1
    while index >= 0:
        _start, end, phase = spans[index]
        if ts < end:
            return phase
        index -= 1
    return "unknown"


def build_kernel_index(events: list[dict[str, Any]]) -> tuple[list[float], list[Any]]:
    """Sorted GPU kernel events, for summing busy time inside an annotation."""
    kernels = sorted(
        (
            (float(e.get("ts", 0.0)), float(e.get("dur", 0.0)), str(e.get("name", "")))
            for e in events
            if e.get("cat") == "kernel" and e.get("ph") == "X"
        ),
        key=lambda k: k[0],
    )
    return [k[0] for k in kernels], kernels


def decode_template(capture_path: str) -> list[tuple[int, str, str, dict[str, str]]]:
    """Ordered (layer, marker, kernel, shape) for one decode step, from the capture.

    Decode replays a CUDA graph and markers do not fire during replay, so a run
    trace carries no decode operator identity at all. The per-bs capture file does:
    it is recorded while the graph is being *built*, with every launch still nested
    inside its `record_function`. Walking launches rather than kernels is essential
    -- capture records ops without executing them, so the GPU bar is empty and a
    kernel-driven loop finds almost nothing (2 events against 2596 launches). The
    file MUST therefore carry `cuda_runtime` events; a CPU-only export silently
    yields an empty template, which is what blocked decode for a long time here.
    """
    events = [e for e in load_events(capture_path) if e.get("ph") == "X"]
    launches = sorted(
        (
            e
            for e in events
            if e.get("cat") == "cuda_runtime" and (e.get("args") or {}).get("kernel")
        ),
        key=lambda e: float(e["ts"]),
    )
    anns = sorted(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and str(e.get("name", "")).startswith("layers.")
        ),
        key=lambda e: float(e["ts"]),
    )
    if not launches or not anns:
        return []
    starts = [float(a["ts"]) for a in anns]
    out: list[tuple[int, str, str, dict[str, str]]] = []
    for launch in launches:
        ts = float(launch["ts"])
        cursor = bisect.bisect_right(starts, ts) - 1
        floor = cursor - 40
        best = None
        # Innermost enclosing annotation wins: attn.indexer.compressor.wkv_gate nests
        # inside attn.indexer, and the fine-grained name is the useful one.
        while cursor >= 0 and cursor > floor:
            ann = anns[cursor]
            if float(ann["ts"]) + float(ann.get("dur", 0.0)) > ts and (
                best is None or float(ann.get("dur", 0.0)) < float(best.get("dur", 0.0))
            ):
                best = ann
            cursor -= 1
        if best is None:
            continue
        parsed = parse_marker(str(best["name"]))
        if parsed is None:
            continue
        layer, path, shape = parsed
        out.append((layer, path, str((launch.get("args") or {}).get("kernel")), shape))
    return out


def collect_decode_ops(
    events: list[dict[str, Any]],
    template: list[tuple[int, str, str, dict[str, str]]],
    families: dict[int, str],
    max_steps: int = 20,
    stats: dict[str, Any] | None = None,
) -> dict[tuple[str, str, str], Op]:
    """Attribute decode kernels by aligning each step against the capture template.

    A fixed offset does not work: replay inserts kernels the capture never recorded
    (a runtime `fillBufferAligned` and an `act_and_mul` per layer here), and the
    first insertion shifts everything after it -- measured 1.9% agreement by index
    against 95.0% with gapped alignment. So the sequences are aligned the way a diff
    aligns text, and whatever fails to align is reported rather than forced onto a
    neighbour.

    Steps are near-identical, so one representative step is aligned properly and the
    resulting position map is reused, with the kernel name re-checked at every
    position; a mismatch drops that kernel instead of mis-attributing it. Only a
    sample of steady steps is summed -- the per-step time is already known exactly
    from the phase labels, and this pass only needs the split between operators.
    """
    import difflib

    kernels = sorted(
        (e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"),
        key=lambda e: float(e["ts"]),
    )
    kstarts = [float(e["ts"]) for e in kernels]
    steps = sorted(
        (
            e
            for e in events
            if e.get("cat") == "gpu_user_annotation"
            and str(e.get("name", "")).startswith("decode[")
        ),
        key=lambda e: float(e["ts"]),
    )
    if not steps or not template:
        return {}

    def kernels_of(step: dict[str, Any]) -> list[dict[str, Any]]:
        start = float(step["ts"])
        end = start + float(step.get("dur", 0.0))
        index = bisect.bisect_left(kstarts, start)
        out = []
        while index < len(kernels) and kstarts[index] < end:
            out.append(kernels[index])
            index += 1
        return out

    # Pick the steadiest window rather than the middle one. Taking the middle
    # excludes allocator warm-up but not KV-cache growth, ramp-down, or the mid-run
    # bumps long-context arrival introduces -- all of which the drift gate rejects.
    walls = [float(st.get("dur") or 0.0) for st in steps]
    lo, hi = find_steady_window(walls, n=max_steps)
    chosen = steps[lo:hi] or steps[-max_steps:]
    if stats is not None:
        stats["window"] = (lo, hi)
    if stats is not None:
        stats["sampled_steps"] = len(chosen)
    ref = kernels_of(chosen[0])
    tnames = [t[2] for t in template]
    rnames = [str(k["name"]) for k in ref]
    matcher = difflib.SequenceMatcher(a=tnames, b=rnames, autojunk=False)
    pos_map: dict[int, int] = {}
    for t_i, r_i, size in matcher.get_matching_blocks():
        for d in range(size):
            pos_map[r_i + d] = t_i + d

    ops: dict[tuple[str, str, str], Op] = {}
    unmatched_us = 0.0
    for step in chosen:
        seq = kernels_of(step)
        for position, event in enumerate(seq):
            t_i = pos_map.get(position)
            name = str(event.get("name", ""))
            if t_i is None or template[t_i][2] != name:
                unmatched_us += float(event.get("dur", 0.0))
                continue
            layer, path, _, shape = template[t_i]
            label = classify_kernel(name)
            key = ("decode", path, name)
            op = ops.get(key)
            if op is None:
                op = ops[key] = Op(
                    phase="decode",
                    path=path,
                    kernel=pretty_kernel(name),
                    kernel_full=name,
                    kclass=label,
                    shape=shape if label == "gemm" else {},
                )
            dur = float(event.get("dur", 0.0))
            fam = families.get(layer, "other")
            op.total_us += dur
            op.count += 1
            op.fire_us.append(dur)
            op.layers.add(layer)
            op.families.add(fam)
            op.class_us[label] = op.class_us.get(label, 0.0) + dur
            op.family_us[fam] = op.family_us.get(fam, 0.0) + dur
            op.family_count[fam] = op.family_count.get(fam, 0) + 1
            op.us_by_layer[layer] = op.us_by_layer.get(layer, 0.0) + dur
            if label == "gemm":
                op.family_gemm[fam] = op.family_gemm.get(fam, 0.0) + dur
                if shape.get("M"):
                    cost = gemm_cost(shape, name)
                    if cost:
                        op.flops += cost["flops"]
                        op.bytes += cost["bytes"]
                        op.peak_key = cost["peak_key"]
                        op.m_values.append(int(shape["M"]))
            ts = float(event["ts"])
            if layer not in op.ts_by_layer or ts < op.ts_by_layer[layer]:
                op.ts_by_layer[layer] = ts
    total = sum(o.total_us for o in ops.values()) + unmatched_us
    if total:
        print(
            f"decode align   : {len(chosen)} steps, "
            f"{(total - unmatched_us) / total:.1%} of kernel time attributed"
            f"  ({unmatched_us / 1000:.1f} ms unmatched)"
        )
    return ops


def phase_steps(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-step wall time for each phase, straight from the framework's own labels.

    This is the number that composes: an operator total over a whole run multiplies
    with nothing, but a per-step time multiplied by the scheduler's step count is an
    end-to-end estimate. Two things have to be handled or it is wrong by 3x:

    * the GPU-side annotation is emitted once **per stream**, so on a three-stream
      decode there are three records per step; they are regrouped by timestamp and
      the step's wall is the union, not the sum;
    * the first prefill carries compilation and autotuning -- 1090 ms against a
      426 ms steady state on the 2026-07-30 trace -- so the median is reported, not
      the mean, and the spread is carried alongside so an outlier stays visible.
    """
    import statistics

    out: dict[str, dict[str, Any]] = {}
    for phase in ("prefill", "decode"):
        marks = [
            e
            for e in events
            if e.get("cat") == "gpu_user_annotation"
            and str(e.get("name", "")).startswith(phase + "[")
        ]
        if not marks:
            continue
        marks.sort(key=lambda e: float(e.get("ts", 0.0)))
        groups: list[list[dict[str, Any]]] = [[marks[0]]]
        for event in marks[1:]:
            if float(event["ts"]) - float(groups[-1][0]["ts"]) < 5000:
                groups[-1].append(event)
            else:
                groups.append([event])
        walls = [
            max(float(e["ts"]) + float(e.get("dur", 0.0)) for e in g)
            - min(float(e["ts"]) for e in g)
            for g in groups
        ]
        walls_sorted = sorted(walls)
        # The median is what excludes a compile-carrying first step, and a median of
        # one sample excludes nothing. Say so rather than let a warm-up number pass
        # as steady state: the same 7237-token prompt took 1090 ms first and 428 ms
        # in steady state on one run, and 2000 ms as the only step on another.
        out[phase] = {
            "steps": len(groups),
            "warmup_suspect": len(groups) < 3,
            "streams": round(len(marks) / max(len(groups), 1), 1),
            "median_us": round(statistics.median(walls), 1),
            "mean_us": round(statistics.mean(walls), 1),
            "min_us": round(walls_sorted[0], 1),
            "max_us": round(walls_sorted[-1], 1),
            "total_us": round(sum(walls), 1),
        }
    return out


def collect_ops(
    events: list[dict[str, Any]],
    n_hash_layers: int = 0,
    family_rules: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, str, str], Op]:
    starts, spans = build_phase_index(events)
    kstarts, kernels = build_kernel_index(events)
    families = layer_families(events, n_hash_layers, family_rules)
    ops: dict[tuple[str, str, str], Op] = {}
    for event in events:
        if event.get("cat") != "gpu_user_annotation":
            continue
        name = str(event.get("name", ""))
        if not name.startswith("layers."):
            continue
        parsed = parse_marker(name)
        if parsed is None:
            continue
        layer, path, shape = parsed
        ts = float(event.get("ts", 0.0))
        phase = phase_at(starts, spans, ts)
        # One row per kernel launch, per the atomic-unit rule. An annotation's own
        # duration is only a SPAN -- how much of it is real GPU work swings from
        # 1.00 to 0.04 depending on the operator -- so the kernels inside it are
        # what get measured, each on its own.
        span = float(event.get("dur", 0.0))
        fam = families.get(layer, "other")
        index = bisect.bisect_left(kstarts, ts)
        end = ts + span
        while index < len(kernels) and kernels[index][0] < end:
            _, dur, kname = kernels[index]
            index += 1
            short = pretty_kernel(kname)
            label = classify_kernel(kname)
            key = (phase, path, kname)
            op = ops.get(key)
            if op is None:
                # The marker's shape describes the module's matmul, so it only
                # prices a gemm-class kernel; siblings are gap tier.
                op = ops[key] = Op(
                    phase=phase,
                    path=path,
                    kernel=short,
                    kernel_full=kname,
                    kclass=label,
                    shape=shape if label == "gemm" else {},
                )
            op.total_us += dur
            op.count += 1
            op.fire_us.append(dur)
            # M moves per prompt while the kernel name does not, so FLOPs and bytes
            # must be accumulated per firing with THAT firing's shape. Keying only
            # on (module, kernel) and pricing the first shape seen would charge one
            # prompt's arithmetic for all of them.
            if label == "gemm" and shape.get("M"):
                cost = gemm_cost(shape, kname)
                if cost:
                    op.flops += cost["flops"]
                    op.bytes += cost["bytes"]
                    op.peak_key = cost["peak_key"]
                    op.m_values.append(int(shape["M"]))
            op.layers.add(layer)
            op.families.add(fam)
            op.class_us[label] = op.class_us.get(label, 0.0) + dur
            op.family_us[fam] = op.family_us.get(fam, 0.0) + dur
            op.family_count[fam] = op.family_count.get(fam, 0) + 1
            op.us_by_layer[layer] = op.us_by_layer.get(layer, 0.0) + dur
            if label == "gemm":
                op.family_gemm[fam] = op.family_gemm.get(fam, 0.0) + dur
            if layer not in op.ts_by_layer or ts < op.ts_by_layer[layer]:
                op.ts_by_layer[layer] = ts
    return ops


def layer_families(
    events: list[dict[str, Any]],
    n_hash_layers: int = 0,
    families: list[dict[str, Any]] | None = None,
) -> dict[int, str]:
    """Label every layer with its archetype, following rules from module_map.yaml.

    Which archetypes exist is a property of the MODEL, so the list and the way each
    one is recognised are data, not code -- a dense model has one family, and the
    next MoE will not be csa/hca/hash. The engine only knows how to apply three
    kinds of rule:

        contains  a layer carrying a marker whose path contains this substring
        leading   the first N layers, N read from a model-config key
        default   everything else

    For DSV4-Pro that resolves CSA vs HCA from marker CONTENT -- a layer with
    `attn.indexer.*` needs an indexer, so it is compressed-sparse -- rather than
    from a hardcoded `hca=17-45:odd` spec. Content follows the model instead of
    tracking it, and it settles a real disagreement: on a c1/TP4 trace a layer-index
    parity heuristic says odd layers are CSA while the markers say the opposite, and
    applying a sparse-attention formula to a dense family is a category error, not
    a rounding error.

    The leading hash layers are the one thing markers CANNOT show: they run the
    same operator sequence as an ordinary layer. Their count is a static model
    property, so it comes from the config -- "layer counts from config, per-layer
    time from the trace". For DSV4-Pro that is 3, leaving 29 CSA + 29 HCA.
    """
    families = families or DEFAULT_FAMILIES
    per_layer: dict[int, set[str]] = {}
    for event in events:
        if event.get("cat") != "gpu_user_annotation":
            continue
        parsed = parse_marker(str(event.get("name", "")))
        if parsed is None:
            continue
        per_layer.setdefault(parsed[0], set()).add(parsed[1])

    fallback = next(
        (f["name"] for f in families if (f.get("detect") or {}).get("default")),
        families[-1]["name"] if families else "layer",
    )
    out: dict[int, str] = {}
    for layer, paths in per_layer.items():
        label = fallback
        for family in families:
            needle = (family.get("detect") or {}).get("contains")
            if needle and any(needle in path for path in paths):
                label = family["name"]
                break
        out[layer] = label

    # Leading layers override content: they look like their neighbours by design.
    leading = next(
        (f for f in families if (f.get("detect") or {}).get("leading")), None
    )
    if leading and n_hash_layers:
        for layer in range(n_hash_layers):
            out[layer] = leading["name"]
    return out


# Order matters: it is the order families appear as columns and as report tabs.
# Overridden per model by module_map.yaml; this is only the shape of the default.
DEFAULT_FAMILIES = [
    {"name": "hash", "detect": {"leading": "num_hash_layers"}},
    {"name": "csa", "detect": {"contains": "attn.indexer."}},
    {"name": "hca", "detect": {"default": True}},
]
# Set once per run from module_map.yaml. A module-level list rather than a
# threaded parameter because it is read in a dozen places that all describe the
# same run, and passing it through each of them would obscure that.
FAMILY_ORDER: list[str] = [f["name"] for f in DEFAULT_FAMILIES]


def merge_by_operator(ops: dict[tuple[str, str, str], Op]) -> list[Op]:
    """Collapse the per-shape rows into one row per operator.

    A prefill run sends several prompts, so the same operator appears once per
    prompt with a different M. Those are repeated samples of one operator, not
    several operators -- leaving them split makes a 20-row table show four
    distinct operators and hides the structure of a layer. FLOPs and bytes are
    summed over every firing, so the aggregate arithmetic intensity is the
    correctly weighted one rather than an average of ratios.
    """
    merged: dict[tuple[str, str], Op] = {}
    for op in ops.values():
        key = (op.phase, op.path, op.kernel)
        keep = merged.get(key)
        if keep is None:
            keep = merged[key] = Op(
                phase=op.phase,
                path=op.path,
                kernel=op.kernel,
                kernel_full=op.kernel_full,
                kclass=op.kclass,
                shape=dict(op.shape),
            )
            keep.m_values = []
        keep.total_us += op.total_us
        keep.count += op.count
        keep.fire_us.extend(op.fire_us)
        keep.layers |= op.layers
        for label, value in op.class_us.items():
            keep.class_us[label] = keep.class_us.get(label, 0.0) + value
        for layer, ts in op.ts_by_layer.items():
            if layer not in keep.ts_by_layer or ts < keep.ts_by_layer[layer]:
                keep.ts_by_layer[layer] = ts
        keep.families |= op.families
        for fam, value in op.family_us.items():
            keep.family_us[fam] = keep.family_us.get(fam, 0.0) + value
        for fam, value in op.family_count.items():
            keep.family_count[fam] = keep.family_count.get(fam, 0) + value
        for layer, value in op.us_by_layer.items():
            keep.us_by_layer[layer] = keep.us_by_layer.get(layer, 0.0) + value
        for fam, value in op.family_gemm.items():
            keep.family_gemm[fam] = keep.family_gemm.get(fam, 0.0) + value
        keep.m_values.extend(op.m_values)
        keep.shape = {k: v for k, v in op.shape.items() if k != "M"} or keep.shape
        keep.flops += op.flops
        keep.bytes += op.bytes
        keep.peak_key = op.peak_key or keep.peak_key
    return list(merged.values())


# The representative layer per family, chosen by typical_layer() and read back
# by price(). A module-level handoff rather than a parameter because the choice
# is a property of the run, not of any one operator.
TYPICAL_LAYER: dict[str, int] = {}


def find_steady_window(
    walls: list[float], n: int = 20, drift_pct_max: float = 2.0
) -> tuple[int, int]:
    """Pick the most stable n-step window.

    Slide a window of n across the per-step durations, score each by stdev, and
    REJECT any whose first-half mean differs from its second-half mean by more than
    drift_pct_max% of the overall mean. The drift gate is what the previous
    `steps[mid:mid+20]` slice lacked: taking the middle of the run assumes the
    middle is steady, which excludes ramp-up but not KV-cache growth, not
    ramp-down, and not the mid-run bumps that long-context arrival introduces.
    Among the accepted windows, lowest stdev wins.
    """
    import statistics

    m = len(walls)
    if m <= n:
        return 0, m
    overall = sum(walls) / m
    drift_max = overall * (drift_pct_max / 100.0)
    best: tuple[float, int] | None = None
    fallback: tuple[float, int] | None = None
    for start in range(m - n + 1):
        window = walls[start : start + n]
        sd = statistics.pstdev(window) if len(window) > 1 else 0.0
        half = n // 2
        drift = abs(sum(window[:half]) / half - sum(window[half:]) / (n - half))
        if fallback is None or sd < fallback[0]:
            fallback = (sd, start)
        if drift > drift_max:
            continue
        if best is None or sd < best[0]:
            best = (sd, start)
    pick = best if best is not None else fallback
    assert pick is not None
    return pick[1], pick[1] + n


def typical_layer(
    families: dict[int, str], layer_us: dict[int, float], n_middle: int = 15
) -> dict[str, int]:
    """One representative layer per family.

    Take the middle `n_middle` layers of the family -- for V4-Pro that is
    hca=17-45:odd, csa=18-46:even -- and keep the single layer whose wall is
    CLOSEST TO THE MEDIAN of that set. Not the first layer, not the mean: the first
    layer of a family often still carries prologue work, and a mean is pulled by
    exactly the outlier the middle window exists to exclude.

    The middle range is used ONLY to choose. Nothing outside the chosen layer is
    aggregated, so the reported structure is one real layer rather than a blend of
    several -- which is what stops operators that exist in only some layers from
    appearing as phantom duplicate rows.
    """
    import statistics

    out: dict[str, int] = {}
    for fam in sorted(set(families.values())):
        layers = sorted(n for n, f in families.items() if f == fam)
        if not layers:
            continue
        if len(layers) > n_middle:
            start = (len(layers) - n_middle) // 2
            layers = layers[start : start + n_middle]
        walls = [layer_us.get(n, 0.0) for n in layers]
        if not any(walls):
            out[fam] = layers[len(layers) // 2]
            continue
        med = statistics.median(walls)
        out[fam] = min(layers, key=lambda n: abs(layer_us.get(n, 0.0) - med))
    return out


def rotate_to_boundary(seq: list[Op]) -> list[Op]:
    """Start the layer at the all-reduce it waits on.

    Timestamps inside one layer read attention -> ... -> MoE -> combine
    all-reduce, so the collective sorts last. But a layer does not BEGIN with its
    mHC pre-stage: that stage consumes a residual which does not exist until the
    previous layer's combine has been reduced across the TP group. The all-reduce
    is the layer's opening dependency, and a table that buries it at the bottom
    hides the thing every other row is waiting on.

    Every layer is structurally identical, so this is the same cyclic sequence read
    from a different starting point -- no number changes, and the collective at the
    top is the same operator that closes the previous layer. The
    attention-side all-reduce (`wo_b`) is genuinely mid-layer and is left alone;
    only the trailing one rotates.
    """
    idx = None
    for i, op in enumerate(seq):
        if op.kclass == "comm":
            idx = i
    if idx is None or idx == 0:
        return seq
    seq[idx].is_layer_boundary = True
    return seq[idx:] + seq[:idx]


def algorithm_order(ops: list[Op], families: dict[int, str]) -> None:
    """Order operators the way the model executes them -- once PER LAYER FAMILY.

    DSV4-Pro's layer semantics are fixed, not inferred: layers 0-2 are hash, then
    HCA and CSA alternate (odd HCA, even CSA). Two families therefore run two
    different sequences, and a single global order can only be right for one of
    them. Picking one reference layer and appending whatever is absent from it
    does not work either: HCA's core attention is 3rd in an HCA layer but would
    land at position 47 of 52. In a tab whose whole promise is "read down the list
    to follow a layer", that is the wrong list.

    So each family gets its own sequence, taken from its first layer, and a merged
    sequence is built for the combined view by aligning the two the way a diff
    aligns text: shared operators anchor, family-exclusive ones land between the
    anchors they actually sit between rather than in a lump at the end.
    """
    import difflib

    layer_us: dict[int, float] = {}
    for op in ops:
        for layer, value in op.us_by_layer.items():
            layer_us[layer] = layer_us.get(layer, 0.0) + value
    reps = typical_layer(families, layer_us)
    TYPICAL_LAYER.clear()
    TYPICAL_LAYER.update(reps)

    seqs: dict[str, list[Op]] = {}
    for fam, layer in reps.items():
        present = [op for op in ops if layer in op.ts_by_layer]
        present.sort(key=lambda o: o.ts_by_layer[layer])
        present = rotate_to_boundary(present)
        seqs[fam] = present
        for i, op in enumerate(present):
            op.order_by_family[fam] = i

    # Merge for the combined view. CSA is the spine: it is the richer family (it
    # carries the indexer chain), so aligning HCA onto it inserts fewer operators
    # than the other way round.
    spine = seqs.get("csa") or max(seqs.values(), key=len, default=[])
    merged: list[Op] = list(spine)
    for fam, seq in seqs.items():
        if seq is spine:
            continue
        a = [id(o) for o in merged]
        b = [id(o) for o in seq]
        out: list[Op] = []
        i = 0
        for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            tag, i1, i2, j1, j2 = block
            if tag in ("equal", "delete"):
                out.extend(merged[i1:i2])
            elif tag == "insert":
                out.extend(seq[j1:j2])
            else:  # replace: keep both, spine first
                out.extend(merged[i1:i2])
                out.extend(seq[j1:j2])
        merged = out

    seen: set[int] = set()
    ordered = [o for o in merged if id(o) not in seen and not seen.add(id(o))]
    # Anything in no family's representative layer keeps its relative order after.
    rest = sorted(
        (op for op in ops if id(op) not in {id(o) for o in ordered}),
        key=lambda o: min(o.ts_by_layer.values()) if o.ts_by_layer else 0.0,
    )
    for i, op in enumerate(ordered + rest):
        op.order = i


# A split-K GEMM is TWO kernels under one marker. AITER splits K into NUM_KSPLIT
# slices so a skinny decode GEMM can fill the CUs: each slice writes an fp32
# partial [M, N] and a second kernel sums them. It is still ONE matmul, so it is
# modelled as one matmul -- `fold_split_k` folds the reduce into it before
# pricing, and `gemm_cost` below never branches on which kernel ran.
#
# The split count is read off the kernel name, never derived. It comes from a
# tuned table (`gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=..-K=..json`, bucket
# `M_LEQ_1`) whose value differs between aiter builds -- 1 in the default file, 4
# in one specialised file, 8 in the build that produced this trace. The same GEMM
# on a build whose table says 1 runs as a single kernel with no reduce at all,
# which is the reason the split must never reach the ceiling: it would make one
# operator's roof depend on which aiter was installed.
_KSPLIT_REDUCE = re.compile(r"ACTUAL_KSPLIT_(\d+)")


def gemm_cost(shape: dict[str, str], kernel: str = "") -> dict[str, Any] | None:
    """FLOPs / Bytes for a plain GEMM, or None when the marker is not one.

        FLOPs = 2 * M * N * K
        Bytes = M*K*sz(a) + K*N*sz(w) + M*N*sz(o)

    Bytes counts one cold pass over A, B and C. It is a lower bound: it assumes
    perfect reuse inside the kernel and ignores every re-read, so an operator that
    genuinely streams its weights more than once shows up as gap, not as a smaller
    ceiling. That is the intended direction of error for a thermometer.

    This is the whole matmul, split-K included. The fp32 partials such a kernel
    writes and reads back are what the implementation costs, not what the operator
    computes, so they sit in the gap beside every other implementation cost.
    `kernel` is accepted so callers can pass what actually ran; nothing branches
    on it, deliberately -- a reduce kernel is not a different operator.
    """
    try:
        m, n, k = (int(shape[key]) for key in ("M", "N", "K"))
    except (KeyError, ValueError):
        return None
    a, w, o = shape.get("a"), shape.get("w"), shape.get("o")
    if a not in DTYPE_BYTES or w not in DTYPE_BYTES or o not in DTYPE_BYTES:
        return None
    flops = 2.0 * m * n * k
    byts = m * k * DTYPE_BYTES[a] + k * n * DTYPE_BYTES[w] + m * n * DTYPE_BYTES[o]
    peak_a, peak_w = DTYPE_PEAK.get(a), DTYPE_PEAK.get(w)
    peaks = [p for p in (peak_a, peak_w) if p]
    peak_key = min(peaks, key=PEAK_ORDER.index) if peaks else "matrix_bf16"
    return {"formula": "gemm", "flops": flops, "bytes": byts, "peak_key": peak_key}


def fold_split_k(ops: dict[Any, Op]) -> dict[Any, Op]:
    """Fold a split-K reduce kernel into the GEMM it finishes.

    Both kernels fired under one marker to compute one matmul, so the operator is
    the matmul: one floor, counted once, with both kernels' time measured against
    it. Two rows each carrying the full GEMM cost billed the matmul twice; giving
    the reduce a reduction model of its own was no better, because it invents an
    operator the model does not contain and leaves the GEMM looking better than
    the module is (0.53 against the fp8 roof, when the matmul takes 12.0 us per
    firing against a 3.70 us floor -- 0.31).

    `count` is NOT summed: the operator fired 1220 times, in two kernels each
    time. Summing would halve every per-firing number, including the one the
    latency regime is decided on.
    """
    for key, red in [
        (k, o) for k, o in ops.items() if _KSPLIT_REDUCE.search(o.kernel_full or "")
    ]:
        main = next(
            (
                o
                for k2, o in ops.items()
                if k2 != key
                and o.phase == red.phase
                and o.path == red.path
                and "NUM_KSPLIT" in (o.kernel_full or "")
                and o.flops > 0
            ),
            None,
        )
        if main is None:
            continue  # an unpaired reduce keeps its own row rather than vanish
        main.total_us += red.total_us
        # One firing of this operator is one firing of EACH kernel, so the two
        # lists are added pairwise rather than concatenated -- both are appended
        # in trace order and have one entry per firing. Concatenating would give
        # a bimodal list (a 7.0 us GEMM beside a 5.0 us reduce) whose median is
        # neither, and `efficiency_p50` compares against a per-firing floor that
        # covers both kernels. Lengths can only differ if the trace lost one, in
        # which case fall back to concatenating rather than mis-pairing.
        if len(main.fire_us) == len(red.fire_us):
            main.fire_us = [a + b for a, b in zip(main.fire_us, red.fire_us)]
        else:
            main.fire_us.extend(red.fire_us)
        for attr in ("class_us", "family_us", "us_by_layer", "family_gemm"):
            dst, src = getattr(main, attr), getattr(red, attr)
            for k2, v in src.items():
                dst[k2] = dst.get(k2, 0.0) + v
        main.layers |= red.layers
        main.families |= red.families
        main.kernel = f"{main.kernel} + splitk reduce"
        ops.pop(key)
    return ops


def decompose(
    work_us: float, measured_us: float, firings: int, floor_us: float
) -> tuple[float, float]:
    """Split an operator's measured time into issue and stall, beside its work.

        work + issue + stall = measured

    `work` is the roofline. `issue` is what firing this operator costs at all --
    the fastest kernel in the run, times the firing count -- minus whatever the
    work already covers. `stall` is the remainder: achieved rate, dependency
    stalls, wave ramp-up.

    The three add up by construction, which is the only reason the split is
    readable: a reader can check that nothing was attributed twice. `recoverable`
    lumps issue and stall together, which is the right number to rank operators
    by and the wrong one to act on -- removing a launch and raising achieved
    bandwidth are different jobs with different ceilings.
    """
    issue = max(0.0, firings * floor_us - work_us)
    stall = max(0.0, measured_us - max(work_us, firings * floor_us))
    return issue, stall


def bench_from_labels(
    events: list[dict[str, Any]],
    steps: dict[str, Any],
    phase: str,
    tp: int,
    dp: int,
    dpon: bool,
    cfg: dict[str, Any] | None = None,
    runtime: dict[str, str] | None = None,
) -> formulas.Bench:
    """Build the operating point the config-driven formulas need.

    bs and tok come straight from ATOM's phase labels. kv_seq_len does not: the
    decode label is `decode[bs=1 tok=1 d=1]` and carries no context length, so it
    is derived as prefill ctx + half the decode steps -- the middle of the range
    the run actually swept -- and the derivation is carried back as a string so the
    report can print it instead of implying the number was measured. It scales the
    attention terms only; nothing else in the model depends on it.
    """
    import re as _re

    bs = tok = ctx = 0
    for event in events:
        name = str(event.get("name", ""))
        if not name.startswith(phase + "["):
            continue
        got = dict(_re.findall(r"(\w+)=(\d+)", name))
        bs = int(got.get("bs", 1))
        tok = int(got.get("tok", 1))
        ctx = int(got.get("ctx", 0))
        break

    dec = steps.get("decode") or {}
    if phase == "prefill":
        kv, note = float(ctx or tok), f"prefill ctx={ctx or tok} from the phase label"
    else:
        # No context length in the decode label, so reconstruct the sweep: the run
        # starts at the prompt length and grows one token per step.
        start = 0.0
        for event in events:
            name = str(event.get("name", ""))
            if name.startswith("prefill["):
                got = dict(_re.findall(r"(\w+)=(\d+)", name))
                start = float(got.get("ctx") or got.get("tok") or 0)
                break
        n = int(dec.get("steps") or 0)
        kv = start + n / 2.0
        note = (
            f"derived, NOT measured: prefill ctx={start:.0f} + {n} decode steps, "
            f"so kv sweeps {start:.0f}..{start + n:.0f}; the midpoint {kv:.0f} is used"
        )
    return formulas.Bench(
        batch=bs or 1,
        seq_len=tok or 1,
        tp=tp,
        dp=dp,
        dpon=dpon,
        kv_seq_len=kv,
        kv_seq_note=note,
        dtypes=formulas.resolve_dtypes(cfg or {}, runtime),
    )


def attn_M_for(bench: formulas.Bench, path: str) -> float:
    """Which token count a formula used: the MoE region sees the gathered batch."""
    return formulas.moe_B(bench) if path.startswith("ffn.") else formulas.attn_M(bench)


def load_model_decl(name: str, root: Path) -> dict[str, Any]:
    """Read configs/models/<name>.yaml, following `extends` for variants.

    Named for the ATOM benchmark catalog's `prefix`, so the string passed to
    --model is the same one the benchmark uses. Quantisation variants are separate
    catalog entries and so separate files, but they inherit structure through
    `extends` rather than repeating it -- a precision does not change a layer
    taxonomy.
    """
    seen: list[str] = []
    merged: dict[str, Any] = {}
    while name:
        if name in seen:
            raise SystemExit(f"model declarations form a cycle: {' -> '.join(seen)}")
        seen.append(name)
        path = root / "configs" / "models" / f"{name}.yaml"
        if not path.is_file():
            have = sorted(q.stem for q in (root / "configs" / "models").glob("*.yaml"))
            raise SystemExit(
                f"no declaration for --model {name} ({path}). "
                f"Declared models: {', '.join(have) or '(none)'}"
            )
        decl = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # Child wins; the parent only fills what the child left unsaid.
        merged = {**decl, **merged}
        name = decl.get("extends")
    return merged


def check_methodology(
    name: str, want: dict[str, int], observed: dict[str, int]
) -> None:
    got = {k: v for k, v in observed.items() if v}
    if got != want:
        raise SystemExit(
            f"--methodology {name} expects layer families {want}, "
            f"trace resolved {got}. Either the trace is not {name}, or the "
            f"family rules in module_map.yaml no longer match this build. "
            f"Refusing to report numbers attributed to the wrong layer family."
        )
    print(f"methodology    : {name} verified ({want})")


def price(
    op: Op,
    peaks: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    bench: formulas.Bench | None = None,
    n_steps: int = 0,
    arch: str | None = None,
) -> dict[str, Any]:
    """Attach roofline columns to one operator, aggregated over every firing."""
    if op.m_values:
        lo, hi = min(op.m_values), max(op.m_values)
        m_text = str(lo) if lo == hi else f"{lo}\u2013{hi}"
    else:
        m_text = ""
    row: dict[str, Any] = {
        "seq": op.order,
        "layer_boundary": op.is_layer_boundary,
        # Per-family execution position. The combined view uses `seq`; a
        # family tab must use its own, or it shows CSA's order under an
        # HCA heading.
        **{f"seq_{fam}": op.order_by_family.get(fam, "") for fam in FAMILY_ORDER},
        "phase": op.phase,
        "op": op.path,
        "kernel": op.kernel,
        "kernel_full": op.kernel_full,
        "kclass": op.kclass,
        "M": m_text,
        "N": op.shape.get("N", ""),
        "K": op.shape.get("K", ""),
        "dtype_a": (op.shape.get("a") or "").replace("torch.", ""),
        "dtype_w": (op.shape.get("w") or "").replace("torch.", ""),
        "dtype_o": (op.shape.get("o") or "").replace("torch.", ""),
        "families": "+".join(f for f in FAMILY_ORDER if f in op.families)
        or "+".join(sorted(op.families)),
        **{f"{fam}_us": round(op.family_us.get(fam, 0.0), 3) for fam in FAMILY_ORDER},
        # Per-family GEMM time, so a family-filtered view can recompute efficiency
        # against that family alone instead of borrowing the whole-run number.
        **{
            f"{fam}_gemm_us": round(op.family_gemm.get(fam, 0.0), 3)
            for fam in FAMILY_ORDER
        },
        # One representative layer's own time, averaged over the steady window --
        # the typical-layer method. The `{fam}_us` columns beside it are
        # TOTALS over every layer of the family; dividing one by the layer count
        # blends layers that are not identical, which is the blend the typical-layer
        # method exists to avoid.
        **{
            f"{fam}_typical_us": (
                round(op.us_by_layer.get(TYPICAL_LAYER[fam], 0.0) / n_steps, 4)
                if fam in TYPICAL_LAYER and n_steps
                else ""
            )
            for fam in FAMILY_ORDER
        },
        **{f"{fam}_typical_layer": TYPICAL_LAYER.get(fam, "") for fam in FAMILY_ORDER},
        "n_layers": len(op.layers),
        "n_samples": op.count,
        "span_us": round(op.total_us, 3),
        "busy_us": round(sum(op.class_us.values()), 3),
        "gemm_us": round(op.class_us.get("gemm", 0.0), 3),
        "comm_us": round(op.class_us.get("comm", 0.0), 3),
        "quant_us": round(op.class_us.get("quant", 0.0), 3),
        "other_us": round(op.class_us.get("other", 0.0), 3),
    }
    row["measured_us"] = row["busy_us"]
    # A total plus its shape. `efficiency` divides a floor by a SUM, so one stalled
    # firing halves it and nothing says so: prefill's `attn.wo_b` all-reduce read
    # 0.35 of the link roof against the MoE one's 0.70, and the whole gap was a
    # single 58.3 ms firing among 61 whose p50 was 987 us -- the two collectives
    # are the same speed. The total is kept as the row's measured time (it IS the
    # time that was spent); these columns say whether that number can be read as a
    # per-firing rate. `top1_share` is the criterion, not the spread: what matters
    # is whether one firing dominates the total, not how long the tail is.
    if op.fire_us:
        ordered = sorted(op.fire_us)
        p50 = ordered[len(ordered) // 2]
        worst = ordered[-1]
        row["p50_per_firing_us"] = round(p50, 4)
        row["max_per_firing_us"] = round(worst, 4)
        share = worst / row["measured_us"] if row["measured_us"] else 0.0
        row["top1_share"] = round(share, 4)
        # One rule, one place, so the report and the CLI cannot disagree about
        # which rows are suspect. `3/n` keeps a row that fired once or twice from
        # flagging itself: there, one firing IS the total and says nothing.
        row["outlier_dominated"] = bool(
            len(ordered) >= 8 and share > max(0.10, 3.0 / len(ordered))
        )
    cost = (
        {
            "formula": "gemm",
            "flops": op.flops,
            "bytes": op.bytes,
            "peak_key": op.peak_key,
        }
        if op.flops > 0 and op.peak_key
        else None
    )
    # No shape in the marker? The operator may still have a closed form -- see
    # formulas.py. The marker is still the identity; the config supplies the
    # dimensions. Flagged as shape_provenance="config" so a reader can tell a
    # measured shape from a derived one without reading this file.
    spec = None
    if cost is None and cfg is not None and bench is not None:
        fn = formulas.lookup(op.path, op.kernel_full or op.kernel, arch)
        if fn is not None:
            # One operator's firings span several layer families, and a
            # family-aware formula (the sparse attentions) charges a different k
            # per family. In DSV4-Pro the leading hash layers and the CSA layers
            # both run the compressed/indexed path -- their indexer kernels fire
            # while the HCA layers' do not (verified: every indexer op reports
            # hash+csa firings and hca_us=0) -- so both price as "csa", and only
            # the dense HCA layers price as "hca". One kind for the whole op would
            # charge every HCA firing k=index_topk instead of k=kv/128, so the sum
            # is per family, weighted by its firing count;
            # families that sum to op.count leave kind-insensitive formulas
            # unchanged. hash and csa map to the same "csa" call -- redundant but
            # harmless, and it keeps the mapping the single fact "hca is dense".
            flops = byts = 0.0
            spec = None
            spec_ct = -1
            try:
                # The kernel this firing actually ran. A formula that can be
                # dispatched several ways -- the collectives are -- reads the
                # path off the name instead of re-deriving it from a threshold
                # this tool cannot observe.
                b_priced = dataclasses.replace(
                    bench, kernel=op.kernel_full or op.kernel
                )
                per_family = op.family_count or {"csa": op.count}
                for fam, ct in per_family.items():
                    kind = "hca" if fam == "hca" else "csa"
                    f_i, b_i, s_i = fn(cfg, b_priced, kind)
                    flops += f_i * ct
                    byts += b_i * ct
                    # Display the dominant family's spec (its per-firing exprs and
                    # peak); the stored flops/bytes are already the weighted total.
                    if s_i is not None and ct > spec_ct:
                        spec, spec_ct = s_i, ct
            except (KeyError, ZeroDivisionError):
                flops = byts = 0.0
                spec = None
            if spec is not None and byts > 0:
                cost = {
                    "formula": fn.__name__,
                    "flops": flops,
                    "bytes": byts,
                    "peak_key": f"matrix_{spec.compute}",
                }
                row["shape_provenance"] = "config"
                # The dtype too. A marker STATES the dtype; a formula ASSERTS it
                # from source reading, and those are not the same kind of fact.
                # Mixing them silently would let an operator carry a ceiling from
                # a paper while its own kernel name says another dtype.
                row["dtype_provenance"] = "asserted"
                # Record the numbers that were actually substituted, at the moment
                # they are substituted. A tooltip reconstructed later from the row
                # would be a second implementation of the same arithmetic, which is
                # exactly how the achieved-throughput column once went 1216x wrong.
                # layer_kind is the mix, not a single value: firings priced as
                # "csa" (hash+csa layers, compressed path) vs "hca" (dense). A
                # single label here is what hid the mispricing.
                kind_counts: dict[str, int] = {}
                for fam, ct in (op.family_count or {"csa": op.count}).items():
                    kk = "hca" if fam == "hca" else "csa"
                    kind_counts[kk] = kind_counts.get(kk, 0) + ct
                kind_mix = ", ".join(
                    f"{kk}x{ct}" for kk, ct in sorted(kind_counts.items())
                )
                row["formula_inputs"] = (
                    f"M={attn_M_for(bench, op.path):,.0f}  kv_seq={bench.kv_seq_len:,.0f}  "
                    f"tp={bench.tp}  dp={bench.dp}  layer_kind={kind_mix}"
                )
                row["formula_note"] = spec.note
                # A path chosen by a software default is a finding in its own
                # right, and it is invisible in a gap. Hoist it out of the note so
                # the table can mark the row without a reader hovering it.
                if "| CONFIG:" in spec.note:
                    row["config_note"] = spec.note.split("| CONFIG:", 1)[1].strip()
                row["flops_expr"] = spec.flops_expr
                row["bytes_expr"] = spec.bytes_expr
                row["formula_peak_key"] = f"matrix_{spec.compute}"
                row["dtype_a"] = row["dtype_a"] or spec.mem
                row["dtype_o"] = row["dtype_o"] or spec.mem

    if cost is None:
        row.update(
            tier="gap",
            formula="",
            flops="",
            bytes="",
            flop_per_byte="",
            t_compute_us="",
            t_memory_us="",
            t_roofline_us="",
            bound="",
            efficiency="",
            gap="",
            recoverable_us="",
            ceiling_tflops="",
        )
        return row

    peak_tflops = float(peaks["compute_tflops"][cost["peak_key"]])
    # Collectives are bound by the interconnect, not by HBM. Pricing an all-reduce
    # against an 8 TB/s HBM roof flatters it by ~100x and hides the only thing the
    # comm roofline says, which is how close the link is to saturated.
    bw_gbps = float(
        peaks["interconnect_bw_gbps"]
        if spec is not None and spec.bw == "interconnect"
        else peaks["mem_bw_gbps"]
    )
    bw_bytes_per_us = bw_gbps * 1e9 / 1e6
    t_compute = cost["flops"] / (peak_tflops * 1e12) * 1e6
    t_memory = cost["bytes"] / bw_bytes_per_us
    t_roofline = max(t_compute, t_memory)
    # Compare like with like: the matmul ceiling against the matmul kernels only.
    # comm / quant / other are reported in their own columns, not folded in here.
    # The row IS the GEMM kernel now, so its own time is the denominator -- no
    # slicing a class back out of a mixed marker.
    per_fire = op.total_us
    efficiency = t_roofline / per_fire if per_fire > 0 else 0.0
    row.update(
        tier="roofline",
        formula=cost["formula"],
        flops=cost["flops"],
        bytes=cost["bytes"],
        flop_per_byte=round(cost["flops"] / cost["bytes"], 2),
        t_compute_us=round(t_compute, 4),
        t_memory_us=round(t_memory, 4),
        t_roofline_us=round(t_roofline, 4),
        # An operator whose bytes cross the fabric is not "memory-bound": its wall
        # is the link, and reading it as HBM pressure points at the wrong fix. The
        # bw_kind was already recorded and simply was not used here.
        bound=(
            "link"
            if spec is not None and spec.bw == "interconnect"
            else ("compute" if t_compute >= t_memory else "memory")
        ),
        efficiency=round(efficiency, 4),
        # The same efficiency against the MEDIAN firing instead of the mean. When
        # the two disagree the row's total is carrying something that is not this
        # operator's steady cost, and which of the two a reader should believe is
        # their call -- so both are printed rather than one being corrected away.
        efficiency_p50=(
            round((t_roofline / op.count) / row["p50_per_firing_us"], 4)
            if op.count and row.get("p50_per_firing_us")
            else ""
        ),
        gap=round(1.0 / efficiency, 3) if efficiency > 0 else "",
        recoverable_us=round(op.total_us * (1.0 - min(efficiency, 1.0)), 3),
        ceiling_tflops=peak_tflops,
        # Blank, not 0.0. A permute, a memset and a collective do no arithmetic,
        # so "0 TFLOP/s" is not a bad score, it is a meaningless one -- and printed
        # as a number it reads as the former. Same rule as the empty roofline cells:
        # nothing means unknown or inapplicable, never zero.
        achieved_tflops=(
            round(cost["flops"] / (per_fire * 1e-6) / 1e12, 1)
            if per_fire > 0 and cost["flops"] > 0
            else ""
        ),
        # The same point read in the units of the wall it actually hits. A
        # memory-bound kernel reported as "11.7 TFLOP/s" tells nobody anything;
        # "5.8 TB/s against an 8 TB/s peak" is immediately judgeable. The two are
        # the same distance from the roofline -- achieved/(AI*BW) == effective/peak
        # -- so this adds no new claim, only a readable one.
        achieved_gbps=(
            round(cost["bytes"] / (per_fire * 1e-6) / 1e9, 1) if per_fire > 0 else ""
        ),
        bw_util=(
            round((cost["bytes"] / (per_fire * 1e-6) / 1e9) / bw_gbps, 4)
            if per_fire > 0
            else ""
        ),
        n_firings=op.count,
        bw_kind=spec.bw if spec is not None else "hbm",
        formula_note=spec.note if spec is not None else "",
    )
    row.setdefault("shape_provenance", "measured")
    row.setdefault("dtype_provenance", "measured" if row.get("dtype_a") else "")
    # Guard the published number, not a proxy for it. The earlier guard tested
    # efficiency, which is computed from t_roofline; a stale duplicate of the
    # achieved-throughput formula elsewhere could therefore be 1216x wrong while
    # efficiency stayed sane and the guard stayed quiet. Test what gets plotted.
    achieved = row.get("achieved_tflops") or 0
    row["over_peak"] = bool(efficiency > 1.0 or float(achieved) > peak_tflops)
    return row


SHEET_FAMILIES = ["csa", "hca"]  # hash is 3 layers of prologue; not a study target

XLSX_COLUMNS = [
    ("seq", "#"),
    ("op", "module"),
    ("kernel", "kernel"),
    ("kernel_full", "kernel (full)"),
    ("kclass", "class"),
    ("stage", "stage"),
    ("M", "M"),
    ("N", "N"),
    ("K", "K"),
    ("dtype_a", "dtype in"),
    ("dtype_w", "dtype weight"),
    ("dtype_o", "dtype out"),
    ("flop_per_byte", "FLOP/byte"),
    ("bound", "bound"),
    ("busy_us", "family busy us"),
    ("gemm_us", "gemm us"),
    ("comm_us", "comm us"),
    ("t_roofline_us", "t_roofline us"),
    ("t_compute_us", "t_compute us"),
    ("t_memory_us", "t_memory us"),
    ("efficiency", "efficiency"),
    ("gap", "gap"),
    ("recoverable_us", "recoverable us"),
    ("issue_us", "issue us"),
    ("stall_us", "stall us"),
    ("ceiling_tflops", "ceiling TFLOPS"),
    ("achieved_tflops", "achieved TFLOPS"),
    ("achieved_gbps", "effective GB/s"),
    ("bw_util", "BW util"),
    ("tier", "tier"),
    ("n_layers", "layers"),
]


def write_xlsx(path, rows, args, peaks, coverage, total, priced) -> None:
    """One sheet per layer family, plus a config sheet.

    Splitting by family is not cosmetic: CSA carries the indexer chain and HCA does
    not, so a single merged sheet averages two different layer architectures into
    one row and hides the thing worth seeing. `family busy us` is that family's
    time only; every roofline column is whole-run and repeats across sheets.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
    except ImportError:
        print("note: openpyxl missing, skipping xlsx")
        return

    book = Workbook()
    book.remove(book.active)
    for family in SHEET_FAMILIES:
        sheet = book.create_sheet(f"{args.phase}_{family}")
        sheet.append([label for _, label in XLSX_COLUMNS])
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        present = [r for r in rows if float(r.get(f"{family}_us") or 0) > 0]
        for row in present:
            values = []
            for key, _ in XLSX_COLUMNS:
                values.append(
                    row.get(f"{family}_us") if key == "busy_us" else row.get(key, "")
                )
            sheet.append(values)
        sheet.freeze_panes = "C2"
        for column, (key, label) in enumerate(XLSX_COLUMNS, start=1):
            width = max(len(label) + 2, 12 if key != "op" else 34)
            sheet.column_dimensions[
                sheet.cell(row=1, column=column).column_letter
            ].width = width

    meta = book.create_sheet("config")
    meta.append(["section", "key", "value"])
    for cell in meta[1]:
        cell.font = Font(bold=True)
    facts = [
        ("run", "trace", Path(args.run_trace).name),
        ("run", "phase", args.phase),
        ("run", "hash_layers_from_config", getattr(args, "_n_hash", "")),
        ("run", "sort", args.sort),
        ("platform", "name", peaks.get("platform")),
        ("platform", "convention", peaks.get("convention")),
        ("platform", "mem_bw_gbps", peaks.get("mem_bw_gbps")),
        *[
            ("platform", f"peak_{k}", v)
            for k, v in (peaks.get("compute_tflops") or {}).items()
        ],
        ("result", "operators", len(rows)),
        ("result", "measured_busy_us", round(total, 1)),
        ("result", "roofline_tier_us", round(priced, 1)),
        ("result", "coverage_priceable_ops", round(coverage, 4)),
        (
            "caveat",
            "coverage_is_not_accuracy",
            (
                "coverage = share of measured time in operators we can price; "
                "the ceiling only speaks to the GEMM kernels inside them"
            ),
        ),
        (
            "caveat",
            "thermometer",
            "magnitudes and bound categories only; raw roofline ~22% error",
        ),
        (
            "caveat",
            "marker_scope",
            "a marker names a module: comm/quant are separate columns",
        ),
        (
            "caveat",
            "fp8_efficiency",
            (
                "fp8 GEMMs read 0.23-0.28 of their ceiling and bf16 ones 0.32-0.46, "
                "but in ABSOLUTE terms every compute-bound GEMM here lands at "
                "0.8-1.4 PFLOPS whatever the input precision -- the fp8 band is a "
                "2x taller ceiling, not a worse kernel. Read it as 'nothing in this "
                "run exceeds ~1.4 PF', and note that fp8 is buying bytes rather "
                "than FLOPs on these shapes"
            ),
        ),
        (
            "caveat",
            "comm_model",
            (
                "collectives are priced at the bandwidth-optimal bound, which is "
                "algorithm-independent. Two of identical size once measured 2x "
                "apart; their per-firing medians agree to 3%, so that gap was one "
                "stalled firing in a total, not a different algorithm"
            ),
        ),
        (
            "caveat",
            "prefill_warmup",
            "if the trace holds one prefill step it is the compile step: 25% occupancy",
        ),
    ]
    for section, key, value in facts:
        meta.append([section, key, value])
    for column, width in (("A", 12), ("B", 26), ("C", 62)):
        meta.column_dimensions[column].width = width
    book.save(path)
    print(
        f"xlsx written  : {path}  (sheets: {', '.join(s.title for s in book.worksheets)})"
    )


def stage_of(path: str, stages: dict[str, str]) -> str:
    """Longest matching prefix wins; a miss reports as 'other' and changes no number."""
    best = ""
    for prefix in stages:
        if (path == prefix or path.startswith(prefix + ".")) and len(prefix) > len(
            best
        ):
            best = prefix
    return stages.get(best, "other")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_trace", help="ATOM torch/kineto trace (.json or .json.gz)")
    parser.add_argument("--peaks", default=None, help="peaks yaml (default: mi355x)")
    parser.add_argument("--map", default=None, help="module_map.yaml")
    parser.add_argument(
        "--phase", default="prefill", choices=["prefill", "decode", "all"]
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path prefix; .json/.csv/.xlsx are written beside it. "
        "Defaults to roofline_<model>_<phase>_c<N> derived from the trace "
        "-- naming a run should not be the caller's problem.",
    )
    parser.add_argument(
        "--capture-trace",
        default=None,
        help="per-bs CUDA-graph capture (capture_traces/bs_<N>_*.json.gz). Required for "
        "--phase decode: replay emits no markers, so operator identity comes from the "
        "capture. It must contain cuda_runtime launch events -- check with "
        '`zcat <f>.json.gz | grep -c \'"cat": "cuda_runtime"\'`.',
    )
    parser.add_argument(
        "--hash-layers",
        type=int,
        default=None,
        help="leading hash layers (DSV4-Pro: 3, from num_hash_layers in the HF "
        "config). Markers cannot detect these -- they run the same operator "
        "sequence as an ordinary layer -- so leaving it unset mislabels them.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name, recorded in identity and used to pick the model "
        "declaration. Inferred from the trace filename when absent -- and the "
        "filename is the weakest source here, it has already been wrong about tp.",
    )
    parser.add_argument(
        "--framework",
        default=None,
        help="atom / sglang / trtllm. Recorded in identity; inferred from the "
        "trace filename when absent.",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="mi355x / mi300x / b200. Recorded in identity; inferred from the "
        "trace filename when absent.",
    )
    parser.add_argument(
        "--methodology",
        default=None,
        help="Validate the layer taxonomy against a canonical spec and hard-fail "
        "on divergence, against the `methodology:` block of that model's "
        "declaration (deepseek-v4-pro: 3 hash + 29 HCA + 29 CSA). A pure "
        "validator with no behavioural effect.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        help="Tensor-parallel width. Inferred from the trace filename when absent, "
        "which the benchmark catalog shows is unreliable (it states tp=8).",
    )
    parser.add_argument(
        "--dp",
        type=int,
        help="Data-parallel width. Previously inferred as tp-when-dp-attn-on, "
        "which is a guess: dp is its own axis.",
    )
    parser.add_argument(
        "--dp-attn",
        choices=["on", "off"],
        help="Whether attention is data-parallel (ATOM --enable-dp-attention). "
        "Changes the per-rank token count in every attention formula.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        help="e.g. fp8. No HF config carries this; ATOM takes it as a server flag.",
    )
    parser.add_argument(
        "--index-cache-dtype",
        help="e.g. fp4 (DSV4 Lightning Indexer). Also a server flag, also not in "
        "the HF config.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="HF config.json; reads num_hash_layers when --hash-layers is omitted",
    )
    parser.add_argument(
        "--sort",
        default="algorithm",
        choices=["algorithm", "recoverable"],
        help="algorithm = execution order inside a layer (default); "
        "recoverable = biggest win first",
    )
    args = parser.parse_args()

    here = Path(__file__).parent
    peaks_path = args.peaks or here / "peaks" / "mi355x.yaml"
    map_path = args.map or here / "module_map.yaml"
    peaks = yaml.safe_load(Path(peaks_path).read_text(encoding="utf-8"))
    mapping = yaml.safe_load(Path(map_path).read_text(encoding="utf-8")) or {}
    stages = mapping.get("stages") or {}
    # Structure comes from the model declaration; `stages` stays in the shared
    # module_map.yaml because grouping for display is not model structure.
    decl = load_model_decl(args.model, here) if args.model else {}
    families = decl.get("families") or mapping.get("families") or DEFAULT_FAMILIES
    FAMILY_ORDER[:] = [f["name"] for f in families]

    if not args.output:
        # Derived, not required: a run should be named by what it is, and the
        # caller should not have to spell that out.
        m = re.search(r"[_.]c(\d+)[_.]", Path(args.run_trace).name)
        conc = m.group(1) if m else "?"
        stem = args.model or "model"
        args.output = f"roofline_{stem}_{args.phase}_c{conc}"

    events = load_events(args.run_trace)
    model_cfg: dict[str, Any] = {}
    if args.model_config:
        model_cfg = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    n_hash = args.hash_layers
    if n_hash is None and model_cfg:
        n_hash = int(model_cfg.get("num_hash_layers", 0))
    if n_hash is None:
        n_hash = 0
        print(
            "note: --hash-layers not given; leading hash layers will be reported as "
            "hca/csa (they are indistinguishable from markers alone)"
        )
    args._n_hash = n_hash
    steps = phase_steps(events)
    if args.phase == "decode":
        if not args.capture_trace:
            raise SystemExit(
                "--phase decode needs --capture-trace: replay emits no markers, so "
                "operator identity has to come from the per-bs capture file."
            )
        template = decode_template(args.capture_trace)
        if not template:
            raise SystemExit(
                f"{args.capture_trace} yielded an empty template. It must contain "
                "cuda_runtime launch events; a CPU-only export has none. Check with "
                '`zcat <file>.json.gz | grep -c \'"cat": "cuda_runtime"\'`.'
            )
        print(f"decode template: {len(template)} launches from {args.capture_trace}")
        fams = {layer: "" for layer, _, _, _ in template}
        by_layer: dict[int, set[str]] = {}
        for layer, path, _, _ in template:
            by_layer.setdefault(layer, set()).add(path)
        synth = [
            {"cat": "gpu_user_annotation", "name": f"layers.{layer}.{path}", "ts": 0.0}
            for layer, paths in by_layer.items()
            for path in paths
        ]
        fams = layer_families(synth, n_hash, families)
        sample_stats: dict[str, Any] = {}
        raw = collect_decode_ops(events, template, fams, stats=sample_stats)
        sampled_steps = int(sample_stats.get("sampled_steps") or 0)
    else:
        raw = collect_ops(events, n_hash, families)
        # Marker-driven phases aggregate every step the trace contains, so the
        # per-step figures divide by the step count the phase labels report.
        sampled_steps = int((steps.get(args.phase) or {}).get("steps") or 0)
    ops = merge_by_operator(fold_split_k(raw))
    if args.phase != "all":
        ops = [op for op in ops if op.phase == args.phase]
    fam_map_for_order = (
        fams if args.phase == "decode" else layer_families(events, n_hash, families)
    )
    algorithm_order(ops, fam_map_for_order)
    # The run's operating point, read from the trace filename and the phase labels.
    # TP/dp come from the filename because ATOM does not annotate them; when the
    # name does not say, the safe reading is 1 -- an over-large tp would silently
    # shrink every per-rank shape.
    # The operating point. Stated, or read off the filename as a last resort.
    #
    # The filename is a weak source and the benchmark catalog proves it: every
    # entry in .github/benchmark/models.json carries `tp` explicitly (8, not the 4
    # this trace happens to use) plus variant flags -- DPA, TBO, MTP3, DSpark --
    # that change the operating point and appear nowhere in a trace name. So the
    # flags are the interface and the filename is the fallback, not the reverse.
    #
    # `dp` in particular was being INFERRED as `tp if dpon else 1`, which is a
    # guess: data-parallel width is its own axis. Only c1/dpoff makes the guess
    # harmless, because both branches then give M = batch.
    # Cache precisions. No HF config carries them; ATOM takes them as server flags,
    # so the tool has to be told.
    runtime = {
        k: v
        for k, v in (
            ("kv_cache_dtype", args.kv_cache_dtype),
            ("index_cache_dtype", args.index_cache_dtype),
        )
        if v
    }
    name = Path(args.run_trace).name
    # The trace filename is the only place ATOM records model / quant / scenario.
    # Weak, but it is what a benchmark run is named by, so record what it says and
    # let the identity block show where each field came from.
    _CHIPS = {
        "platform": r"(mi\d{3}x|b\d{3}|h\d{3})",
        "framework": r"(atom|sglang|trtllm|vllm)",
        "model": r"(dsv\d+(?:-pro)?|glm-[\d-]+|qwen[\d.]+|llama-[\d.]+|kimi-\w+)",
        "quant": r"(?:dsv\d+|glm[\d-]*|qwen[\d.]*)-(fp4|fp8|mxfp4|mxfp8|bf16)",
        "run": r"(run\d{8})",
        "scenario": r"_(summarize|chat|agentic|code)_",
    }
    cfg_chips = {}
    chip_source = {}
    for key, pat in _CHIPS.items():
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            cfg_chips[key] = m.group(1).lower()
            chip_source[key] = "trace filename"
    # Stated beats inferred, and the identity block records which it was. The
    # filename is the weakest source in this tool -- it has already been wrong
    # about tp -- so it stays a fallback, never the interface.
    for key, value in (
        ("model", args.model),
        ("framework", args.framework),
        ("platform", args.platform),
    ):
        if value:
            cfg_chips[key] = value.lower()
            chip_source[key] = "stated"

    def _from_name(key: str, default: int) -> int:
        m = re.search(rf"[_.]{key}(\d+)[_.]", name)
        return int(m.group(1)) if m else default

    tp = args.tp or _from_name("tp", 1)
    dpon = args.dp_attn == "on" if args.dp_attn else ("dpon" in name)
    dp = args.dp if args.dp is not None else (tp if dpon else 1)
    stated = [
        f"{k}={v}"
        for k, v in (("tp", args.tp), ("dp", args.dp), ("dp-attn", args.dp_attn))
        if v is not None
    ]
    # The architecture selects the marker map, keyed as ATOM keys its own model
    # registry. Unmapped means nothing is priced -- loudly, rather than borrowing
    # another model's vocabulary.
    arch = (model_cfg.get("architectures") or [None])[0] if model_cfg else None
    if arch and arch not in formulas.MARKER_MAPS:
        print(
            f"NOTE: no marker map for {arch}. Only shape-carrying markers will be "
            f"priced; operators needing a cost model are left blank. Mapped: "
            f"{', '.join(sorted(formulas.MARKER_MAPS)) or '(none)'}"
        )
    bench = (
        bench_from_labels(events, steps, args.phase, tp, dp, dpon, model_cfg, runtime)
        if model_cfg and args.phase in ("prefill", "decode")
        else None
    )
    if bench is not None:
        print(
            f"operating point: bs={bench.batch} tok={bench.seq_len} tp={tp} "
            f"dp={dp} dp-attn={'on' if dpon else 'off'}"
            + (
                f"   (stated: {', '.join(stated)})"
                if stated
                else "   (all inferred from the trace filename -- pass --tp/--dp/"
                "--dp-attn to state them)"
            )
        )
        print(
            f"architecture   : {arch or '(no --model-config)'}"
            f"   marker map: "
            f"{len(formulas.MARKER_MAPS.get(arch, [])) if arch else 0} entries"
        )
        print(f"kv_seq_len     : {bench.kv_seq_len:,.0f}  ({bench.kv_seq_note})")
        d = bench.dtypes
        print(
            "dtypes         : "
            + " · ".join(
                f"{role}={getattr(d, role)}[{d.source.get(role, '?')}]"
                for role in ("linear_w", "expert_w", "act", "kv", "index")
            )
        )
    rows = [
        price(op, peaks, model_cfg or None, bench, sampled_steps, arch) for op in ops
    ]
    fam_map = fam_map_for_order
    observed: dict[str, int] = {
        fam: sum(1 for v in fam_map.values() if v == fam) for fam in FAMILY_ORDER
    }
    if args.methodology:
        want = (
            decl.get("methodology")
            if decl.get("model") == args.methodology
            else load_model_decl(args.methodology, here).get("methodology")
        )
        if not want:
            raise SystemExit(
                f"--methodology {args.methodology}: that declaration has no "
                f"`methodology:` block to check against."
            )
        check_methodology(args.methodology, want, observed)
    layer_counts = {
        "total_from_config": (
            int(model_cfg.get("num_hidden_layers", 0)) if model_cfg else 0
        ),
        "hash_from_config": n_hash,
        "observed_per_family": observed,
    }

    # Regime, measured rather than assumed. No kernel in this run runs faster than
    # a few microseconds however little work it does, so an operator whose entire
    # roofline time is below that floor is bound by neither ceiling -- reporting it
    # as "memory-bound, efficiency 0.00" names the wrong constraint and points at
    # the wrong fix (bandwidth instead of batching or fusion).
    #
    # The floor is taken from the run itself: the fastest per-firing kernel measured
    # anywhere in it. It is NOT kernel-launch overhead, which is around 1 us on this
    # stack -- launch is only about a quarter of it. The rest is per-kernel fixed
    # cost that a roofline does not model either (memory latency on a cold access,
    # wave ramp on a kernel too small to fill the machine). Calling the whole 4 us
    # "launch" would point at the wrong fix a second time: fusing kernels removes
    # the launch quarter, batching removes the rest. Nothing here is hardcoded, so a
    # faster stack or a larger batch moves the line on its own.
    # n_samples, not n_firings: n_firings is set only on priced rows, and the
    # fastest kernel in a run is often a gap-tier one, which keying on n_firings
    # would exclude from the search.
    fires = [
        float(r["busy_us"]) / int(r["n_samples"])
        for r in rows
        if int(r.get("n_samples") or 0) > 0 and float(r.get("busy_us") or 0) > 0
    ]
    latency_floor = min(fires) if fires else 0.0
    n_lat = 0
    for row in rows:
        n = int(row.get("n_firings") or 0)
        if row.get("tier") != "roofline" or not n:
            row["regime"] = ""
            continue
        per = float(row["t_roofline_us"]) / n
        row["t_roofline_per_firing_us"] = round(per, 4)
        if per < latency_floor:
            row["regime"] = "latency"
            n_lat += 1
        else:
            row["regime"] = row["bound"]
        # The same decomposition the step table uses, per operator, and it sums to
        # this row's measured time: work + issue + stall. `recoverable_us` lumps
        # the last two together, which is the right number to rank by but the
        # wrong one to act on -- removing a launch and raising achieved bandwidth
        # are different jobs. `issue` is what the run's own fastest kernel costs,
        # times this operator's firings, minus whatever its work already covers.
        issue, stall = decompose(
            float(row["t_roofline_us"]), float(row["measured_us"]), n, latency_floor
        )
        row["issue_us"] = round(issue, 3)
        row["stall_us"] = round(stall, 3)
    # A roofline is a FLOOR, so a row measured FASTER than its own floor needs an
    # explanation, and there are two very different ones:
    #
    #   * the formula counts work the kernel does not do, or
    #   * an input to the formula is itself an over-estimate. `active_experts`
    #     assumes uniform routing and real routing concentrates, so a skewed MoE
    #     run loads fewer expert weights than the floor charges -- the rows say so
    #     in their own note, and it is the first thing to check on a MoE row that
    #     lands here, ahead of the two below.
    #   * the bytes never came from HBM. MI355X carries 256 MB of Infinity Cache,
    #     and this roofline prices every byte at HBM bandwidth by construction, so
    #     an operator whose working set fits in cache can legitimately beat the HBM
    #     roof. That is not an error in the measurement, it is the HBM ceiling not
    #     being the binding constraint for that operator.
    #
    # Working-set size separates the two well enough to be worth printing: under the
    # cache, cache residency is plausible; far above it, the formula is wrong. Either
    # way the row is flagged and kept out of any claim, because a floor that exceeds
    # its own measurement cannot support one.
    over = [
        r
        for r in rows
        if r.get("tier") == "roofline" and float(r.get("efficiency") or 0) > 1.0
    ]
    for r in over:
        r["formula_overestimates"] = True
    # The kernel name is evidence the formula's author did not have to guess at.
    # Where it carries a dtype token, it outranks an assertion made from a paper or
    # a source comment -- that is exactly how the indexer ended up priced against
    # the fp4 ceiling while running an fp8 kernel.
    DTYPE_TOKENS = [
        (re.compile(r"afp8|a8w8|fp8gemm|_fp8|fp8_"), "fp8"),
        (re.compile(r"wfp4|mxfp4|_fp4"), "fp4"),
        (re.compile(r"hgemm_bf16|bf16gemm|_bf16"), "bf16"),
    ]
    contradicted = []
    for r in rows:
        if r.get("dtype_provenance") != "asserted":
            continue
        # Zero-FLOP operators are exempt. `mxfp4_moe_sort_kernel` names the data it
        # permutes, not arithmetic it performs -- the kernel does no maths at all,
        # so a dtype token in its name says nothing about a ceiling.
        if not float(r.get("flops") or 0):
            continue
        text = str(r.get("kernel_full") or r.get("kernel") or "").lower()
        hints = {d for pat, d in DTYPE_TOKENS if pat.search(text)}
        # Storage and ceiling are two answers, and the name may carry either. The
        # indexer stores fp4 and computes fp8; flagging it against mem alone
        # reported a contradiction where the model was right.
        # Plus whatever the byte expression actually counted. A mixed-precision op
        # cannot state one `mem`: the compressor scatter reads bf16 and writes fp8,
        # and the expression -- "read bf16 ... write fp8" -- is the formula's real
        # claim, so that is what the kernel name should be checked against.
        claimed = {
            r.get("dtype_a"),
            r.get("dtype_o"),
            str(r.get("formula_peak_key") or "").replace("matrix_", ""),
        }
        expr = str(r.get("bytes_expr") or "") + str(r.get("flops_expr") or "")
        claimed |= {d for _, d in DTYPE_TOKENS if re.search(rf"\b{d}\b", expr)}
        if hints and not (hints & claimed):
            contradicted.append((r, sorted(hints)))
    if contradicted:
        print(
            f"DTYPE CHECK: {len(contradicted)} operators whose kernel NAME carries a "
            f"dtype the formula does not assert. The name is the stronger evidence:"
        )
        for r, hints in contradicted[:8]:
            print(
                f"  {r['op'][:32]:32s} {r.get('formula', ''):20s} "
                f"formula says {r.get('dtype_a')}, kernel says {'/'.join(hints)}"
                f"   [{r.get('kernel', '')[:34]}]"
            )

    # Fix-then-sweep, as a standing check rather than a habit. A GEMM has a
    # closed-form cost by definition, so an unpriced kernel whose name says GEMM is
    # a missing needle, not an operator that resists modelling. The same miss has
    # now happened three times -- one operator's GEMM reaching the trace through
    # aiter, aiter mono-tile and rocBLAS/Tensile under three different names -- and
    # each time it was spotted by eye. This reports it every run instead.
    GEMMISH = ("cijk", "gemm", "mfma", "hgemm", "mono_tile", "xdl", "blockscale")
    missed = [
        r
        for r in rows
        if r.get("tier") != "roofline"
        and any(
            g in str(r.get("kernel_full") or r.get("kernel") or "").lower()
            for g in GEMMISH
        )
    ]
    if missed:
        busy = sum(float(r.get("busy_us") or 0) for r in rows) or 1.0
        share = sum(float(r.get("busy_us") or 0) for r in missed) / busy
        print(
            f"UNPRICED GEMMS: {len(missed)} kernels name themselves a GEMM but no "
            f"needle claims them ({share:.1%} of GPU time). Either add the needle or "
            f"record why the marker is not the operator:"
        )
        for r in sorted(missed, key=lambda r: -(float(r.get("busy_us") or 0)))[:6]:
            print(
                f"  {float(r['busy_us']) / busy:6.2%}  {r['op'][:34]:34s} "
                f"{str(r.get('kernel'))[:40]}"
            )

    # Capacity from peaks; see the note there on why it is uncertain by 2x and why
    # no cache ROOF is drawn (AMD publishes no Infinity Cache bandwidth).
    cache = peaks.get("cache") or {}
    LLC_BYTES = float(cache.get("llc_bytes") or 256e6)
    LLC_LOW = LLC_BYTES / 2
    if over:
        print(
            f"FLAGGED: {len(over)} operators are measured FASTER than their own "
            f"roofline floor. Not used in any total:"
        )
        for r in sorted(over, key=lambda r: -float(r["efficiency"]))[:8]:
            ws = float(r.get("bytes") or 0) / max(int(r.get("n_firings") or 1), 1)
            if r.get("bw_kind") == "interconnect":
                # A collective's bytes cross the fabric; cache residency has nothing
                # to say about it. If it beats its own floor the link model is
                # wrong -- most likely the wrong collective (a reduce-scatter moves
                # message/N, not the whole message).
                why = (
                    "collective beats its own link model -> wrong collective or "
                    "wrong link count, not cache"
                )
            elif ws <= LLC_LOW:
                why = (
                    f"working set fits even the conservative {LLC_LOW / 1e6:,.0f} MB "
                    f"cache -> HBM was never the ceiling"
                )
            elif ws <= LLC_BYTES:
                why = (
                    f"working set is between {LLC_LOW / 1e6:,.0f} and "
                    f"{LLC_BYTES / 1e6:,.0f} MB -- cache residency depends on which "
                    f"capacity applies, so the cause is undecided"
                )
            else:
                why = "working set far exceeds cache -> the formula overcounts"
            r["overestimate_reason"] = why
            print(
                f"  {float(r['efficiency']):6.2f}x  {r['op'][:32]:32s} "
                f"{r.get('formula', ''):18s} roof {float(r['t_roofline_us']) / 1000:6.2f} ms "
                f"vs {float(r['busy_us']) / 1000:6.2f} ms  |  {ws / 1e6:6.1f} MB/firing: {why}"
            )

    if latency_floor:
        lat_us = sum(float(r["busy_us"]) for r in rows if r.get("regime") == "latency")
        busy = sum(float(r.get("busy_us") or 0) for r in rows) or 1.0
        print(
            f"latency floor  : {latency_floor:.2f} us/firing = fastest kernel in the "
            f"run (~1 us of that is launch; the rest is per-kernel fixed cost); "
            f"{n_lat} operators price below it = {lat_us / busy:.0%} of GPU time, "
            f"bound by neither ceiling"
        )
    skewed = sorted(
        (r for r in rows if r.get("outlier_dominated")),
        key=lambda r: -float(r.get("top1_share") or 0),
    )
    if skewed:
        print(
            f"one-firing totals: {len(skewed)} operator(s) whose measured time is "
            f"dominated by a single firing -- their efficiency is a statement about "
            f"that firing, not about the operator"
        )
        for r in skewed[:5]:
            print(
                f"  {r['op']:<28} {r.get('formula') or '-'!s:<20} "
                f"{float(r['top1_share']):5.0%} of {float(r['measured_us']) / 1000:8.1f} ms "
                f"in one of {int(r['n_samples']):5d} firings   "
                f"eff {r['efficiency']} vs p50 {r.get('efficiency_p50')}"
            )

    for row in rows:
        row["stage"] = stage_of(row["op"], stages)
    if args.sort == "algorithm":
        rows.sort(key=lambda r: r["seq"])
    else:
        rows.sort(key=lambda r: -(r["recoverable_us"] or 0))

    total = sum(r["measured_us"] for r in rows)
    priced = sum(r["measured_us"] for r in rows if r["tier"] == "roofline")
    # Two different coverages, and conflating them overstates what the ceiling
    # speaks to. `priced` is time spent in operators we CAN price; but inside those
    # same markers only the GEMM kernels are compared against the matrix ceiling --
    # their comm and quant kernels are not. So the ceiling's real reach is the
    # narrower number, and t_roofline is a floor for that slice alone.
    gemm_priced = sum(
        float(r.get("measured_us") or 0) for r in rows if r["tier"] == "roofline"
    )
    t_roofline_sum = sum(
        float(r.get("t_roofline_us") or 0) for r in rows if r["tier"] == "roofline"
    )
    coverage = priced / total if total else 0.0
    ceiling_reach = gemm_priced / total if total else 0.0

    out = Path(f"{args.output}.csv")
    if rows:
        # Gap-tier rows carry fewer keys than roofline-tier ones, so the header has
        # to be the union in first-seen order -- taking rows[0] alone drops columns
        # (or raises) depending on which tier happens to sort first.
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)

    write_xlsx(f"{args.output}.xlsx", rows, args, peaks, coverage, total, priced)

    summary = {
        "run_trace": args.run_trace,
        "phase": args.phase,
        # Everything needed to place this run in a model x concurrency grid without
        # parsing a filename. A directory of these is a site; a JSON that cannot say
        # which model and which concurrency it is cannot be filed, and the fields
        # were being printed to stdout and then thrown away.
        "identity": {
            "architecture": arch,
            "model": cfg_chips.get("model"),
            "quant": cfg_chips.get("quant"),
            "platform": peaks.get("platform"),
            "framework": cfg_chips.get("framework"),
            "run": cfg_chips.get("run"),
            "scenario": cfg_chips.get("scenario"),
            "source": chip_source,
            "phase": args.phase,
            "concurrency": bench.batch if bench else None,
            "tokens_per_step": bench.seq_len if bench else None,
            "tp": tp,
            "dp": dp,
            "dp_attn": dpon,
            "stated": stated,
            "dtypes": (
                {
                    role: {
                        "value": getattr(bench.dtypes, role),
                        "source": bench.dtypes.source.get(role),
                    }
                    for role in ("linear_w", "expert_w", "act", "kv", "index")
                }
                if bench
                else {}
            ),
        },
        "platform": peaks.get("platform"),
        "peaks_convention": peaks.get("convention"),
        # The peaks these numbers were computed WITH, not the peaks a reader's copy
        # of the yaml happens to hold. A renderer with its own copy of the peaks
        # drifts the moment one is corrected, and then the dots and the ceiling
        # drawn beside them disagree on the same page.
        "peaks": {
            "compute_tflops": peaks.get("compute_tflops", {}),
            "mem_bw_gbps": peaks.get("mem_bw_gbps"),
            "interconnect_bw_gbps": peaks.get("interconnect_bw_gbps"),
        },
        "n_operators": len(rows),
        "measured_total_us": round(total, 1),
        "roofline_tier_us": round(priced, 1),
        "coverage": round(coverage, 4),
        "ceiling_reach": round(ceiling_reach, 4),
        "gemm_priced_us": round(gemm_priced, 1),
        "t_roofline_sum_us": round(t_roofline_sum, 1),
        "sampled_steps": sampled_steps,
        "latency_floor_us": round(latency_floor, 3),
        # Layer counts as the run actually shows them, next to what the config
        # declares. They do not agree and the disagreement is real: hash layers are
        # invisible to markers (they carry the same operator signatures as their
        # neighbours), so the tool's csa/hca families absorb them. Reporting only
        # the config numbers would hide that; reporting only the observed ones
        # would lose the fact that 3 hash layers exist at all.
        "layer_counts": layer_counts,
        "kv_seq_len": round(bench.kv_seq_len, 1) if bench else "",
        "kv_seq_note": bench.kv_seq_note if bench else "",
        "phase_steps": steps,
        "rows": rows,
    }
    Path(f"{args.output}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    bad = [r for r in rows if r.get("over_peak")]
    if bad:
        print("!! IMPOSSIBLE: these operators exceed their ceiling -- do not publish")
        for row in bad:
            print(
                f"   {row['op']} M={row['M']} achieved {row['achieved_tflops']} TFLOP/s"
                f" vs ceiling {row['ceiling_tflops']}  (eff {row['efficiency']})"
            )
        print()

    print(f"operators      : {len(rows)}  ({args.phase})")
    print(f"measured total : {total / 1000:.1f} ms")
    print(f"priceable ops  : {priced / 1000:.1f} ms   ({coverage:.1%} of busy)")
    print(
        f"  vs ceiling   : {gemm_priced / 1000:.1f} ms   ({ceiling_reach:.1%} of busy)"
        f"  <- GEMM kernels only, the ceiling's real reach"
    )
    print(
        f"  floor        : {t_roofline_sum / 1000:.1f} ms   "
        f"gap {gemm_priced / t_roofline_sum:.2f}x"
        if t_roofline_sum
        else ""
    )
    print(f"unbounded      : {(total - gemm_priced) / 1000:.1f} ms   no floor exists")
    print()
    for phase, stat in steps.items():
        print(
            f"{phase:8s} steps : {stat['steps']:5d}  median {stat['median_us'] / 1000:8.2f} ms/step"
            f"  (min {stat['min_us'] / 1000:.2f} / max {stat['max_us'] / 1000:.2f}"
            f", {stat['streams']:.0f} stream-record/step)  total {stat['total_us'] / 1e6:.2f} s"
            + (
                "   !! n<3: the median cannot exclude a compile-carrying first step"
                if stat["warmup_suspect"]
                else ""
            )
        )
    print(f"written        : {out}  and  {args.output}.json")
    print()
    header = (
        f"{'#':>3s} {'module':30s} {'kernel':26s} {'cls':5s} {'M':>11s} "
        f"{'csa':>7s} {'hca':>7s} {'FLOP/B':>7s} {'bound':7s} {'eff':>6s} {'recov':>7s}"
    )
    print(header)
    print("-" * len(header))

    def ms(value: Any) -> str:
        """Gap-tier rows leave numeric columns empty; render that, do not crash."""
        try:
            return f"{float(value) / 1000:.2f}"
        except (TypeError, ValueError):
            return "-"

    for row in rows:
        eff = f"{row['efficiency']:.3f}" if row["efficiency"] != "" else "   gap"
        ai = f"{row['flop_per_byte']:.1f}" if row["flop_per_byte"] != "" else "-"
        print(
            f"{row['seq']:3d} {row['op'][:30]:30s} {row['kernel'][:26]:26s} "
            f"{row['kclass']:5s} {row['M']!s:>11s} "
            f"{ms(row.get('csa_us')):>7s} {ms(row.get('hca_us')):>7s} "
            f"{ai:>7s} {row['bound'] or '-':7s} {eff:>6s} {ms(row.get('recoverable_us')):>7s}"
        )


if __name__ == "__main__":
    main()
