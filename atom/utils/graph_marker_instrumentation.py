# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional


_GRAPH_MARKER_RE = re.compile(
    r"""torch\.ops\.aiter\.graph_marker\.default\(
        \s*[^,]+,\s*
        (?P<q>['"])(?P<name>.*?)(?P=q)
        \s*\)""",
    re.VERBOSE,
)

_SUBGRAPH_ID_RE = re.compile(r"artifact_shape_[^/]+_subgraph_(\d+)")

_GRAPH_MARKER_LINE_RE = re.compile(
    r"""^(?P<indent>\s*)
        (?P<lhs>[A-Za-z_]\w*)\s*=\s*
        torch\.ops\.aiter\.graph_marker\.default\(
            \s*(?P<arg>[^,]+?)\s*,\s*
            (?P<q>['"])(?P<name>.*?)(?P=q)\s*
        \)\s*$""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Marker:
    idx: int
    indent: str
    name: str


def _iter_py_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _ensure_record_function_import(lines: list[str]) -> None:
    # If already imported or referenced via qualified name, do nothing.
    if any(
        ("record_function" in l and ("import" in l or "from torch" in l))
        for l in lines
    ):
        return

    # Insert `from torch.profiler import record_function` after the first
    # real `import torch` line outside the initial docstring.
    in_doc = False
    for i, line in enumerate(lines):
        if i == 0 and line.lstrip().startswith('"""'):
            in_doc = True
        if in_doc and line.rstrip().endswith('"""') and i != 0:
            in_doc = False
            continue
        if in_doc:
            continue
        if re.match(r"^\s*import\s+torch\b", line):
            lines.insert(i + 1, "from torch.profiler import record_function\n")
            return

    # Fallback: put it near the top (after any shebang/encoding if present).
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    lines.insert(insert_at, "from torch.profiler import record_function\n")


def _collect_markers(lines: list[str]) -> list[_Marker]:
    out: list[_Marker] = []
    for i, line in enumerate(lines):
        m = _GRAPH_MARKER_RE.search(line)
        if not m:
            continue
        indent = re.match(r"^(\s*)", line).group(1)  # type: ignore[union-attr]
        out.append(_Marker(idx=i, indent=indent, name=m.group("name")))
    return out


def _prefix_and_kind(name: str) -> Optional[tuple[str, str]]:
    if name.endswith("_start"):
        return name[: -len("_start")], "start"
    if name.endswith("_end"):
        return name[: -len("_end")], "end"
    return None


def _already_wrapped(lines: list[str], indent: str, prefix: str, start_idx: int, end_idx: int) -> bool:
    needle = f'{indent}with record_function("{prefix}"):\n'
    for i in range(start_idx, min(end_idx, len(lines))):
        if lines[i] == needle:
            return True
    return False


def _wrap_region_with_record_function(
    lines: list[str],
    *,
    start_marker_idx: int,
    end_marker_idx: int,
    prefix: str,
    indent: str,
    layer_id: Optional[int] = None,
) -> None:
    """
    Transform:
      <indent>... graph_marker(..., "<prefix>_start")
      <indent>LINE_A
      <indent>LINE_B
      <indent>... graph_marker(..., "<prefix>_end")
    Into:
      <indent>... graph_marker(..., "<prefix>_start")
      <indent>with record_function("<prefix>"):
      <indent>    LINE_A
      <indent>    LINE_B
      <indent>... graph_marker(..., "<prefix>_end")
    """
    if end_marker_idx <= start_marker_idx + 1:
        return

    if layer_id is not None and layer_id >= 0:
        tag = f"layer_{layer_id}_{prefix}"
    else:
        tag = prefix
    with_line = f'{indent}with record_function("{tag}"):\n'
    insert_at = start_marker_idx + 1

    # If we already inserted a record_function line in a previous run, upgrade it
    # in-place (e.g. "mlp" -> "layer_0_mlp") and exit without touching indentation.
    if insert_at < len(lines) and lines[insert_at].startswith(f"{indent}with record_function("):
        if lines[insert_at] != with_line:
            lines[insert_at] = with_line
        return

    # Otherwise, insert a new record_function wrapper and indent the region.
    lines.insert(insert_at, with_line)

    # Re-indent the region between start_marker and end_marker (exclusive of end marker).
    region_start = insert_at + 1
    region_end = end_marker_idx + 1  # end marker shifted down by 1 due to insertion
    indent_prefix = indent
    extra = " " * 4
    for i in range(region_start, region_end):
        line = lines[i]
        if line.strip() == "":
            continue
        if line.startswith(indent_prefix):
            lines[i] = indent_prefix + extra + line[len(indent_prefix):]


def _layer_id_from_wrapper_path(path: str) -> Optional[int]:
    """
    Derive layer id from wrapper file path:
      .../artifact_shape_<shape>_subgraph_<N>/... -> layer_id = N - 1
    Returns None if the pattern isn't found.
    """
    m = _SUBGRAPH_ID_RE.search(path)
    if not m:
        return None
    try:
        subgraph_id = int(m.group(1))
    except ValueError:
        return None
    return subgraph_id - 1


def _strip_runtime_graph_markers(lines: list[str]) -> bool:
    """
    Remove runtime overhead of graph markers in generated wrapper code.

    - Replace `x = torch.ops.aiter.graph_marker.default(y, '...')` with `x = y`
    - Drop assert_size_stride / assert_alignment lines that specifically refer to
      `torch.ops.aiter.graph_marker.default` (they become redundant).
    """
    out: list[str] = []
    changed = False
    for line in lines:
        m = _GRAPH_MARKER_LINE_RE.match(line.rstrip("\n"))
        if m:
            indent = m.group("indent")
            lhs = m.group("lhs")
            arg = m.group("arg").strip()
            out.append(f"{indent}{lhs} = {arg}\n")
            changed = True
            continue

        if (
            ("assert_size_stride" in line or "assert_alignment" in line)
            and "torch.ops.aiter.graph_marker.default" in line
        ):
            changed = True
            continue

        out.append(line)

    if changed:
        lines[:] = out
    return changed


def instrument_record_functions_in_file(path: str, *, strip_markers: bool = True) -> bool:
    """
    Returns True if the file was modified.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return False

    markers = _collect_markers(lines)
    if not markers:
        return False

    # Build intervals by matching <prefix>_start / <prefix>_end.
    stack: dict[str, _Marker] = {}
    intervals: list[tuple[_Marker, _Marker, str]] = []
    for mk in markers:
        pk = _prefix_and_kind(mk.name)
        if pk is None:
            continue
        prefix, kind = pk
        if kind == "start":
            stack[prefix] = mk
        else:
            start_mk = stack.pop(prefix, None)
            if start_mk is None:
                continue
            # Use start indent as the wrapping indent (generated code is consistent).
            intervals.append((start_mk, mk, prefix))

    # Even if we can't form any intervals, we still might want to strip marker
    # calls from already-instrumented wrappers (best-effort).
    has_intervals = bool(intervals)

    layer_id = _layer_id_from_wrapper_path(path)

    # Apply from bottom to top so indices stay valid.
    wrapped_or_upgraded = False
    if has_intervals:
        for start_mk, end_mk, prefix in sorted(intervals, key=lambda t: t[0].idx, reverse=True):
            _wrap_region_with_record_function(
                lines,
                start_marker_idx=start_mk.idx,
                end_marker_idx=end_mk.idx,
                prefix=prefix,
                indent=start_mk.indent,
                layer_id=layer_id,
            )
            wrapped_or_upgraded = True

    stripped = False
    if strip_markers:
        # Only strip marker calls if we either:
        # - wrapped/upgraded this run, or
        # - the file already contains record_function blocks (previous run)
        already_has_record = any("with record_function(" in l for l in lines)
        if wrapped_or_upgraded or already_has_record:
            stripped = _strip_runtime_graph_markers(lines)

    changed = wrapped_or_upgraded or stripped
    if changed:
        if wrapped_or_upgraded:
            _ensure_record_function_import(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return changed


def instrument_record_functions_in_dir(root: str, *, strip_markers: bool = True) -> int:
    """
    Walk `root` and instrument all generated `.py` wrapper files.
    Returns the number of modified files.
    """
    changed = 0
    for fp in _iter_py_files(root):
        if instrument_record_functions_in_file(fp, strip_markers=strip_markers):
            changed += 1
    return changed


