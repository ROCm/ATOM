#!/usr/bin/env python3
"""Convert AIPerf per-request records to Chrome Trace Event JSON.

The output can be opened locally with chrome://tracing or
https://ui.perfetto.dev.  It deliberately uses only the Python standard
library so the conversion can run on a headless benchmark server.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

NS_PER_MS = 1_000_000
NS_PER_US = 1_000
PID = 1
CACHE_ANOMALY_THRESHOLD = 0.80


def metric_value(metrics: dict[str, Any], name: str, default: Any = None) -> Any:
    """Return an AIPerf metric value from its {value, unit} wrapper."""
    entry = metrics.get(name)
    if not isinstance(entry, dict):
        return default
    return entry.get("value", default)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records and report malformed line numbers."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            records.append(record)
    return records


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _session_value(metadata: dict[str, Any], session_key: str) -> str:
    value = metadata.get(session_key)
    if value is None:
        value = (
            metadata.get("root_correlation_id")
            or metadata.get("x_correlation_id")
            or metadata.get("conversation_id")
            or "unknown"
        )
    return str(value)


def _short_id(value: str, length: int = 12) -> str:
    return value if len(value) <= length else f"{value[:length]}…"


def _event(
    name: str,
    *,
    ts_us: int,
    tid: int,
    category: str,
    args: dict[str, Any] | None = None,
    duration_us: int | None = None,
    color: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "name": name,
        "cat": category,
        "ph": "X" if duration_us is not None else "i",
        "pid": PID,
        "tid": tid,
        "ts": ts_us,
        "args": args or {},
    }
    if duration_us is not None:
        event["dur"] = duration_us
    else:
        event["s"] = "t"
    if color is not None:
        event["cname"] = color
    return event


def _record_sort_key(record: dict[str, Any]) -> tuple[float, int]:
    metadata = record.get("metadata", {})
    start = _number(metadata.get("request_start_ns"))
    return (
        start if start is not None else float("inf"),
        int(metadata.get("turn_index", 0)),
    )


def _cache_observation(metrics: dict[str, Any]) -> tuple[float | None, bool | None]:
    input_tokens = _number(metric_value(metrics, "input_sequence_length"))
    cache_tokens = _number(metric_value(metrics, "usage_prompt_cache_read_tokens"))
    if input_tokens is None or cache_tokens is None or input_tokens <= 0:
        return None, None
    ratio = cache_tokens / input_tokens
    return ratio, ratio < CACHE_ANOMALY_THRESHOLD


def _tree_key(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("root_correlation_id")
        or metadata.get("x_correlation_id")
        or metadata.get("conversation_id")
        or "unknown"
    )


def _request_session_key(metadata: dict[str, Any], tree: str) -> str:
    return str(
        metadata.get("x_correlation_id") or metadata.get("conversation_id") or tree
    )


def _pack_request_rows(
    items: list[tuple[dict[str, Any], dict[str, Any], str, float, float]],
) -> tuple[dict[int, int], int]:
    """Pack overlapping requests into non-overlapping rows."""
    order = sorted(
        range(len(items)), key=lambda index: (items[index][3], items[index][4])
    )
    row_ends: list[float] = []
    rows: dict[int, int] = {}
    for index in order:
        start_ns, end_ns = items[index][3], items[index][4]
        for row, row_end in enumerate(row_ends):
            if start_ns >= row_end:
                row_ends[row] = end_ns
                rows[id(items[index][0])] = row
                break
        else:
            rows[id(items[index][0])] = len(row_ends)
            row_ends.append(end_ns)
    return rows, len(row_ends)


def _assign_tree_slots(
    spans: dict[str, tuple[float, float]], concurrency: int
) -> dict[str, int]:
    """Assign non-overlapping tree lifetimes to a fixed number of slots."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    slot_ends: list[float] = []
    assignments: dict[str, int] = {}
    for tree, (start_ns, end_ns) in sorted(spans.items(), key=lambda item: item[1][0]):
        available = next(
            (slot for slot, slot_end in enumerate(slot_ends) if start_ns >= slot_end),
            None,
        )
        if available is None:
            if len(slot_ends) < concurrency:
                available = len(slot_ends)
                slot_ends.append(end_ns)
            else:
                # Keep the requested number of tracks for an incomplete or
                # inconsistent export whose tree spans appear over capacity.
                available = min(range(concurrency), key=slot_ends.__getitem__)
                slot_ends[available] = max(slot_ends[available], end_ns)
        else:
            slot_ends[available] = end_ns
        assignments[tree] = available
    return assignments


def _build_slot_layout(
    usable: list[tuple[dict[str, Any], dict[str, Any], str, float, float]],
    concurrency: int,
) -> tuple[dict[int, tuple[int, int]], dict[str, tuple[int, float, float]], int]:
    """Return request placement, tree spans, and the number of used slots."""
    trees: dict[str, list[tuple[dict[str, Any], dict[str, Any], str, float, float]]] = (
        defaultdict(list)
    )
    tree_sessions: dict[
        str, dict[str, list[tuple[dict[str, Any], dict[str, Any], str, float, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    for item in usable:
        tree = _tree_key(item[1])
        session = _request_session_key(item[1], tree)
        trees[tree].append(item)
        tree_sessions[tree][session].append(item)

    spans = {
        tree: (min(item[3] for item in items), max(item[4] for item in items))
        for tree, items in trees.items()
    }
    assignments = _assign_tree_slots(spans, concurrency)
    placement: dict[int, tuple[int, int]] = {}
    tree_layout: dict[str, tuple[int, float, float]] = {}
    used_slots = max(assignments.values(), default=-1) + 1

    for tree, members in tree_sessions.items():
        slot = assignments[tree]
        root_session = (
            tree
            if tree in members
            else min(
                members,
                key=lambda session: (
                    min(item[3] for item in members[session]),
                    min((item[1].get("agent_depth") or 0) for item in members[session]),
                ),
            )
        )
        root_items = members.get(root_session, [])
        root_rows, root_row_count = _pack_request_rows(root_items)
        for item in root_items:
            placement[id(item[0])] = (slot, root_rows[id(item[0])])

        child_row_ends: list[float] = []
        children = sorted(
            (session for session in members if session != root_session),
            key=lambda session: min(item[3] for item in members[session]),
        )
        for session in children:
            child_items = members[session]
            child_rows, child_row_count = _pack_request_rows(child_items)
            child_start = min(item[3] for item in child_items)
            child_end = max(item[4] for item in child_items)
            row_base = 0
            for row in range(len(child_row_ends) + 1):
                if all(
                    child_row_ends[row + offset] <= child_start
                    for offset in range(child_row_count)
                    if row + offset < len(child_row_ends)
                ):
                    row_base = row
                    break
            child_row_ends.extend(
                [0.0] * (row_base + child_row_count - len(child_row_ends))
            )
            for offset in range(child_row_count):
                child_row_ends[row_base + offset] = child_end
            for item in child_items:
                placement[id(item[0])] = (
                    slot,
                    root_row_count + row_base + child_rows[id(item[0])],
                )

        tree_layout[tree] = (slot, spans[tree][0], spans[tree][1])
    return placement, tree_layout, used_slots


def _append_wait_prefill_analysis(
    events: list[dict[str, Any]],
    usable: list[tuple[dict[str, Any], dict[str, Any], str, float, float]],
    placement: dict[int, tuple[int, int]],
    *,
    inspect_slot: int,
    wait_tid: int | None,
    overlap_tid: int,
    origin_ns: float,
) -> None:
    """Add wait/TTFB and wait∩prefill events for one fixed slot."""
    for record, metadata, _, start_ns, end_ns in usable:
        slot, _ = placement[id(record)]
        if slot != inspect_slot:
            continue
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        waiting_ms = _number(metric_value(metrics, "http_req_waiting"))
        if waiting_ms is None or waiting_ms <= 0:
            continue
        sending_ms = _number(metric_value(metrics, "http_req_sending")) or 0.0
        wait_start_ns = start_ns + sending_ms * NS_PER_MS
        wait_end_ns = min(end_ns, wait_start_ns + waiting_ms * NS_PER_MS)
        request_number = metadata.get("session_num", metadata.get("turn_index", "?"))
        if wait_tid is not None:
            events.append(
                _event(
                    f"wait/TTFB #{request_number} | {waiting_ms:.1f} ms",
                    ts_us=round((wait_start_ns - origin_ns) / NS_PER_US),
                    tid=wait_tid,
                    category="http",
                    args={
                        "request_num": request_number,
                        "wait_start_ns": int(wait_start_ns),
                        "wait_end_ns": int(wait_end_ns),
                        "wait_ttfb_ms": waiting_ms,
                        "root_correlation_id": _tree_key(metadata),
                    },
                    duration_us=max(
                        0, round((wait_end_ns - wait_start_ns) / NS_PER_US)
                    ),
                )
            )

        ttft_ms = _number(metric_value(metrics, "time_to_first_token"))
        if ttft_ms is None or ttft_ms < 0:
            continue
        prefill_end_ns = min(end_ns, start_ns + ttft_ms * NS_PER_MS)
        overlap_start_ns = max(start_ns, wait_start_ns)
        overlap_end_ns = min(prefill_end_ns, wait_end_ns)
        overlap_ns = overlap_end_ns - overlap_start_ns
        if overlap_ns <= 0:
            continue
        prefill_ms = max(0.0, (prefill_end_ns - start_ns) / NS_PER_MS)
        overlap_ms = overlap_ns / NS_PER_MS
        events.append(
            _event(
                f"wait∩prefill #{request_number} | {overlap_ms:.1f} ms",
                ts_us=round((overlap_start_ns - origin_ns) / NS_PER_US),
                tid=overlap_tid,
                category="overlap",
                args={
                    "request_num": request_number,
                    "overlap_ms": overlap_ms,
                    "prefill_ms": prefill_ms,
                    "prefill_covered_pct": (
                        100.0 * overlap_ms / prefill_ms if prefill_ms else 0.0
                    ),
                    "wait_ttfb_ms": waiting_ms,
                    "time_to_first_token_ms": ttft_ms,
                    "root_correlation_id": _tree_key(metadata),
                },
                duration_us=round(overlap_ns / NS_PER_US),
            )
        )


def _flow_event(
    name: str,
    phase: str,
    *,
    tid: int,
    ts_us: int,
    flow_id: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "cat": "flow",
        "ph": phase,
        "pid": PID,
        "tid": tid,
        "ts": ts_us,
        "id": flow_id,
        "args": args,
    }


def _append_ttft_summary(
    events: list[dict[str, Any]],
    usable: list[tuple[dict[str, Any], dict[str, Any], str, float, float]],
    placement: dict[int, tuple[int, int]],
    *,
    summary_tid: int,
    origin_ns: float,
) -> None:
    """Add one TTFT overview track and flow links to slot prefill rows."""
    for record, metadata, _, start_ns, end_ns in usable:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        ttft_ms = _number(metric_value(metrics, "time_to_first_token"))
        if ttft_ms is None or ttft_ms < 0:
            continue
        slot, row = placement[id(record)]
        first_token_ns = min(end_ns, start_ns + ttft_ms * NS_PER_MS)
        prefill_us = max(0, round((first_token_ns - start_ns) / NS_PER_US))
        start_us = round((start_ns - origin_ns) / NS_PER_US)
        request_number = metadata.get("session_num", metadata.get("turn_index", "?"))
        request_id = str(metadata.get("x_request_id") or id(record))
        flow_id = f"ttft:{request_id}"
        input_tokens = metric_value(metrics, "input_sequence_length")
        output_tokens = metric_value(metrics, "output_sequence_length")
        cache_ratio, cache_anomaly = _cache_observation(metrics)
        args = {
            "request_num": request_number,
            "x_request_id": metadata.get("x_request_id"),
            "root_correlation_id": _tree_key(metadata),
            "slot_index": slot,
            "slot_row": row,
            "time_to_first_token_ms": ttft_ms,
            "input_tokens": input_tokens,
            "usage_prompt_tokens": metric_value(metrics, "usage_prompt_tokens"),
            "prompt_cache_read_tokens": metric_value(
                metrics, "usage_prompt_cache_read_tokens"
            ),
            "cache_read_ratio_pct": (
                100.0 * cache_ratio if cache_ratio is not None else None
            ),
            "cache_anomaly": cache_anomaly,
            "output_tokens": output_tokens,
            "usage_completion_tokens": metric_value(metrics, "usage_completion_tokens"),
            "target_track": f"slot {slot:02d} row {row}",
        }
        events.append(
            _event(
                f"slot {slot:02d} TTFT #{request_number} | {ttft_ms:.1f} ms",
                ts_us=start_us,
                tid=summary_tid,
                category="ttft",
                args=args,
                duration_us=prefill_us,
                color="terrible" if cache_anomaly else None,
            )
        )
        events.append(
            _flow_event(
                "TTFT → slot prefill",
                "s",
                tid=summary_tid,
                ts_us=start_us,
                flow_id=flow_id,
                args=args,
            )
        )
        events.append(
            _flow_event(
                "TTFT → slot prefill",
                "f",
                tid=10_000 + slot * 100 + row + 1,
                ts_us=start_us,
                flow_id=flow_id,
                args=args,
            )
        )


def _append_decode_summary(
    events: list[dict[str, Any]],
    usable: list[tuple[dict[str, Any], dict[str, Any], str, float, float]],
    placement: dict[int, tuple[int, int]],
    *,
    summary_tid: int,
    origin_ns: float,
) -> None:
    """Add one decode overview track and flow links to slot decode rows."""
    for record, metadata, _, start_ns, end_ns in usable:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        ttft_ms = _number(metric_value(metrics, "time_to_first_token"))
        if ttft_ms is None or ttft_ms < 0:
            continue
        slot, row = placement[id(record)]
        first_token_ns = min(end_ns, start_ns + ttft_ms * NS_PER_MS)
        decode_us = max(0, round((end_ns - first_token_ns) / NS_PER_US))
        request_number = metadata.get("session_num", metadata.get("turn_index", "?"))
        decode_ms = _number(metric_value(metrics, "full_decode_duration"))
        output_tokens = metric_value(metrics, "output_sequence_length")
        args = {
            "request_num": request_number,
            "x_request_id": metadata.get("x_request_id"),
            "root_correlation_id": _tree_key(metadata),
            "slot_index": slot,
            "slot_row": row,
            "decode_ms": decode_ms,
            "decode_interval_ms": decode_us / 1_000.0,
            "output_tokens": output_tokens,
            "target_track": f"slot {slot:02d} row {row}",
        }
        start_us = round((first_token_ns - origin_ns) / NS_PER_US)
        request_id = str(metadata.get("x_request_id") or id(record))
        flow_id = f"decode:{request_id}"
        events.append(
            _event(
                f"slot {slot:02d} decode #{request_number} | {decode_us / 1_000:.1f} ms",
                ts_us=start_us,
                tid=summary_tid,
                category="decode",
                args=args,
                duration_us=decode_us,
            )
        )
        events.append(
            _flow_event(
                "decode → slot decode",
                "s",
                tid=summary_tid,
                ts_us=start_us,
                flow_id=flow_id,
                args=args,
            )
        )
        events.append(
            _flow_event(
                "decode → slot decode",
                "f",
                tid=10_000 + slot * 100 + row + 1,
                ts_us=start_us,
                flow_id=flow_id,
                args=args,
            )
        )


def _append_prefill_decode_summary(
    events: list[dict[str, Any]],
    usable: list[tuple[dict[str, Any], dict[str, Any], str, float, float]],
    placement: dict[int, tuple[int, int]],
    *,
    summary_tid: int,
    origin_ns: float,
) -> None:
    """Add prefill and decode intervals to one combined overview track."""
    for record, metadata, _, start_ns, end_ns in usable:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        ttft_ms = _number(metric_value(metrics, "time_to_first_token"))
        if ttft_ms is None or ttft_ms < 0:
            continue
        slot, row = placement[id(record)]
        first_token_ns = min(end_ns, start_ns + ttft_ms * NS_PER_MS)
        prefill_us = max(0, round((first_token_ns - start_ns) / NS_PER_US))
        decode_us = max(0, round((end_ns - first_token_ns) / NS_PER_US))
        request_number = metadata.get("session_num", metadata.get("turn_index", "?"))
        input_tokens = metric_value(metrics, "input_sequence_length")
        output_tokens = metric_value(metrics, "output_sequence_length")
        cache_ratio, cache_anomaly = _cache_observation(metrics)
        args = {
            "request_num": request_number,
            "x_request_id": metadata.get("x_request_id"),
            "root_correlation_id": _tree_key(metadata),
            "slot_index": slot,
            "slot_row": row,
            "time_to_first_token_ms": ttft_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_cache_read_tokens": metric_value(
                metrics, "usage_prompt_cache_read_tokens"
            ),
            "cache_read_ratio_pct": (
                100.0 * cache_ratio if cache_ratio is not None else None
            ),
            "cache_anomaly": cache_anomaly,
        }
        start_us = round((start_ns - origin_ns) / NS_PER_US)
        first_token_us = round((first_token_ns - origin_ns) / NS_PER_US)
        events.append(
            _event(
                f"slot {slot:02d} prefill #{request_number} | {ttft_ms:.1f} ms",
                ts_us=start_us,
                tid=summary_tid,
                category="prefill",
                args=args,
                duration_us=prefill_us,
                color="terrible" if cache_anomaly else "yellow",
            )
        )
        events.append(
            _event(
                f"slot {slot:02d} decode #{request_number} | {decode_us / 1_000:.1f} ms",
                ts_us=first_token_us,
                tid=summary_tid,
                category="decode",
                args=args,
                duration_us=decode_us,
                color="thread_state_runnable",
            )
        )


def _usable_records(
    records: Iterable[dict[str, Any]],
    *,
    phase: str | None,
    session_key: str,
    session_id: str | None,
) -> list[tuple[dict[str, Any], dict[str, Any], str, float, float]]:
    usable = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if phase is not None and metadata.get("benchmark_phase") != phase:
            continue
        current_session = _session_value(metadata, session_key)
        if session_id is not None and current_session != session_id:
            continue
        start = _number(metadata.get("request_start_ns"))
        end = _number(metadata.get("request_end_ns"))
        if start is None:
            continue
        if end is None:
            end = start
        end = max(start, end)
        usable.append((record, metadata, current_session, start, end))
    return sorted(usable, key=lambda item: (item[3], int(item[1].get("turn_index", 0))))


def build_trace(
    records: Iterable[dict[str, Any]],
    *,
    phase: str | None = "profiling",
    session_key: str = "root_correlation_id",
    session_id: str | None = None,
    concurrency: int | None = None,
    inspect_slot: int | None = None,
    inspect_all_slots: bool = False,
) -> dict[str, Any]:
    """Build a Chrome Trace Event document from AIPerf records.

    When ``concurrency`` is provided, root trees are assigned to a fixed
    number of reusable slots, matching AIPerf's agentic replay swim-lane
    layout. Child x-correlation sessions are packed into rows below the root
    row within each slot.
    """
    if (inspect_slot is not None or inspect_all_slots) and concurrency is None:
        raise ValueError("slot analysis requires concurrency")
    if inspect_slot is not None and inspect_all_slots:
        raise ValueError("choose inspect_slot or inspect_all_slots, not both")
    if (
        inspect_slot is not None
        and concurrency is not None
        and not 0 <= inspect_slot < concurrency
    ):
        raise ValueError("inspect_slot must be within the configured concurrency slots")
    usable = _usable_records(
        records,
        phase=phase,
        session_key=session_key,
        session_id=session_id,
    )
    if not usable:
        filters = f"phase={phase!r}"
        if session_id is not None:
            filters += f", session={session_id!r}"
        raise ValueError(f"no usable AIPerf records found ({filters})")

    origin_ns = min(item[3] for item in usable)
    events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": PID,
            "tid": 0,
            "args": {"name": "AIPerf request timeline"},
        }
    ]

    grouped: dict[
        str, list[tuple[dict[str, Any], dict[str, Any], str, float, float]]
    ] = defaultdict(list)
    for item in usable:
        grouped[item[2]].append(item)

    slot_layout = None
    tree_layout: dict[str, tuple[int, float, float]] = {}
    if concurrency is not None:
        placement, tree_layout, _used_slots = _build_slot_layout(usable, concurrency)
        grouped_by_track: dict[
            tuple[int, int],
            list[tuple[dict[str, Any], dict[str, Any], str, float, float]],
        ] = defaultdict(list)
        rows_by_slot: defaultdict[int, int] = defaultdict(lambda: 1)
        for item in usable:
            slot, row = placement[id(item[0])]
            grouped_by_track[(slot, row)].append(item)
            rows_by_slot[slot] = max(rows_by_slot[slot], row + 1)

        summary_tid = 9_000
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": PID,
                "tid": summary_tid,
                "args": {"name": "all slots • TTFT"},
            }
        )
        events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": PID,
                "tid": summary_tid,
                "args": {"sort_index": -1},
            }
        )
        decode_summary_tid = 9_001
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": PID,
                "tid": decode_summary_tid,
                "args": {"name": "all slots • decode"},
            }
        )
        events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": PID,
                "tid": decode_summary_tid,
                "args": {"sort_index": -2},
            }
        )
        combined_summary_tid = 9_002
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": PID,
                "tid": combined_summary_tid,
                "args": {"name": "all slots • prefill+decode"},
            }
        )
        events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": PID,
                "tid": combined_summary_tid,
                "args": {"sort_index": -3},
            }
        )

        if inspect_all_slots:
            analysis_slots = set(range(concurrency))
        elif inspect_slot is not None:
            analysis_slots = {inspect_slot}
        else:
            analysis_slots = set()
        special_rows_by_slot: dict[int, tuple[int | None, int]] = {}
        for slot in analysis_slots:
            special_row_start = rows_by_slot[slot]
            if inspect_all_slots:
                rows_by_slot[slot] += 1
                special_rows_by_slot[slot] = (None, special_row_start)
            else:
                rows_by_slot[slot] += 2
                special_rows_by_slot[slot] = (
                    special_row_start,
                    special_row_start + 1,
                )

        for slot in range(concurrency):
            overview_tid = 10_000 + slot * 100
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": PID,
                    "tid": overview_tid,
                    "args": {"name": f"slot {slot:02d} overview", "slot_index": slot},
                }
            )
            events.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": PID,
                    "tid": overview_tid,
                    "args": {"sort_index": slot * 100},
                }
            )
            for row in range(rows_by_slot[slot]):
                tid = overview_tid + row + 1
                special_rows = special_rows_by_slot.get(slot)
                if (
                    special_rows
                    and special_rows[0] is not None
                    and row == special_rows[0]
                ):
                    track_name = f"slot {slot:02d} wait/TTFB"
                elif special_rows and row == special_rows[1]:
                    track_name = f"slot {slot:02d} wait∩prefill"
                else:
                    track_name = f"slot {slot:02d} row {row}"
                events.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": PID,
                        "tid": tid,
                        "args": {
                            "name": track_name,
                            "slot_index": slot,
                            "slot_row": row,
                        },
                    }
                )
                events.append(
                    {
                        "name": "thread_sort_index",
                        "ph": "M",
                        "pid": PID,
                        "tid": tid,
                        "args": {"sort_index": slot * 100 + row + 1},
                    }
                )

        for slot, (wait_row, overlap_row) in special_rows_by_slot.items():
            _append_wait_prefill_analysis(
                events,
                usable,
                placement,
                inspect_slot=slot,
                wait_tid=(
                    10_000 + slot * 100 + wait_row + 1 if wait_row is not None else None
                ),
                overlap_tid=10_000 + slot * 100 + overlap_row + 1,
                origin_ns=origin_ns,
            )
        _append_ttft_summary(
            events,
            usable,
            placement,
            summary_tid=summary_tid,
            origin_ns=origin_ns,
        )
        _append_decode_summary(
            events,
            usable,
            placement,
            summary_tid=decode_summary_tid,
            origin_ns=origin_ns,
        )
        _append_prefill_decode_summary(
            events,
            usable,
            placement,
            summary_tid=combined_summary_tid,
            origin_ns=origin_ns,
        )

        for tree, (slot, start_ns, end_ns) in tree_layout.items():
            events.append(
                _event(
                    f"tree {_short_id(tree)}",
                    ts_us=round((start_ns - origin_ns) / NS_PER_US),
                    tid=10_000 + slot * 100,
                    category="session-tree",
                    args={
                        "root_correlation_id": tree,
                        "slot_index": slot,
                        "tree_start_ns": int(start_ns),
                        "tree_end_ns": int(end_ns),
                    },
                    duration_us=round((end_ns - start_ns) / NS_PER_US),
                )
            )

        track_entries = [
            (
                f"slot {slot:02d} row {row}",
                10_000 + slot * 100 + row + 1,
                grouped_by_track[(slot, row)],
                slot,
                row,
            )
            for slot in range(concurrency)
            for row in range(rows_by_slot[slot])
            if grouped_by_track[(slot, row)]
        ]
        sessions = sorted(tree_layout)
        session_count = len(tree_layout)
        slot_count = concurrency
        slot_layout = placement
    else:
        sessions = sorted({item[2] for item in usable})
        tids = {value: index + 1 for index, value in enumerate(sessions)}
        track_entries = [
            (session, tids[session], grouped[session], None, None)
            for session in sessions
        ]
        session_count = len(sessions)
        slot_count = None

    if concurrency is None:
        for session_index, session in enumerate(sessions, start=1):
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": PID,
                    "tid": tids[session],
                    "args": {
                        "name": f"{session_key}={_short_id(session)}",
                        "session_id": session,
                        "session_index": session_index,
                    },
                }
            )

    for session, tid, track_items, slot_index, slot_row in track_entries:
        previous_end_ns: float | None = None
        for record, metadata, _, start_ns, end_ns in track_items:
            metrics = record.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}

            relative_start_us = round((start_ns - origin_ns) / NS_PER_US)
            request_duration_us = round((end_ns - start_ns) / NS_PER_US)
            ttft_ms = _number(metric_value(metrics, "time_to_first_token"))
            decode_ms = _number(metric_value(metrics, "full_decode_duration"))
            sending_ms = _number(metric_value(metrics, "http_req_sending"))
            waiting_ms = _number(metric_value(metrics, "http_req_waiting"))
            input_tokens = metric_value(metrics, "input_sequence_length")
            usage_prompt_tokens = metric_value(metrics, "usage_prompt_tokens")
            prompt_cache_read_tokens = metric_value(
                metrics, "usage_prompt_cache_read_tokens"
            )
            output_tokens = metric_value(metrics, "output_sequence_length")
            usage_completion_tokens = metric_value(metrics, "usage_completion_tokens")
            cache_ratio, cache_anomaly = _cache_observation(metrics)

            request_number = metadata.get(
                "session_num", metadata.get("turn_index", "?")
            )
            request_session = (
                _request_session_key(metadata, _tree_key(metadata))
                if concurrency is not None
                else session
            )
            request_args = {
                "session_id": request_session,
                "root_correlation_id": _tree_key(metadata),
                "session_num": request_number,
                "turn_index": metadata.get("turn_index"),
                "x_request_id": metadata.get("x_request_id"),
                "conversation_id": metadata.get("conversation_id"),
                "source_kind": metadata.get("source_kind"),
                "parent_correlation_id": metadata.get("parent_correlation_id"),
                "phase": metadata.get("benchmark_phase"),
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "time_to_first_token_ms": ttft_ms,
                "full_decode_duration_ms": decode_ms,
                "input_tokens": input_tokens,
                "usage_prompt_tokens": usage_prompt_tokens,
                "prompt_cache_read_tokens": prompt_cache_read_tokens,
                "cache_read_ratio_pct": (
                    100.0 * cache_ratio if cache_ratio is not None else None
                ),
                "cache_anomaly": cache_anomaly,
                "output_tokens": output_tokens,
                "usage_completion_tokens": usage_completion_tokens,
                "error": record.get("error"),
            }
            if slot_index is not None:
                request_args["slot_index"] = slot_index
                request_args["slot_row"] = slot_row
            input_label = (
                f" | input={int(input_tokens)} tok" if input_tokens is not None else ""
            )
            output_label = (
                f" | output={int(output_tokens)} tok"
                if output_tokens is not None
                else ""
            )
            events.append(
                _event(
                    f"request #{request_number}{input_label}{output_label}",
                    ts_us=relative_start_us,
                    tid=tid,
                    category="request",
                    args=request_args,
                    duration_us=request_duration_us,
                    color="terrible" if cache_anomaly else None,
                )
            )

            if previous_end_ns is not None and start_ns > previous_end_ns:
                events.append(
                    _event(
                        "session idle",
                        ts_us=round((previous_end_ns - origin_ns) / NS_PER_US),
                        tid=tid,
                        category="session",
                        args={"idle_ms": (start_ns - previous_end_ns) / NS_PER_MS},
                        duration_us=round((start_ns - previous_end_ns) / NS_PER_US),
                    )
                )

            if sending_ms is not None and sending_ms > 0:
                events.append(
                    _event(
                        f"HTTP send #{request_number}",
                        ts_us=relative_start_us,
                        tid=tid,
                        category="http",
                        args={"duration_ms": sending_ms},
                        duration_us=round(sending_ms * 1_000),
                    )
                )

            if ttft_ms is not None and ttft_ms >= 0:
                first_token_ns = min(end_ns, start_ns + ttft_ms * NS_PER_MS)
                first_token_us = round((first_token_ns - origin_ns) / NS_PER_US)
                prefill_us = max(0, round((first_token_ns - start_ns) / NS_PER_US))
                decode_us = max(0, request_duration_us - prefill_us)
                events.append(
                    _event(
                        f"prefill #{request_number}",
                        ts_us=relative_start_us,
                        tid=tid,
                        category="inference",
                        args={"time_to_first_token_ms": ttft_ms},
                        duration_us=prefill_us,
                        color="terrible" if cache_anomaly else None,
                    )
                )
                events.append(
                    _event(
                        f"decode #{request_number}",
                        ts_us=first_token_us,
                        tid=tid,
                        category="inference",
                        args={"full_decode_duration_ms": decode_ms},
                        duration_us=decode_us,
                    )
                )
                events.append(
                    _event(
                        f"first token #{request_number}",
                        ts_us=first_token_us,
                        tid=tid,
                        category="milestone",
                        args={"time_to_first_token_ms": ttft_ms},
                    )
                )

            if waiting_ms is not None and waiting_ms > 0:
                wait_start_ns = start_ns + (sending_ms or 0) * NS_PER_MS
                events.append(
                    _event(
                        f"HTTP wait/TTFB #{request_number}",
                        ts_us=round((wait_start_ns - origin_ns) / NS_PER_US),
                        tid=tid,
                        category="http",
                        args={"duration_ms": waiting_ms},
                        duration_us=round(waiting_ms * 1_000),
                    )
                )

            ack_ns = _number(metadata.get("request_ack_ns"))
            if ack_ns is not None:
                events.append(
                    _event(
                        f"server ack #{request_number}",
                        ts_us=round((ack_ns - origin_ns) / NS_PER_US),
                        tid=tid,
                        category="milestone",
                        args={"ack_ns": int(ack_ns)},
                    )
                )
            events.append(
                _event(
                    f"response complete #{request_number}",
                    ts_us=round((end_ns - origin_ns) / NS_PER_US),
                    tid=tid,
                    category="milestone",
                    args={"end_ns": int(end_ns)},
                )
            )
            previous_end_ns = max(previous_end_ns or end_ns, end_ns)

    return {
        "displayTimeUnit": "ms",
        "traceEvents": events,
        "metadata": {
            "format": "AIPerf profile_export.jsonl → Chrome Trace Event",
            "phase": phase,
            "session_key": session_key,
            "session_id": session_id,
            "origin_ns": int(origin_ns),
            "record_count": len(usable),
            "session_count": session_count,
            "slot_count": slot_count,
            "tree_count": len(tree_layout) if slot_layout is not None else None,
            "inspect_slot": inspect_slot,
        },
    }


def list_sessions(
    records: Iterable[dict[str, Any]],
    *,
    phase: str | None,
    session_key: str,
) -> list[tuple[str, int]]:
    counts: defaultdict[str, int] = defaultdict(int)
    for record in records:
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if phase is not None and metadata.get("benchmark_phase") != phase:
            continue
        counts[_session_value(metadata, session_key)] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="AIPerf profile_export.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output trace JSON (default: <input>.trace.json)",
    )
    parser.add_argument(
        "--phase",
        default="profiling",
        help="benchmark phase to include; use 'all' to include every phase",
    )
    parser.add_argument(
        "--session-key",
        choices=("root_correlation_id", "x_correlation_id", "conversation_id"),
        default="root_correlation_id",
        help="metadata field used as a trace lane/session",
    )
    parser.add_argument(
        "--session-id",
        help="only export one session ID; use --list-sessions to discover IDs",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="use fixed reusable root-tree slots (for example, 48)",
    )
    parser.add_argument(
        "--inspect-slot",
        type=int,
        help="add wait/TTFB and wait∩prefill tracks after this slot",
    )
    parser.add_argument(
        "--inspect-all-slots",
        action="store_true",
        help="add wait/TTFB and wait∩prefill tracks after every slot",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="list session IDs and record counts, then exit",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    phase = None if args.phase == "all" else args.phase
    if args.list_sessions:
        for session, count in list_sessions(
            records, phase=phase, session_key=args.session_key
        ):
            print(f"{count:6d}  {session}")
        return

    output = args.output or args.input.with_name(f"{args.input.stem}.trace.json")
    trace = build_trace(
        records,
        phase=phase,
        session_key=args.session_key,
        session_id=args.session_id,
        concurrency=args.concurrency,
        inspect_slot=args.inspect_slot,
        inspect_all_slots=args.inspect_all_slots,
    )
    output.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    print(
        f"wrote {output} "
        f"({trace['metadata']['record_count']} requests, "
        f"{trace['metadata']['session_count']} sessions)"
    )


if __name__ == "__main__":
    main()
