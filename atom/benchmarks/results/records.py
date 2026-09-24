# SPDX-License-Identifier: MIT
"""Normalize AIPerf and capture random-client requests without prompt text."""

import asyncio
import atexit
import math
import time

from .io import JsonlWriter

LATENCY_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}
for _short, _long in (
    ("ns", "nanosecond"),
    ("us", "microsecond"),
    ("ms", "millisecond"),
    ("s", "second"),
):
    LATENCY_UNITS[_long] = LATENCY_UNITS[_long + "s"] = LATENCY_UNITS[_short]
LATENCY_UNITS["sec"] = 1.0


def number(value):
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def seconds(value, default_unit="ms"):
    unit = str(
        value.get("unit", default_unit) if isinstance(value, dict) else default_unit
    ).lower()
    n = number(value)
    if n is None:
        return None
    if unit not in LATENCY_UNITS:
        raise ValueError(f"Unsupported latency unit: {unit!r}")
    return n * LATENCY_UNITS[unit]


def timestamp(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) or (isinstance(value, str) and value.isdecimal()):
        return str(int(value))
    # A float nanosecond timestamp has already lost precision. Do not call it exact.
    raise ValueError("Nanosecond timestamps must be integers or decimal strings")


def from_aiperf(row, index):
    m, meta = row.get("metrics", {}), row.get("metadata", {})
    error = row.get("error")
    result = {
        "request_id": str(
            meta.get("x_request_id") or meta.get("request_id") or f"aiperf-{index}"
        ),
        "correlation_id": meta.get("x_correlation_id") or meta.get("correlation_id"),
        "session_id": meta.get("x_correlation_id")
        or meta.get("session_id")
        or (str(meta["session_num"]) if meta.get("session_num") is not None else None),
        "conversation_id": meta.get("conversation_id"),
        "root_session_id": meta.get("root_correlation_id"),
        "parent_id": meta.get("parent_correlation_id") or meta.get("parent_session_id"),
        "turn_index": meta.get("turn_index"),
        "phase": meta.get("benchmark_phase", "unknown"),
        "status": (
            "cancelled"
            if meta.get("was_cancelled")
            else "error" if error else "success"
        ),
        "error": error,
        "start_ns": timestamp(meta.get("request_start_ns")),
        "end_ns": timestamp(meta.get("request_end_ns")),
        "input_tokens": number(m.get("input_sequence_length")),
        "output_tokens": number(m.get("output_sequence_length")),
        "expected_output_tokens": number(m.get("expected_output_sequence_length")),
        "ttft_s": seconds(m.get("time_to_first_token")),
        "e2el_s": seconds(m.get("request_latency")),
        "native_itl_s": seconds(m.get("inter_token_latency")),
        "full_response_itl_s": seconds(m.get("full_response_inter_token_latency")),
        "full_decode_duration_s": seconds(m.get("full_decode_duration")),
        "http_duration_s": seconds(m.get("http_req_duration")),
        "cached_input_tokens": number(m.get("usage_prompt_cache_read_tokens")),
        "usage_input_tokens": number(m.get("usage_prompt_tokens")),
        "token_count_source": "aiperf_export",
        "time_source": "aiperf_request_timestamps",
        "source": {
            "format": "aiperf",
            "record_index": index,
            **{
                k: meta[k]
                for k in (
                    "source_trace_id",
                    "source_outer_idx",
                    "source_inner_idx",
                    "source_kind",
                    "worker_id",
                    "phase_index",
                    "phase_name",
                )
                if k in meta
            },
        },
    }
    return {key: value for key, value in result.items() if value is not None}


def full_response_itl(record):
    explicit = number(record.get("full_response_itl_s"))
    if explicit is not None and explicit > 0:
        return explicit
    tokens = number(record.get("output_tokens"))
    if tokens is None or tokens < 2:
        return None
    duration = number(record.get("full_decode_duration_s"))
    if duration is None:
        ttft = number(record.get("ttft_s"))
        if ttft is None:
            return None
        if record.get("start_ns") and record.get("end_ns"):
            duration = (int(record["end_ns"]) - int(record["start_ns"])) / 1e9 - ttft
        else:
            http = number(record.get("http_duration_s"))
            if http is not None:
                duration = http - ttft
    return duration / (tokens - 1) if duration is not None and duration > 0 else None


class RequestRecorder:
    def __init__(self, path, tokenizer=None):
        self.writer = JsonlWriter(path)
        self.tokenizer = tokenizer
        self.sequence = 0
        atexit.register(self.close)

    async def call(self, fn, request_func_input, pbar=None, phase="profiling"):
        request = request_func_input
        request_id = f"{phase}-{self.sequence}"
        self.sequence += 1
        start = time.time_ns()
        row = {
            "request_id": request_id,
            "phase": phase,
            "status": "error",
            "start_ns": str(start),
            "input_tokens": request.prompt_len,
            "expected_output_tokens": request.output_len,
            "source": {"format": "atom-random"},
        }
        try:
            output = await fn(request_func_input=request, pbar=pbar)
            tokens = output.output_tokens
            token_source = "server_usage" if tokens is not None else "unavailable"
            if tokens is None and output.success and self.tokenizer is not None:
                tokens = len(
                    self.tokenizer(
                        output.generated_text, add_special_tokens=False
                    ).input_ids
                )
                output.output_tokens = tokens
                token_source = "client_tokenizer"
            ttft = getattr(output, "content_ttft_s", None)
            ttft = output.ttft if ttft is None else ttft
            full_duration = getattr(output, "full_response_duration_s", None)
            row.update(
                {
                    "status": "success" if output.success else "error",
                    "error": output.error or None,
                    "start_ns": str(getattr(output, "request_start_ns", 0) or start),
                    "end_ns": str(
                        getattr(output, "request_end_ns", 0) or time.time_ns()
                    ),
                    "ttft_s": ttft,
                    "harness_ttft_s": output.ttft,
                    "first_content_ns": (
                        str(output.first_content_ns)
                        if getattr(output, "first_content_ns", None)
                        else None
                    ),
                    "e2el_s": full_duration or output.latency,
                    "full_decode_duration_s": (
                        full_duration - ttft if full_duration is not None else None
                    ),
                    "harness_e2el_s": output.latency,
                    "output_tokens": tokens,
                    "token_count_source": token_source,
                    "time_source": "unix_ns_with_monotonic_duration",
                    # Preserve chunk measurements separately from per-request full-response ITL.
                    "chunk_itls_s": output.itl,
                }
            )
            return output
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            row["error"] = "client cancelled"
            raise
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row.setdefault("end_ns", str(time.time_ns()))
            self.writer.write(row)

    def close(self):
        atexit.unregister(self.close)
        self.writer.close()
