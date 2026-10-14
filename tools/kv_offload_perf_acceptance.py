"""One-arm, streaming OpenAI-compatible performance acceptance client.

CPU-only manifest generation/tests; `run` contacts an ALREADY running server.
Never starts/stops a server or changes its configuration. Requires aiohttp only
for `run`. Generate ONE manifest per case/round/concurrency and reuse it for ON
and OFF. A new manifest receives a random document namespace by default.

TTFT includes time to the first nonempty content/reasoning delta. TPOT
is (last_content - first_content)/(usage.completion_tokens - 1), NOT a percentile
of SSE inter-chunk gaps. An SSE content event can contain multiple tokens; the
client does not infer token counts from text or chunks. E2E ends after [DONE].
"""

import argparse
import asyncio
import codecs
import hashlib
import json
import logging
import math
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)
VALIDATOR_VERSION = 2

DOCS = 12
REPETITIONS = 500
REQUESTS = 240
OUTPUT_TOKENS = 64
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_EVENTS = 8192
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
WINDOW = "first measured request dispatched through last measured request completed; serial prime and metrics scrapes excluded"
TIMING = {
    "clock": "time.perf_counter (monotonic)",
    "content_fields": ["content", "reasoning_content", "reasoning"],
    "ttft": "first nonempty content event - request dispatch",
    "tpot": "(last nonempty content event - first nonempty content event)/(completion_tokens - 1)",
    "e2e": "response closed after [DONE] - request dispatch",
    "percentile": "linear interpolation at (n - 1) * q (type 7), across requests",
    "warning": "SSE events may contain multiple tokens; TPOT is a client-observed per-request average, NOT per-token ITL; one content event yields TPOT=0",
}
METRICS = {
    "external_queries": "vllm:external_prefix_cache_queries_total",
    "external_hits": "vllm:external_prefix_cache_hits_total",
    "local_queries": "vllm:prefix_cache_queries_total",
    "local_hits": "vllm:prefix_cache_hits_total",
}
SOURCE_METRIC = "vllm:prompt_tokens_by_source_total"
SOURCE_KEYS = {
    "source_external": "external_kv_transfer",
    "source_local": "local_cache_hit",
    "source_compute": "local_compute",
}
ALL_METRICS = {
    **METRICS,
    **{
        key: f'{SOURCE_METRIC}{{source="{source}"}}'
        for key, source in SOURCE_KEYS.items()
    },
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
SAMPLE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{(?:[^\"{}]|\"(?:\\.|[^\"\\])*\")*\})?\s+(\S+)(?:\s+\S+)?\s*$"
)
LABEL = re.compile(
    r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:\\[\\"n]|[^"\\])*)"\s*(?:,|$)'
)


class AcceptanceError(ValueError):
    pass


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def payload(manifest, request):
    return {
        "model": manifest["model"],
        "messages": [
            {
                "role": "user",
                "content": manifest["documents"][request["document_id"]]
                + "\nQuery "
                + request["tail"]
                + ": reply.",
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1 if request["phase"] == "prime" else OUTPUT_TOKENS,
        "ignore_eos": True,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def build_manifest(case, round_number, model, concurrency=24, namespace=None):
    if not SAFE_NAME.fullmatch(case) or case in (".", ".."):
        raise AcceptanceError("case must be a safe 1-80 character filename component")
    if type(round_number) is not int or round_number < 1:
        raise AcceptanceError("round must be a positive integer")
    if type(concurrency) is not int or not 1 <= concurrency <= REQUESTS:
        raise AcceptanceError("concurrency must be between 1 and 240")
    if not isinstance(model, str) or not model.strip():
        raise AcceptanceError("model is required")
    nonce = namespace if namespace is not None else uuid.uuid4().hex
    if not isinstance(nonce, str) or not SAFE_NAME.fullmatch(nonce):
        raise AcceptanceError("namespace must be a safe 1-80 character component")
    document_namespace = f"{nonce}/{case}/round-{round_number:04d}/conc-{concurrency}"
    result = {
        "schema_version": 1,
        "case": case,
        "round": round_number,
        "namespace_nonce": nonce,
        "document_namespace": document_namespace,
        "model": model,
        "concurrency": concurrency,
        "recipe": {
            "documents": DOCS,
            "word_repetitions": REPETITIONS,
            "requests": REQUESTS,
            "output_tokens": OUTPUT_TOKENS,
            "prime_output_tokens": 1,
            "temperature": 0.0,
            "seed": 0,
            "ignore_eos": True,
        },
        "window_definition": WINDOW,
        "documents": [
            f"Dossier {document_namespace} {i:03d}.\n" + WORDS * REPETITIONS
            for i in range(DOCS)
        ],
        "prime": [],
        "measured": [],
    }
    for phase, count, key in (
        ("prime", DOCS, "prime"),
        ("measured", REQUESTS, "measured"),
    ):
        for i in range(count):
            request = {
                "request_id": f"{phase}-{i:06d}",
                "phase": phase,
                "ordinal": i,
                "document_id": i % DOCS,
                "tail": f"p{i:03d}" if phase == "prime" else f"measured-{i:06d}",
            }
            request["payload_sha256"] = sha256(canonical(payload(result, request)))
            result[key].append(request)
    return result


def load_manifest(path):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    expected = build_manifest(
        manifest["case"],
        manifest["round"],
        manifest["model"],
        manifest["concurrency"],
        manifest["namespace_nonce"],
    )
    if manifest != expected:
        raise AcceptanceError(
            "manifest differs from the fixed recipe or payload hashes; do not edit/reuse a manifest for a different round"
        )
    return manifest, raw, sha256(raw)


class SSEDecoder:
    """Incremental UTF-8 + SSE framing, including CRLF split across TCP reads."""

    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
        self.buffer = ""
        self.data = []
        self.event_type = "message"
        self.event_size = 0

    def feed(self, raw, eof=False):
        self.buffer += self.decoder.decode(raw, final=eof)
        events = []
        while True:
            match = re.search(r"[\r\n]", self.buffer)
            if match is None or (
                not eof and match.group() == "\r" and match.end() == len(self.buffer)
            ):
                break
            end = match.end()
            if self.buffer[match.start() : end + 1] == "\r\n":
                end += 1
            line, self.buffer = self.buffer[: match.start()], self.buffer[end:]
            self.event_size += len(line)
            if self.event_size > 1024 * 1024:
                raise AcceptanceError("SSE event exceeds 1 MiB")
            if not line:
                if self.data:
                    events.append((self.event_type, "\n".join(self.data)))
                self.data, self.event_type, self.event_size = [], "message", 0
            elif not line.startswith(":"):
                field, sep, value = line.partition(":")
                value = value.removeprefix(" ")
                if field == "data":
                    self.data.append(value if sep else "")
                elif field == "event":
                    self.event_type = value
        if len(self.buffer) + self.event_size > 1024 * 1024:
            raise AcceptanceError("SSE event exceeds 1 MiB")
        return events


def validate_usage(usage, expected_tokens):
    if not isinstance(usage, dict):
        raise AcceptanceError("missing complete usage object")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise AcceptanceError(f"usage.{key} must be a nonnegative integer")
    if usage["prompt_tokens"] <= 0:
        raise AcceptanceError("usage.prompt_tokens must be positive")
    if usage["completion_tokens"] != expected_tokens:
        raise AcceptanceError(
            f"completion_tokens={usage['completion_tokens']}, expected {expected_tokens}"
        )
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise AcceptanceError("usage.total_tokens does not equal prompt + completion")


class StreamState:
    def __init__(self, record, expected_tokens):
        self.record = record
        self.expected_tokens = expected_tokens
        self.done = False
        self.first_content = None
        self.last_content = None
        self.finish_reason = None

    def accept(self, event_type, data, elapsed):
        if self.done:
            raise AcceptanceError("SSE event after [DONE]")
        if len(self.record["events"]) >= MAX_RESPONSE_EVENTS:
            raise AcceptanceError("response exceeds SSE event limit")
        event = {"at_s": elapsed, "event_type": event_type, "data": data}
        self.record["events"].append(event)
        if event_type == "error":
            raise AcceptanceError(f"server SSE error: {data[:400]}")
        if data.strip() == "[DONE]":
            self.done = True
            return
        item = json.loads(data)
        if not isinstance(item, dict):
            raise AcceptanceError("SSE JSON payload must be an object")
        if item.get("error") is not None:
            raise AcceptanceError(f"server streaming error: {item['error']}")
        if item.get("model"):
            self.record["response_models"] = sorted(
                set(self.record.get("response_models", [])) | {item["model"]}
            )
        usage = item.get("usage")
        if usage is not None:
            if self.record.get("usage") is not None and self.record["usage"] != usage:
                raise AcceptanceError("conflicting usage objects in one response")
            self.record["usage"] = usage
        choices = item.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise AcceptanceError("expected at most one streaming choice")
        if not choices:
            return  # usage-only tail chunks are normal
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise AcceptanceError("invalid choice/index")
        already_finished = self.finish_reason is not None
        reason = choice.get("finish_reason")
        if reason is not None:
            if reason not in ("length", "stop") or self.finish_reason is not None:
                raise AcceptanceError(f"unexpected or repeated finish_reason: {reason}")
            self.finish_reason = reason
            self.record["finish_reason"] = reason
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise AcceptanceError("choice.delta must be an object")
        content = {}
        for field in TIMING["content_fields"]:
            value = delta.get(field)
            if value is not None and not isinstance(value, str):
                raise AcceptanceError(f"delta.{field} must be a string or null")
            if value:
                content[field] = value
        if content:
            if already_finished:
                raise AcceptanceError("content after terminal finish_reason")
            if self.first_content is None:
                self.first_content = elapsed
            self.last_content = elapsed
            event["content_characters"] = {
                key: len(value) for key, value in content.items()
            }
            event["token_count"] = None  # chunks are not tokens
            self.record["content_event_count"] += 1

    def finish(self):
        if not self.done:
            raise AcceptanceError("SSE stream ended without [DONE]")
        validate_usage(self.record.get("usage"), self.expected_tokens)
        if self.finish_reason is None:
            raise AcceptanceError("missing terminal finish_reason")
        if self.first_content is None and self.expected_tokens > 1:
            raise AcceptanceError("no nonempty content/reasoning event")
        return {
            "ttft_s": self.first_content,
            "tpot_s": (
                (self.last_content - self.first_content) / (self.expected_tokens - 1)
                if self.expected_tokens > 1
                else None
            ),
            "first_content_s": self.first_content,
            "last_content_s": self.last_content,
        }


async def stream_request(
    session, url, manifest, request, sink, clock=time.perf_counter
):
    body = canonical(payload(manifest, request))
    if sha256(body) != request["payload_sha256"]:
        raise AcceptanceError("request payload hash mismatch before dispatch")
    started = clock()
    record = {
        **request,
        "started_monotonic_s": started,
        "started_utc": utc_now(),
        "status": "failed",
        "usage": None,
        "events": [],
        "content_event_count": 0,
    }
    state = StreamState(record, 1 if request["phase"] == "prime" else OUTPUT_TOKENS)
    try:
        async with session.post(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            allow_redirects=False,
        ) as response:
            record["http_status"] = response.status
            if response.status != 200:
                error_body = (await response.content.read(4096)).decode(
                    "utf-8", errors="replace"
                )
                raise AcceptanceError(f"HTTP {response.status}: {error_body}")
            if (
                response.headers.get("Content-Type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
                != "text/event-stream"
            ):
                raise AcceptanceError("response Content-Type is not text/event-stream")
            decoder = SSEDecoder()
            received_bytes = 0
            async for raw in response.content.iter_any():
                received = clock() - started
                received_bytes += len(raw)
                record["received_bytes"] = received_bytes
                if received_bytes > MAX_RESPONSE_BYTES:
                    raise AcceptanceError("response exceeds byte limit")
                for event_type, data in decoder.feed(raw):
                    state.accept(event_type, data, received)
            for event_type, data in decoder.feed(b"", eof=True):
                state.accept(event_type, data, clock() - started)
            if decoder.buffer or decoder.data:
                raise AcceptanceError("incomplete SSE frame at response EOF")
            record.update(state.finish())
        record["status"] = "passed"
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        record["error"] = "request cancelled"
        raise
    except Exception as exc:
        logger.exception("Streaming request failed")
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["completed_monotonic_s"] = clock()
        record["e2e_s"] = record["completed_monotonic_s"] - started
        sink(record)
    return record


def parse_labels(raw):
    if not raw or raw == "{}":
        return ()
    text, pos, labels = raw[1:-1], 0, {}
    while pos < len(text):
        match = LABEL.match(text, pos)
        if match is None:
            raise AcceptanceError(f"invalid Prometheus labels: {raw}")
        key, value = match.group(1, 2)
        if key in labels:
            raise AcceptanceError(f"duplicate label {key}")
        labels[key] = re.sub(
            r'\\([\\"n])', lambda m: "\n" if m.group(1) == "n" else m.group(1), value
        )
        pos = match.end()
    return tuple(sorted(labels.items()))


def parse_metrics(text):
    """Exact metric names; sum ALL label series, never last-line-wins."""
    result = {key: {} for key in ALL_METRICS}
    reverse = {name: key for key, name in METRICS.items()}
    source_reverse = {source: key for key, source in SOURCE_KEYS.items()}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\s{]", line, maxsplit=1)[0]
        if name not in reverse and name != SOURCE_METRIC:
            continue
        match = SAMPLE.fullmatch(line)
        if match is None:
            raise AcceptanceError(f"malformed target metric: {line}")
        labels = parse_labels(match.group(2))
        value = float(match.group(3))
        if not math.isfinite(value) or value < 0:
            raise AcceptanceError(f"nonfinite or negative metric {name}")
        if name == SOURCE_METRIC:
            source = dict(labels).get("source")
            if source not in source_reverse:
                raise AcceptanceError(f"unknown prompt token source: {source}")
            key = source_reverse[source]
        else:
            key = reverse[name]
        series = result[key]
        if labels in series:
            raise AcceptanceError(f"duplicate metric series: {name}{labels}")
        series[labels] = value
    return result


def metric_deltas(before_text, after_text, arm):
    before, after = parse_metrics(before_text), parse_metrics(after_text)
    external_absent = all(
        not snap[key]
        for snap in (before, after)
        for key in ("external_hits", "external_queries")
    )
    if external_absent and arm == "on":
        raise AcceptanceError(
            "ON requires both external metric families in before and after"
        )
    deltas, totals = {}, {}
    for key, metric_name in ALL_METRICS.items():
        if external_absent and key.startswith("external_"):
            deltas[key], totals[key] = None, {
                "before": None,
                "after": None,
                "series": None,
            }
            continue
        if not before[key] or not after[key]:
            raise AcceptanceError(f"missing metric {metric_name} in before or after")
        if before[key].keys() != after[key].keys():
            raise AcceptanceError(f"metric label set changed: {metric_name}")
        if any(after[key][labels] < value for labels, value in before[key].items()):
            raise AcceptanceError(f"counter reset: {metric_name}")
        left, right = sum(before[key].values()), sum(after[key].values())
        deltas[key] = sum(
            after[key][labels] - value for labels, value in before[key].items()
        )
        totals[key] = {"before": left, "after": right, "series": len(before[key])}
    return deltas, totals, external_absent


def assess_supply(before_text, after_text, arm, prompt_tokens):
    deltas, totals, absent = metric_deltas(before_text, after_text, arm)
    return assess_delta_supply(deltas, totals, absent, arm, prompt_tokens)


def assess_delta_supply(deltas, totals, absent, arm, prompt_tokens):
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise AcceptanceError(
            "actual prompt token denominator must be a positive integer"
        )
    if arm not in ("on", "off") or type(absent) is not bool:
        raise AcceptanceError("invalid arm or external metric availability")
    if absent and arm == "on":
        raise AcceptanceError("ON requires external metrics")
    for key in ALL_METRICS:
        value = deltas.get(key)
        if absent and key.startswith("external_"):
            if value is not None:
                raise AcceptanceError("absent external metrics must not have a delta")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise AcceptanceError(
                f"missing, nonfinite or negative counter delta: {key}"
            )
    errors = []
    for prefix in ("external", "local"):
        hits, queries = deltas[prefix + "_hits"], deltas[prefix + "_queries"]
        if hits is not None and hits > queries:
            errors.append(f"{prefix} hits exceed queries")
    local = deltas["local_hits"] / prompt_tokens
    tier = None if absent else deltas["external_hits"] / prompt_tokens
    recompute = None if tier is None else 1 - local - tier
    for name, value in (("local", local), ("tier", tier), ("recompute", recompute)):
        if value is not None and not 0 <= value <= 1:
            errors.append(f"{name} fraction is outside [0,1]: {value}")
    actual_external = deltas["source_external"] / prompt_tokens
    actual_local = deltas["source_local"] / prompt_tokens
    actual_compute = deltas["source_compute"] / prompt_tokens
    for name, value in (
        ("actual_external", actual_external),
        ("actual_local", actual_local),
        ("actual_compute", actual_compute),
    ):
        if not 0 <= value <= 1:
            errors.append(f"{name} fraction is outside [0,1]: {value}")
    source_total = sum(deltas[key] for key in SOURCE_KEYS)
    if source_total != prompt_tokens:
        errors.append(
            f"prompt token source deltas sum to {source_total}, but actual usage prompt tokens={prompt_tokens}; metrics lag/other traffic/accounting mismatch"
        )
    # This recipe isolates external reuse: enabling local HBM prefix hits is
    # a different experiment even when the external supply threshold passes.
    if arm == "on" and (deltas["local_hits"] != 0 or deltas["source_local"] != 0):
        errors.append(
            "ON requires local HBM lookup hits=0 and actual local_cache_hit tokens=0 for this recipe"
        )
    if arm == "on" and actual_external <= 0.50:
        errors.append(
            "ON actual external_kv_transfer must be strictly >50% of actual prompt tokens; lookup hits alone do not prove retrieval"
        )
    if arm == "off" and (deltas["source_external"] != 0 or deltas["source_local"] != 0):
        errors.append(
            "OFF actual external transfer and local cache hit tokens must both be zero"
        )
    if arm == "on" and not (tier is not None and tier > 0.50):
        errors.append("ON tier supply must be strictly >50% of actual prompt tokens")
    if arm == "off" and (
        deltas["local_hits"] != 0 or (not absent and deltas["external_hits"] != 0)
    ):
        errors.append(
            "OFF requires measured local_hits=0 and external_hits=0 when external metrics exist"
        )
    return {
        "counter_deltas": deltas,
        "counter_sums": totals,
        "actual_prompt_tokens": prompt_tokens,
        "tier_fraction_of_actual_prompt": tier,
        "tier_fraction_definition": "external lookup hits / actual usage prompt tokens; NOT proof of completed transfer",
        "actual_transfer_fraction_of_prompt": actual_external,
        "actual_local_fraction_of_prompt": actual_local,
        "actual_recompute_fraction_of_prompt": actual_compute,
        "actual_source_token_sum": source_total,
        "actual_source_metric": SOURCE_METRIC,
        "local_fraction_of_actual_prompt": local,
        "recompute_fraction_of_actual_prompt": recompute,
        "recompute_definition": "1 - local_hits/actual_prompt_tokens - external_hits/actual_prompt_tokens; not computed when external metrics are absent",
        "external_metrics_absent": absent,
        "external_note": (
            "OFF external families absent from BOTH scrapes; unknown, NOT zero; tier and recompute fractions unavailable"
            if absent
            else "external families present in both scrapes"
        ),
        "gates_passed": not errors,
        "errors": errors,
    }


def percentile(values, q):
    values = sorted(values)
    if not values:
        raise AcceptanceError("cannot compute percentile of an empty sample")
    index = (len(values) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def summarize_performance(records, expected_count=REQUESTS):
    if len(records) != expected_count or any(
        item["status"] != "passed" for item in records
    ):
        raise AcceptanceError(
            "performance requires every fixed-count measured request to pass; no partial-success statistics"
        )
    if len({item["request_id"] for item in records}) != expected_count:
        raise AcceptanceError("duplicate measured request IDs")
    for item in records:
        validate_usage(item["usage"], OUTPUT_TOKENS)
    first = min(item["started_monotonic_s"] for item in records)
    last = max(item["completed_monotonic_s"] for item in records)
    wall = last - first
    if wall <= 0:
        raise AcceptanceError("measurement window must be positive")
    prompt = sum(item["usage"]["prompt_tokens"] for item in records)
    output = sum(item["usage"]["completion_tokens"] for item in records)
    return {
        "window_definition": WINDOW,
        "first_dispatch_monotonic_s": first,
        "last_completion_monotonic_s": last,
        "window_s": wall,
        "requests": expected_count,
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": prompt + output,
        "request_per_s": expected_count / wall,
        "prompt_tok_per_s": prompt / wall,
        "output_tok_per_s": output / wall,
        "total_tok_per_s": (prompt + output) / wall,
        "latency_ms": {
            name: {
                f"p{int(q * 100)}": 1000
                * percentile([item[name + "_s"] for item in records], q)
                for q in (0.50, 0.90, 0.99)
            }
            for name in ("ttft", "tpot", "e2e")
        },
        "content_event_count": sum(item["content_event_count"] for item in records),
        "single_content_event_requests": sum(
            item["content_event_count"] == 1 for item in records
        ),
        "timing_definition": TIMING,
    }


async def collect_measured(session, url, manifest, sink):
    iterator = iter(manifest["measured"])

    async def worker():
        for request in iterator:
            await stream_request(session, url, manifest, request, sink)

    workers = [asyncio.create_task(worker()) for _ in range(manifest["concurrency"])]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker_task in workers:
            if not worker_task.done():
                worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


async def scrape_metrics(session, url, output_dir, name, timeout):
    metadata = {
        "url": url,
        "started_utc": utc_now(),
        "started_monotonic_s": time.perf_counter(),
    }
    try:
        async with session.get(url, timeout=timeout, allow_redirects=False) as response:
            raw = await response.read()
            Path(output_dir, name + ".prom").write_bytes(raw)
            metadata.update(
                {
                    "http_status": response.status,
                    "sha256": sha256(raw),
                    "bytes": len(raw),
                }
            )
            if response.status != 200:
                raise AcceptanceError(f"metrics HTTP {response.status}")
            return raw.decode("utf-8")
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["completed_monotonic_s"] = time.perf_counter()
        write_json(Path(output_dir, name + ".json"), metadata)


def validate_base_url(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise AcceptanceError(
            "base URL must be http(s)://host:port without credentials/path/query/fragment"
        )
    return value.rstrip("/")


async def run_arm(args):
    # Lazy import: manifest/compare and pure CPU tests have no HTTP dependency.
    import aiohttp

    manifest, raw, manifest_sha = load_manifest(args.manifest)
    base = validate_base_url(args.base_url)
    if (
        min(
            args.request_timeout,
            args.read_timeout,
            args.metrics_timeout,
            args.round_timeout,
        )
        <= 0
    ):
        raise AcceptanceError("timeouts must be positive")
    if args.round_timeout > 600 or args.metrics_timeout >= args.round_timeout:
        raise AcceptanceError("round timeout must be <=600s and exceed metrics timeout")
    parent = (
        Path(args.output_root)
        / manifest["case"]
        / f"round-{manifest['round']:04d}"
        / f"conc-{manifest['concurrency']}"
    )
    output_dir = parent / args.arm
    opposite = parent / ("off" if args.arm == "on" else "on") / "manifest.json"
    if opposite.exists() and sha256(opposite.read_bytes()) != manifest_sha:
        raise AcceptanceError(
            "paired ON/OFF manifest SHA mismatch; refusing before HTTP"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "manifest.json").open("xb") as handle:
        handle.write(raw)
    summary = {
        "schema_version": 1,
        "status": "failed",
        "case": manifest["case"],
        "round": manifest["round"],
        "arm": args.arm,
        "concurrency": manifest["concurrency"],
        "document_namespace": manifest["document_namespace"],
        "manifest_sha256": manifest_sha,
        "base_url": base,
        "started_utc": utc_now(),
        "window_definition": WINDOW,
        "errors": [],
        "performance": None,
        "supply": None,
        "pair_manifest_check": (
            "matched" if opposite.exists() else "other arm not yet present"
        ),
        "isolation_requirement": "Caller must verify target server identity/configuration and exclude other traffic; aggregate metric deltas cannot establish isolation.",
    }
    summary["validator_version"] = VALIDATOR_VERSION
    summary["round_timeout_s"] = args.round_timeout
    prime, measured, before, after = [], [], None, None
    log = (output_dir / "requests.jsonl").open("x", encoding="utf-8")

    def sink(record):
        (prime if record["phase"] == "prime" else measured).append(record)
        log.write(canonical(record).decode("utf-8") + "\n")
        log.flush()
        if record["status"] == "passed":
            # Drain liveness event after complete SSE + usage validation, NOT a score.
            completed = sum(item["status"] == "passed" for item in prime + measured)
            print(
                f"PERF_COMPLETED requests={completed} phase={record['phase']} prompt_tokens={record['usage']['prompt_tokens']}",
                flush=True,
            )
        if record["status"] != "passed":
            print(
                f"{record['request_id']}: {record['status']}: {record.get('error')}",
                file=sys.stderr,
                flush=True,
            )

    timeout = aiohttp.ClientTimeout(
        total=args.request_timeout, sock_read=args.read_timeout
    )
    round_started = time.perf_counter()
    work_deadline = round_started + args.round_timeout - args.metrics_timeout
    round_deadline = round_started + args.round_timeout
    try:
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=manifest["concurrency"]),
            trust_env=False,
        ) as session:

            async def work():
                nonlocal before
                for request in manifest["prime"]:
                    result = await stream_request(
                        session, base + "/v1/chat/completions", manifest, request, sink
                    )
                    if result["status"] != "passed":
                        raise AcceptanceError(
                            "serial prime failed; measured window not started"
                        )
                before = await scrape_metrics(
                    session,
                    base + "/metrics",
                    output_dir,
                    "metrics_before",
                    args.metrics_timeout,
                )
                metric_deltas(
                    before, before, args.arm
                )  # reject missing metrics before paying for a window
                await collect_measured(
                    session, base + "/v1/chat/completions", manifest, sink
                )

            try:
                # Reserve the final scrape budget inside the 600s round bound.
                await asyncio.wait_for(
                    work(), timeout=max(0, work_deadline - time.perf_counter())
                )
                # A synchronous sink or a final ready task can delay the event
                # loop beyond wait_for's timer without raising TimeoutError.
                if time.perf_counter() >= work_deadline:
                    raise asyncio.TimeoutError
            except asyncio.TimeoutError:
                summary["errors"].append(
                    f"round deadline: work budget {args.round_timeout - args.metrics_timeout}s exhausted; remaining budget reserved for metrics_after"
                )
            except asyncio.CancelledError:
                summary["errors"].append(
                    "run cancelled; partial requests are not a successful benchmark"
                )
            except Exception as exc:
                logger.exception("Performance workload failed")
                summary["errors"].append(f"{type(exc).__name__}: {exc}")
            finally:
                try:
                    after = await scrape_metrics(
                        session,
                        base + "/metrics",
                        output_dir,
                        "metrics_after",
                        args.metrics_timeout,
                    )
                except (Exception, asyncio.CancelledError) as exc:
                    logger.exception("Final metrics scrape failed")
                    summary["errors"].append(
                        f"metrics_after: {type(exc).__name__}: {exc}"
                    )
    finally:
        log.close()
    summary["round_elapsed_s"] = time.perf_counter() - round_started
    if time.perf_counter() >= round_deadline:
        summary["errors"].append(
            "round deadline exceeded including final metrics scrape"
        )
    summary["counts"] = {
        "planned_prime": DOCS,
        "recorded_prime": len(prime),
        "planned_measured": REQUESTS,
        "recorded_measured": len(measured),
        "passed_measured": sum(item["status"] == "passed" for item in measured),
        "failed_measured": sum(item["status"] == "failed" for item in measured),
        "cancelled_measured": sum(item["status"] == "cancelled" for item in measured),
        "not_started_measured": REQUESTS - len(measured),
    }
    summary["prompt_tokens_by_request"] = {
        item["request_id"]: (
            item["usage"].get("prompt_tokens")
            if isinstance(item.get("usage"), dict)
            else None
        )
        for item in measured
    }
    try:
        summary["performance"] = summarize_performance(measured)
        if before is None or after is None:
            raise AcceptanceError("both successful metrics scrapes are required")
        summary["supply"] = assess_supply(
            before, after, args.arm, summary["performance"]["prompt_tokens"]
        )
        summary["errors"].extend(summary["supply"]["errors"])
    except Exception as exc:
        logger.exception("Performance validation failed")
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    if len(prime) != DOCS or any(item["status"] != "passed" for item in prime):
        summary["errors"].append(
            "all 12 serial prime requests must pass with completion_tokens=1"
        )
    if not summary["errors"]:
        summary["status"] = "passed"
    summary["completed_utc"] = utc_now()
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(output_dir / "summary.json"),
                "manifest_sha256": manifest_sha,
                "errors": summary["errors"],
                "performance": summary["performance"],
                "supply": summary["supply"],
            },
            indent=2,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def compare_summaries(on, off):
    for value, arm in ((on, "on"), (off, "off")):
        if value.get("status") != "passed" or value.get("arm") != arm:
            raise AcceptanceError(f"{arm.upper()} summary is not a passing {arm} arm")
        if value.get("validator_version") != VALIDATOR_VERSION:
            raise AcceptanceError(
                "validator version mismatch; legacy summaries require separate revalidation"
            )
        if value.get("errors"):
            raise AcceptanceError("passing summary contains errors")
        supply = value["supply"]
        checked = assess_delta_supply(
            supply["counter_deltas"],
            supply["counter_sums"],
            supply["external_metrics_absent"],
            arm,
            value["performance"]["prompt_tokens"],
        )
        if not checked["gates_passed"] or checked != supply:
            raise AcceptanceError(
                "supply fails current validation or contains inconsistent derived fields"
            )
    for key in (
        "manifest_sha256",
        "case",
        "round",
        "concurrency",
        "document_namespace",
        "window_definition",
        "prompt_tokens_by_request",
    ):
        if on.get(key) != off.get(key) or key not in on:
            raise AcceptanceError(f"ON/OFF mismatch: {key}")
    if on["window_definition"] != WINDOW:
        raise AcceptanceError("unexpected measurement window definition")
    for value in (on, off):
        if (
            value["performance"]["requests"] != REQUESTS
            or value["performance"]["completion_tokens"] != REQUESTS * OUTPUT_TOKENS
        ):
            raise AcceptanceError(
                "ON/OFF comparison requires exactly 240 x 64 output tokens per arm"
            )
        by_request = value["prompt_tokens_by_request"]
        if set(by_request) != {f"measured-{i:06d}" for i in range(REQUESTS)}:
            raise AcceptanceError("comparison requires all 240 measured request IDs")
        if any(
            type(tokens) is not int or tokens <= 0 for tokens in by_request.values()
        ):
            raise AcceptanceError("invalid per-request prompt token count")
        if sum(by_request.values()) != value["performance"]["prompt_tokens"]:
            raise AcceptanceError("per-request and total prompt tokens differ")
        for key in ("request_per_s", "output_tok_per_s", "total_tok_per_s"):
            rate = value["performance"][key]
            if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
                raise AcceptanceError(f"invalid performance rate: {key}")
    return {
        "status": "passed",
        "manifest_sha256": on["manifest_sha256"],
        "case": on["case"],
        "round": on["round"],
        "concurrency": on["concurrency"],
        "on_over_off": {
            key: on["performance"][key] / off["performance"][key]
            for key in ("request_per_s", "output_tok_per_s", "total_tok_per_s")
        },
        "note": "One paired round only; no noise-floor or statistical-significance claim.",
    }


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "manifest",
        help="CPU-only: create a fixed recipe and unique namespace; reuse this file for ON/OFF",
    )
    create.add_argument("--case", required=True)
    create.add_argument("--round", type=int, required=True)
    create.add_argument("--model", default="DeepSeek-V4.1-Flash")
    create.add_argument("--concurrency", type=int, default=24)
    create.add_argument(
        "--namespace",
        help="optional nonce (round/case/concurrency are also included); default random UUID",
    )
    create.add_argument("--output", required=True)
    run = commands.add_parser(
        "run", help="HTTP client only: run one arm against an existing server"
    )
    run.add_argument("--manifest", required=True)
    run.add_argument("--arm", choices=("on", "off"), required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument(
        "--round-timeout",
        type=float,
        default=600,
        help="round deadline <=600s, includes serial prime/window/final metrics; final scrape budget is reserved",
    )
    run.add_argument(
        "--request-timeout",
        type=float,
        default=600,
        help="hard total seconds per request; also bounded by round deadline",
    )
    run.add_argument(
        "--read-timeout", type=float, default=120, help="maximum silent socket seconds"
    )
    run.add_argument("--metrics-timeout", type=float, default=15)
    compare = commands.add_parser(
        "compare",
        help="CPU-only: check a completed pair before reporting ON/OFF throughput ratios",
    )
    compare.add_argument("--on-summary", required=True)
    compare.add_argument("--off-summary", required=True)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.command == "manifest":
            manifest = build_manifest(
                args.case, args.round, args.model, args.concurrency, args.namespace
            )
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = canonical(manifest) + b"\n"
            with path.open("xb") as handle:
                handle.write(raw)
            print(
                json.dumps(
                    {
                        "manifest": str(path),
                        "manifest_sha256": sha256(raw),
                        "document_namespace": manifest["document_namespace"],
                        "requests": REQUESTS,
                        "concurrency": manifest["concurrency"],
                    }
                )
            )
            return 0
        if args.command == "run":
            return asyncio.run(run_arm(args))
        result = compare_summaries(
            json.loads(Path(args.on_summary).read_text()),
            json.loads(Path(args.off_summary).read_text()),
        )
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("FAILED: interrupted/cancelled", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("Performance client failed")
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
