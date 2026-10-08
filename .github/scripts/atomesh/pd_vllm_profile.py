"""Bounded native PD diagnostic; all profiling controls use direct endpoints."""

import argparse
import asyncio
import json
import math
import re
import statistics
import time
import uuid
from pathlib import Path

import httpx


def token_counters(text):
    result = {}
    for line in text.splitlines():
        if line.startswith("vllm:prompt_tokens_by_source_total{"):
            source = re.search(r'source="([^"]+)"', line)
            if source:
                key = source.group(1)
                result[key] = result.get(key, 0) + float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:request_success_total{"):
            result["request_success"] = result.get("request_success", 0) + float(
                line.rsplit(" ", 1)[1]
            )
    return result


def counter_delta(before, after):
    return {key: after[key] - value for key, value in before.items() if key in after}


COMPOSITION_SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")
COMPOSITION_METRICS = (
    "vllm:prompt_tokens_by_source_total",
    "vllm:request_success_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:diffusion_num_denoising_steps_total",
    "vllm:diffusion_num_canvas_positions_total",
    "vllm:diffusion_num_committed_tokens_total",
)


def composition_series(text, strict=False):
    """Keep labeled counters separate; missing samples are never baseline zero."""
    series = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^\s{]+)(\{.*\})?\s+(\S+)(?:\s+\S+)?", line)
        if match and match[1] in COMPOSITION_METRICS:
            labels = sorted(re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[2] or ""))
            key = match[1] + json.dumps(labels, separators=(",", ":"))
            if strict:
                require(key not in series, f"Duplicate metric series: {key}")
            assert key not in series, f"Duplicate metric series: {key}"
            value = float(match[3])
            if strict:
                require(
                    math.isfinite(value) and value >= 0 and value.is_integer(),
                    "Invalid counter",
                )
            assert math.isfinite(value) and value >= 0 and value.is_integer(), (
                "Invalid counter",
                key,
                value,
            )
            series[key] = value
    return series


def composition_totals(series, strict=False):
    totals = dict.fromkeys((*COMPOSITION_SOURCES, "request_success"))
    source_groups = {}
    success_groups = set()
    for key, value in series.items():
        if key.startswith("vllm:prompt_tokens_by_source_total"):
            labels = dict(json.loads(key[len("vllm:prompt_tokens_by_source_total") :]))
            source = labels.pop("source", None)
            if strict:
                require(
                    source in COMPOSITION_SOURCES, f"Unknown prompt source: {source}"
                )
            assert source in COMPOSITION_SOURCES, f"Unknown prompt source: {source}"
            group = tuple(sorted(labels.items()))
            source_groups.setdefault(group, set()).add(source)
        elif key.startswith("vllm:request_success_total"):
            labels = dict(json.loads(key[len("vllm:request_success_total") :]))
            labels.pop("finished_reason", None)
            success_groups.add(tuple(sorted(labels.items())))
            source = "request_success"
        else:
            continue
        if value is None:
            return None
        totals[source] = (totals[source] or 0) + value
    if strict:
        require(
            source_groups.keys() == success_groups
            and all(
                sources == set(COMPOSITION_SOURCES)
                for sources in source_groups.values()
            ),
            "Missing per-engine source/success metrics",
        )
    assert source_groups.keys() == success_groups and all(
        sources == set(COMPOSITION_SOURCES) for sources in source_groups.values()
    ), "Missing per-engine source/success metrics"
    return totals


class OutputNotFreshError(ValueError):
    pass


def require(condition, message="Invalid bounded-perf evidence"):
    if not condition:
        raise AssertionError(message)


async def run(args):
    if (
        args.mode == "bounded-perf"
        and args.output.exists()
        and any(args.output.iterdir())
    ):
        raise OutputNotFreshError(
            "bounded-perf requires a fresh empty output directory"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    http_timings = {}
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:

        async def post(url, path, body=None, headers=None):
            if args.mode == "bounded-perf":
                with (args.output / "http-requests.jsonl").open("a") as output:
                    output.write(
                        json.dumps(
                            {"url": url + path, "request": body, "headers": headers}
                        )
                        + "\n"
                    )
            if args.mode == "bounded-perf":
                http_start = time.perf_counter()
            response = await client.post(url + path, json=body, headers=headers)
            if args.mode == "bounded-perf":
                http_ms = 1000 * (time.perf_counter() - http_start)
                http_timings[url] = http_ms
                with (args.output / "http-responses.jsonl").open("a") as output:
                    output.write(
                        json.dumps(
                            {
                                "url": url + path,
                                "request": body,
                                "headers": headers,
                                "status_code": response.status_code,
                                "http_ms": http_ms,
                                "response": response.text,
                            }
                        )
                        + "\n"
                    )
            response.raise_for_status()
            return response

        async def snapshot(tag):
            counters = {}
            for role in ("prefill", "decode"):
                response = await client.get(getattr(args, role) + "/metrics")
                response.raise_for_status()
                (args.output / f"{tag}-{role}.metrics.txt").write_text(response.text)
                counters[role] = token_counters(response.text)
            return counters

        async def reset():
            for role in ("prefill", "decode"):
                response = await post(getattr(args, role), "/reset_prefix_cache")
                result = response.json()
                with (args.output / "cache-resets.jsonl").open("a") as output:
                    output.write(json.dumps({"role": role, "response": result}) + "\n")
                assert (
                    result.get("success") is True
                ), f"{role} cache reset failed: {result}"

        async def complete(url, prompt, count, params=None, request_id=None):
            body = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": count,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "stream": False,
                "return_token_ids": True,
            }
            if params is not None:
                body["kv_transfer_params"] = params
            response = await post(
                url,
                "/v1/completions",
                body,
                {"X-Request-Id": request_id or uuid.uuid4().hex},
            )
            result = response.json()
            try:
                if args.mode == "bounded-perf":
                    check_completion(result, prompt, count)
                assert result.get("choices"), result
                choice = result["choices"][0]
                # API echo is the original prompt, not the hybrid producer's N-1
                # effective prompt used only for source accounting.
                prompt_ids = choice.get("prompt_token_ids")
                assert (
                    isinstance(prompt_ids, list)
                    and all(type(token) is int for token in prompt_ids)
                    and prompt_ids == prompt
                ), f"Invalid prompt_token_ids: {choice}"
                output_ids = choice.get("token_ids")
                assert (
                    isinstance(output_ids, list)
                    and all(type(token) is int and token >= 0 for token in output_ids)
                    and len(output_ids) == count
                ), f"Invalid token_ids for max_tokens={count}: {choice}"
            except (AssertionError, AttributeError, KeyError, TypeError):
                with (args.output / "completion-validation-failures.jsonl").open(
                    "a"
                ) as output:
                    output.write(
                        json.dumps({"url": url, "request": body, "response": result})
                        + "\n"
                    )
                raise
            if params is None:
                # Save validated ordinary references before any reset, PD, or metrics failure.
                with (args.output / "direct-references.jsonl").open("a") as output:
                    output.write(
                        json.dumps({"url": url, "request": body, "response": result})
                        + "\n"
                    )
            return result

        async def pd(prompt, count, tag):
            request_id = "native-pd-" + uuid.uuid4().hex
            start = time.perf_counter()
            prefill = await complete(
                args.prefill,
                prompt,
                1,
                {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_tp_size": args.tp,
                    "remote_dp_size": 1,
                    "transfer_id": "moriio-" + uuid.uuid4().hex,
                },
                request_id,
            )
            after_prefill = time.perf_counter()
            transfer = prefill["kv_transfer_params"]
            if args.mode == "bounded-perf":
                require(
                    transfer.get("remote_block_ids") and transfer.get("remote_host"),
                    "Missing handoff",
                )
            else:
                assert (
                    transfer["remote_block_ids"] and transfer["remote_host"]
                ), transfer
            record = {
                "tag": tag,
                "request_id": request_id,
                "prompt_tokens": len(prompt),
                "prefill": prefill,
                "prefill_ms": 1000 * (after_prefill - start),
            }
            records.append(record)
            if args.mode != "bounded-perf":
                (args.output / "requests.json").write_text(
                    json.dumps(records, indent=2)
                )
            decode = await complete(
                args.decode,
                prompt,
                count,
                {
                    **transfer,
                    "remote_tp_size": args.tp,
                    "remote_dp_size": 1,
                    "do_remote_decode": False,
                    "do_remote_prefill": True,
                },
                request_id,
            )
            finished = time.perf_counter()
            record.update(
                {
                    "decode_ms": 1000 * (finished - after_prefill),
                    "decode": decode,
                }
            )
            if args.mode == "bounded-perf":
                record["client_inclusive_elapsed_ms"] = 1000 * (finished - start)
                record["p_http_ms"] = http_timings[args.prefill]
                record["d_http_ms"] = http_timings[args.decode]
                record["client_http_total_ms"] = (
                    record["p_http_ms"] + record["d_http_ms"]
                )
            (args.output / "requests.json").write_text(json.dumps(records, indent=2))
            return decode

        async def measured_pd(prompt, count, tag, cache_state):
            evidence = {
                "tag": tag,
                "status": "PENDING_REVIEW",
                "cache_state": cache_state,
                "prompt_tokens": len(prompt),
                "producer_effective_prompt_tokens_expected": len(prompt)
                - int(args.hybrid),
                "dcp_block_tokens": 128 * args.dcp,
                "decoder_protocol_tail_tokens": 1,
                "attribution": "serial request; no concurrent workload permitted",
            }
            before = await snapshot(tag + "-before")
            try:
                response = await pd(prompt, count, tag)
                evidence["http_completed"] = True
                evidence["finish_reason"] = response["choices"][0].get("finish_reason")
                # Metrics are emitted on the engine stats interval. Bound the wait.
                for attempt in range(12):
                    after = await snapshot(tag + "-after")
                    deltas = {
                        role: counter_delta(before[role], after[role])
                        for role in before
                    }
                    if all(d.get("request_success", 0) >= 1 for d in deltas.values()):
                        break
                    await asyncio.sleep(0.5)
                evidence["token_counter_deltas"] = deltas
                d = deltas["decode"]
                evidence["producer_effective_prompt_tokens_observed"] = sum(
                    deltas["prefill"].get(k, 0)
                    for k in (
                        "local_compute",
                        "local_cache_hit",
                        "external_kv_transfer",
                    )
                )
                required = {
                    "local_compute",
                    "local_cache_hit",
                    "external_kv_transfer",
                    "request_success",
                }
                if required <= d.keys() and d["request_success"] == 1:
                    if cache_state == "cold-reset-both":
                        assert (
                            d["external_kv_transfer"] > 0
                        ), "No external KV token consumption"
                    elif d["external_kv_transfer"] == 0:
                        evidence["review_reason"] = (
                            "Warm APC may hide transfer; not remote-transfer proof"
                        )
                    assert (
                        d["local_compute"] >= 1
                    ), "Missing READ protocol tail computation"
                    assert sum(d[k] for k in required - {"request_success"}) == len(
                        prompt
                    ), "Prompt token accounting mismatch"
                    evidence["accounting_checked"] = True
                else:
                    evidence["accounting_checked"] = False
                    evidence["review_reason"] = (
                        "Missing/delayed or non-isolated metrics"
                    )
                assert (
                    evidence["finish_reason"] == "length"
                ), "Incomplete decode response"
                return response
            except Exception as exc:
                evidence["status"] = "FAIL"
                evidence["error"] = repr(exc)
                try:
                    after = await snapshot(tag + "-failure")
                    evidence["token_counter_deltas"] = {
                        role: counter_delta(before[role], after[role])
                        for role in before
                    }
                except (httpx.HTTPError, OSError, ValueError) as metric_exc:
                    evidence["metrics_error"] = repr(metric_exc)
                raise
            finally:
                (args.output / f"{tag}-evidence.json").write_text(
                    json.dumps(evidence, indent=2) + "\n"
                )

        if args.mode == "bounded-perf":
            require(
                args.hybrid and (not getattr(args, "cache_composition", False)),
                "Invalid bounded-perf evidence",
            )
            tpot_prefix = "vllm:request_time_per_output_token_seconds_"
            spec_names = dict(
                zip(
                    ("drafts", "draft_tokens", "accepted_tokens"),
                    COMPOSITION_METRICS[2:5],
                )
            )
            metadata = {
                "mode": args.mode,
                "concurrency": 1,
                "output_tokens": 128,
                "lengths": [1025, 2050, 3073],
                "apc_effect_evaluated": False,
                "cache_policy": "APC configuration retained; reset both roles before every request",
                "timing_scope": "instrumented client P-to-D diagnostic elapsed; D-leg server amortized mean TPOT; not stream ITL, TTFT or GPU kernel time",
                "client_inclusive_elapsed_ms_definition": "same clock before P through validated D response; includes per-request journal, validation, request construction and handoff; variable instrumentation overhead; excludes growing historical evidence rewrites",
                "client_http_total_ms_definition": "p_http_ms+d_http_ms, each measured around await client.post; excludes client inter-request gap, not end-to-end; journal delays may indirectly affect subsequent D state",
                "endpoint_requirement": "exclusive endpoints; C1 alone does not exclude other traffic",
                "tpot_definition": "server (last_token_ts-first_token_ts)/(actual generation_tokens-1); no TP scaling or client denominator",
                "bytes": "UNKNOWN",
                "ack": "UNKNOWN",
                "speedup_evaluated": False,
            }

            def save(tag, evidence):
                (args.output / f"{tag}-evidence.json").write_text(
                    json.dumps(evidence, indent=2, allow_nan=False) + "\n"
                )

            pinned_series = {}
            last_scrape = {}

            async def bounded_snapshot(tag):
                result = {}
                for role in ("prefill", "decode"):
                    response = await client.get(getattr(args, role) + "/metrics")
                    (args.output / f"{tag}-{role}.metrics.txt").write_text(
                        response.text
                    )
                    response.raise_for_status()
                    parsed = composition_series(response.text, strict=True)
                    for line in response.text.splitlines():
                        match = re.fullmatch(
                            r"([^\s{]+)(\{.*\})?\s+(\S+)(?:\s+\S+)?", line
                        )
                        if not match or match[1] not in (
                            tpot_prefix + "sum",
                            tpot_prefix + "count",
                        ):
                            continue
                        labels = sorted(
                            re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[2] or "")
                        )
                        key = match[1] + json.dumps(labels, separators=(",", ":"))
                        value = float(match[3])
                        require(key not in parsed, f"Duplicate timing series: {key}")
                        require(
                            math.isfinite(value) and value >= 0,
                            "Invalid timing counter",
                        )
                        if match[1].endswith("count"):
                            require(value.is_integer(), "Non-integer timing count")
                        parsed[key] = value
                    keys = set(parsed)
                    if role in pinned_series:
                        require(
                            keys == pinned_series[role],
                            "Survey metric label drift/missing series",
                        )
                        require(
                            all(parsed[key] >= last_scrape[role][key] for key in keys),
                            "Counter regression between scrapes",
                        )
                    else:
                        pinned_series[role] = keys
                    last_scrape[role] = parsed
                    result[role] = parsed
                return result

            def group_values(series, name):
                return {
                    key[len(name) :]: value
                    for key, value in series.items()
                    if key.startswith(name + "[")
                }

            def validate_groups(series):
                # Success labels identify engines (DP engines, not TP ranks).
                groups = {
                    json.dumps(
                        sorted(
                            (k, v)
                            for k, v in json.loads(labels)
                            if k != "finished_reason"
                        ),
                        separators=(",", ":"),
                    )
                    for labels in group_values(series, COMPOSITION_METRICS[1])
                }
                require(len(groups) == 1, "Expected one fixed DP-engine label group")
                for name in (
                    *spec_names.values(),
                    tpot_prefix + "sum",
                    tpot_prefix + "count",
                ):
                    values = group_values(series, name)
                    require(set(values) == groups, f"Missing/misaligned metric: {name}")
                    require(
                        all(value is not None for value in values.values()),
                        f"Missing metric: {name}",
                    )

            def check_completion(response, prompt, count):
                require(
                    isinstance(response.get("choices"), list)
                    and len(response["choices"]) == 1,
                    "Expected one completion",
                )
                choice = response["choices"][0]
                require(
                    isinstance(choice.get("text"), str),
                    "Completion text must be a string",
                )
                for key, expected in (
                    ("prompt_token_ids", len(prompt)),
                    ("token_ids", count),
                ):
                    ids = choice.get(key)
                    require(
                        isinstance(ids, list)
                        and len(ids) == expected
                        and all(type(token) is int and token >= 0 for token in ids),
                        f"Invalid {key}",
                    )
                require(choice.get("finish_reason") == "length", "Incomplete response")
                require(
                    choice.get("prompt_token_ids") == prompt,
                    "Invalid bounded-perf evidence",
                )
                usage = response.get("usage")
                require(isinstance(usage, dict), "Missing usage")
                for key, expected in {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": count,
                    "total_tokens": len(prompt) + count,
                }.items():
                    require(
                        type(usage.get(key)) is int and usage[key] == expected,
                        "Incorrect usage",
                    )

            async def bounded_request(tag, prompt, reference=None):
                evidence = {
                    **metadata,
                    "tag": tag,
                    "status": "PENDING_REVIEW",
                    "prompt_tokens": len(prompt),
                    "timing": None,
                    "speculative_counters": None,
                    "timing_status": "UNKNOWN",
                }
                save(tag, evidence)
                try:
                    for role in ("prefill", "decode"):
                        response = await post(
                            getattr(args, role),
                            "/reset_prefix_cache?reset_external=false&reset_running_requests=false",
                        )
                        require(
                            response.json().get("success") is True, "Cache reset failed"
                        )
                    before = await bounded_snapshot(tag + "-before")
                    evidence["labeled_counter_baseline"] = before
                    save(tag, evidence)
                    for series in before.values():
                        composition_totals(series, strict=True)
                        validate_groups(series)
                    if reference is None:
                        response = await complete(args.prefill, prompt, 128)
                        evidence["direct"] = response
                    else:
                        response = await pd(prompt, 128, tag)
                        evidence["direct"] = reference
                        evidence["pd"] = response
                        for key in (
                            "client_inclusive_elapsed_ms",
                            "p_http_ms",
                            "d_http_ms",
                            "client_http_total_ms",
                        ):
                            evidence[key] = records[-1][key]
                        records[-1]["direct"] = reference
                        (args.output / "requests.json").write_text(
                            json.dumps(records, indent=2)
                        )
                    save(tag, evidence)
                    check_completion(response, prompt, 128)
                    evidence["actual_completion_tokens"] = {
                        "prefill": (
                            1
                            if reference is not None
                            else response["usage"]["completion_tokens"]
                        ),
                        "decode": (
                            response["usage"]["completion_tokens"]
                            if reference is not None
                            else 0
                        ),
                    }
                    if reference is not None:
                        check_completion(records[-1]["prefill"], prompt, 1)
                        evidence["direct_pd_token_ids_equal"] = (
                            response["choices"][0]["token_ids"]
                            == reference["choices"][0]["token_ids"]
                        )
                        evidence["direct_pd_text_equal"] = (
                            response["choices"][0]["text"]
                            == reference["choices"][0]["text"]
                        )
                        save(tag, evidence)
                        require(
                            evidence["direct_pd_token_ids_equal"]
                            and evidence["direct_pd_text_equal"],
                            "Direct/PD mismatch",
                        )
                    expected = {
                        "prefill": {
                            "local_compute": len(prompt) - int(reference is not None),
                            "local_cache_hit": 0,
                            "external_kv_transfer": 0,
                            "request_success": 1,
                        },
                        "decode": {
                            "local_compute": int(reference is not None),
                            "local_cache_hit": 0,
                            "external_kv_transfer": (
                                len(prompt) - 1 if reference is not None else 0
                            ),
                            "request_success": int(reference is not None),
                        },
                    }
                    previous = None
                    for attempt in range(16):
                        after = await bounded_snapshot(f"{tag}-after-{attempt:02d}")
                        labeled = {
                            role: {
                                key: (
                                    after[role][key] - before[role][key]
                                    if key in before[role] and key in after[role]
                                    else None
                                )
                                for key in before[role].keys() | after[role].keys()
                            }
                            for role in before
                        }
                        evidence["labeled_counter_deltas"] = labeled
                        save(tag, evidence)
                        totals, timing, speculative = {}, {}, {}
                        for role, series in labeled.items():
                            require(
                                all(
                                    value is not None and value >= 0
                                    for value in series.values()
                                ),
                                "Missing/regressing counter",
                            )
                            validate_groups(series)
                            totals[role] = composition_totals(series, strict=True)
                            require(
                                all(
                                    (
                                        totals[role][key] <= value
                                        for key, value in expected[role].items()
                                    )
                                ),
                                "Non-isolated or incorrect source accounting",
                            )
                            count = sum(
                                group_values(series, tpot_prefix + "count").values()
                            )
                            total = sum(
                                group_values(series, tpot_prefix + "sum").values()
                            )
                            require(
                                count <= expected[role]["request_success"],
                                "Non-isolated timing count",
                            )
                            require(
                                count != 0 or total == 0,
                                "Timing sum without observation",
                            )
                            timing[role] = {
                                "count": count,
                                "sum_seconds": total,
                                "mean_tpot_seconds": total if count == 1 else None,
                            }
                            speculative[role] = {
                                key: sum(group_values(series, name).values())
                                for key, name in spec_names.items()
                            }
                        evidence.update(
                            token_counter_deltas=totals,
                            timing=timing,
                            speculative_counters=speculative,
                        )
                        save(tag, evidence)
                        ready = totals == expected and all(
                            timing[role]["count"] == values["request_success"]
                            for role, values in expected.items()
                        )
                        if ready and labeled == previous and attempt >= 6:
                            for role, values in speculative.items():
                                require(
                                    values["accepted_tokens"] <= values["draft_tokens"],
                                    "Accepted exceeds drafted tokens",
                                )
                                active = role == (
                                    "decode" if reference is not None else "prefill"
                                )
                                if active:
                                    require(
                                        values["drafts"] > 0
                                        and values["draft_tokens"] > 0,
                                        "No observed draft activity",
                                    )
                                if expected[role]["request_success"] == 0:
                                    require(
                                        not any(values.values()),
                                        "Unexpected speculative activity",
                                    )
                            evidence["accounting_checked"] = True
                            evidence["timing_status"] = "OBSERVED_SINGLE_REQUEST"
                            save(tag, evidence)
                            return response, evidence
                        previous = labeled if ready else None
                        await asyncio.sleep(0.5)
                    raise AssertionError("Missing/delayed/non-isolated metrics")
                except Exception as exc:
                    evidence.update(status="FAIL", error=repr(exc))
                    save(tag, evidence)
                    raise

            try:
                response = await post(
                    args.prefill,
                    "/tokenize",
                    {
                        "model": args.model,
                        "prompt": "\n".join(
                            f"Record {i}: The service reads a buffer and computes a result."
                            for i in range(600)
                        ),
                    },
                )
                tokens = response.json()["tokens"]
                require(
                    isinstance(tokens, list) and len(tokens) >= 3073,
                    "Invalid bounded-perf evidence",
                )
                require(
                    all(type(token) is int and token >= 0 for token in tokens),
                    "Invalid bounded-perf evidence",
                )
                (args.output / "workload.json").write_text(
                    json.dumps({**metadata, "tokens": tokens}, indent=2)
                )
                references = {}
                for length in metadata["lengths"]:
                    prompt = tokens[:length]
                    reference, _ = await bounded_request(f"reference-{length}", prompt)
                    references[length] = reference
                    await bounded_request(f"correctness-{length}", prompt, reference)
                await bounded_request("warmup", tokens[:1025], references[1025])
                samples = []
                for index in range(3):
                    _, evidence = await bounded_request(
                        f"timed-{index}", tokens[:1025], references[1025]
                    )
                    samples.append(
                        {
                            "tag": evidence["tag"],
                            **{
                                key: evidence[key]
                                for key in (
                                    "client_inclusive_elapsed_ms",
                                    "p_http_ms",
                                    "d_http_ms",
                                    "client_http_total_ms",
                                )
                            },
                            "decode_tpot_seconds": evidence["timing"]["decode"][
                                "mean_tpot_seconds"
                            ],
                        }
                    )
                summary = {
                    key: {
                        "min": min(s[key] for s in samples),
                        "median": statistics.median(s[key] for s in samples),
                        "max": max(s[key] for s in samples),
                    }
                    for key in (
                        "client_inclusive_elapsed_ms",
                        "client_http_total_ms",
                        "decode_tpot_seconds",
                    )
                }
                (args.output / "complete.json").write_text(
                    json.dumps(
                        {
                            **metadata,
                            "status": "PENDING_REVIEW",
                            "samples": samples,
                            "summary": summary,
                            "direct_pd_token_id_checks": 7,
                        },
                        indent=2,
                    )
                )
            except Exception as exc:
                (args.output / "failure.json").write_text(
                    json.dumps(
                        {**metadata, "status": "FAIL", "error": repr(exc)}, indent=2
                    )
                )
                raise
            return

        if getattr(args, "cache_composition", False):
            assert args.hybrid, "Cache composition requires hybrid P N-1 semantics"
            metadata = {
                "profile": "hash-partial-tail-under4096",
                "requested_hash_block_tokens": 128,
                "physical_geometry": "UNKNOWN_PENDING_LOG_AUDIT",
                "bytes": "UNKNOWN",
                "ack": "UNKNOWN",
                "speculative_certification": False,
                "phase": args.phase,
            }

            async def composition_snapshot(tag):
                result = {}
                for role in ("prefill", "decode"):
                    response = await client.get(getattr(args, role) + "/metrics")
                    response.raise_for_status()
                    (args.output / f"{tag}-{role}.metrics.txt").write_text(
                        response.text
                    )
                    result[role] = composition_series(response.text)
                return result

            async def composition_reset(roles):
                for role in roles:
                    response = await post(
                        getattr(args, role),
                        "/reset_prefix_cache?reset_external=false"
                        "&reset_running_requests=false",
                    )
                    result = response.json()
                    with (args.output / "cache-resets.jsonl").open("a") as output:
                        output.write(
                            json.dumps({"role": role, "response": result}) + "\n"
                        )
                    assert (
                        result.get("success") is True
                    ), f"{role} cache reset failed: {result}"

            async def drain(tag, before, expected, evidence):
                # No filler requests: poll only, and require a second stable scrape.
                previous = None
                for attempt in range(12):
                    after = await composition_snapshot(f"{tag}-after-{attempt:02d}")
                    labeled = {
                        role: {
                            key: (
                                after[role][key] - before[role][key]
                                if key in after[role] and key in before[role]
                                else None
                            )
                            for key in before[role].keys() | after[role].keys()
                        }
                        for role in before
                    }
                    evidence["labeled_counter_deltas"] = labeled
                    deltas = {}
                    for role in before:
                        required_names = COMPOSITION_METRICS[:2]
                        required = {
                            key: value
                            for key, value in labeled[role].items()
                            if key.startswith(required_names)
                        }
                        assert all(value is not None for value in required.values()), (
                            "Missing before/after metric series",
                            role,
                            required,
                        )
                        assert all(value >= 0 for value in required.values()), (
                            "Counter regression",
                            role,
                            required,
                        )
                        totals = composition_totals(required)
                        assert totals is not None and all(
                            value is not None for value in totals.values()
                        ), ("Missing source/success metrics", role, totals)
                        assert totals["request_success"] <= expected[role][0], (
                            "Non-isolated request count",
                            role,
                            totals,
                        )
                        assert (
                            sum(totals[k] for k in COMPOSITION_SOURCES)
                            <= expected[role][1]
                        ), ("Non-isolated source accounting", role, totals)
                        deltas[role] = totals
                    evidence["token_counter_deltas"] = deltas
                    ready = all(
                        deltas[role]["request_success"] == count
                        and sum(deltas[role][k] for k in COMPOSITION_SOURCES) == tokens
                        for role, (count, tokens) in expected.items()
                    )
                    if ready and labeled == previous:
                        evidence["accounting_checked"] = True
                        evidence["speculative_counters"] = {
                            role: {
                                key: value
                                for key, value in labeled[role].items()
                                if key.startswith(COMPOSITION_METRICS[2:])
                            }
                            or None
                            for role in labeled
                        }
                        return deltas
                    previous = labeled if ready else None
                    await asyncio.sleep(0.5)
                raise AssertionError("Missing/delayed or non-isolated metrics")

            async def composition_request(
                tag, prompt, reference=None, cold_d=True, warm_p=False
            ):
                evidence = {
                    **metadata,
                    "tag": tag,
                    "status": "PENDING_REVIEW",
                    "prompt_tokens": len(prompt),
                    "producer_effective_prompt_tokens_expected": len(prompt)
                    - int(reference is not None),
                    "decoder_prompt_tokens_expected": (
                        len(prompt) if reference is not None else 0
                    ),
                    "producer_cache_hit_required": warm_p,
                    "consumer_cache_hit_required": not cold_d,
                    "decoder_cold_required": cold_d,
                    "attribution": "serial request; no concurrent workload permitted",
                }
                try:
                    before = await composition_snapshot(tag + "-before")
                    evidence["labeled_counter_baseline"] = before
                    for role in before:
                        totals = composition_totals(before[role])
                        assert totals is not None and all(
                            value is not None for value in totals.values()
                        ), ("Missing baseline source/success metrics", role, totals)
                    if reference is None:
                        response = await complete(args.prefill, prompt, 16)
                        evidence["direct"] = response
                        expected = {"prefill": (1, len(prompt)), "decode": (0, 0)}
                    else:
                        response = await pd(prompt, 16, tag)
                        records[-1]["direct"] = reference
                        (args.output / "requests.json").write_text(
                            json.dumps(records, indent=2)
                        )
                        evidence["direct_pd_token_ids_equal"] = (
                            response["choices"][0]["token_ids"]
                            == reference["choices"][0]["token_ids"]
                        )
                        evidence["direct_pd_text_equal"] = (
                            response["choices"][0]["text"]
                            == reference["choices"][0]["text"]
                        )
                        expected = {
                            "prefill": (1, len(prompt) - 1),
                            "decode": (1, len(prompt)),
                        }
                    assert (
                        response["choices"][0].get("finish_reason") == "length"
                    ), "Incomplete response"
                    d = await drain(tag, before, expected, evidence)
                    if reference is None:
                        assert d["prefill"]["local_compute"] == len(
                            prompt
                        ), "Reference not cold"
                    if reference is not None:
                        assert evidence[
                            "direct_pd_token_ids_equal"
                        ], "Direct/PD token IDs differ"
                        assert evidence[
                            "direct_pd_text_equal"
                        ], "Direct/PD text differs"
                        p, decoder = d["prefill"], d["decode"]
                        assert (
                            p["external_kv_transfer"] == 0
                        ), "Unexpected producer external tokens"
                        assert (
                            decoder["local_compute"] >= 1
                        ), "Missing READ tail computation"
                        if cold_d:
                            assert (
                                decoder["local_cache_hit"] == 0
                            ), "Decoder cache not cold"
                            assert (
                                decoder["external_kv_transfer"] > 0
                            ), "No external KV token consumption"
                        missing_hits = [
                            role
                            for role, required in (
                                ("prefill", warm_p),
                                ("decode", not cold_d),
                            )
                            if required and d[role]["local_cache_hit"] == 0
                        ]
                        if missing_hits:
                            evidence["status"] = "APC_NOT_EXERCISED"
                            evidence["apc_not_exercised_roles"] = missing_hits
                            evidence["finding"] = (
                                "Required cache hit not observed; not PD unsupported"
                            )
                        if not warm_p:
                            assert p["local_cache_hit"] == 0, "Producer cache not cold"
                    return response, evidence
                except Exception as exc:
                    evidence["status"] = "FAIL"
                    evidence["error"] = repr(exc)
                    raise
                finally:
                    (args.output / f"{tag}-evidence.json").write_text(
                        json.dumps(evidence, indent=2) + "\n"
                    )

            response = await post(
                args.prefill,
                "/tokenize",
                {
                    "model": args.model,
                    "prompt": "\n".join(
                        f"Record {i}: The service reads a buffer and computes a result."
                        for i in range(300)
                    ),
                },
            )
            tokens = response.json()["tokens"]
            assert len(tokens) >= 1153
            prompt = tokens[:1026]
            branch = tokens[:1153]
            alternate = next((token for token in tokens if token != branch[1024]), None)
            assert alternate is not None, "Need a distinct token for branch divergence"
            branch[1024] = alternate
            workload = {
                **metadata,
                "prompt": prompt,
                "branch": branch,
                "output_tokens": 16,
            }
            (args.output / "workload.json").write_text(json.dumps(workload, indent=2))
            references = []
            for name, item in (("base", prompt), ("branch", branch)):
                await composition_reset(("prefill", "decode"))
                reference, _ = await composition_request(
                    "composition-reference-" + name, item
                )
                references.append(reference)
            stages = (
                ("cold", prompt, references[0], ("prefill", "decode"), True, False),
                ("p-warm-d-cold", prompt, references[0], ("decode",), True, True),
                ("both-warm", prompt, references[0], (), False, True),
                (
                    "branch-prime",
                    prompt,
                    references[0],
                    ("prefill", "decode"),
                    True,
                    False,
                ),
                ("branch", branch, references[1], ("decode",), True, True),
            )
            findings = []
            for name, item, reference, roles, cold_d, warm_p in stages:
                await composition_reset(roles)
                _, evidence = await composition_request(
                    "composition-" + name, item, reference, cold_d, warm_p
                )
                if evidence["status"] == "APC_NOT_EXERCISED":
                    findings.append(evidence["tag"])
            (args.output / "complete.json").write_text(
                json.dumps(
                    {
                        **metadata,
                        "status": "APC_NOT_EXERCISED" if findings else "PENDING_REVIEW",
                        "apc_not_exercised": findings,
                        "requests": len(records),
                        "direct_pd_token_id_checks": len(stages),
                    },
                    indent=2,
                )
                + "\n"
            )
            return

        if args.mode == "smoke":
            text = "Explain this engineering record.\n" + "\n".join(
                f"Record {i}: The service reads a buffer and computes a result."
                for i in range(300)
            )
            response = await post(
                args.prefill, "/tokenize", {"model": args.model, "prompt": text}
            )
            tokens = response.json()["tokens"]
            assert len(tokens) >= 2050
            lengths = (127, 128, 129, 1023, 1024, 1025, 2049, 2050)
            (args.output / "workload.json").write_text(
                json.dumps(
                    {
                        "mode": "smoke",
                        "phase": args.phase,
                        "tokens": tokens,
                        "lengths": lengths,
                        "output_tokens": 16,
                        "scope": f"Two nodes TP{args.tp}/DCP{args.dcp}, clean main MoRIIO READ; no LMCache",
                        "hybrid": args.hybrid,
                        "dcp_block_tokens": 128 * args.dcp,
                        "reference": "producer ordinary request without transfer params",
                        "cache_policy": "reset both roles before each cold request; warm separately",
                    },
                    indent=2,
                )
            )
            await snapshot("before")
            for length in lengths:
                await reset()
                direct = await complete(args.prefill, tokens[:length], 16)
                await reset()
                split = await measured_pd(
                    tokens[:length], 16, f"correctness-{length}", "cold-reset-both"
                )
                # Keep the direct reference even if the comparison fails.
                records[-1]["direct"] = direct
                (args.output / "requests.json").write_text(
                    json.dumps(records, indent=2)
                )
                equal = split["choices"][0]["text"] == direct["choices"][0]["text"]
                ids_equal = (
                    split["choices"][0]["token_ids"]
                    == direct["choices"][0]["token_ids"]
                )
                evidence_path = args.output / f"correctness-{length}-evidence.json"
                evidence = json.loads(evidence_path.read_text())
                evidence["direct_pd_text_equal"] = equal
                evidence["direct_pd_token_ids_equal"] = ids_equal
                if not equal or not ids_equal:
                    evidence["status"] = "FAIL"
                evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
                assert equal and ids_equal, (length, direct, split)
            await reset()
            await measured_pd(tokens[:1025], 16, "warm-prime", "cold-reset-both")
            await measured_pd(tokens[:1025], 16, "warm-replay", "warm-no-reset")
            await snapshot("after")
            (args.output / "complete.json").write_text(
                json.dumps(
                    {
                        "status": "PENDING_REVIEW",
                        "requests": len(records),
                        "direct_pd_text_checks": len(lengths),
                        "direct_pd_token_id_checks": len(lengths),
                        "output_tokens": 16,
                        "natural_accuracy_evaluated": False,
                        "full_c16_evaluated": False,
                        "remote_transfer_requires_log_and_metrics_review": True,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"Native PD smoke complete: {args.output}", flush=True)
            return

        text = "Analyze this engineering record and explain the timing.\n" + "\n".join(
            f"Record {i}: The service processes an input buffer, computes a result, "
            "and reuses unchanged intermediate results."
            for i in range(3000)
        )
        response = await post(
            args.prefill, "/tokenize", {"model": args.model, "prompt": text}
        )
        tokens = response.json()["tokens"]
        assert len(tokens) > 40000
        (args.output / "workload.json").write_text(
            json.dumps(
                {
                    "tokens": tokens,
                    "phase": args.phase,
                    "scope": "Two nodes TP8/DCP8, native MoRIIO READ, no LMCache; diagnostic only, not C16 acceptance",
                }
            )
        )
        await snapshot("before")
        # The first output token remains target-model verified even in the
        # synthetic-acceptance benchmark phase. Longer accuracy needs natural SD.
        for length in (32767, 32768, 32769, 20224):
            await reset()
            direct = await complete(args.prefill, tokens[:length], 1)
            await reset()
            split = await pd(tokens[:length], 1, f"correctness-{length}")
            records[-1]["direct"] = direct
            (args.output / "requests.json").write_text(json.dumps(records, indent=2))
            assert (
                split["choices"][0]["text"] == direct["choices"][0]["text"]
                and split["choices"][0]["token_ids"]
                == direct["choices"][0]["token_ids"]
            ), (length, direct, split)
        await snapshot("correctness")
        for concurrency in (1, 16):
            prompts = [tokens[i * 16 : i * 16 + 1024] for i in range(concurrency)]
            await reset()
            await asyncio.gather(
                *(pd(prompt, 128, f"warmup-c{concurrency}") for prompt in prompts)
            )
            await reset()
            await snapshot(f"timed-c{concurrency}-before")
            await asyncio.gather(
                *(pd(prompt, 128, f"timed-c{concurrency}") for prompt in prompts)
            )
            await snapshot(f"timed-c{concurrency}-after")
        for tag, prompts, count in (
            ("long-prefill", [tokens[:32768]], 16),
            ("decode-c16", [tokens[i * 16 : i * 16 + 1024] for i in range(16)], 128),
        ):
            await reset()
            started = []
            try:
                for url in (args.prefill, args.decode):
                    await post(url, "/start_profile")
                    started.append(url)
                await asyncio.gather(
                    *(pd(prompt, count, f"profile-{tag}") for prompt in prompts)
                )
            finally:
                for url in reversed(started):
                    await post(url, "/stop_profile")
            await snapshot(tag)
        await snapshot("after")
        (args.output / "complete.json").write_text(
            json.dumps(
                {
                    "requests": len(records),
                    "first_token_checks": 4,
                    "natural_accuracy_evaluated": False,
                    "full_c16_evaluated": False,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Native PD diagnostic complete: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("profile", "smoke", "bounded-perf"), default="profile"
    )
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dcp", type=int, default=8)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--cache-composition", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except OutputNotFreshError:
        raise
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.json").write_text(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": repr(exc),
                    "mode": args.mode,
                },
                indent=2,
            )
            + "\n"
        )
        raise
