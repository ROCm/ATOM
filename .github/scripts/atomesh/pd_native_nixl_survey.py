"""Bounded CPU-restored producer -> NIXL consumer survey; externally owned servers.

Requires fresh, exclusive P Multi[NIXL, CPU Offloading] and D NIXL servers.
No launches, fillers, performance measurements, or automatic cross-node PASS.
"""

import argparse
import asyncio
import json
import math
import uuid
from pathlib import Path

import httpx

SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")
ERRORS = (
    "vllm:nixl_num_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total",
    "vllm:nixl_num_kv_expired_reqs_total",
    "vllm:num_preemptions_total",
)
NATIVE = "vllm:kv_offload_total_bytes_total"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def metrics(text):
    result = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        key, value = line.rsplit(" ", 1)
        name = key.split("{", 1)[0]
        if name.endswith(("_total", "_sum", "_count")) and any(
            part in name
            for part in (
                "nixl_",
                "kv_offload_",
                "prompt_tokens_by_source",
                "request_success",
                "num_preemptions",
            )
        ):
            require(key not in result, "Duplicate metric series")
            result[key] = float(value)
            require(math.isfinite(result[key]), "Nonfinite metric")
    return result


def total(values, name, label=""):
    selected = [
        v for k, v in values.items() if k.split("{", 1)[0] == name and label in k
    ]
    return sum(selected) if selected and all(v is not None for v in selected) else None


def deltas(before, after):
    return {
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


def native_bytes(values, direction):
    # Flat counters are lazy on this frozen source; never invent their baseline.
    flat = total(
        values,
        f"vllm:kv_offload_{'load' if direction == 'CPU_to_GPU' else 'store'}_bytes_total",
    )
    compatibility = total(values, NATIVE, f'transfer_type="{direction}"')
    if flat is not None and compatibility is not None:
        require(flat == compatibility, "Native flat/compatibility bytes disagree")
    return flat if flat is not None else compatibility


def check_errors(values):
    for role, series in values.items():
        require(
            all(v is None or v >= 0 for v in series.values()), "Metric counter reset"
        )
        for name in ERRORS:
            value = total(series, name)
            require(value == 0, f"{role}: missing/nonzero {name}")
        for reason in ("error", "abort"):
            require(
                total(
                    series, "vllm:request_success_total", f'finished_reason="{reason}"'
                )
                == 0,
                f"{role}: missing/nonzero request {reason} counter",
            )
        for key, value in series.items():
            if "allocation_failure" in key:
                require(
                    value is None or value == 0, f"{role}: native allocation failure"
                )


def phase_ready(values, name):
    check_errors(values)
    pd = name == "cpu_restore_then_nixl"
    restore = name in ("cpu_restore", "cpu_restore_then_nixl")
    expected = {"P": (1, 0, 1024) if restore else (1025, 0, 0)}
    if pd:
        expected["D"] = None  # NIXL may account external=N before tail replay.
    else:
        expected["D"] = (0, 0, 0)
    ready = True
    for role, source_expected in expected.items():
        series = values[role]
        success = total(series, "vllm:request_success_total")
        count = 1 if role == "P" or pd else 0
        require(success is None or success <= count, f"{role}: non-isolated requests")
        source = tuple(
            total(series, "vllm:prompt_tokens_by_source_total", f'source="{s}"')
            for s in SOURCES
        )
        ready &= success == count
        if any(v is None for v in source):
            ready = False
        elif source_expected is None:
            require(sum(source) <= 1025, "D prompt accounting exceeds N")
            ready &= (
                sum(source) == 1025
                and source[1] == 0
                and source[2] > 0
                and source[0] < 1025
            )
        else:
            require(
                sum(source) <= sum(source_expected),
                f"{role}: prompt accounting exceeds expected",
            )
            ready &= source == source_expected
        byte_count = total(series, "vllm:nixl_bytes_transferred_count")
        byte_sum = total(series, "vllm:nixl_bytes_transferred_sum")
        if role == "D" and pd:
            ready &= (
                byte_count is not None
                and byte_count > 0
                and byte_sum is not None
                and byte_sum > 0
            )
        else:
            require(
                byte_count in (None, 0) and byte_sum in (None, 0),
                f"{role}: unexpected NIXL reception",
            )
            ready &= byte_count == 0 and byte_sum == 0
    load = native_bytes(values["P"], "CPU_to_GPU")
    store = native_bytes(values["P"], "GPU_to_CPU")
    ready &= load is not None and (load > 0 if restore else load == 0)
    ready &= store is not None and (store > 0 if name == "cold_store" else store >= 0)
    return ready


def check_response(response, prompt, count, reference=None):
    require(
        isinstance(response.get("id"), str) and bool(response["id"]),
        "Missing response ID",
    )
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "Expected one choice")
    choice = choices[0]
    require(choice.get("finish_reason") == "length", "Incomplete response")
    prompt_ids = choice.get("prompt_token_ids")
    require(
        isinstance(prompt_ids, list)
        and all(type(t) is int and t >= 0 for t in prompt_ids)
        and prompt_ids == prompt,
        "Prompt IDs changed/missing",
    )
    ids = choice.get("token_ids")
    require(
        isinstance(ids, list)
        and len(ids) == count
        and all(type(t) is int and t >= 0 for t in ids),
        "Generated IDs changed/missing",
    )
    require(isinstance(choice.get("text"), str), "Missing response text")
    require(
        response.get("usage", {}).get("prompt_tokens") == len(prompt),
        "Wrong prompt usage",
    )
    require(
        response["usage"].get("completion_tokens") == count, "Wrong completion usage"
    )
    if reference is not None:
        require(
            ids == reference["choices"][0]["token_ids"]
            and choice["text"] == reference["choices"][0]["text"],
            "Exact IDs/text mismatch",
        )


async def run(args, client, *, poll_interval=1, settle_polls=12, max_polls=20):
    """Run serial HTTP protocol; optional polling controls are for CPU mock tests."""
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "RUNNING",
        "scope": "P native CPU roundtrip then P CPU restore -> D NIXL; not CPU-to-CPU transfer",
        "model": args.model,
        "endpoints": {"P": args.prefill, "D": args.decode},
        "phases": [],
        "resets": [],
        "limits": [
            "Externally launched fresh exclusive servers and topology require launcher audit",
            "Native allocation-failure series is lazy; missing remains unknown, never synthetic zero",
            "No group-resolved DMA, performance, graph, speculation or accuracy claim",
        ],
    }
    sequence = 0

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def post(role, path, body=None, request_id=None):
        nonlocal sequence
        sequence += 1
        record = {"role": role, "path": path, "request_id": request_id, "request": body}
        output = args.output / f"{sequence:03d}-{role}-http.json"
        try:
            response = await client.post(
                result["endpoints"][role] + path,
                json=body,
                headers={"X-Request-Id": request_id or uuid.uuid4().hex},
                timeout=120,
            )
            record.update(status_code=response.status_code, raw_response=response.text)
            response.raise_for_status()
            return response.json()
        finally:
            output.write_text(json.dumps(record, indent=2) + "\n")

    async def snapshot(tag):
        nonlocal sequence
        value = {}
        for role, url in result["endpoints"].items():
            sequence += 1
            response = await client.get(url + "/metrics", timeout=15)
            (args.output / f"{sequence:03d}-{tag}-{role}.prom").write_text(
                response.text
            )
            response.raise_for_status()
            value[role] = metrics(response.text)
        return value

    async def reset(before, tag):
        record = {"tag": tag, "role": "P", "before": before}
        result["resets"].append(record)
        response = await post(
            "P", "/reset_prefix_cache?reset_external=false&reset_running_requests=false"
        )
        record["response"] = response
        save()
        require(
            response.get("success") is True,
            "GPU-only reset failed; no filler or external reset fallback",
        )
        after = before
        for _ in range(settle_polls):
            await asyncio.sleep(poll_interval)
            after = await snapshot(tag)
            delta = deltas(before, after)
            check_errors(delta)
            require(
                all(total(delta[r], "vllm:request_success_total") == 0 for r in delta),
                "Request contamination during reset",
            )
            require(
                all(
                    v is None or v == 0
                    for series in delta.values()
                    for v in series.values()
                ),
                "Delayed counters crossed reset boundary",
            )
        record.update(after=after, deltas=deltas(before, after))
        save()
        return after

    try:
        save()
        tokenized = await post(
            "P",
            "/tokenize",
            {
                "model": args.model,
                "prompt": "Explain the following cache experiment in clear English. "
                "Distinguish local computation, CPU restoration and network delivery.\n"
                + "\n".join(
                    f"Record {i}: A research team stores earlier prompt state in CPU "
                    "memory, clears GPU prefix entries, restores the state, and sends "
                    "it to a second worker. Explain why completed byte counters and "
                    "matching generated tokens provide different kinds of evidence."
                    for i in range(150)
                ),
            },
        )
        tokens = tokenized.get("tokens")
        require(
            isinstance(tokens, list)
            and len(tokens) >= 1025
            and all(type(t) is int and t >= 0 for t in tokens),
            "Invalid tokenizer output",
        )
        prompt = tokens[:1025]
        require(len(set(prompt)) > 8, "Insufficient prompt diversity")
        result["prompt_token_ids"] = prompt
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 32,
            "return_token_ids": True,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "stream": False,
        }
        before = await snapshot("initial")
        check_errors(before)
        for role, series in before.items():
            require(
                total(series, "vllm:request_success_total") == 0,
                f"{role}: server is not fresh",
            )
            require(
                all(
                    total(series, "vllm:prompt_tokens_by_source_total", f'source="{s}"')
                    == 0
                    for s in SOURCES
                ),
                f"{role}: initial prompt counters missing/nonzero",
            )
            require(
                total(series, "vllm:nixl_bytes_transferred_count") == 0
                and total(series, "vllm:nixl_bytes_transferred_sum") == 0,
                f"{role}: initial NIXL counters missing/nonzero",
            )
        require(
            native_bytes(before["P"], "CPU_to_GPU") == 0
            and native_bytes(before["P"], "GPU_to_CPU") == 0,
            "Initial native counters missing/nonzero",
        )
        reference = None
        for name in (
            "cold_store",
            "cpu_restore",
            "cpu_restore_then_nixl",
            "no_load_control",
        ):
            if name != "cold_store":
                before = await reset(before, name + "-reset")
            phase = {
                "name": name,
                "before": before,
                "accounting_checked": False,
                "request_id": "native-nixl-" + uuid.uuid4().hex,
            }
            result["phases"].append(phase)
            save()
            request_id = phase["request_id"]
            if name == "cpu_restore_then_nixl":
                prefill = await post(
                    "P",
                    "/v1/completions",
                    {
                        **body,
                        "max_tokens": 1,
                        "kv_transfer_params": {
                            "do_remote_decode": True,
                            "do_remote_prefill": False,
                            "remote_engine_id": None,
                            "remote_block_ids": None,
                            "remote_host": None,
                            "remote_port": None,
                        },
                    },
                    request_id,
                )
                phase["prefill"] = prefill
                check_response(prefill, prompt, 1)
                transfer = prefill.get("kv_transfer_params")
                require(
                    isinstance(transfer, dict)
                    and transfer.get("do_remote_prefill") is True
                    and all(
                        transfer.get(k)
                        for k in (
                            "remote_engine_id",
                            "remote_block_ids",
                            "remote_host",
                            "remote_port",
                        )
                    ),
                    "Missing NIXL handoff metadata",
                )
                # No metrics, sleep, reset or filler while producer retention is live.
                response = await post(
                    "D",
                    "/v1/completions",
                    {**body, "kv_transfer_params": transfer},
                    request_id,
                )
                require(
                    prefill["choices"][0]["token_ids"]
                    == response.get("choices", [{}])[0].get("token_ids", [])[:1],
                    "Handoff first token differs",
                )
            else:
                payload = (
                    {**body, "kv_transfer_params": {"max_load_tokens": 0}}
                    if name == "no_load_control"
                    else body
                )
                response = await post("P", "/v1/completions", payload, request_id)
            phase["response"] = response
            check_response(response, prompt, 32, reference)
            if reference is None:
                require(
                    len(set(response["choices"][0]["token_ids"])) > 1,
                    "Degenerate single-token repeated reference; no discriminating control",
                )
                reference = response
            save()
            for attempt in range(max_polls):
                await asyncio.sleep(poll_interval)
                after = await snapshot(name + f"-{attempt}")
                delta = deltas(before, after)
                phase.update(after=after, deltas=delta)
                save()
                # Absolute check also catches first appearance of a lazy error counter.
                for series in after.values():
                    require(
                        all(
                            v == 0
                            for k, v in series.items()
                            if "allocation_failure" in k
                        ),
                        "Native allocation failure",
                    )
                ready = phase_ready(delta, name)
                if ready and attempt + 1 >= settle_polls:
                    phase["accounting_checked"] = True
                    break
            require(
                phase["accounting_checked"],
                f"{name}: missing/delayed or incorrect accounting; stopped",
            )
            before = after
            save()
        result["status"] = "PENDING_REVIEW"
        return result
    except asyncio.CancelledError:
        result.update(
            status="FAIL", error="Client cancelled or exceeded total deadline"
        )
        raise
    except Exception as exc:
        result.update(status="FAIL", error=repr(exc))
        try:
            result["failure_metrics"] = await snapshot("failure")
        except (httpx.HTTPError, OSError, ValueError, RuntimeError) as metric_exc:
            result["failure_metrics_error"] = repr(metric_exc)
        raise
    finally:
        save()


async def main(args):
    async with httpx.AsyncClient(trust_env=False) as client:
        # Bounds all polling and requests, including error evidence collection.
        await asyncio.wait_for(run(args, client), timeout=900)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prefill", "decode", "model"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))
