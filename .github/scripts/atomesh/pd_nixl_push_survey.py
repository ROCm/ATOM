"""Bounded dense NIXL PUSH and missing-consumer lease control.

Externally owned P/D: full Qwen3 BF16, TP1/PP1/DCP1, eager, APC off,
NixlPushConnector/UCX, explicit engine IDs, unchanged default lease30s.
No server management, cache reset, retries, or active-DMA cancellation test.
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
    "vllm:num_preemptions_total",
)
EXPIRY = "vllm:nixl_num_kv_expired_reqs_total"
BYTES = "vllm:nixl_bytes_transferred_sum"
COUNT = "vllm:nixl_bytes_transferred_count"
SUCCESS = "vllm:request_success_total"
SOURCE = "vllm:prompt_tokens_by_source_total"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def metrics(text):
    values = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        key, value = line.rsplit(" ", 1)
        name = key.split("{", 1)[0]
        if name.endswith(("_total", "_sum", "_count")) and any(
            part in name
            for part in (
                "nixl_",
                "prompt_tokens_by_source",
                "request_success",
                "num_preemptions",
            )
        ):
            require(key not in values, "Duplicate metric series")
            values[key] = float(value)
            require(math.isfinite(values[key]), "Nonfinite metric")
    return values


def total(values, name, label=""):
    found = [v for k, v in values.items() if k.split("{", 1)[0] == name and label in k]
    return sum(found) if found and all(v is not None for v in found) else None


def differences(before, after):
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


def check_response(response, prompt, count, reference=None):
    require(
        isinstance(response.get("id"), str) and response["id"], "Missing response ID"
    )
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "Expected one choice")
    choice = choices[0]
    for key, expected in (("prompt_token_ids", len(prompt)), ("token_ids", count)):
        ids = choice.get(key)
        require(
            isinstance(ids, list)
            and len(ids) == expected
            and all(type(t) is int and t >= 0 for t in ids),
            f"Invalid {key}",
        )
    require(choice["prompt_token_ids"] == prompt, "Prompt IDs changed")
    require(
        choice.get("finish_reason") == "length" and isinstance(choice.get("text"), str),
        "Incomplete response",
    )
    usage = response.get("usage", {})
    require(
        type(usage.get("prompt_tokens")) is int
        and usage["prompt_tokens"] == len(prompt),
        "Wrong prompt usage",
    )
    require(
        type(usage.get("completion_tokens")) is int
        and usage["completion_tokens"] == count,
        "Wrong completion usage",
    )
    if reference is not None:
        require(
            choice["token_ids"] == reference["choices"][0]["token_ids"]
            and choice["text"] == reference["choices"][0]["text"],
            "Exact IDs/text mismatch",
        )


def accounting(delta, kind, pumps=0):
    """Require complete measured baselines; only initialization tolerates missing bytes."""
    paired = kind in ("warmup_EXCLUDED", "healthy_before", "healthy_after")
    orphan = kind == "missing_consumer"
    ready = True
    for role, values in delta.items():
        require(
            all(v is None or v >= 0 for v in values.values()), "Metric counter reset"
        )
        for name in ERRORS:
            value = total(values, name)
            require(value is None or value == 0, f"{role}: {name} increased")
            ready &= value == 0
        for reason in ("error", "abort"):
            value = total(values, SUCCESS, f'finished_reason="{reason}"')
            require(value is None or value == 0, f"{role}: {reason} request")
            ready &= value == 0
        count = 1 + pumps if role == "P" else int(paired)
        n = 1025 + pumps if role == "P" else 1025 * int(paired)
        success = total(values, SUCCESS)
        require(success is None or success <= count, "Non-isolated request counts")
        ready &= (
            success == count
            and total(values, SUCCESS, 'finished_reason="length"') == count
        )
        source = [total(values, SOURCE, f'source="{s}"') for s in SOURCES]
        expected = [n, 0, 0] if role == "P" else [0, 0, n]
        if all(v is not None for v in source):
            require(sum(source) <= n, "Prompt accounting exceeds expected")
        ready &= source == expected
        expiry = total(values, EXPIRY)
        expected_expiry = int(orphan and role == "P")
        require(expiry is None or expiry <= expected_expiry, "Unexpected lease expiry")
        ready &= expiry == expected_expiry
        for metric in (BYTES, COUNT):
            observed = total(values, metric)
            if kind == "warmup_EXCLUDED" and observed is None:
                # Only absent initialization baselines are excluded, not known activity.
                continue
            positive = paired and role == "P"
            if not positive:
                require(observed is None or observed == 0, "Unexpected completed WRITE")
            if kind == "direct_reference" and observed is None:
                continue
            ready &= observed is not None and (
                observed > 0 if positive else observed == 0
            )
    return ready


async def run(
    args,
    client,
    *,
    poll_interval=1,
    settle_polls=12,
    max_polls=20,
    pump_interval=4,
    max_pumps=10,
):
    """Execute public push protocol; injected polling bounds support HTTP mock tests."""
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "RUNNING",
        "phases": [],
        "model": args.model,
        "endpoints": {"P": args.prefill, "D": args.decode},
        "producer": {
            "engine_id": args.prefill_engine_id,
            "host": args.prefill_kv_host,
            "port": args.prefill_side_channel_port,
        },
        "default_lease_seconds": 30,
        "lifecycle_log_audit": "PENDING_INDEPENDENT_ID_MATCH",
        "limits": [
            "External launcher must certify fresh full-model APC-off TP1 servers and cross-node UCX topology",
            "Expiry requires artifact log match for the orphan internal ID and zero remote workers",
            "No DMA quiescence, active WRITE cancellation, exact allocator reclamation or orphan recovery claim",
        ],
    }
    sequence = 0

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def post(role, path, body, request_id):
        nonlocal sequence
        sequence += 1
        target = args.output / f"{sequence:03d}-{role}-http.json"
        record = {"role": role, "path": path, "request_id": request_id, "request": body}
        try:
            response = await client.post(
                result["endpoints"][role] + path,
                json=body,
                headers={"X-Request-Id": request_id},
                timeout=120,
            )
            record.update(status_code=response.status_code, raw_response=response.text)
            response.raise_for_status()
            return response.json()
        finally:
            target.write_text(json.dumps(record, indent=2) + "\n")

    async def snapshot(tag):
        nonlocal sequence
        output = {}
        for role, url in result["endpoints"].items():
            sequence += 1
            response = await client.get(url + "/metrics", timeout=10)
            (args.output / f"{sequence:03d}-{tag}-{role}.prom").write_text(
                response.text
            )
            response.raise_for_status()
            output[role] = metrics(response.text)
        return output

    async def observe(phase, *, pumps=0, poll=True):
        attempts = max_polls if poll else 1
        ready = False
        for attempt in range(attempts):
            await asyncio.sleep(poll_interval)
            after = await snapshot(phase["name"])
            delta = differences(phase["before"], after)
            phase.update(after=after, deltas=delta)
            # Missing baseline cannot hide a first-observed positive failure.
            for values in after.values():
                for name in ERRORS:
                    value = total(values, name)
                    require(
                        value is None or value == 0,
                        "Absolute failure/preemption counter is nonzero",
                    )
            ready = accounting(delta, phase["name"], pumps)
            save()
            if ready and (not poll or attempt + 1 >= settle_polls):
                break
        if poll:
            require(ready, f"{phase['name']}: missing/delayed or incorrect accounting")
            phase["accounting_checked"] = True
        return after, ready

    def phase(name, before):
        item = {
            "name": name,
            "request_id": f"push-{name}-" + uuid.uuid4().hex,
            "before": before,
            "accounting_checked": False,
        }
        result["phases"].append(item)
        save()
        return item

    def payloads(body, request_id):
        p = {
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
        }
        d = {
            **body,
            "kv_transfer_params": {
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "remote_engine_id": args.prefill_engine_id,
                "remote_host": args.prefill_kv_host,
                "remote_port": args.prefill_side_channel_port,
                "tp_size": 1,
                "pp_size": 1,
                "remote_request_id": request_id,
            },
        }
        return p, d

    async def pair(name, before, body, reference):
        if name != "warmup_EXCLUDED":
            for values in before.values():
                require(
                    all(
                        total(values, metric) is not None
                        for metric in (BYTES, COUNT, EXPIRY, *ERRORS, SUCCESS)
                    ),
                    "Measured baseline has missing metrics",
                )
        item = phase(name, before)
        p, d = payloads(body, item["request_id"])
        responses = await asyncio.gather(
            post("P", "/v1/completions", p, item["request_id"]),
            post("D", "/v1/completions", d, item["request_id"]),
            return_exceptions=True,
        )
        item["responses"] = {
            role: (
                {"error": repr(response)}
                if isinstance(response, BaseException)
                else response
            )
            for role, response in zip(("P", "D"), responses)
        }
        save()
        for response in responses:
            if isinstance(response, BaseException):
                # This is a remote request failure, not an invalid argument type.
                raise RuntimeError(  # noqa: TRY004
                    f"Paired request failed: {response!r}"
                )
        check_response(responses[0], body["prompt"], 1)
        check_response(responses[1], body["prompt"], 32, reference)
        after, _ = await observe(item)
        return after

    try:
        save()
        text = "Explain this cache experiment in clear English.\n" + "\n".join(
            f"Record {i}: A producer computes a prompt and sends cached attention state to a consumer. Compare completed transfers, generated words and missing-consumer lease expiry without confusing network delivery with model accuracy."
            for i in range(150)
        )
        response = await post(
            "P", "/tokenize", {"model": args.model, "prompt": text}, uuid.uuid4().hex
        )
        tokens = response.get("tokens")
        require(
            isinstance(tokens, list)
            and len(tokens) >= 1025
            and all(type(t) is int and t >= 0 for t in tokens),
            "Invalid tokenize response",
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
        for values in before.values():
            require(total(values, SUCCESS) == 0, "Servers must be fresh")
            require(
                all(total(values, SOURCE, f'source="{s}"') == 0 for s in SOURCES),
                "Initial source counters missing/nonzero",
            )
            require(
                all(total(values, name) == 0 for name in (*ERRORS, EXPIRY)),
                "Initial error counters missing/nonzero",
            )
        item = phase("direct_reference", before)
        reference = await post("P", "/v1/completions", body, item["request_id"])
        item["response"] = reference
        check_response(reference, prompt, 32)
        require(
            len(set(reference["choices"][0]["token_ids"])) > 1,
            "Degenerate repeated-token reference",
        )
        before, _ = await observe(item)
        before = await pair("warmup_EXCLUDED", before, body, reference)
        before = await pair("healthy_before", before, body, reference)
        item = phase("missing_consumer", before)
        item["request_id"] = "push-orphan-" + uuid.uuid4().hex
        item.update(
            consumer_request_submitted=False,
            pumps=[],
            log_match_required="P internal request ID: releasing expired KV blocks, retrieved by 0 remote workers",
        )
        p, _ = payloads(body, item["request_id"])
        item["response"] = await post("P", "/v1/completions", p, item["request_id"])
        check_response(item["response"], prompt, 1)
        save()
        async with asyncio.timeout(60):
            for index in range(max_pumps):
                await asyncio.sleep(pump_interval)
                pump_before = await snapshot("pump-before")
                pump_id = "push-pump-" + uuid.uuid4().hex
                pump = {"request_id": pump_id, "before": pump_before}
                item["pumps"].append(pump)
                pump["response"] = await post(
                    "P",
                    "/v1/completions",
                    {**body, "prompt": prompt[:1], "max_tokens": 1},
                    pump_id,
                )
                check_response(pump["response"], prompt[:1], 1)
                after, ready = await observe(item, pumps=index + 1, poll=False)
                pump.update(
                    after=after,
                    deltas=differences(pump_before, after),
                    attribution="Ordinary P pump; delayed orphan stats may share this window",
                )
                save()
                if ready:
                    break
            before, _ = await observe(item, pumps=len(item["pumps"]))
        before = await pair("healthy_after", before, body, reference)
        result.update(status="PENDING_REVIEW", final_metrics=before)
        return result
    except asyncio.CancelledError:
        result.update(status="FAIL", error="Cancelled or total deadline exceeded")
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
        await asyncio.wait_for(run(args, client), timeout=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "prefill",
        "decode",
        "model",
        "prefill-engine-id",
        "prefill-kv-host",
    ):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--prefill-side-channel-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(1 <= args.prefill_side_channel_port <= 65535, "Invalid side-channel port")
    asyncio.run(main(args))
