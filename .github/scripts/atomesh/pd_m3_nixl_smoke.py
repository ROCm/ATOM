"""Cold NIXL pull evidence; default M3 or explicit V4 profile, never automatic PASS.

The historical module name remains for existing harness/import compatibility.
"""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import httpx


def metrics(text):
    # Preserve all label sets (including any future group labels), not just sums.
    return {
        line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith(
            (
                "vllm:nixl_",
                "vllm:prompt_tokens_by_source_total",
                "vllm:request_success_total",
            )
        )
    }


def total(values, name, label=""):
    matches = [
        v for k, v in values.items() if k.split("{", 1)[0] == name and label in k
    ]
    return sum(matches) if matches and all(v is not None for v in matches) else None


def metric_deltas(before, after):
    return {
        role: {
            key: (
                values[key] - before[role][key]
                if key in values and key in before[role]
                else None
            )
            for key in values.keys() | before[role].keys()
        }
        for role, values in after.items()
    }


SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")
FAILURES = (
    "vllm:nixl_num_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total",
    "vllm:nixl_num_kv_expired_reqs_total",
)


def accounting_ready(deltas, length, pd):
    ready = True
    for role in (("prefill", "decode") if pd else ("prefill",)):
        values = deltas[role]
        assert all(v is None or v >= 0 for v in values.values()), "Metric counter reset"
        failures = [total(values, name) for name in FAILURES]
        assert all(v is None or v == 0 for v in failures), (role, failures)
        success = total(values, "vllm:request_success_total")
        assert success is None or success <= 1, "Non-isolated request counters"
        sources = [
            total(values, "vllm:prompt_tokens_by_source_total", f'source="{s}"')
            for s in SOURCES
        ]
        if success != 1 or any(v is None for v in sources):
            ready = False
            continue
        local, cache, external = sources
        assert cache == 0, "Unexpected APC hit"
        assert sum(sources) <= length, "Prompt token accounting exceeds N"
        if sum(sources) != length:
            ready = False
            continue
        if role == "prefill":
            assert (
                local == length and external == 0
            ), "Producer did not compute full prompt"
        else:
            # NIXL stats can report external=N/local=0 before scheduler tail replay.
            assert external > 0 and local < length, "No external KV consumption"
            byte_sum = total(values, "vllm:nixl_bytes_transferred_sum")
            byte_count = total(values, "vllm:nixl_bytes_transferred_count")
            ready &= (
                byte_sum is not None
                and byte_sum > 0
                and byte_count is not None
                and byte_count > 0
            )
        ready &= all(v is not None for v in failures)
    return ready


def check_usage(response, length, count):
    assert response["choices"][0]["finish_reason"] == "length", response
    assert response["usage"]["prompt_tokens"] == length, response
    assert response["usage"]["completion_tokens"] == count, response


def check_token_ids(response, prompt, count):
    choice = response["choices"][0]
    assert choice.get("prompt_token_ids") == prompt, "Prompt IDs changed or missing"
    ids = choice.get("token_ids")
    assert isinstance(ids, list) and len(ids) == count, "Generated IDs missing/invalid"
    assert all(type(token) is int for token in ids), "Generated IDs must be integers"


async def run(args):
    profile = getattr(args, "model_profile", "m3")
    lengths = {"m3": (127, 129, 513), "v4": (255, 257, 513)}[profile]
    exact_ids = profile == "v4"
    args.output.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:

        async def post(url, path, body, request_id):
            response = await client.post(
                url + path, json=body, headers={"X-Request-Id": request_id}
            )
            response.raise_for_status()
            return response.json()

        async def snapshot(tag):
            result = {}
            for role in ("prefill", "decode"):
                response = await client.get(getattr(args, role) + "/metrics")
                response.raise_for_status()
                (args.output / f"{tag}-{role}.prom").write_text(response.text)
                result[role] = metrics(response.text)
            return result

        async def await_accounting(before, tag, length, pd, evidence, key):
            for attempt in range(16):
                after = await snapshot(f"{tag}-{attempt}")
                deltas = metric_deltas(before, after)
                evidence[key] = deltas
                evidence[key + "_unknown_series"] = {
                    role: [name for name, value in values.items() if value is None]
                    for role, values in deltas.items()
                }
                if accounting_ready(deltas, length, pd):
                    return after
                await asyncio.sleep(1)
            evidence["accounting_checked"] = False
            evidence["review_reason"] = (
                f"{tag}: missing/delayed metrics; isolation unconfirmed; stopped subsequent requests"
            )
            (args.output / "pending.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
            return None

        tokenized = await post(
            args.prefill,
            "/tokenize",
            {"model": args.model, "prompt": "Explain this engineering record. " * 600},
            uuid.uuid4().hex,
        )
        tokens = tokenized["tokens"]
        assert len(tokens) >= 513
        for length in lengths:
            request_id = f"{profile}-nixl-" + uuid.uuid4().hex
            evidence = {
                "status": "PENDING_REVIEW",
                "request_id": request_id,
                "prompt_tokens": length,
                "producer_prompt_tokens_expected": length,
                "cache_policy": "APC disabled; serial cold requests; no warm claim",
                "group_transfer_bytes": "UNKNOWN: upstream metrics are per-engine, not per-cache-group",
                "protocol": "upstream nixl_integration/toy_proxy_server.py pull handoff",
            }
            path = args.output / f"request-{length}.json"
            body = {
                "model": args.model,
                "prompt": tokens[:length],
                "max_tokens": 16,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "stream": False,
            }
            if exact_ids:
                body["return_token_ids"] = True
            before = None
            try:
                # Flush reference metrics before measuring PD, or stop the sequence.
                before = await snapshot(f"{length}-reference-before")
                evidence["reference"] = await post(
                    args.prefill, "/v1/completions", body, request_id + "-reference"
                )
                check_usage(evidence["reference"], length, 16)
                if exact_ids:
                    check_token_ids(evidence["reference"], body["prompt"], 16)
                before = await await_accounting(
                    before,
                    f"{length}-reference-after",
                    length,
                    False,
                    evidence,
                    "reference_metric_deltas",
                )
                if before is None:
                    return
                # The flushed snapshot is the PD baseline; no intervening workload.
                evidence["pd_baseline"] = before
                params = {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                }
                prefill = await post(
                    args.prefill,
                    "/v1/completions",
                    {**body, "max_tokens": 1, "kv_transfer_params": params},
                    request_id,
                )
                evidence["prefill"] = prefill
                path.write_text(json.dumps(evidence, indent=2) + "\n")
                transfer = prefill["kv_transfer_params"]
                assert transfer.get("do_remote_prefill") is True, transfer
                assert all(
                    transfer.get(k)
                    for k in (
                        "remote_engine_id",
                        "remote_block_ids",
                        "remote_host",
                        "remote_port",
                    )
                ), transfer
                evidence["remote_block_groups"] = transfer["remote_block_ids"]
                # Forward returned metadata unchanged, exactly as upstream toy proxy.
                decode = await post(
                    args.decode,
                    "/v1/completions",
                    {**body, "kv_transfer_params": transfer},
                    request_id,
                )
                evidence["decode"] = decode
                check_usage(prefill, length, 1)
                check_usage(decode, length, 16)
                if exact_ids:
                    check_token_ids(prefill, body["prompt"], 1)
                    check_token_ids(decode, body["prompt"], 16)
                    evidence["generated_token_ids_equal"] = (
                        decode["choices"][0]["token_ids"]
                        == evidence["reference"]["choices"][0]["token_ids"]
                    )
                    assert evidence[
                        "generated_token_ids_equal"
                    ], "Direct/PD token ID mismatch"
                after = await await_accounting(
                    before,
                    f"{length}-after",
                    length,
                    True,
                    evidence,
                    "labeled_metric_deltas",
                )
                if after is None:
                    return
                evidence["accounting_checked"] = True
                d = evidence["labeled_metric_deltas"]["decode"]
                evidence["external_tokens"] = total(
                    d,
                    "vllm:prompt_tokens_by_source_total",
                    'source="external_kv_transfer"',
                )
                evidence["nixl_bytes"] = total(d, "vllm:nixl_bytes_transferred_sum")
                evidence["direct_pd_text_equal"] = (
                    decode["choices"][0]["text"]
                    == evidence["reference"]["choices"][0]["text"]
                )
                assert evidence["direct_pd_text_equal"], "Direct/PD text mismatch"
            except Exception as exc:
                evidence["status"] = "FAIL"
                evidence["error"] = repr(exc)
                if before is not None:
                    try:
                        evidence["failure_metrics"] = await snapshot(
                            f"{length}-failure"
                        )
                    except (httpx.HTTPError, OSError, ValueError) as metric_exc:
                        evidence["metrics_error"] = repr(metric_exc)
                raise
            finally:
                path.write_text(json.dumps(evidence, indent=2) + "\n")
        (args.output / "complete.json").write_text(
            json.dumps(
                {
                    "status": "PENDING_REVIEW",
                    "requests": 3,
                    "lengths": list(lengths),
                    "output_tokens": 16,
                    "review_required": "Audit all cache groups, bytes, completion, failures and delayed P counters; no accuracy claim",
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill", "decode", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-profile", choices=("m3", "v4"), default="m3")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.json").write_text(
            json.dumps({"status": "FAIL", "error": repr(exc)}) + "\n"
        )
        raise
