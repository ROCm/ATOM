"""Bounded native READ smoke, adapted from survey commit 6ccefa405e1a."""

import argparse
import asyncio
import json
import re
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
    return {key: value - before.get(key, 0) for key, value in after.items()}


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:

        async def post(url, path, body=None, headers=None):
            response = await client.post(url + path, json=body, headers=headers)
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
                assert result.get("success") is True, (
                    f"{role} cache reset failed: {result}"
                )

        async def complete(url, prompt, count, params=None, request_id=None):
            metrics = await client.get(url + "/metrics")
            metrics.raise_for_status()
            successes = token_counters(metrics.text).get("request_success", 0)
            body = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": count,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "stream": False,
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
            assert result.get("choices"), result
            for attempt in range(12):
                metrics = await client.get(url + "/metrics")
                metrics.raise_for_status()
                if token_counters(metrics.text).get("request_success", 0) > successes:
                    break
                await asyncio.sleep(0.5)
            else:
                raise AssertionError("Request completion metrics did not drain")
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
            assert transfer["remote_block_ids"] and transfer["remote_host"], transfer
            record = {
                "tag": tag,
                "request_id": request_id,
                "prompt_tokens": len(prompt),
                "prefill": prefill,
                "prefill_ms": 1000 * (after_prefill - start),
            }
            records.append(record)
            (args.output / "requests.json").write_text(json.dumps(records, indent=2))
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
            record.update(
                {
                    "decode_ms": 1000 * (time.perf_counter() - after_prefill),
                    "decode": decode,
                }
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
                    if (
                        cache_state == "cold-reset-both"
                        and len(prompt) > 128 * args.dcp + 1
                    ):
                        assert d["external_kv_transfer"] > 0, (
                            "No external KV token consumption"
                        )
                    elif d["external_kv_transfer"] == 0:
                        evidence["review_reason"] = (
                            "Warm APC may hide transfer; not remote-transfer proof"
                        )
                    assert d["local_compute"] >= 1, (
                        "Missing READ protocol tail computation"
                    )
                    assert sum(d[k] for k in required - {"request_success"}) == len(
                        prompt
                    ), "Prompt token accounting mismatch"
                    evidence["accounting_checked"] = True
                else:
                    raise AssertionError("Missing/delayed or non-isolated metrics")
                assert evidence["finish_reason"] == "length", (
                    "Incomplete decode response"
                )
                evidence["status"] = "PASS_ACCOUNTING"
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

        text = "Explain this engineering record.\n" + "\n".join(
            f"Record {i}: The service reads a buffer and computes a result."
            for i in range(300)
        )
        response = await post(
            args.prefill, "/tokenize", {"model": args.model, "prompt": text}
        )
        tokens = response.json()["tokens"]
        # K3 producer omits its last token; sub-block prompts may legitimately
        # export no KV. Cross the second DCP block while retaining a full first.
        lengths = (1026, 2047, 2048, 2049, 2050)
        assert len(tokens) >= max(lengths) + 64
        (args.output / "workload.json").write_text(
            json.dumps(
                {
                    "tokens": tokens,
                    "lengths": lengths,
                    "phase": args.phase,
                    "output_tokens": 16,
                    "tp": args.tp,
                    "dcp": args.dcp,
                    "scope": "direct MoRI READ; no LMCache or multi-owner GPU coverage",
                    "concurrent_lanes": 4,
                    "concurrent_waves": 2,
                },
                indent=2,
            )
        )
        await snapshot("before")
        for length in lengths:
            await reset()
            direct = await complete(args.decode, tokens[:length], 16)
            await reset()
            split = await measured_pd(
                tokens[:length], 16, f"cold-{length}", "cold-reset-both"
            )
            records[-1]["direct"] = direct
            (args.output / "requests.json").write_text(json.dumps(records, indent=2))
            assert split["choices"][0]["text"] == direct["choices"][0]["text"], (
                length,
                direct,
                split,
            )
        await reset()
        prime = await measured_pd(tokens[:2050], 16, "warm-prime", "cold-reset-both")
        warm = await measured_pd(tokens[:2050], 16, "warm-replay", "warm-no-reset")
        assert warm["choices"][0]["text"] == prime["choices"][0]["text"]

        # Two joined waves, with independent references and an intervening reset.
        # This exercises post-completion reuse, not proof of physical block reuse.
        prompts = [tokens[i * 16 : i * 16 + 2050] for i in range(4)]
        references = [await complete(args.decode, p, 32) for p in prompts]
        waves = []
        for wave in range(2):
            await reset()
            before = await snapshot(f"reuse-{wave}-before")
            replies = await asyncio.gather(
                *(
                    pd(prompt, 32, f"reuse-{wave}-lane-{lane}")
                    for lane, prompt in enumerate(prompts)
                )
            )
            for reference, reply in zip(references, replies):
                assert reply["choices"][0]["finish_reason"] == "length"
                assert reply["choices"][0]["text"] == reference["choices"][0]["text"]
            for attempt in range(12):
                after = await snapshot(f"reuse-{wave}-after")
                delta = counter_delta(before["decode"], after["decode"])
                if delta.get("request_success", 0) >= 4:
                    break
                await asyncio.sleep(0.5)
            waves.append(
                {
                    "wave": wave,
                    "decode_delta": delta,
                    "direct_pd_text_checks": len(replies),
                }
            )
            (args.output / "reuse.json").write_text(json.dumps(waves, indent=2))
            assert delta.get("request_success") == 4
            assert delta.get("external_kv_transfer", 0) > 0
        await snapshot("after")
        (args.output / "complete.json").write_text(
            json.dumps(
                {
                    "status": "PASS_SMOKE_PENDING_RUNTIME_REVIEW",
                    "requests": len(records),
                    "direct_pd_text_checks": len(lengths) + 8,
                    "warm_text_checks": 1,
                    "request_errors": 0,
                    "retries": 0,
                    "natural_accuracy_evaluated": False,
                    "full_replay_sync_read_proven": False,
                    "physical_block_reuse_proven": False,
                    "runtime_review": "Correlate FULL dispatch/replay and MoRI READ wait/completion "
                    "in both rank logs; capture success is not replay evidence.",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Native PD smoke complete: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dcp", type=int, default=8)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.json").write_text(
            json.dumps({"status": "FAIL", "error": repr(exc), "retries": 0}, indent=2)
            + "\n"
        )
        raise
