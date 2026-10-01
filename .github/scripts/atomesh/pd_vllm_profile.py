"""Bounded native PD diagnostic; all profiling controls use direct endpoints."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:

        async def post(url, path, body=None, headers=None):
            response = await client.post(url + path, json=body, headers=headers)
            response.raise_for_status()
            return response

        async def snapshot(tag):
            for role in ("prefill", "decode"):
                response = await client.get(getattr(args, role) + "/metrics")
                response.raise_for_status()
                (args.output / f"{tag}-{role}.metrics.txt").write_text(response.text)

        async def reset():
            await asyncio.gather(
                *(
                    post(url, "/reset_prefix_cache")
                    for url in (args.prefill, args.decode)
                )
            )

        async def complete(url, prompt, count, params=None, request_id=None):
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
                    "remote_tp_size": 8,
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
                "prompt_tokens": len(prompt),
                "prefill_ms": 1000 * (after_prefill - start),
                "request_id": request_id,
                "prefill": prefill,
            }
            records.append(record)
            (args.output / "requests.json").write_text(json.dumps(records, indent=2))
            decode = await complete(
                args.decode,
                prompt,
                count,
                {
                    **transfer,
                    "remote_tp_size": 8,
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
        comparisons = []
        for length in (32767, 32768, 32769, 20224):
            await reset()
            direct = await complete(args.decode, tokens[:length], 1)
            await reset()
            await snapshot(f"correctness-{length}-before")
            split = await pd(tokens[:length], 1, f"correctness-{length}")
            await snapshot(f"correctness-{length}-after")
            comparisons.append({"length": length, "direct": direct, "split": split})
            (args.output / "comparisons.json").write_text(
                json.dumps(comparisons, indent=2)
            )
            assert split["choices"][0]["text"] == direct["choices"][0]["text"], (
                length,
                direct,
                split,
            )
        await snapshot("correctness")
        await snapshot("after")
        (args.output / "complete.json").write_text(
            json.dumps(
                {
                    "requests": len(records),
                    "first_token_checks": 4,
                    "natural_accuracy_evaluated": False,
                    "full_c16_evaluated": False,
                    "zeroing_audit": "Check all rank logs for schedule exclusions, zero_kernel, read_post, and read_done",
                    "full_graph_evaluated": False,
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
    asyncio.run(run(parser.parse_args()))
