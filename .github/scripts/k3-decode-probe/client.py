"""Exercise full-model D generation with locally computed prompt KV."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import aiohttp
import msgpack
import zmq
import zmq.asyncio


async def post(session, port, route, body, request_id):
    async with session.post(
        f"http://127.0.0.1:{port}/{route}",
        json=body,
        headers={"X-Request-Id": request_id},
    ) as response:
        value = await response.json()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {value}")
        return value


async def prepare(session, args):
    lengths = [int(n) for n in args.lengths.split(",")]
    filler = (
        "The library opens at nine in the morning and closes at six in the "
        "evening. Books are sorted by author and subject. Visitors may borrow "
        "books for two weeks and return them at the front desk.\n"
    )
    filler_response = await post(
        session,
        args.port,
        "tokenize",
        {
            "model": "Kimi-K3",
            "prompt": filler * 800,
            "add_special_tokens": False,
        },
        uuid.uuid4().hex,
    )
    filler_tokens = filler_response["tokens"]
    rows = []
    for i in range(args.n):
        length = lengths[i % len(lengths)]
        start = f"Record {i}. Read the background and answer the final question.\n"
        end = (
            f"\nQuestion: A shop has {i + 12} boxes with 7 pencils in each box. "
            "It sells 5 pencils. How many pencils are left?\nAnswer:"
        )
        encoded = []
        for text in (start, end):
            response = await post(
                session,
                args.port,
                "tokenize",
                {
                    "model": "Kimi-K3",
                    "prompt": text,
                    "add_special_tokens": False,
                },
                uuid.uuid4().hex,
            )
            encoded.append(response["tokens"])
        a, c = encoded
        needed = length - len(a) - len(c)
        repeated = filler_tokens * ((needed + len(filler_tokens) - 1) // len(filler_tokens))
        prompt = a + repeated[:needed] + c
        assert len(prompt) == length
        rows.append({"case": i, "prompt": prompt, "expected_answer": (i + 12) * 7 - 5})
    args.prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Prepared {len(rows)} prompts in {args.prompts}", flush=True)


async def run(session, args):
    cases = [json.loads(line) for line in args.prompts.open()]
    if args.n:
        cases = cases[: args.n]
    if args.cases:
        selected = {int(n) for n in args.cases.split(",")}
        cases = [r for r in cases if r["case"] in selected]
    reference = {}
    if args.reference:
        reference = {
            r["case"]: r
            for r in (json.loads(line) for line in args.reference.open())
            if r.get("ok") and "response" in r
        }
    sem = asyncio.Semaphore(args.concurrency)
    ctx = zmq.asyncio.Context()
    sockets = {}
    results = []

    async def release(kv):
        # One ACK to every producer rank for this homogeneous TP simulation.
        tp = int(kv["tp_size"])
        dp = int(kv.get("remote_dp_rank", 0))
        local_dp_size = int(kv.get("remote_dp_size_local", 1))
        local_dp = dp % local_dp_size if local_dp_size else dp
        for rank in range(tp):
            port = int(kv["remote_notify_port"]) + local_dp * tp + rank
            address = f"tcp://{kv['remote_host']}:{port}"
            if address not in sockets:
                sock = ctx.socket(zmq.DEALER)
                sock.setsockopt(zmq.LINGER, 2000)
                sock.setsockopt(zmq.SNDTIMEO, 5000)
                sock.connect(address)
                sockets[address] = sock
            await sockets[address].send(
                msgpack.dumps(
                    {
                        "type": "release",
                        "transfer_id": kv["transfer_id"],
                        "consumer_tp_size": tp,
                    }
                )
            )

    async def one(case, iteration):
        async with sem:
            row = {
                "case": case["case"],
                "iteration": iteration,
                "input_tokens": len(case["prompt"]),
                "mode": args.mode,
                "request_id": f"seq-{uuid.uuid4().hex}",
            }
            body = {
                "model": "Kimi-K3",
                "prompt": case["prompt"],
                "temperature": 0,
                "seed": 1234,
                "ignore_eos": True,
                "max_tokens": args.max_tokens,
                "return_token_ids": True,
            }
            if args.logprobs is not None:
                body["logprobs"] = args.logprobs
            if args.cache_salt:
                body["cache_salt"] = args.cache_salt
            if args.mode == "prefill":
                body["max_tokens"] = 1
                body["kv_transfer_params"] = {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_tp_size": args.tp,
                    "remote_dp_size": 1,
                    "transfer_id": f"transfer-{uuid.uuid4()}",
                }
            start = time.monotonic()
            try:
                value = await post(
                    session,
                    args.port,
                    "v1/completions",
                    body,
                    row["request_id"],
                )
                row.update(seconds=time.monotonic() - start, response=value)
                if args.mode == "prefill":
                    kv = value.get("kv_transfer_params")
                    if not kv or not kv.get("remote_block_ids"):
                        raise RuntimeError("P did not export block IDs")
                    await release(kv)
                    row["simulated_release_sent"] = True
                json.dumps(value, allow_nan=False)
                row["ok"] = value["usage"]["completion_tokens"] == body["max_tokens"]
                if args.mode == "generate":
                    row["expected_answer"] = case["expected_answer"]
                    row["starts_with_expected_answer"] = (
                        value["choices"][0]["text"]
                        .lstrip()
                        .startswith(str(case["expected_answer"]))
                    )
                if case["case"] in reference:
                    ref = reference[case["case"]]["response"]["choices"][0]
                    choice = value["choices"][0]
                    a, b = choice["token_ids"], ref["token_ids"]
                    row["token_ids_equal"] = a == b
                    row["first_token_equal"] = a[:1] == b[:1]
                    row["common_prefix_tokens"] = next(
                        (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                        min(len(a), len(b)),
                    )
            except (
                aiohttp.ClientError,
                TimeoutError,
                RuntimeError,
                KeyError,
                ValueError,
                TypeError,
                zmq.ZMQError,
            ) as exc:
                row.update(
                    ok=False,
                    seconds=time.monotonic() - start,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(row)
            with args.out.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "response"}),
                flush=True,
            )

    async def metrics(suffix):
        async with session.get(f"http://127.0.0.1:{args.port}/metrics") as response:
            response.raise_for_status()
            args.out.with_suffix(f".{suffix}.metrics").write_text(await response.text())

    await metrics("before")
    start = time.monotonic()
    try:
        await asyncio.gather(
            *(
                one(case, iteration)
                for iteration in range(args.repeat)
                for case in cases
            )
        )
        await asyncio.sleep(1)
        await metrics("after")
    finally:
        for sock in sockets.values():
            sock.close()
        ctx.term()
    summary = {
        "requests": len(results),
        "ok": sum(r["ok"] for r in results),
        "seconds": time.monotonic() - start,
    }
    if reference:
        summary.update(
            comparable_requests=sum("token_ids_equal" in r for r in results),
            token_matches=sum(r.get("token_ids_equal", False) for r in results),
            first_token_matches=sum(r.get("first_token_equal", False) for r in results),
        )
    print(json.dumps(summary), flush=True)
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if summary["ok"] != summary["requests"]:
        raise SystemExit(1)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["prepare", "prefill", "generate"], required=True
    )
    parser.add_argument("--port", type=int, default=2584)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cache-salt")
    parser.add_argument("--lengths", default="12289,16385,24577")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--cases")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--logprobs", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.mode != "prepare" and args.out is None:
        parser.error("--out is required for prefill/generate")
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        if args.mode == "prepare":
            await prepare(session, args)
        else:
            await run(session, args)


if __name__ == "__main__":
    asyncio.run(main())
