"""Hold P source KV with a delayed mock ACK, then expect fatal missing-ACK expiry."""

import argparse
import asyncio
import json
import math
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import msgpack
import zmq
import zmq.asyncio
from client import post, prepare


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


async def metrics(session, port, path):
    async with session.get(f"http://127.0.0.1:{port}/metrics") as response:
        response.raise_for_status()
        text = await response.text()
    path.write_text(text)
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("vllm:kv_cache_usage_perc{")
    ]
    assert values and all(math.isfinite(v) and 0 <= v <= 1 for v in values), text
    return sum(values)


async def reset(session, port):
    for _ in range(120):
        value = await post(session, port, "reset_prefix_cache", {}, uuid.uuid4().hex)
        if value.get("success"):
            return
        await asyncio.sleep(1)
    raise RuntimeError("P prefix cache remains held after mock ACK")


async def export(session, port, case, out):
    request_id, transfer_id = f"lease-{uuid.uuid4().hex}", f"transfer-{uuid.uuid4()}"
    started = time.monotonic()
    value = await post(
        session,
        port,
        "v1/completions",
        {
            "model": "Kimi-K3",
            "prompt": case["prompt"],
            "temperature": 0,
            "seed": 1234,
            "ignore_eos": True,
            "max_tokens": 1,
            "return_token_ids": True,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_tp_size": 8,
                "remote_dp_size": 1,
                "transfer_id": transfer_id,
            },
        },
        request_id,
    )
    assert value["usage"]["completion_tokens"] == 1
    kv = value["kv_transfer_params"]
    assert kv["transfer_id"] == transfer_id and kv["remote_block_ids"]
    assert int(kv["tp_size"]) == 8
    write(
        out,
        {
            "request_id": request_id,
            "transfer_id": transfer_id,
            "input_tokens": len(case["prompt"]),
            "response": value,
            "request_seconds": time.monotonic() - started,
            "exported_at": time.time(),
        },
    )
    return kv


async def mock_release(kv):
    context = zmq.asyncio.Context()
    sockets = []
    try:
        tp = int(kv["tp_size"])
        dp = int(kv.get("remote_dp_rank", 0))
        local_size = int(kv.get("remote_dp_size_local", 1))
        local_dp = dp % local_size if local_size else dp
        for rank in range(tp):
            socket = context.socket(zmq.DEALER)
            sockets.append(socket)
            socket.setsockopt(zmq.LINGER, 2000)
            socket.setsockopt(zmq.SNDTIMEO, 5000)
            port = int(kv["remote_notify_port"]) + local_dp * tp + rank
            socket.connect(f"tcp://{kv['remote_host']}:{port}")
            await socket.send(
                msgpack.dumps(
                    {
                        "type": "release",
                        "transfer_id": kv["transfer_id"],
                        "consumer_tp_size": tp,
                    }
                )
            )
    finally:
        for socket in sockets:
            socket.close()
        context.term()


async def run(args):
    args.results.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        prompts = args.results / "prompts.jsonl"
        await prepare(
            session,
            SimpleNamespace(port=args.port, prompts=prompts, n=2, lengths="65537"),
        )
        cases = [json.loads(line) for line in prompts.read_text().splitlines()]
        await reset(session, args.port)
        kv = await export(
            session, args.port, cases[0], args.results / "delayed-export.json"
        )
        print("P-PROBE holding source KV for 70 seconds before mock ACK", flush=True)
        hold_start = time.monotonic()
        await asyncio.sleep(70)
        async with session.get(f"http://127.0.0.1:{args.port}/health") as response:
            response.raise_for_status()
        usage = await metrics(
            session, args.port, args.results / "held-before-ack.metrics"
        )
        assert usage > 0, "source KV was freed before the delayed mock ACK"
        held_seconds = time.monotonic() - hold_start
        await mock_release(kv)
        await reset(session, args.port)
        for _ in range(30):
            usage_after = await metrics(
                session, args.port, args.results / "released-after-ack.metrics"
            )
            if usage_after == 0:
                break
            await asyncio.sleep(1)
        assert usage_after == 0, "source KV remains allocated after mock ACK and reset"
        write(
            args.results / "delayed-ack-complete.json",
            {
                "transfer_id": kv["transfer_id"],
                "held_seconds": held_seconds,
                "kv_usage_before_ack": usage,
                "kv_usage_after_ack": usage_after,
                "mock_ack_ranks": 8,
                "scope": "Mock ACK only; no consumer RDMA READ",
            },
        )
        print("P-PROBE delayed mock ACK released source KV", flush=True)
        kv = await export(
            session, args.port, cases[1], args.results / "expiry-export.json"
        )
        write(
            args.results / "expiry-armed.json",
            {
                "transfer_id": kv["transfer_id"],
                "lease_seconds": 180,
                "ack_sent": False,
                "armed_at": time.time(),
            },
        )
        print("P-PROBE withholding ACK; expecting fatal lease expiry", flush=True)
        await asyncio.sleep(240)
        raise RuntimeError(
            "Missing-ACK lease did not terminate the server within 240 seconds"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--results", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
