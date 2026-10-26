# Concurrency correctness probe: N distinct greedy chat requests at once; the share whose answer
# mentions Paris.   paris.py <N,...>
import asyncio
import resource
import sys
import time

import aiohttp

resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))  # 2048 sockets

import os

URL = "http://localhost:8000/v1/chat/completions"
PAD = "Please read carefully. " * int(
    os.environ.get("PAD", "0")
)  # ~5 tokens each: longer prompts
MODEL = "/data/Qwen3.5-35B-A3B-MXFP4"


async def one(sess, i):
    body = {
        "model": MODEL,
        "max_tokens": 48,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": f"Request {i}: {PAD}in one short sentence, what is the capital of France?",
            }
        ],
    }
    async with sess.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=900)) as r:
        d = await r.json()
        return d["choices"][0]["message"].get("content") or ""


async def wave(n):
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as sess:
        t = time.time()
        outs = await asyncio.gather(
            *[one(sess, i) for i in range(n)], return_exceptions=True
        )
    ok = sum(1 for o in outs if isinstance(o, str) and "Paris" in o)
    badi = [i for i, o in enumerate(outs) if not (isinstance(o, str) and "Paris" in o)]
    where = ""
    if badi:
        q = [
            sum(1 for i in badi if k * n // 4 <= i < (k + 1) * n // 4) for k in range(4)
        ]
        where = f"; failed indices by quarter {q}, first {badi[:6]}; e.g. {str(outs[badi[0]])[:80]!r}"
    print(
        f"N={n:5d}: {ok}/{n} mention Paris ({time.time() - t:.0f} s){where}", flush=True
    )


for n in [int(x) for x in sys.argv[1].split(",")]:
    asyncio.run(wave(n))
