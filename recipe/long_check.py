"""Greedy long-context needle retrieval (milestone6 copy of /shared_nfs/ygan/max/bin/long_check.py, two fixes).

usage: long_check.py <port> <out.json> [tokens ...]   (default 20000 80000 130000)
env:   NEEDLE_TIMEOUT_S  hard wall-clock limit per needle request (default 180; past gates took 0.9-10.6 s per needle); a request over it is a miss
       ("error": "timeout ...") and the next needle starts, so a hung server fails the gate fast.
Prompts of ~N tokens: numbered records with a needle, ask for it back. A request past 512 index blocks
(~65K tokens) takes the indexer context-parallel scoring path. Prints "LONG hits=h/n".
milestone6 fixes:
 1. seed >= 7 bug: the needle index was n * (3 + seed) // 10, i.e. >= n (a vault that is not in the prompt) for
    seed >= 7. The position now uses p = seed % 7 (< 7): needle = n * (3 + p) // 10 <= 0.9 n. Seeds 0..6 (every
    gate so far: 3 sizes -> seeds 0, 1, 2) give exactly the old prompts; the record codes still use the full seed.
 2. per-needle timeout (SIGALRM, NEEDLE_TIMEOUT_S, default 180 s) instead of urlopen's 1800 s socket timeout.
"""

import hashlib
import json
import os
import signal
import sys
import time
import urllib.request

NEEDLE_TIMEOUT_S = int(os.environ.get("NEEDLE_TIMEOUT_S", "180"))


def prompt(n_tokens, seed):
    lines = []
    n = n_tokens // 17
    p = seed % 7  # < 7: the needle stays inside the records (old bug: seed >= 7 -> needle >= n)
    needle = n * (3 + p) // 10
    assert 0 <= needle < n, (n, seed, needle)
    for i in range(n):
        h = hashlib.md5(f"{seed}-{i}".encode()).hexdigest()[:8]
        lines.append(f"Record {i}: the access code of vault {i} is {h}.")
    code = hashlib.md5(f"{seed}-{needle}".encode()).hexdigest()[:8]
    q = f"\n\nWhat is the access code of vault {needle}? Answer with the code only."
    return "\n".join(lines) + q, code


class NeedleTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise NeedleTimeout(f"timeout after {NEEDLE_TIMEOUT_S} s")


def ask(url, text, max_tokens):
    body = json.dumps(
        {
            "model": "minimax-m3.1",
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(NEEDLE_TIMEOUT_S)
    try:
        with urllib.request.urlopen(req, timeout=NEEDLE_TIMEOUT_S + 5) as r:
            return json.loads(r.read())
    finally:
        signal.alarm(0)


def main():
    url = f"http://127.0.0.1:{int(sys.argv[1])}/v1/chat/completions"
    out = sys.argv[2]
    sizes = [int(x) for x in sys.argv[3:]] or [20000, 80000, 130000]
    res = []
    for seed, n in enumerate(sizes):
        text, code = prompt(n, seed)
        t = time.time()
        try:
            r = ask(url, text, 2048)
            m = r["choices"][0]["message"]
            content = m.get("content") or ""
            res.append(
                {
                    "tokens": r["usage"]["prompt_tokens"],
                    "code": code,
                    "hit": code in content,
                    "hit_any": code in content + (m.get("reasoning_content") or ""),
                    "out": content[-300:],
                    "completion_tokens": r["usage"]["completion_tokens"],
                    "s": round(time.time() - t, 1),
                }
            )
        except BaseException as e:  # noqa: BLE001  (NeedleTimeout, URLError, socket timeout, bad JSON)
            if isinstance(e, KeyboardInterrupt):
                raise
            res.append({"tokens": n, "code": code, "hit": False, "hit_any": False, "error": repr(e),
                        "s": round(time.time() - t, 1)})
        print(n, res[-1].get("tokens"), "hit", res[-1]["hit"], res[-1].get("s"), "s", flush=True)
        json.dump(res, open(out, "w"), indent=1)  # partial results survive an outer timeout
    json.dump(res, open(out, "w"), indent=1)
    print(f"LONG hits={sum(r['hit'] for r in res)}/{len(res)}")


if __name__ == "__main__":
    main()
