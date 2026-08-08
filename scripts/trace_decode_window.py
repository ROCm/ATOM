"""Drive a long-context load and capture a profiler window once the node is busy.

A trace is only worth reading if it covers the regime being asked about, so this
waits for the decode batch to actually build up before starting the profiler
rather than tracing the prefill ramp. Prompts are long enough (well past the
8192-token hot buffer) that every step goes through the cold-pool swap path.

The server must have been launched with TORCH_PROFILER_DIR set.

  docker exec -e PYTHONPATH=/it-share/yajizhan/code/ATOM atom_pp4pd_test \\
    python3 scripts/trace_decode_window.py --requests 20 --ctx-tokens 60000
"""

import argparse
import json
import threading
import time
import urllib.request

FILLER = (
    "The city archives record countless mundane details about daily civic life. "
    "Clerks filed reports on weather, road repairs, market prices, and festivals. "
    "None of these ordinary entries carry any special significance whatsoever. "
)


def post(url: str, payload: dict, timeout: int = 1200):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="http://127.0.0.1:30000")
    ap.add_argument("--decode", default="http://127.0.0.1:8020")
    ap.add_argument("--model", default="/mnt/models/GLM-5.2-MXFP4")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--ctx-tokens", type=int, default=60000)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--settle-secs", type=int, default=180)
    ap.add_argument("--decode-log", default="/tmp/decode.log")
    ap.add_argument(
        "--live-marker",
        default="Engine Core: output send",
        help="log line whose count must be rising before the window opens; use "
        "'Scheduled prefill batch' with --decode/--decode-log pointed at prefill",
    )
    ap.add_argument("--window-secs", type=int, default=6)
    args = ap.parse_args()

    # ~4 chars/token is close enough; the point is "far past the hot buffer".
    body = FILLER * max(1, args.ctx_tokens * 4 // len(FILLER))
    done = []

    def fire(i: int):
        try:
            post(
                f"{args.mesh}/v1/chat/completions",
                {
                    "model": args.model,
                    "messages": [
                        {"role": "user", "content": f"[req {i}]\n{body}\nSummarize."}
                    ],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                },
            )
            done.append(i)
        except (OSError, ValueError) as e:  # a straggler must not abort the capture
            print(f"  req {i} failed: {type(e).__name__}: {e}", flush=True)

    threads = [
        threading.Thread(target=fire, args=(i,), daemon=True)
        for i in range(args.requests)
    ]
    print(f"firing {args.requests} requests of ~{args.ctx_tokens} tokens", flush=True)
    for t in threads:
        t.start()

    # Wait for decode to actually be running rather than sleeping a guessed
    # interval: these prompts often stop well short of max_tokens, so a fixed
    # settle lands the window either in the prefill ramp or after the last token.
    # Decode is "live" when the engine's output counter is still moving.
    def output_sends() -> int:
        try:
            with open(args.decode_log, errors="replace") as f:
                return sum(1 for ln in f if args.live_marker in ln)
        except OSError:
            return -1

    deadline = time.time() + args.settle_secs
    prev = output_sends()
    live = 0
    while time.time() < deadline:
        time.sleep(2)
        cur = output_sends()
        if cur > prev:
            live += 1
            if live >= 2:
                print(f"decode is live ({cur - prev} sends in the last 2s)", flush=True)
                break
        else:
            live = 0
        prev = cur
    else:
        print("WARNING: never saw decode activity; window may be empty", flush=True)

    print("start_profile", flush=True)
    print(post(f"{args.decode}/start_profile", {}, timeout=120), flush=True)
    time.sleep(args.window_secs)
    print("stop_profile", flush=True)
    print(post(f"{args.decode}/stop_profile", {}, timeout=600), flush=True)

    for t in threads:
        t.join(timeout=600)
    print(f"completed {len(done)}/{args.requests} requests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
