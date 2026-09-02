"""Check that requests terminate on their own, the way lm_eval needs them to.

The other probes grade *what* the model said. This one grades *whether it
stopped*, which is the failure that keeps lm_eval from finishing: a request
that never emits EOS runs to max_tokens, and enough of those make an eval look
hung even though every request eventually returns.

``probe_varied_prompts.py`` cannot detect this. Its FACTS prompts hit the
completion endpoint with no stop sequence, so rambling to max_tokens after a
short answer is correct behaviour there, not a fault. This probe instead sends
prompts that terminate reliably on a known-good config, and treats
``finish_reason != "stop"`` as the failure.

It also reproduces the access pattern lm_eval has and the other probes lack: a
long few-shot preamble **shared verbatim by every request**, so all but the
first are prefix-cache hits. GSM8K works this way, and under rapidserve that
path is delicate -- V4's SWA-on-a-prefix-cache-hit tail gate has to round-trip
through a second process, because the prefill forward writes the sliding-window
KV but only decode owns the SlidingWindowPool (disagg_types.BlockAssignment).
Distinct prompts, as in the other probes, produce almost no hits and so miss it
entirely.

The first request is reported separately as the one that *populates* the cache;
every later request *hits* it. If only the hit requests fail to stop, the fault
is in the prefix-cache path, not in decoding generally -- and that split is the
whole point of the tool.

Usage::

    # default: shared few-shot prefix, chat endpoint
    python tools/probe_eos.py --n 32

    # control: same prompts, no shared prefix (should isolate the cache path)
    python tools/probe_eos.py --n 32 --no-shared-prefix

    # compare against a known-good config
    python tools/probe_eos.py --n 32 --save /tmp/eos_dpa.json
    python tools/probe_eos.py --n 32 --compare /tmp/eos_dpa.json
"""

import argparse
import collections
import concurrent.futures
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from probe_varied_prompts import one_request

# Short factual questions with unambiguous answers. A working instruct model
# answers each in a sentence or two and emits EOS; there is no reason for any
# of these to reach max_tokens.
QUESTIONS = [
    "What is the capital of France?",
    "How many days are in a leap year?",
    "What is the chemical symbol for gold?",
    "Who wrote 'Romeo and Juliet'?",
    "What is the largest planet in our solar system?",
    "What is 12 multiplied by 12?",
    "Which ocean lies between Africa and Australia?",
    "What is the freezing point of water in Celsius?",
    "Who was the first person to walk on the moon?",
    "What is the square root of 144?",
    "What colour do you get mixing blue and yellow?",
    "How many continents are there?",
    "What is the tallest mountain on Earth?",
    "In which year did the Second World War end?",
    "What is the smallest prime number?",
    "What is the currency of Japan?",
]

# Stands in for GSM8K's few-shot preamble: long enough to span many KV blocks,
# identical across requests, so every request after the first is a cache hit.
FEWSHOT = """You are a careful assistant. Answer each question directly and \
concisely, then stop. Here are worked examples of the expected style.

Question: What is the capital of Spain?
Answer: The capital of Spain is Madrid.

Question: How many sides does a hexagon have?
Answer: A hexagon has six sides.

Question: What is the boiling point of water at sea level in Celsius?
Answer: Water boils at 100 degrees Celsius at sea level.

Question: Who painted the Mona Lisa?
Answer: The Mona Lisa was painted by Leonardo da Vinci.

Question: What is 7 times 8?
Answer: 7 times 8 is 56.

Question: What is the largest mammal?
Answer: The largest mammal is the blue whale.

Question: Which planet is closest to the Sun?
Answer: Mercury is the planet closest to the Sun.

Question: What language is spoken in Brazil?
Answer: The language spoken in Brazil is Portuguese.

Now answer the following question in the same style.

"""


def build_payload(args, question: str) -> dict:
    prefix = FEWSHOT if args.shared_prefix else ""
    if args.completion:
        return {
            "model": args.model,
            "prompt": f"{prefix}Question: {question}\nAnswer:",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": f"{prefix}{question}"}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }


def report(results: list, args) -> int:
    reasons = collections.Counter(
        r.get("finish_reason") for r in results if r.get("ok")
    )
    errors = [r for r in results if not r.get("ok")]

    print(f"\n-- finish_reason over {len(results)} requests --")
    for reason, n in reasons.most_common():
        print(f"  {n:4d}x  {reason}")
    if errors:
        print(f"  {len(errors):4d}x  transport error")
        for e in errors[:3]:
            print(f"        {e['error']}")

    # The split that localises the fault: request 0 populates the shared
    # prefix, every later request hits it.
    ok = [r for r in results if r.get("ok")]
    if args.shared_prefix and len(ok) > 1:
        first_bad = ok[0]["finish_reason"] != "stop"
        rest = ok[1:]
        rest_bad = sum(1 for r in rest if r["finish_reason"] != "stop")
        print("\n-- prefix-cache split --")
        print(f"  populating request: {'FAILED to stop' if first_bad else 'stopped'}")
        print(f"  cache-hit requests: {rest_bad}/{len(rest)} failed to stop")
        if rest_bad and not first_bad:
            print(
                "\n  Only cache-hit requests failed. That points at the "
                "prefix-cache\n  path rather than decoding in general — under "
                "rapidserve, V4's SWA\n  tail gate on a hit round-trips through "
                "the prefill process.\n  Re-run with --no-shared-prefix to "
                "confirm it goes away."
            )

    runaway = [r for r in ok if r["finish_reason"] != "stop"]
    if runaway:
        print(f"\n-- tail of {min(3, len(runaway))} runaway completion(s) --")
        for r in runaway[:3]:
            tail = " ".join(r["text"].split()[-25:])
            print(f"  ...{tail}")

    print("\n-- verdict --")
    if errors:
        print("  Transport errors — server may be unhealthy; fix that first.")
        return 2
    if runaway:
        print(
            f"  FAIL: {len(runaway)}/{len(ok)} requests never emitted EOS.\n"
            "  These are what stall lm_eval: each burns the full max_tokens."
        )
        return 1
    print("  PASS: every request terminated on its own.")
    return 0


def compare_runs(results: list, path: str) -> int:
    with open(path) as f:
        ref = json.load(f)["results"]
    n = min(len(ref), len(results))
    flips = []
    for i in range(n):
        a, b = results[i], ref[i]
        if not (a.get("ok") and b.get("ok")):
            continue
        if a["finish_reason"] != b["finish_reason"]:
            flips.append((i, b["finish_reason"], a["finish_reason"]))
    print(f"\n-- vs {path} --")
    if not flips:
        print("  No termination differences.")
        return 0
    print(f"  {len(flips)} request(s) changed termination (reference -> this):")
    for i, rb, ra in flips[:10]:
        print(f"    #{i:3d}  {rb} -> {ra}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="default")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--completion",
        action="store_true",
        help="use the completion endpoint instead of chat (also switch --url)",
    )
    ap.add_argument(
        "--no-shared-prefix",
        dest="shared_prefix",
        action="store_false",
        help="drop the few-shot preamble, so requests share no cacheable prefix",
    )
    ap.add_argument("--save", metavar="FILE")
    ap.add_argument("--compare", metavar="FILE")
    args = ap.parse_args()

    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(args.n)]
    print(f"Sending {len(questions)} requests to {args.url}")
    print(
        f"  shared_prefix={args.shared_prefix} max_tokens={args.max_tokens} "
        f"concurrency={args.concurrency} temp={args.temperature}"
    )

    def send(q):
        return one_request(args.url, build_payload(args, q), args.timeout)

    # Request 0 goes first and alone: it populates the shared prefix so that
    # every later request is a cache hit. Firing them all at once would race,
    # and several would miss the cache for reasons unrelated to the bug.
    results = [send(questions[0])]
    rest = questions[1:]
    if rest:
        if args.concurrency <= 1:
            results.extend(send(q) for q in rest)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                results.extend(pool.map(send, rest))

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"saved -> {args.save}")

    rc = report(results, args)
    if args.compare:
        rc = compare_runs(results, args.compare) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
