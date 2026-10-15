#!/usr/bin/env python3
"""Three-arm KV-offload accuracy probe with explicit reuse validity gates.

Blind questions omit the synthetic document; cold questions carry a fresh nonce
at its head; warm questions reuse a primed document. Compare correctness, not
bit-identical text: even HBM prefix reuse can change a greedy continuation.

This samples semantic correctness, not complete PAGE/STATE byte equality. The
model's cursor check detects some missing STATE restores, but a matching cursor
does not establish correct contents or request identity.

Supply percentages use actual prompt tokens summed over the arm, not external
lookup queries (which exclude the HBM-hit prefix). Run against an otherwise idle
server: metrics are server-wide counters. Missing/reset metrics invalidate a run.

Usage: kv_offload_accuracy_arms.py [port] [questions] [facts] [max_tokens] [model]
       [--concurrency N] [--min-tier-share PCT] [--min-hbm-share PCT]
       [--min-reuse-share PCT] [--out results.json]

Defaults retain the single-client tier-only check. For HBM-only validation use
--min-tier-share 0 --min-hbm-share 50. For a mixed-source arm set both thresholds
BEFORE running; do not lower them afterwards to make an empty test pass.
"""

import argparse
import json
import math
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path

METRICS = (
    "external_prefix_cache_queries_total",
    "external_prefix_cache_hits_total",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
)
HEAD = (
    "You are reading a facility register. Answer only with the access code, "
    "nothing else.\n\n"
)


class AccuracyRegression(ValueError):
    pass


def make_data(facts, questions, seed):
    rng = random.Random(seed)
    alpha = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    codes = [
        "".join(rng.choice(alpha) for _ in range(2))
        + "-"
        + "".join(rng.choice("0123456789") for _ in range(4))
        for _ in range(facts)
    ]
    doc = "\n".join(
        f"Record {i:04d}: the access code for sector {i:04d} is {codes[i]}."
        for i in range(facts)
    )
    return codes, doc, rng.sample(range(facts), 3 * questions)


def parse_metrics(text):
    out = {}
    pattern = re.compile(r"^vllm:(\w+)(?:\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?$")
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match is None or match[1] not in METRICS:
            continue
        value = float(match[2])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"metric {match[1]} must be finite and nonnegative")
        out[match[1]] = out.get(match[1], 0.0) + value
    missing = set(METRICS) - out.keys()
    if missing:
        raise ValueError(f"missing metrics: {sorted(missing)}")
    return out


def metric_delta(before, after):
    delta = {key: after[key] - before[key] for key in METRICS}
    if any(value < 0 for value in delta.values()):
        raise ValueError("metric counter decreased during the arm")
    return delta


def summarize(answers, delta):
    tokens = sum(tokens for _, tokens in answers)
    if tokens <= 0:
        raise ValueError("arm has no prompt tokens")
    tier = delta["external_prefix_cache_hits_total"]
    hbm = delta["prefix_cache_hits_total"]
    if tier + hbm > tokens:
        raise ValueError("reported reuse exceeds the arm's prompt token budget")
    return {
        "correct": sum(correct for correct, _ in answers),
        "questions": len(answers),
        "prompt_tokens": tokens,
        "tier_hits": tier,
        "hbm_hits": hbm,
        "tier_share": 100 * tier / tokens,
        "hbm_share": 100 * hbm / tokens,
        "reuse_share": 100 * (tier + hbm) / tokens,
        "metrics_delta": delta,
    }


def validate(args, blind, cold, warm):
    if blind["correct"] * 4 > args.questions:
        raise ValueError("blind arm scored too high; questions do not need the prefix")
    if cold["correct"] * 4 < args.questions * 3:
        raise ValueError("cold arm scored too low; task/budget is not a valid control")
    if cold["reuse_share"] > 5:
        raise ValueError("cold arm reused more than 5% of its prompt from HBM/tier")
    for source in ("tier", "hbm", "reuse"):
        minimum = getattr(args, f"min_{source}_share")
        if warm[f"{source}_share"] < minimum:
            raise ValueError(f"warm {source} supply below the declared {minimum}% gate")
    if warm["correct"] < cold["correct"]:
        raise AccuracyRegression("warm accuracy lower than the cold control")


def request_json(url, payload=None):
    req = urllib.request.Request(
        url,
        None if payload is None else json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900 if payload is not None else 10) as h:
        return json.load(h)


def tier(url):
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as h:
        return parse_metrics(h.read().decode())


def ask(args, doc, sector, *, with_doc, nonce=None):
    body = HEAD
    if with_doc:
        if nonce is not None:
            body += f"Register revision {nonce}.\n\n"
        body += doc + "\n\n"
    body += f"Question: what is the access code for sector {sector:04d}?"
    response = request_json(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": body}],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "seed": 0,
            "stream": False,
        },
    )
    choice = response["choices"][0]
    tokens = response["usage"]["prompt_tokens"]
    text = choice["message"]["content"]
    if type(tokens) is not int or tokens <= 0 or not isinstance(text, str):
        raise ValueError("response lacks valid prompt usage or answer text")
    return {
        "sector": sector,
        "text": text,
        "prompt_tokens": tokens,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": response["usage"].get("completion_tokens"),
    }


def run(args, name, codes, doc, sectors, salt, *, with_doc, unique=False):
    url = f"http://127.0.0.1:{args.port}"
    before = tier(url)

    def one(item):
        i, sector = item
        result = ask(
            args,
            doc,
            sector,
            with_doc=with_doc,
            nonce=f"{name}-{i}-{salt}" if unique else None,
        )
        result["expected"] = codes[sector]
        result["correct"] = codes[sector] in result["text"]
        return result

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, enumerate(sectors)))
    summary = summarize(
        [(r["correct"], r["prompt_tokens"]) for r in results],
        metric_delta(before, tier(url)),
    )
    for r in results:
        if not r["correct"] and with_doc:
            print(
                f"  {name} sector {r['sector']:04d} wanted {r['expected']}: "
                f"{r['text'].strip()[:80]!r}",
                flush=True,
            )
    print(
        f"ARM {name} correct {summary['correct']}/{len(sectors)} "
        f"prompt_tokens={summary['prompt_tokens']} "
        f"tier={summary['tier_hits']:.0f}/{summary['prompt_tokens']} "
        f"({summary['tier_share']:.1f}%) "
        f"hbm={summary['hbm_hits']:.0f}/{summary['prompt_tokens']} "
        f"({summary['hbm_share']:.1f}%)",
        flush=True,
    )
    return {**summary, "responses": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", default="8021")
    parser.add_argument("questions", nargs="?", type=int, default=16)
    parser.add_argument("facts", nargs="?", type=int, default=900)
    parser.add_argument("max_tokens", nargs="?", type=int, default=640)
    parser.add_argument("model", nargs="?", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--min-tier-share", type=float, default=50)
    parser.add_argument("--min-hbm-share", type=float, default=0)
    parser.add_argument("--min-reuse-share", type=float, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if (
        min(args.questions, args.concurrency, args.max_tokens) <= 0
        or args.facts < 3 * args.questions
    ):
        parser.error(
            "positive questions/concurrency/budget and facts >= 3*questions required"
        )
    for source in ("tier", "hbm", "reuse"):
        value = getattr(args, f"min_{source}_share")
        if not 0 <= value <= 100:
            parser.error("supply gates must be finite percentages in [0, 100]")
    if args.min_tier_share + args.min_hbm_share > 100:
        parser.error("tier and HBM minimum shares cannot sum to more than 100%")
    # Exclusive creation checks the output path before sending any workload.
    output = args.out.open("x") if args.out else None
    report = {
        "configuration": {**vars(args), "out": str(args.out) if args.out else None},
        "arms": {},
        "status": "invalid",
    }
    try:
        codes, doc, sectors = make_data(args.facts, args.questions, args.seed)
        salt = f"{os.getpid():d}-{time.time_ns():d}"
        report["cold_nonce"] = salt
        print(
            f"document facts={args.facts} questions_per_arm={args.questions} "
            f"max_tokens={args.max_tokens} concurrency={args.concurrency}",
            flush=True,
        )
        report["prime"] = ask(args, doc, sectors[0], with_doc=True)
        for i, name in enumerate(("blind", "cold", "warm")):
            report["arms"][name] = run(
                args,
                name,
                codes,
                doc,
                sectors[i * args.questions : (i + 1) * args.questions],
                salt,
                with_doc=name != "blind",
                unique=name == "cold",
            )
        validate(args, *(report["arms"][name] for name in ("blind", "cold", "warm")))
        report["status"] = "passed"
        print("VALID test: accuracy and all declared supply gates passed", flush=True)
        return 0
    except (OSError, HTTPException, ValueError, KeyError, TypeError, IndexError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"INVALID: {report['error']}", flush=True)
        return 2 if isinstance(exc, AccuracyRegression) else 1
    finally:
        if output is not None:
            with output:
                json.dump(report, output, indent=2)
                output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
