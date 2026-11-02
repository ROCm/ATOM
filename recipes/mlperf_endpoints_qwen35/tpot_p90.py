#!/usr/bin/env python3
"""MLPerf-style interactivity and output tok/s of minutes [a, b), from the client's events.jsonl.

TPOT of a response = (complete - recv_first) / (its tokens - 1); tokens are counted with the
model's tokenizer over the response's output, reasoning and tool-call text. The rules count the
tokens after the first output-bearing chunk, which with MTP often holds 2-3 tokens, so this p90
TPOT is slightly lower than the official one. tps_per_user = 1000 / p90 TPOT (ms).
Responses are those issued in the window, measured from the first issued request.

usage: tpot_p90.py <events.jsonl> <tokenizer dir> [a=2] [b=8]
"""

import collections
import json
import re
import sys

from transformers import AutoTokenizer

HEAD = re.compile(
    r'"event_type":"sample\.(issued|recv_first|complete)","timestamp_ns":(\d+),'
    r'"sample_uuid":"([^"]*)"'
)


def strings(x):
    """Every text fragment of a TextModelOutput: output, reasoning, tool-call names / arguments."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, list):
        for y in x:
            yield from strings(y)
    elif isinstance(x, dict):
        for key, y in x.items():
            if key in ("arguments", "name", "content") or isinstance(y, (list, dict)):
                yield from strings(y)


def main():
    events, tok_dir = sys.argv[1], sys.argv[2]
    a = float(sys.argv[3]) if len(sys.argv) > 3 else 2
    b = float(sys.argv[4]) if len(sys.argv) > 4 else 8
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    stamps = collections.defaultdict(dict)
    text = {}
    t0 = None
    with open(events, errors="replace") as f:
        for line in f:
            m = HEAD.match(line, 1)
            if not m:
                continue
            kind, t, uid = m[1], int(m[2]), m[3]
            if kind == "issued" and t0 is None:
                t0 = t
            stamps[uid][kind] = t
            if kind == "complete":
                data = json.loads(line).get("data")
                if isinstance(data, list) and data and data[0] == "TextModelOutput":
                    text[uid] = "".join(strings(data[1:]))
    lo, hi = t0 + a * 60e9, t0 + b * 60e9
    uids = [
        u
        for u, e in stamps.items()
        if u in text
        and all(k in e for k in ("issued", "recv_first", "complete"))
        and lo <= e["issued"] < hi
    ]
    counts = [
        len(ids)
        for ids in tok([text[u] for u in uids], add_special_tokens=False)["input_ids"]
    ]
    tpot = sorted(
        (stamps[u]["complete"] - stamps[u]["recv_first"]) / 1e6 / (n - 1)
        for u, n in zip(uids, counts)
        if n > 1
    )

    def pct(q):
        return tpot[min(len(tpot) - 1, int(q * len(tpot)))]

    print(
        f"window min {a:g}-{b:g}: {len(uids)} responses, "
        f"{sum(counts) / ((b - a) * 60):,.0f} output tok/s (client-counted)"
    )
    print(
        f"  TPOT ms: p50 {pct(0.5):.2f}  p90 {pct(0.9):.2f}  p99 {pct(0.99):.2f}  ->  "
        f"tps_per_user (1000/p90) {1000 / pct(0.9):.1f}  (1000/p50 {1000 / pct(0.5):.1f})"
    )


if __name__ == "__main__":
    main()
