# Time profile of an endpoints run: per 5-min bin, turns completed, output rate, TTFT, decode
# rate a turn, prompt size (chars) of the turns issued, turn index.   ep_drift.py <events.jsonl>
import collections
import json
import statistics as st
import sys

iss, first, comp = {}, {}, {}
t0 = None
with open(sys.argv[1]) as f:
    lines = list(f)
for line in lines:
    et = line[15:45]
    if et.startswith("session.start_performance"):
        t0 = json.loads(line)["timestamp_ns"]
        continue
    if not et.startswith("sample."):
        continue
    d = json.loads(line)
    u = d["sample_uuid"]
    if d["event_type"] == "sample.issued":
        msgs = d["data"][3] or []
        chars = sum(
            len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or ""))
            for m in msgs
        )
        iss[u] = (d["timestamp_ns"], chars, d["turn"], d["conversation_id"])
    elif d["event_type"] == "sample.recv_first":
        first[u] = d["timestamp_ns"]
    elif d["event_type"] == "sample.complete":
        out = d["data"]
        n = len(
            json.dumps(out[1:])
        )  # output characters (all fields), calibrated to tokens below
        comp[u] = (d["timestamp_ns"], n)
import re

with open(sys.argv[1].replace("events.jsonl", "report.txt")) as f:
    rep = f.read()
osl = float(
    re.search(r"OSL per-turn mean \(accuracy, all turns\): ([0-9.]+)", rep).group(1)
)
cpt = sum(n for _, n in comp.values()) / (osl * len(comp))  # characters a token
print(f"output: {cpt:.2f} chars a token (OSL mean {osl})")
comp = {u: (t, n / cpt) for u, (t, n) in comp.items()}
B = 300e9
bins = collections.defaultdict(
    lambda: {"done": 0, "tok": 0, "ttft": [], "rate": [], "chars": [], "turn": []}
)
for u, (ti, ch, turn, cid) in iss.items():
    b = int((ti - t0) // B)
    bins[b]["chars"].append(ch)
    bins[b]["turn"].append(turn)
for u, (tc, n) in comp.items():
    b = int((tc - t0) // B)
    bins[b]["done"] += 1
    bins[b]["tok"] += n
    if u in first and u in iss:
        bins[b]["ttft"].append((first[u] - iss[u][0]) / 1e6)
        dt = (tc - first[u]) / 1e9
        if n > 16 and dt > 0:
            bins[b]["rate"].append(n / dt)
print(
    "bin(min) | turns done | out tok/s | TTFT med ms | decode tok/s/turn med | issued prompt chars med (~tokens) | turn med"
)
for b in sorted(bins):
    x = bins[b]
    med = lambda v: st.median(v) if v else float("nan")
    print(
        f"{b*5:3d}-{b*5+5:<3d} | {x['done']:5d} | {x['tok']/300:7.0f} | {med(x['ttft']):7.0f} | {med(x['rate']):6.1f} | {med(x['chars']):8.0f} (~{med(x['chars'])/3.6/1000:.0f}k) | {med(x['turn']):4.0f}"
    )
