# Full-run length at concurrency c from a short run: per-turn service time t(L) = a + b L (fit on the
# short run's turns, L = prompt chars / 3.6), applied to every assistant turn of the dataset, plus the
# injected delays and the measured client gap; / c (and >= the longest trajectory).
#   ep_extrap.py <events.jsonl> <dataset.jsonl> <c>
import collections
import itertools
import json
import sys

import numpy as np

ev, ds, c = sys.argv[1], sys.argv[2], int(sys.argv[3])
iss, comp, conv = {}, {}, collections.defaultdict(list)
with open(ev) as f:
    lines = list(f)
for line in lines:
    if '"sample.issued"' in line[:40]:
        d = json.loads(line)
        msgs = d["data"][3] or []
        L = (
            sum(
                len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or ""))
                for m in msgs
            )
            / 3.6
        )
        iss[d["sample_uuid"]] = (d["timestamp_ns"], L)
        conv[d["conversation_id"]].append(d["sample_uuid"])
    elif '"sample.complete"' in line[:40]:
        d = json.loads(line)
        comp[d["sample_uuid"]] = d["timestamp_ns"]
X = np.array([iss[u][1] for u in comp if u in iss])
Y = np.array([(comp[u] - iss[u][0]) / 1e9 for u in comp if u in iss])
b, a = np.polyfit(X, Y, 1)
gaps = []
for us in conv.values():
    us = [u for u in us if u in comp]
    us.sort(key=lambda u: iss[u][0])
    gaps += [(iss[v][0] - comp[u]) / 1e9 for u, v in itertools.pairwise(us)]
print(
    f"short run: {len(Y)} turns, context {X.min()/1e3:.0f}k-{X.max()/1e3:.0f}k (median {np.median(X)/1e3:.0f}k), "
    f"service median {np.median(Y):.2f} s mean {Y.mean():.2f} s; fit t = {a:.2f} s + {b*1e3*1e3:.2f} s/Mtok * L; gap mean {np.mean(gaps):.2f} s"
)
with open(ds) as f:
    rows = [json.loads(l) for l in f][1:]
convs = collections.defaultdict(list)
for r in rows:
    convs[r["conversation_id"]].append(r)
n = lambda v: len(v if isinstance(v, str) else json.dumps(v)) / 3.6 if v else 0
per_conv = []
for rs in convs.values():
    rs.sort(key=lambda r: int(r["turn"]))
    ctx = n(rs[0].get("system", "")) + n(rs[0].get("tools", ""))
    t = 0.0
    for r in rs:
        if r["role"] == "assistant":
            t += a + b * ctx
        t += float(r.get("delay_seconds") or 0)
        ctx += sum(
            n(r.get(k, ""))
            for k in ("content", "reasoning_content", "tool_calls", "tool_results")
        )
    per_conv.append(t)
per_conv = np.array(per_conv)
tot = per_conv.sum()
est = max(tot / c, per_conv.max())
print(
    f"dataset: sum over trajectories {tot/3600:.1f} h; longest trajectory {per_conv.max()/60:.0f} min; "
    f"full run at c={c}: ~{est/3600:.1f} h (10-hour cap)"
)
