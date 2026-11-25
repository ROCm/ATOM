#!/usr/bin/env python3
"""screen_compare.py NEW_RUN NEW_GPUS BASE_RUN BASE_GPUS [SECONDS]: a 25-min screen against the same first SECONDS
(default 1500) of a baseline run (same warmup, same trace order). Client side (p2d subwindow.py logic: requests that
started in the slice; records-span and tail-free throughput) and server side (pd_balance over the slice). Prints a
table and writes NEW_RUN/screen_compare.json."""

import json
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
new, ng, base, bg = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
secs = sys.argv[5] if len(sys.argv) > 5 else "1500"


def client(run, gpus):
    out = subprocess.run([sys.executable, str(S / "subwindow.py"), str(run), "0", secs, gpus], capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out)


def server(run, tag):
    name = f"pd_balance_first{int(float(secs))}s.json"
    subprocess.run([sys.executable, str(S / "pd_balance.py"), str(run), "--to-s", secs, "--out", name],
                   capture_output=True, text=True, check=True)
    return json.loads((run / name).read_text())


def flat(c, s):
    row = {f"client.{k}": v for k, v in c.items() if k not in ("from_s", "to_s")}
    p = s["prefill"].get("prefill0", {})
    for k in ("prompt_tok_s", "computed_tok_s", "hit_hbm", "hit_store", "queue_time_s_mean", "ttft_engine_s_mean",
              "batch_tokens_mean", "batches_s"):
        row[f"prefill.{k}"] = p.get(k)
    for k in ("waiting", "parked_store_loads", "held_kv_share", "deferred_free_requests"):
        row[f"prefill.{k}.mean"] = (p.get(k) or {}).get("mean")
    sl = s["prefill"].get("store_loads", {})
    row["store_loads.loads_s"] = sl.get("loads_s")
    row["store_loads.total_ms.mean"] = (sl.get("total_ms") or {}).get("mean")
    row["store_loads.pending.mean"] = (sl.get("loads_pending") or {}).get("mean")
    for d, v in sorted(s["decode"].items()):
        for k in ("finished_req_s", "batch_mean", "itl_ms_mean", "queue_time_s_mean", "pd_kv_transfer_s_mean",
                  "decode_held_prompt", "share_samples_credit_ge_0.85", "gpu_forward_busy"):
            row[f"{d}.{k}"] = v.get(k)
        row[f"{d}.running.mean"] = v["running"]["mean"]
        row[f"{d}.credit_q.mean"] = v["credit_wait_queue"]["mean"]
        row[f"{d}.occupancy.mean"] = (v.get("credit_occupancy") or {}).get("mean")
        row[f"{d}.kv.mean"] = v["kv_usage"]["mean"]
        row[f"{d}.deferred"] = v.get("deferred")
        row[f"{d}.little"] = v.get("little_s_per_req")
    return row


a = flat(client(new, ng), server(new, "new"))
b = flat(client(base, bg), server(base, "base"))
rows = []
for k in list(dict.fromkeys(list(b) + list(a))):
    va, vb = a.get(k), b.get(k)
    d = f"{(va - vb) / abs(vb) * 100:+.1f}%" if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and vb else ""
    rows.append({"metric": k, "base": vb, "new": va, "delta": d})
    print(f"{k:<44} | {str(vb)[:44]:<44} | {str(va)[:44]:<44} | {d}")
(new / "screen_compare.json").write_text(json.dumps({"base": str(base), "new": str(new), "seconds": secs, "rows": rows},
                                                    indent=1))
