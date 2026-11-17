"""Rebuild the agentic result_summary numbers (tps, TPOT, TTFT, OSL) from events.jsonl with the MLCommons
client's own code: EventRecord msgspec decoding, the OSL / TPOT tokenization rules (OslTrigger /
TpotTrigger -> BatchTokenizer.count_sync) and the MetricsTable window (tracked = issued between
START and STOP_PERFORMANCE_TRACKING; duration = START -> last tracked completion; tps = OSL total /
duration, the non-poisson branch of Report.from_snapshot). ISL is skipped (not in tps / TPOT).
usage: replay_report.py <events.jsonl> [procs=48]"""

# ruff: noqa: I001  (the client is imported after its src dir is put on sys.path)
import json, re, sys, time
from multiprocessing import Pool

sys.path.insert(0, "/app/mlcommons-endpoints/src")
import msgspec
from inference_endpoint.core.record import EventRecord, EventType
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    extract_tokenization_input,
)
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    BatchTokenizer,
    MessageInput,
    TextInput,
)

TOK = "/data/Qwen3.5-35B-A3B-MXFP4"
_bt = None


def _init():
    global _bt
    _bt = BatchTokenizer(TOK, live_workers=1, n_workers=0)


def _count(items):
    return [
        (
            _bt.count_sync(a) if a is not None else -1,
            _bt.count_sync(b) if b is not None else -1,
        )
        for a, b in items
    ]


def tpot_input(out):  # TpotTrigger._extract_tokenization_input
    if out.reasoning or out.tool_calls:
        return MessageInput(*out.as_message_parts_after_first_chunk())
    t = out.text_after_first_chunk()
    return TextInput(t) if t else None


def main():
    E = sys.argv[1]
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    dec = msgspec.json.Decoder(EventRecord, dec_hook=EventType.decode_hook)
    head = re.compile(
        rb'^\{"event_type":"([a-z_.]+)","timestamp_ns":(\d+),"sample_uuid":"([0-9a-f]*)"'
    )
    tracking = False
    start = stop = None
    issued = {}
    first = {}
    rows = []
    n_iss = n_iss_tracked = n_err = n_complete_untracked = 0
    t0 = time.time()
    with open(E, "rb") as f:
        for line in f:
            m = head.match(line)
            if not m:
                continue
            ev, ts, uuid = m.group(1).decode(), int(m.group(2)), m.group(3).decode()
            if ev == "session.start_performance_tracking":
                tracking = True
                start = ts
            elif ev == "session.stop_performance_tracking":
                tracking = False
                stop = ts
            elif ev == "sample.issued":
                n_iss += 1
                if tracking and uuid not in issued:
                    issued[uuid] = ts
                    n_iss_tracked += 1
            elif ev == "sample.recv_first":
                if uuid in issued and uuid not in first:
                    first[uuid] = ts
            elif ev.startswith("error."):
                n_err += 1
            elif ev == "sample.complete":
                if uuid not in issued:
                    n_complete_untracked += 1
                    continue
                out = dec.decode(line).data
                osl_in = (
                    extract_tokenization_input(out)
                    if isinstance(out, TextModelOutput)
                    else None
                )
                tp_in = (
                    tpot_input(out)
                    if isinstance(out, TextModelOutput) and uuid in first
                    else None
                )
                rows.append((issued.pop(uuid), first.get(uuid), ts, osl_in, tp_in))
    print(
        f"parsed in {time.time()-t0:.0f} s: {n_iss} issued ({n_iss_tracked} tracked), {len(rows)} tracked completed, "
        f"{len(issued)} tracked never completed, {n_err} error events",
        flush=True,
    )
    t0 = time.time()
    chunks = [
        [(r[3], r[4]) for r in rows[i : i + 500]] for i in range(0, len(rows), 500)
    ]
    with Pool(P, initializer=_init) as pool:
        counts = [c for part in pool.map(_count, chunks) for c in part]
    print(f"tokenized in {time.time()-t0:.0f} s", flush=True)
    osl = [c[0] for c in counts if c[0] > 0]
    tpot = sorted(
        (r[2] - r[1]) / c[1]
        for r, c in zip(rows, counts)
        if r[1] is not None and c[1] > 0
    )
    ttft = sorted(r[1] - r[0] for r in rows if r[1] is not None)
    last = max(r[2] for r in rows)
    dur = (last - start) / 1e9
    pct = lambda v, q: v[min(len(v) - 1, int(q * len(v)))]
    res = dict(  # noqa: C408
        tracked_duration_s=dur,
        stop_tracking_at_s=(stop - start) / 1e9 if stop else None,
        n_tracked_completed=len(rows),
        osl_total=sum(osl),
        osl_mean=sum(osl) / len(osl),
        tps=sum(osl) / dur,
        qps=len(rows) / dur,
        tpot_ms={q: pct(tpot, q) / 1e6 for q in (0.5, 0.9, 0.99)},
        tpot_mean_ms=sum(tpot) / len(tpot) / 1e6,
        ttft_ms={q: pct(ttft, q) / 1e6 for q in (0.5, 0.9, 0.99)},
    )
    res["tps_per_user_p90"] = 1000 / res["tpot_ms"][0.9]
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
