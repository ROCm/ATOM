#!/usr/bin/env python3
"""A1 (C4 roofline): several fixed-shape phases on ONE server, each with its own profiler windows. Stdlib only, runs in
the container. Token-id prompts (seeded, 1000..150000), /v1/completions, so no chat template / tokenization cost.

Per phase (JSON list in --phases):
  1. prefill the phase's decode prompts (max_tokens=1) into the prefix cache (or reuse earlier phases' prompts);
  2. stream them (ignore_eos, dec_tokens each): reference client step for ref_s (no profiler), then a profiler window
     of win_s seconds of steady decode; wait until the streams end;
  3. on the idle server, pf_reps C4-shaped requests: the first decode prompt's block-aligned prefix + pf_new fresh
     tokens, streamed with max_tokens=pf_osl: TTFT and every chunk gap (the request's own bs1 decode right after its
     prefill, i.e. the AgentX C4 per-request pattern); then one more inside a profiler window when pf_profile.
usage: python3 a1_fixed.py --phases '[...]' --summary out.json [--server-log log]"""
import argparse, http.client, json, os, random, statistics, sys, threading, time, urllib.parse, urllib.request

LO, HI, BLOCK = 1000, 150000, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8790")
    ap.add_argument("--model", required=True)
    ap.add_argument("--phases", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    U = urllib.parse.urlparse(a.url)
    rng = random.Random(a.seed)
    phases = json.loads(a.phases)
    S = {"phases": {}, "args": vars(a)}
    prompts = {}

    def log(m):
        print(f"{time.strftime('%F %T')} {m}", flush=True)

    def ids(n):
        return [rng.randrange(LO, HI) for _ in range(n)]

    def post(path, body=None, timeout=3600):
        data = json.dumps(body).encode() if body is not None else b""
        req = urllib.request.Request(a.url + path, data=data, method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw.strip() else {}

    def payload(prompt, mt, stream):
        return {"model": a.model, "prompt": prompt, "max_tokens": mt, "temperature": a.temperature, "ignore_eos": True,
                "stream": stream}

    def log_pos():
        return os.path.getsize(a.server_log) if a.server_log and os.path.exists(a.server_log) else 0

    def scheduled(since):
        if not a.server_log or not os.path.exists(a.server_log):
            return []
        with open(a.server_log, errors="replace") as f:
            f.seek(since)
            return [ln.strip()[:200] for ln in f.read().splitlines() if "Scheduled prefill batch" in ln]

    class Stream(threading.Thread):
        def __init__(self, prompt, mt):
            super().__init__(daemon=True)
            self.prompt, self.mt, self.t, self.err, self.t0 = prompt, mt, [], None, None
            self.first = threading.Event()

        def run(self):
            try:
                conn = http.client.HTTPConnection(U.hostname, U.port or 80, timeout=3600)
                self.t0 = time.time()
                conn.request("POST", "/v1/completions", json.dumps(payload(self.prompt, self.mt, True)),
                             {"Content-Type": "application/json"})
                r = conn.getresponse()
                self.ack = time.time()
                if r.status != 200:
                    raise RuntimeError(f"http {r.status}: {r.read(300)!r}")
                while True:
                    ln = r.readline()
                    if not ln:
                        break
                    if ln.startswith(b"data: ") and not ln.startswith(b"data: [DONE]"):
                        self.t.append(time.time())
                        self.first.set()
                self.end = time.time()
            except Exception as e:  # noqa: BLE001
                self.err = repr(e)[:300]
            finally:
                self.first.set()

    def step_stats(streams, lo, hi):
        gaps = []
        for s in streams:
            ts = [t for t in list(s.t) if lo <= t <= hi]
            gaps += [y - x for x, y in zip(ts, ts[1:]) if y - x >= 0.0015]
        if not gaps:
            return None
        gaps.sort()
        q = lambda p: 1e3 * gaps[int(p * (len(gaps) - 1))]
        return {"median_ms": 1e3 * statistics.median(gaps), "p10_ms": q(0.1), "p90_ms": q(0.9), "n": len(gaps)}

    def one_request(prompt, mt):
        s = Stream(prompt, mt)
        s.start()
        s.join(1800)
        if s.err or not s.t:
            return {"err": s.err}
        g = [1e3 * (y - x) for x, y in zip(s.t, s.t[1:])]
        return {"ttft_ms": 1e3 * (s.t[0] - s.t0), "ack_ms": 1e3 * (s.ack - s.t0), "gaps_ms": [round(x, 3) for x in g],
                "decode_ms": 1e3 * (s.t[-1] - s.t[0]), "chunks": len(s.t),
                "gap_median_ms": statistics.median(g) if g else None}

    warm = ids(2048)
    post("/v1/completions", payload(warm, 16, False))
    log("warm done")
    for ph in phases:
        nm = ph["name"]
        P = {"spec": ph}
        log(f"=== phase {nm}: {ph}")
        dec = []
        for src in ph.get("reuse", []):
            dec += prompts[src]
        for i in range(ph.get("bs", 0) - len(dec)):
            p = ids(ph["ctx"])
            t0 = time.time()
            post("/v1/completions", payload(p, 1, False))
            log(f"  prompt {i}: {len(p)} tokens prefilled in {time.time() - t0:.1f} s")
            P.setdefault("prompt_prefill_s", []).append(time.time() - t0)
            dec.append(p)
        prompts[nm] = dec
        time.sleep(ph.get("settle_s", 0))
        if dec and ph.get("dec_tokens", 0):
            pos = log_pos()
            streams = [Stream(p, ph["dec_tokens"]) for p in dec]
            for s in streams:
                s.start()
            for s in streams:
                s.first.wait(1800)
            bad = [s.err for s in streams if s.err]
            if bad:
                log(f"ABORT stream errors {bad[:2]}")
                S["phases"][nm] = P
                json.dump(S, open(a.summary, "w"), indent=1)
                os._exit(1)
            time.sleep(1.0)
            r0 = time.time()
            time.sleep(ph.get("ref_s", 3))
            P["ref"] = step_stats(streams, r0, time.time())
            log(f"  ref step {P['ref']}")
            if ph.get("win_s", 0) > 0:
                w0 = time.time()
                post("/start_profile", timeout=120)
                w1 = time.time()
                time.sleep(ph["win_s"])
                w2 = time.time()
                alive = sum(s.is_alive() for s in streams)
                post("/stop_profile", timeout=1800)
                w3 = time.time()
                P["window"] = {"start": w0, "started": w1, "stop": w2, "stopped": w3, "alive": alive,
                               "step": step_stats(streams, w1, w2)}
                log(f"  window step {P['window']['step']} alive {alive}/{len(streams)} stop took {w3 - w2:.1f} s")
            for s in streams:
                s.join(1800)
            P["decode_sched"] = scheduled(pos)[:4]
        if ph.get("pf_new", 0) and dec:
            pre = dec[0][: (len(dec[0]) // BLOCK) * BLOCK]
            reps = []
            for i in range(ph.get("pf_reps", 3)):
                pos = log_pos()
                r = one_request(pre + ids(ph["pf_new"]), ph.get("pf_osl", 128))
                r["sched"] = scheduled(pos)[:2]
                reps.append(r)
                log(f"  C4 request {len(pre)}+{ph['pf_new']}: ttft {r.get('ttft_ms', 0):.1f} ms, gap median "
                    f"{r.get('gap_median_ms')} first gaps {r.get('gaps_ms', [])[:8]} last {r.get('gaps_ms', [])[-2:]}")
                time.sleep(0.5)
            P["pf_reps"] = reps
            if ph.get("pf_profile"):
                pos = log_pos()
                post("/start_profile", timeout=120)
                time.sleep(0.3)
                r = one_request(pre + ids(ph["pf_new"]), ph.get("pf_osl", 128))
                time.sleep(0.3)
                t2 = time.time()
                post("/stop_profile", timeout=1800)
                r["stop_s"] = time.time() - t2
                r["sched"] = scheduled(pos)[:2]
                P["pf_window"] = r
                log(f"  profiled C4 request: ttft {r.get('ttft_ms', 0):.1f} ms gaps {r.get('gaps_ms', [])[:10]}")
        S["phases"][nm] = P
        json.dump(S, open(a.summary, "w"), indent=1)
    log("DONE")
    os._exit(0)


if __name__ == "__main__":
    main()
