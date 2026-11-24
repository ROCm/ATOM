#!/usr/bin/env python3
"""ladder_stats.py [--table] [--json] [--save] TAG [TAG...] -- per-point AgentX numbers + server-side regime stats.

Official (summarize_ix.py = InferenceX's own aggregation; use these for the frontier):
  tput_ix   sum(ISL+OSL) / record span / 4 GPUs;  ix_p90 = 1/p90(full-response ITL) (dashboard x-axis)
  ix_p50    1/p50(full-response ITL);  e2eN90 = 1/p90(E2EL/OSL);  ok = aiperf validity (coverage >= 0.95, err <= 10%)
Client side (aiperf profiling phase only, TP4 => /4 GPUs):
  aipTput   aiperf total_token_throughput / 4 (phase window incl. grace -> lower than tput_ix; NEVER a score)
  otput     output_token_throughput / 4
  i_p90itl  1000 / p90(ITL ms)          (summarize.py's interactivity)
  i_p50itl  1000 / p50(ITL ms)
  i_user50  median over requests of per-request output tok/s/user (aiperf output_token_throughput_per_user p50)
  i_mean    1000 / mean(ITL ms)
  e2e50     median over requests of OSL / E2E latency
Server side (from results/TAG/srvmetrics.prom.gz snapshots diffed over the profiling window, and the server log):
  run/wait  mean (max) running / waiting requests over the window
  dec_rows  mean real decode rows per decode step; dist = share of decode steps by row bucket
  pfKtok/s  real (uncached) prefill tokens per second of window (pf_tok = per prefill step in the detail line)
  decB/pfB  estimated share of wall time in decode / prefill forwards: gpu_forward_seconds (rank 0, needs
            ATOM_ENABLE_METRICS_DEVICE_TIMER=1) buckets <= 50 ms = decode steps, > 50 ms = prefill steps, each
            bucket priced at its midpoint (rough; the raw event sums overlap and exceed wall time at high load)
  decMs/pfMs estimated mean decode / prefill step duration (same split)
  kvMx      max referenced-KV usage; evict = paged blocks evicted in window (log [Pool Pressure]); preempt
  hit%      server GPU prefix hit over the profiling window (atom cached/full counters diffed between snapshots).
            InferenceX's server_gpu_cache_hit_rate (gpu_hit in the json) uses aiperf's cumulative-since-start stats
            and so includes the warmup's cache-pressure phase (reads lower).
  lost%     theoretical (aiperf, infinite cache) minus hit% = prompt share re-prefilled because the pool
            evicted it; recompute_frac (detail) = that loss as a share of all prefill tokens
  Server stats come from results/TAG/srvmetrics.prom.gz (ladder.sh's 30 s /metrics snapshots); without it they fall
  back to aiperf's server_metrics_export.json, whose counter/histogram stats include the warmup.
  mono      share of decode steps on the mono path (ATOM_MONO_STATS_EVERY), by token count
"""
import ast, datetime as dt, gzip, json, math, os, re, sys
from collections import defaultdict

D = "/shared_nfs/ygan/p1/c17"
R = f"{D}/results"
# milestone4 fix: milestone3 pointed at {D}/bin/summarize_ix.py, which never existed, so official() ALWAYS fell back
# to the aiperf aggregate (that is why lease logs showed tput_ix 57785 for m3_C_035_1); use the team's copy.
SUMMARIZE_IX = "/shared_nfs/ygan/max/bin/summarize_ix.py"
SAVE = False  # --save: write results/TAG/ladder_stats.json (only for your own tags)


def pct(a, q):
    a = sorted(a)
    if not a:
        return float("nan")
    k = (len(a) - 1) * q / 100
    f = math.floor(k)
    c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


def ts_of(s):  # aiperf naive ISO (container clock = UTC)
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()


def rdir(tag):
    return tag if os.path.isabs(tag) else f"{R}/{tag}"


def client(tag):
    d = rdir(tag)
    out = {}
    agg = None
    try:
        agg = json.load(open(f"{d}/profile_export_aiperf.json"))
    except Exception:
        pass
    recs = []
    try:
        for line in open(f"{d}/profile_export.jsonl"):
            r = json.loads(line)
            md = r.get("metadata", {})
            if md.get("benchmark_phase", "profiling") != "profiling" or md.get("was_cancelled"):
                continue
            recs.append(r)
    except Exception:
        pass
    if not recs and agg is None:
        return None

    def g(m, k):
        v = m.get(k)
        return v.get("value") if isinstance(v, dict) else v

    itl, ttft, e2e, isl, osl, user, t0, t1 = [], [], [], [], [], [], [], []
    for r in recs:
        m, md = r["metrics"], r["metadata"]
        i, t, e, o, n = (g(m, k) for k in ("inter_token_latency", "time_to_first_token", "request_latency",
                                            "output_sequence_length", "input_sequence_length"))
        if i:
            itl.append(i)
            user.append(1000 / i)
        if t is not None:
            ttft.append(t)
        if e and o:
            e2e.append(o / (e / 1000))
        if n:
            isl.append(n)
        if o:
            osl.append(o)
        t0.append(md.get("request_start_ns", 0) / 1e9)
        t1.append(md.get("request_end_ns", 0) / 1e9)
    out["n"] = len(recs)
    out["isl"] = sum(isl) / len(isl) if isl else float("nan")
    out["osl"] = sum(osl) / len(osl) if osl else float("nan")
    out["ttft50"], out["ttft90"], out["ttft99"] = pct(ttft, 50), pct(ttft, 90), pct(ttft, 99)
    out["i_p90itl"] = 1000 / pct(itl, 90) if itl else float("nan")
    out["i_p50itl"] = 1000 / pct(itl, 50) if itl else float("nan")
    out["i_mean"] = 1000 / (sum(itl) / len(itl)) if itl else float("nan")
    out["i_user50"] = pct(user, 50)
    out["e2e50"] = pct(e2e, 50)
    if agg:
        a = lambda k, s="avg": (agg.get(k) or {}).get(s, float("nan"))
        out["tput"] = a("total_token_throughput") / 4
        out["otput"] = a("output_token_throughput") / 4
        out["i_user50"] = a("output_token_throughput_per_user", "p50")
        out["hit"] = a("overall_usage_prompt_cache_read_pct")
        out["theo_hit"] = a("theoretical_prefix_cache_hit")
        out["eff_conc"] = a("effective_concurrency")
        out["eff_dec"] = a("effective_decode_concurrency")
        out["eff_pf"] = a("effective_prefill_concurrency")
        out["dur"] = a("benchmark_duration")
        out["t_start"], out["t_end"] = ts_of(agg["start_time"]), ts_of(agg["end_time"])
        out["agg_n"] = a("request_count")
    else:  # aborted run: estimate from records
        span = max(t1) - min(t0)
        out["tput"] = (sum(isl) + sum(osl)) / span / 4
        out["otput"] = sum(osl) / span / 4
        out["t_start"], out["t_end"] = min(t0), max(t1)
        out["dur"] = span
        out["partial"] = True
    return out


# ---------------- server metrics snapshots ----------------
LINE = re.compile(r'^(atom:[a-zA-Z0-9_]+)(\{[^}]*\})?\s+(\S+)')


def parse_prom(path):
    snaps = []
    cur = None
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("# ts "):
                cur = {"_ts": float(line.split()[2])}
                snaps.append(cur)
                continue
            m = LINE.match(line)
            if m and cur is not None:
                try:
                    cur[m.group(1) + (m.group(2) or "")] = float(m.group(3))
                except ValueError:
                    pass
    return [s for s in snaps if len(s) > 5]


def hist(s, name, rank0=False):
    """{le: cumulative count}, sum, count for a histogram in snapshot s (first label set / tp_rank 0)."""
    b, sm, ct = {}, 0.0, 0.0
    for k, v in s.items():
        if not k.startswith(name):
            continue
        if rank0 and 'tp_rank="' in k and 'tp_rank="0"' not in k:
            continue
        rest = k[len(name):]
        if rest.startswith("_bucket"):
            le = re.search(r'[{,]le="([^"]+)"', rest).group(1)
            b[float(le)] = b.get(float(le), 0) + v
        elif rest.startswith("_sum"):
            sm += v
        elif rest.startswith("_count"):
            ct += v
    return b, sm, ct


def hdiff(s0, s1, name, rank0=False):
    b0, sm0, c0 = hist(s0, name, rank0)
    b1, sm1, c1 = hist(s1, name, rank0)
    return {le: b1[le] - b0.get(le, 0) for le in b1}, sm1 - sm0, c1 - c0


def hquant(b, q):
    les = sorted(b)
    if not les or b[les[-1]] <= 0:
        return float("nan")
    tot = b[les[-1]]
    for le in les:
        if b[le] >= q * tot:
            return le
    return les[-1]


def val(s, name):
    return sum(v for k, v in s.items() if k == name or k.startswith(name + "{"))


def fwd_split(fb, fsum, fct, W, out):
    """Decode / prefill split of the rank-0 forward-time histogram: decode steps all land <= 50 ms and prefill steps
    above it (at c32 the <= 0.05 s count equals the decode-step count); each bucket priced at its midpoint."""
    if not fb or not fct:
        return
    prev_le, prev_c, dsum_e, psum_e = 0.0, 0, 0.0, 0.0
    for le in sorted(fb):
        c = fb[le] - prev_c
        mid = (prev_le + le) / 2 if le != float("inf") else prev_le * 1.5
        if le <= 0.05:
            dsum_e += c * mid
        else:
            psum_e += c * mid
        prev_le, prev_c = (le if le != float("inf") else prev_le), fb[le]
    out["dec_busy_e"], out["pf_busy_e"] = dsum_e / W, psum_e / W
    out["busy_raw"] = fsum / W  # raw event sum; the timed spans overlap, so this can exceed 1
    if out.get("dec_steps"):
        out["dec_step_ms_e"] = dsum_e / out["dec_steps"] * 1000
    if out.get("pf_steps"):
        out["pf_step_ms_e"] = psum_e / out["pf_steps"] * 1000


def server(tag, t_start, t_end):
    out = {}
    p = f"{rdir(tag)}/srvmetrics.prom.gz"
    if not os.path.exists(p):
        p = p[:-3]
    if not os.path.exists(p):
        return out
    snaps = parse_prom(p)
    if len(snaps) < 2:
        return out
    pre = [s for s in snaps if s["_ts"] <= t_start + 5]
    s0 = pre[-1] if pre else snaps[0]
    post = [s for s in snaps if s["_ts"] >= t_end - 5]
    s1 = post[0] if post else snaps[-1]
    win = [s for s in snaps if s0["_ts"] <= s["_ts"] <= s1["_ts"]]
    W = s1["_ts"] - s0["_ts"]
    out["win"] = W
    for g, key in (("atom:requests_running", "run"), ("atom:requests_waiting", "wait"),
                   ("atom:kv_cache_usage_ratio", "kv")):
        xs = [val(s, g) for s in win]
        out[key], out[key + "_max"] = sum(xs) / len(xs), max(xs)
    ev = [s.get('atom:scheduler_kv_cache_blocks{dp_rank="0",engine_role="default",state="evictable"}', float("nan")) for s in win]
    tot = val(s1, "atom:kv_cache_blocks_total")
    out["kv_evictable"] = (sum(ev) / len(ev)) / tot if tot else float("nan")
    b, sm, ct = hdiff(s0, s1, "atom:decode_batch_size")
    out["dec_steps"], out["dec_rows"] = ct, (sm / ct if ct else float("nan"))
    les = sorted(b)
    dist, prev = {}, 0
    for le in les:
        if le == float("inf"):
            continue
        dist[int(le)] = (b[le] - prev) / ct if ct else 0
        prev = b[le]
    out["dec_dist"] = {k: round(v, 4) for k, v in dist.items() if v > 0.0005}
    out["dec_rows_p50"], out["dec_rows_p90"] = hquant(b, .5), hquant(b, .9)
    b, sm, ct = hdiff(s0, s1, "atom:prefill_batch_tokens")
    out["pf_steps"], out["pf_tok"] = ct, (sm / ct if ct else float("nan"))
    out["pf_tokens"] = sm
    out["pf_tok_p90"] = hquant(b, .9)
    fb, fsum, fct = hdiff(s0, s1, "atom:gpu_forward_seconds", rank0=True)
    fwd_split(fb, fsum, fct, W, out)
    b, sm, ct = hdiff(s0, s1, "atom:request_queue_time_seconds")
    out["queue_mean"] = sm / ct if ct else float("nan")
    out["queue_p90"] = hquant(b, .9)
    dc = lambda n: val(s1, n) - val(s0, n)
    full, cached = dc("atom:prefix_cache_full_tokens_total"), dc("atom:prefix_cache_cached_tokens_total")
    out["srv_hit"] = 100 * cached / full if full else float("nan")
    out["preempt"] = dc("atom:preemptions_total")
    out["gen_tok"] = dc("atom:generation_tokens_total")
    out["prompt_tok"] = dc("atom:prompt_tokens_total")
    out["lmc_loaded"] = dc("atom:lmcache_loaded_tokens_total")
    out["lmc_saved"] = dc("atom:lmcache_saved_tokens_total")
    out["kv_blocks_total"] = tot
    return out


def server_aiperf(tag):
    """Same keys as server(), from aiperf's own profiling-phase scrape (server_metrics_export.json[.gz])."""
    p = f"{rdir(tag)}/server_metrics_export.json"
    if not os.path.exists(p):
        p = f"{rdir(tag)}/server_metrics_export.full.json.gz"
    if not os.path.exists(p):
        return {}
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as fh:
        d = json.load(fh)
    m = d.get("metrics", {})
    out = {}
    pr = d.get("summary", {}).get("phase_time_ranges", {}).get("profiling", {})
    W = (pr.get("end_ns", 0) - pr.get("start_ns", 0)) / 1e9 if pr else float("nan")
    out["win"] = W

    def ser(name, want=None):
        ss = (m.get(name) or {}).get("series") or []
        for s in ss:
            lab = s.get("labels") or {}
            if want and any(lab.get(k) != v for k, v in want.items()):
                continue
            return s
        return None

    for g, key in (("atom:requests_running", "run"), ("atom:requests_waiting", "wait"),
                   ("atom:kv_cache_usage_ratio", "kv")):
        s = ser(g)
        if s:
            out[key], out[key + "_max"] = s["stats"]["avg"], s["stats"]["max"]
    s = ser("atom:scheduler_kv_cache_blocks", {"state": "evictable"})
    t = ser("atom:kv_cache_blocks_total")
    if s and t and t["stats"]["avg"]:
        out["kv_evictable"] = s["stats"]["avg"] / t["stats"]["avg"]
        out["kv_blocks_total"] = t["stats"]["avg"]

    def h(name, want=None):
        s = ser(name, want)
        if not s:
            return None, 0.0, 0.0
        b = {(float("inf") if k == "+Inf" else float(k)): v for k, v in (s.get("buckets") or {}).items()}
        return b, s["stats"].get("sum", 0.0), s["stats"].get("count", 0.0)

    b, sm, ct = h("atom:decode_batch_size")
    if b is not None:
        out["dec_steps"], out["dec_rows"] = ct, (sm / ct if ct else float("nan"))
        dist, prev = {}, 0
        for le in sorted(b):
            if le == float("inf"):
                continue
            dist[int(le)] = (b[le] - prev) / ct if ct else 0
            prev = b[le]
        out["dec_dist"] = {k: round(v, 4) for k, v in dist.items() if v > 0.0005}
        out["dec_rows_p50"], out["dec_rows_p90"] = hquant(b, .5), hquant(b, .9)
    b, sm, ct = h("atom:prefill_batch_tokens")
    if b is not None:
        out["pf_steps"], out["pf_tok"], out["pf_tokens"] = ct, (sm / ct if ct else float("nan")), sm
        out["pf_tok_p90"] = hquant(b, .9)
    fb, fsum, fct = h("atom:gpu_forward_seconds", {"tp_rank": "0"})
    fwd_split(fb, fsum, fct, W, out)
    b, sm, ct = h("atom:request_queue_time_seconds")
    if b is not None:
        out["queue_mean"] = sm / ct if ct else float("nan")
        out["queue_p90"] = hquant(b, .9)

    def tot(name):
        s = ser(name)
        return s["stats"].get("total", 0.0) if s else 0.0

    full, cached = tot("atom:prefix_cache_full_tokens"), tot("atom:prefix_cache_cached_tokens")
    out["srv_hit"] = 100 * cached / full if full else float("nan")
    out["preempt"] = tot("atom:preemptions")
    out["gen_tok"] = tot("atom:generation_tokens")
    out["prompt_tok"] = tot("atom:prompt_tokens")
    out["lmc_loaded"] = tot("atom:lmcache_loaded_tokens")
    out["lmc_saved"] = tot("atom:lmcache_saved_tokens")
    return out


# ---------------- server log ----------------
TS = re.compile(r'^\[atom (\d\d):(\d\d):(\d\d)\]')
ENG = re.compile(r'Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, '
                 r'Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%')
POOL = re.compile(r'\[Pool Pressure\] paged: (\d+)/(\d+) used, (\d+) reusable-free, (\d+) vacant.*evicted: (\d+)')
MONO = re.compile(r'mono steps by token count: (\{.*\})')


def serverlog(tag, t_start, t_end):
    out = {}
    p = f"{D}/logs/server_{os.path.basename(tag)}.log"
    if not os.path.exists(p):
        return out
    day0 = dt.datetime.fromtimestamp(t_start, dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    base = day0.timestamp()
    last_sod, add = -1, 0.0
    eng, pools, monos, pf_lines = [], [], [], 0
    mono_pre = None
    with open(p, errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if not m:
                continue
            sod = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            if last_sod >= 0 and sod < last_sod - 3600:
                add += 86400
            last_sod = sod
            t = base + sod + add
            mm = MONO.search(line)
            if mm and t <= t_start:
                mono_pre = ast.literal_eval(mm.group(1))
            if not (t_start <= t <= t_end):
                continue
            if (e := ENG.search(line)):
                eng.append(tuple(float(x) for x in e.groups()))
            elif (q := POOL.search(line)):
                pools.append(tuple(int(x) for x in q.groups()))
            elif mm:
                monos.append(ast.literal_eval(mm.group(1)))
            elif "Scheduled prefill batch" in line:
                pf_lines += 1
    if eng:
        out["log_run_mean"] = sum(e[2] for e in eng) / len(eng)
        out["log_wait_mean"] = sum(e[3] for e in eng) / len(eng)
        out["log_wait_max"] = max(e[3] for e in eng)
        out["log_kv_max"] = max(e[4] for e in eng)
        out["log_pf_tps"] = sum(e[0] for e in eng) / len(eng)
        out["log_gen_tps"] = sum(e[1] for e in eng) / len(eng)
    if pools:
        out["evicted"] = pools[-1][4] - pools[0][4]
        out["pool_used_max"] = max(q[0] for q in pools) / pools[0][1]
        out["pool_vacant_min"] = min(q[3] for q in pools) / pools[0][1]
    if monos:
        last = monos[-1]
        first = mono_pre or {"mono": {}, "original": {}}
        diff = {k: {n: last[k].get(n, 0) - first[k].get(n, 0) for n in last[k]} for k in ("mono", "original")}
        diff = {k: {n: c for n, c in v.items() if c} for k, v in diff.items()}
        tm, to = sum(diff["mono"].values()), sum(diff["original"].values())
        out["mono_share"] = tm / (tm + to) if tm + to else float("nan")
        out["mono_steps"] = diff
    return out


def agg_fallback(tag):
    """When summarize_ix.py cannot read profile_export.jsonl (seen: a 32 KB NUL hole from NFS in ladder_mbt8k_c32),
    take the same quantities from aiperf's aggregate: ix = 1000 / p(full_response_inter_token_latency), tput =
    usage_total_tokens.sum / benchmark_duration.avg / 4 GPUs == the official record-span number (benchmark_duration is
    the record span; e.g. m3_C_035_1 214959905 / 902.0608 / 4 = 59574.7). milestone4 fix (aiperf-anomaly NOTES): NOT
    total_token_throughput / 4, whose phase window includes aiperf's grace wait (-3% when a leaked join times out,
    with nothing stalled)."""
    try:
        agg = json.load(open(f"{rdir(tag)}/profile_export_aiperf.json"))
    except Exception:
        return {}
    fr = agg.get("full_response_inter_token_latency") or {}
    md = agg.get("metadata") or {}
    tok = (agg.get("usage_total_tokens") or {}).get("sum")
    bd = (agg.get("benchmark_duration") or {}).get("avg")
    return {"tput_ix": tok / bd / 4 if tok and bd else None,
            "ix_p90": 1000 / fr["p90"] if fr.get("p90") else None, "ix_p50": 1000 / fr["p50"] if fr.get("p50") else None,
            "valid": md.get("submission_valid", True), "invalid_reasons": md.get("submission_invalid_reasons"),
            "theo_hit_ix": (agg.get("theoretical_prefix_cache_hit") or {}).get("avg", float("nan")) / 100,
            "span": bd, "official_src": "aiperf aggregate: usage_total_tokens.sum / benchmark_duration.avg / 4"}


def official(tag):
    """InferenceX numbers via summarize_ix.py (the team's exact re-implementation of the published metrics)."""
    import subprocess
    try:
        out = subprocess.run([sys.executable, SUMMARIZE_IX, rdir(tag), "--json"],
                             capture_output=True, text=True, timeout=300).stdout
        d = json.loads(out[out.index("["):])[0]
    except Exception:
        return agg_fallback(tag)
    if d.get("tput_per_gpu") is None:
        return agg_fallback(tag)
    e = d.get("errors") or {}
    return {"tput_ix": d.get("tput_per_gpu"), "ix_p90": d.get("intvty_p90"), "ix_p50": d.get("intvty_p50"),
            "e2eN_p90": d.get("e2e_norm_intvty_p90"), "span": d.get("span_s"), "isl_ix": d.get("mean_isl"),
            "valid": e.get("valid"), "invalid_reasons": e.get("reasons"), "coverage": e.get("coverage"),
            "gpu_hit": d.get("gpu_cache_hit"), "theo_hit_ix": d.get("theoretical_cache_hit"), "al": d.get("al"),
            "ttft50_ix": d.get("ttft_p50"), "ttft90_ix": d.get("ttft_p90")}


def stats(tag):
    c = client(tag)
    if c is None:
        return None
    s = {"tag": tag, **c}
    s.update(official(tag))
    # aiperf's "profiling" server-metric stats are cumulative since server start (warmup included: e.g. c32 prefill
    # tokens 16.7M vs 7.2M in the window); prefer our own snapshots diffed over the exact profiling window
    srv = server(tag, c["t_start"], c["t_end"])
    if not srv.get("dec_steps"):
        srv = server_aiperf(tag)
        srv["srv_src"] = "aiperf(cumulative incl. warmup)"
    else:
        srv["srv_src"] = "snapshots"
    s.update(srv)
    s.update(serverlog(tag, c["t_start"], c["t_end"]))
    th = s.get("theo_hit_ix")
    gh = s["srv_hit"] / 100 if s.get("srv_hit") == s.get("srv_hit") and s.get("srv_hit") else None
    if th and gh:
        s["hit_lost"] = th - gh                      # prefix hit lost to the finite pool (fraction of all prompt tokens)
        s["recompute_frac"] = (th - gh) / (1 - gh) if gh < 1 else float("nan")  # share of prefill that is re-prefill
    if s.get("pf_tokens") and s.get("win"):
        s["pf_tok_s"] = s["pf_tokens"] / s["win"]
    if SAVE:
        try:
            json.dump(s, open(f"{rdir(tag)}/ladder_stats.json", "w"), indent=1, default=str)
        except Exception:
            pass
    return s


def f(x, w, p=0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return " " * (w - 1) + "-"
    return f"{x:{w}.{p}f}"


HDR = (f"{'tag':26s} {'n':>4s} {'tput_ix':>8s} {'ix_p90':>6s} {'ix_p50':>6s} {'e2eN90':>6s} {'ok':>3s} "
       f"{'aipTput':>7s} {'ISL':>6s} "
       f"{'ttft50':>7s} {'ttft90':>7s} {'hit%':>5s} {'lost%':>5s} {'effC':>5s} {'run':>5s} {'wait':>5s} {'decRows':>7s} "
       f"{'pfKtok/s':>8s} {'decB':>5s} {'pfB':>5s} {'decMs':>6s} {'pfMs':>6s} {'kvMx':>5s} {'evict':>6s} {'mono%':>5s}")


def row(s):
    ok = {True: "ok", False: "BAD"}.get(s.get("valid"), "?")
    return (f"{s['tag']:26s} {s.get('n', 0):4d} {f(s.get('tput_ix'), 8)} {f(s.get('ix_p90'), 6, 1)} "
            f"{f(s.get('ix_p50'), 6, 1)} {f(s.get('e2eN_p90'), 6, 1)} {ok:>3s} {f(s.get('tput'), 7)} "
            f"{f((s.get('isl_ix') or s.get('isl') or 0) / 1000, 5)}K "
            f"{f(s.get('ttft50'), 7)} {f(s.get('ttft90'), 7)} {f(s.get('srv_hit'), 5, 1)} "
            f"{f(100 * s['hit_lost'] if 'hit_lost' in s else None, 5, 2)} {f(s.get('eff_conc'), 5, 2)} "
            f"{f(s.get('run'), 5, 2)} {f(s.get('wait'), 5, 2)} {f(s.get('dec_rows'), 7, 2)} "
            f"{f(s['pf_tok_s'] / 1000 if 'pf_tok_s' in s else None, 8, 1)} "
            f"{f(s.get('dec_busy_e'), 5, 2)} {f(s.get('pf_busy_e'), 5, 2)} {f(s.get('dec_step_ms_e'), 6, 2)} "
            f"{f(s.get('pf_step_ms_e'), 6)} {f(100 * s['kv_max'] if 'kv_max' in s else None, 5, 1)} "
            f"{f(s.get('evicted'), 6)} {f(100 * s['mono_share'] if 'mono_share' in s else None, 5, 1)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    table = "--table" in args
    js = "--json" in args
    SAVE = "--save" in args
    tags = [a for a in args if not a.startswith("--")]
    print(HDR)
    for t in tags:
        s = stats(t)
        if s is None:
            print(f"{t:26s} missing")
            continue
        print(row(s) + (" (partial)" if s.get("partial") else ""))
        if not table:
            print(f"   dec_dist={s.get('dec_dist')} pf_steps={s.get('pf_steps')} pf_tok_p90={s.get('pf_tok_p90')} "
                  f"queue_mean={f(s.get('queue_mean'), 0, 2)}s official_gpu_hit={s.get('gpu_hit')} "
                  f"theo_hit={s.get('theo_hit_ix')} recompute_frac={s.get('recompute_frac')} src={s.get('srv_src')} "
                  f"preempt={s.get('preempt')} mono={s.get('mono_steps')} lmc_loaded={s.get('lmc_loaded')}")
        if js:
            print(json.dumps(s, default=str))
