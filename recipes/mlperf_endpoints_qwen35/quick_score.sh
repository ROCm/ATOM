#!/bin/bash
# Quick A/B scoring (option 2): wait for the client's scored window to close (STOP_PERFORMANCE_TRACKING, then every
# turn tracked in the window completed), score it from events.jsonl with the MLCommons client's own code
# (replay_report.py: same OSL / TPOT rules and window as result_summary.json; matched it to the token at c1536),
# write <run dir>.quick_summary.json, then stop the run (client, harness, server). Not for record runs: those go to
# completion for inline accuracy and the official result_summary.json.
#   usage (inside the container): quick_score.sh <run dir> [poll s=60]
set -u
RUN=$1; POLL=${2:-60}
ST=$RUN/status.txt
until grep -q "^BENCH" $ST 2>/dev/null; do grep -qE "DIED|FAIL" $ST 2>/dev/null && { echo "run failed before BENCH"; exit 1; }; sleep 10; done
b=$(date -u -d "$(grep -m1 ^BENCH $ST | grep -oE '[0-9T:-]+Z$')" +%s)
# The client finished by itself (low concurrency: no user completes its trajectories before the issue cap, so there is
# no stop event, and events.jsonl moves to the report dir): score from its official result_summary.json instead.
score_complete() {
  RS=$(ls $RUN/c*/report/performance/result_summary.json 2>/dev/null | head -1)
  grep -qE "^DONE" $ST && [ -n "$RS" ] || return 1
  python3 - "$RS" "$RUN.quick_summary.json" <<'PY' | tee -a $ST
import json, sys
d = json.load(open(sys.argv[1])); p90 = d["tpot"]["percentiles"]["90.0"] / 1e6
o = {"source": "result_summary.json (complete run)", "complete": d["complete"], "tps": d["tps"], "qps": d["qps"],
     "tpot_ms": {"0.9": p90}, "tps_per_user_p90": 1000 / p90, "tracked_duration_s": d["duration_ns"] / 1e9,
     "n_tracked_completed": d["n_samples_completed"], "osl_mean": d["output_sequence_lengths"]["avg"]}
json.dump(o, open(sys.argv[2], "w"), indent=1)
print("QUICK tps %.1f  p90 TPOT %.3f ms -> %.1f tok/s/user  window %.0f s  turns %d  OSL %.1f  (complete run)"
      % (o["tps"], p90, o["tps_per_user_p90"], o["tracked_duration_s"], o["n_tracked_completed"], o["osl_mean"]))
PY
  echo "QUICK_DONE $(date -u +%FT%TZ)" >> $ST; exit 0
}
EV=""
until [ -n "$EV" ]; do
  for d in $(ls -dt /dev/shm/benchmark_cli_benchmark_* 2>/dev/null); do
    [ -f $d/events/events.jsonl ] && [ $(stat -c %W $d 2>/dev/null || stat -c %Y $d) -ge $((b - 120)) ] && { EV=$d/events/events.jsonl; break; }
  done
  [ -z "$EV" ] && { score_complete; sleep 10; }
done
echo "[$(TZ=America/Chicago date +%T) CT] events: $EV"
until LC_ALL=C grep -q -m1 '"session.stop_performance_tracking"' $EV; do grep -qE "^DONE|DIED" $ST && break; sleep $POLL; done
echo "[$(TZ=America/Chicago date +%T) CT] scored window closed; waiting for its tracked turns"
score_complete
while true; do
  [ -f $EV ] || { sleep 20; score_complete; }
  out=$(/app/mlcommons-endpoints/venv/bin/python $(dirname "$0")/replay_report.py $EV 48 2>/dev/null)
  open=$(echo "$out" | grep -oE '[0-9]+ tracked never completed' | grep -oE '^[0-9]+')
  [ "${open:-1}" = 0 ] && break
  echo "  $open tracked turns still open"; sleep $POLL
done
echo "$out" | sed -n '/^{/,$p' > $RUN.quick_summary.json
echo "$out" | head -1
python3 -c "import json;d=json.load(open('$RUN.quick_summary.json'));print('QUICK tps %.1f  p90 TPOT %.3f ms -> %.1f tok/s/user  window %.0f s  turns %d  OSL %.1f'%(d['tps'],d['tpot_ms']['0.9'],d['tps_per_user_p90'],d['tracked_duration_s'],d['n_tracked_completed'],d['osl_mean']))" | tee -a $ST
echo "[$(TZ=America/Chicago date +%T) CT] stopping the run"
pkill -f "[i]nference-endpoint benchmark"; pkill -f "[b]ench_point.sh"; pkill -f "[r]un_ep.sh"; pkill -f "[a]tom.entrypoints.openai_server"
sleep 10; pkill -9 -f "[a]tom.entrypoints.openai_server|ATOM[:][:]|[M]odelRunner"; pkill -9 -f "[i]nference-endpoint"
echo "QUICK_DONE $(date -u +%FT%TZ)" >> $ST
