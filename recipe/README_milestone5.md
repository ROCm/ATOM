# milestone5 recipe (prepared by prefill-dispatch, round 15): not run
- New arms on p1/integrated9 b790686e8 (= int8 + SP prefill dispatch by step tokens):
  C4  = C3 + SP piecewise graph 513-1792 tokens (buckets 1024/1280/1536/1792) + SP TBO >= 10240 tokens (gen mode).
  C4T = C3 + SP TBO >= 10240 only (the bit-identical part; keeps the TP path below 2048).
- Launch as milestone4 (series.sh / launch.sh, CAP <= 2400 s per ATOM lease), e.g. C4:m5_C4_<node>_1 C3:m5_C3_<node>_1.
- Expected (prefill-dispatch NOTES, estimates): C4 - C3 = -4.7..-4.8 GPU s per 900 s window (0.53% of wall), sim +0.2% throughput
  (sd 0.3), TTFT mean -0.017 s. Below the 0.55% run-to-run range: n >= 2 per arm.
- Before the e2e: a multi-request TBO bitwise check (one 4-GPU lease) is still open; C4's < 2048 graph part is SP numerics on a
  range that was TP (needs the GSM8K gate per the rules, or run C4T).
- m5-smoke (round 17): arm C8 = C3 + scheduler-owned midpoint SP TBO (p1/integrated12 30ba3fd20 = int8 + balance_full.patch):
  ATOM_P1_SP_TBO=1 _PAIR=1 _MODE=gen _MIN_TOKENS=10240 (inert) + ATOM_P1_SCHED_BALANCE_TBO_MIN=10240 _TBO_MAX=24575
  ATOM_P1_SCHED_BALANCE_SNAP_PCT=10 (NOT ..._TBO_SNAP_PCT: unread). Launch e.g.
  GPUS=0,1,2,3 CAP=2400 bash series.sh C8:m5_C8_034_1 C3:m5_C3_034_1. Smoke: ../NOTES.md (clean; one allocator retry after an
  idle 22K pair -> fallback ATOM_P1_SCHED_BALANCE_TBO_MAX=16384). Needs its own GSM8K gate (batch composition changes).
