#!/usr/bin/env bash
# Ladder test: does a PARTIAL prefix match resume from an intermediate rung?
#
# The 8x8 script cannot see this. Its 1213-token prompt crosses exactly one
# rung -- the prompt-end anchor -- which is the case that worked even when
# decode asked `checkpoint_cut` once for the whole prompt. The ladder only
# exists on prompts long enough to cross several, and it only MATTERS when a
# later request shares part of a prompt rather than all of it.
#
#   wave 1:  8 x LONG prompt            (crosses several rungs)
#   wave 2:  8 x a PREFIX of that       (shares ~60%, then diverges)
#
# Wave 2's shared prefix ends mid-prompt, so it can only resume from a
# checkpoint at or below that point:
#
#   one checkpoint per prompt (old)  -> the only image sits near the END of
#                                       wave 1's prompt, past the match, so
#                                       nothing is reachable:  cached=0
#   a ladder (new)                   -> the largest rung below the match is
#                                       reachable:             cached>0
#
# That is the whole discriminator, and it needs 16 requests rather than a
# benchmark.
#
# Usage: PORT=8000 MODEL=... ./prefix_cache_ladder.sh [logfile]

set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
RANKS="${RANKS:-8}"
# ~10 tokens per repetition. 5000 -> ~50k tokens, which at a 16384 budget is
# four chunks and three interior rungs plus the anchor.
LONG_REPS="${LONG_REPS:-5000}"
# The shared fraction. 0.6 puts the divergence point well inside the prompt,
# between two rungs, so the resume has to step BACK to one.
SHORT_REPS="${SHORT_REPS:-3000}"
LOG="${1:-}"

send() {  # send <rank> <wave> <reps>
  python3 - "$MODEL" "$1" "$2" "$3" <<'PY' |
import json, sys
model, rank, wave, reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
base = "You are a helpful assistant. " + "The quick brown fox jumps over the lazy dog. " * reps
print(json.dumps({
    "model": model,
    "prompt": f"{base} Question w{wave}r{rank}.",
    "max_tokens": 8,
    "temperature": 0,
    "data_parallel_rank": int(rank),
}))
PY
  curl -s -o /dev/null -w "  w$2 r$1 http=%{http_code} ttft~%{time_total}s\n" \
    "http://localhost:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' --data-binary @-
}

echo "--- wave 1: $RANKS long prompts (~$((LONG_REPS / 100))00 tokens), one per rank ---"
for r in $(seq 0 $((RANKS - 1))); do send "$r" 1 "$LONG_REPS" & done
wait

echo "--- settling 5s ---"
sleep 5

echo "--- wave 2: $RANKS prompts sharing only the first $SHORT_REPS reps ---"
for r in $(seq 0 $((RANKS - 1))); do send "$r" 2 "$SHORT_REPS" & done
wait

if [[ -n "$LOG" ]]; then
  echo
  echo "=== 1. was a ladder built? (expect several rungs, not one) ==="
  grep -o "prefill-cut\] seq [0-9]*: [0-9]* rung(s) \[[^]]*\]" "$LOG" | tail -n "$RANKS"
  echo
  echo "=== 2. did prefill chunk? (expect repeated 16384s per request) ==="
  grep -o "prefill iter [0-9.]*ms | reqs=[0-9]* | tokens=[0-9]*" "$LOG" \
    | grep -o "tokens=[0-9]*" | sort | uniq -c | sort -rn | head
  echo
  echo "=== 3. THE TEST: wave-2 resume point ==="
  echo "    cached=0 on every rank  -> nothing below the match was reachable"
  echo "    cached>0                -> an interior rung was reachable (ladder)"
  grep -o "prefill-cut\] seq [0-9]*: [0-9]* rung(s).*cached=[0-9]*" "$LOG" \
    | grep -o "cached=[0-9]*" | sort | uniq -c
  grep -o "drain#1 start=[0-9-]*" "$LOG" | sort | uniq -c
  echo
  echo "=== 4. the cap: rungs per prompt (expect <= MAX_PREFILL_CHECKPOINTS) ==="
  echo "    a count that DEGRADES across requests means reservations are not"
  echo "    being released -- the pinning cost leaking rather than bounded."
  grep -o "prefill-cut\] seq [0-9]*: [0-9]* rung(s)" "$LOG" \
    | grep -o "[0-9]* rung" | sort -n | uniq -c
  echo
  echo "=== 5. the cap keeps the TAIL (first rung should not be the lowest) ==="
  grep -o "prefill-cut\] seq [0-9]*: [0-9]* rung(s) \[[^]]*\]" "$LOG" | tail -3
  echo
  echo "=== 6. LEAK CHECK: KV usage once everything has drained ==="
  echo "    should fall back to ~0%. Elevated with Running:0/Waiting:0 means"
  echo "    units reserved by dropped rungs were never released."
  grep -o "Decode Engine [0-9]*:.*Running: [0-9]* reqs, Waiting: [0-9]* reqs, GPU KV cache usage: [0-9.]*%" "$LOG" \
    | tail -n "$RANKS"
  echo
  echo "=== 7. images actually written (one scatter per consumed rung) ==="
  grep -c "prefix-hash.*publish" "$LOG" || true
fi
