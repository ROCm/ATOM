#!/usr/bin/env bash
# Prefix-cache smoke test for rapidserve + DP attention.
#
# Two waves of 8 requests sharing one long prefix. Wave 1 warms every rank;
# wave 2, five seconds later, should hit on every rank.
#
# Each request PINS its `data_parallel_rank` rather than trusting the load
# balancer. That matters: every DP rank runs its own BlockManager over its own
# KV pool, so a wave that happens to land twice on rank 3 and never on rank 5
# leaves rank 5 cold and rank 3 warm, and the aggregate is unreadable. Pinning
# rank i to request i makes each rank see exactly one request per wave.
#
# Usage: PORT=8000 MODEL=deepseek-ai/DeepSeek-V4-Pro ./prefix_cache_8x8.sh [logfile]

set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
RANKS="${RANKS:-8}"
LOG="${1:-}"

# ~1200 tokens of natural text. Natural, not random token ids: decoding random
# ids to text and re-encoding does not round-trip, so a "shared prefix" built
# that way is not guaranteed to survive as the same leading tokens. Plain
# English re-encodes to itself.
PREFIX=$(python3 -c 'print("You are a helpful assistant. " + "The quick brown fox jumps over the lazy dog. " * 120, end="")')

send() {  # send <rank> <wave>
  local rank="$1" wave="$2"
  # The unique tail lands in the trailing partial block, which the reuse
  # ceiling excludes anyway (prefill must forward at least one block to
  # produce logits) -- so it costs nothing measurable and keeps the two waves
  # from being byte-identical requests.
  python3 - "$PREFIX" "$rank" "$wave" "$MODEL" <<'PY' |
import json, sys
prefix, rank, wave, model = sys.argv[1:5]
print(json.dumps({
    "model": model,
    "prompt": f"{prefix} Question w{wave}r{rank}.",
    "max_tokens": 8,
    "temperature": 0,
    "data_parallel_rank": int(rank),
}))
PY
  curl -s -o /dev/null -w "  rank %{url_effective} http=%{http_code} t=%{time_total}s\n" \
    "http://localhost:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' --data-binary @-
}

wave() {  # wave <n>
  local n="$1"
  echo "--- wave $n: $RANKS requests, one per DP rank ---"
  for r in $(seq 0 $((RANKS - 1))); do
    send "$r" "$n" &
  done
  wait
}

wave 1
echo "--- settling 5s so wave 1 publishes before wave 2 probes ---"
sleep 5
wave 2

if [[ -n "$LOG" ]]; then
  echo
  echo "=== wave 2 probes (expect block_id != -1, tok_match=True) ==="
  grep "prefix-hash" "$LOG" | grep probe | tail -n $((RANKS * 2))
  echo
  echo "=== per-rank hit summary ==="
  # One line per rank: how many probes found a block vs missed. A rank that
  # only ever misses is a rank whose pool never saw wave 1.
  grep "prefix-hash" "$LOG" | grep probe | sed 's/.*\(dp=[0-9-]*\).*block_id=\([0-9-]*\).*/\1 \2/' \
    | awk '{ if ($2 == "-1") miss[$1]++; else hit[$1]++ }
           END { for (d in miss) printf "%s miss=%d hit=%d\n", d, miss[d], hit[d]+0;
                 for (d in hit) if (!(d in miss)) printf "%s miss=0 hit=%d\n", d, hit[d] }' \
    | sort
  echo
  echo "=== admissions: pending_start > 0 means a hit was found ==="
  grep "prefix-publish" "$LOG" | grep -o "pending_start=[0-9-]*" | sort | uniq -c
fi
