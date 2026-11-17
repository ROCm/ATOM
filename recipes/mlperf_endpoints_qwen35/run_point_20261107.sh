#!/bin/bash
# One MLPerf Endpoints agentic point of the 2026-11-07 configuration (agentic_20261107.env), with the submission
# client settings: stop_issuing_on_first_user_complete false, inline accuracy, salt and tool delay, 6743 trajectories
# (11 x 613), a 10-minute issue cap (the client's max_issue_duration, not a kill). The client runs to its own end
# (report/performance/result_summary.json + inline accuracy); METRICS_ISL=0 skips the ISL metric, whose end-of-run
# tokenization of ~2-4B prompt tokens on one core otherwise holds the report back for hours.
#   usage: run_point_20261107.sh [concurrency=1536] [run dir=$R/c<concurrency>_20261107]
#   env:   AITER_DIR (aiter tree; ATOM is always this script's tree), R (work dir with the dataset, default /work/qwen35_ep)
#          INTERVAL / PREFILL_TOKENS override the per-concurrency defaults below; EXTRA="K=V ..." is exported after
#          the env files (for A/B flags); CAP_MIN (10), NUM_TRAJ (6743).
# Decode interval (decode steps after each prefill) by concurrency, measured 2026-11-06/07 with ATOM_GDN_STATE_MIDSTEP
# (complete runs): each point is the non-dominated one of its bracket.
#   c >= 1536: 10 with a 20,480-token prefill budget (the highest tok/s; 20 is the balanced point)
#   c 1024-1535: 24 (from the trend: 8 and 16 are dominated by c1536 at 20)   c 768-1023: 32   c 512-767: 48
#   c 384-511: 80   c 256-383: 128   c < 256: 96 (192 does not pay at c128)
set -e
C=${1:-1536}
HERE=$(cd "$(dirname "$0")" && pwd)
# Always this recipe's own tree: images export ATOM_DIR to their copy, which must not win.
export ATOM_DIR=$(cd "$HERE/../.." && pwd)
AIT=${AITER_DIR:?set AITER_DIR to the aiter tree}
export R=${R:-/work/qwen35_ep}
RUN=${2:-$R/c${C}_20261107}
_CAP=${CAP_MIN:-10}  # read before the recipe env (which sets CAP_MIN=600) is sourced
cd "$HERE"
set -a; . ./agentic_20261101.env; . ./agentic_20261107.env; set +a
[ -n "${EXTRA:-}" ] && export $EXTRA
if   [ "$C" -ge 1536 ]; then I=10;  P=20480
elif [ "$C" -ge 1024 ]; then I=24;  P=16384
elif [ "$C" -ge 768 ];  then I=32;  P=16384
elif [ "$C" -ge 512 ];  then I=48;  P=16384
elif [ "$C" -ge 384 ];  then I=80;  P=16384
elif [ "$C" -ge 256 ];  then I=128; P=16384
else                         I=96;  P=16384; fi
export ATOM_PREFILL_DECODE_INTERVAL=${INTERVAL:-$I} PREFILL_TOKENS=${PREFILL_TOKENS:-$P}
if [ "$PREFILL_TOKENS" -gt 16384 ]; then
  export ATOM_TOKEN_GRAPH_SIZES=${ATOM_TOKEN_GRAPH_SIZES:-256,512,1024,1536,2048,3072,4096,5120,6144,7168,8192,10240,12288,14336,16384,$PREFILL_TOKENS}
fi
export PYTHONPATH=/opt/qwen35/mori/python:/opt/flydsl-0.4.0:$AIT
export AITER_LOG_LEVEL=WARNING
# submission client settings (never STOP_FIRST=1)
export NUM_TRAJ=${NUM_TRAJ:-6743} STOP_FIRST=0 ACC_INLINE=1 CAP_MIN=$_CAP METRICS_TOKENIZER_WORKERS=16
export METRICS_ISL=${METRICS_ISL:-0}
echo "c=$C model $MODEL decode interval $ATOM_PREFILL_DECODE_INTERVAL, prefill budget $PREFILL_TOKENS tokens, KV $KV, MXFP8_1X32 $ATOM_MXFP8_1X32"
[ -s $R/dataset/mlperf_agentic_inference_dataset/mlperf_agentic_inference_dataset.jsonl ] || bash ./get_dataset.sh
ulimit -c 0
rm -rf /root/.cache/atom/*
mkdir -p $RUN; env | sort > $RUN.launch_env
STOP_S=0 exec ./bench_point.sh "$C" "$RUN"
