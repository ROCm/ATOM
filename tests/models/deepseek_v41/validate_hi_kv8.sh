#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# The V4.1 mono hi path on ATOM's fp8 KV pool (``--kv_cache_dtype fp8``) on 4 MI355X / MI350X GPUs:
#
#   1. the device unit checks on one GPU: the kv8 programs' rows against ATOM's quantize_fp8 / dequantize_rows
#      (tests/test_deepseek_v41_mono_hi_kv8_gpu.py) and the cache bridge's fp8 export / import (test_mono_hi_bridge.py
#      -k kv8_atom);
#   2. GSM8K (5-shot, lm_eval local-completions) on a TP4 server with the hi path forced (ATOM_DSV41_MONO_HI=1: a
#      deployment it refuses is an error) and on the same server with it off (ATOM's own mono decode), both fp8 KV,
#      real DSpark acceptance; the hi run's log must say the hi path serves (no "V4.1 mono hi off").
#
# Usage: MODEL_PATH=<DeepSeek-V4.1-Flash> GPUS=0,1,2,3 bash tests/models/deepseek_v41/validate_hi_kv8.sh [OUT_DIR]
# Optional env: PYTHON (python3), EXTRA_PYTHONPATH (prepended after the repo), EXTRA_ENV ("VAR=VALUE ..."),
#   PORT (8190), LIMIT (GSM8K questions, 400; 0: all), CONC (lm_eval concurrency, 8), READY_MIN (40),
#   SKIP_UNIT=1, SKIP_E2E=1.
# Exit 0 when every step passed; the summary is OUT_DIR/summary.txt.

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${1:-$PWD/hi_kv8_validate_$(date -u +%Y%m%dT%H%M%SZ)}
PY=${PYTHON:-python3}
PORT=${PORT:-8190}
LIMIT=${LIMIT:-400}
CONC=${CONC:-8}
GPUS=${GPUS:-0,1,2,3}
: "${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint}"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
FAILED=0
note() { echo "$*" | tee -a "$OUT/summary.txt"; }

if [ "${SKIP_UNIT:-0}" != 1 ]; then
    first=${GPUS%%,*}
    (cd "$ROOT" && HIP_VISIBLE_DEVICES=$first $PY -m pytest -q -p no:cacheprovider -rs \
        tests/test_deepseek_v41_mono_hi_kv8_gpu.py "tests/models/deepseek_v41/test_mono_hi_bridge.py::test_kv8_atom") \
        > "$OUT/unit.log" 2>&1
    rc=$?
    note "unit (HIP $first): rc $rc -- $(tail -n 1 "$OUT/unit.log")"
    grep -q "skipped" "$OUT/unit.log" && { note "unit: a check skipped (not an MI355X?)"; FAILED=1; }
    [ $rc -eq 0 ] || FAILED=1
fi

serve() {  # serve <name> <ATOM_DSV41_MONO_HI>: start, wait, GSM8K, stop
    local name=$1 hi=$2 log="$OUT/$1.server.log"
    # shellcheck disable=SC2086
    env HIP_VISIBLE_DEVICES="$GPUS" ATOM_DSV41_MONO_HI="$hi" OMP_NUM_THREADS=4 ATOM_DISABLE_MMAP=true \
        AITER_LOG_LEVEL=WARNING PYTHONUNBUFFERED=1 ${EXTRA_ENV:-} \
        setsid $PY -u -m atom.entrypoints.openai_server --model "$MODEL_PATH" --served-model-name v41 \
        --trust-remote-code --host 127.0.0.1 --server-port "$PORT" --tensor-parallel-size 4 \
        --kv_cache_dtype fp8 --index-cache-dtype fp4 --gpu-memory-utilization 0.75 \
        --max-num-batched-tokens 16384 --block-size 16 --level 3 --cudagraph-mode FULL \
        --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,16,24,32,48]' \
        --method dspark --num-speculative-tokens 5 > "$log" 2>&1 &
    local pid=$! t=0
    until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
        sleep 20; t=$((t + 20))
        if ! kill -0 $pid 2> /dev/null || [ $t -gt $((${READY_MIN:-40} * 60)) ]; then
            note "$name: the server did not come up (see $log)"; kill -- -$pid 2> /dev/null; FAILED=1; return
        fi
    done
    grep -E "V4.1 mono hi" "$log" | head -n 5 > "$OUT/$name.hi_lines.txt"
    local lim=(); [ "$LIMIT" != 0 ] && lim=(--limit "$LIMIT")
    $PY -m lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 "${lim[@]}" \
        --model_args "model=v41,tokenizer=$MODEL_PATH,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=$CONC,max_retries=3,tokenized_requests=False" \
        --output_path "$OUT/$name.lm_eval" > "$OUT/$name.lm_eval.log" 2>&1
    note "$name (ATOM_DSV41_MONO_HI=$hi): $(grep -E 'exact_match.*(strict|flexible)' "$OUT/$name.lm_eval.log" | tr -s ' ' | head -n 2 | tr '\n' ' ')"
    kill -- -$pid 2> /dev/null; wait $pid 2> /dev/null
    sleep 30  # (the GPUs' memory released)
}

if [ "${SKIP_E2E:-0}" != 1 ]; then
    serve hi_fp8 1
    if grep -q "V4.1 mono hi off" "$OUT/hi_fp8.server.log" 2> /dev/null; then
        note "hi_fp8: the hi path did not serve: $(grep -m1 'V4.1 mono hi off' "$OUT/hi_fp8.server.log")"; FAILED=1
    fi
    serve atom_fp8 0
fi
note "done: $([ $FAILED = 0 ] && echo PASS || echo FAIL) ($OUT)"
exit $FAILED
