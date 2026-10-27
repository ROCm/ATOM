#!/bin/bash
# One 30-minute MLPerf Endpoints (agentic) point for Qwen3.5-35B-A3B on gfx1250: a fresh server,
# smoke and prefix-cache checks, the MLCommons client, an output check. Runs inside the serving
# container.  usage: run_ep.sh <concurrency>
#
# Concurrency: 1-2048, a power of two or three times one (1 2 3 4 6 8 12 16 24 ... 768 1024
# 1536 2048). c <= 16: TP4 + MTP3; above: DP4 attention + EP4 MoE (MODE=tp|dp, SPEC=<n> override).
# 30-minute points (defaults): CAP_MIN=30, STOP_FIRST=1, TRAJ_X=4 (trajectories = max(613, 4 c):
# the cap, not an exhausted pool, ends every point), TURN_TIMEOUT=14400 (the benchmark's own; the drain
# after the cap lasts until the in-flight turns finish).
# Full compliant run: CAP_MIN=600 STOP_FIRST=0. SMOKE_ONLY=1 stops after the server checks.
C=$1
R=${R:-/work/qwen35_ep}; RUNS=${RUNS:-$R/runs}; D=$RUNS/c$C; ST=$RUNS/status.txt; mkdir -p $D
S=$(cd "$(dirname "$0")" && pwd)   # serve_ep.sh, mkcfg.py, check_outputs.py live next to this script
MODEL=${MODEL:-/data/Qwen3.5-35B-A3B-MXFP4}
DATASET=${DATASET:-$R/dataset/mlperf_agentic_inference_dataset/mlperf_agentic_inference_dataset.jsonl}
EP=${EP:-/app/mlcommons-endpoints}; VENV=${VENV:-$EP/venv}; BASE=$EP/examples/10_Agentic_Inference/qwen_agentic_benchmark.yaml

valid() { local m=$1; [ "$m" -ge 1 ] && [ "$m" -le 2048 ] || return 1
  while [ $((m % 2)) -eq 0 ]; do m=$((m / 2)); done; [ $m -eq 1 ] || [ $m -eq 3 ]; }
[[ "$C" =~ ^[0-9]+$ ]] && valid "$C" || { echo "usage: run_ep.sh <c>, c in 1-2048, 2^k or 3*2^k (got '$C')"; exit 2; }
MODE=${MODE:-$([ "$C" -le 16 ] && echo tp || echo dp)}
export SPEC=${SPEC:-$([ "$MODE" = tp ] && echo 3 || echo 0)}
# sequences an engine: every concurrent trajectory, with 25% slack under DP for the session-affinity
# imbalance, at most SEQ_CAP=512. On the 2026-10-26 gfx1250 images an engine decoding more than 512
# sequences returns partly garbage (see the recipe's Known issues); concurrency above that queues.
if [ "$MODE" = dp ]; then N=$(( (C * 5 / 4 + 3) / 4 )); [ $N -lt 16 ] && N=16; else N=$(( C < 64 ? 64 : C )); fi
[ $N -gt ${SEQ_CAP:-512} ] && N=${SEQ_CAP:-512}
echo "START $(date -u +%FT%TZ) c=$C mode=$MODE spec=$SPEC max_num_seqs=$N" >> $ST

# the images ship only versioned ROCm libs (libamdhip64.so.7, ...); aiter JIT modules such as
# top_p_sampling_from_probs link against the unversioned names and fail to build without these
for d in /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib /opt/rocm/lib; do
  for f in $d/*.so.[0-9]*; do b=${f%%.so.*}.so; [ -e $b ] || ln -s $(basename $f) $b 2>/dev/null; done
done
pkill -f atom.entrypoints.openai_server; sleep 10; pkill -9 -f atom.entrypoints.openai_server; pkill -9 -f 'ATOM[:][:]'; sleep 5
rm -rf ~/.cache/atom
MAXSEQ=$N bash $S/serve_ep.sh $MODE $D & SPID=$!
for i in $(seq 1 360); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER_DIED c=$C $(date -u +%FT%TZ)" >> $ST; exit 1; }
  curl -s -m 5 localhost:8000/v1/models | grep -q Qwen && { echo "READY $(date -u +%FT%TZ)" >> $ST; break; }
  sleep 10
done
grep -q '^HIP_FORCE_DEV_KERNARG=1$' $D/server_env.txt || { echo "KERNARG_NOT_SET" >> $ST; exit 1; }
# smoke: the client's sampling params (top_k 20, presence_penalty 1.5), a tool, and its session header
curl -s -m 900 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -H 'X-Session-ID: smoke-1' -d '{
  "model":"'$MODEL'","max_tokens":2048,"temperature":1.0,"top_k":20,"top_p":0.95,
  "presence_penalty":1.5,"repetition_penalty":1.0,"chat_template_kwargs":{"preserve_thinking":true},
  "tools":[{"type":"function","function":{"name":"get_order","description":"Look up an order","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}}}],
  "messages":[{"role":"user","content":"Where is my order 43E33A3CAA? Use the tool."}]}' > $D/smoke.json
grep -q '"tool_calls":\[{' $D/smoke.json && echo "SMOKE_OK" >> $ST || { echo "SMOKE_FAIL (no tool call): $(head -c 300 $D/smoke.json)" >> $ST; exit 1; }
# prefix cache: the same greedy request twice (the second hits the cache) must give the same text
P=$(python3 -c "print(' '.join(f'Line {i}: the quick brown fox jumps over the lazy dog.' for i in range(400)))")
for i in 1 2; do
  curl -s -m 300 localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"'$MODEL'",
    "prompt":"'"$P"' Summarize the text above in one sentence.","max_tokens":64,"temperature":0}' |
    python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])" > $D/prefix_$i.txt
done
cmp -s $D/prefix_1.txt $D/prefix_2.txt && echo "PREFIX_OK" >> $ST || { echo "PREFIX_MISMATCH" >> $ST; exit 1; }
[ "${SMOKE_ONLY:-0}" = 1 ] && { echo "SMOKE_ONLY_DONE $(date -u +%FT%TZ)" >> $ST; pkill -f atom.entrypoints.openai_server; exit 0; }

$VENV/bin/python $S/mkcfg.py $BASE $DATASET $MODEL $C $D/report $D/client.yaml > /dev/null || { echo "CFG_FAIL c=$C" >> $ST; exit 1; }
echo "BENCH c=$C $(date -u +%FT%TZ)" >> $ST
DRAIN=${TURN_TIMEOUT:-14400}
(cd $EP && timeout $(( ${CAP_MIN:-30} * 60 + DRAIN + 900 )) $VENV/bin/inference-endpoint benchmark from-config --config $D/client.yaml --mode perf > $D/client.log 2>&1); rc=$?
curl -s -m 10 localhost:8000/metrics > $D/server_metrics.txt 2>/dev/null
kill -0 $SPID 2>/dev/null || echo "SERVER_DIED during c=$C rc=$rc" >> $ST
echo "DONE c=$C rc=$rc $(date -u +%FT%TZ) $(grep -m1 -E 'Results: [0-9]+/' $D/client.log | sed 's/.*Results/Results/')" >> $ST
# outputs must be text, not token salad (the >512-sequence failure mode): flag a point above 0.5%
[ -f $D/report/events.jsonl ] && echo "OUTPUTS c=$C $(python3 $S/check_outputs.py $D/report/events.jsonl | head -1)" >> $ST
pkill -f atom.entrypoints.openai_server
