#!/bin/bash
set -euo pipefail

ROUTER_URL="${ROUTER_URL:-http://10.24.112.168:8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/DeepSeek-V4-Pro/}"
ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-32,64,128,256,512,1024}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
RESULT_DIR="${RESULT_DIR:-/workspace/benchmark_results}"

if [[ ! -d /tmp/sglang-benchmark/bench_serving ]]; then
    rm -rf /tmp/sglang-benchmark
    mkdir -p /tmp/sglang-benchmark
    git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/sglang-benchmark/bench_serving
fi

mkdir -p "${RESULT_DIR}"

IFS=',' read -ra ISLS <<< "${ISL_LIST}"
IFS=',' read -ra CONCS <<< "${CONC_LIST}"

for ISL in "${ISLS[@]}"; do
    for CONC in "${CONCS[@]}"; do
        echo ""
        echo "========================================="
        echo "[bench] ISL=${ISL} OSL=${OSL} CONC=${CONC}"
        echo "========================================="

        PYTHONDONTWRITEBYTECODE=1 python /tmp/sglang-benchmark/bench_serving/benchmark_serving.py \
            --model="${MODEL_PATH}" \
            --backend=vllm \
            --base-url="${ROUTER_URL}" \
            --dataset-name=random \
            --random-input-len="${ISL}" \
            --random-output-len="${OSL}" \
            --random-range-ratio "${RANDOM_RANGE_RATIO}" \
            --num-prompts=$(( CONC * 10 )) \
            --max-concurrency="${CONC}" \
            --trust-remote-code \
            --num-warmups=$(( 2 * CONC )) \
            --request-rate=inf \
            --ignore-eos \
            --save-result \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --result-dir="${RESULT_DIR}" \
            --result-filename="pd-atom-mesh-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json"
    done
done

echo ""
echo "========================================="
echo "[bench] summary"
echo "========================================="

python3 -c "
from pathlib import Path
import json

result_dir = Path('${RESULT_DIR}')
json_files = sorted(result_dir.glob('pd-atom-mesh-*.json'))
if not json_files:
    print('No result files found')
    exit(0)

print(f\"{'Config':<25} {'TTFT(ms)':>10} {'ITL(ms)':>10} {'Throughput(tok/s)':>18}\")
print('-' * 65)
for f in json_files:
    d = json.load(open(f))
    isl = d.get('random_input_len', '?')
    osl = d.get('random_output_len', '?')
    conc = d.get('max_concurrency', '?')
    ttft = d.get('mean_ttft_ms', 0)
    itl = d.get('mean_itl_ms', 0)
    tp = d.get('output_throughput', 0)
    print(f'{isl}/{osl} c={conc:<6} {ttft:>10.1f} {itl:>10.2f} {tp:>18.1f}')
"

echo "[bench] results saved to ${RESULT_DIR}"
