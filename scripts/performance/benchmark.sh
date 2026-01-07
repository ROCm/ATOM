#!/bin/bash
set -euo pipefail

# step 1: please pip install pandas openpyxl
# step 3: change the variables as below
# step 4: bash benchmark.sh

##################### need changed #######################
MODEL_PATH="/data/models/deepseek-ai/DeepSeek-V3.2-Exp/"
ISL_LIST=(1023 2047 4093 6143)
CONC_LIST=(8 4 1)
OSL=1024
PORT=8888
##########################################################

LOG_FILE="result.txt"
# clean old file
if [ -f "$LOG_FILE" ]; then
    rm "$LOG_FILE"
fi

# health check
curl -sf "http://localhost:$PORT/health" > /dev/null || {
    echo "ERROR: Server not running on port $PORT at vllm backend"
    exit 1
}

for ISL in "${ISL_LIST[@]}"; do

    for CONC in "${CONC_LIST[@]}"; do
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "Start Test ISL=$ISL, OSL=$OSL, CONC=$CONC" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
        
        python -m atom.benchmarks.benchmark_serving \
            --model="$MODEL_PATH" \
            --backend=vllm \
            --base-url="http://localhost:$PORT" \
            --dataset-name=random \
            --random-input-len="$ISL" \
            --random-output-len="$OSL" \
            --random-range-ratio 0.8 \
            --num-prompts=$(( CONC * 10)) \
            --max-concurrency="$CONC" \
            --request-rate=inf \
            --ignore-eos \
            --percentile-metrics="ttft,tpot,itl,e2el"  2>&1 | tee -a "$LOG_FILE"
        
        echo -e "\n" | tee -a "$LOG_FILE"
    done
done

if [ -f "$LOG_FILE" ]; then
    python save_csv.py $LOG_FILE
fi