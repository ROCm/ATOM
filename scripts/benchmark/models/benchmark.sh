#!/bin/bash
set -euo pipefail


if [ ! -e "$MODEL_PATH" ]; then
    echo "model '$MODEL_PATH' does not exist, please set MODEL_PATH firstly."
    exit 1
fi

if [ -f "$LOG_FILE" ]; then
    rm "$LOG_FILE"
fi

# health check
curl -sf "http://localhost:$PORT/health" > /dev/null || {
    echo "ERROR: Server not running on port $PORT at vllm backend"
    exit 1
}

for ISL in "${ISL_LOOP[@]}"; do
    for CONC in "${CONC_LOOP[@]}"; do
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
            --random-range-ratio 1 \
            --num-prompts=$(( CONC * 2)) \
            --max-concurrency="$CONC" \
            --request-rate=inf \
            --ignore-eos \
            --percentile-metrics="ttft,tpot,itl,e2el"  2>&1 | tee -a "$LOG_FILE"
        
        echo -e "\n" | tee -a "$LOG_FILE"
    done
done

for COMBINATION in "${ISL_OSL_CONC[@]}"; do

    IFS=':' read -r ISL OSL CONC <<< "$COMBINATION"
    
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