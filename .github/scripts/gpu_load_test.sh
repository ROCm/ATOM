#!/bin/bash

#############################################
# GPU Load Test for DeepSeek-R1 Model
# Tests if all GPUs load simultaneously or if any GPU is slow
#############################################

MODEL_NAME="deepseek-ai/DeepSeek-R1-0528"
MODEL_LOCAL_PATH="/data/deepseek-ai/DeepSeek-R1-0528"
TENSOR_PARALLEL=8
KV_CACHE_DTYPE="fp8"
TEMPERATURE=0

LOG_FILE="/tmp/gpu_load_test_$(hostname)_$(date +%Y%m%d_%H%M%S).log"

echo "========================================="
echo "GPU Load Test - DeepSeek-R1"
echo "========================================="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Log: $LOG_FILE"
echo ""

# Check if model exists locally
echo "Checking for model..."
if [ -f "$MODEL_LOCAL_PATH/config.json" ]; then
    echo "Found model locally at: $MODEL_LOCAL_PATH"
    MODEL_PATH="$MODEL_LOCAL_PATH"
else
    echo "Model not found locally"
    echo "Will download from HuggingFace: $MODEL_NAME"
    MODEL_PATH="$MODEL_NAME"
fi

# Pre-flight GPU check
echo ""
echo "========================================="
echo "GPU Status"
echo "========================================="
if command -v rocm-smi &> /dev/null; then
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c 'GPU' || echo 'Unknown')
    echo "GPU Count: $GPU_COUNT"
    echo ""
    echo "Temperatures:"
    rocm-smi --showtemp 2>&1 | grep "Temperature (Sensor junction)" | head -8
else
    echo "rocm-smi not available"
fi
echo ""

# Remove existing container
docker rm -f atom_inference 2>/dev/null

# Run the test
echo "========================================="
echo "Starting Model Load Test..."
echo "========================================="
echo ""

docker run \
  --name atom_inference \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /data:/data \
  -e HF_HOME=/data/huggingface_cache \
  -e NCCL_DEBUG=WARN \
  -e RCCL_DEBUG=WARN \
  --shm-size=16G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  rocm/atom-dev:latest \
  bash -c '
    MODEL_RUNNER="/app/ATOM/atom/model_engine/model_runner.py"

    # Add timing instrumentation
    if ! grep -q "^import time$" "$MODEL_RUNNER"; then
        sed -i "1a import time" "$MODEL_RUNNER"
    fi

    # Instrument model loading with timing
    sed -i "/load_model(self.model, config.model, config.hf_config, config.load_dummy)/i\\
        load_start_time = time.time()\\
        logger.info(f\"[LOAD START] GPU {self.rank} | Time: {load_start_time:.6f}\")" \
        "$MODEL_RUNNER"

    sed -i "/load_model(self.model, config.model, config.hf_config, config.load_dummy)/a\\
        load_elapsed = time.time() - load_start_time\\
        logger.info(f\"[LOAD DONE] GPU {self.rank} | Duration: {load_elapsed:.2f}s | Time: {time.time():.6f}\")" \
        "$MODEL_RUNNER"

    # Run inference
    python3 -m atom.examples.simple_inference \
      --model "'"$MODEL_PATH"'" \
      --kv_cache_dtype "'"$KV_CACHE_DTYPE"'" \
      -tp "'"$TENSOR_PARALLEL"'" \
      --temperature "'"$TEMPERATURE"'"
  ' 2>&1 | tee "$LOG_FILE"

# Analyze results
echo ""
echo "========================================="
echo "DETAILED ANALYSIS RESULTS"
echo "========================================="
echo ""

# Check if test completed
LOAD_COUNT=$(grep -c "\[LOAD DONE\]" "$LOG_FILE" 2>/dev/null || echo 0)

if [ "$LOAD_COUNT" -eq 0 ]; then
    echo "FAIL: Test did not complete successfully"
    echo "   No GPU load completion markers found in log"
    echo ""
    echo "Possible issues:"
    echo "  - Model download failed"
    echo "  - Docker container crashed"
    echo "  - Out of memory"
    echo ""
    echo "Check log file: $LOG_FILE"
    exit 1
fi

if [ "$LOAD_COUNT" -lt 8 ]; then
    echo "WARNING: Only $LOAD_COUNT GPUs completed loading (expected 8)"
    echo ""
fi

# Extract and analyze timing data
echo "PER-GPU LOAD TIME ANALYSIS"
echo "========================================="
echo ""

# Get all load times with GPU IDs
LOAD_DATA=$(grep "\[LOAD DONE\]" "$LOG_FILE" | \
  sed 's/.*\[atom\] //' | \
  grep -oP 'GPU \K\d+.*Duration: [0-9.]+s' | \
  sort -t' ' -k1 -n)

# Find min and max for comparison
MIN_TIME=$(echo "$LOAD_DATA" | awk '{print $3}' | sed 's/s$//' | sort -n | head -1)
MAX_TIME=$(echo "$LOAD_DATA" | awk '{print $3}' | sed 's/s$//' | sort -n | tail -1)

echo "GPU | Load Time | Delta from Fastest | Status"
echo "----|-----------|-------------------|------------------"

while IFS= read -r line; do
    GPU=$(echo "$line" | awk '{print $1}')
    TIME=$(echo "$line" | awk '{print $3}' | sed 's/s$//')
    DELTA=$(awk "BEGIN {printf \"%.2f\", $TIME - $MIN_TIME}")
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($TIME - $MIN_TIME) / $MIN_TIME * 100}")

    # Determine status
    if (( $(echo "$DELTA < 1" | bc -l) )); then
        STATUS="Excellent"
        MARKER=""
    elif (( $(echo "$DELTA < 5" | bc -l) )); then
        STATUS="Good"
        MARKER=""
    elif (( $(echo "$DELTA < 10" | bc -l) )); then
        STATUS="Moderate"
        MARKER="  <--"
    else
        STATUS="SLOW"
        MARKER="  <-- PROBLEM"
    fi

    printf " %1s  | %8.2fs | +%6.2fs (%5.1f%%) | %-20s%s\n" "$GPU" "$TIME" "$DELTA" "$PERCENT" "$STATUS" "$MARKER"
done <<< "$LOAD_DATA"

echo ""

# Calculate comprehensive statistics
echo "STATISTICAL SUMMARY"
echo "========================================="
echo ""

STATS=$(grep "\[LOAD DONE\]" "$LOG_FILE" | grep -oP 'Duration: \K[0-9.]+' | awk '{
    times[NR] = $1;
    sum += $1;
    if(NR==1) { min=max=$1 }
    if($1 < min) { min=$1 }
    if($1 > max) { max=$1 }
}
END {
    if(NR > 0) {
        avg = sum / NR;
        delta = max - min;
        variance = (delta / avg) * 100;

        # Calculate median
        n = asort(times, sorted);
        if(n % 2) {
            median = sorted[(n+1)/2];
        } else {
            median = (sorted[n/2] + sorted[n/2+1]) / 2;
        }

        # Calculate standard deviation
        sum_sq_diff = 0;
        for(i=1; i<=NR; i++) {
            diff = times[i] - avg;
            sum_sq_diff += diff * diff;
        }
        stddev = sqrt(sum_sq_diff / NR);

        printf "GPUs Tested:     %d\n", NR;
        printf "Min Load Time:   %.2fs\n", min;
        printf "Max Load Time:   %.2fs\n", max;
        printf "Average:         %.2fs\n", avg;
        printf "Median:          %.2fs\n", median;
        printf "Std Deviation:   %.2fs\n", stddev;
        printf "Delta (Max-Min): %.2fs\n", delta;
        printf "Variance:        %.2f%%\n", variance;
        printf "DELTA_VALUE:%.2f\n", delta;
        printf "VARIANCE_VALUE:%.2f\n", variance;
        printf "MIN_VALUE:%.2f\n", min;
        printf "MAX_VALUE:%.2f\n", max;
    }
}')

echo "$STATS" | grep -v "_VALUE:" | while IFS=: read -r key value; do
    if [ -n "$key" ]; then
        printf "  %-18s %s\n" "$key:" "$value"
    fi
done

echo ""

# Extract key values for decision making
DELTA=$(echo "$STATS" | grep "^DELTA_VALUE:" | cut -d: -f2)
VARIANCE=$(echo "$STATS" | grep "^VARIANCE_VALUE:" | cut -d: -f2)

# Determine overall status
echo "DIAGNOSTIC ASSESSMENT"
echo "========================================="
echo ""

if (( $(echo "$DELTA < 1" | bc -l) )); then
    echo "Status: EXCELLENT"
    echo ""
    echo "All GPUs loaded within ${DELTA}s of each other."
    echo "All GPUs are loading the model simultaneously."
    echo "No GPU hardware issues detected."

elif (( $(echo "$DELTA < 5" | bc -l) )); then
    echo "Status: GOOD"
    echo ""
    echo "All GPUs loaded within ${DELTA}s variance."
    echo "Minor variance detected but within acceptable range."

elif (( $(echo "$DELTA < 10" | bc -l) )); then
    echo "Status: MODERATE VARIANCE"
    echo ""
    echo "Delta is ${DELTA}s - some imbalance detected."
    echo "Recommended: run test again to check if same GPUs are slow."

else
    echo "Status: HIGH VARIANCE - ISSUE DETECTED"
    echo ""
    echo "Delta is ${DELTA}s - significant GPU load imbalance!"
    echo ""

    # Identify slow GPUs
    SLOW_GPUS=$(echo "$LOAD_DATA" | awk -v min="$MIN_TIME" '{
        time = $3;
        gsub(/s$/, "", time);
        if(time > min + 10) {
            print $1;
        }
    }' | tr '\n' ',' | sed 's/,$//')

    if [ -n "$SLOW_GPUS" ]; then
        echo "Slow GPU(s) identified: $SLOW_GPUS"
        echo ""
    fi

    echo "Most likely causes:"
    echo "  1. Storage I/O bottleneck"
    echo "  2. Model shard distribution"
    echo "  3. NUMA/memory issues"
    echo "  4. PCIe link degradation"
fi

echo ""
echo "========================================="
echo "TEST SUMMARY"
echo "========================================="
echo "Hostname:     $(hostname)"
echo "Test Date:    $(date)"
echo "Model:        $MODEL_PATH"
echo "GPUs Tested:  $LOAD_COUNT"
echo "Full Log:     $LOG_FILE"
echo "========================================="
