#!/bin/bash
set -euo pipefail

# step 1: please pip install openpyxl
# step 3: change the configs as below
# step 4: bash benchmark.sh

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_SCRIPT="$CUR_DIR/../benchmark.sh"
SAVE_CSV_FILE="$CUR_DIR/../save_csv.py"

(
######################### CONFIGS ###########################
MODEL_PATH="/data/pretrained-models/Qwen3-235B-A22B-Thinking-2507_moe_w_mxfp4_a_mxfp4_kv_fp8"
ISL_LOOP=(1024 8192)
CONC_LOOP=(4 8 16 32 64)
OSL=1024
ISL_OSL_CONC=(1024:8192:4 1024:8192:8 1024:8192:16 1024:8192:32 1024:8192:64)
PORT=8888
LOG_FILE="qwen_fp4"
########################## CONFIGS ##########################
#curl -s -S -X POST http://127.0.0.1:8888/start_profile
source $BENCHMARK_SCRIPT
#curl -s -S -X POST http://127.0.0.1:8888/stop_profile
python $SAVE_CSV_FILE $LOG_FILE
)