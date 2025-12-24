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
MODEL_PATH="/mnt/raid0/models/deepseek-ai/DeepSeek-V3.2-Exp/"
ISL_LOOP=(1023 8191)
CONC_LOOP=(1 8 64)
OSL=1024
ISL_OSL_CONC=(65535:1024:1 65535:1024:8 1023:1024:128 1023:8192:64)
PORT=8888
LOG_FILE="deepseek32"
########################## CONFIGS ##########################

source $BENCHMARK_SCRIPT
python $SAVE_CSV_FILE $LOG_FILE
)