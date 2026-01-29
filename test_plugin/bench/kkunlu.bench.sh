#/bin/bash

#下載kunlun-benchmark
# wget https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/ai_perf/pack/kunlun_benchmark/main/ubuntu22.04/py3.12.8/kunlun-benchmark.tar.gz

#解壓就好了, 安裝幾個缺的library
# pip install jsonlines prettytable oss2

MODEL=/data/pretrained-models/Qwen3-235B-A22B-Instruct-2507-FP8

# 3-3.6K, 0.3-0.5K
MIN_INPUT_LEN=3000
MAX_INPUT_LEN=3600
MIN_OUTPUT_LEN=300
MAX_OUTPUT_LEN=500
CON=220
NUM=880

# 0.8-1K, 1.6-2K
# MIN_INPUT_LEN=800
# MAX_INPUT_LEN=1000
# MIN_OUTPUT_LEN=1600
# MAX_OUTPUT_LEN=2000
# CON=512
# NUM=2048

# 3.6-4.4K, 1.8-2.2K
#MIN_INPUT_LEN=3600
#MAX_INPUT_LEN=4400
#MIN_OUTPUT_LEN=1800
#MAX_OUTPUT_LEN=2200
#CON=260
#NUM=1040

echo "ATOM Model=${MODEL}"
echo "ATOM MIN_INPUT_LEN=${MIN_INPUT_LEN}, MAX_INPUT_LEN=${MAX_INPUT_LEN}, MIN_OUTPUT_LEN=${MIN_OUTPUT_LEN}, MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN}, CON=${CON}, NUM=${NUM}"

sleep 2

/home/zejchen/plugin/ATOM/test_plugin/bench/kunlun-benchmark/kunlun-benchmark \
  vllm \
  server \
  --port 8000 \
  --work_mode manual \
  --max_input_len ${MAX_INPUT_LEN} \
  --min_input_len ${MIN_INPUT_LEN} \
  --max_output_len ${MAX_OUTPUT_LEN} \
  --min_output_len ${MIN_OUTPUT_LEN} \
  --concurrency ${CON} \
  --query_num ${NUM} \
  --result_dir ./qwen235b_tp8_test \
  --model_path ${MODEL} \
  --is_sla False \
  --sla_decode 50 \
  --sla_prefill 3000 \
  --tp 8 2>&1 | tee log.bench.log

echo "ATOM Model=${MODEL}"
echo "ATOM MIN_INPUT_LEN=${MIN_INPUT_LEN}, MAX_INPUT_LEN=${MAX_INPUT_LEN}, MIN_OUTPUT_LEN=${MIN_OUTPUT_LEN}, MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN}, CON=${CON}, NUM=${NUM}"
