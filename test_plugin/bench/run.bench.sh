#!bin/bash

MODEL=/home/zejchen/model/Qwen3-235B-A22B-Instruct-2507-FP8

# 1K/1K
ISL=1000
OSL=1000
CON=256
#CON=128
NUM=$(( CON * 4 ))

# 4K/1K
#ISL=4000
#OSL=1000
#CON=128
#CON=64
#NUM=$(( CON * 4 ))

# 10K/1K
#ISL=10000
#OSL=1000
#CON=64
#CON=32
#NUM=$(( CON * 4 ))

echo "ATOM Model=${MODEL}"
echo "ATOM ISL=${ISL}, OSL=${OSL}, NUM=${NUM}, CON=${CON}"

sleep 2

git clone https://github.com/kimbochen/bench_serving.git
python bench_serving/benchmark_serving.py \
    --model=$MODEL \
    --backend=vllm \
    --base-url=http://localhost:8000 \
    --dataset-name=random \
    --random-input-len=$ISL \
    --random-output-len=$OSL \
    --random-range-ratio 1 \
    --num-prompts=${NUM} \
    --max-concurrency=${CON} \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=./ \
    2>&1 | tee log.bench.log

echo "ATOM Model=${MODEL}"
echo "ATOM ISL=${ISL}, OSL=${OSL}, NUM=${NUM}, CON=${CON}"
rm -rf ./*.json
