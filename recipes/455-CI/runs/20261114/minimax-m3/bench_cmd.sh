#!/bin/bash
# runs in the container; needs CONC and RESULT_DIR; recipe "8K/1K 的 bench（所有模型共用）"
MODEL_PATH='/models/MiniMax-M3-MXFP4'; PORT=8890
# Recipe "8K/1K 的 bench（所有模型共用）", verbatim. Caller sets MODEL_PATH PORT CONC RESULT_DIR; runs in the test container.
mkdir -p $RESULT_DIR
python3 -m atom.benchmarks.benchmark_serving \
  --model=$MODEL_PATH --backend=vllm --base-url=http://127.0.0.1:$PORT \
  --dataset-name=random --random-input-len=8192 --random-output-len=1024 --random-range-ratio=0.8 \
  --max-concurrency=$CONC --num-prompts=$((CONC * 10)) --num-warmups=$((CONC * 2)) \
  --request-rate=inf --ignore-eos --trust-remote-code \
  --save-result --result-dir=$RESULT_DIR --result-filename=raw.json \
  --percentile-metrics="ttft,tpot,itl,e2el"
