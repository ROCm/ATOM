MODEL=/data/pretrained-models/Qwen3-235B-A22B-Thinking-2507_moe_w_mxfp4_a_mxfp4_kv_fp8
ISL=$1
OSL=$2
CONC=$3
PORT=8000
RESULT_FILENAME="qwen3_235b_a22b_instrct_2507_FP8_isl${ISL}_osl${OSL}_conc${CONC}_infrrate"
log_file="log_i${input_len}_o${output_len}_c${concurrency}.log"
# Remember to use scripts in this repo!
git clone https://github.com/kimbochen/bench_serving.git
python bench_serving/benchmark_serving.py \
--model=$MODEL --backend=vllm --base-url=http://localhost:$PORT \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL \
--random-range-ratio 1 \
--num-prompts=$(( $CONC * 4)) \
--max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics="ttft,tpot,itl,e2el" \
--result-dir=./ --result-filename=$RESULT_FILENAME.json 2>&1 | tee $log_file
