addr=localhost
port=8000

url=http://${addr}:${port}/v1/completions
model=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8

task=gsm8k

echo "addr=${addr}"
echo "port=${port}"
echo "url=${url}"
echo "model=${model}"
echo "task=${task}"

sleep 1

lm_eval --model local-completions \
        --model_args model=${model},base_url=${url},num_concurrent=65,max_retries=1,tokenized_requests=False \
        --tasks ${task} \
        --num_fewshot 3 \
        2>&1 | tee log.lmeval.log

# --output_path: Path to save evaluation results
# --log_samples: Save per-document outputs and inputs
