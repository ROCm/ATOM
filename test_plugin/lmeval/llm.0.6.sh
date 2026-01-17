addr=localhost
port=9090
url=http://${addr}:${port}/v1/completions
model=/raid/data/model/Qwen3-0.6B
bs=1
task=gsm8k

echo "url=${url}"
echo "model=${model}"
echo "task=${task}"

lm_eval \
    --model local-completions \
    --tasks ${task} \
    --model_args model=${model},base_url=${url} \
    --batch_size ${bs} \
    --seed 123 \
    --limit 10 \
    2>&1 | tee log.lmeval.log