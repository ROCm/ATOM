ACC_DIR=/results/deepseek-v4.1-flash/gsm8k_tp4
HF_HUB_OFFLINE=0 lm_eval --model local-completions \
    --model_args "model=/models/DeepSeek-V4.1-Flash,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k --num_fewshot 3 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
