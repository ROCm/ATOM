ACC_DIR=/results/minimax-m3/gsm8k
HF_HUB_OFFLINE=0 lm_eval --model local-completions   --model_args "model=/models/MiniMax-M3-MXFP4,base_url=http://127.0.0.1:8890/v1/completions,num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True"   --tasks gsm8k --num_fewshot 3 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
