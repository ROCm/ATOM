ACC_DIR=/results/glm-5.2/gsm8k
HF_HUB_OFFLINE=0 lm_eval --model local-completions \
  --model_args "model=/models/GLM-5.2-MXFP4,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 3 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
