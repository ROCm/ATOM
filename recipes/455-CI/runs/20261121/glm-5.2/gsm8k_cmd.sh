ACC_DIR=/results/glm-5.2/gsm8k
HF_HUB_OFFLINE=0 lm_eval --model local-chat-completions --apply_chat_template \
  --model_args "model=/models/GLM-5.2-MXFP4,base_url=http://127.0.0.1:8000/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=64,timeout=1800,tokenized_requests=False,max_length=1048576" \
  --tasks gsm8k --gen_kwargs max_tokens=16384,temperature=0,top_p=1 --num_fewshot 5 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
