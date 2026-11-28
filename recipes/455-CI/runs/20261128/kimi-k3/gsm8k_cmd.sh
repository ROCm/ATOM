ACC_DIR=/results/kimi-k3/gsm8k
HF_HUB_OFFLINE=0 lm_eval --model local-chat-completions --apply_chat_template --tasks gsm8k --num_fewshot 5 \
  --model_args "model=moonshotai/Kimi-K3,tokenizer=/models/Kimi-K3,base_url=http://127.0.0.1:8000/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=32,timeout=1800,tokenized_requests=False,max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
