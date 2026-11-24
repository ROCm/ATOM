ACC_DIR=/results/qwen-3.8-flash/gsm8k_acc
cat > "$ACC_DIR/qwen38_gsm8k.yaml" <<'EOF'
tag:
  - math_word_problems
task: gsm8k
dataset_path: openai/gsm8k
dataset_name: main
output_type: generate_until
training_split: train
fewshot_split: train
test_split: test
doc_to_text: "Question: {{question}}\nEnd your response with the answer on the last line, formatted as: #### [number]\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
      - "\\.$"
generation_kwargs:
  until:
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: 5
filter_list:
  - name: strict-match
    filter:
      - function: regex
        group_select: -1
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
      - function: take_first
  - name: flexible-extract
    filter:
      - function: regex
        group_select: -1
        regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
      - function: take_first
metadata:
  version: 3.0
EOF
HF_HUB_OFFLINE=0 lm_eval --model local-chat-completions --apply_chat_template --tasks "$ACC_DIR/qwen38_gsm8k.yaml" \
  --model_args "model=/models/Qwen3.8-Flash-Next-Quark-MXFP4,base_url=http://127.0.0.1:8000/v1/chat/completions,api_key=EMPTY,eos_string=,max_retries=5,num_concurrent=65,timeout=1800,tokenized_requests=False,max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
