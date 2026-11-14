SERVED_NAME='/models/Deepseek-S3_sq_a05_v2_mlperf6_1/'; TOKENIZER='/data/Deepseek-S3_sq_a05_v2_mlperf6_1'; SERVER='127.0.0.1:8000'; CLIENT='/mnt/data/zejchen/ci455/client'
# point_yaml <name> <并发> <轨迹数> <warmup|run> <report_dir>
point_yaml() {
  cat << YAML
name: "$1"
version: "1.0"
type: "online"
model_params:
  name: "$SERVED_NAME"
  tokenizer_name: "$TOKENIZER"
  temperature: 1.0
  top_p: 1.0
  max_new_tokens: 8192
  streaming: "on"
datasets:
  - name: agentic_combined
    type: performance
    path: $CLIENT/data/mlperf_agentic_inference_dataset.jsonl
    accuracy_config:
      eval_method: agentic_inference_inline
      num_repeats: 1
    agentic_inference:
      turn_timeout_s: 86400.0
      enable_salt: true
      inject_tool_delay: true
      num_trajectories_to_issue: $3
      stop_issuing_on_first_user_complete: true
settings:
  runtime:
YAML
  if [ "$4" = warmup ]; then
    printf '    scheduler_random_seed: 1\n    dataloader_random_seed: 2\n'
  else
    printf '    max_issue_duration_ms: 600000\n'
  fi
  cat << YAML
  timeouts:
    run_timeout_s: 14400
  load_pattern:
    type: agentic_inference
    target_concurrency: $2
  client:
    warmup_connections: 0
    max_idle_time: 0.5
endpoint_config:
  endpoints: ["http://$SERVER"]
  api_type: openai
report_dir: $5
YAML
}
run_point() { (ulimit -n 524288 && cd $CLIENT/endpoints && UV_FROZEN=1 uv run inference-endpoint benchmark from-config --config "$1" --mode perf); }

NW=$(( CONC < 16 ? 2 * CONC : CONC ))
point_yaml warmup-c$CONC $CONC $NW warmup $RESULT_DIR/warmup > $RESULT_DIR/warmup.yaml
point_yaml run-c$CONC $CONC 613 run $RESULT_DIR/report > $RESULT_DIR/run.yaml
run_point $RESULT_DIR/warmup.yaml > $RESULT_DIR/warmup.log 2>&1
run_point $RESULT_DIR/run.yaml > $RESULT_DIR/bench.log 2>&1
