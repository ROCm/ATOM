#!/bin/bash
# mkpoint_yaml.sh <name> <concurrency> <trajectories> <issue cap, e.g. 30m, or -> <seeds: submission|warmup>
# Writes one benchmark YAML to stdout. stop_issuing_on_first_user_complete is on, which skips the tail of the pass:
# these runs compare configurations and cannot be submitted. "submission" binds the run to the Endpoints seed set
# (cohort 2026-10-C1, set A) the Kimi-K3 runs used; the ruleset pins seeds only, it has no per-model rules.
# "warmup" uses other seeds, so its trajectory order and salts differ from the measured run's, and it leaves no
# prefix-cache entries that run could hit.
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
NAME=$1 C=$2 N=$3 CAP=$4 SEEDS=$5
cat << YAML
name: "$NAME"
version: "1.0"
type: "online"
YAML
if [ "$SEEDS" = submission ]; then
  cat << YAML
submission_ref:
  model: MiniMaxAI/MiniMax-M3
  ruleset: mlperf-endpoints-v1.0-2026-10-C1-A
YAML
fi
cat << YAML
model_params:
  name: "/models/$MODEL"
  tokenizer_name: "$MODEL_PATH"
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
      num_trajectories_to_issue: $N
      stop_issuing_on_first_user_complete: true
settings:
YAML
if [ "$CAP" != - ] || [ "$SEEDS" = warmup ]; then
  echo "  runtime:"
  if [ "$CAP" != - ]; then echo "    max_issue_duration_ms: $CAP"; fi
  if [ "$SEEDS" = warmup ]; then printf '    scheduler_random_seed: 1\n    dataloader_random_seed: 2\n'; fi
fi
cat << YAML
  load_pattern:
    type: agentic_inference
    target_concurrency: $C
  client:
    warmup_connections: 0
    max_idle_time: 0.5
endpoint_config:
  endpoints: ["http://127.0.0.1:$API_PORT"]
  api_type: openai
report_dir: $CLIENT/results/$NAME
YAML
