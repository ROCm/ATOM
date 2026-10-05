#!/usr/bin/env bash
PROMPT=$(python -c 'print("You are a helpful assistant. " + "The quick brown fox jumps over the lazy dog. " * 120)')

for i in 1 2; do
  curl -s localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d "$(python - "$PROMPT" "$i" <<'EOF'
import json, sys
print(json.dumps({
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "prompt": sys.argv[1] + " Question " + sys.argv[2],
    "max_tokens": 8,
    "data_parallel_rank": 3,
}))
EOF
)" > /dev/null
  sleep 5
done