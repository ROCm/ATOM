curl -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/shared/data/amd_int/models/deepseek-ai/DeepSeek-V4-Flash",
        "prompt": "The capital of China",
        "max_tokens": 32,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1.0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": false,
        "ignore_eos": false,
        "n": 1,
        "seed": 123
}'