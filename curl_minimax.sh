curl -X POST "http://localhost:8014/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "The capital of China is", "temperature": 0, "top_p": 1, "top_k": 1, "repetition_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "stream": false, "ignore_eos": false, "n": 1, "seed": 123, "max_tokens": 20}'