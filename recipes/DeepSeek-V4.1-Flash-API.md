# DeepSeek-V4.1-Flash text API recipe

This recipe covers ordinary APIs and beta text APIs merged into ROCm/ATOM main. Use commit `10b99a085ba46e5b3a125f0114cec7c71e401552` or a later release containing it. The examples target the ATOM Python API server directly. Image serving is a separate deployment recipe.

## 1. Install and start the basic service

Run inside an ATOM development container with ROCm, PyTorch and the matching AITER dependencies already installed. Use the checkpoint directory containing the model weights, tokenizer and model-specific encoder. Follow the [model recipe](https://github.com/ROCm/ATOM/blob/main/recipes/DeepSeek-V4.1-Flash.md) for the hardware/dependency build; this recipe does not replace that setup.

```bash
cd /path/to/ATOM
git fetch origin lirzhang/v41-api-recipe-20261109
git checkout -b v41-api-recipe origin/lirzhang/v41-api-recipe-20261109
python3 -m pip install -e '.[structured-output]'

export MODEL_PATH=/models/deepseek-ai/DeepSeek-V4.1-Flash
export HIP_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export USE_ATOMESH_ENTRYPOINTS=0
export ATOM_MONO_ENABLE=0
export ATOM_DISABLE_MMAP=true

python3 -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --trust-remote-code \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --server-port 8000 --port 10000 \
  -tp 4 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 --max-num-seqs 16 \
  --max-num-batched-tokens 4096 --attn-prefill-chunk-size 4096 \
  --enable_prefix_caching --block-size 16 \
  --state-checkpoint-interval-tokens 8192 \
  --level 0 --enforce-eager \
  --tool-call-parser dsml_v41
```

This is a conservative TP4, text-only starting point, with no speculative decoding. It can exercise Chat logprobs and Schema together. The 32K limit is the configured prompt-plus-output budget, not a claim about the checkpoint's maximum supported context. Choose GPUs and ports that are free. TP2 is also supported on suitable MI350-class hardware; use two visible GPUs and `-tp 2` after checking host/GPU memory capacity.

Do not enable synthetic acceptance when checking output correctness: omit `--spec-decode-acceptance-length`, `--spec-decode-acceptance-rate` and synthetic benchmark environment settings. Start from a clean shell rather than inheriting a benchmark configuration.

The structured-output extra installs llguidance, jsonschema and google-re2. Ordinary requests do not require a schema in the launch command: schema/strict tool definitions are supplied per request.

Wait until startup finishes, then inspect model discovery:

```bash
export BASE_URL=http://127.0.0.1:8000
curl -sS "$BASE_URL/v1/models"
```

From another host, replace `127.0.0.1` with the reachable server address. The launch above does not configure API authentication; use your deployment's access control. Request `model` must match `DeepSeek-V4.1-Flash`, the served alias configured above.

## 2. Request examples

The standalone request files are in [examples/deepseek-v41-api](examples/deepseek-v41-api). After starting the server, open a second shell and run:

```bash
cd /path/to/ATOM/recipes/examples/deepseek-v41-api
export BASE_URL=http://127.0.0.1:8000
```

Then run the curl command beside each example. Each JSON file can also be copied and edited independently.

### Plain Chat

`chat.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "What is 2 + 2? Answer briefly."
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "thinking": {
    "type": "disabled"
  }
}
```

```bash
curl -sS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @chat.json
```

Read the answer from `choices[0].message.content`.

### Streaming Chat

`chat-stream.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "What is 2 + 2? Answer briefly."
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "thinking": {
    "type": "disabled"
  },
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

```bash
curl -N -sS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @chat-stream.json
```

SSE chunks carry `choices[].delta`. V4.1 usage is on the finish chunk; earlier chunks have null usage when requested. The stream ends with `[DONE]`.

### Thinking / reasoning effort

`chat-thinking.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "Explain why the sum of two odd integers is even."
    }
  ],
  "thinking": {
    "type": "enabled"
  },
  "reasoning_effort": "high",
  "max_tokens": 2048
}
```

```bash
curl -sS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @chat-thinking.json
```

Read reasoning from `message.reasoning_content` and the visible answer from `message.content`. Use a larger token budget for thinking. V4.1 effort values are `none`, `low`, `high`, `max`; aliases `minimal`→`low`, `medium`/`xhigh`→`high`. An explicit disabled thinking switch takes precedence. Avoid conflicting controls.

### Responses

`responses.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "input": "What is 2 + 2? Answer briefly.",
  "reasoning": {
    "effort": "none"
  },
  "max_output_tokens": 256
}
```

```bash
curl -sS "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  --data-binary @responses.json
```

Read assistant text from the output message content blocks. Supply full input/history on each request; this deployment does not persist Responses conversations. The explicitly documented unsupported stateful fields are ignored for V4.1.

### Streaming Responses

`responses-stream.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "input": "What is 2 + 2? Answer briefly.",
  "reasoning": {
    "effort": "none"
  },
  "max_output_tokens": 256,
  "stream": true
}
```

```bash
curl -N -sS "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  --data-binary @responses-stream.json
```

Consume typed SSE events, including reasoning/text deltas and a terminal `response.completed`, `response.incomplete` or `response.failed`. Responses does not end with `[DONE]`.

### Anthropic Messages

`messages.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "What is 2 + 2? Answer briefly."
    }
  ],
  "thinking": {
    "type": "disabled"
  },
  "max_tokens": 256
}
```

```bash
curl -sS "$BASE_URL/v1/messages" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  --data-binary @messages.json
```

Read the visible answer from text blocks in `content`. For streaming add `"stream": true` and consume Anthropic events. The endpoint is `/v1/messages`, not `/v1/chat/completions`.

### Anthropic thinking

`messages-thinking.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "What is 2 + 2? Answer briefly."
    }
  ],
  "thinking": {
    "type": "enabled",
    "budget_tokens": 1024
  },
  "max_tokens": 2048,
  "output_config": {
    "effort": "high"
  }
}
```

```bash
curl -sS "$BASE_URL/v1/messages" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  --data-binary @messages-thinking.json
```

`thinking.type` controls activation independently of `output_config.effort`. The DeepSeek-compatible endpoint ignores `budget_tokens`; use `output_config.effort` to select effort.

### Chat token logprobs

`chat-logprobs.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "What is 2 + 2? Answer briefly."
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "thinking": {
    "type": "disabled"
  },
  "logprobs": true,
  "top_logprobs": 5
}
```

```bash
curl -sS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @chat-logprobs.json
```

Read `choices[0].logprobs.content`; reasoning probabilities use the corresponding reasoning list when thinking is enabled. Entries include token, bytes, logprob and top_logprobs. `top_logprobs` accepts 0–20 and requires `logprobs: true`. These are raw target-model probabilities before sampling filters/masks. A selected token below the top-20 rank threshold uses the wire sentinel `-9999.0`. Probability fields apply to Chat, not Responses, Messages or FIM.

### JSON object

`chat-json.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "Return a JSON object with city Paris and temperature_c 20."
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "thinking": {
    "type": "disabled"
  },
  "response_format": {
    "type": "json_object"
  }
}
```

```bash
curl -sS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @chat-json.json
```

Ask for JSON in the prompt and pass `response_format`. Parse `message.content` as JSON. This constrains JSON syntax, not arbitrary business semantics.

### Chat JSON Schema

`chat-schema.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "Return city Paris and temperature_c 20."
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "thinking": {
    "type": "disabled"
  },
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string"
          },
          "temperature_c": {
            "type": "number"
          }
        },
        "required": [
          "city",
          "temperature_c"
        ],
        "additionalProperties": false
      }
    }
  }
}
```

```bash
curl -sS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @chat-schema.json
```

The schema belongs in the request, not the startup command. Parse `message.content` and validate it against your schema. `finish_reason: length` means the answer was truncated and must not be treated as complete JSON.

### Responses JSON Schema

`responses-schema.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "input": "Return city Paris and temperature_c 20.",
  "reasoning": {
    "effort": "none"
  },
  "max_output_tokens": 256,
  "text": {
    "format": {
      "type": "json_schema",
      "name": "weather",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string"
          },
          "temperature_c": {
            "type": "number"
          }
        },
        "required": [
          "city",
          "temperature_c"
        ],
        "additionalProperties": false
      }
    }
  }
}
```

```bash
curl -sS "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  --data-binary @responses-schema.json
```

Responses uses `text.format`, rather than Chat’s `response_format`. Read the visible output text; `response.incomplete` signals a token-budget truncation.

### Ordinary function tools

`chat-tools.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in Paris? Use get_weather."
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "thinking": {
    "type": "disabled"
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": [
            "city"
          ],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": {
    "type": "function",
    "function": {
      "name": "get_weather"
    }
  }
}
```

```bash
curl -sS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @chat-tools.json
```

The reply contains `message.tool_calls`. The client executes the function, then sends a new Chat request containing the original user message, the returned assistant message and `{"role":"tool","tool_call_id":"<returned id>","content":"{\"city\":\"Paris\",\"temperature_c\":20}"}`. Keep the tool definition in the follow-up. Tools are not executed by ATOM. Required/named tool choice needs thinking disabled on the V4.1 Chat contract.

### Beta fill-in-the-middle

`fim.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "prompt": "def add(a, b):\n",
  "suffix": "\n\nprint(add(2, 3))",
  "temperature": 0,
  "max_tokens": 128
}
```

```bash
curl -sS "$BASE_URL/beta/completions" \
  -H "Content-Type: application/json" \
  --data-binary @fim.json
```

Return `choices[0].text` is the generated middle: compose `prompt + generated text + suffix`. FIM does not guarantee the completed code is correct; compile/test it. `echo` cannot be combined with `suffix`; `logprobs` is rejected on this endpoint.

### Beta assistant-prefix continuation

`prefix.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "List three primary colors."
    },
    {
      "role": "assistant",
      "content": "The three primary colors are",
      "prefix": true
    }
  ],
  "thinking": {
    "type": "disabled"
  },
  "temperature": 0,
  "max_tokens": 128
}
```

```bash
curl -sS "$BASE_URL/beta/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @prefix.json
```

Only the final assistant message may carry `prefix: true`. The returned content is the continuation; prepend your original prefix to display the complete answer. Prefix cannot be combined with structured output, strict tools, images or token-ID prompts.

### Beta strict function tools

`strict-tools.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in Paris? Use get_weather."
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "thinking": {
    "type": "disabled"
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": [
            "city"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ],
  "tool_choice": {
    "type": "function",
    "function": {
      "name": "get_weather"
    }
  }
}
```

```bash
curl -sS "$BASE_URL/beta/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @strict-tools.json
```

`strict: true` is enforced on `/beta/chat/completions`; `/v1/chat/completions` continues to ignore it. All object properties must be required and `additionalProperties` must be false. Execute and return results using the ordinary tool roundtrip above. Streamed strict calls are held until validated and arrive as completed calls rather than incremental argument fragments.

### Flash web_search declaration compatibility

`flash-search-declaration.json`:

```json
{
  "model": "DeepSeek-V4.1-Flash",
  "input": "What is 2 + 2? Answer briefly.",
  "reasoning": {
    "effort": "none"
  },
  "max_output_tokens": 256,
  "tools": [
    {
      "type": "web_search"
    }
  ]
}
```

```bash
curl -sS "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  --data-binary @flash-search-declaration.json
```

This declaration is accepted for Flash compatibility, but no search backend runs and no fresh web facts/citations are guaranteed. To implement real search, supply a normal client-executed function tool and return its results.

## 3. Optional DSpark and graphs

After the basic service works, stop that instance and restart with:

- DSpark: add `--method dspark --num-speculative-tokens 5`.
- Graph execution: remove `--enforce-eager` and use the model recipe's supported graph settings. Keep `--level 0` for this starting recipe.

Keep forced/synthetic acceptance disabled. Ordinary Chat logprobs work with DSpark. JSON Schema and beta strict tools work with DSpark separately from logprobs; **Schema/strict tools + speculative decoding + logprobs is rejected**. Use the basic non-speculative launch if that combination is needed.

Access `/beta/*` directly on this Python API server. The beta route implementation is not currently routed through Atomesh DP/P-D mesh; do not substitute a mesh frontend for this recipe.

## 4. Boundaries and errors

- JSON Schema is a supported subset, not every JSON Schema predicate. External references and unsupported predicates fail before enqueue. Prefer simple typed objects, required fields, enums and bounds; unsupported unbounded `uniqueItems` and complex negations are rejected.
- Schema and strict tools require the DSML tool parser and text serving; no P/D, pipeline parallelism, custom stop strings, ignored EOS, relaxed MTP or synthetic acceptance. Strict schemas additionally reject `minLength`, `maxLength`, `minItems`, `maxItems`, `patternProperties`; supported strict formats are email, hostname, ipv4, ipv6 and uuid. Local definitions/references must follow the documented strict rules.
- With thinking enabled, use `tool_choice: "auto"` for Chat tools; the model may choose to answer without calling a tool. Disable thinking for required/named selection.
- Use a list for ordinary `/v1` stop strings, e.g. `"stop": ["END"]`. Beta accepts a string as well. Do not combine custom stops with Schema/strict tools.
- An unknown served model returns HTTP 400. Invalid fields can produce HTTP 400/422; inspect the returned error body rather than only the HTTP code. Streaming errors/truncation must be handled as terminal failure/incomplete events.
- Enabled Schema/strict constraints and logprobs add work; performance depends on workload and topology. This recipe makes no zero-overhead claim. Start with those request options absent when measuring the ordinary path.
- Image input and vision, including its remaining qualification work, are excluded from this text recipe. Never assume successful text generation establishes image understanding.

## 5. Merge references

API compatibility: [#2844](https://github.com/ROCm/ATOM/pull/2844). Chat/DSpark probabilities: [#2856](https://github.com/ROCm/ATOM/pull/2856). JSON constraints and model metadata: [#2862](https://github.com/ROCm/ATOM/pull/2862). Beta FIM/prefix: [#3036](https://github.com/ROCm/ATOM/pull/3036). Beta strict tools: [#3090](https://github.com/ROCm/ATOM/pull/3090). Requirements tracker: [#2864](https://github.com/ROCm/ATOM/issues/2864).

Prepared against main `10b99a08` on 2026-11-09. Commands and payloads were checked against source; this documentation and example set was not newly GPU-tested as a bundle. Prior real-model validation is recorded in the linked PRs. The issue's historical draft/unmerged labels are stale.
