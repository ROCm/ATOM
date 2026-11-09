# DeepSeek-V4.1-Flash API recipe

This recipe covers ordinary APIs, beta text APIs and image inputs merged into ROCm/ATOM main. Use latest `main`, including #3111 and #3115 (merged by `0a99b2b9`); the runtime source is not the documentation branch. The examples target the ATOM Python API server directly. The basic non-DSpark launch below also serves images when the complete multimodal checkpoint is installed.

## 1. Install and start the basic service

Run inside an ATOM development container with ROCm, PyTorch and the matching AITER dependencies already installed. Use the checkpoint directory containing the model weights, tokenizer and model-specific encoder. For images, include the vision tower, aligner and image processor files from the full checkpoint. Follow the [model recipe](https://github.com/ROCm/ATOM/blob/main/recipes/DeepSeek-V4.1-Flash.md) for the hardware/dependency build; this recipe does not replace that setup.

```bash
cd /path/to/ATOM
git switch main
git pull --ff-only origin main
python3 -m pip install -e '.[structured-output]' pillow

unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
unset ATOM_DSV41_BENCHMARK_SYNTHETIC ATOM_ENABLE_RELAXED_MTP

export MODEL_PATH=/models/deepseek-ai/DeepSeek-V4.1-Flash
export HIP_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export USE_ATOMESH_ENTRYPOINTS=0
export ATOM_MONO_ENABLE=0
export ATOM_DISABLE_MMAP=true

python3 -c "import torch, aiter, llguidance, jsonschema, re2; from PIL import Image; assert torch.version.hip, 'ROCm PyTorch required'; assert torch.cuda.device_count() == 4, 'Expected four visible GPUs'; print(torch.version.hip, torch.cuda.device_count())"

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
  --level 0 --cudagraph-mode FULL \
  --cudagraph-capture-sizes '[1,2,4,8,16]' \
  --timeout-keep-alive 900 \
  --tool-call-parser dsml_v41
```

This is a conservative native TP4 starting point, with no speculative decoding or DP attention. It can exercise Chat logprobs and Schema together. The 32K limit is the configured prompt-plus-output budget, not a claim about the checkpoint's maximum supported context. Choose GPUs and ports that are free. TP2 is also supported on suitable MI350-class hardware; use two visible GPUs and `-tp 2` after checking host/GPU memory capacity.

Do not enable synthetic acceptance when checking output correctness: omit `--spec-decode-acceptance-length`, `--spec-decode-acceptance-rate` and synthetic benchmark environment settings. Start from a clean shell rather than inheriting a benchmark configuration.

The structured-output extra installs llguidance, jsonschema and google-re2; Pillow is needed to decode API images. The editable install does not install a ROCm PyTorch build, AITER or the complete kernel build toolchain: use the supported ATOM development environment for your GPU architecture. Do not reuse an older image's AITER blindly after updating main; follow the current model/development setup instructions. Ordinary requests do not require a schema in the launch command: schema/strict tool definitions are supplied per request.

The import/device check in the startup block runs before weight loading and checks the same Python environment. `--server-port 8000` is the HTTP port clients use; `--port 10000` is internal engine communication. Confirm both ports and the GPUs are available. The graph buckets are request counts up to this recipe's `--max-num-seqs 16`; adjust both together for a larger deployment. Level 0 disables torch compilation optimizations, while the explicit FULL mode still enables CUDA graphs. The first launch may take time to load weights and compile kernels/capture graphs.

Wait until startup finishes, then inspect model discovery:

```bash
export BASE_URL=http://127.0.0.1:8000
curl --fail-with-body -sS "$BASE_URL/v1/models"
```

From another host, replace `127.0.0.1` with the reachable server address. The launch above does not configure API authentication; use your deployment's access control. Request `model` must match `DeepSeek-V4.1-Flash`, the served alias configured above.

## 2. Request examples

The standalone request files are in [examples/deepseek-v41-api](examples/deepseek-v41-api) on the documentation branch. Run the server from latest `main`; extract the examples into a separate client directory without switching the serving checkout:

```bash
cd /path/to/ATOM
git fetch origin refs/heads/lirzhang/v41-api-recipe-20261109:refs/remotes/origin/lirzhang/v41-api-recipe-20261109
mkdir -p /tmp/v41-api-examples
git archive origin/lirzhang/v41-api-recipe-20261109 recipes/examples/deepseek-v41-api | tar -x -C /tmp/v41-api-examples
cd /tmp/v41-api-examples/recipes/examples/deepseek-v41-api
export BASE_URL=http://127.0.0.1:8000
```

Use curl 7.76 or later for `--fail-with-body`: it preserves the error body and returns a nonzero exit status on HTTP 4xx/5xx. Then run the curl command beside each example. Each JSON file can also be copied and edited independently.

### Run a local smoke check

The included checker uses Python's standard library to submit the examples and retain each JSON/SSE response. Run it after `/v1/models` is reachable:

```bash
python3 check-api.py --base-url "$BASE_URL"
# Include the supplied base64 image examples on the native, non-DSpark service:
python3 check-api.py --base-url "$BASE_URL" --images --output-dir api-check-with-images
```

The checker fails on HTTP errors, missing visible answers/probabilities, malformed example JSON output, missing tool calls, or incomplete/failed stream termination. It writes `summary.json` and raw responses under the output directory. Inspect image answers and generated FIM code as well: these checks are a smoke test, not broad model-quality or performance qualification. `--urls` additionally enables the public-URL image examples and requires server-side internet access.

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
curl --fail-with-body -sS "$BASE_URL/v1/chat/completions" \
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
curl --fail-with-body -N -sS "$BASE_URL/v1/chat/completions" \
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
curl --fail-with-body -sS "$BASE_URL/v1/chat/completions" \
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
curl --fail-with-body -sS "$BASE_URL/v1/responses" \
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
curl --fail-with-body -N -sS "$BASE_URL/v1/responses" \
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
curl --fail-with-body -sS "$BASE_URL/v1/messages" \
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
curl --fail-with-body -sS "$BASE_URL/v1/messages" \
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
curl --fail-with-body -sS "$BASE_URL/v1/chat/completions" \
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
curl --fail-with-body -sS "$BASE_URL/v1/chat/completions" \
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
curl --fail-with-body -sS "$BASE_URL/v1/chat/completions" \
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
curl --fail-with-body -sS "$BASE_URL/v1/responses" \
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
curl --fail-with-body -sS "$BASE_URL/v1/chat/completions" \
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
curl --fail-with-body -sS "$BASE_URL/beta/completions" \
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
curl --fail-with-body -sS "$BASE_URL/beta/chat/completions" \
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
curl --fail-with-body -sS "$BASE_URL/beta/chat/completions" \
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
curl --fail-with-body -sS "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  --data-binary @flash-search-declaration.json
```

This declaration is accepted for Flash compatibility, but no search backend runs and no fresh web facts/citations are guaranteed. To implement real search, supply a normal client-executed function tool and return its results.

## 3. Optional DSpark

After the basic service works, stop that instance and restart with:

Add `--method dspark --num-speculative-tokens 5` to the same startup command.

The basic launch explicitly selects FULL CUDA graphs at `--level 0`. `--enforce-eager` is only an optional debugging flag to disable graphs when isolating a problem; it is not part of the normal launch.

Keep forced/synthetic acceptance disabled. Ordinary Chat logprobs work with DSpark. JSON Schema and beta strict tools work with DSpark separately from logprobs; **Schema/strict tools + speculative decoding + logprobs is rejected**. Use the basic non-speculative launch if that combination is needed.

Access `/beta/*` directly on this Python API server. The beta route implementation is not currently routed through Atomesh DP/P-D mesh; do not substitute a mesh frontend for this recipe.

## 4. Image requests

### Serving configuration

Use the basic launch in section 1 with the full multimodal checkpoint. Keep `ATOM_MONO_ENABLE=0` and `--level 0`; use the explicit FULL graph configuration from section 1. **Do not add `--method dspark` or `--enable-dp-attention` for images.** If you switched the service to DSpark for section 3, stop that instance and restart the basic service before sending images. The model rejects unsupported image deployments instead of answering from text alone.

The same native instance can handle ordinary text requests and images. Begin with one image request at a time; tune prefill chunks/concurrency only after checking visual answers on your workload. Neither eager nor graph execution guarantees visual correctness.

### Ready-to-run local image examples

The example directory includes `sample-red.png`, a generated red square, and request files for all three APIs. The client sends image bytes; it does not ask the server to read your local path.

```bash
cd /tmp/v41-api-examples/recipes/examples/deepseek-v41-api
export BASE_URL=http://127.0.0.1:8000

curl --fail-with-body -sS "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @chat-image.json
curl --fail-with-body -sS "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' --data-binary @responses-image.json
curl --fail-with-body -sS "$BASE_URL/v1/messages" \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' --data-binary @messages-image.json
```

Each request asks for the square's color, so check that the visible answer describes red. This is a simple image-content smoke check, not a visual benchmark. For streaming, use the corresponding `chat-image-stream.json`, `responses-image-stream.json` or `messages-image-stream.json` and add `curl -N`. Stream framing is the same as each endpoint's text stream.

To use your own image, the included standard-library-only helper writes the six request files (JSON/SSE × three interfaces):

```bash
python3 build-image-requests.py /path/on/the/client/photo.png
# Optional second image: preserves first/second order and asks to compare them.
python3 build-image-requests.py first.png --second second.jpg
```

The helper reads the image on the **client**, base64-encodes it, and writes JSON. Use JPEG, PNG, GIF or WebP. Change its `--prompt` or edit the text block in the generated JSON to ask your own question. GIF means its first frame; this is not video serving.

### Protocol shapes

- **Chat:** a `user` message contains a text block plus `{"type":"image_url","image_url":{"url":"data:image/png;base64,<encoded bytes>","detail":"auto"}}`.
- **Responses:** `input` contains a `user` message with `input_text` and `{"type":"input_image","image_url":"data:image/png;base64,<encoded bytes>","detail":"auto"}`.
- **Messages:** a `user` message contains `text` and `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"<encoded bytes>"}}`. Here `data` is raw base64, without a `data:` prefix.

The bundled JSON files contain actual sample bytes, rather than the placeholders above.

### Public image URLs

Three additional files (`chat-image-url.json`, `responses-image-url.json`, `messages-image-url.json`) use a public HTTPS example image. Replace that URL with a public image URL you control for repeatable tests; the server must have egress access.

- Chat: `image_url.url` is the public URL.
- Responses: `input_image.image_url` is the public URL.
- Messages: image `source` becomes `{"type":"url","url":"https://..."}`.

URLs and redirects must resolve to public addresses. Local paths, `file://`, private/loopback/link-local URLs and file references are refused by default. For a trusted intranet image store, the operator can explicitly set `ATOM_IMAGE_URL_ALLOW_PRIVATE=1` before launch; that permits internal destinations and system proxies. Base64 data avoids a server-side download.

### Placement and limits

| Interface | Supported placement |
| --- | --- |
| Chat | Image parts in `user` messages |
| Responses | `user` and `developer` messages; image blocks in `function_call_output` / `custom_tool_call_output` output |
| Messages | `user` messages; image blocks in `tool_result.content` |

Images in `system` or `assistant` messages are rejected. For multi-turn requests, resend the relevant image-bearing history in that interface's format. Tool-result images require the preceding matching tool call and its returned call ID; they are not standalone user messages.

JPEG, PNG, GIF (first frame) and WebP are detected from their bytes. Per image: 32 MiB, up to 8192 px per side; from 15 images onward, up to 4096 px per side. Per request: at most 600 images, 64 MiB of image bytes and 256 megapixels of decoded images. Public URLs are at most 8192 characters; a download has a 60-second limit. Prompt image tokens plus requested output must also fit the configured model context. These input limits are not a recommendation to start with a 600-image workload.

For Chat and Responses, `detail: "low"` downscales to fit 512×512. `auto`, `high` and `original` retain the incoming dimensions at the API loader; the model processor still performs its own processing.

There is no Files API: `file_id` and Anthropic `file` sources are rejected. Chat's `file_data` form accepts image data URLs, not arbitrary documents. Image output/generation, audio and video are outside this recipe.

## 5. Boundaries and errors

- JSON Schema is a supported subset, not every JSON Schema predicate. External references and unsupported predicates fail before enqueue. Prefer simple typed objects, required fields, enums and bounds; unsupported unbounded `uniqueItems` and complex negations are rejected.
- Schema and strict tools require the DSML tool parser and text serving; no P/D, pipeline parallelism, custom stop strings, ignored EOS, relaxed MTP or synthetic acceptance. Strict schemas additionally reject `minLength`, `maxLength`, `minItems`, `maxItems`, `patternProperties`; supported strict formats are email, hostname, ipv4, ipv6 and uuid. Local definitions/references must follow the documented strict rules.
- With thinking enabled, use `tool_choice: "auto"` for Chat tools; the model may choose to answer without calling a tool. Disable thinking for required/named selection.
- Use a list for ordinary `/v1` stop strings, e.g. `"stop": ["END"]`. Beta accepts a string as well. Do not combine custom stops with Schema/strict tools.
- An unknown served model returns HTTP 400. Invalid fields can produce HTTP 400/422; inspect the returned error body rather than only the HTTP code. Streaming errors/truncation must be handled as terminal failure/incomplete events.
- Enabled Schema/strict constraints and logprobs add work; performance depends on workload and topology. This recipe makes no zero-overhead claim. Start with those request options absent when measuring the ordinary path.
- Images use the native multimodal path and cannot be combined with DSpark, DP attention, P/D, Schema/strict tools, Chat logprobs or beta assistant prefixes. Visual quality remains unstable across concurrency, chunk size and execution mode; validate answer content on representative images. A successful HTTP response or graph capture does not establish image accuracy.

## 6. Merge references

API compatibility: [#2844](https://github.com/ROCm/ATOM/pull/2844). Chat/DSpark probabilities: [#2856](https://github.com/ROCm/ATOM/pull/2856). JSON constraints and model metadata: [#2862](https://github.com/ROCm/ATOM/pull/2862). Beta FIM/prefix: [#3036](https://github.com/ROCm/ATOM/pull/3036). Beta strict tools: [#3090](https://github.com/ROCm/ATOM/pull/3090). Chat image input: [#3111](https://github.com/ROCm/ATOM/pull/3111). Responses/Messages image input: [#3115](https://github.com/ROCm/ATOM/pull/3115). Requirements tracker: [#2864](https://github.com/ROCm/ATOM/issues/2864).

Source review and live checks were refreshed against main `b986b653` on 2026-11-09. All 25 example payloads passed the current request-model checks. The native TP4 launch above loaded the full V4.1-Flash checkpoint and captured CUDA graphs successfully. All 22 text and inline-image examples passed the live smoke checker, including Chat logprobs, Chat/Responses JSON Schema, beta FIM/prefix/strict tools and streaming termination. The three non-streaming image examples correctly identified the red sample; a separate Chat tool-result round trip used the supplied weather result in its answer. These are small functional examples, not a throughput benchmark or a general image-accuracy evaluation. The three public-URL examples, optional DSpark launch, large-image workloads and tool-result images were not live-tested in this run. The issue's historical draft/unmerged labels are stale. See [API image input](../docs/api_image_input.md) for protocol/limits and [V4.1 vision](../docs/deepseek_v41_vision.md) for runtime and visual-quality limitations.
