# ATOM serving & benchmarking guide

ATOM (AiTer Optimized Model) is AMD's lightweight LLM inference engine built on
[AITER](https://github.com/ROCm/aiter) kernels for ROCm/HIP GPUs.  This guide
covers the OpenAI-compatible serving API, programmatic engine usage, benchmarking
tools, profiling, and speculative decoding.

## Quick reference

```bash
# Start the OpenAI-compatible server
python -m atom.entrypoints.openai_server --model <model_name_or_path> --kv_cache_dtype fp8

# Run the online serving benchmark
python -m atom.benchmarks.benchmark_serving \
    --backend vllm --model <model_name_or_path> \
    --base-url http://localhost:8000 \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --num-prompts 1000 --request-rate inf --ignore-eos

# Simple inference example
python -m atom.examples.simple_inference --model <model_name_or_path> --kv_cache_dtype fp8

# Offline profiling
python -m atom.examples.profile_offline --model <model_name_or_path> --kv_cache_dtype fp8

# Accuracy validation with lm-eval
lm_eval --model local-completions \
    --model_args model=<model>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
    --tasks gsm8k --num_fewshot 5
```

## OpenAI-compatible server

The server is implemented in `atom/entrypoints/openai_server.py` using FastAPI
and Uvicorn.  It exposes OpenAI-compatible HTTP endpoints so that existing
clients (curl, OpenAI SDK, lm-eval) work without modification.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (ChatCompletionRequest -> ChatCompletionResponse) |
| `POST` | `/v1/completions` | Text completion (CompletionRequest -> CompletionResponse) |
| `GET`  | `/v1/models` | List available models |
| `GET`  | `/health` | Health check (returns `{"status": "ok"}`) |
| `POST` | `/start_profile` | Start torch profiler on the engine |
| `POST` | `/stop_profile` | Stop torch profiler and flush traces |

### Request models

**ChatCompletionRequest** fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `Optional[str]` | `None` | Model name (validated against the loaded model) |
| `messages` | `Optional[List[ChatMessage]]` | `None` | List of chat messages (`role`, `content`) |
| `prompt` | `Optional[List[ChatMessage]]` | `None` | Alias for `messages` |
| `temperature` | `Optional[float]` | `1.0` | Sampling temperature |
| `top_p` | `Optional[float]` | `1.0` | Nucleus sampling threshold |
| `max_tokens` | `Optional[int]` | `256` | Maximum tokens to generate |
| `stop` | `Optional[List[str]]` | `None` | Stop strings |
| `ignore_eos` | `Optional[bool]` | `False` | Ignore end-of-sequence token |
| `stream` | `Optional[bool]` | `False` | Enable server-sent events streaming |
| `seed` | `Optional[int]` | `None` | Random seed |

**CompletionRequest** fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `Optional[str]` | `None` | Model name |
| `prompt` | `str` | (required) | Text prompt |
| `temperature` | `Optional[float]` | `1.0` | Sampling temperature |
| `top_p` | `Optional[float]` | `1.0` | Nucleus sampling threshold |
| `max_tokens` | `Optional[int]` | `256` | Maximum tokens to generate |
| `stop` | `Optional[List[str]]` | `None` | Stop strings |
| `ignore_eos` | `Optional[bool]` | `False` | Ignore end-of-sequence token |
| `stream` | `Optional[bool]` | `False` | Enable SSE streaming |

### Response models

Both `ChatCompletionResponse` and `CompletionResponse` include:

- `id` — unique request identifier (e.g. `chatcmpl-<uuid>` or `cmpl-<uuid>`)
- `object` — `"chat.completion"` or `"text_completion"`
- `created` — Unix timestamp
- `model` — model name
- `choices` — list of generated completions
- `usage` — token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`)
  plus `ttft_s`, `tpot_s`, and `latency_s` timing fields

Streaming responses use the SSE (Server-Sent Events) protocol with
`data: [DONE]\n\n` as the termination signal.

#### Delivery under load

The API server is a single Python process, so at high concurrency the fixed
per-chunk cost of delivering tokens (detokenize, coroutine wakeup, JSON encode,
socket write) can cap throughput before the GPU does. Two things keep that cost
down:

- **Backlog merging.** Each request's chunks land in a `StreamOutputCollector`
  (`atom/entrypoints/openai/streaming_dispatch.py`), which holds at most one
  chunk per stream: anything arriving behind an unread one merges into it.
  Nothing is held back waiting for more, so a consumer that keeps up sees
  exactly one chunk per engine step.
- **msgspec frame encoding** (`atom/entrypoints/openai/sse.py`), roughly 5.8x
  cheaper per frame than `json.dumps`.

**A token *can* be delivered later than the engine produced it, by a bounded
amount.** Two stages downstream of the collector read the text for markers —
the reasoning channel's delimiters
(`atom/entrypoints/openai/reasoning.py`) and the opening tags of whichever
tool-call format this model uses (`atom/entrypoints/openai/tool_parser/`) —
and neither may hand out a byte that could turn out to be the first character
of one. Both ask the same
question through `MarkerScanner`
(`atom/entrypoints/openai/marker_scanner.py`): release everything except the
longest *suffix* of the buffer that is a prefix of some marker. The wait is
therefore bounded by the longest marker a format declares, a few dozen bytes,
and is usually zero — a chunk whose tail cannot begin a marker is released
whole.

This is worth stating because it used to be unbounded. The rule was "hold
everything once a marker's first character appears *anywhere* in the buffer",
which one `<` in an ordinary answer — `if (a < b)` — satisfied forever, and
the buffer was never cleared while it held. The whole answer then arrived in
a single frame at end of stream, indistinguishable from a hang to a streaming
client, and the scan over that ever-growing buffer made the cost quadratic in
the response length.

Two waits are longer than that. Text inside the reasoning channel is held until
its end marker — not a stall: it is reasoning, and it is delivered as
`reasoning_content` as it arrives. And once a marker that *opens a tool-call
region* appears, everything from it onward belongs to the format until it can
parse the region, which for a real call is its closing tag and for an answer
that merely quotes a marker is end of stream. Nothing is lost there — the
region is released verbatim once it turns out not to be a call — but it does
arrive late, and `atom:stream_longest_silence_seconds` now reports it while it
is happening.

"Opens a region" is asked of the format, not assumed of every marker it
declares. Kimi-K3 declares thirteen and only two of them mean a tool call; the
other eleven are channel framing that wraps every answer it gives, including
`<|open|>response<|sep|>` at the very start. Treating those as a handover meant
a K3 response streamed *nothing* — measured, 324 of 324 characters in one frame
at EOS — which was the common path for that model rather than an edge case.

A start marker is not a promise, and that applies to the handover markers too.
An answer *quoting* one opens a region that then parses to no call, and every
format releases that region verbatim rather than deleting it. K3 was the one
without such a branch: it cut the answer at a quoted call opener and lost 62
characters with no event and `finish_reason` still `stop`.

**The tool's name does not wait for its arguments.** A region is buffered
until it closes, so on a 20 KB file write the client learned *which* tool was
being called only after 5030 of 5040 tokens. Every format carries the name in
its opener, so it is sent as soon as the region reveals it — measured across
all six, chunk 11–21 instead of 225–248.

Two things have to be true before a name goes out. It is one the request
declared in `tools`, and what follows it is this format's own next token
rather than English -- prose can name a real tool, so the first test alone let
"the model writes `<tool_call><function=get_weather>` and then..." announce
`get_weather`. SGLang's cursor parsers announce with neither check and will
emit a call named after whatever follows the tag.

The same pair gates the *unclosed-region* branch of `parse` in all five XML-ish
formats, which exists for a call cut off at `max_tokens` and could not tell
that from prose: it produced a complete zero-argument call, deleted the rest of
the sentence and reported `finish_reason: tool_calls`, so an agentic client ran
a tool nobody asked for. K3 applies only the second test — it carries argument
types on the wire, so it is never handed `tools` and has no declared names to
check a name against.

Where the model *wrote* the name, that name wins. DSML also infers a dropped
tool name from the parameter signature, for a documented malform that omits the
`<invoke>` wrapper entirely; reaching that inference for a merely *truncated*
call scored a different declared tool than the one in the opener — which is
also the opener `peek_name` reads, so the announcement and the parse then
disagreed about the same bytes.

The peek reads a bounded prefix and stops once that prefix has gone by without
a name. Running the format's regex over the whole region on every chunk is
quadratic in the response -- 3.0 → 9.8 → 36 → 137 ms across 2k/4k/8k/16k
tokens, which is the shape `marker_scanner` exists to retire, one layer up.

Peek and parse read *one* rule. Each format writes down the tokens that may
follow the name inside a call -- another parameter, or the close of the very
block the name opened -- and both callers test against that tuple. They used
to encode it twice, a follower set in the peek regex and a truncation test in
`parse`, and four of the five disagreed: Qwen's peek accepted `</tool_call>`,
which closes the *outer* wrapper and leaves the `<function=` block open, so
`parse` read the same bytes as prose and the name went out for a call that
never came. The peek also takes the request's `tools` now, because MiniMax
names a parameter by its own tag and telling `<city>` from `<br>` needs the
schema.

What remains is the honest case: a call the model really was making, cut off
by `max_tokens`. The name is correct information, and `finish_reason` keys on
the arguments so nothing downstream counts it as a call.

Should a format's peek and its parse disagree anyway, the mismatch is logged
and recovered from, not raised. The caller is `flush`, on a stream whose 200
is already sent, so an exception there reaches the client as a connection cut
mid-frame with no `[DONE]` — and on the `n>1` path takes the other choices
with it. The announced name cannot be retracted, but it can be left with no
arguments, which is the same shape every unfulfilled announcement takes and
which nothing downstream counts as a call; the parsed call goes out whole at
the next index.

Kimi-K2 does not announce at all. Its call index and id travel on the wire
(`functions.NAME:INDEX`) and an announcement has to carry both before the
entry that supplies them has arrived; every announced call went out at index
0, so a client accumulating by index overwrote the first call with the
second.

Arguments still wait for the region to close. SGLang streams those too, as
JSON fragments; a response cut short then leaves the client holding an
unterminated object. The residual cost here is narrower: a response truncated
mid-call has sent a name and no arguments, and `finish_reason` / `stop_reason`
key on the *arguments* precisely so that dangling name is not reported as a
tool the client should run.

**Which tool-call format a model uses is decided at startup, not from its
output.** `--tool-call-parser` defaults to `auto`, which renders the model's
chat template with a tools payload — the template's own instructions for
calling one — and runs the `_DETECT_ORDER` cascade on the result. It reads a
Jinja template or a model-side Python encoder (`<model>/encoding/encoding_*.py`,
which is how DeepSeek-V4 ships its), and logs the format it chose. When nothing
is recognised it says so and tool calls are delivered as plain text. There is
no fallback to reading the output — not on either path, which is the point: the
non-streaming path used to run the cascade over the response whenever no format
had been resolved, so an answer that merely quoted another format's section
token had everything from the token onward deleted with `stream=false` and
arrived whole with `stream=true`. A guess is silent, and it is also two
different answers to one request.

**`stream=false` and `stream=true` deliver the same text.** A format's `parse`
returns the content byte-for-byte when it found no tool call; the only thing it
may remove is a marker it declares itself (Kimi-K3's channel tokens, which the
streaming path removes too). Whitespace is not that — every format used to
`.strip()`, which cost a code-block answer its trailing newline on one path
only. The property suite generates this check from the parser registry, so a
format added later is held to it without a new case being written.

The *reasoning* split is held to the same rule one stage earlier, and was not.
Two ways: `</think>` was matched only at position 0, so a model that answers,
opens a `<think>` block and answers again had it extracted when streamed and
handed over as literal tags with the chain of thought inside `content` when
not — and both halves were then `.strip()`ed, which is the trailing-newline
bug above, in the stage before it. A model writes `</think>\n\nThe answer.`;
`stream=true` delivers `"\n\nThe answer."` at every real chunk size and
`stream=false` delivered `"The answer."`. Measured over 12544 (dialect, shape,
chunking) comparisons, the two agreed byte-for-byte on 50% of them; they now
agree on all of them, and the property that says so is byte-exact rather than
word-level.

The streaming filter also stopped eating the newline after its end marker. It
only ever saw what happened to be buffered when the marker arrived, so the
same answer kept those bytes at one chunk size and lost them at another —
there was no chunk-invariant behaviour on that whitespace for the other path
to match even if it had wanted to.

**`tool_choice: "none"` suppresses the call, not the answer.** It used to be
enforced where the events are *sent* — twelve places across two endpoints —
while the parser went on consuming the region, so the model's own words were
deleted and nothing took their place: 89 characters of a 95-character answer,
no event, `finish_reason: stop`. The rule now lives at the one place the
parser is chosen, which is also the right reading — the request said this
cannot be a call, so it is prose — and it costs less, since nothing is parsed
in order to be discarded. `/v1/messages` reads the field too, in Anthropic's
`{"type": "none"}` spelling; it previously parsed it off the request and used
it nowhere, so a client that forbade tool calls got `tool_use` blocks and
`stop_reason: tool_use` anyway.

**`thinking` is answered in the prompt, not in the response.** On
`/v1/messages`, `thinking: {"type": "disabled"}` sets the chat template's own
reasoning switch, so the model emits no chain of thought — there is then none
to separate, none to discard, and none for the tool parser to misread.
*Separation* stays unconditional, exactly as on `/v1/chat/completions`: the
tool parser is a second reader of the same text, so a chain of thought left in
it is one the tool parser will try to parse.

That ordering is the whole of it. Handling an unwanted chain of thought *after*
generating it fails three different ways — discarding it returns an empty
message for a reasoning model stopped at `max_tokens`; relabelling it as `text`
hands the client the thing it declined; and leaving it unseparated feeds it to
the tool parser, which is a second reader of the same text and read one model's
musing about `<function=NAME>` as a call to a tool named `NAME`. SGLang answers
the same field the same way (`apply_reasoning_enabled`), and vLLM gets it
structurally by having no such field: its reasoning parser runs unconditionally
and `include_reasoning` only suppresses the result after the split.

Which kwarg carries the switch is resolved at startup by rendering the template
twice and comparing, because a template silently ignores a kwarg it does not
read. On this box: Qwen3/Qwen3.5 `enable_thinking`, Kimi-K3 `thinking`,
MiniMax-M3 `thinking_mode="disabled"`, DeepSeek-V4 `thinking_mode="chat"`.
A model whose template has no switch is named in the startup log. Its reasoning
cannot be prevented, so `thinking: {"type": "disabled"}` is answered the only
way left: the text is still separated, and the `thinking` blocks are withheld.
That is the one downstream suppression there is, and it is reached only when
the prompt could not carry the answer — without it an explicit opt-out was
honoured at neither layer. A response that was *nothing but* reasoning then
ends on an empty text block, which is the honest reply to "do not think".

Two details that bite: `{"type": "disabled"}` is a non-empty object, so testing
the field for truthiness read the standard off-switch as on; and an *absent*
`thinking` leaves the model's own default alone rather than switching reasoning
off, at both layers or neither, so an existing caller's answers do not change.

**A stalled response is visible while it is stalled.** Every SSE frame leaves
through `_client_stream`, which times the gap before each one and registers it,
and `atom:stream_longest_silence_seconds` reports the age of the oldest gap
in flight. Zero when every stream has just been served; non-zero and growing is
a response whose client is receiving nothing. A gap longer than 30 seconds also
logs a line naming the request — the gauge cannot see a stall that has already
recovered by scrape time. Neither costs a timer: `asyncio.wait_for` measured
1.38 us per frame per stream against 0.07 us for a timestamp and a dict entry.
This exists because the symptom that started this work was ten minutes of
silence with every metric looking healthy.

Measured at the frame and not at `StreamOutputCollector.get`, which is where it
started and which cannot see the thing it was built for. The collector is where
a stream waits for the *engine*, but the reasoning read-ahead and the tool-call
read-ahead sit between it and the socket, and while either withholds, the
collector wakes on every token. Measured: an answer quoting a tool marker fed
126 tokens and sent the client 6 frames, and the gauge read zero. At the frame
it reads the silence.

The wait for the *first* frame is still excluded, and moving out did not
change that — a claim this paragraph made and did not hold. Every response
generator awaits the collector before yielding anything, so that wait is
admission, queueing and prefill: timing it put 0.2 s on the gauge for a
request 200 ms into a queue with no token yet produced, which is
`atom:requests_waiting` under another name, and past the threshold would log a
line per admitted request blaming the read-ahead.

One consequence matters when reading benchmark output. ITL is sampled once per
received SSE chunk (`backend_request_func.py`, `benchmark_serving.py`), so
merging N tokens into one chunk removes N-1 samples and stretches the gaps that
remain: **every ITL statistic - mean, median and p99 alike - inflates by roughly
the merge factor**, without any token being delivered later. Measured on
Qwen3.5-27B-FP8 tp4 at concurrency 2048, mean ITL read 191.8 ms against a TPOT
of 126.6 ms, while the same workload with merging disabled read 122.9 ms against
a TPOT of 123.3 ms.

**Compare TPOT, not ITL, whenever merging is active.** It is the only
token-normalized latency in the report (`latency - ttft` over `output_len - 1`),
so it stays honest at any merge factor. The ratio ITL/TPOT is itself the useful
number: it *is* the merge factor, and a value near 1.0 means the frontend is
keeping up and nothing ever merged.

### Server startup

```bash
python -m atom.entrypoints.openai_server \
    --model <model_name_or_path> \
    --kv_cache_dtype fp8 \
    --host 0.0.0.0 \
    --server-port 8000
```

Server-specific CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--server-port` | `8000` | HTTP port (note: `--port` is for internal engine communication) |
| `--timeout-keep-alive` | `5` | Seconds an idle keep-alive connection is held. Pooling clients hold their end longer (aiohttp defaults to 15s), so a caller that pauses for longer than this reuses a socket the server already closed and has to re-send. Raise it past the caller's idle window to avoid that |
| `--disable-uvicorn-access-log` | off | Stop uvicorn logging a line per HTTP request. It copies a `LogRecord` and writes to the same stdout as the engine, on the event loop |
| `--tool-call-parser` | `auto` | Tool-call wire format. `auto` reads it from the model's chat template at startup (Jinja, or a model-side `encoding/encoding_*.py`); a name — `dsml`, `glm`, `kimi`, `kimi_k3`, `minimax`, `qwen` — overrides. When neither resolves, tool calls are delivered as plain text and the startup log says so; the format is never guessed from output. An unknown name is refused rather than silently disabling tool parsing, and on atomesh that refusal happens before the weights load. Accepted by the atomesh entrypoint too, where it is also forwarded to the mesh router, which declares a flag of the same name and its own vocabulary for it |

All `EngineArgs` arguments are also accepted (see Section 7 for the full list).

### Example: curl

```bash
# Non-streaming chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'

# Streaming text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 64,
    "stream": true
  }'
```

## Programmatic API (LLMEngine)

The `LLMEngine` class in `atom/model_engine/llm_engine.py` provides a
Python-native interface for inference without running an HTTP server.

### Initialization

```python
from atom import LLMEngine, SamplingParams

engine = LLMEngine(model="deepseek-ai/DeepSeek-R1", kv_cache_dtype="fp8",
                   tensor_parallel_size=8)
```

`LLMEngine.__init__(model, **kwargs)` accepts all `Config` field names as
keyword arguments (e.g. `tensor_parallel_size`, `kv_cache_dtype`,
`max_model_len`, `data_parallel_size`, `gpu_memory_utilization`).

### SamplingParams

Defined in `atom/sampling_params.py`:

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_strings: Optional[list[str]] = None
```

### Core methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `(prompts: list[str], sampling_params) -> list[dict]` | Synchronous batch generation; blocks until all prompts complete |
| `add_request` | `(prompt_or_tokens_list, sampling_params_list, stream_callback=None)` | Submit requests for asynchronous processing |
| `step` | `() -> list[Sequence]` | Retrieve completed sequences |
| `is_finished` | `() -> bool` | Check whether all pending requests have completed |
| `start_profile` | `()` | Start torch profiler on all workers |
| `stop_profile` | `()` | Stop torch profiler and write traces |
| `print_mtp_statistics` | `()` | Print speculative decoding acceptance statistics |

### Synchronous generation example

```python
from atom import LLMEngine, SamplingParams

engine = LLMEngine(model="meta-llama/Meta-Llama-3-8B", kv_cache_dtype="fp8")
params = SamplingParams(temperature=0.6, max_tokens=256)

outputs = engine.generate(["Explain quantum computing in simple terms."], params)
for out in outputs:
    print(out["text"])
```

Each output dictionary contains: `text`, `token_ids`, `latency`,
`finish_reason`, `num_tokens_input`, `num_tokens_output`, `ttft`, and `tpot`.

### Asynchronous / streaming usage

```python
engine.add_request(
    prompt_or_tokens_list=["Hello world", "How are you?"],
    sampling_params_list=SamplingParams(temperature=0.8, max_tokens=128),
    stream_callback=my_callback,  # called per-token with RequestOutput
)

while not engine.is_finished():
    completed = engine.step()
    # process completed sequences
```

## Simple inference

The `atom/examples/simple_inference.py` script provides a quick way to validate
model loading and generation.

### Usage

```bash
python -m atom.examples.simple_inference \
    --model meta-llama/Meta-Llama-3-8B \
    --kv_cache_dtype fp8 \
    --temperature 0.6
```

### What it does

1. Parses all `EngineArgs` plus `--temperature` (default `0.6`).
2. Creates an `LLMEngine` via `EngineArgs.from_cli_args(args).create_engine()`.
3. Applies the model's chat template to four built-in prompts (English and
   Chinese) with `enable_thinking=True`.
4. Runs a warmup generation, then generates completions for the batch.
5. Calls `llm.print_mtp_statistics()` to report speculative decoding stats
   (if MTP is enabled).

## Benchmarking

ATOM ships a comprehensive online serving benchmark in
`atom/benchmarks/benchmark_serving.py` (adapted from vLLM's benchmarking
tooling).

### Metrics

The `BenchmarkMetrics` dataclass tracks:

| Metric | Abbreviation | Description |
|--------|--------------|-------------|
| Time to First Token | **TTFT** | Latency from request submission to the first generated token |
| Time per Output Token | **TPOT** | Average latency per output token (excluding the first) |
| Inter-Token Latency | **ITL** | Latency between successive output tokens |
| End-to-End Latency | **E2EL** | Total latency from request send to full response receipt |
| Request Throughput | -- | Completed requests per second |
| Output Token Throughput | -- | Generated tokens per second |
| Total Token Throughput | -- | (input + output) tokens per second |
| Request Goodput | -- | Requests per second meeting SLO targets |
| Concurrency | -- | Average in-flight requests (sum of per-request end-to-end latency / benchmark duration) |
| Accept Length | -- | Speculative decoding only: mean tokens per model forward (1 + accepted draft tokens), from `/debug/mtp_stats`; printed only when spec-decode is enabled |
| Acceptance Rate | -- | Speculative decoding only: fraction of drafted tokens accepted (accepted / drafted), from `/debug/mtp_stats`; printed only when spec-decode is enabled |

For each latency metric, mean, median, standard deviation, and configurable
percentiles (default: P99) are reported.

### Key CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backend` | `vllm` | Backend type. Choices: `tgi`, `vllm`, `lmdeploy`, `deepspeed-mii`, `openai`, `openai-chat`, `tensorrt-llm`, `scalellm`, `sglang` |
| `--model` | (required) | Model name or path |
| `--base-url` | `None` | Server base URL (e.g. `http://localhost:8000`) |
| `--host` | `127.0.0.1` | Server host (used when `--base-url` is not set) |
| `--port` | `8000` | Server port (used when `--base-url` is not set) |
| `--endpoint` | `/v1/completions` | API endpoint path |
| `--dataset-name` | `sharegpt` | Dataset type: `sharegpt`, `burstgpt`, `sonnet`, `random`, `hf` |
| `--dataset-path` | `None` | Path to dataset file or HuggingFace dataset ID |
| `--num-prompts` | `1000` | Number of prompts to benchmark |
| `--request-rate` | `inf` | Requests per second (`inf` = send all at once) |
| `--burstiness` | `1.0` | Burstiness factor (1.0 = Poisson process) |
| `--max-concurrency` | `None` | Maximum concurrent requests |
| `--ignore-eos` | `False` | Ignore EOS token in generation |
| `--save-result` | `False` | Save results to JSON |
| `--result-dir` | `None` | Directory for result JSON files |
| `--result-filename` | `None` | Custom filename for results |
| `--percentile-metrics` | `ttft,tpot,itl` | Comma-separated metrics to report percentiles for |
| `--metric-percentiles` | `99` | Comma-separated percentile values (e.g. `25,50,75,99`) |
| `--goodput` | `None` | SLO targets as `KEY:VALUE` pairs (e.g. `ttft:100 tpot:50`) |
| `--profile` | `False` | Enable torch profiler during the benchmark run |
| `--tokenizer` | `None` | Custom tokenizer name or path |
| `--seed` | `0` | Random seed |

**Random dataset options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--random-input-len` | `1024` | Input token length |
| `--random-output-len` | `128` | Output token length |
| `--random-range-ratio` | `1.0` | Length variation ratio |
| `--random-prefix-len` | `0` | Fixed prefix token length |
| `--use-chat-template` | `False` | Apply chat template to random prompts |

### Backend request functions

Defined in `atom/benchmarks/backend_request_func.py`:

| Backend Key | Function | Protocol |
|-------------|----------|----------|
| `vllm` | `async_request_openai_completions` | OpenAI Completions API (streaming) |
| `openai` | `async_request_openai_completions` | OpenAI Completions API (streaming) |
| `openai-chat` | `async_request_openai_chat_completions` | OpenAI Chat Completions API (streaming) |
| `tgi` | `async_request_tgi` | TGI `generate_stream` |
| `tensorrt-llm` | `async_request_trt_llm` | TRT-LLM `generate_stream` |
| `deepspeed-mii` | `async_request_deepspeed_mii` | DeepSpeed-MII |
| `lmdeploy` | `async_request_openai_completions` | OpenAI Completions API |
| `scalellm` | `async_request_openai_completions` | OpenAI Completions API |
| `sglang` | `async_request_openai_completions` | OpenAI Completions API |

Each function uses `RequestFuncInput` and returns a `RequestFuncOutput` with
timing data (`ttft`, `itl`, `latency`, `tpot`).

### Full benchmark example

```bash
# 1. Start the server
python -m atom.entrypoints.openai_server \
    --kv_cache_dtype fp8 -tp 8 --model deepseek-ai/DeepSeek-R1

# 2. Run benchmark
MODEL=deepseek-ai/DeepSeek-R1
ISL=1024
OSL=1024
CONC=128
PORT=8000
RESULT_FILENAME=Deepseek-R1-result

python -m atom.benchmarks.benchmark_serving \
    --model=$MODEL --backend=vllm --base-url=http://localhost:$PORT \
    --dataset-name=random \
    --random-input-len=$ISL --random-output-len=$OSL \
    --random-range-ratio 0.8 \
    --num-prompts=$(( $CONC * 10 )) \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=./ --result-filename=$RESULT_FILENAME.json
```

## Profiling

ATOM supports PyTorch profiling via environment variables, HTTP endpoints, and
the programmatic API.

### Configuration

| Mechanism | Description |
|-----------|-------------|
| `--torch-profiler-dir <dir>` | CLI arg to set the trace output directory |
| `ATOM_TORCH_PROFILER_DIR` env var | Sets the default `torch_profiler_dir` in `Config` |
| `ATOM_PROFILER_MORE=1` env var | Enables detailed profiling: `record_shapes`, `with_stack`, `profile_memory` |
| `ATOM_PROFILER_TIMEOUT=<seconds>` env var | Overrides the `stop_profile` timeout; default is 300 seconds |
| `ATOM_ENABLE_DETAILED_ANNOTATION=1` env var | Appends attention FLOP aggregates (`sqsq`, `sqsk`, `sk`) to the `prefill[]`/`decode[]` trace labels while profiling is active (see [CUDA-graph capture traces](#cuda-graph-capture-traces)) |

When a profiler directory is configured, each worker saves traces to a
rank-specific subdirectory:

- Multi-GPU with DP: `{profiler_dir}/dp{dp_rank}_tp{rank}/`
- Single-GPU / TP-only: `{profiler_dir}/rank_{rank}/`

Traces are saved in gzip-compressed TensorBoard format and can be viewed with
`tensorboard --logdir <profiler_dir>` or Chrome's `chrome://tracing`.

### Online profiling (HTTP)

While the server is running, start and stop profiling with HTTP requests:

```bash
# Start profiling
curl -s -S -X POST http://127.0.0.1:8000/start_profile

# ... run your workload ...

# Stop profiling and flush traces
curl -s -S -X POST http://127.0.0.1:8000/stop_profile
```

The server must be started with `--torch-profiler-dir` or with
`ATOM_TORCH_PROFILER_DIR` set for these endpoints to produce traces.
For large traces, set `ATOM_PROFILER_TIMEOUT` higher before starting the server.

### Programmatic profiling

```python
engine = LLMEngine(model="Qwen/Qwen3-0.6B", torch_profiler_dir="./traces")

engine.start_profile()
outputs = engine.generate(prompts, sampling_params)
engine.stop_profile()
# Traces written to ./traces/rank_0/
```

### Offline profiling script

`atom/examples/profile_offline.py` provides a self-contained offline profiling
workflow:

```bash
python -m atom.examples.profile_offline \
    --model Qwen/Qwen3-0.6B \
    --kv_cache_dtype fp8 \
    --torch-profiler-dir ./profiler_traces \
    --input-length 128 \
    --output-length 32 \
    --bs 4
```

Script-specific arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-length` | `128` | Approximate input prompt length in tokens |
| `--output-length` | `32` | Output generation length in tokens |
| `--bs` | `1` | Batch size (number of parallel requests) |
| `--random-input` | `False` | Use random token input instead of predefined text |

If `--torch-profiler-dir` is not specified, the script defaults to
`./profiler_traces`.

### Profiling during benchmarks

The benchmark tool can trigger profiling automatically via `--profile`:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model <model> --backend vllm \
    --base-url http://localhost:8000 \
    --dataset-name random --num-prompts 100 \
    --profile
```

This sends `POST /start_profile` before the benchmark and
`POST /stop_profile` after completion.

### CUDA-graph capture traces

During CUDA-graph capture (server bring-up), ATOM can emit one trace file per
captured batch size instead of a single combined blob. This makes each graph's
capture cost easy to inspect in isolation and keeps individual trace files
small. Capture-trace profiling is gated on `--mark-trace` (with
`--torch-profiler-dir`/`ATOM_TORCH_PROFILER_DIR` set).

Each file covers one full iteration of the capture loop: the warmup forward
followed by the graph capture itself. Both are needed — inside
`torch.cuda.graph(...)` the stream is in capture mode, so kernel launches are
recorded as graph nodes rather than dispatched, and a trace of that region
alone has an empty GPU track. The warmup forward is where the kernels actually
run.

The traces are written to:

```
{profiler_dir}/capture_traces/bs_<bs>_q_<max_q_len>_rank<rank>.json.gz
```

where `<bs>` is the captured batch size, `<max_q_len>` the query-length bucket
(`1` without speculative decoding, `mtp_k + 1` with a drafter, and one file per
bucket when DSpark expands them — see
[Speculative decoding](#speculative-decoding-mtp)), and `<rank>` the worker
rank. Each file is a gzip-compressed Chrome trace viewable with
`chrome://tracing` or TensorBoard.

Like the run-phase profiler, these traces carry `record_shapes`, `with_stack`,
and `profile_memory` only when `ATOM_PROFILER_MORE=1`. Leave it unset unless you
need the shapes or Python stacks — stack capture runs on every rank and
noticeably stretches server bring-up.

To additionally annotate the run-phase `prefill[]`/`decode[]` labels with the
attention FLOP aggregates used for roofline analysis, set
`ATOM_ENABLE_DETAILED_ANNOTATION=1` (see [Configuration](#configuration)). The added
fields are `sqsq` (Σ N_Q²), `sqsk` (Σ N_Q·N_KV), and `sk` (Σ N_KV), summed over
every request in the forward. These are attention-quadratic terms only — a full
roofline still requires GEMM FLOPs and bytes moved.

## Speculative decoding (MTP)

ATOM supports Multi-Token Prediction (MTP) for DeepSeek models using the
Eagle-style speculative decoding framework.

### Architecture

- **EagleProposer** (`atom/spec_decode/eagle.py`): Loads and runs the draft
  (MTP) model to propose speculative tokens.  Supports the `DeepSeekMTPModel`
  architecture via `DeepSeekMTP`.
- **RejectionSampler** (`atom/model_ops/rejection_sampler.py`): Implements
  greedy rejection sampling with a Triton kernel.  Compares draft token IDs
  against target model argmax and accepts matching prefixes; appends a bonus
  token if all drafts are accepted.

### Configuration

Enable MTP via CLI arguments:

```bash
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    --kv_cache_dtype fp8 -tp 8 \
    --method mtp \
    --num-speculative-tokens 1
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `None` | Speculative method: `mtp` (DeepSeek MTP) or `eagle3` (EAGLE 3 / EAGLE 3.1 — see [`eagle3_speculative_decoding.md`](eagle3_speculative_decoding.md)) |
| `--num-speculative-tokens` | `1` | Number of draft tokens per iteration (draft model runs this many autoregressive steps) |
| `--draft-model` | `None` | Path or HF repo of the speculative draft model. Required for `--method eagle3`; the draft's `config.json` drives EAGLE 3 vs EAGLE 3.1 toggles automatically |
| `--spec-decode-acceptance-length` | `None` | Benchmark-only: force a mean acceptance length in `[1, num_speculative_tokens + 1]`, ignoring real draft/target agreement. See [Forced acceptance length](#forced-acceptance-length) |
| `--spec-decode-acceptance-rate` | `None` | The same knob as a rate in `[0, 1]`, i.e. `(length - 1) / num_speculative_tokens`. Mutually exclusive with the above |

### MTP statistics

ATOM tracks acceptance statistics at runtime:

- **total_draft_tokens**: Total number of draft tokens proposed
- **total_accepted_tokens**: Number of draft tokens accepted by rejection sampling
- **acceptance_rate**: Ratio of accepted to draft tokens

Statistics are logged every 1000 draft tokens and can be printed on demand:

```python
engine.print_mtp_statistics()
```

Example output:
```text
MTP Statistics:
  Total draft tokens: 5000
  Accepted tokens:    4250
  Acceptance rate:    85.00%
```

### How rejection sampling works

1. The draft model generates `num_speculative_tokens` token predictions
   autoregressively using argmax.
2. The target model verifies all draft tokens in a single forward pass.
3. The `rejection_greedy_sample_kernel` (Triton) compares each draft token
   against the target model's argmax:
   - If they match, the token is accepted.
   - On the first mismatch, the target model's token replaces it and all
     subsequent draft tokens are discarded.
   - If all draft tokens match, a bonus token from the target model is
     appended.

### Forced acceptance length

Speculative throughput is dominated by how many tokens each target forward
emits, so a run cannot be compared against another engine unless both accept at
the same rate. `--spec-decode-acceptance-length` pins that number: the sampler
stops comparing draft against target and instead accepts draft tokens with a
fixed per-position probability, hitting the requested mean acceptance length.
It exists to benchmark the serving system while a draft head is still training,
and to replay a published acceptance-length figure such as an
[InferenceX golden AL](https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/README.md).

```bash
python -m atom.entrypoints.openai_server \
    --model /models/Kimi-K3 \
    --draft-model /models/Kimi-K3-DSpark \
    --method dspark \
    --num-speculative-tokens 7 \
    --spec-decode-acceptance-length 3.78
```

Acceptance length counts the target's own guaranteed token, matching vLLM's
`synthetic_acceptance_length` and SGLang's `SGLANG_SIMULATE_ACC_LEN`, so a
published figure goes in unchanged. The budget is spent on the earliest
positions — length `3.78` over 7 draft slots accepts 2 tokens always and a 3rd
with probability `0.78` — which is the minimum-variance schedule vLLM and
SGLang also use, so the accepted-length distribution matches and not just its
mean. Read the realized value back from `average_tokens_per_forward` on
`/debug/mtp_stats` (or the `atom:mtp_average_tokens_per_forward` metric).

Two caveats:

- Generated text is meaningless, because tokens are accepted without agreeing
  with the target. Never run an accuracy evaluation with this enabled.
- It cannot be combined with the DSpark confidence scheduler
  (`--dspark-config '{"confidence_schedule": true}'`), which picks each
  request's verify length at runtime; a short one silently caps acceptance
  below the requested length, so the combination is rejected at startup.

The full reference — the resolved schedule, the rate-based spelling, and how to
replay a golden AL curve — is in
[`forced_acceptance_length.md`](forced_acceptance_length.md).

## Deployment examples

### Single-GPU

```bash
python -m atom.entrypoints.openai_server \
    --model Qwen/Qwen3-0.6B \
    --kv_cache_dtype fp8
```

### Multi-GPU with tensor parallelism

```bash
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    --kv_cache_dtype fp8 \
    -tp 8
```

### Docker deployment

```bash
# Pull the ROCm PyTorch image
docker pull rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Launch container
docker run -it --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME:/home/$USER \
    -v /mnt:/mnt \
    -v /data:/data \
    --shm-size=16G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Inside the container
pip install amd-aiter
git clone https://github.com/ROCm/ATOM.git && cd ATOM && pip install .

# Start serving
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    --kv_cache_dtype fp8 -tp 8
```

### Engine CLI arguments (EngineArgs)

These arguments are available for all entrypoints (server, examples, and any
script using `EngineArgs.add_cli_args`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3-0.6B` | Model name or path |
| `--trust-remote-code` | `False` | Trust remote code from HuggingFace |
| `--tensor-parallel-size`, `-tp` | `1` | Tensor parallel size |
| `--data-parallel-size`, `-dp` | `1` | Data parallel size |
| `--enforce-eager` | `False` | Disable CUDA graph capture; use eager execution |
| `--enable_prefix_caching` | `False` | Enable prefix caching |
| `--port` | `8006` | Internal engine communication port |
| `--kv_cache_dtype` | `bf16` | KV cache dtype: `bf16` or `fp8` |
| `--block-size` | `16` | KV cache block size |
| `--max-model-len` | `None` | Maximum context length (defaults to HF config) |
| `--max-num-batched-tokens` | `16384` | Maximum tokens per batch |
| `--max-num-seqs` | `512` | Maximum sequences per batch |
| `--gpu-memory-utilization` | `0.9` | GPU memory utilization (0.0 to 1.0) |
| `--scheduler-delay-factor` | `0.0` | Delay factor before scheduling next prompt |
| `--cudagraph-capture-sizes` | `[1,2,4,...,256]` | Batch sizes for CUDA graph capture |
| `--level` | `3` | Compilation level (0-3); 3 = torch.compile |
| `--load_dummy` | `None` | Dummy weights (no checkpoint read). Bare flag / `=empty`: skip load (uninitialized). `=zero`: all-zero. `=xavier`: xavier for bf16, constant target magnitude for fp4/fp8 |
| `--enable-expert-parallel` | `False` | Enable expert parallelism for MoE |
| `--enable-dp-attention` | `False` | Enable data-parallel attention |
| `--torch-profiler-dir` | `None` | Directory for torch profiler traces |
| `--method` | `None` | Speculative decoding method (`mtp`) |
| `--num-speculative-tokens` | `1` | Number of speculative tokens per step |

## Accuracy validation

ATOM supports accuracy validation through the
[lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) framework via
the OpenAI-compatible API.

### Setup

```bash
pip install lm-eval[api]
```

### Run evaluation

Start an ATOM server, then run lm-eval against it:

```bash
# Start server
python -m atom.entrypoints.openai_server \
    --model meta-llama/Meta-Llama-3-8B \
    --kv_cache_dtype fp8

# Run evaluation
lm_eval --model local-completions \
    --model_args model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
    --tasks gsm8k \
    --num_fewshot 5
```

Any lm-eval task can be used.  The `local-completions` model type sends
requests to the `/v1/completions` endpoint, making it compatible with the ATOM
server without modification.

## Source files

| File | Description |
|------|-------------|
| `atom/entrypoints/openai_server.py` | OpenAI-compatible API server (FastAPI + Uvicorn) |
| `atom/entrypoints/openai/streaming_dispatch.py` | `StreamBatchDispatcher` (per-engine-step cross-thread dispatch), `StreamOutputCollector` (per-request delivery, folds a backlog) and the silence watchdog |
| `atom/entrypoints/openai/sse.py` | SSE frame encoding (`data_frame`, `event_frame`) on a shared msgspec encoder |
| `atom/entrypoints/openai/marker_scanner.py` | `MarkerScanner` — the one rule for how much of a stream is safe to release |
| `atom/entrypoints/openai/reasoning.py` | Splits the reasoning channel from the answer; seeded per request by `prompt_starts_in_reasoning` |
| `atom/entrypoints/openai/chat_encoders.py` | Renders the chat template, and the two startup probes of it: `render_probe_prompt` (what the prompt tells the model) and `chat_template_source` (what the template does with a reply) |
| `atom/entrypoints/openai/tool_parser/registry.py` | Which format a model emits, resolved once at startup from its chat template |
| `atom/entrypoints/openai/tool_parser/` | Per-format tool-call parsing; each format declares its `START_MARKERS` |
| `atom/model_engine/llm_engine.py` | `LLMEngine` programmatic API |
| `atom/sampling_params.py` | `SamplingParams` dataclass |
| `atom/model_engine/arg_utils.py` | `EngineArgs` CLI argument definitions and engine factory |
| `atom/examples/simple_inference.py` | Simple batch inference example |
| `atom/examples/profile_offline.py` | Offline profiling tool |
| `atom/benchmarks/benchmark_serving.py` | Online serving benchmark (`BenchmarkMetrics`, dataset sampling, result reporting) |
| `atom/benchmarks/backend_request_func.py` | Async HTTP request functions for each backend (`RequestFuncInput`, `RequestFuncOutput`, `ASYNC_REQUEST_FUNCS`) |
| `atom/benchmarks/benchmark_utils.py` | `convert_to_pytorch_benchmark_format` utility |
| `atom/spec_decode/eagle.py` | `EagleProposer` -- MTP draft model for DeepSeek speculative decoding |
| `atom/model_ops/rejection_sampler.py` | `RejectionSampler` with Triton greedy rejection kernel |
| `atom/config.py` | `Config`, `CompilationConfig`, `SpeculativeConfig` dataclasses |
| `atom/model_engine/model_runner.py` | `ModelRunner` with `start_profiler`/`stop_profiler` and MTP statistics |
