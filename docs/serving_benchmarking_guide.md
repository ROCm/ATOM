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
parse the region.

##### What is held, and for how long

The rule is that a byte is buffered only while its **destination field** is
undecided — `content` or `tool_calls` — because an SSE frame cannot be taken
back. Four cases, and only two of them wait:

| Where the byte is | Held? | Until |
|---|---|---|
| Before any start marker | no | — `MarkerScanner`, bounded by the longest marker and asserted there |
| Inside a region, before the name is legible | **yes** | the format can name a declared tool, usually the first 30–70 characters |
| Inside a region, after that — the argument values | **yes** | the region closes |
| After the region closed | no | — released as it arrives |

The name goes out as soon as it is legible and the request declared it
(`tool_call_start`), so a client learns *which* tool is being called at chunk
7–24 rather than after the whole payload.

**Argument values wait, deliberately.** vLLM and SGLang stream them as JSON
fragments for their JSON-shaped formats; a response cut off by `max_tokens`
then leaves the client accumulating an object it cannot parse. Buffering them
means five of the six formats here hand back *valid* JSON even for a truncated
call — the half-written value becomes a string — which streaming fragments
gives up. (Kimi-K2 is the exception: it passes the model's bytes through, so a
truncated call already yields `{"city": "Par`. That is a separate defect.)
Both of those engines buffer for their tag-shaped formats too, K3 included, so
this is not a gap against them.

**The region closes on the call's own closer**, not on the wrapper's. Every
format's grammar lists the call closer among the terminators of an argument
value — `</function>` for Qwen, `</invoke>` for DSML and MiniMax,
`<|close|>call` for K3 — so a model writing one inside a parameter ends the
parameter, and the literal can never hide in a value. The *wrapper* closer
(`</tool_call>`) can, which is why it serves only as a trigger to look and
never as the answer. Declared as `CALL_SELF_CLOSERS`; `REGION_END_MARKERS`
overrides it where a region is larger than one call, which is Kimi-K2's
section.

Getting that wrong is expensive and was: while only Kimi declared a region
end, everything a model wrote *after* its tool call waited for end of stream —
0 of 397 characters streamed on five of six formats, and "call a tool, then
explain the result" is the ordinary agentic shape.

**A region that never closes is still held to end of stream.** An answer that
merely quotes its own opener is the case: measured on a GLM answer naming the
tag at character 29 and then explaining for 1234 more, 98% arrives in one
frame at EOS. Nothing is lost — the region is released verbatim once it turns
out not to be a call — and `atom:stream_longest_silence_seconds` reports the
wait while it happens. vLLM and SGLang both do the same here, and SGLang's K3
detector drops the text rather than releasing it.

A probe that gave up on a region producing nothing after N bytes was written
for this and reverted. It rests on acceptance being monotone in how many bytes
have arrived, and that is false: MiniMax gates its in-progress test on the
first tag being in the declared schema, and DSML's wrapper-less and
direct-JSON branches match no prefix at all — so real calls over N bytes were
delivered as raw text with `finish_reason: stop` on three of the six formats.
It was quadratic besides, because giving up re-fed bytes that immediately
reopened a region with a fresh budget: 1.19 ms to 18.2 s on a 250 KB answer,
in the request coroutine. Fixing the latency needs the *format* to say "this
can no longer become a call", which is a different question from "does not
parse yet" and one no format answers today.

A second version was written and reverted for the same reason: bounded to the
256-byte peek window, asked once, and restricted to `tool_choice: "none"`
where nothing would be dispatched anyway. It measured clean on every shape the
corpus carries — and the corpus carries one form per format, because it is
generated from `render_call`. DSML's direct-JSON body needs the whole object,
so a real call with a payload past the window is invisible to any head-sized
peek and would have gone out as text.
`TestAGiveUpProbeStaysReverted` now carries that shape, with a positive
control asserting DSML actually accepts it: the first draft of that test
invented both shapes from this paragraph rather than from the parser, DSML
accepted neither, and it therefore proved nothing.

"Opens a region" is asked of the format, not assumed of every marker it
declares. Kimi-K3 declares 16 and only two of them mean a tool call; the
rest are channel framing that wraps every answer it gives, including
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
being called only after 5030 of 5040 tokens. Four of the six formats can
recognise a call that has not finished arriving, and for those the name is
sent as soon as the region reveals it — chunk 11–21 instead of 225–248.

The name is read out of `parse_region` itself, over the region so far and with
`at_end=False`. That is what makes the early name and the parsed call agree:
same function, same enumeration, the second run seeing a superset of the
first's bytes. Every format used to answer this with a regex of its own, and
four of the five that had one disagreed with their own parse — Qwen's peek
accepted `</tool_call>`, which closes the *outer* wrapper and leaves the
`<function=` block open; DeepSeek-V4's skipped a self-closing
`<invoke name="x"/>` its parse returned first, putting three tool calls on the
`/v1/messages` wire for a response containing two. There is no separate peek
now, so there is nothing left to disagree.

`at_end` is the only difference between the two questions. With it — the
region has closed, or the stream has — a token cut off part-way through is all
there will ever be, so a prefix counts, which is what a call truncated by
`max_tokens` looks like. Without it a prefix means "not yet", and accepting
one let a chunk boundary landing one character into `<br>` name a tool for
prose. Same bytes, announced at one chunk size and silent at another.

A name only goes out for a tool the request declared. Prose can name a real
tool, so that alone is not enough — the follower test above is the other half.
SGLang's cursor parsers announce with neither check and will emit a call named
after whatever follows the tag.

The read is bounded to `Region.head` and stops once that prefix has gone by
without a name. Running the format's regex over the whole region on every
chunk is quadratic in the response — 3.0 → 9.8 → 36 → 137 ms across
2k/4k/8k/16k tokens, the shape `marker_scanner` exists to retire, one layer up.

Kimi-K2 and Kimi-K3 do not name a call whose *arguments are still arriving*: a
K2 entry is invisible until `<|tool_call_end|>` and a K3 call until
`<|close|>call`, so on a large payload the name arrives with the arguments.
A call short enough to fit inside `Region.head` is named early by all six,
because the whole call is in the window and `parse_region` sees a finished one.
Nothing declares which formats are which -- the property suite measures both
facts independently (can the parse read a call in progress; did the name land
before the arguments on an 800-byte payload) and asserts they agree. A class
attribute used to stand in for this, outlived its only reader when the
give-up probe below was reverted, and took these two paragraphs false with it.

Arguments still wait for the region to close. SGLang streams those too, as
JSON fragments; a response cut short then leaves the client holding an
unterminated object. On `/v1/chat/completions` a name with no arguments is
harmless — clients accumulate by index and wait for `finish_reason`, which
keys on the arguments. On `/v1/messages` it is not: a `content_block_start` of
type `tool_use` carries `"input": {}` and is, on its own, a complete
zero-argument call, with no frame for "the name is known, the arguments are
coming". So that endpoint opens the block when the arguments arrive, not when
the name does.

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

**`stream=false` and `stream=true` deliver the same text**, and not because a
test compares them: `stream=false` is `read_whole`, which is the streaming
engine over a single chunk. There is no second implementation to disagree
with. A format used to be read twice — once by a `parse` taking the whole
output, once by a `process`/`flush` state machine of its own — and both had to
decide where content ends, whether an unclosed tag is a call, what a region
that parses to nothing means, and which bytes are framing. Six formats, two
copies, four rules; three rounds of review found the copies disagreeing about
all four.

A format now declares only what is particular to it: the literals that must
not be split (`START_MARKERS`), which of those hand the stream over
(`opens_region`, the rest being framing the reader drops), and what one
region's bytes mean (`parse_region`, returning the calls and the two offsets
that bracket its own markup). Everything else — reading ahead, releasing
content, the rule that a start marker is not a promise, stamping call indices,
and handing back the answer that follows the markup — is the engine's, once.

The content comes back byte-for-byte except for markers the format declares.
Whitespace is not one — every format used to `.strip()`, which cost a
code-block answer its trailing newline on one path only. Text *after* a call
is not one either: five of the six deleted it, and the property suite now
holds every registered format to delivering it, at four chunk sizes.

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

**Which reasoning dialect a model speaks is decided at startup too**, from the
same evidence as the tool-call format and by the same kind of function
(`resolve_dialect`, on the chat-template source). It used to be decided twice
per response and differently each time: the non-streaming split tried each
registered dialect in order and took the first that matched, while the
streaming filter carried no dialect at all and closed the channel on the
*union* of every dialect's end markers. A `<think>` model answering a question
about Kimi's wire format therefore ended its chain of thought at the quoted
`<|open|>response<|sep|>` when streamed and at the real `</think>` when not —
24 characters of the answer filed as reasoning on one path, a raw `</think>`
shipped to the user on the other. `ReasoningChannel` now carries the dialect
and whether the output begins inside the channel, with one accessor per
delivery mode, so the two cannot be handed different answers. A template that
names no dialect falls back to the inline-`<think>` one, which is a no-op for
a model that never writes the tag.

Whether the output *begins* inside the channel is per request, not per model.
A request that switches reasoning off renders a prompt that does not open it,
and the model-level fact — a template that closes a block it never opens, as
DeepSeek-R1's does — used to be OR-ed in regardless. On such a model an
ordinary answer to a request that had asked for no thinking came back entirely
as `reasoning_content`, with `content` empty.

**`tool_choice: "none"` suppresses the call, not the answer.** It used to be
enforced where the events are *sent* — twelve places across two endpoints —
while the parser went on consuming the region, so the model's own words were
deleted and nothing took their place: 89 characters of a 95-character answer,
no event, `finish_reason: stop`. The rule now lives at the one place the
parser is *asked*, as `suppress_calls`. What that suppresses is dispatch: the
region is read exactly as a permitted call's would be and only the calls are
dropped, so the answer around them survives and the model's raw wire markup
does not reach the client. A reply that was nothing but a forbidden call
therefore has empty `content` — the model produced no answer.

Not by using no parser at all, which is where the first fix for that went.
Dropping the parser drops everything else a parser does, and a format whose
framing wraps *every* answer then leaks it: Kimi-K3's `Hello there.` arrived
as `<|open|>response<|sep|>Hello there.<|close|>response<|sep|><|end_of_msg|>`
the moment a request said `none`. The format is still read; only dispatch is
suppressed. `/v1/messages` reads the field too, in Anthropic's
`{"type": "none"}` spelling; it previously parsed it off the request and used
it nowhere, so a client that forbade tool calls got `tool_use` blocks and
`stop_reason: tool_use` anyway.

Forwarding it to the chat template is a separate step, and one that used to be
a 500. The handler passes template controls it cannot know the model reads —
`response_format`, `tool_choice`, `thinking_effort`, and whatever a client puts
in `chat_template_kwargs`. A Jinja template silently ignores a kwarg it does
not read; a model-shipped Python encoder raises `TypeError`. So on DeepSeek-V4
and Kimi-K3, which ship encoders instead of templates, any request carrying one
of those was an unhandled exception. The adapter now reads the encoder's
signature once at startup and passes on only what it can take.

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

**A stalled response is visible while it is stalled — on the OpenAI server.**
Every SSE frame from `openai_server` leaves through `_client_stream`, which
times the gap before each one and registers it,
and `atom:stream_longest_silence_seconds` reports the age of the oldest gap
in flight. Zero when every stream has just been served; non-zero and growing is
a response whose client is receiving nothing. A gap longer than 30 seconds also
logs a line naming the request — the gauge cannot see a stall that has already
recovered by scrape time. Neither costs a timer: `asyncio.wait_for` measured
1.38 us per frame per stream against 0.07 us for a timestamp and a dict entry.
This exists because the symptom that started this work was ten minutes of
silence with every metric looking healthy.

The atomesh standalone entrypoint has none of it. Its frames leave through
`ChatCompletionStreamState.drain` / `CompletionStreamState.drain`, polled by
the Rust router, which builds no `FrameWait`; and `AtomMetricsExporter` is
constructed only by `openai_server`, so that deployment exposes no `/metrics`
route at all. A stalled atomesh stream is therefore invisible rather than
reported as zero.

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
| `--tool-call-parser` | `auto` | Tool-call wire format. `auto` reads it from the model's chat template at startup (Jinja, or a model-side `encoding/encoding_*.py`); a name — `dsml`, `glm`, `kimi`, `kimi_k3`, `minimax`, `qwen` — overrides. When neither resolves, tool calls are delivered as plain text and the startup log says so; the format is never guessed from output. On the OpenAI server an unknown name is refused at startup, before the weights load, rather than silently disabling tool parsing. The atomesh entrypoint deliberately does not refuse: it shares the flag with the mesh router, which declares its own vocabulary for it, so a name ATOM does not recognise is logged at INFO, forwarded to the router, and ATOM falls back to reading the chat template. Check the log for `is not one of ATOM's formats` if a format you specified is not taking effect |

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
| `--enable-log-stats` / `--no-enable-log-stats` | `True` | Emit the periodic engine-status line (throughput, running/waiting reqs, KV usage, prefix-cache hit rate) |
| `--throughput-log-interval` | `10.0` | Seconds between engine-status lines |
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
| `atom/entrypoints/openai/reasoning.py` | Splits the reasoning channel from the answer; `ReasoningChannel` carries the model's dialect and whether the output begins inside the channel, and has one accessor per delivery mode |
| `atom/entrypoints/openai/reasoning_dialects.py` | The dialects, and `resolve_dialect`, which picks one from the chat template at startup |
| `atom/entrypoints/openai/kimi_k3_tokens.py` | Kimi-K3's channel tokens, split by owner: what the reasoning stage strips and what only the tool parser may |
| `atom/entrypoints/openai/tool_parser/stream.py` | The one reader: the engine both delivery modes run through |
| `atom/entrypoints/openai/chat_encoders.py` | Renders the chat template, and the two startup probes of it: `render_probe_prompt` (what the prompt tells the model) and `chat_template_source` (what the template does with a reply) |
| `atom/entrypoints/openai/tool_parser/registry.py` | Which format a model emits, resolved once at startup from its chat template |
| `atom/entrypoints/openai/tool_parser/` | Per-format tool-call syntax; each format declares its markers and a `parse_region`, and writes no reader of its own |
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
