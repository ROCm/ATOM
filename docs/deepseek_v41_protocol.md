# DeepSeek-V4.1 chat, tools and reasoning

The native V4.1 text protocol runs on the Python OpenAI-compatible entrypoints
and the Rust mesh tool parser. ATOM loads the checkpoint's standalone
`encoding/encoding.py` when `config.json` declares `model_type="deepseek_v41"`;
keep that directory alongside the tokenizer and weights. An ambiguous encoder
directory is rejected by the existing discovery policy.

Prompt construction uses the official encoder from revision
dba1be0a40aa45a94ad051997016db3960a90277, SHA256
502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1.
ATOM adapts request fields; it does not carry a second prompt implementation.

## Request controls

The chat endpoint accepts reasoning_effort as an exact integer from 1 to 100,
or "low" (50), "high" (75), "max" (100). The official default is 75. The
prefix occurs only at the start of a conversation in thinking mode.

For example, send this JSON to /v1/chat/completions:

```json
{
  "messages": [{"role": "user", "content": "What is 2 + 2?"}],
  "thinking": {"type": "enabled"},
  "reasoning_effort": 37,
  "temperature": 0,
  "max_tokens": 128
}
```

Explicit thinking controls override server defaults and chat_template_kwargs.
An effort alone does not enable thinking over an existing off setting.
thinking.effort takes precedence over the top-level effort; reasoning_effort
set to "none" disables thinking. The adapter translates the common
thinking_effort control to the official encoder's reasoning_effort argument.
Explicit API effort overrides a native effort supplied in template defaults.

The encoder's thinking_mode="chat"/"thinking", context and drop_thinking
controls remain available through chat_template_kwargs. Text content blocks
are preserved until the official encoder joins them; Jinja models retain their
existing text conversion.

Earlier assistant reasoning is dropped according to the official rules.
Tool conversations preserve it. Mid-conversation system messages retain their
System token and assistant transition. Top-level tools attach to the first
system message, or a synthetic one when no initial system turn exists. Tool
results are merged and sorted by the preceding assistant's tool_call IDs.

## Tool calls and streaming

The V4.1 DSML dialect uses leading spaces in calls, invoke and parameter tags.
It is detected once from the rendered tools prompt, and can be selected
explicitly with --tool-call-parser dsml_v41. The original dsml selection keeps
V4's tag spelling.

The Rust mesh registry also exposes dsml_v41. Specific DeepSeek-V4.1 model
mappings precede the V4 family mapping. Both dialects share DsmlParser's
streaming state machine; only markers differ. Typed JSON parameters wait for
their closing parameter tag, and partial string values preserve whitespace.
Python and Rust consume the same fixtures for qualified names, nested JSON,
null/boolean values, Unicode, multiple calls and arbitrary valid UTF-8 splits.

```xml
<｜DSML｜ calls>
<｜DSML｜ invoke name="math::add">
<｜DSML｜ parameter name="a" string="false">17</｜DSML｜ parameter>
<｜DSML｜ parameter name="b" string="false">25</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>
```

Both delivery modes use the existing streaming engine. The V4.1 class declares
its tag spellings and reuses V4's parameter coercion, truncation recovery and
markup spans. There is no second state machine or replacement of tags in
generated text.

A tool schema may supply namespace beside function, inside function, or as a
qualified function name. ATOM uses namespace::name as the OpenAI function
identity in responses and schema lookup; callers can send that name back
unchanged. This is equivalent to the reference encoder's separate namespace
field. Exactly one namespace is supported, and conflicting namespaces or
duplicate qualified names are rejected.

Reasoning is delivered in reasoning_content, answer text in content, and calls
in tool_calls. The existing tool_choice="none" suppression is preserved.
Named/required tool choices and response_format do not add constrained
decoding in this phase.

## Validation

- The Rust mesh tool-parser regression and the Python streaming/complete parser
  checks consume the **same fixture file**, at every UTF-8 boundary and
  character stream, so the two implementations cannot drift apart silently.
- Differential tests check official prompts, numeric budgets, old reasoning,
  mid-conversation system updates, structured text and reversed tool results.
  Typed namespace calls match the official non-streaming parser at every
  possible two-chunk boundary.
- The actual chat handler's argument merge is tested, including defaults,
  explicit on/off, numeric effort, and HTTP 400 before generation for invalid
  numeric budgets and invalid native effort controls.
- A real TP4 ModelRunner/Scheduler run with packed KV/index storage and
  PIECEWISE graphs: Chinese chat returned `中国的首都是北京。`; thinking returned
  reasoning plus the final answer `4`; the model called `math::add` with
  `a=17, b=25` and answered `17 + 25 = 42。` after the tool result. All ranks
  generated identical tokens, and character-by-character reasoning/tool parsing
  matched whole-output parsing for all four replies.

That TP4 run goes through ModelRunner/Scheduler, not a listening HTTP server;
the HTTP route's control handling is tested against a synthetic generation
source instead. These are protocol acceptance checks, not a tool-use benchmark
or a general-quality score.

Reproduce the CPU and reference checks:

```bash
ATOM_DSV41_MODEL=/mnt/DeepSeek-V4.1-Flash \
pytest -q -rs tests/entrypoints
```

## CPU overhead and boundaries

Seven-repeat median measurements on this host, excluding tokenizer work:

| Workload | Official encoder | ATOM adapter |
|---|---:|---:|
| Single user turn | 4.34 us | 5.28 us |
| 32-turn text history | 191.85 us | 194.58 us |
| 64 declared tools | 467.51 us | 465.94 us |

The difference in the tools row is measurement noise. The adapter adds about
1–3 us in these cases. With 64-character stream packets, V4/V4.1 DSML took
65.54/65.76 us for a 64-character argument and 6.23/6.15 ms for a 128 KiB
argument. These are CPU microbenchmarks, not end-to-end latency improvements.

Encoder discovery belongs to `chat_encoders.py`, request translation to
`deepseek_v41_encoder.py`, and DSML syntax to `deepseekv41_tool_parser.py`; the
common stream engine owns buffering and dispatch. The protocol changes no
model, attention, MoE, scheduler or graph kernel, and it does not by itself
enable image requests, constrained decoding, speculative decoding or new
distributed configurations. Image execution and multimodal embedding lifetime
are covered in [image requests](deepseek_v41_vision.md).
