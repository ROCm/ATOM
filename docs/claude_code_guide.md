# Using Claude Code with ATOM

[Claude Code](https://code.claude.com/docs) talks to its backend over the Anthropic Messages API. ATOM serves that API at `/v1/messages`, next to the OpenAI-compatible endpoints, so Claude Code can use a model served by ATOM without a proxy. Streaming, tool calls and thinking work over this endpoint, and ATOM's prefix cache serves the prompt prefix Claude Code resends on every turn.

## Prerequisites

- An ATOM server (see [Quickstart](quickstart.rst)) serving a model whose chat template supports tool calls
- Claude Code installed (see the [Claude Code docs](https://code.claude.com/docs))

## Step 1: start ATOM

```bash
python -m atom.entrypoints.openai_server \
  --model <your-model-path> \
  --host 127.0.0.1 \
  --server-port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len <N> \
  --trust-remote-code
```

- `<N>` is the context length Step 2 sizes Claude Code's window from. Claude Code's system prompt and tool definitions alone take about 14K tokens per request (GLM-5.2's tokenizer), so use `131072` or more; at 32768, reading six 3K-token files was enough to make compaction thrash. Check `usage.input_tokens` on the first turn to see what the fixed prompt takes with your model.
- Prefix caching is on by default (`--enable_prefix_caching`). Leave it on: Claude Code resends the same system prompt, tool definitions and history on every turn, so a warm turn only prefills the new tokens.
- The tool-call parser is picked from the model's chat template (`--tool-call-parser auto`). Check the startup log: `Tool-call format: <name>` means a parser was found. `No known tool-call format ...` or `Could not render ...'s chat template with a tools payload ...` means tool calls would come back as plain text, and Claude Code cannot work, so pass `--tool-call-parser` explicitly.
- ATOM does not authenticate requests, and `--host` defaults to `0.0.0.0` (all interfaces). Bind it to localhost as above, or put it behind a firewall.

Find the model ID the server reports:

```bash
curl http://localhost:8000/v1/models
```

## Step 2: point Claude Code at ATOM

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_MODEL=<model-id-from-step-1>
export ANTHROPIC_DEFAULT_OPUS_MODEL=$ANTHROPIC_MODEL
export ANTHROPIC_DEFAULT_SONNET_MODEL=$ANTHROPIC_MODEL
export ANTHROPIC_DEFAULT_HAIKU_MODEL=$ANTHROPIC_MODEL
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384   # 4096 when <N> is 32768
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=$(( <N> - CLAUDE_CODE_MAX_OUTPUT_TOKENS ))

claude
```

What each setting is for:

- `ANTHROPIC_BASE_URL` has no `/v1` suffix; Claude Code appends `/v1/messages` itself.
- `ANTHROPIC_AUTH_TOKEN` can be any value. Set it anyway: without a credential, Claude Code sends your claude.ai login token to the server at `ANTHROPIC_BASE_URL`.
- ATOM serves one model and returns 400 for any other model name. The `ANTHROPIC_DEFAULT_*_MODEL` variables make the `opus` / `sonnet` / `haiku` aliases, subagents that name an alias, and background tasks resolve to the served model.
- **Context window.** Claude Code does not recognize the served model ID, so it assumes a 32000-token `max_tokens` and a 1M-token window (200K before Claude Code v2.1.323), and compacts the conversation only as it approaches that window. `CLAUDE_CODE_MAX_CONTEXT_TOKENS` declares the real window so that it compacts in time. It applies on its own when Claude Code does not recognize the ID and the ID has no `[1m]`. An ID that contains a Claude model name, such as `org/claude-opus-4-8`, resolves to that model and ignores the variable; use `--served-model-name` to avoid one. Size it as above:
  - **Leave room for the output budget.** The fixed prompt plus `CLAUDE_CODE_MAX_OUTPUT_TOKENS` must fit inside the declared window, or Claude Code refuses every request with "Prompt is too long" before sending it.
  - **Requests can still overrun it.** Claude Code's compaction request carries the whole conversation plus the instruction to summarize, and is not held to the declared window. In testing at `<N>` = 32768 (a declared window of 28672), requests reached ATOM at up to 38.7K tokens; a smaller declared window, or exact counts from a `count_tokens` endpoint, did not prevent this. A large `<N>` keeps it rare. ATOM answers such a request with a 400 in one of the two wordings Claude Code acts on: `prompt is too long`, on which it compacts (also the answer when the KV pool cannot hold the prompt), or ``input length and `max_tokens` exceed context limit``, on which it retries with a smaller `max_tokens`.
- **Mid-conversation system messages.** For a model ID it does not recognize, Claude Code sends some reminders as `role: "system"` entries inside `messages`. `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` leaves them in place. A `modelOverrides` entry that maps a Claude model released before Opus 4.8 to the served ID stops them, but Claude Code then treats the ID as that model and ignores `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. ATOM renders them as system messages. If the chat template accepts a system message only first, ATOM returns a 400 `Unexpected role "system" in messages …`, and Claude Code resends the text inside user turns. #3610 (open) folds these messages into user turns for such templates.

## Checking prompt caching

ATOM ignores `cache_control` markers and caches prompt prefixes automatically. A hit is reported in `usage.cache_read_input_tokens`; `cache_creation_input_tokens` is always `0`.

Send the same request twice. With a cold cache the first reports `0`, and the second reports the cached prefix length. The cache works in whole blocks (16 tokens by default, larger for some models such as DeepSeek-V4 at 256, and times the DCP size under decode context parallelism), so the request needs a prompt longer than a block:

```bash
SYSTEM=$(python -c 'print("You are a helpful assistant. " * 100)')
curl -s http://localhost:8000/v1/messages \
  -H 'content-type: application/json' \
  -d "{\"model\": \"<model-id>\", \"max_tokens\": 32, \"system\": \"$SYSTEM\",
       \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}]}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
```

## Notes

- ATOM does not serve `/v1/messages/count_tokens` (the server log shows it as a 404). Claude Code falls back to a character-based estimate, so `/context` shows approximate counts.
- With `claude -p`, Claude Code prints `[claude-code:unrecognized_model]` to stderr for the served model ID (in interactive sessions it goes to the debug log). This is a diagnostic, not an error.

## Troubleshooting

### `Requested model 'X' does not match server model 'Y'`

One of the model variables in Step 2 is unset or differs from the ID in `curl http://localhost:8000/v1/models`. Use `--served-model-name` on the server if you want a shorter ID.

### Connection refused

Make sure ATOM is running and listening on the expected host and port: `curl http://localhost:8000/health`.
