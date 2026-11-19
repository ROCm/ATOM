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

- `<N>` is the context length Step 2 sizes Claude Code's window from. Claude Code's system prompt and tool definitions alone take about 14K tokens per request (GLM-5.2's tokenizer), so use 128K or more; at 32K, reading six 3K-token files was enough to make compaction thrash. Check `usage.input_tokens` on the first turn to see what the fixed prompt takes with your model.
- Prefix caching is on by default (`--enable_prefix_caching`). Leave it on: Claude Code resends the same system prompt, tool definitions and history on every turn, so a warm turn only prefills the new tokens.
- The tool-call parser is picked from the model's chat template (`--tool-call-parser auto`). Check the startup log: `Tool-call format: <name>` means a parser was found; `No known tool-call format ...` means tool calls would come back as plain text, and Claude Code cannot work, so pass `--tool-call-parser` explicitly.
- ATOM does not authenticate requests. Keep it bound to localhost as above, or put it behind a firewall if you bind `0.0.0.0`.

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
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384      # 4096 when <N> is 32K
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=$(( <N> - CLAUDE_CODE_MAX_OUTPUT_TOKENS ))
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1  # until #3588

claude
```

What each setting is for:

- `ANTHROPIC_BASE_URL` has no `/v1` suffix; Claude Code appends `/v1/messages` itself.
- `ANTHROPIC_AUTH_TOKEN` can be any value. Set it anyway: without a credential, Claude Code sends your claude.ai login token to the server at `ANTHROPIC_BASE_URL`.
- ATOM serves one model and returns 400 for any other model name. The `ANTHROPIC_DEFAULT_*_MODEL` variables make the `opus` / `sonnet` / `haiku` aliases, subagents that name an alias, and background tasks resolve to the served model.
- **Context window.** Claude Code does not recognize the served model ID, so it assumes a 1M-token window and a 32000-token `max_tokens`, and compacts the conversation only as it approaches that window. `CLAUDE_CODE_MAX_CONTEXT_TOKENS` declares the real window so that it compacts in time. This only works when the served model ID does not start with `claude-`; use `--served-model-name` if it does. Size it as above:
  - **Leave room for the output budget.** The fixed prompt plus `CLAUDE_CODE_MAX_OUTPUT_TOKENS` must fit inside the declared window, or Claude Code refuses every request with "Prompt is too long" before sending it.
  - **Requests can still overrun it.** ATOM does not serve `/v1/messages/count_tokens`, so Claude Code estimates token counts from characters, and its compaction request carries the whole conversation. In testing at `<N>` = 32768 (a declared window of 28672), requests reached ATOM at up to 38.7K tokens, and a smaller declared window did not prevent this. Until #3588, ATOM silently cuts the end of such a prompt. That drops the latest turn, or, for a compaction request, the instruction to summarize, and in testing the compaction then failed. With #3588 it returns a 400 that Claude Code recovers from by compacting. A large `<N>` keeps this rare.
- `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` stops Claude Code sending `role: "system"` entries in the middle of `messages` and other pre-release fields. Until #3588, ATOM accepts those entries without error but renders their block content as a Python repr, so the model would see garbled reminders. Drop the variable once #3588 is in your build.

## Checking prompt caching

ATOM ignores `cache_control` markers and caches prompt prefixes automatically. A hit is reported in `usage.cache_read_input_tokens`; `cache_creation_input_tokens` is always `0`.

Send the same request twice. With a cold cache the first reports `0`, and the second reports the cached prefix length. The cache works in whole blocks (16 tokens by default, times the DCP size under decode context parallelism), so the request needs a prompt longer than a block:

```bash
SYSTEM=$(python -c 'print("You are a helpful assistant. " * 100)')
curl -s http://localhost:8000/v1/messages \
  -H 'content-type: application/json' \
  -d "{\"model\": \"<model-id>\", \"max_tokens\": 32, \"system\": \"$SYSTEM\",
       \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}]}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
```

## Notes

- ATOM does not serve `/v1/messages/count_tokens` (the server log shows it as a 404). Claude Code falls back to a character-based estimate, so `/context` shows approximate counts and requests can overrun the declared window.
- Claude Code prints `[claude-code:unrecognized_model]` for the served model ID. This is a diagnostic, not an error.

## Troubleshooting

### `Requested model 'X' does not match server model 'Y'`

One of the model variables in Step 2 is unset or differs from the ID in `curl http://localhost:8000/v1/models`. Use `--served-model-name` on the server if you want a shorter ID.

### Connection refused

Make sure ATOM is running and listening on the expected host and port: `curl http://localhost:8000/health`.
