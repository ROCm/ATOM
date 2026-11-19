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

- `<N>` is the context length you will declare to Claude Code in Step 2. Claude Code's system prompt and tool definitions are large (about 14K tokens per request with GLM-5.2's tokenizer), so use a model with a long context; check `usage.input_tokens` on the first turn to see how much of `<N>` the fixed prompt already takes.
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
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=$((<N> - 16384))
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1

claude
```

What each setting is for:

- `ANTHROPIC_BASE_URL` has no `/v1` suffix; Claude Code appends `/v1/messages` itself.
- `ANTHROPIC_AUTH_TOKEN` can be any value. Set it anyway: without a credential, Claude Code sends your claude.ai login token to the server at `ANTHROPIC_BASE_URL`.
- ATOM serves one model and returns 400 for any other model name. The `ANTHROPIC_DEFAULT_*_MODEL` variables make the `opus` / `sonnet` / `haiku` aliases, subagents that name an alias, and background tasks resolve to the served model.
- **Context window.** Claude Code does not recognize the served model ID, so it assumes a 1M-token window and a 32000-token `max_tokens`, and compacts the conversation only as it approaches that window. ATOM cannot hold that: once prompt plus `max_tokens` exceeds `--max-model-len`, it either rejects the request with an error Claude Code does not recognize as "prompt too long", or silently cuts the end of the prompt, which drops the latest turn. Declaring `CLAUDE_CODE_MAX_CONTEXT_TOKENS` as `<N>` minus the output budget makes Claude Code compact before either happens. This only works when the served model ID does not start with `claude-`; use `--served-model-name` if it does.
- `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` stops Claude Code sending `role: "system"` entries in the middle of `messages` and other pre-release fields. ATOM accepts those entries without error but does not render their block content correctly, so the model would see garbled reminders.

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

- ATOM does not serve `/v1/messages/count_tokens`. Claude Code falls back to a character-based estimate, so `/context` shows approximate counts.
- Claude Code prints `[claude-code:unrecognized_model]` for the served model ID. This is a diagnostic, not an error.

## Troubleshooting

### `Requested model 'X' does not match server model 'Y'`

One of the model variables in Step 2 is unset or differs from the ID in `curl http://localhost:8000/v1/models`. Use `--served-model-name` on the server if you want a shorter ID.

### Connection refused

Make sure ATOM is running and listening on the expected host and port: `curl http://localhost:8000/health`.
