# Using Hermes Agent with ATOM

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is an open-source AI agent with a built-in learning loop, multi-platform messaging support, and tool execution. This guide shows how to run Hermes Agent locally using ATOM as the inference backend.

## Prerequisites

- ATOM server running (see [Quickstart](quickstart.rst))
- Python 3.10+
- `pip` or `uv` package manager

## Step 1: Start ATOM

```bash
python -m atom.entrypoints.openai_server \
  --model <your-model-path> \
  --host 0.0.0.0 \
  --server-port 8000 \
  --tensor-parallel-size 8 \
  --trust-remote-code
```

Verify the server is ready:

```bash
curl http://localhost:8000/v1/models
```

Note the model ID from the response — you will use it in Step 3.

## Step 2: Install Hermes Agent

```bash
# Clone the repo
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create a virtual environment and install
uv venv ~/.hermes-venv
source ~/.hermes-venv/bin/activate
uv pip install -e ".[all]"
```

## Step 3: Configure Hermes to use ATOM

Create (or edit) `~/.hermes/config.yaml`:

```yaml
model:
  default: "<model-id-from-step-1>"
  provider: "atom"
  base_url: "http://localhost:8000/v1"

custom_providers:
  - name: ATOM
    base_url: "http://localhost:8000/v1"
    model: "<model-id-from-step-1>"
```

For example, if ATOM serves `Qwen/Qwen3-32B`:

```yaml
model:
  default: "Qwen/Qwen3-32B"
  provider: "atom"
  base_url: "http://localhost:8000/v1"

custom_providers:
  - name: ATOM
    base_url: "http://localhost:8000/v1"
    model: "Qwen/Qwen3-32B"
```

## Step 4: Run Hermes

```bash
source ~/.hermes-venv/bin/activate
cd hermes-agent
hermes
```

You should see the Hermes Agent TUI with your ATOM model name in the header.

## Alternative: Environment Variables

Instead of `config.yaml`, you can configure via environment variables in a `.env` file in the `hermes-agent` directory:

```bash
HERMES_INFERENCE_PROVIDER=atom
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy
```

## Alternative: CLI Flags

```bash
hermes --provider custom --model <model-id>
```

## How It Works

Hermes Agent treats ATOM as any OpenAI-compatible endpoint via the `custom` provider (same as vLLM, LM Studio, or llama.cpp). All inference requests go through the standard `/v1/chat/completions` API. Features supported:

- **Streaming** — real-time token output
- **Tool calling** — Hermes sends tool schemas, ATOM returns structured tool calls
- **Reasoning** — thinking models (e.g., Kimi-K2) return `reasoning_content` alongside the response
- **Multi-turn conversations** — full chat history maintained by Hermes
- **Context compression** — Hermes auto-compresses long conversations when approaching the context limit

## Troubleshooting

### Model name mismatch

If you get `Requested model X does not match server model Y`, make sure `model.default` in your config matches the exact model ID returned by `curl http://localhost:8000/v1/models`.

### Connection refused

Make sure ATOM is running and listening on the expected host/port. Check with `curl http://localhost:8000/health`.

### Empty responses

Thinking models (e.g., Kimi-K2-Thinking) use tokens for reasoning before generating the final answer. Increase `max_tokens` in your config if responses appear empty:

```yaml
model:
  max_tokens: 8192
```
