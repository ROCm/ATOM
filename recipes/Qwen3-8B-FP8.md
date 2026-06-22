# Qwen3-8B-FP8 (block-128) on RX 9070 XT (gfx1201) via ROCm/ATOM

Verified path on RX 9070 XT (gfx1201). Attention and GEMM run through
Triton; same backend setup and the **build-aiter-for-gfx1201** prerequisite
([ROCm/aiter#3846](https://github.com/ROCm/aiter/issues/3846)) as the
[Ministral-3-8B recipe](./Ministral-3-8B.md).

## Model

[`Qwen/Qwen3-8B-FP8`](https://huggingface.co/Qwen/Qwen3-8B-FP8) —
official Qwen release, FineGrainedFP8 quant with
`weight_block_size=[128, 128]`, `activation_scheme="dynamic"`.
36 layers, hidden=4096, head_dim=128, num_q_heads=32, num_kv_heads=8 (GQA),
vocab=151936.

```bash
hf download Qwen/Qwen3-8B-FP8 \
  --local-dir /mnt/sda1/carhuang/models/Qwen3-8B-FP8
```

## Required env

```bash
export ATOM_USE_UNIFIED_ATTN=1   # route through TritonMHABackend (aiter triton unified_attention)
export ATOM_USE_TRITON_GEMM=1
export AITER_ROPE_NATIVE_BACKEND=1
export AITER_LOG_LEVEL=WARNING
export ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT=0
export ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT=0
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
```

**Fused RMSNorm+Quant / SiLU+Quant**: set
`ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT=1` and
`ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT=1` to fuse
normalization/activation with FP8 quantization. Requires HIP
`rmsnorm_quant` to JIT-compile on gfx1201 — test before enabling.

## Required CLI flags

- `--level 0` — torch.compile not supported with this backend
- `--block-size 64` — required with `ATOM_USE_UNIFIED_ATTN=1` + bf16 KV (the engine asserts `block_ratio == 1`; default 16 fails)
- `--kv_cache_dtype bf16` or `--kv_cache_dtype fp8` (FP8 KV halves cache memory)
- `-tp 1` — TP > 1 not exercised

CUDAGraph capture works at all default decode batch sizes.

## OpenAI-compatible server

```bash
python3 -m atom.entrypoints.openai_server \
  --model /mnt/sda1/carhuang/models/Qwen3-8B-FP8 \
  --level 0 --kv_cache_dtype bf16 --block-size 64 \
  --max-model-len 16384 \
  --server-port 30000
```

## gsm8k via lm_eval (5-shot, generate-until)

```bash
OPENAI_API_KEY=dummy lm_eval \
  --model local-completions \
  --model_args model=/mnt/sda1/carhuang/models/Qwen3-8B-FP8,base_url=http://localhost:30000/v1/completions,tokenizer=/mnt/sda1/carhuang/models/Qwen3-8B-FP8,tokenized_requests=False,max_length=4096,num_concurrent=4 \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 --limit 50
```

## Verified results on RX 9070 XT (gfx1201, 16 GB), BF16 KV, single stream

| ISL / OSL | Mode | TTFT (ms) | TPOT (ms) | Output tok/s |
|---|---|---:|---:|---:|
| 18 / 80   | cudagraph | 39  | **18.5** | 53.3 |
| 549 / 256 | cudagraph | 86  | **18.6** | **52.9** |
| 549 / 256 | eager     | 93  | 28.6 | 35.6 |

gsm8k 5-shot, n=50:

| Mode | strict | flex |
|---|---:|---:|
| eager     | 0.88 ± 0.05 | 0.88 ± 0.05 |
| cudagraph | **0.86 ± 0.05** | **0.86 ± 0.05** |


## ATOM + Qwen3-8B + Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) can drive this ATOM server as an
OpenAI-compatible backend. The generic steps are in the
[Hermes guide](../docs/hermes_agent_guide.md); below are the **gfx1201 / 16 GB specifics** that the
generic guide does not cover.

> **The one thing that matters on 16 GB:** this card's KV cache caps total context at **~19.8K
> tokens** (util 0.9), but Hermes' *default* toolset builds a **~19.6K-token** system prompt. That
> over-limit request is **silently parked** by ATOM (`"will never be scheduled"`), so the chat just
> hangs. The fix is to **serve the max context the card allows** *and* **restrict Hermes to a small
> toolset** so its prompt stays a few thousand tokens.

### 1. Start the server (context-tuned for Hermes)

Export the [Required env](#required-env) first, then:

```bash
python3 -m atom.entrypoints.openai_server \
  --model /mnt/sda1/carhuang/models/Qwen3-8B-FP8 \
  --level 0 --kv_cache_dtype bf16 --block-size 64 \
  --max-model-len 19456 --gpu-memory-utilization 0.9 \
  -tp 1 --server-port 30001
```

- `--max-model-len 19456` is about the ceiling on 16 GB (KV pool ≈ 1237 blocks × 16 tokens).
- Do **not** raise `--gpu-memory-utilization` to 0.95, and do **not** set
  `PYTORCH_ALLOC_CONF=expandable_segments:True` — both crash this path during CUDA-graph capture
  (HIP "memory access fault" / out-of-memory).

> **Stopping cleanly:** `pkill -f openai_server` leaves the engine-core `multiprocessing.spawn`
> children alive **holding VRAM**. Also run `pkill -f spawn_main` (or just stop the container),
> otherwise the next launch OOMs on a "full" GPU.

### 2. Register ATOM as a Hermes provider

Add a named provider to `~/.hermes/config.yaml` under `providers:`. This leaves your existing
`model.default` untouched, so it is non-destructive:

```yaml
providers:
  atom:
    base_url: http://localhost:30001/v1
    api_key: dummy
    api_mode: chat_completions
    model: /mnt/sda1/carhuang/models/Qwen3-8B-FP8
    models:
      - /mnt/sda1/carhuang/models/Qwen3-8B-FP8
```

> Hermes hardcodes a 64K-token minimum context (`agent/model_metadata.py:MINIMUM_CONTEXT_LENGTH`)
> and, when it can't detect the real window, defaults to 256K — so do **not** set a truthful
> `context_length` (< 64K) here or Hermes refuses to start. Keeping requests under the real limit is
> done by the toolset restriction below, not by the declared context length.

### 3. Quick CLI smoke test

```bash
OPENAI_API_KEY=dummy hermes \
  --provider atom -m /mnt/sda1/carhuang/models/Qwen3-8B-FP8 \
  -t memory,todo -z "hi"
```

`-t memory,todo` restricts the toolset so the prompt is ~2K tokens. A successful run returns a
normal reply (e.g. *"Hello! How can I assist you today?"*).

### 4. Browser chat via the Hermes dashboard

Hermes ships a web dashboard with an in-browser **Chat** tab (`--tui`). Launch it with the ATOM
provider and a small toolset baked into the environment:

```bash
export HERMES_INFERENCE_PROVIDER=atom HERMES_TUI_PROVIDER=atom
export HERMES_MODEL=/mnt/sda1/carhuang/models/Qwen3-8B-FP8
export HERMES_INFERENCE_MODEL=/mnt/sda1/carhuang/models/Qwen3-8B-FP8
export OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:30001/v1
export HERMES_TUI_TOOLSETS="memory,todo,clarify,mcp-off"   # keep the prompt small!
export HERMES_TUI_SKILLS=""

hermes dashboard --host 0.0.0.0 --port 9119 --tui --no-open --insecure
```

Open the **Chat** tab from another machine:

- Same LAN: `http://<host-lan-ip>:9119`
- Anywhere, via SSH tunnel (recommended): `ssh -L 9119:localhost:9119 <user>@<host>` then browse
  `http://localhost:9119`

`--insecure` is required to bind off-localhost. The dashboard exposes API keys and the agent's
shell tools (effectively remote code execution), so only expose it on a trusted network or keep it
behind the SSH tunnel (`--host 127.0.0.1`).

### Why the toolset must stay small

| Hermes toolset | Prompt size | Fits the ~19.4K window? |
|---|---:|---|
| default (~17 toolsets + skills) | ~19,605 tok | ✗ overflows → request hangs |
| `memory,todo,clarify` | ~2,000 tok | ✓ ample room for the conversation |

This 16 GB card serves ~19.8K tokens of context at most, while Hermes is designed for ≥64K. Keep the
toolset minimal; re-enabling heavy tools (browser / terminal / file / web) or letting a conversation
grow long will overflow the window again. For the full Hermes agent experience, use a GPU with
≥64K-token KV capacity.
