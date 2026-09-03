# Qwen3.8 Usage Guide

[Qwen3.8](https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4) reuses the Qwen3.5 hybrid attention stack — one full-attention layer every four layers (`full_attention_interval=4`), Gated-Delta-Net (GDN) linear attention in between — and ships in two shapes that ATOM's native backend covers:

| Checkpoint | HF architecture | `model_type` | Layers | Experts | Weights |
|---|---|---|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `Qwen3_5ForConditionalGeneration` | `qwen3_5` | 64 | — (dense) | BF16 (128x128 block FP8 also supported) |
| [amd/Qwen3.8-2.4T-A95B-Quark-MXFP4](https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4) | `Qwen3_5MoeForCausalLM` | `qwen3_5_moe_text` | 92 | 512 routed, top-10 | Quark MXFP4 (per-group fp4, group size 32, e8m0 scales) |

Both checkpoints declare 262144 max positions; the commands below cap context at 32768, which is what the nightly accuracy runs use.

For the vLLM plugin backend, see [Qwen3.8 with ATOM vLLM Plugin Backend](atom_vllm/Qwen3.8.md).

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching Qwen3.8 with ATOM

Both configurations share the same environment variables:

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0
```

### Qwen3.8-27B (TP1, FP8 KV cache)

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-27B \
  --kv_cache_dtype fp8 -tp 1 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable_prefix_caching
```

### Qwen3.8-2.4T-A95B-Quark-MXFP4 (TP8, BF16 KV cache)

```bash
python -m atom.entrypoints.openai_server \
  --model amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --kv_cache_dtype bf16 -tp 8 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable_prefix_caching
```

## Thinking controls

ATOM's OpenAI layer normalizes thinking controls to `thinking` and `thinking_effort`. Qwen3.8's Hugging Face chat template reads `enable_thinking` and `reasoning_effort` instead, so
[`apply_chat_template`](../atom/entrypoints/openai/chat_encoders.py) mirrors the ATOM names onto the template names before rendering:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-27B",
  "messages": [{"role": "user", "content": "Janet has 3 apples and buys 4 more. How many?"}],
  "thinking": true,
  "thinking_effort": "low",
  "max_tokens": 8192
}'
```

Without the mirror the request silently falls back to the template's default effort, which produces much longer reasoning traces and different accuracy — verify with the rendered prompt if a run looks off:

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.8-27B')
print(tok.apply_chat_template([{'role':'user','content':'hi'}],
      tokenize=False, add_generation_prompt=True, reasoning_effort='low')[-400:])
"
```

## `min_tokens`

The sampling recipe Qwen recommends for Qwen3.8 (`temperature=1.0, top_p=0.95, top_k=20`) can sample EOS at the very first decode step, which returns an empty completion. `min_tokens` (accepted on both `/v1/chat/completions` and `/v1/completions`, and on `SamplingParams`) blocks that:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-27B",
  "messages": [{"role": "user", "content": "2+2?"}],
  "min_tokens": 1,
  "max_tokens": 8192
}'
```

The floor is enforced at every layer that can end a request:

- **At the sampler**, [`apply_min_tokens_mask`](../atom/model_ops/sampler.py) masks EOS, the server's configured `stop_token_ids`, and request-level `stop_token_ids` to `-inf` before sampling while the request is below its floor. Masking logits (rather than dropping a sampled EOS) keeps the sampled distribution intact.
- **At the scheduler**, `Scheduler.postprocess` checks the completion length at the exact stop-token position. This matters when speculative decoding returns several tokens in one step: a later accepted draft cannot make an earlier EOS valid retroactively.
- **At the frontend**, text-level stop strings are not checked until the request has passed its floor. Stop strings cannot be enforced in the token-only scheduler because they may cross token boundaries.

`max_tokens` remains outside the floor guard: the ceiling always wins, and `min_tokens > max_tokens` is rejected up front.

Further semantics:

- `0 <= min_tokens <= max_tokens` is validated in `SamplingParams`.
- Speculative decoding is supported. [`apply_min_tokens_mask_with_spec_decode`](../atom/model_ops/sampler.py) masks only the leading draft rows a request can still reach below its floor — a draft position reachable only after the tokens ahead of it in the same forward is no longer below it. The bonus row is scored at `num_draft_tokens` past the current length, which is exactly where it lands on the only path that keeps it (all drafts accepted). ATOM's rejection sampler is argmax-based, so a masked terminal token simply stops being `target_argmax`: a draft that proposed EOS too early is rejected and the sequence continues from the target's own token.
- Batches where every request has `min_tokens=0` skip the mask, the per-step stop-set construction and the scheduler guard entirely, so the default path costs nothing.

## Stopping

`min_tokens` is one end of a stop story that follows vLLM's split, so it is worth naming the whole of it. Two kinds of stop condition, decided in two places, because only one of them can be decided without a tokenizer:

| Condition | Decided by | Notes |
|---|---|---|
| EOS | scheduler | silenced by `ignore_eos` |
| `config.stop_token_ids` | scheduler | the model's other end-of-turn ids; `ignore_eos` silences these too, since they are EOS by another name |
| `stop_token_ids` (per request) | scheduler | the client named them, so they fire even under `ignore_eos` |
| `max_tokens` | scheduler | always wins |
| `stop` / `stop_strings` | **frontend** | matched on detokenized text, then the request is aborted |

Stop strings are matched on text rather than on pre-encoded token ids because a client's spelling of one need not tokenize the way the model emits it — `"five,"` is `[52670, 11]` standing alone but `[4236, 11]` mid-sentence — and a token-level match simply never fires, with nothing reporting it.

- `include_stop_str_in_output` (default `False`) decides whether the matched stop string stays in the returned text. `False` matches OpenAI, vLLM and TGI.
- `finish_reason` reads `stop_sequence`. Which stop string matched is not reported: OpenAI's schema has no field for it, and vLLM likewise keeps it off its OpenAI endpoint.
- The abort is asynchronous, so a few tokens may be generated after the match; they are dropped rather than returned.

## Accuracy

Install `lm-eval` first:
```bash
pip install lm-eval[api]
```

[`scripts/test_qwen38_gsm8k.sh`](../scripts/test_qwen38_gsm8k.sh) launches the server, waits for readiness, runs GSM8K and tears everything down:

```bash
# 27B, TP1, port 8000
HIP_VISIBLE_DEVICES=0 scripts/test_qwen38_gsm8k.sh Qwen/Qwen3.8-27B 1 8000

# 2.4T MXFP4, TP8, BF16 KV cache
KV_CACHE_DTYPE=bf16 scripts/test_qwen38_gsm8k.sh amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 8 8000
```

Useful overrides: `LIMIT=50` for a quick subset, `EVAL_MODE=completion` for the non-chat endpoint, `GEN_KWARGS=...` to change sampling, `KEEP_SERVER_ALIVE=1` to keep the server up. Results, server log and `run_config.txt` land in `outputs/qwen38_gsm8k/<timestamp>/`.

The equivalent manual command:

```bash
lm_eval --model local-chat-completions \
  --model_args model=Qwen/Qwen3.8-27B,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=1200,max_gen_toks=8192,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 \
  --gen_kwargs 'do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_tokens=1,reasoning_effort=low' \
  --apply_chat_template --fewshot_as_multiturn --log_samples
```

Measured on MI355X, full 1319-sample GSM8K, 5-shot:

| Checkpoint | TP | KV cache | flexible-extract | strict-match |
|---|---|---|---|---|
| Qwen3.8-27B | 1 | fp8 | 0.9727 | 0.9712 |
| Qwen3.8-27B (repeat, same recipe) | 1 | fp8 | 0.9689 | 0.9697 |
| Qwen3.8-2.4T-A95B-Quark-MXFP4 | 8 | bf16 | 0.9833 | 0.9826 |

The two 27B rows are the same configuration run twice: `do_sample=True` at `temperature=1.0` gives a run-to-run spread within one stderr (±0.0048), so treat a single run's third decimal as noise.

`reasoning_effort=low` is part of the recipe, not a detail. The same two configurations measured 0.9704 (27B) and 0.9810 (2.4T) when the effort silently fell back to the template default — about 0.0023 lower in both cases, i.e. at the edge of run-to-run noise, but only the pinned-effort recipe is reproducible.

## Nightly coverage

Both configurations are registered in [`.github/benchmark/models_accuracy.json`](../.github/benchmark/models_accuracy.json) at `test_level: nightly`. The gate is `accuracy_threshold` (0.96 for 27B, 0.97 for the MXFP4 MoE); `accuracy_baseline` only feeds the dashboard.
