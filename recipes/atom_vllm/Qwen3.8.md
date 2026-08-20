# Qwen3.8 with ATOM vLLM Plugin Backend

This recipe shows how to run Qwen3.8 with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md). For the native ATOM engine, see [Qwen3.8 Usage Guide](../Qwen3.8.md).

Covered checkpoints:

| Checkpoint | HF architecture | Weights | Layout |
|---|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `Qwen3_5ForConditionalGeneration` | BF16 | TP1 |
| [amd/Qwen3.8-2.4T-A95B-Quark-MXFP4](https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4) | `Qwen3_5MoeForCausalLM` | Quark MXFP4, 512 experts top-10 | TP8 + EP8 |

`Qwen3_5MoeForCausalLM` is the **text-only** MoE entry point. ATOM maps it to `Qwen3_5MoeForCausalLMVllm` in [`register.py`](../../atom/plugin/vllm/register.py) / [`model_wrapper.py`](../../atom/plugin/vllm/model_wrapper.py); the wrapper declares `IsHybrid` and supplies the GDN mamba-state dtype, shape and copy functions so vLLM can size the linear-attention state cache. The `*ForConditionalGeneration` architectures remain the multimodal path.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0
```

### Qwen3.8-27B (TP1, MI355X)

```bash
vllm serve Qwen/Qwen3.8-27B \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --reasoning-parser qwen3
```

### Qwen3.8-2.4T-A95B-Quark-MXFP4 (TP8 + EP8, MI355X)

```bash
vllm serve amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype auto \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --reasoning-parser qwen3
```

`--reasoning-parser qwen3` splits the thinking trace out of `message.content` into `reasoning_content`. Leave it off and graders that read `content` will see the full trace.

## Step 3: Performance Benchmark

```bash
ISL=1000
OSL=100
CONC=4

vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model Qwen/Qwen3.8-27B \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --random-range-ratio 0.0 \
    --max-concurrency "${CONC}" \
    --num-prompts "$(( CONC * 8 ))" \
    --trust_remote_code \
    --num-warmups "${CONC}" \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el
```

## Step 4: Accuracy Validation

```bash
lm_eval --model local-chat-completions \
    --model_args model=Qwen/Qwen3.8-27B,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=1200,max_gen_toks=8192,tokenized_requests=False,trust_remote_code=True \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 \
    --gen_kwargs 'do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_tokens=1,reasoning_effort=low' \
    --apply_chat_template --fewshot_as_multiturn --log_samples
```

Two parts of `--gen_kwargs` are load-bearing:

- `reasoning_effort=low` — Qwen3.8's chat template gates thinking depth on it; the default effort produces far longer traces and a different score.
- `min_tokens=1` — at `temperature=1.0` a request can sample EOS on the first decode step and return an empty completion, which GSM8K scores as wrong.

Nightly baselines recorded in [`.github/benchmark/oot_models_accuracy.json`](../../.github/benchmark/oot_models_accuracy.json) (5-shot GSM8K, flexible-extract):

| Configuration | Baseline | Gate (`accuracy_test_threshold`) |
|---|---|---|
| Qwen3.8-27B TP1 | 0.9757 | 0.96 |
| Qwen3.8-2.4T-A95B-Quark-MXFP4 TP8+EP8 | 0.9795 | 0.97 |

Both entries are `nightly` / `P0` on the `atom-plugin-acc-validation-runner-vllm` runner and are selectable by name in the [accuracy validation workflow](../../.github/workflows/atom-vllm-accuracy-validation.yaml).

## Architecture Notes

Qwen3.8 alternates full attention with Gated-Delta-Net linear attention (one full layer every four). The GDN decode path needs request-indexed metadata, which interacts with vLLM's FULL CUDA graph padding:

- vLLM pads a captured decode batch with zero-length rows at the **end** of `query_start_loc_cpu`. [`AtomGDNAttentionMetadataBuilder`](../../atom/plugin/vllm/gdn_backend.py) compacts that padded batch back to the real request prefix after `build()`.
- The compaction slices `[:real_num_decodes]` instead of indexing with a GPU boolean mask. Building a device mask and using it for advanced indexing is not CUDA-graph safe on ROCm; the validation therefore stays on CPU.
- The layout assumptions are asserted rather than assumed. If vLLM ever stops padding at the tail, emits multi-token decode rows, or changes the block-table shape, the builder raises a `RuntimeError` naming the broken assumption instead of silently reading the wrong state slots.

This path is only entered when `use_full_cuda_graph` is set and the batch is pure decode (no prefills, no spec decode).
