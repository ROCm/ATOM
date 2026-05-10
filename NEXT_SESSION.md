# Next-session pickup notes — ATOM gfx1201 / Ministral-3

## Headline

End-to-end **prefill** of Mistral-3-8B works on gfx1201 (RX 9070 XT) using the
torch-native fallback path. Engine boots, warmup completes, and a real prompt
gets through all 34 transformer layers + sampler before hitting the explicit
`NotImplementedError` at `TorchNativeMetadataBuilder.prepare_decode`.

The only remaining piece for a working `simple_inference` / `openai_server`
greedy generation is **decode + paged KV-cache write**. That's the focus of
the next session.

## Reproduce furthest-progress run

```bash
ssh -i /home/carhuang/id_rsa_carhuang carhuang@agent-tr9980x-01
docker exec -it atom_gfx1201 bash -lc '
  cd /tmp && \
  ATOM_USE_TRITON_GEMM=1 AITER_LOG_LEVEL=WARNING \
  AITER_ROPE_NATIVE_BACKEND=1 \
  ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT=0 \
  ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT=0 \
  ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0 \
  python3 -m atom.examples.simple_inference \
    --model /mnt/sda1/carhuang/models/Ministral-3-8B-Instruct-2512 \
    --enforce-eager --level 0 -tp 1 --kv_cache_dtype bf16 \
    --max-model-len 256 --max-tokens 4 \
    --gpu-memory-utilization 0.85'
```

Expected progression: `Model load done` → `warmup_model done` → engine ready
→ first real prompt prefill completes → `NotImplementedError:
TorchNativeMetadataBuilder.prepare_decode is a TODO`.

## Branch state (`carhuang/support_gfx1201_mistral3`, local-only on remote)

| commit | what works |
|---|---|
| `93e6013` | Mistral3 model + loader fixes — model loads cleanly |
| `4f848a9` | Backend scaffold + selector wiring |
| `c983d98` | Torch-native attention impl (prefill) + RMSNorm fallback |
| `e2a0e1b` | FP8 GEMM + SiLU+Mul + sampler fallbacks; KV-budget unblocked |

## What works on gfx1201 today (and via what)

| op | replaced by | file |
|---|---|---|
| RMSNorm (with/without residual) | torch RMSNorm | `atom/model_ops/layernorm.py` |
| Per-tensor FP8 linear (qkv_proj, o_proj, gate_up_proj, down_proj) | dequant + `F.linear` | `atom/model_ops/linear.py` |
| YaRN-scaled RoPE | `forward_native` via `AITER_ROPE_NATIVE_BACKEND=1` | env var (no patch) |
| SiluAndMul (SwiGLU) | existing `forward_native` | `atom/model_ops/activation.py` |
| Mixed Gumbel sampler | torch Gumbel-max + argmax | `atom/model_ops/sampler.py` |
| Attention prefill | per-seq SDPA loop using cu_seqlens | `atom/model_ops/attentions/torch_native_attn.py` |
| `compute_block_bytes` (KV budget) | rough placeholder so engine boots | same file |

## Decode work — the actual remaining piece

The two TODOs still raising in `torch_native_attn.py`:

1. **`TorchNativeMetadataBuilder.prepare_decode(batch, bs)`** — must build
   `AttentionMetaData` with at minimum: `slot_mapping`, `context_lens`,
   `block_tables`, `cu_seqlens_q` (decode is `cu_seqlens_q[i+1]-cu_seqlens_q[i]==1`),
   `max_seqlen_q=1`, `max_seqlen_k=max(context_lens)`. Reference implementation:
   `atom/model_ops/attentions/aiter_attention.py:prepare_decode` (lines ~529-620).
   Strip aiter-specific fields (`kv_indptr`, `kv_indices`, persistent worker
   buffers); we won't need them. Returns `(attn_metadata, positions_tensor)`.

2. **`TorchNativeAttentionImpl.forward` decode path** (and KV-cache write).
   Today the prefill path is the whole forward. Add a branch on
   `is_prefill==False`:
   - Read current K/V from the new q/k/v inputs.
   - Write them into the paged KV pool at `slot_mapping`. The KV pool is
     stored on the parent `PagedAttention` instance as `self.kv_cache`
     (or `module.k_cache`/`module.v_cache` after `build_kv_cache_tensor`).
     **Currently we don't allocate a KV pool** — so before this works we
     also need to override `allocate_kv_cache_tensors` /
     `build_kv_cache_tensor` to actually create the tensors.
   - Gather historical K/V from the pool using `block_tables` + the
     new `slot_mapping`, then SDPA: query is [bs, num_heads, 1, d],
     keys are [bs, num_heads, ctx_len, d], no causal mask needed for
     decode (length-1 query).

A minimal-correctness shortcut to consider:
**stateless decode** — recompute the full prefill on every step using the
growing input ids, never store a KV cache. Wildly inefficient (O(N²) per
token) but correct, and avoids the entire KV-cache machinery for a first
greedy/gsm8k run. Could be the fastest path to lm_eval results.

## Validation milestones, in order

1. `simple_inference` greedy generation completes at least one real
   sentence. Print the output and eyeball that it's English.
2. Spin up `openai_server` and curl `/v1/chat/completions` with a tiny
   prompt; check the response is sane.
3. `lm_eval --model local-completions --base_url http://localhost:30000/v1/completions --tasks gsm8k --num_fewshot 5 --apply_chat_template`
   — first real accuracy number.

## Required env vars (record verbatim, keep in repo recipes/Ministral-3-8B.md when committing)

```
ATOM_USE_TRITON_GEMM=1
AITER_LOG_LEVEL=WARNING
AITER_ROPE_NATIVE_BACKEND=1
ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT=0
ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT=0
ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
```

## Caveats / known issues to revisit

* 238 `activation_scale` checkpoint tensors get silently dropped during
  load. Currently fine because our FP8 linear fallback ignores
  `input_scale`. Once we want fully native FP8 (no dequant) we'll need to
  fix the loader to merge q/k/v static scales into `qkv_proj.input_scale`.
* `--enforce-eager --level 0` are still required. CUDAGraph capture will
  break the dispatch-by-arch checks; revisit only after decode works.
* `--kv_cache_dtype bf16` only. FP8 KV is gated on real KV cache + a
  quant/dequant step we don't have.
* The KV-cache "allocation mismatch" warning at boot is the placeholder
  `compute_block_bytes` lying about the pool size. Harmless until decode
  needs it.
