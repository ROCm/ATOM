# Kimi-K3 KDA prefill fusion — deferred end-to-end validation

The kernel work is complete and gated: 10/10 bitwise GPU parity from a cold
Triton cache, whole-branch review with zero Critical findings.

Step 1 (serving bring-up) has since been **run and passed** on this box under
the vLLM plugin — engine init 34 s, VRAM 94% on all 8 TP ranks, coherent
greedy generation, both vendored Triton kernels autotuned live in 8-way TP.
Steps 2-3 remain open.

An earlier revision of this file claimed the Kimi-K3 checkpoint was absent
from the development box. That was wrong: it is at
`/workspace/shared/data/amd_int/models/Kimi-K3`, and the steps below run
here.

Nothing below is optional. Bitwise parity certifies that the fused kernels
compute what the reference computed; it does not certify the end-to-end
accuracy or the performance win that motivated the branch.

## Step 1 — Serving bring-up

Per CLAUDE.md, in order:

```bash
# Stale compile cache causes silent failures after kernel changes.
rm -rf /root/.cache/atom/*
export AITER_LOG_LEVEL=WARNING

python -m atom.entrypoints.openai_server \
  --model Kimi-K3 \
  --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.93 \
  --block-size 128 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'
```

**Do not confirm with `curl /health`** — it returns OK even when the model
never loaded. Confirm with:

```bash
rocm-smi --showmemuse   # pass: VRAM% > 0 on all 8 TP ranks
```

Then check a generation is coherent:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Kimi-K3", "prompt": "What is 2+2?", "max_tokens": 32}'
```

On any server or GPU error, run `/debug-guide` first — do not blindly retry.

## Step 2 — Profile: confirm the launches actually went away

Capture a trace of a prefill-heavy request and inspect the KDA region. All
seven items from the spec's inventory should be absent:

| # | Item | Removed by |
|---|---|---|
| 1 | `gather_kda_state_kernel` | vendored indexed h0 gather + call site |
| 2 | the `beta.float()` cast copy | call site |
| 3 | redundant fp32 compute (subsumed by 2) | call site |
| 4 | `@input_guard` contiguous copies | vendored entry |
| 5 | the output `zeros_like` | `o=` out-param + call site |
| 6 | `index_copy_` into `ssm_state` | inplace ht scatter + call site |
| 7 | the `out.copy_` d2d | `o=` out-param + call site |

**Pass:** the KDA prefill region drops from nine launches plus copies to seven
launches, with no d2d copies between the conv and `o_norm`.

If any item is still present, report which one and what is still calling it
rather than adjusting expectations.

## Step 3 — Accuracy (gsm8k)

```bash
lm_eval \
  --model local-completions \
  --model_args "model=Kimi-K3,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --seed 42
```

Threshold from `recipes/Kimi-K3.md` — both filters:

| Filter | Minimum |
|---|---:|
| flexible-extract | 0.9538 |
| strict-match | 0.9538 |

**If gsm8k regresses, the first hypothesis is the dropped `beta.float()` in
`_run_kda`.** Restore it and re-measure before investigating anything else.

The reasoning for dropping it: fla allocates the sigmoid output fp32 regardless
of input dtype (`fla/ops/common/gate.py:59`), so the widening was a redundant
d2d copy — the gsm8k regression that originally motivated it was a bf16 sigmoid
*output*, which cannot occur on this path. The `.contiguous()` that replaced it
is load-bearing for a different reason (`beta` is a non-contiguous column slice)
and must stay. See the comment at the `chunk_kda` call in `atom/models/kimi_k3.py`.
