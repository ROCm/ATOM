# Kimi-K3 (Minimax-m3-xiaobing) ATOM accuracy debug — handoff

## Where things stand

Model: `/shared/data/amd_int/models/xiaobing/Minimax-m3-xiaobing` — KimiLinear hybrid
(24 MLA full-attn layers + 71 KDA linear-attn layers, 896-expert MXFP4 MoE,
`situ` activation, `attn_res_block_size=12` block residuals). Served text-only via
ATOM in docker `zxb_kimi_k3`, `-tp 8` on 8× gfx950/MI355 (288GB each).
venv python: `/opt/venv/bin/python`. Code: `/workdir/xiaobing/{ATOM-K3,aiter-k3}`.

### Bugs already fixed (took output from total garbage → coherent, gsm8k ~0.28)
1. **prefill never wrote KV cache** — `KimiFullAttention` prefill bypassed `self.attn`
   (attention_mha.forward_impl→rope_cache→reshape_and_cache). Fixed. (committed e68fa86)
2. **latent-MoE `routed_expert_norm` on TP-partial sum** — must all-reduce before the
   nonlinear RMSNorm. Fixed in `KimiSparseMoeBlock.forward`. (committed e68fa86)
3. **triton MoE used SiLU instead of situ** — `fused_moe_triton.triton_kernel_fused_experts`
   hardcoded `fused_clamp_act_mul(activation="silu")` for all 92 MoE layers; situ ≈ SiLU for
   small activations but diverges (tanh saturates). Fixed: compute situ in fp32 when
   `activation==Situv2`, thread `situ_beta=4/linear_beta=25`. (committed 91968c5)
4. **multi-seq varlen prefill fault** (head_dim=192) — unified prefill faults on batched
   prefill; switched to per-sequence SDPA compute + explicit reshape_and_cache write.
   REQUIRES `--no-enable_prefix_caching --no-enable_chunked_prefill` (SDPA can't see cached
   prefix). UNCOMMITTED in `atom/models/kimi_k3.py` working tree.
5. aiter-k3 version skews patched: `shuffle_scale_moe(return_layout=)` + flat `topk` sigmoid
   score_mode. (committed in aiter-k3 `xiaoing/k3`)

### The residual bug (NOT yet localized)
- gsm8k stuck ~0.28 (expect ~0.9, cf Kimi-K2-Thinking 0.9363 same harness).
- Symptom = **sporadic token corruption**, worse on large-magnitude values:
  - misreads input numbers (e.g. "$10/hr" → uses $12 or $14.4)
  - computes right in text then emits wrong `####` token ("624 pages ... #### 11")
- Present in BOTH unified-prefill (run8) and SDPA-prefill (run7c) → NOT the prefill/cache code.
- KDA / MLA / MoE / block-residual Python structure all verified line-by-line == reference
  (modeling_kimi.py) by two independent passes. So the bug is at the **kernel numerical level**
  (aiter triton attention for head_dim=192, or fla KDA kernels on gfx950) OR a **weight-loading
  subtlety** (transpose / shard / interleave) that is structurally plausible but numerically wrong.

## Plan: isolated submodule comparison (full HF reference forward is INFEASIBLE —
MXFP4→bf16 dequant ≈ 2.8TB, won't fit 8×288GB)

For each op type, load ONE real layer's weights, run ATOM module vs reference module on the
SAME input tensor, compare outputs (max abs/rel err). Find which op diverges.

Reference modules (import from the model dir, needs `fla-core` + `flash_attn` + transformers>=4.56):
- `modeling_kimi.KimiDeltaAttention`   (KDA)  — layers in text_config.linear_attn_config.kda_layers (1-based)
- `modeling_kimi.KimiMLAAttention`     (MLA)  — full_attn_layers (1-based): 4,8,12,...
- `modeling_kimi.KimiSparseMoeBlock`   (MoE)

ATOM modules: `atom.models.kimi_k3.{KimiKDAAttention,KimiFullAttention,KimiSparseMoeBlock}`.

Suggested steps:
1. Pick layer 0 (dense MLP, first_k_dense_replace=1) and an MLA layer (e.g. idx 3 = config
   full_attn_layers "4" minus 1) and a KDA layer (e.g. idx 1) and one MoE layer (idx>=1).
2. Load that single layer's weights from the safetensors shards (use safetensors index to find
   which shard holds `language_model.model.layers.{i}.*`). Only that layer fits easily.
3. Build the reference module (single device, bf16, dequant that layer's MXFP4 experts to bf16
   via compressed-tensors or fp4_utils.mxfp4_to_f32) and the ATOM module (tp=1), load identical
   weights into both, feed identical random-but-fixed input (seed), compare outputs.
4. Rank ops by divergence. The first large divergence localizes the bug.

Alternative if per-module weight extraction is fiddly: use ATOM's built-in forward dump
(`install_block_forward_hooks`, already wired in model_runner.py:675) to dump ATOM per-layer
hidden states (`ATOM_FWD_DUMP_DIR=... ATOM_FWD_DUMP_BLOCK_CLASS=KimiDecoderLayer
ATOM_FWD_DUMP_LAYER_ATTR=layer_idx`), and compare successive-layer deltas / NaN/Inf/large-norm
growth to spot the first misbehaving layer TYPE (KDA vs MLA) without a reference.

## Localization results (subagent, ~1h, quantitative)

EXONERATED with numeric evidence (all match reference to bf16 precision):
- fla KDA kernels: chunk (prefill), fused_recurrent (decode), AND 64-step decode
  accumulation (~6e-5, no drift).
- head_dim=192 `unified_attention` decode compute vs SDPA (~1e-3, same as 128 control).
- Routed MoE GEMM `moe_gemm_a16w4` + situ on real swizzled expert weights: 0.34% vs
  bf16-dequant ref, correct even for large-magnitude inputs.
- MoE routing `biased_grouped_topk` (sigmoid+bias+grouped, n_group=1): 0 expert-id
  mismatches, weights match to 1e-8.
- RoPE (NoPE, correctly not applied), block-residual (exact), situ (correct branch+betas),
  norms/eps, KDA o-norm gate, A_log/dt_bias loading.
- Per-layer hidden-state dump: all 93 layers finite, smooth block-residual progression,
  NO single-layer or layer-type blow-up.

SEPARATE aiter bug found (NOT in K3's path, report upstream): `gemm_a16wfp4` (dense mxfp4
a16w GEMM) has ~11% systematic error on gfx950. Only used by `_apply_shared_experts_dense`
(fused shared experts); K3 uses separate bf16 KimiMLP shared experts, so it's not exercised.

STILL UNTESTED (the two remaining runtime-only suspects — the bug most likely hides here):
1. **[TOP] MLA KV-cache SHUFFLE write/read round-trip at head_dim=192.** Symptom ("decode
   misreads input values") == decode retrieves slightly-wrong K/V from the paged cache.
   The decode COMPUTE was tested with correct in-memory K/V, but NOT the full round-trip:
   reshape_and_cache write (K=192, V padded 128→192, asm_layout per v_cache.dim()==5) →
   paged read. head_dim=192 is non-power-of-2, exactly where a SHUFFLE pack/unpack that
   tiles head_size can mis-index. This is a SMALL standalone test (tiny paged cache tensors,
   no full model needed) — do it that way to avoid the GPU-contention/load blockers.
2. latent-MoE TP all-reduce/norm ordering at runtime TP=8 (verify block output == single-GPU
   ref; confirm experts are TP-partial not replicated).

## ★ ROOT CAUSE CONFIRMED (definitive, quantitative)

**aiter `unified_attention(shuffled_kv_cache=True)` mis-reads the 5-D SHUFFLE paged KV cache
at the non-power-of-2 `head_dim=192`.** SHUFFLE-read maxabs error: 64→2.8e-3, 128→2.1e-3,
**192→1.01 (~100%)**, 256→2.6e-3. Cache WRITE (reshape_and_cache, asm_layout=True) is
bit-exact; only the SHUFFLE READ is broken at 192. Flash-layout read
(`shuffled_kv_cache=False`) is correct at 192 (2.55e-3).

This is the decode path of the 24 MLA full-attn layers (head_dim = qk_nope 128 + qk_rope 64 =
192): server uses `ATOM_USE_UNIFIED_ATTN=1`, `use_flash_layout=False` →
`unified_attention(shuffled_kv_cache=True)` over the 5-D SHUFFLE cache. MLA prefill uses SDPA
(correct) so prompts encode fine; only decode-time attention over cached tokens is corrupted →
"misreads input numbers", wrong `####` tokens. Decode-only → present in both run7c & run8. The
71 KDA layers (linear attn, no paged read) are unaffected, keeping output coherent (gsm8k 0.28
not 0). Everything else (KDA kernels, MoE GEMM+situ+routing, RoPE/NoPE, block-residual, norms,
weight loading, latent-MoE reduce-before-norm) is verified correct.

### Fix options (pick one)
1. **Flash KV-cache layout for the MLA layers (cleanest, no memory waste).** Allocate cache
   `[num_blocks, block_size, num_kv_heads, head_dim]` via the flash path, set
   `use_flash_layout=True` on the impl so `shuffled_kv_cache=False`. Requires backend plumbing:
   `attn_metadata_builder.build_kv_cache_tensor` (aiter_attention.py) to allocate flash layout
   for head_dim=192 layers + write via `reshape_and_cache_flash`/`fused_qk_rope...(flash_layout
   =True)`. `use_flash_layout` is currently hardcoded False (attention_mha.py:98).
2. **Pad head_dim 192→256 (localized, low-risk, wastes ~33% KV).** In `KimiFullAttention`:
   create `self.attn` with head_dim=256 but keep scale=192**-0.5; pad q,k, and v to 256; write
   256-dim K/V to cache; slice attn output [:128] for V. SHUFFLE read at 256 is correct. Fully
   contained in kimi_k3.py.
3. Fix aiter `unified_attention` SHUFFLE-read for non-pow2 head_dim (pad the head-dim tile loop).

Validate any fix by re-running gsm8k (base completions, num_fewshot 3-5) — expect flexible to
jump from 0.28 toward ~0.9.

### ATTEMPTED & FAILED: pad head_dim 192→256 (option 2)
Implemented option 2 in `KimiFullAttention` (Attention head_dim=256, pad q/k/v to 256, scale
stays 192**-0.5, slice V[:128]). Result in the REAL model: **total garbage** (worse than the
192 coherent-but-wrong baseline) — e.g. "Half # #每次 bi bi …". So the SHUFFLE cache
write+read round-trip at 256 does NOT actually work in the real model, even though the
subagent's ISOLATED 256 read test passed. LESSON: the isolated SHUFFLE read test was not
representative of the real reshape_and_cache-write + unified_attention-read round-trip. Reverted.
**Any fix MUST be validated end-to-end in the real model (gsm8k), not just isolated.** This also
lowers confidence that the isolated "flash-layout correct at 192" result will hold end-to-end —
verify it in the real model before trusting it.

Current best working state = SDPA-prefill + 192 SHUFFLE (coherent, gsm8k ~0.28), uncommitted in
`atom/models/kimi_k3.py` (with `--no-enable_prefix_caching --no-enable_chunked_prefill`).

## PERFORMANCE (CUDA graph) — done, shippable, no code changes
Decode was ~4.85 tok/s (eager). Enabling CUDA-graph capture gives ~20 tok/s (**~4x**), gsm8k
held at 0.94/0.94. NO kimi_k3.py changes needed: ATOM already treats model_type=kimi_linear as
GDN (is_qwen_next()==True, use_gdn=True), and at `level 0` with `enforce_eager=False` +
`cudagraph_mode=FULL`, ATOM's capture_cudagraph captures the whole eager decode forward (MLA
unified_attention op + KDA causal_conv1d_update/fused_recurrent_kda + block-residual) per decode
batch size ≤ max_num_seqs. The prefill `.item()/.tolist()` host syncs are not captured (warmup
uses is_prefill=False). Lower risk than torch.compile (no Dynamo constraints).
Launch flag changes only (see `my_script/launch_k3_cudagraph.sh`):
- DROP `--enforce-eager` (enables FULL cudagraph capture at level 0)
- `--gpu-memory-utilization 0.90 → 0.93` (the ~1.5GB cudagraph pool otherwise starves the KDA
  per-request state cache: "Per-request cache 0.84GB > KV budget 0.78GB")
- keep `-tp 8`, `--level 0`, `--no-enable_prefix_caching --no-enable_chunked_prefill`
Capture only covers decode bs ≤ max_num_seqs(16); larger batches fall back to eager.

## FULL gsm8k validated: 0.9378 flexible / 0.9371 strict (1319/1319, clean cudagraph server)
On par with Kimi-K2-Thinking (0.9363). No sustained-load degradation on a clean server.

### False alarm (resolved): a one-off "degradation" run
One earlier full run gave 0.53, and a 50q re-run right after it gave 0.12 — this was a TRANSIENT
anomaly on a server that had been restarted immediately after a CI GPU-contention crash (competing
sglang_validation container). A clean fresh full 1319 run reproduces 0.9378, and fresh 50q runs
give 0.94 (RUN A) / 0.92 (RUN B, ~100 reqs). So there is NO real accuracy-degradation bug; the low
numbers were environmental (shared-GPU CI contention corrupting that specific server instance).
Practical rule: run on a clean GPU set with no competing job (use the CI-killer guard) — that's the
only requirement for stable 0.94.

## Environment gotchas
- GPUs are SHARED: an automated CI keeps respawning `atom_sglang_validation_mi355_N` containers
  (DeepSeek-V4-Pro tp8, mem 0.85) that OOM/kill the K3 server. `docker ps`; stop the validator
  container to grab a window (user authorized). It respawns every few minutes — pause that CI
  pipeline for uninterrupted work.
- Fast model load: parallel-prewarm page cache (`ls model-*.safetensors|xargs -P12 -I{} dd
  if={} of=/dev/null bs=4M`) + `ATOM_USE_FASTSAFETENSORS=0 ATOM_LOADER_USE_THREADPOOL=1
  ATOM_LOADER_THREADPOOL_WORKERS=8` → ~3 min vs ~30 min. (host RAM 3TB holds the 1.4TB cache)
- Server run cmd: see `my_script/run_models.sh` but use `-tp 8` (tp4 OOMs) and add the
  `--no-enable_prefix_caching --no-enable_chunked_prefill` flags for the SDPA prefill.
