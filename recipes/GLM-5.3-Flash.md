# GLM-5.3-Flash Bring-Up Notes

> **Status: serving-accurate under ATOM on 8× MI355X, long context included.**
> `atom/models/glm5_next.py` loads every text-model parameter (the unsupported
> vision tower is skipped) and scores **gsm8k 0.9682 / 0.9689** at 3-shot on the
> dense path and **0.9659 / 0.9666** at 16-shot on the pooled k-pool path, over
> all 1319 questions (chat, TP8, bf16 KV; see §7). Per-layer hidden states match
> the transformers reference to cosine ≥ 0.9997 at all 45 layers. Not yet done:
> the MTP draft layer or multimodal serving. See §8.

```bash
python -m atom.examples.simple_inference --model /models/GLM-5.3-Flash -tp 4 \
    --kv_cache_dtype bf16 --max-tokens 64
```

Needs an aiter with `chunk_kimi_delta_attn` and `mla_decode_fwd(causal=...)` —
`rocm/atom-dev:nightly_202608270231` or newer. Earlier images fail with
`No module named 'aiter.ops.triton.kimi_delta_attn'` or
`mla_decode_fwd() got an unexpected keyword argument 'causal'`.

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is a natively
multimodal MoE model from Z.ai — 320B total / 18B active, 1M context, FP8 weights,
text + image + video. Architecture: `Glm5NextForConditionalGeneration`, `model_type`
`glm5_next`. See the [GLM-5 technical report](https://arxiv.org/abs/2602.15763).

## 1. Architecture

45 text layers in a repeating hybrid pattern. The checkpoint also contains a
24-layer vision tower, which ATOM deliberately skips until its input processor
and serving path exist:

| Component | Shape / setting |
| --- | --- |
| Layers | 45 (+ layer 45 = MTP draft) |
| Attention pattern | 34 × KDA linear attention, 11 × DSA (layers 3, 7, 11, … 43) |
| MLA | `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 256, `v_head_dim` 256, 64 heads |
| **Positional encoding** | **NoPE** — `qk_rope_head_dim == 0`, `mla_use_nope`. Position comes from the KDA layers' causal conv + recurrence. |
| KDA | 64 heads × 128 head_dim, conv kernel 4, `gate_lower_bound` -5.0, **per-channel** (diagonal) decay |
| DSA indexer | 32 heads × 128, `index_topk` 2048, `index_kpool` 4 + compress + always-select-tail |
| mHC | `hc_mult` 4 residual streams, `hc_sinkhorn_iters` 20, at both attn and FFN sites |
| MoE | 288 routed (8/token) + 1 shared, `moe_intermediate_size` 2048, sigmoid + `noaux_tc`, `routed_scaling_factor` 2.5 |
| Dense layers | first 3 (`first_k_dense_replace`), `intermediate_size` 12288 |
| Quantization | block FP8 e4m3, `weight_block_size` [128, 128], dynamic activations |
| Vision (not served) | 24 layers, hidden 1024, image 448, patch 14, spatial merge 2, temporal patch 2 → `out_hidden_size` 4096 |

Two structural points that differ from every model ATOM currently serves:

* **The residual is 4-wide.** `inputs_embeds` is expanded to `[B, S, hc_mult, D]` at
  the embedding and stays that way through all 45 layers, collapsing to `[B, S, D]`
  via an *unweighted mean* (`HyperHead`) right before the final norm. Every sub-layer
  collapses in (`pre`) and expands out (`post`, `comb`).
* **The whole text model is NoPE.** There is no rotary embedding anywhere in the
  text path — not in MLA, not in the indexer.

## 2. Mapping onto existing ATOM components

Nearly every hard piece already exists in ATOM from recently-landed models. This is
assembly plus one new op, not a from-scratch port.

| GLM-5.3-Flash | ATOM equivalent | Fit |
| --- | --- | --- |
| KDA linear attention | `KimiKDAAttention` (`models/kimi_k3.py`), aiter `kimi_delta_attn` Triton kernels | Very close. Same **separate `q/k/v_conv1d`** layout as the checkpoint, per-head `A_log`, per-channel `dt_bias`, `f_a`/`f_b` forget gate, and it already reads `linear_attn_config.gate_lower_bound`. |
| mHC hyper-connections | `hc_split_sinkhorn` (`model_ops/sparse_attn_v4.py`), `Block.hc_pre`/`hc_post` (`models/deepseek_v4.py`) | **Math-exact.** Same sigmoid gates, same Sinkhorn schedule including the special first iteration, same `HC_POST_MULT = 2.0`. Checkpoint tensor names (`hc_attn_fn`/`base`/`scale`) are already what `Block` expects, and `hc_attn_fn` is `[24, 16384]` = exactly its `mixes` layout. `dim=4096` satisfies the fused aiter `mhc_pre`/`mhc_post` `% 512 == 0` constraint. |
| k-pool DSA indexer | **new** — dispatch in `model_ops/glm5_next/indexer.py`, kernels in `model_ops/glm5_next/kpool.py`, with CPU geometry tests and direct GPU kernel/reference parity tests | DeepSeek-V4's `Compressor` pools the same way at `compress_ratio=4` with an `ape` term, but overlapping + RoPE'd. GLM's is non-overlapping and NoPE. |
| MLA | `model_ops/attention_mla.py` via `MLAModules`, NoPE via `NoPositionalRotaryEmbedding` | Needs the rope block materialized at `_ROPE_PAD = 64` lanes of zeros, giving the 576-wide KV entry every ROCm MLA kernel assumes. A zero-*width* slice is not viable — see §4d. |
| MoE 288 × sigmoid/`noaux_tc` | `model_ops/fused_moe`, `models/glm4_moe.py`, `deepseek_v2.py` | Direct. |
| Block FP8 128×128 | existing DeepSeek block-FP8 path | Direct. |
| MTP (layer 45) | `deepseek_mtp.py` / `glm4_moe_mtp.py` | Layer 45 is a full DSA layer plus `eh_proj`/`enorm`/`hnorm`/`shared_head.norm`. `index_share_for_mtp_iteration` means it reuses the main model's top-k. |
| Vision tower | Not shipped on this text-only path | Building and loading an unreachable tower consumes VRAM and exposed its weights to text-only packed-name rewrites. Add it with the processor and end-to-end multimodal coverage. |

### Checkpoint → model weight remap

The checkpoint does not match the `transformers` module tree. The authoritative
remap for *transformers* is in `transformers/conversion_mapping.py` under
`"glm5_next"`:

| Checkpoint | Model |
| --- | --- |
| `layers.N.hc_attn_{fn,base,scale}` | `layers.N.attn_hc.{fn,base,scale}` |
| `layers.N.hc_ffn_{fn,base,scale}` | `layers.N.ffn_hc.{fn,base,scale}` |
| `self_attn.{A_log,dt_bias,f_a_proj,f_b_proj}` | `self_attn.forget_gate.{...}` |
| `self_attn.{q,k,v}_conv1d.weight` | `self_attn.conv1d.weight` (concat dim 0, **q,k,v order**) |
| `mlp.experts.*.{gate,up}_proj.weight` | `mlp.experts.gate_up_proj` (merge modulelist dim 0, concat dim 1) |
| `mlp.experts.*.down_proj.weight` | `mlp.experts.down_proj` (merge modulelist dim 0) |

Everything lives under `model.language_model.*` / `model.visual.*`; `lm_head` is
top-level and BF16.

ATOM needs less of this. It declares the mHC parameters flat on the layer exactly
as the checkpoint names them (`hc_attn_fn`, ...), keeps `q/k/v_conv1d` separate
like the checkpoint (Kimi-K3 already does), and folds the low-rank KDA output
gate after load, so its whole `weights_mapping` is one rule:
`"model.language_model." -> "model."`. `model.visual.*` is dropped via
`skip_weight_prefixes`, and checkpoint layer 45 (MTP) is dropped automatically by
the loader's past-last-layer filter. Only the expert and q/k/v fusions go through
`packed_modules_mapping`.

## 3. What is validated

The serving implementation is split between
`atom/model_ops/glm5_next/{indexer,kpool}.py`; there is no second, dead indexer
implementation. Its contracts are covered at three levels:

* `tests/model_ops/test_glm5_kpool_geometry.py` runs on CPU and pins the shared
  producer/metadata output width, including `ATOM_GLM5_KPOOL=0`.
* `tests/model_ops/test_glm5_kpool_kernels.py` runs on ROCm and compares the
  production pooling/Hadamard/query-quant kernels directly with their torch
  references. It also asserts that query quantization uses AITER's
  architecture-dependent FP8 dtype and max.
* The 16-shot GSM8K row in §7 drives pooled writes, scoring, expansion and MLA
  consumption end to end. The state tests additionally cover chunk-boundary
  tails, checkpoint copy, relocation and invalid CUDAGraph slots.

During bring-up, a separate `transformers>=5.16` harness compared the real
layer-3 weights over sequence lengths 7 / 64 / 300 / 2048 / 3000. That harness
is not carried in this tree because ATOM pins transformers 5.12.1; its measured
results remain bring-up evidence rather than a misleading CI test of different
code.

## 4. Bugs found during bring-up

(a)–(c) are upstream, in `transformers` and aiter. (d) was this port's own, and
is the one that decided whether the model works at all.

**a) `transformers` mis-quantizes the KDA forget gate.** The checkpoint's
`quantization_config.modules_to_not_convert` names it
`model.layers.N.self_attn.f_a_proj`, but two things break the match: the entries use
a `model.layers.` prefix while the real keys are `model.language_model.layers.`, and
the `glm5_next` conversion mapping renames those tensors to
`self_attn.forget_gate.f_a_proj` *before* the FP8 quantizer runs. Result: all 68
forget-gate linears (34 KDA layers × 2) are wrapped in `FP8Linear` while still
holding BF16 weights, with a freshly-initialised `weight_scale_inv`. Confirmed
directly:

```
0.self_attn.forget_gate.f_a_proj    FP8Linear   w=torch.bfloat16  scale=(1, 32)
```

This silently corrupts the KDA decay for anyone running this checkpoint under
transformers. Worked around by swapping such modules back to `nn.Linear` after load.

**b) The `finegrained-fp8` hub Triton kernel does not compile on gfx950.**
`kernels-community/finegrained-fp8` loads fine, but compiling it aborts in LLVM:

```
llvm/ADT/Sequence.h:275: iota_range(T, T, bool): Assertion `Begin <= End' failed.
```

Independent of (a) — it still fires after the forget-gate fix. Every block-FP8 Linear
and the MoE experts route through this kernel, so no FP8 `glm5_next` forward runs on
MI355X without a substitute. Replaced with `fp8_aiter_backend.py`, which routes to
`aiter.gemm_a8w8_blockscale` — the block-FP8 GEMM ATOM already ships — and matches a
torch dequant reference at cosine 0.9997 (the residual is FP8 activation quant, which
is what the checkpoint was trained for).

**c) aiter kernels launch on the current CUDA device, not the tensor's device.**
Found while wiring (b). With a multi-GPU `device_map`, accelerate's hooks move tensors
to `cuda:1..3` but never change the CUDA context, so `torch.cuda.current_device()`
stays `0`. Ordinary torch ops dispatch on the tensor's device; aiter's do not — they
launch on the current device and silently read and write the wrong GPU's memory:

```
[fp8-verify] FAIL in grouped_matmul call #21: finite=False
    devices: ['cuda:1', 'cuda:1', 'cuda:1'] current=0
```

All inputs were finite and well-scaled; the output came back NaN. It reproduces only
past the first device boundary, so the first ~20 calls look fine — the failure mode is
a model that loads, runs, and emits garbage. transformers warns about exactly this for
DeepGEMM in its FP8 loader; aiter has the same constraint and no such guard. Fixed by
wrapping every aiter call in `with torch.cuda.device(tensor.device)`. Not a concern for
ATOM proper (one device per rank), but it bites any multi-device single-process use.

**d) A zero-WIDTH rope slice is not a viable way to express NoPE on ROCm.**
The obvious reading of `qk_rope_head_dim == 0` is to leave it at 0 and let the
MLA's rope half be an empty slice. That was this port's original choice and it is
wrong in two independent ways, both measured:

* **Accuracy collapses silently.** The paged MLA entry is sized
  `kv_lora_rank + qk_rope_head_dim`, so it comes out 512 while aiter's asm decode
  kernel is built for a 576-wide query. That kernel only *asserts* the 576 on the
  gfx1250 path, and `cfg_mla_asm` never dispatches on head_size, so on gfx950 the
  mismatch is computed rather than rejected. gsm8k measured **0.0099 /
  0.0000**: 69% of replies empty, and of the rest an extraction-free audit found
  the gold value in only 8.8% (a healthy run gives ~98%). Prefill is unaffected —
  it goes through `flash_attn_varlen`, which is head-dim generic — so a
  prefill-only per-layer cosine check reads 0.9997 and clears a broken model.
* **It also crashes outright.** `KV_PeDim == 0` makes every
  `tl.arange(0, KV_PeDim)` an `arange(0, 0)`, which Triton rejects at compile
  time, and upstream aiter's three `gather_kv_b_proj` kernels have no guard for
  it (`git show HEAD:...gather_kv_b_proj.py | grep -c 'KV_PeDim > 0'` → 0). It
  fires only on the cached-prefix prefill path, so a single-prompt demo passes
  and any concurrent run dies ~15 s in with
  `NameError('kv_pe_data is not defined')`.

The fix is `_ROPE_PAD = 64` in `atom/models/glm5_next.py`: materialize the rope
block at 64 lanes and hold it at zero. The apparent objection — `qk_nope_head_dim`
is already 256 and CK caps head_dim at 256 — conflates two constraints that apply
to **different tensors**. The latent/cache/decode side wants 576; the per-head
qk/prefill side wants ≤ 256. `MLAModules.rope_is_zero_pad` drops the zero lanes at
every `flash_attn_varlen_func` site, which is exact, so both are satisfied at once.

Two things this needs that are easy to miss. The padded width must be declared to
the cache allocator as `config.mla_kv_entry_dim`: `KimiMLAGDNBackend` shadows the
plain MLA allocator and sizes the pool from the raw config, so without it the pool
is built 512 wide under a 576-wide write and the server dies at startup with
`shape '[..., -1, 576]' is invalid`. And the pad must be appended by
`_ZeroRopePad`, which is deliberately **not** an `nn.Module` — wrapping `q_b_proj`
in one inserts a level into the parameter path and the weights silently never
load.

## 5. Reference oracle

A working `transformers` reference on this hardware, for diffing the ATOM port
against. Loads in ~131 s across 4× MI355X and generates coherent text.

**The harness itself is not in this tree.** It was bring-up scaffolding: it
pins `transformers==5.16.1` against ATOM's 5.12.1, so it cannot even be
imported from an ATOM environment, and most of it exists only to work around
the gfx950 `finegrained-fp8` failure of §4b. What it established is recorded
here and in §6. Rebuilding it takes four things, and the two that are not
obvious are the reason this section exists:

* **A separate image** — ROCm torch plus `transformers==5.16.1` and
  `kernels==0.16.0`, installed `--no-deps` so pip does not replace ROCm torch
  with the CUDA wheel. `glm5_next` does not exist in earlier transformers.
* **A replacement for the `finegrained-fp8` hub kernel**, which does not
  compile on gfx950 (§4b) and which every block-FP8 Linear and every MoE expert
  goes through. Two work, and building *both* is the point: route block-FP8
  through `aiter.gemm_a8w8_blockscale` — the same DeepSeek-style 128×128 kernel
  the checkpoint wants, quantising the activation as its
  `activation_scheme: dynamic` intends — or dequantise the weights and run a
  plain torch matmul, which is slower and keeps BF16 activations. Running the
  two against each other on every call, and reporting the first divergence with
  shapes and devices, is how the aiter multi-GPU bug of §4c was found; neither
  backend alone shows it.
* **`device_map="auto"` over 4× MI355X**, and the §4c warning applies: without
  that fix the model loads, runs, and emits NaN past the first device boundary.
* **A fixed prompt, greedy**, with the logits dumped, so §6 has something to
  diff against.

Measured on 4× MI355X, 21-token prompt, greedy:

| FP8 backend | decode |
| --- | --- |
| `aiter` (`gemm_a8w8_blockscale`) | 4.25 tok/s |
| `torch` (dequant reference) | 2.68 tok/s |

Both are far off what ATOM will do — this path is `device_map="auto"` pipeline
parallelism with one GPU active at a time, eager attention, no paged KV, a Python
loop over experts, and the dense k-pool indexer. It exists to be *correct*, not fast.

Reference next-token distribution for `"Give three reasons why the sky appears blue."`
(21 tokens, chat template, greedy):

```
    785   23.8750  'The'
   1654   17.3750  'We'
 154842   17.1250  '</think>'
```

and the greedy continuation, which confirms the model is in its default thinking mode:

```
The user is asking why the sky appears blue. This is a classic physics question
about Rayleigh scattering. Let me think about the actual scientific reasons.
```

## 6. Validation of the ATOM port

**Per-layer hidden states vs the reference**, 21-token prompt, TP4, BF16 KV.
ATOM's side comes from its own `ATOM_FWD_DUMP_DIR` block hooks; the reference
side from the same prompt under per-decoder-layer hooks on the §5 harness.
Cosine of the mHC residual `[21, 4, 4096]` at each layer:

| layer | 0 | 3 | 7 | 11 | 19 | 27 | 35 | 43 | 44 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cosine | 0.99972 | 0.99976 | 0.99970 | 0.99999 | 1.00003 | 0.99999 | 1.00002 | 0.99998 | 0.99998 |

It does **not** degrade with depth, which is the thing to check — a wired-up-wrong
component compounds, and this does not. Relative error holds at 1–2%, consistent
with FP8 activations plus a different kernel stack. (Cosine marginally above 1.0
is fp32 rounding in a reduction over 344k elements.)

Two traps worth knowing about when reproducing this:

* **Dump the right forward.** `ATOM_FWD_DUMP_ONE_SHOT` defaults on and writes only
  the *first* call, which is the warmup pass over 16384 dummy tokens. Comparing
  against that shows cosine 0.08 at layer 0 and looks like a catastrophic bug.
  Set `ATOM_FWD_DUMP_ONE_SHOT=0` and take the call whose row count equals the
  prompt length.
* **Teacher-forced rank-1 agreement is not a clean metric once trajectories
  fork.** Scoring ATOM's generated tokens under the reference — one forward
  over `prompt + ATOM's own tokens`, asking at each position what probability
  the reference gives the token ATOM picked — ATOM gets 67% rank-1 / mean
  p 0.58, against a baseline of 98.4% / 0.86 for the reference's own torch-FP8
  output rescored with the aiter-FP8
  backend. That gap is mostly an artifact: the baseline sequences never diverge,
  while ATOM's forks at position 4 — on `' why'` (p 0.63) vs `' about'` (p 0.11),
  a genuinely split distribution — and every later position is then scored
  against a prefix the reference would not have written. The per-layer cosines
  above are the load-bearing evidence, not this number.

**Vision prototype (bring-up only, not shipped).** The external reference
harness loaded `model.visual.*` and compared a prototype tower against
`Glm5NextVisionModel` on merged `[n_tokens, 4096]` outputs:

| dtype + attention | (1,2,2) | (1,4,4) | (1,8,12) | (2,4,4) |
| --- | --- | --- | --- | --- |
| fp32 + SDPA (correctness) | **bit-exact** | **bit-exact** | **bit-exact** | **bit-exact** |
| bf16 + aiter (serving) | .999555 | .999845 | .994098 | .999677 |

These numbers record the prototype's math, not supported ATOM behavior. The
tower was removed from this text-only landing because no processor can produce
its inputs under the pinned transformers version; loading it only consumed VRAM
and left unexercised packed-weight mappings in production.

GLM's expert SwiGLU limit (10.0) is attached to `FusedMoE`; current ATOM
backends forward it into their fused activation path. The dense layers use
`swiglu_oai_split` with the same limit.

**Performance**, 21-token prompt, TP4, eager, batch 1, no MTP:

| | TTFT | TPOT | decode |
| --- | --- | --- | --- |
| ATOM | 3.12 s | 45 ms | ~22 tok/s |
| transformers + aiter FP8 | — | — | 4.25 tok/s |
| transformers + torch FP8 | — | — | 2.68 tok/s |

The model now carries `@support_torch_compile`; TP8 level-3 compilation and
whole-forward CUDA graph capture are smoke-tested. Sharing the identity RoPE
cache across all 11 MLA layers reduced measured `peak_torch` from 42.71 GiB to
41.47 GiB per TP rank. MTP remains unsupported.

## 7. Measured serving accuracy

`lm_eval` gsm8k, all 1319 questions, TP8, bf16 KV, chat +
`--fewshot_as_multiturn`, greedy. `--max-model-len` matches the context column:

| | context | flexible-extract | strict-match |
| --- | --- | --- | --- |
| 3-shot (dense MLA) | 2048 | **0.9682** | **0.9689** |
| 16-shot (pooled k-pool) | 8192 | **0.9659** | **0.9666** |

Guarded by the `GLM-5.3-Flash` and `GLM-5.3-Flash-kpool-16shot` entries in
`.github/benchmark/models_accuracy.json` (threshold 0.94). SGLang publishes
0.9704–0.9757 for this model, so the port is in line.

Post-review validation on 2026-09-01, after merging current `main` and enabling
level-3 compilation, produced **0.9682 / 0.9682**. The final ATOM-style refactor
(shared NoPE cache, split indexer dispatch, no D2H scalar read, ATOM Linear gate)
produced **0.9659 / 0.9666** on the same 1319-question run, exactly matching the
catalog baseline and comfortably clearing the 0.94 threshold.

**Only the 16-shot row exercises the pooled path.** GSM8K prompts are short —
3-shot is ~389 tokens, 5-shot ~645 — so at or below `index_topk` the indexer
selects every token, `attention_mla` runs dense, and the pooled selection is
computed and thrown away. 16-shot prompts run 2763–3591 tokens, so every one of
the 1319 questions exceeds 2048 and pooled scoring plus pooled top-k decide what
the model attends to. Do not quote a short-context score as evidence that pooled
selection works; it only shows the pooled writes did no harm.

Use `--max-model-len 8192`, not 4096: lm_eval samples its shots at random, the
longest prompt is 3591 tokens, and 3591 + `max_gen_toks` 512 overflows a 4096
cap — 4 requests 400 and lm_eval then aborts the whole eval.

Three things about scoring this model that will otherwise waste a run:

* **Chat mode with few-shot, always.** lm_eval's gsm8k filters assume the
  few-shot `#### N` convention; this model answers in markdown/LaTeX. 0-shot chat
  scores strict-match 0.0000 on replies that are ~98% correct.
* **Never trust a low score before an extraction-free audit.** Ask whether the
  gold value appears anywhere in the reply. That is what separates a filter
  artefact from a real regression — it read 8.8% for the broken NoPE
  representation (§4d) and 98.0% after the fix.
* **The model is nondeterministic.** Six identical greedy requests give 2–3
  distinct outputs, and two runs of identical code disagree on ~28 of 1319
  questions in each direction. Judge changes on the 1319-question aggregate, or
  better on a per-question paired comparison; never on sample text.

## 8. Remaining work

1. ~~**Contexts beyond 2048.**~~ Done —
   `model_ops/glm5_next/{indexer,kpool}.py` implements the paged/ragged pooled
   indexer; measured at 16-shot in §7 and checked directly against its torch
   kernel references.
2. **MTP draft layer** (checkpoint layer 45: `eh_proj` / `enorm` / `hnorm` /
   `shared_head.norm`, plus its own indexer). `index_share_for_mtp_iteration`
   means it reuses the main model's top-k.
3. **Parallel feature coverage.** PCP, DCP and TBO remain explicitly rejected
   until their pooled-index metadata/state layouts have dedicated tests.
4. **Multimodal serving.** Land the image processor, input builder, tower and
   packed-weight tests together. `Glm5NextProcessor` only exists in transformers
   >= 5.16 while ATOM pins 5.12.1; video additionally needs frame sampling.
   Until then `model.visual.*` is skipped so text serving does not pay its VRAM.
5. **Performance**: tune the newly-enabled compiled path, add MTP speculative
   decoding, and drop the `hc` torch fallback once the fused path is trusted.
6. **Upstream the transformers FP8 bug** (§4a) and the gfx950 Triton failure (§4b).

Upstream, for reference: sglang PR #36507 (16.6k lines, 144 files) and vLLM PR
#53906 (12.5k lines, 85 files). Both are NVIDIA-first — sglang ships `.cuh` + TileLang
k-pool kernels and its ROCm CI is red; vLLM puts the model under
`vllm/models/glm5next/nvidia/`. Neither is a shortcut for AMD.
