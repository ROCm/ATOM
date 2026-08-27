# GLM-5.3-Flash Bring-Up Notes

> **Status: in progress — not yet servable under ATOM.**
> This documents the architecture analysis, the ATOM component mapping, the
> validated pieces, and the reference oracle to build the port against. The ATOM
> model itself (`atom/models/glm5_next.py`) is not written yet.

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is a natively
multimodal MoE model from Z.ai — 320B total / 18B active, 1M context, FP8 weights,
text + image + video. Architecture: `Glm5NextForConditionalGeneration`, `model_type`
`glm5_next`. See the [GLM-5 technical report](https://arxiv.org/abs/2602.15763).

## 1. Architecture

45 text layers in a repeating hybrid pattern plus a 24-layer vision tower:

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
| Vision | 24 layers, hidden 1024, image 448, patch 14, spatial merge 2, temporal patch 2 → `out_hidden_size` 4096 |

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
| k-pool DSA indexer | **new** — `model_ops/kpool_indexer.py` (this branch) | DeepSeek-V4's `Compressor` pools the same way at `compress_ratio=4` with an `ape` term, but overlapping + RoPE'd. GLM's is non-overlapping and NoPE. |
| MLA | `model_ops/attention_mla.py`; NoPE via the `_NoPositionalRotaryEmbedding` trick in `kimi_k3.py` | Needs a NoPE path (`qk_rope_head_dim == 0`) through the MLA backends. |
| MoE 288 × sigmoid/`noaux_tc` | `model_ops/fused_moe`, `models/glm4_moe.py`, `deepseek_v2.py` | Direct. |
| Block FP8 128×128 | existing DeepSeek block-FP8 path | Direct. |
| MTP (layer 45) | `deepseek_mtp.py` / `glm4_moe_mtp.py` | Layer 45 is a full DSA layer plus `eh_proj`/`enorm`/`hnorm`/`shared_head.norm`. `index_share_for_mtp_iteration` means it reuses the main model's top-k. |
| Vision tower | `kimi_k3_vl.py`, `qwen3_5_vl.py` | Standard ViT: fused `qkv`, `q_norm`/`k_norm`, gated MLP, `downsample`, `merger`. All BF16. |

### Checkpoint → model weight remap

The checkpoint does not match the `transformers` module tree; the authoritative
remap is in `transformers/conversion_mapping.py` under `"glm5_next"`. ATOM's loader
must reproduce it:

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

## 3. What is validated

**`atom/model_ops/kpool_indexer.py`** — selects byte-identical token indices to
`transformers`' `Glm5NextTextIndexer` on the real layer-3 weights, over sequence
lengths 7 / 64 / 300 / 2048 / 3000 and with left padding of 5 and 17 tokens
(`seq=3000` exceeds `index_topk`, so genuine sparse pool selection is exercised).

Unit tests (CPU, synthetic weights): `tests/model_ops/test_kpool_indexer.py`.

To re-run the weights-based parity check you need the checkpoint and one GPU:

```bash
# transformers >= 5.16 is required for glm5_next; install --no-deps into a ROCm
# image so pip does not replace ROCm torch with the CUDA wheel.
pip install --no-cache-dir --no-deps \
    "transformers==5.16.1" "tokenizers>=0.23.1,<0.24" "accelerate" \
    "kernels==0.16.0" "kernels-data"
python kpool_parity_atom.py   # see §5
```

## 4. Two upstream bugs found during bring-up

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

## 5. Reference oracle

A working `transformers` reference on this hardware, for diffing the ATOM port
against. Loads in ~131 s across 4× MI355X and generates coherent text.

Artifacts (on the bring-up machine):

The harness lives in [`recipes/glm5_3_flash/`](glm5_3_flash):

| File | Purpose |
| --- | --- |
| `Dockerfile` | ROCm torch 2.10 + transformers 5.16.1 reference image (`glm53-ref:tf5161`) |
| `ref_run.py` | loads the checkpoint, dumps oracle logits, generates |
| `fp8_aiter_backend.py` | routes block-FP8 through `aiter.gemm_a8w8_blockscale` (default) |
| `fp8_torch_fallback.py` | torch-only dequant bundle; slower, BF16 activations, cross-check |
| `aiter_fp8_check.py` | aiter block-FP8 GEMM vs torch dequant on a real GLM weight |
| `kpool_parity_atom.py` | ATOM k-pool op vs `transformers`, on real weights |

Env knobs on `ref_run.py`: `GLM53_FP8_BACKEND=aiter|torch`, `GLM53_MAX_NEW_TOKENS`,
`GLM53_FP8_VERIFY=1` (runs both backends every call and reports the first divergence
with shapes and devices — how §4c was found).

Measured on 4× MI355X, 21-token prompt, greedy:

| FP8 backend | decode |
| --- | --- |
| `aiter` (`gemm_a8w8_blockscale`) | 4.25 tok/s |
| `torch` (dequant reference) | 2.68 tok/s |

Both are far off what ATOM will do — this path is `device_map="auto"` pipeline
parallelism with one GPU active at a time, eager attention, no paged KV, a Python
loop over experts, and the dense k-pool indexer. It exists to be *correct*, not fast.

On the bring-up machine: weights at `/raid/carhuang/models/GLM-5.3-Flash`
(62 shards, 305.8 GiB, verified), oracle logits at
`/raid/carhuang/glm53_out/ref_logits.pt` and `ref_top10.json`.

```bash
docker build -t glm53-ref:tf5161 recipes/glm5_3_flash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --shm-size 64G --ipc=host --privileged -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v /raid/carhuang/models/GLM-5.3-Flash:/models/GLM-5.3-Flash:ro \
  -v /raid/carhuang/glm53_out:/out \
  -v $PWD/recipes/glm5_3_flash:/w -w /w \
  --entrypoint python3 glm53-ref:tf5161 -u ref_run.py
```

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

## 6. Remaining work

1. `atom/models/glm5_next.py` — config normalisation (`linear_attn_config` →
   the `linear_*` names `KimiKDAAttention` reads), the mHC decoder block, KDA and
   NoPE-MLA layers, MoE, and the weight remap above.
2. A paged/ragged k-pool indexer that reads pooled state from the KV cache instead of
   rebuilding pools densely each step; `kpool_indexer.py` is its correctness oracle.
3. Hybrid cache sizing — a paged KV pool for the 11 DSA layers *and* a KDA recurrent
   state pool for the 34 linear layers (upstream notes the KDA pool is what usually
   caps concurrency).
4. NoPE path through the MLA attention backends.
5. Vision tower + processor; MTP draft layer.
6. Register `Glm5NextForConditionalGeneration` in
   `model_runner.py:support_model_arch_dict`.

Upstream, for reference: sglang PR #36507 (16.6k lines, 144 files) and vLLM PR
#53906 (12.5k lines, 85 files). Both are NVIDIA-first — sglang ships `.cuh` + TileLang
k-pool kernels and its ROCm CI is red; vLLM puts the model under
`vllm/models/glm5next/nvidia/`. Neither is a shortcut for AMD.
