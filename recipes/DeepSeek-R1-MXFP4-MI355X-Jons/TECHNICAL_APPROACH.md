# Technical Approach

## Goal
Maximize `tput_per_gpu = total_token_throughput / 8` for `DeepSeek-R1-0528-MXFP4` on 8× MI355X (gfx950) against AMD's benchmark harness (`dsr1_benchmark perf`) at ISL=8192 / OSL=1024 for concurrency levels 4, 32, and 128. Must pass GSM8K accuracy ≥ 0.93.

## Final Stack (Submitted)

### conc=128: `tp8_spec3_bigbatch` — 3537.19 tok/s/GPU (TOP-1)
```
python3 -m atom.entrypoints.openai_server \
  --model /share4/teamK/DeepSeek-R1-0528-MXFP4 \
  --server-port 8888 -tp 8 \
  --kv_cache_dtype fp8 \
  --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 256
```

### conc=4: `c4_level3` — 711.28 tok/s/GPU
```
python3 -m atom.entrypoints.openai_server \
  --model /share4/teamK/DeepSeek-R1-0528-MXFP4 \
  --server-port 8888 -tp 8 \
  --kv_cache_dtype fp8 \
  --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 \
  --level 3 \
  --cudagraph-capture-sizes "[1,2,4,8]"
```

## Knob Catalogue — What Helped vs What Hurt

### Helps at conc=128
- `--max-num-batched-tokens 131072` + `--max-num-seqs 256` — enables aggressive prefill batching for 128 concurrent users. **+~10% over vanilla.**
- MTP speculative decoding with `--num-speculative-tokens 3` — gives ~2.3 tokens/forward via MTP acceptance.
- Default `--method mtp` is the only supported speculative method in ATOM v0.1.2.

### Helps at conc=4
- `--level 3` + `--cudagraph-capture-sizes "[1,2,4,8]"` — tight cudagraph capture covering the small batch sizes seen at conc=4 with spec=3.
- `--num-speculative-tokens 3` (max for fp8 MLA path on this build — see Constraints).

### Confirmed Dead Knobs (regressions or crashes on this build)
| Knob | Effect | Reason |
|------|--------|--------|
| `--enable-dp-attention` | Tensor shape mismatch (20480 vs 16384) | v0.1.2 DP-attn bug |
| `--enable-expert-parallel` (without MoRI tuning) | MoRI symmetric heap OOM | Default heap = 2 GB |
| `--data-parallel-size > 1` | `recvBytes` / process group init failure | RCCL/MoRI conflict |
| `--enable_prefix_caching` | `NoneType.shape` per request | v0.1.2 bug |
| `--num-speculative-tokens ≥ 4` (fp8 KV) | C++ assert: `qo_len <= 4` | Hard cap in `asm_mla.cu:281` |
| `--kv_cache_dtype bf16` + `--num-speculative-tokens ≥ 4` | GSM8K = 0.05 (output broken) | DSR1 MTP head not trained for spec > 3 |
| `--kv_cache_dtype bf16` at TP=8 | ~25% throughput regression vs fp8 | 2× KV bandwidth |
| TP=4 with default cudagraph capture | GPU memory access fault (MoE) | TP=4 + batch=65 (GSM8K eval concurrency) outside captured graphs |
| AMD env stack alone (`HIP_FORCE_DEV_KERNARG=1`, `AITER_ENABLE_VSKIP=1`, `AMD_DIRECT_DISPATCH=1`, `GPU_MAX_HW_QUEUES=8`) | -40% at conc=128 | Co-tuned vars; need pairing with level=3/cudagraph |
| `--max-num-seqs > 256` | Crash | Session 012 confirmed |
| `--enforce-eager` | -77% | Cudagraph load-bearing |
| `--block-size 128` | -2% | Slight regression |

## Structural Ceiling — What We Could Not Solve

To exceed our submitted numbers we would need **higher speculative acceptance per forward step**. Two empirically-proven blockers prevent this on `rocm/atom:rocm7.2.1-ubuntu24.04-pytorch2.9.1-atom0.1.2`:

1. **AITER fp8 MLA decode kernel hard-caps `qo_len ≤ 4`** (`/app/aiter-test/csrc/py_itfs_cu/asm_mla.cu:281`). The precompiled `.co` binaries in `/app/aiter-test/hsa/gfx950/mla/` only ship `qSeqLen ∈ {1, 2, 4}` for fp8. No source `.s` files are present to rebuild for qSeqLen > 4. This caps MTP at spec=3 in fp8.

2. **DSR1's single MTP layer (`num_nextn_predict_layers = 1`) was trained for spec=3**. Empirically tested: at TP=4 + bf16 + spec=4 we get GSM8K=0.0561 (broken). At spec=5: GSM8K=0.0508. Output collapses to random tokens above spec=3.

We additionally investigated unlocking EAGLE/NEXTN via SGLang v0.5.9-rocm700-mi35x, which would allow tree speculation with higher acceptance. We found **at least 3 cascading bugs** in SGLang's MTP+TP=8+MXFP4 load path:
- `channel_quant_to_tensor_quant` shape mismatch (`fp8_utils.py:1035`)
- `quark_post_load_weights` UnboundLocalError on fp8 input (`quark/utils.py:214`)
- `apply_fp8_linear` receives tuple instead of tensor (`fp8_utils.py:1105`)

Partial patches are in `sglang_patches/deepseek_weight_loader.py`. Full fix is multi-day work beyond the window.

## Prototype Work Product Beyond the Submission

We additionally built a **Triton fp8 MLA decode kernel** (`atom_patches/triton_mla_fp8_multi.py`) that supports arbitrary `qo_len` up to 8, intended to bypass the AITER kernel cap. It passes GSM8K (0.9447 at qo_len=4 baseline against the ASM kernel) but is **~8× slower than the ASM kernel** due to Triton's lack of native fp8 dot product support on AMD. With more time it could be optimized to be competitive. We include it for completeness.

## Methodology Notes
- All numbers are from `dsr1_benchmark perf` (the AMD-provided harness). Submissions used `dsr1_benchmark submit Jons`.
- Each run loads the model fresh, runs GSM8K validation, then runs the perf bench. Total ~12-15 minutes per run.
- Variance on c4_level3 has σ ≈ 14 tok/s/GPU around mean ~715. The 711 submission landed at the low end of variance; historical peak from this same config is 736 (May 11).
- Benchmark harness binary computes `tput_per_gpu = total_token_throughput / 8.0` (hardcoded — verified by `strings`).
