# Kimi-K3.1 original checkpoint

Native ATOM path for the original Kimi-K3.1 checkpoint
(`moonshotai/Kimi-K3.1-Pro`) on gfx950 / MI355X. The launch below scores
97.65% on GSM8K (5-shot, full set, mono on; see Validated scope).

## Precision

- Routed expert matrices keep the checkpoint's MXFP4. Attention, the dense MLP
  and the latent MoE down/up matrices keep their BF16 weights; KDA recurrence
  controls, convolutions and output norms keep their FP32 storage.
- MLA KV is BF16 (`--kv_cache_dtype bf16`); the AgentX launch below runs FP8
  KV (`--kv_cache_dtype fp8`). MLA K/V projections and the absorbed BMM
  weights stay BF16.
- The DSA index cache must be FP8 or FP4: pass `--index-cache-dtype fp8`, or
  `fp4` on gfx950 (the AgentX launch below adds `--index-cache-layout mfma`).
  Left unset it follows the KV cache dtype (BF16), which the NoPE index writer
  does not support. FP8 applies to the indexer's Q/K activations for MQA
  scoring, not to weights.

## Model wiring

- `flatten_dsa_config` (`atom/config.py`) flattens the nested `dsa_config`
  before attention pools are chosen; only the `dsa_layers` own index planes.
- All 24 MLA layers carry a NoPE indexer: 32 heads, head dimension 128, causal
  top-k 2048, no index sharing. Queries come from the normalized q-lora latent.
- `FusedIndexerProjection` concatenates q_a, kv_a, index K and index head
  weights in one BF16 projection; all four checkpoint regions must load, and a
  quantization scheme (checkpoint or online) that would quantize it is
  rejected while the model is built.
- Loading rejects incomplete expert regions (unless
  `ATOM_LOADER_STRICT_COVERAGE=false`) and warns about parameters nothing
  loaded, checkpoint tensors nothing took, and float tensors converted to
  another dtype. Check the load log: a missed parameter keeps its init value
  and fails silently.

## AITER requirements

Validated with AITER `main` at `dbaffe23ed` plus
[ROCm/aiter#7067](https://github.com/ROCm/aiter/pull/7067); the AgentX launch
below needs a newer AITER, named there. K3.1 relies on:

- Custom all-reduce summing ranks in a fixed order on every reduce path
  (#6954), so a token's sum does not depend on its row offset or the receiving
  rank.
- The D192/V128 OPUS attention scanning KV forward on both passes, so batch
  composition does not change BF16 attention order.
- The gfx950 BF16 MLA prefill code objects waiting for the first KV/index
  prefetch before reading it.
- From #7067: the register top-k path taking the FP32 `e_score_correction_bias`
  with BF16 router logits, and the MXFP4 MoE stage2 honoring the tuned reduce
  mode (bit-reproducible routed-expert output). Use AITER's normal merged
  tuning tables; pointing `AITER_CONFIG_FMOE` at a single CSV drops required
  shared entries.

After updating AITER, rebuild its JIT modules before starting the server: an
existing `aiter/jit/module_*.so` is loaded without being compared to its
sources. Clear `aiter/jit/` (modules and `build/`) and let the first start
rebuild them, or rebuild them in one process. Do not set `AITER_REBUILD=1` on
the TP server: every rank would rebuild every module it uses.

## Launch

TP4 cannot hold the resident checkpoint; use TP8.

```bash
export AITER_LOG_LEVEL=WARNING

python -m atom.entrypoints.openai_server \
  --model /data/Kimi-K3.1-Pro --trust-remote-code -tp 8 \
  --decode-context-parallel-size 8 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --max-model-len 131072 --max-num-seqs 64 --block-size 128
```

- Mono decode is on by default (`ATOM_MONO_ENABLE=1`; set it to 0 to run
  the original decode path). It serves decode steps of up to 64 rows, a plain
  decode or a speculative verify (16-row token groups past 16): the 69 KDA
  layers run K1 (the attention seam through the KDA recurrence) and K2
  (`o_proj` plus the latent MoE), the 24 MLA layers keep their attention and
  run K2, and their DSA indexer runs as one mono launch: under DCP equal to TP
  over the ranks' own keys, without DCP over the replicated cache (a request's
  rows owned by one rank, a long one's keys split; not yet validated on
  K3.1). A wider step runs the
  original path. It works with ReplaySSM (`ATOM_ENABLE_REPLAYSSM=1`) and with
  a DSpark drafter (`--method dspark --draft-model /data/Kimi-K3-DSpark
  --num-speculative-tokens 7`, at `--max-num-seqs 8` so a step stays within 64
  rows). The kernels compile on each width's first step (each rank compiles
  its share); a cold FlyDSL cache adds a few minutes to the first requests.
- DCP is optional (`--decode-context-parallel-size`); it uses the shared FP8
  index-key exchange for prefill and the score-only candidate merge
  (`flydsl_dcp_topk_merge`) for decode. KDA keeps the complete token sequence
  on every rank.
- Local prefix caching (`--enable_prefix_caching`) reuses KDA's PAGE-backed
  state checkpoints. MLA and index keys share scheduler block identity, and a
  hit requires a restorable recurrent checkpoint.

## Validated scope

`moonshotai/Kimi-K3.1-Pro` (the checkpoint now published; same tensors as the
first K3.1 release, new values): GSM8K 5-shot on the full set (1319 questions)
with the launch above, mono on: flexible-extract 97.65%, strict-match 97.65%
(±0.42), no fault. The results below are from the first K3.1 release.

GSM8K 5-shot on the full set (1319 questions) with the launch above and mono
off: flexible-extract 97.19%, strict-match 97.19% (±0.45), no fault.

Mono, paired against the original path on the same runs (TP8 DCP8, ISL / OSL
1024): GSM8K 96.66% vs 96.13–96.51% at concurrency 16, disagreements within
the same-code band; throughput at concurrency 64 2713 -> 3038 tok/s (+12%).
With DSpark (pinned acceptance length 4.59) and ReplaySSM, concurrency 1 / 4 /
8: 321 -> 562, 919 -> 1354, 1656 -> 1931 tok/s; GSM8K mono with ReplaySSM
96.21%, the original path with ReplaySSM 96.21%; 5-shot with mono, DSpark
(no pinned acceptance) and ReplaySSM at `--max-num-seqs 8`: 96.97% (±0.47).

Only these configurations have been validated: TP8 with DCP8,
`--kv_cache_dtype bf16`, `--index-cache-dtype fp8`, CUDA graphs on, with mono
on or off, and with the DSpark drafter and ReplaySSM above; and the AgentX
launch below (FP8 KV, the FP4 index cache in the `mfma` layout, LMCache MP
offload, DSpark and ReplaySSM, mono off), with the GSM8K result that closes
its section.
PCP is rejected while the model is built (the indexer's projections live in
the fused q_a projection, and the PCP indexer projects k itself). Nothing
else is rejected at startup, but online quantization of the other layers,
PP, expert parallelism, DP attention, other speculative methods, plugins,
TBO and KV transfer other than that LMCache MP offload are untested. The
vision wrapper is registered, but image inference has not been validated.

## AgentX at CONC 16

The SemiAnalysis/Weka AgentX replay (AIPerf scenario `inferencex-agentx-mvp`,
dataset `semianalysis_cc_traces_weka_062126`) at concurrency 16. This launch
differs from the one above: FP8 KV, the FP4 index cache in the `mfma` layout,
block size 64, prefix caching with an LMCache MP CPU tier, and DSpark with
seven draft tokens at a synthetic acceptance length.

It needs a newer AITER than AITER requirements names: the `mfma` index cache
scores with `mqa_logits_mxfp4_config` and `mqa_logits_mxfp4_schedule`
([ROCm/aiter#6952](https://github.com/ROCm/aiter/pull/6952)) and writes
through `indexer_qk_rope_quant_and_cache` with `scale_layout="opus"`
([ROCm/aiter#7798](https://github.com/ROCm/aiter/pull/7798)). The
`b038a7dff` row and its GSM8K ran on AITER `342aa1dc1` plus ROCm/aiter#7798
at `86719be77`, with the `Kimi-K3.1-Pro-DSpark-preview` draft at revision
`6a2ef20a`.

Start the LMCache server, then ATOM:

```bash
lmcache server --host 127.0.0.1 --port 5585 --chunk-size 1024 --null-block-id -1 \
  --separate-object-groups --supported-transfer-mode lmcache_driven \
  --l1-size-gb 1799 --l1-use-lazy --eviction-policy LRU \
  --l1-read-ttl-seconds 900 --eviction-trigger-watermark 0.98 &

export AITER_LOG_LEVEL=WARNING AITER_SITUV2_A4W4=1 AITER_FLYDSL_USE_HERD=1 \
  AITER_FLYDSL_HERD_MIN_M=64 AITER_REUSE_IDENTICAL_COMM_GROUPS=0 \
  AITER_CUSTOM_AR_DISABLE_LAMPORT=1 ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1 \
  ATOM_GDN_SSM_DTYPE=fp32 ATOM_ENABLE_REPLAYSSM=1 ATOM_MONO_ENABLE=0 \
  ATOM_STATE_CHECKPOINT_DEMAND=0 ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0 \
  ATOM_DCP_PREFILL_QREP=1 ATOM_MLA_ABSORB_BF16=1 PYTHONNOUSERSITE=1 \
  PYTHONHASHSEED=0 LMCACHE_CHUNK_SIZE=1024 OFFLOAD_MP_LOOKUP_THREAD=1
unset ATOM_PREFILL_DECODE_INTERVAL ATOM_PREFILL_DELAYER_MAX_QUEUE_MS \
  AITER_QUICK_REDUCE_QUANTIZATION AITER_FLYDSL_STAGE2_FP8 \
  PYTORCH_HIP_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF

python -m atom.entrypoints.openai_server \
  --model /data/Kimi-K3.1-Pro --served-model-name kimi-k3.1-pro \
  --trust-remote-code --host 127.0.0.1 --server-port 8100 \
  --tool-call-parser kimi_k3 -tp 8 --decode-context-parallel-size 8 \
  --kv_cache_dtype fp8 --index-cache-dtype fp4 --index-cache-layout mfma \
  --block-size 64 --enable_prefix_caching --max-num-seqs 16 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 \
  --state-checkpoint-interval-tokens -1 --level 3 --cudagraph-mode FULL \
  --cudagraph-capture-sizes "[$(seq -s, 1 128)]" \
  --method dspark --draft-model /data/Kimi-K3.1-Pro-DSpark-preview \
  --num-speculative-tokens 7 --spec-decode-acceptance-length 3.69 \
  --kv-transfer-config '{"kv_connector":"lmcache_mp","kv_role":"offload","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5585,"lmcache.mp.read_ttl_seconds":900,"lmcache.mp.transfer_deadline_s":300}}'
```

- `--gpu-memory-utilization 0.85`, not 0.94: at 0.94 a two-request prefill
  near 317K tokens of context exhausts VRAM and hangs the server
  ([LLM-Booster#564](https://github.com/wuhuikx/LLM-Booster/issues/564)).
- The prefill delayer stays off: holding prefills back raises TTFT at this
  concurrency.
- `--spec-decode-acceptance-length` commits draft tokens unverified
  ([`docs/forced_acceptance_length.md`](../docs/forced_acceptance_length.md)).
  It is for performance runs only; drop it for GSM8K.

Once the server answers, run the client from the ATOM tree.
`agentic_prepare` finds or installs the `agentx` harness, and
`run_aiperf_agentic` takes the server URL, the concurrency and the output
directory:

```bash
MODEL_PATH=/data/Kimi-K3.1-Pro AIPERF_MODEL=kimi-k3.1-pro \
  AIPERF_SCENARIO=inferencex-agentx-mvp CONC=16 \
  bash -c 'source .github/scripts/aiperf_agentic.sh && agentic_prepare &&
    run_aiperf_agentic http://127.0.0.1:8100 16 ./agentx-c16'
```

Each row is measured over the first 3600 s of the profiling phase in
`profile_export.jsonl`, counting the requests that end inside it: throughput
per GPU is their input plus output tokens / 3600 / 8, P90 interactivity is
1 / p90(ITL). The `2c946f623` row selected the same index cache with
`ATOM_DSA_FP4_OPUS=1` and ran with the KV budget of utilization 0.94.

| Run | Throughput / GPU | TTFT p50 / mean | ITL p50 | P90 interactivity | Requests |
|---|---:|---:|---:|---:|---:|
| ATOM `b038a7dff`, this launch, 2026-11-14 | 8225.0 | 0.87 / 1.37 s | 13.61 ms | 42.76 | 1869 |
| ATOM `2c946f623` (local build), 2026-11-12 | 7958.5 | 0.81 / 1.31 s | 14.82 ms | 40.36 | 1791 |
| SGLang, InferenceX CI run 42324316724 | 9238.8 | 0.70 / 1.21 s | 11.62 ms | 62.67 | 2074 |

GSM8K 5-shot at `b038a7dff` without `--spec-decode-acceptance-length`:
flexible-extract 97.73%, strict-match 97.65% (±0.42). That run set
`ATOM_DCP_PREFILL_A2A_FP8=1` and `--gpu-memory-utilization 0.94`.
