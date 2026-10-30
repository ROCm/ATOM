# Qwen3.8-Flash-Next Mono service

This service variant combines the Qwen Mono target path with compact MTP
draft execution and a global-certificate vocabulary screen. The screen keeps
the original BF16 head weights and refines eligible logits using those weights.

The configuration follows Ling Zhang's `l33_final6` Native ATOM deployment:
`amd/Qwen3.8-Flash-Next-Quark-MXFP4`, TP1, MTP3, FP8 KV, BF16 index cache,
PTPC FP8 target attention projections, BF16 draft, and 128 pinned host/UVA PLE
shards. The combined author tree is `3a7cf774c1e8d1429233b7eaa672bd1676fab7b7`.
Relevant PRs are [PLE host tables #2771](https://github.com/ROCm/ATOM/pull/2771),
[draft MRoPE graphs #2663](https://github.com/ROCm/ATOM/pull/2663), and
[top-k #2665](https://github.com/ROCm/ATOM/pull/2665). Their live heads may differ
from the tested combined tree.

Use the matching author AITER wheel `0.1.1.dev1+g4d1980603` and its source patch.
Its extensions use pybind11 3.1.0. Keep all AITER extensions from a consistent
build; mixing them with extensions built using pybind11 3.0.4 produces an ABI
registration error. On this ROCm deployment, set `TMPDIR` to local container
storage. A shared-filesystem `TMPDIR` caused a reproducible HIP startup crash.

## Execution and ownership

| Request shape | Target | Draft |
| --- | --- | --- |
| C1, MTP3, four verify rows | Mono HC/BA, projection/conv, router and compact expert chain; Native sparse attention and Mono ReplaySSM seams | Compact selected rows; global certificate head at one row |
| C2/C4, MTP3, eight/sixteen verify rows | Existing Mono target with Native expert seam | Compact selected rows; Native head for wider inputs |
| Prefill or unsupported step | Original model path | Original path where compact execution is ineligible |

`dispatch.py` owns step admission, `quark.py` owns the target layer loop and
deferred HC state, and `draft/` owns the compact draft path. `head/binding.py`
prepares the screened head through the model's existing weight preparation
hook. Runtime modules live under `atom.models.qwen4_exp_mono`; no external
experimental directories are required on `PYTHONPATH`.

`replayssm.py` contains the Mono serial split-V variant: it stages new shared
k/g records at the free tail during a flush, then publishes them after all
replay CTAs finish. The C1 GDN finish fuses that publication with its output
stage. Shared replay arithmetic, UT execution and cursor commits are reused;
the public ReplaySSM implementation is unchanged.

## Launch configuration

Enable the measured combination explicitly:

```bash
export ATOM_MONO_ENABLE=1
export QWEN_HC_BA=1 QWEN_HEAD_SCREEN=1 QWEN_GLOBAL_HEAD=1
export QWEN_GDN_PROJECTION_CONV=1 QWEN_GDN_FINISH=1
export QWEN_ROUTER_PARTITION=1
export QWEN_DRAFT_COMPACT=1 QWEN_DRAFT_TAIL=1 QWEN_DRAFT_QSA_FUSED=1
export QWEN_DRAFT_QSA_PROBE=0 QWEN_QSA_COMPACT_EMIT16K=0
export QWEN_MOE_HC_VALIDATE=0
export QWEN_MOE_HC_RUN=/tmp/qwen-mono-artifacts
export TMPDIR=/tmp AITER_LOG_LEVEL=WARNING
mkdir -p "$QWEN_MOE_HC_RUN"

python -m atom.entrypoints.openai_server \
  --model /models/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --served-model-name amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
  --trust-remote-code -tp 1 --method mtp --num-speculative-tokens 3 \
  --gpu-memory-utilization 0.95 --max-num-seqs 160 \
  --max-num-batched-tokens 16384 --max-model-len 262144 \
  --block-size 64 --kv_cache_dtype fp8 --index-cache-dtype bf16 \
  --enable-prefix-caching --state-checkpoint-interval-tokens 8192 \
  --timeout-keep-alive 60 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,40,48,56,64]' \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","*mlp.gate","*shared_expert_gate","*mlp.experts*","*mlp.shared_expert*","*hyper_connection*","*indexer*","*.ple.*","model.visual.*","mtp.*"]}'
```

This is the author's low concurrency configuration with KV offload disabled.
The C32/C64 LMCache configuration is a separate deployment. Do not apply the
older BF16 tuned-row workaround to the l33 AITER configuration.

For accuracy, use real MTP verification: omit
`--spec-decode-acceptance-length`. The author GSM8K protocol uses the full 1,319
test questions, five-shot chat prompts with few-shot examples as separate
turns, temperature 0, top-p 1 and a 12,288-token output budget. A C1 client
exercises the one-row head and four-row target paths.

For the fixed-load performance comparison, use 65,536 input tokens, 4,000
output tokens, and `--spec-decode-acceptance-length 2.32`. Measure Native,
candidate, Native in the same environment and compare the cleanup against
the frozen pre-cleanup source. This diagnostic workload is distinct from
the author's 3,600-second AgentX trace benchmark. Historical measurements
do not establish performance for a new dependency snapshot.
