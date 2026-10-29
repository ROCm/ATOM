#!/bin/bash
# Qwen3.5-35B-A3B-MXFP4 MLPerf Endpoints (agentic) server on gfx1250, after the DSv4.1 endpoints
# recipe's server. usage: serve_ep.sh <tp|dp> <outdir>   (run_ep.sh calls it; see the recipe .md)
# tp: TP4 (the interactive points). dp: DP4 attention + EP4 MoE (the high-concurrency points:
# KV is not duplicated across ranks, ~2x the TP4 capacity). SPEC=<n>: MTP with n draft tokens.
MODE=$1; OUT=$2; mkdir -p $OUT; cd ${ATOM_DIR:-/app/ATOM}
MODEL=${MODEL:-/data/Qwen3.5-35B-A3B-MXFP4}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3} OMP_NUM_THREADS=4 AITER_LOG_LEVEL=WARNING
export HF_HUB_OFFLINE=1 PYTHONFAULTHANDLER=1 FLYDSL_GPU_ARCH=gfx1250
# FlyDSL 0.4.0 first when the image keeps an older default (the gfx1250 fw-bringup images from
# 2026-10-24: 0.3.3 in site-packages, 0.4.0 in /opt): the FlyDSL prefill FMHA at head_dim 256 needs it
[ -d /opt/flydsl-0.4.0 ] && export PYTHONPATH=/opt/flydsl-0.4.0${PYTHONPATH:+:$PYTHONPATH}
# platform knobs as in the DSv4.1 recipe: device kernargs, no async scratch reclaim (gfx12.5
# page-fault workaround), SDMA/SVM/XNACK, GC threshold
export HIP_FORCE_DEV_KERNARG=1 HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
export HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 ATOM_GC_THRESHOLD=20000,50,50
# aiter custom all-reduce (TP4 only). TP4 GSM8K with it is 0.86 once prefill attention runs on
# FlyDSL (aiter FMHA v_hdim 256); RCCL (MAX_SIZE=0) cannot be captured into hipGraphs on this host.
# From 16 KB up: a 128 KB floor sent every decode all-reduce of a batch of up to 8 (x 4 verify tokens
# x 2048 x bf16 = 128 KB) to RCCL.
[ "$MODE" = tp ] && export AITER_CUSTOM_AR_MIN_SIZE=${AITER_CUSTOM_AR_MIN_SIZE:-16384} \
  AITER_CUSTOM_AR_MAX_SIZE=${AITER_CUSTOM_AR_MAX_SIZE:-8388608}
# TP decode step (c16): verify and bonus rows sampled in one call, the sampled ids synced over the
# custom all-gather instead of an RCCL broadcast, one launch per sigmoid gate, the tokenizer cache for
# Qwen's NFC normalizer, aiter's register-resident top-p for the top-k-first sampler (the generic
# kernel spills, and every launch then waits on a scratch grow). Each one: the environment wins.
# Then the MoE route count staged once per forward (passed to aiter's fused_moe as num_valid_routes:
# needs aiter alizaidy/qwen35-c16-perf, set ATOM_MOE_PRECOMPUTED_NVR=0 with an older aiter), and the
# MTP draft: replicated fc and embedding (no all-gather / all-reduce), both pre-fc norms in one launch,
# the argmax packs exchanged by one all-reduce, the GDN kv-indptr rebuild in one launch. And the
# sampler's softmax over the 248k vocab split across programs a row (one block a row otherwise).
if [ "$MODE" = tp ]; then
  export ATOM_SPEC_FUSED_VERIFY_SAMPLE=${ATOM_SPEC_FUSED_VERIFY_SAMPLE:-1}
  export ATOM_TP_TOKEN_SYNC_ALLGATHER=${ATOM_TP_TOKEN_SYNC_ALLGATHER:-1}
  export ATOM_FUSED_SIGMOID_GATE=${ATOM_FUSED_SIGMOID_GATE:-1}
  export ATOM_TOKENIZE_CACHE_PER_SEGMENT_NORMALIZER=${ATOM_TOKENIZE_CACHE_PER_SEGMENT_NORMALIZER:-1}
  export AITER_TOPK_FIRST_SMALL_K_TOPP=${AITER_TOPK_FIRST_SMALL_K_TOPP:-1}
  export ATOM_MOE_PRECOMPUTED_NVR=${ATOM_MOE_PRECOMPUTED_NVR:-1}
  export ATOM_MTP_REPLICATED_FC=${ATOM_MTP_REPLICATED_FC:-1}
  export ATOM_QWEN35_MTP_REPLICATED_EMBED=${ATOM_QWEN35_MTP_REPLICATED_EMBED:-1}
  export ATOM_MTP_DUAL_PRE_FC_NORM=${ATOM_MTP_DUAL_PRE_FC_NORM:-1}
  export ATOM_ARGMAX_HEAD_AR_GATHER=${ATOM_ARGMAX_HEAD_AR_GATHER:-1}
  export ATOM_MTP_FUSED_KV_INDPTR=${ATOM_MTP_FUSED_KV_INDPTR:-1}
  export ATOM_SAMPLER_SPLIT_SOFTMAX=${ATOM_SAMPLER_SPLIT_SOFTMAX:-1}
fi
# DP decode step (c128 DP4/EP4): the sampling and tokenizer settings of the TP block above (verify
# and bonus rows sampled in one call, register-resident top-p, the split-row softmax, the per-segment
# normalizer cache): +11% c128 TPS, TPOT 6.70 -> 6.18 ms. The rest of the TP block is TP-only.
if [ "$MODE" = dp ]; then
  export ATOM_SPEC_FUSED_VERIFY_SAMPLE=${ATOM_SPEC_FUSED_VERIFY_SAMPLE:-1}
  export AITER_TOPK_FIRST_SMALL_K_TOPP=${AITER_TOPK_FIRST_SMALL_K_TOPP:-1}
  export ATOM_SAMPLER_SPLIT_SOFTMAX=${ATOM_SAMPLER_SPLIT_SOFTMAX:-1}
  export ATOM_TOKENIZE_CACHE_PER_SEGMENT_NORMALIZER=${ATOM_TOKENIZE_CACHE_PER_SEGMENT_NORMALIZER:-1}
  # DP prefill runs eager and host-bound (~110 ms a step whatever its size): aiter's cached launch
  # paths for the bf16 GEMMs and the FlyDSL MoE kernels (need aiter alizaidy/qwen35-c128-host-attn;
  # ignored by an older aiter), and prefill token graphs on steps every rank prefills, with buckets
  # up to the 16384-token chunk and up to 16 prefill requests a rank recorded
  export AITER_GEMM_FAST_LAUNCH=${AITER_GEMM_FAST_LAUNCH:-1}
  export AITER_FLYDSL_FAST_LAUNCH=${AITER_FLYDSL_FAST_LAUNCH:-1}
  export ATOM_DP_PREFILL_GRAPH=${ATOM_DP_PREFILL_GRAPH:-1}
  export ATOM_GDN_GRAPH_MAX_SEQS=${ATOM_GDN_GRAPH_MAX_SEQS:-16}
  export ATOM_TOKEN_GRAPH_SIZES=${ATOM_TOKEN_GRAPH_SIZES:-256,512,1024,1536,2048,3072,4096,6144,8192,12288,16384}
fi
# keep Triton autotune results across server starts: the GDN chunk kernels otherwise re-tune on the
# first eager multi-request prefill, a ~35 s stall a fresh server
export TRITON_CACHE_AUTOTUNING=${TRITON_CACHE_AUTOTUNING:-1}
# --level 0 (no torch.compile; decode graphs are still captured): at level 3 TP4 hung ~7 min
# into c8 (all ranks stuck behind the sampled-token broadcast, NCCL watchdog at 10 min).
LEVEL=${LEVEL:-0}; KV=${KV:-bf16}; MAX_LEN=${MAX_LEN:-262144}
# 262144 is the model's native context; it fits 98% of the dataset's conversations.
# Prefix caching stays on (the default): every agentic turn resends the whole history.
ARGS=(--model $MODEL --trust-remote-code --host 0.0.0.0 --server-port 8000 -tp 4
  --kv_cache_dtype $KV --gpu-memory-utilization 0.9 --max-model-len $MAX_LEN
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384
  --level $LEVEL --disable_uvicorn_access_log)
[ "$LEVEL" = 3 ] && ARGS+=(--cudagraph-mode FULL)
# decode graph sizes up to MAXSEQ (sequences an engine): dense past 256 too, or a batch of 300
# pads to the next size. CAPS="[...]" overrides.
caps() { python3 -c "
b = [1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256]
b += list(range(320, 1025, 64)) + list(range(1152, 2049, 128)) + list(range(2304, 8193, 256))
print([s for s in b if s < $1] + [$1])" | tr -d ' '; }
case $MODE in
  # MAXSEQ: run_ep.sh sets it from the concurrency (sequences an engine, at most 512)
  tp) MAXSEQ=${MAXSEQ:-64}
      ARGS+=(--max-num-seqs $MAXSEQ --cudagraph-capture-sizes "${CAPS:-$(caps $MAXSEQ)}") ;;
  # as the image's scripts/qwen35/serve_qwen35_dp4ep4.sh (mori v2 dispatch: aiter's flydsl dispatch gives NaN
  # logits with uneven DP token counts) + the DSv4.1 DEP recipe's session affinity: a conversation's
  # turns stay on the DP rank that holds its prefix cache
  dp) MAXSEQ=${MAXSEQ:-16}
      export ATOM_MORI_V2=1 MEGA_DISPATCH=mori MORI_GPU_ARCHS=gfx1250 ATOM_DP_LM_HEAD_MODE=allgather
      export MORI_AMD_SMI_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamd_smi.so.27
      export ATOM_DP_SESSION_AFFINITY=1 AITER_CUSTOM_AR_MAX_SIZE=0
      # after any rank prefills, every rank runs 8 decode-only passes before new prefill: without it, once
      # the KV cache fills the prefill delayer releases prefill every step, ranks prefill and decode in the
      # same lockstep step and each decode waits for the prefill beside it (see the recipe's "Prefill /
      # decode interval"). An ATOM_PREFILL_DECODE_INTERVAL in the environment wins; 0 is the old scheduling.
      export ATOM_PREFILL_DECODE_INTERVAL=${ATOM_PREFILL_DECODE_INTERVAL:-8}
      # the sigmoid gates as one launch each on DP points too: the MoE shared-expert combine (three
      # eager kernels a layer) and the attention output gate (two a full-attention layer)
      export ATOM_FUSED_SIGMOID_GATE=${ATOM_FUSED_SIGMOID_GATE:-1}
      # MHA prefill and decode on aiter's unified attention over the paged cache, with 64-token KV blocks:
      # no KV gather before the prefill FMHA, and the 64-token decode / verify configs (see the recipe's
      # "DP4 + EP4 c2048"). Needs aiter alizaidy/gfx1250-qwen-support from 3d3228c1: before it the 2D
      # kernel took the block-table stride as a constant and recompiled for every prefill batch.
      export ATOM_USE_UNIFIED_ATTN=${ATOM_USE_UNIFIED_ATTN:-1}
      ARGS+=(--enable-dp-attention --enable-expert-parallel --max-num-seqs $MAXSEQ --cudagraph-capture-sizes "${CAPS:-$(caps $MAXSEQ)}"
        --block-size ${BLOCK_SIZE:-64}) ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
# SPEC=<n>: MTP speculative decoding (graphed draft steps); the checkpoint keeps the MTP
# layer BF16 and gfx1250 has no BF16 fused MoE, so its routed experts are quantized to MXFP4 at load
SPEC=${SPEC:-0}
[ "$SPEC" -gt 0 ] && ARGS+=(--method mtp --num-speculative-tokens $SPEC
  --online_quant_config '{"layer_quant_config":{"mtp.layers.*.mlp.experts":"mxfp4"}}')
# TRACE_DIR=<dir>: torch profiler on every rank, armed only; run_ep.sh's TRACE_AT_MIN takes the window
# (traces under $TRACE_DIR/rank_<r>/). Unset, nothing changes.
[ -n "$TRACE_DIR" ] && mkdir -p $TRACE_DIR && ARGS+=(--torch-profiler-dir $TRACE_DIR --mark-trace)
[ -n "$EXTRA_ENV" ] && export $EXTRA_ENV
printf '%q ' python3 -m atom.entrypoints.openai_server "${ARGS[@]}" $EXTRA_ARGS > $OUT/server_cmd.txt; echo >> $OUT/server_cmd.txt
env | grep -E '^(ATOM|AITER|FLYDSL|HIP|HSA|MORI|MEGA|TRITON)_' | sort > $OUT/server_env.txt
# PY_ENTRY: another entry point taking the same arguments (debugging wrappers)
exec python3 -u ${PY_ENTRY:--m atom.entrypoints.openai_server} "${ARGS[@]}" $EXTRA_ARGS > $OUT/server.log 2>&1
