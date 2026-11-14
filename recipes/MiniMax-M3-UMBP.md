# MiniMax-M3 — TP4 with a Mori UMBP CPU tier

MiniMax-M3 (MXFP4) on one node, TP4 + EAGLE3, with the KV offload tier served
by a Mori UMBP standalone server through `umbp_store`. This is the
[MiniMax-M3-Agentic-InferenceX.md](MiniMax-M3-Agentic-InferenceX.md)
configuration with the LMCache CPU tier replaced by UMBP. Read that recipe for
what each server flag does.

Like the LMCache tier there, the CPU tier pays at **high concurrency** (40 and
48 on the AgentX ladder), where the working set outgrows the HBM prefix cache.
Below that, run without offload.

For the UMBP side (requirements, server settings, checks, limitations) see
[UMBP-KV-Offload.md](UMBP-KV-Offload.md). PD with UMBP is not supported yet; the
planned setup is in [mesh/Agentic-MiniMax-M3-UMBP.md](mesh/Agentic-MiniMax-M3-UMBP.md).

```bash
MODEL=/shared_nfs/huggingface_models/amd/MiniMax-M3-MXFP4
DRAFT=/shared_nfs/huggingface_models/Inferact/MiniMax-M3-EAGLE3-GQA
CONC=40                       # or 48
export UMBP_ADDR=unix:///tmp/umbp/server.sock
```

## 0. Container

Any ATOM container whose Mori includes UMBP and ROCm/mori#791 (see
[requirements](UMBP-KV-Offload.md#1-requirements)). No `--privileged` is
needed. Host memory must be lockable, and the IPC namespace shared so the
server and the workers can map each other's memory:

```bash
docker run -d --name m3_umbp \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
  --network host --ipc host --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -v "$MODEL":"$MODEL":ro -v "$DRAFT":"$DRAFT":ro \
  --entrypoint sleep <atom-image> infinity
```

Everything below runs inside the container.

## 1. UMBP server

The ranks run on GPUs `0,1,4,5`, two per socket (same reason as the LMCache
tier: four ranks pinning host memory on one NUMA node starve each other). The
server gets the same GPU list and a pool spread over both NUMA nodes, sized for
the node. 1 TiB matches four ranks × 256 GiB of a per-rank CPU tier.

```bash
UMBP_BIN=$(python3 -c 'import mori.umbp,pathlib;print(pathlib.Path(mori.umbp.__file__).resolve().parents[1]/"umbp_standalone_server")')
mkdir -p /tmp/umbp && rm -f /tmp/umbp/server.sock*
ROCR_VISIBLE_DEVICES=0,1,4,5 \
UMBP_DRAM_CAPACITY=$((1024 * (1 << 30))) \
UMBP_DRAM_NUMA_NODE=0,1 \
UMBP_DRAM_USE_HUGEPAGES=0 \
UMBP_DRAM_PREFAULT_THREADS=16 \
UMBP_SSD_ENABLED=0 \
UMBP_STANDALONE_METRICS_PORT=9877 \
MORI_UMBP_LOG_LEVEL=INFO \
  nohup "$UMBP_BIN" "$UMBP_ADDR" > umbp_server_c${CONC}.log 2>&1 &

# Ready only when a real client connects (prefaulting takes minutes).
until python3 - <<PY 2>/dev/null
import mori.umbp as u
c = u.UMBPConfig(); p = u.UMBPStandaloneProcessConfig()
p.address = "$UMBP_ADDR"; p.auto_start = False; p.startup_timeout_ms = 10000
c.standalone_process = p
u.UMBPClient(c)
PY
do sleep 5; done; echo "umbp ready"
```

## 2. ATOM server

```bash
rm -rf /root/.cache/atom/*
env \
  ROCR_VISIBLE_DEVICES=0,1,4,5 \
  NCCL_IB_DISABLE=1 \
  RCCL_IB_DISABLE=1 \
  PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 \
  AITER_LOG_LEVEL=WARNING \
  AITER_SITUV2_A4W4=1 \
  AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
  AITER_FLYDSL_STAGE2_FP8=1 \
  ATOM_FORCE_ATTN_TRITON=1 \
  ATOM_PA_FLYDSL=1 \
  ATOM_PA_FLYDSL_PLAN=1 \
  ATOM_M3_INDEXER_CP=1 \
  ATOM_PREFIX_CACHE_POLICY=slru \
  ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5 \
  python3 -u -m atom.entrypoints.openai_server \
    --model "$MODEL" --served-model-name "$MODEL" --trust-remote-code \
    --host 0.0.0.0 --server-port 8890 --port 8896 \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 --index-cache-dtype fp8 --block-size 128 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs $((2 * CONC)) \
    --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]' \
    --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","vision_tower","multi_modal_projector","patch_merge_mlp","*block_sparse_moe"]}' \
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}' \
    --enable-prefix-caching \
    --method eagle3 --draft-model "$DRAFT" --num-speculative-tokens 3 \
    --spec-decode-acceptance-length 2.78 \
    --kv-transfer-config '{"kv_connector":"umbp_store","kv_role":"offload","kv_connector_extra_config":{"umbp_store.address":"unix:///tmp/umbp/server.sock"}}' \
  > server_c${CONC}.log 2>&1 &
```

What differs from the LMCache variant:

- no `LMCACHE_*` variables;
- `--kv-transfer-config` names `umbp_store` and the server address;
- the UMBP server from step 1 must be running.

Everything else is the same. `PYTHONHASHSEED=0` and the SLRU HBM prefix-cache
policy stay.

- `--spec-decode-acceptance-length 2.78` (equivalently
  `--spec-decode-acceptance-rate 0.5933`) is for throughput runs only: outputs
  become synthetic. Drop it for the accuracy checks in step 4.
- `ATOM_M3_INDEXER_CP=1` needs `--tensor-parallel-size 4`; it pays from CONC 15
  up (see the InferenceX recipe).
- Leave `--max-model-len` unset (warm-up replays very long contexts).

Wait for the server, and check it is using the GPUs:

```bash
until curl -sf localhost:8890/health >/dev/null; do sleep 10; done
rocm-smi --showmemuse        # GPUs 0,1,4,5 well above 0%
grep -m4 "umbp_store\|UMBP" server_c${CONC}.log
```

## 3. Benchmark

Use the AgentX client exactly as in
[MiniMax-M3-Agentic-InferenceX.md § aiperf](MiniMax-M3-Agentic-InferenceX.md#aiperf):
SemiAnalysis workload `semianalysis_cc_traces_weka_062126`, 3600 s, seed 42,
10 warm-up requests per lane, `--server-metrics http://localhost:8890/metrics`.

One launch per concurrency point. Restart **both** the ATOM server and the
UMBP server between points, so each point starts from an empty pool.

While it runs:

```bash
grep "OFFLOAD-STORE-STATS" server_c${CONC}.log | tail -4   # puts / gets / save_held_chunks per rank
grep "OFFLOAD-LOOKUP-STATS" server_c${CONC}.log | tail -1  # uneven_lookups should stay low
curl -s localhost:9877/metrics | grep -i -E "dram|evict|commit" | head
```

Read the CPU tier's contribution from the realised hit (prompt tokens actually
prefilled), not from the engine's `Prefix cache hit rate`, which counts the
GPU tier only.

## 4. Accuracy

Restart the server **without** `--spec-decode-acceptance-length`, then:

```bash
lm_eval --model local-chat-completions --apply_chat_template --fewshot_as_multiturn \
  --model_args model=$MODEL,tokenizer=$MODEL,base_url=http://localhost:8890/v1/chat/completions,num_concurrent=64,max_retries=3,max_gen_toks=16384,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k --num_fewshot 5 --seed 0
```

Compare against the same server with
`--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'`
(plus the `LMCACHE_*` variables of the InferenceX recipe).

GSM8K prompts are short and mostly hit HBM, so they barely exercise the load
path. To check KV read-back as well, follow
[UMBP-KV-Offload.md § Check that it works](UMBP-KV-Offload.md#4-check-that-it-works):
push long prompts out of HBM, resend them, and confirm they load from UMBP
and still answer correctly.

## Stop

```bash
pkill -f 'atom.entrypoints.openai_server' || true
pkill -f umbp_standalone_server || true
```
