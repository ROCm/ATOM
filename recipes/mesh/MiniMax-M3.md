# MiniMax-M3 PD Disaggregation with atomesh

PD-disaggregated serving for MiniMax-M3 (MXFP4 and MXFP8) using the ATOM
native backend, Mooncake RDMA KV transfer, and atomesh routing. Covers four
topologies, each with optional EAGLE3 speculative decoding.

## Prerequisites

- AMD MI355X GPUs (4 GPUs per instance, TP=4)
- RDMA network connectivity (RoCE or InfiniBand) for KV cache transfer
- Model weights accessible at the same path on all nodes
- Checkpoints:
  - MXFP4: [`amd/MiniMax-M3-MXFP4`](https://huggingface.co/amd/MiniMax-M3-MXFP4)
  - MXFP8: [`MiniMaxAI/MiniMax-M3-MXFP8`](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8)
  - EAGLE3 draft: [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3)

## Quick Reference

| Topology | Nodes | GPUs | Prefill flags | Decode flags | Typical CONC |
|----------|------:|-----:|---------------|--------------|-------------|
| 1P+1D (single-node) | 1 | 8 | TP=4 | TP=4 | 1–256 |
| 1P+1D + EAGLE3 | 1 | 8 | TP=4, eagle3 | TP=4, eagle3 | 1–256 |
| 2P+1D DPA | 3 | 4 each | TP=4, DPA+TBO | TP=4, DPA | 256–1024 |
| 2P+1D DPA + EAGLE3 | 3 | 4 each | TP=4, DPA+TBO, eagle3 | TP=4, DPA, eagle3 | 256–1024 |

## Environment Setup

Start container(s) with the RDMA-aware docker script:

```bash
DOCKER_IMAGE=rocm/atom-dev:latest bash atom/mesh/scripts/docker_start.sh
docker exec -it atom_mesh bash
```

For multi-node topologies, start a container on **each node** (separate
containers avoid ATOM port 29500 conflicts).

All commands below run **inside the container**.

### Common Env Vars

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')
export PYTHONUNBUFFERED=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
rm -rf /root/.cache/atom/* 2>/dev/null || true
```

## Online Quant Config Reference

The `--online_quant_config` flag converts attention and dense MLP linear weights
to PTPC FP8 at load time. The `exclude_layer` list differs by model variant and
parallelism mode:

| Mode | Model | `exclude_layer` |
|------|-------|-----------------|
| TP-only (1P+1D) | MXFP4 | `"lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"` |
| TP-only (1P+1D) | MXFP8 | same as MXFP4 above |
| DPA (2P+1D) | MXFP4 | same as MXFP4 above |
| DPA (2P+1D) | MXFP8 | `"lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*.gate.*", "*.block_sparse_moe.experts*"` |

The DPA + MXFP8 variant uses a broader exclude pattern (`*.gate.*`,
`*.block_sparse_moe.experts*`) to avoid quantizing the MoE gate and expert
weights that are already in FP8 format.

---

## 1P+1D — Single-Node (MXFP4)

Prefill on GPU 0-3, decode on GPU 4-7, router on port 8000.

### Prefill Server (GPU 0-3)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

python3 -m atom.entrypoints.openai_server \
    --model amd/MiniMax-M3-MXFP4 \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --block-size 128 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}' \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    2>&1 | tee prefill.log
```

### Decode Server (GPU 4-7)

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model amd/MiniMax-M3-MXFP4 \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --block-size 128 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]" \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    2>&1 | tee decode.log
```

### Router

```bash
atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${NODE_IP}:8010" \
    --decode  "http://${NODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker \
    --prometheus-port 29100
```

### Using MXFP8 Instead

Replace the model path and keep the same `--online_quant_config`:

```bash
--model MiniMaxAI/MiniMax-M3-MXFP8
```

For 1P+1D (non-DPA) mode, both MXFP4 and MXFP8 use the same
`"*block_sparse_moe"` exclude pattern.

### Adding EAGLE3

Append these three flags to **both** the prefill and decode server commands:

```
--method eagle3 \
--draft-model Inferact/MiniMax-M3-EAGLE3 \
--num-speculative-tokens 3
```

The router command stays the same.

---

## 2P+1D DPA — Multi-Node (MXFP4)

Two prefill instances + one decode instance across 3 nodes. Each instance uses
TP=4 with Data-Parallel Attention (DPA). Prefill instances additionally enable
Token-Budget Optimization (TBO).

### Topology

```
Node 0 (prefill-1)  ─┐
                      ├──▶  atomesh router (:8000) ──▶ Client
Node 1 (prefill-2)  ─┤
                      │
Node 2 (decode)     ──┘
```

### Prefill Server (Node 0 — prefill-1)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

python3 -m atom.entrypoints.openai_server \
    --model amd/MiniMax-M3-MXFP4 \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --enable-dp-attention \
    --enable-tbo prefill \
    --block-size 128 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}' \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    2>&1 | tee prefill.log
```

Repeat on **Node 1 (prefill-2)** with the same command.

### Decode Server (Node 2)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

python3 -m atom.entrypoints.openai_server \
    --model amd/MiniMax-M3-MXFP4 \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --enable-dp-attention \
    --block-size 128 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 32768 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]" \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    2>&1 | tee decode.log
```

Key differences from prefill:
- `kv_role: kv_consumer`
- `--enable-dp-attention` without `--enable-tbo` (TBO is prefill-only)
- `--max-num-seqs 1024` — higher batch capacity for decode throughput
- `--cudagraph-capture-sizes` — pre-captures graphs for common batch sizes

### Router

```bash
export PREFILL_IP_1=<prefill-node-0-ip>
export PREFILL_IP_2=<prefill-node-1-ip>
export DECODE_IP=<decode-node-2-ip>

atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP_1}:8010" \
    --prefill "http://${PREFILL_IP_2}:8010" \
    --decode  "http://${DECODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker \
    --prometheus-port 29100
```

Two `--prefill` flags — atomesh round-robins requests across both prefill
instances.

### Using MXFP8 Instead (DPA)

Replace the model path **and** change the `--online_quant_config` exclude list:

```bash
--model MiniMaxAI/MiniMax-M3-MXFP8 \
--online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*.gate.*", "*.block_sparse_moe.experts*"]}'
```

This applies to **both** prefill and decode servers.

### Adding EAGLE3

Append these three flags to **both** the prefill and decode server commands:

```
--method eagle3 \
--draft-model Inferact/MiniMax-M3-EAGLE3 \
--num-speculative-tokens 3
```

The router command stays the same.

---

## Verify

After the router is up, send a test request:

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"amd/MiniMax-M3-MXFP4","prompt":"Hello","max_tokens":32,"temperature":0}'
```

## GSM8K Accuracy (via Router)

```bash
lm_eval --model local-chat-completions \
    --model_args "model=amd/MiniMax-M3-MXFP4,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=3,max_gen_toks=16384" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 65 \
    --apply_chat_template \
    --fewshot_as_multiturn
```

For 2P+1D DPA topologies, increase `num_concurrent` to `256` or higher.

## Serving Benchmark (via Router)

```bash
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
    --model=amd/MiniMax-M3-MXFP4 \
    --backend=vllm \
    --base-url=http://127.0.0.1:8000 \
    --dataset-name=random \
    --random-input-len="${ISL}" \
    --random-output-len="${OSL}" \
    --random-range-ratio=0.8 \
    --num-prompts=$(( CONC * 10 )) \
    --max-concurrency="${CONC}" \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics="ttft,tpot,itl,e2el"
```

For 2P+1D DPA topologies, sweep higher concurrency: `CONC=256,512,768,1024`.
