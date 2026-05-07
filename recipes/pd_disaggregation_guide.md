# PD Disaggregation with Mooncake (RDMA)

Prefill-Decode disaggregation splits inference into two stages on separate nodes:
- **Producer** (prefill): runs prompt prefill, pushes KV cache via RDMA
- **Consumer** (decode): receives KV cache, runs autoregressive decode

## Prerequisites

- Two nodes with AMD MI300X GPUs (8 GPUs each for TP=8)
- RDMA network connectivity between nodes (RoCE or InfiniBand)
- Mooncake package installed (`pip install mooncake`)
- Producer and consumer should be in the **same network partition** for best accuracy

## Quick Start

### Step 0: Find local IP

On each node, find the network IP (not loopback):

```bash
export LOCAL_IP=$(ip addr show | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d/ -f1 | head -1)
echo "Local IP: ${LOCAL_IP}"
```

### Step 1: Start Proxy (on producer node)

The proxy handles routing between producer and consumer:

```bash
python -m atom.kv_transfer.disaggregation.proxy --port 10001
```

### Step 2: Start Producer (prefill node)

```bash
ATOM_DISABLE_MMAP=true \
NCCL_SOCKET_IFNAME=lo \
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-R1/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8003 \
  --kv-transfer-config '{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "proxy_ip": "'"${LOCAL_IP}"'",
    "proxy_ping_port": 36367,
    "http_port": 8003
  }' \
  2>&1 | tee producer.log
```

### Step 3: Start Consumer (decode node)

Replace `PRODUCER_IP` with the producer node's IP:

```bash
export PRODUCER_IP=<producer-node-ip>

ATOM_DISABLE_MMAP=true \
NCCL_SOCKET_IFNAME=lo \
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-R1/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8004 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "proxy_ip": "'"${PRODUCER_IP}"'",
    "proxy_ping_port": 36367,
    "http_port": 8004
  }' \
  2>&1 | tee consumer.log
```

### Step 4: Validate Accuracy

Run GSM8K evaluation against the consumer endpoint:

```bash
lm_eval --model local-chat-completions \
  --model_args "model=DeepSeek-R1,base_url=http://${CONSUMER_IP}:8004/v1,tokenizer_backend=huggingface,pretrained=/data/models/DeepSeek-R1/" \
  --tasks gsm8k_cot \
  --batch_size 1 \
  --limit 100 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --predict_only \
  --log_samples \
  --output_path results/ \
  --gen_kwargs "max_tokens=8192,temperature=0.6"
```

Expected accuracy: ~0.95-0.96 (matching non-PD baseline).
```
ewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.96|±  | 0.028|
|     |       |strict-match    |     5|exact_match|↑  | 0.96|±  | 0.028|
```