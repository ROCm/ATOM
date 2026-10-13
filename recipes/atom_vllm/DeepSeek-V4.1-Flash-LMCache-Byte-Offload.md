# DeepSeek-V4.1-Flash — LMCache byte offload on the vLLM plugin

vLLM 0.31, TP4, CUDA graphs on.

V4.1's pool holds two currencies. PAGE (the compressed history) is a function
of the tokens, so a block-table hit restores it. STATE (the per-request window
ring, the compressor's partial group, the Engram cursor) is not. The connector
owns a tier for STATE and caps every prefix hit to a boundary whose state it
stored, which is what makes prefix caching admissible for this model.

Without the connector, prefix caching is refused at startup.

## Serving

```bash
export PYTHONPATH=/workspace/ATOM-dsv41-lmc031
export HIP_VISIBLE_DEVICES=0,1,2,3
export ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1     # ATOM's own graph gate
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=64         # GiB of CPU tier
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_CACHE_POLICY=ATOM_SLRU

vllm serve /data/amd_int/models/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 127.0.0.1 --port 8021 \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector",
    "kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector",
    "kv_role":"kv_both","kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{"atom.offload.backend":"inproc",
      "atom.offload.v41.state_interval":4096}}'
```

Three settings silently disable the tier if changed:
`kv_load_failure_policy` other than `recompute` turns a tier miss into a user
500; `--no-enable-prefix-caching` leaves nothing to reuse; and on the `mp`
backend the tier server needs `--no-l1-use-lazy` or `--shm-name` is ignored
and transfers fall back to ZMQ pickle.

The OFF arm is the same command without `--kv-transfer-config` and with
`--no-enable-prefix-caching` — V4.1 refuses prefix caching without the
connector, so the pair is "reuse vs none", not "tier vs HBM".

## Accuracy

`tools/kv_offload_accuracy_arms.py` — three arms over one document of invented
facts, each asking for one of them by index. It exits non-zero when the
experiment did not happen, not when the score is low.

```bash
python3 tools/kv_offload_accuracy_arms.py 8021 16 64 96 DeepSeek-V4.1-Flash
```

- `blind` — no document. Must collapse, or the questions did not need the prefix.
- `cold` — the document with a per-question nonce at its HEAD, so every prefix
  hash is unique. Must score high and take ~0% from the tier.
- `warm` — the shared document, already in the tier. The measurement.

Discriminating power, measured by zeroing restored PAGE blocks while the cold
arm stayed at 32/32: 25% of blocks → 15/16, 100% → 1/16.

### Result

```
ARM blind correct  0/16   tier 0.0%
ARM cold  correct 16/16   tier 0.0%
ARM warm  correct 16/16   tier 95.2%
VALID. stale-cursor refusals 0, cursor_mismatch 0
```

gsm8k cannot be used here: 5-shot re-samples exemplars per question, so no two
requests share a prefix and the tier supplies nothing (measured: 1,150,678
queries, 0 hits).

## Throughput

`dsv41_lmc/perf_pairs.sh` — three ON/OFF pairs, alternating, three rounds each.
The container is restarted between arms and the port checked silent, so an arm
cannot measure the previous server. Reuse is designed rather than raced: 12
documents primed one at a time before the window, every request in the window
carrying a unique tail, `ignore_eos` with fixed `max_tokens` so decode work is
identical across arms.

```bash
PAIRS=3 ROUNDS=3 bash dsv41_lmc/perf_pairs.sh
```

### Result

Steady state (rounds 2-3, medians), both arms with graphs on:

| | ON | OFF |
|---|---|---|
| throughput | **12.778 req/s** | 5.112 req/s |
| ratio | **2.500x** | |
| p50 latency | 1.87 s | 4.67 s |
| tier supply (of prompt tokens) | 95.4% | 0.0% |
| local HBM hit | 0.0% | 0.0% |
| within-arm spread | 1.17% | 0.30% |

The OFF arm held 5.003-5.115 req/s across all nine rounds, so 2.5x is far
outside the noise.

Read 95.4% as the ceiling, not a general speed-up: this workload is built for
reuse, and the gain tracks how much of a prompt the tier supplies.

One arm is excluded: pair 1's ON arm stopped at 192 of 240 requests and its
remaining rounds produced nothing. The engine did not crash -- the log shows
`[shutdown] API server: shutdown triggered` with the engine healthy -- and
what triggered it is not known. The exclusion was not pre-registered. The six
surviving ON rounds come from two independent pairs that agree
(12.69/12.80 and 12.84/12.76).

## State checkpoint images: present, off, not load-bearing

`ATOM_V41_STATE_IMAGES=1` registers STATE as a `MambaSpec` group so vLLM
allocates and prefix-caches it. It does not work and the gate stays shut.

vLLM's align-mode prefix caching requires the backend to have left recoverable
state at each complete block boundary -- registering a cached block IS that
claim -- and a V4.1 forward leaves state only at the end of the query, while
`checkpoint_offsets` asks for an interior offset. Zeroing
`num_prefill_checkpoint_blocks` does not opt out: align mode caches the last
full block anyway and the group still answers that boundary (measured with a
probe on vLLM's own finder, `MambaManager ... max=1280 -> hit=1280`).

Reviving it means exporting at a given offset through
`MambaPrefillCheckpointExporter`, which for V4.1 means splitting
`engram_compress`'s plan there -- a model-op change, not a flag.

With both the images and DSpark on, #2494's `_build_v41_proxy_draft_groups`
splits groups proxy-vs-rest and merges the rest, which asserts one spec type;
"rest" would hold a `MambaSpec` and a `SlidingWindowMLASpec`. Unreachable while
the images default off.
