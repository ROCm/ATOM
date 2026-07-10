# Prefill Context Parallel (PCP) Guide

For long-context serving, prefill is bottlenecked on the **sequence dimension**:
the DSA indexer scores every query against all history, and that cost grows with
sequence length and is replicated across Tensor-Parallel (TP) ranks. Plain TP
shards weights/heads/experts, not tokens, so it cannot reduce this cost.

**Prefill Context Parallel (PCP)** is an independent parallelism dimension that
splits the **prefill token sequence** across the PCP process group, so each GPU
processes only `1/pcp` of the tokens during prefill. This cuts the per-GPU
prefill work (and the indexer's sequence-length cost) to `1/pcp`, lowering TTFT
and raising long-prefill throughput. Decode is left unchanged. PCP composes with
TP and Expert Parallelism (EP): the total world size is `world = tp × pcp`.

```
  pcp = 2, prefill tokens: 0 1 2 3 4 5
    GPU (pcp rank 0):  0   2   4      ← each GPU processes 1/pcp of the tokens
    GPU (pcp rank 1):    1   3   5
  Full KV is kept on every rank; the 1/pcp outputs are all-gathered back to the
  full sequence before the LM head. Decode runs as usual (no split).
```

> **Model support.** PCP currently supports **DeepSeek-V4** only. Support for
> more models will be added incrementally.

## When to use PCP

- **Best fit**: long-context / large-prompt prefill on DeepSeek-V4, where prefill
  TTFT dominates.
- **Combine with**: `--enable-tbo` (prefill) to overlap the MoE communication PCP
  introduces (see [Overlapping communication with TBO](#overlapping-communication-with-tbo)).
  TBO is only usable with `ATOM_PCP_MOE_MERGE=1` **and `-tp 1`** (see
  [Constraints & Compatibility](#constraints--compatibility)).
- **Requires**: `world = tp × pcp` GPUs, e.g. `-tp 4 -pcp 2` on 8 GPUs.
- **Little benefit / avoid**: decode-heavy or short-prompt workloads. PCP only
  shards **prefill** tokens to lower TTFT on long sequences; decode is left
  unsharded and runs redundantly across the PCP ranks. For long-decode
  (large `output_len`) workloads PCP can **hurt TPOT** — do not enable it there;
  use TP/EP as usual.

## Quick Reference

| Flag / Variable | Default | Purpose |
|-----------------|---------|---------|
| `-pcp N` / `--prefill-context-parallel-size N` | `1` | Enable PCP with size `N` (`world = tp × pcp`) |
| `ATOM_PCP_MOE_MERGE` | `1` | Whether to shard MoE across the PCP ranks too |
| `--enable-tbo [prefill\|all]` | off | Overlap compute with PCP communication. With PCP, only prefill TBO is supported, and only when `ATOM_PCP_MOE_MERGE=1` and `-tp 1` |
| `--no-enable_chunked_prefill` | chunked on | Disable chunked prefill. Recommended with PCP (see [Tuning for long sequences](#tuning-for-long-sequences)) |
| `--max-num-batched-tokens N` | `16384` | Max tokens scheduled per step. Raise (e.g. `131072`) for long-sequence PCP so a full long prompt is prefilled in one step |

| Goal (8 GPUs) | Command |
|------|---------|
| Long-context prefill | `-tp 4 -pcp 2` |
| Long-context prefill + overlap | `-tp 1 -pcp 8 --enable-tbo` |
| Disable PCP (baseline) | `-tp 8 -pcp 1` |

> **TBO requires `-tp 1`.** Two-batch overlap is only supported with
> `ATOM_PCP_MOE_MERGE=1` **and TP=1** (all GPUs go to PCP, e.g. `-tp 1 -pcp 8`).

## CLI usage

```bash
-pcp N                          # or --prefill-context-parallel-size N; world = tp × pcp
--enable-tbo                    # prefill-only TBO overlap; requires ATOM_PCP_MOE_MERGE=1 and -tp 1 (prefill only supported with PCP)
```

```bash
ATOM_PCP_MOE_MERGE=1            # default: shard MoE across PCP ranks (gather/scatter)
ATOM_PCP_MOE_MERGE=0            # run MoE per-rank on its 1/pcp shard, no extra MoE comm
```

`ATOM_PCP_MOE_MERGE` only has an effect when PCP is enabled (`pcp > 1`):

| Value | MoE behaviour | When to use |
|---|---|---|
| `1` (default, recommended) | PCP is folded into the MoE tensor/expert sharding, so MoE weights are also sharded across PCP ranks. Lowers per-GPU MoE weight memory, at the cost of one hidden gather/scatter per MoE layer (which TBO overlaps). | Most deployments |
| `0` | Each GPU runs MoE independently on its `1/pcp` shard with no extra MoE communication; MoE weights are replicated across PCP ranks. | Avoid extra MoE comm and have memory headroom for replicated MoE weights |

## Launching server

### DeepSeek-V4: TP4 + PCP2 (8 GPUs)

```bash
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4 \
    -tp 4 -pcp 2 \
    --kv_cache_dtype fp8
```

### DeepSeek-V4: TP1 + PCP8 + prefill TBO overlap (8 GPUs)

TBO requires `ATOM_PCP_MOE_MERGE=1` (the default) **and `-tp 1`** — put every GPU
into PCP.

```bash
ATOM_PCP_MOE_MERGE=1 \
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4 \
    -tp 1 -pcp 8 \
    --enable-tbo \
    --kv_cache_dtype fp8
```

Tips:
- `-tp 8 -pcp 1` (or omitting `-pcp`) disables PCP and serves as the baseline.
- `--enable-tbo` overlaps the MoE communication introduced by
  `ATOM_PCP_MOE_MERGE=1`. It only helps in that mode: with `ATOM_PCP_MOE_MERGE=0`
  there is no MoE communication to overlap, so TBO is auto-disabled (a warning is
  logged).
- `--enable-tbo` additionally requires `-tp 1`; with `tp > 1` it **may hang** under
  long-sequence / high-concurrency workloads (not yet supported). Give all GPUs to
  PCP instead, e.g. `-tp 1 -pcp 8`.
- Under PCP, TBO uses a **request-boundary split** (each micro-batch is a whole
  subset of requests) — the non-default TBO split mode — instead of the
  token-midpoint split used without PCP. `ATOM_TBO_PREFILL_TOKEN_SPLIT`
  therefore has no effect when PCP is enabled.
- `--torch-profiler-dir ./log` can be added to collect traces for performance
  analysis.

## Tuning for long sequences

PCP targets long-prefill TTFT, but the default scheduler settings work against it. When serving long prompts:

- **Disable chunked prefill** (`--no-enable_chunked_prefill`), or enlarge the chunk (`--attn_prefill_chunk_size`). It is on by default and splits a long prompt across steps, so PCP gets fewer tokens to shard per step and TBO's request-boundary split (needs ≥2 whole sequences per step) falls back to the non-overlapped path.
- **Raise `--max-num-batched-tokens`** to `131072` (≥ `input_len`; ≥ `2 × input_len` for TBO's balanced split). The default `16384` forces a long prompt across multiple steps instead of prefilling it in one.
- **Avoid PCP for long-decode workloads.** PCP shards only prefill; decode runs unsharded. A large `output_len` is decode-dominated, where PCP adds no benefit and can raise TPOT — use plain TP/EP.

Example (long-context prefill, chunked off, larger batch budget):

```bash
ATOM_PCP_MOE_MERGE=1 \
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4 \
    -tp 1 -pcp 8 --enable-tbo \
    --no-enable_chunked_prefill \
    --max-num-batched-tokens 131072 \
    --kv_cache_dtype fp8
```

## Performance baseline

Benchmark against a running server with a long input length (PCP targets prefill):

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-V4 --backend=vllm --base-url=http://localhost:7777 \
  --dataset-name=random \
  --random-input-len=32768 --random-output-len=512 \
  --num-prompts=128 --max-concurrency=64 \
  --request-rate=inf --ignore-eos
```

Compare `-tp 4 -pcp 2` against the `-tp 8` baseline and watch **Mean TTFT** and
output throughput; the gap widens as `--random-input-len` grows.

> PCP was introduced in [ROCm/ATOM#1220](https://github.com/ROCm/ATOM/pull/1220),
> which reported, on 8×MI308 for `-tp 4 -pcp 2` vs `-tp 8`, a **35–43%** Mean-TTFT
> reduction and up to **~49%** higher throughput on long prefill. Actual gains
> depend on model, sequence length, and hardware.

### PCP + TBO (prefill overlap)

TBO's benefit is **hardware-dependent**:

- **MI308**: current testing shows PCP + TBO gives **almost no gain** over PCP
  alone — the overlap does not meaningfully hide the MoE communication on this
  hardware. Prefer plain PCP (`-tp 4 -pcp 2`) here.
- **MI355**: PCP + TBO **does** deliver a speedup. Enable it with
  `ATOM_PCP_MOE_MERGE=1 -tp 1 -pcp 8 --enable-tbo`.

Because TBO requires `-tp 1`, compare TBO configs against the matching `-tp 1`
PCP baseline (not the `-tp 4 -pcp 2` numbers above).

## Constraints & Compatibility

| Constraint | Notes |
|-----------|-------|
| Models | DeepSeek-V4 only (more coming) |
| World size | `tp × pcp ≤ 8`; multi-node not yet validated |
| PCP + DP-attention | Not supported (raises at startup) |
| PCP + TBO decode (`--enable-tbo all`) | Not supported (raises at startup); use `--enable-tbo` prefill-only |
| `ATOM_PCP_MOE_MERGE=0` + `--enable-tbo` | TBO auto-disabled (warning logged) |
| PCP + TBO with `tp > 1` | **Not supported.** May hang under long-sequence / high-concurrency workloads. TBO requires `-tp 1` (all GPUs in PCP, e.g. `-tp 1 -pcp 8`) |

PCP + TBO **prefill** is supported only with `ATOM_PCP_MOE_MERGE=1` **and
`-tp 1`**. Decode is unchanged by PCP in all configurations.

## How it works

1. At the start of the prefill forward, the token sequence is split round-robin
   across the PCP ranks (token `i` → rank `i % pcp`), padded so the count divides
   evenly.
2. Each rank runs attention / indexer / compressor on its `1/pcp` token shard.
   The full KV is kept on every rank (all-gathered as needed), so the attention
   kernels are unchanged.
3. MoE either runs on the local `1/pcp` shard (`ATOM_PCP_MOE_MERGE=0`) or gathers
   to the full sequence and scatters back (`=1`, default).
4. After the final layer, the `1/pcp` hidden states are all-gathered back to the
   full sequence, the original token order is restored, and the LM head runs.
5. Decode is untouched: every rank keeps the full KV and runs normally, so PCP
   adds no decode-time cost.

## Source Files

| File | Description |
|------|-------------|
| `atom/model_engine/arg_utils.py` | `--prefill-context-parallel-size`, `--enable-tbo` CLI |
| `atom/utils/envs.py` | `ATOM_PCP_MOE_MERGE` |
| `atom/distributed/pcp_utils.py` | PCP communication and helper primitives |
| `atom/models/deepseek_v4.py` | DeepSeek-V4 PCP forward path and MoE handling |
| `atom/model_ops/attentions/deepseek_v4_attn.py` | PCP attention metadata (incl. PCP + TBO prefill) |
| `atom/model_engine/model_runner.py` | PCP token split and PCP + TBO grouping |
| `atom/model_engine/llm_engine.py` | PCP / TBO / DP-attention validation |
