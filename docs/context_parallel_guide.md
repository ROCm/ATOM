# Context Parallel Guide (PCP & DCP)

ATOM has two independent context-parallel dimensions that shard the **token
sequence** instead of weights/heads:

- **[Prefill Context Parallel (PCP)](#prefill-context-parallel-pcp-guide)** —
  shards the **prefill** token sequence to lower long-prefill TTFT. Adds GPUs
  (`world = tp × pcp`). Decode is untouched.
- **[Decode Context Parallel (DCP)](#decode-context-parallel-dcp-guide)** —
  shards the **KV cache** (and decode attention) along the sequence across
  existing TP GPUs (`world = tp`, `tp % dcp == 0`) to cut per-GPU KV memory and
  decode attention cost. Prefill KV writes are sharded too.

They target opposite phases and can be used independently.

---

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

---

# Decode Context Parallel (DCP) Guide

For long-context / large-batch decode, the bottleneck is the **KV cache**. With
**MLA** (Multi-head Latent Attention, e.g. DeepSeek) the KV is a single latent
head, so Tensor Parallel (TP) cannot shard it — every TP rank holds the **full**
KV cache and repeats the full decode attention. KV memory therefore does not
shrink with TP, capping context length and batch size.

**Decode Context Parallel (DCP)** shards the KV cache along the **sequence
dimension** across a DCP process group: token `i` lives on rank `i % dcp`, so
each GPU stores only `1/dcp` of the KV and does `1/dcp` of the decode attention
work. This lowers per-GPU KV memory (longer context / larger batch) and decode
attention cost. Unlike PCP, **DCP does not add GPUs** — it sub-partitions the
existing TP ranks, so `world = tp` and `tp` must be divisible by `dcp`.

```
  dcp = 2, KV tokens: 0 1 2 3 4 5
    GPU (dcp rank 0):  0   2   4      ← each GPU keeps 1/dcp of the KV cache
    GPU (dcp rank 1):    1   3   5
  Decode: all-gather Q (all heads) → each rank runs attention over its local
  1/dcp KV (returns LSE) → LSE-correct + reduce-scatter combines the partial
  outputs back to each rank's head slice.
```

> **Model support.** DCP supports **dense MLA** models (e.g. DeepSeek-V3 / R1),
> **DeepSeek Sparse Attention (DSA / sparse MLA)** models (e.g.
> DeepSeek-V3.2-Exp), and **hybrid KDA + MLA** models (**Kimi-K3**), covering
> both prefill and decode. For **dense MLA**, both the ATOM server and the
> vllm-atom plugin paths are validated. For **sparse MLA (DSA)** and **Kimi-K3**,
> the ATOM server path is validated; the **vllm-atom plugin path is not yet
> verified**.
>
> **Not yet supported: DSA + DCP + MTP.** Speculative decode (MTP, `q > 1`) is
> only available for dense MLA (gfx950); combining it with sparse attention under
> DCP is rejected at runtime — see the DCP Constraints & Compatibility table below.

## When to use DCP

- **Best fit**: long-context / large-batch **decode** on MLA models — dense MLA
  (V3 / R1), sparse MLA / DSA (V3.2-Exp), and hybrid KDA + MLA (Kimi-K3) — where
  the full-replicated KV cache limits context length or batch size.
- **Requires**: `tp % dcp == 0`; `world = tp` (DCP reuses TP GPUs, it does *not*
  add any). E.g. `-tp 8 -dcp 8` or `-tp 8 -dcp 2` on 8 GPUs.
- **Composes with**: prefix caching and chunked prefill (both supported under
  DCP); `--kv-cache-dtype fp8` (per-tensor scale); **speculative decode** —
  **MTP** (`--method mtp`, `num_speculative_tokens` 1–3; **gfx950 only**) on
  dense MLA, and **DSpark** (`--method dspark`) on Kimi-K3.
- **Little benefit / avoid**: short-context, KV-memory-plentiful workloads —
  DCP adds per-step decode communication (Q all-gather + output reduce-scatter)
  that isn't worth it when KV memory isn't the constraint.

## Quick Reference

| Flag / Variable | Default | Purpose |
|-----------------|---------|---------|
| `-dcp N` / `--decode-context-parallel-size N` | `1` | Enable DCP with size `N`. `world = tp`; requires `tp % N == 0` |
| `--dcp-config '{...}'` | see below | JSON dict of the four DCP knobs (`interleave_size`, `enable_query_replication`, `enable_project_before_merge`, `comm_backend`). Unknown keys raise. See [`--dcp-config`](#--dcp-config-the-four-dcp-knobs) |
| `--kv-cache-dtype fp8` | `auto` | Supported with DCP (per-tensor scale). `auto`/`bf16` also fine |
| `--enable_prefix_caching` | off | Supported with DCP |
| `--enable_chunked_prefill` / `--no-enable_chunked_prefill` | on | Chunked prefill is supported with DCP; on by default |

| Goal (8 GPUs) | Command |
|------|---------|
| Max KV capacity (all ranks in one DCP group) | `-tp 8 -dcp 8` |
| Partial DCP | `-tp 8 -dcp 2` (four DCP groups of 2) |
| Disable DCP (baseline) | `-tp 8 -dcp 1` (or omit `-dcp`) |

## CLI usage

```bash
-dcp N                          # or --decode-context-parallel-size N; world = tp, tp % N == 0
--dcp-config '{"..."}'          # JSON dict of DCP tuning knobs; see below
```

## `--dcp-config`: the four DCP knobs

Everything DCP-specific beyond `-dcp N` lives in one JSON dict rather than four
top-level flags. It is parsed straight into `DCPConfig` (`atom/config.py`), and
**unknown keys raise** so a typo fails at startup instead of silently doing
nothing.

```bash
--dcp-config '{"interleave_size": 16, "enable_query_replication": true}'
```

| Key | Type | Default | What it does |
|-----|------|---------|--------------|
| `interleave_size` | int | `1` | KV-cache interleave granularity `S`: token `i` lives on DCP rank `(i // S) % W`. `1` = token-level round-robin |
| `enable_query_replication` | bool | `true` | Replicate the MLA query projection across the DCP group at load time, so each rank produces the whole group's head set locally and the per-step decode AllGather Q disappears |
| `enable_project_before_merge` | bool | `true` | Apply the V up-projection *before* the DCP output merge, so the merge exchanges `v_head_dim` per head instead of `kv_lora_rank` |
| `comm_backend` | str | `"a2a"` | Which collective pattern merges the per-rank partial attention: `"a2a"` or `"ag_rs"` |

> **Three of the four default to the new behaviour.** A control run that wants
> the old path must say so **explicitly** — passing nothing re-runs the new
> configuration, which silently turns an A/B into a no-op:
>
> ```bash
> --dcp-config '{"enable_query_replication": false, "enable_project_before_merge": false, "comm_backend": "ag_rs"}'
> ```

### `interleave_size`

Block-level interleave (`S > 1`) keeps `S` consecutive tokens on one rank
instead of striping every token. Constraints, all asserted at startup:

- `1 <= interleave_size <= kv_cache_block_size`
- `kv_cache_block_size % interleave_size == 0` when `S > 1` — the local-index
  math `(i // (S*W)) * S + i % S` depends on it
- `S > 1` requires `-dcp > 1`, and is **incompatible with speculative decode**
  (the `qlen>1` verify cprr MLA kernel assumes token-level interleave)

### `enable_query_replication` (QREP)

Removes one collective per decode step by paying for it once at load time. It is
**auto-disabled with a warning** (not an error) when the combination is not
wired, so it can default on without breaking mixed runs:

| Condition | Reason logged |
|-----------|---------------|
| `-dcp 1` | `decode_context_parallel_size <= 1 (no DCP group)` — no AllGather Q to remove |
| speculative decode (MTP / eagle3 / DSpark) | `speculative decode (qlen>1 cprr path)` |
| fp4 (`ATOM_USE_TRITON_MXFP4_BMM=1`) | `fp4 (mxfp4) BMM weights` — different scale structure |

Check the server log for `query_replication disabled: ...` to see whether it
actually took effect; the flag being `true` is not the same as QREP running.

Note it costs KV budget: replicating the query heads shrinks the KV pool by
roughly 5% (measured on DeepSeek-R1 tp8/dcp8: 235 016 → 221 020 blocks).

### `enable_project_before_merge` (PBM)

The merge is a per-(token, head) scalar weighting plus a cross-rank sum, and
`W_V` is a per-head linear map, so the two commute — the merged output is the
same either way, but merging *after* the projection moves `v_head_dim` per head
instead of `kv_lora_rank`. The payload ratio is `kv_lora_rank / v_head_dim`:

| Model | Ratio |
|-------|-------|
| DeepSeek-R1 / V3.2 | 4x |
| GLM-5.2 (`v_head_dim=256`) | 2x |

Covers both decode and sparse prefill. Costs a DCP-group-wide copy of `W_V`
(gathered at load time). Auto-disabled for fp4 and for `-dcp 1`.

### `comm_backend`

Both backends compute the same thing and are **mathematically equivalent but not
bitwise identical** (different summation order, and `a2a` round-trips the fp32
LSE through two 16-bit halves of a bf16 buffer).

| Value | Pattern | Collectives |
|-------|---------|-------------|
| `a2a` (default) | one all-to-all with the LSE packed alongside the output, combine done locally | **1** |
| `ag_rs` | AllGather LSE + local correct + ReduceScatter output | 2 |

Byte counts are about the same; what `a2a` saves is one collective's launch and
sync.

## Launching server

### ATOM server — DeepSeek-R1: TP8 + DCP8 (8 GPUs)

```bash
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    -tp 8 -dcp 8 \
    --kv_cache_dtype fp8        # or bf16; fp8 uses a per-tensor scale
```

### vllm-atom plugin — DeepSeek-R1: TP8 + DCP8 (8 GPUs)

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8 \
    --kv-cache-dtype bfloat16 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

### ATOM server — DeepSeek-R1: TP8 + DCP8 + MTP (8 GPUs, gfx950)

```bash
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    -tp 8 -dcp 8 \
    --kv_cache_dtype fp8 \       # bf16 or fp8; both support MTP under DCP
    --method mtp --num-speculative-tokens 3
```

### ATOM server — Kimi-K3: TP8 + DCP8 (8 GPUs, gfx950)

`-dcp 8` is the only addition to the [Kimi-K3 recipe](../recipes/Kimi-K3.md)
launch; every other flag keeps its recipe value. See
[DCP on Kimi-K3](#dcp-on-kimi-k3-hybrid-kda--mla) for what DCP does and does not
shard on a hybrid model.

```bash
python -m atom.entrypoints.openai_server \
    --model moonshotai/Kimi-K3 \
    --kv_cache_dtype fp8 -tp 8 -dcp 8 \
    --trust-remote-code \
    --max-model-len 16384 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --block-size 128 \
    --no-enable_prefix_caching \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'
```

### ATOM server — Kimi-K3: TP8 + DCP8 + DSpark (8 GPUs, gfx950)

Add the [DSpark](../recipes/DSpark.md) flags to the command above; the draft is
a separate checkpoint. See
[DCP + Speculative Decode](#dcp--speculative-decode-mtp--dspark).

```bash
python -m atom.entrypoints.openai_server \
    --model moonshotai/Kimi-K3 \
    --draft-model Inferact/Kimi-K3-DSpark \
    --kv_cache_dtype fp8 -tp 8 -dcp 8 \
    --method dspark --num-speculative-tokens 2 \
    --trust-remote-code \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.93 \
    --block-size 128 \
    --no-enable_prefix_caching
```

Tips:
- `-tp 8 -dcp 1` (or omitting `-dcp`) disables DCP and serves as the baseline.
- `--kv_cache_dtype fp8` further lowers KV memory; DCP uses a per-tensor scale
  (per-token / per-group fp8 layouts are not yet supported).
- **MTP under DCP is gfx950-only** and works with both bf16 and fp8 KV cache for
  `num_speculative_tokens` 1–3 — see
  [DCP + Speculative Decode](#dcp--speculative-decode-mtp--dspark).

## How it works

1. **KV cache layout.** The cache is interleaved across the DCP group: token `i`
   lives on rank `i % dcp`, so each rank holds only `1/dcp` of the KV. Blocks are
   allocated in *virtual blocks* of `block_size × dcp` global tokens: a single
   block id — shared by all ranks — maps, on each rank, to that rank's own
   physical block of `block_size` interleaved (every-`dcp`-th) tokens. Since one
   block-table entry now spans `dcp×` more tokens, a sequence of a given length
   needs `dcp×` fewer block-table entries than the same `block_size` without DCP.
2. **Prefill KV write.** New-token KV is written interleaved via `slot_mapping`
   (`-1` for tokens this rank does not own). The cached prefix (prefix-cache /
   chunked-prefill) **context** is read by gathering the local compressed KV,
   **AllGather** across the DCP group, reorganizing to per-sequence layout
   (`reorg_kvcache`), then `kv_b_proj` + attention — producing the context
   `(out, LSE)` (LSE-merged across chunks when there is more than one). That
   context output is then LSE-merged with the new-token (suffix) self-attention
   to form the final output (standard chunked-prefill prefix+suffix merge).
   (This is the *compressed-KV AllGather* scheme.)
3. **Decode.** All-gather Q (all heads) → each rank runs attention over its local
   `1/dcp` KV and returns per-token LSE → all-gather LSE, correct each rank's
   partial output, and reduce-scatter so every rank ends with its head slice.
4. **fp8 KV cache.** Per-tensor scale. Decode all-gathers the *quantized* fp8 Q
   (a copy-only collective — safe); the prefill context path dequantizes the
   AllGathered compressed KV before `kv_b_proj`.

## DCP on Kimi-K3 (hybrid KDA + MLA)

Kimi-K3 is not a pure MLA model: of its 93 decoder layers, **24 are MLA
full-attention** and **69 are KDA linear-attention**. DCP shards the paged latent
KV of the 24 MLA layers exactly as it does on a dense MLA model. The KDA layers
hold a **per-request recurrent state** rather than a paged cache, so there is
nothing for DCP to shard there and that state stays replicated — the per-GPU
memory DCP frees on K3 is the MLA share only, not the whole attention footprint.

**Query-head width is the K3-specific constraint.** K3 has 96 query heads, so at
`-tp 8` each rank owns 12 and the DCP decode gathers `12 × dcp`: 24 at dcp2, 48
at dcp4, 96 at dcp8. aiter's `mla_decode_fwd` dispatches natively on 16 / 32 / 64
/ 128 heads only; the remaining multiples of 16 (48, 80, 96, 112) are folded onto
the 16-head kernel, and that fold reinterprets head groups as extra sequence rows
(`total_s *= ori_nhead // 16`) without touching `kv_indptr`, which desynchronises
the row-to-global-position mapping the round-robin causal mask depends on. So the
**gathered** width is padded up to the next natively dispatched one (24 → 32,
48 → 64, 96 → 128) by `mla_dcp_kernel_num_heads`. The pad lives entirely inside
`_forward_decode` — applied after the all-gather and stripped before the
cross-rank LSE combine — so it never costs collective traffic. Widths past 128
fall back to the folded kernel with a warning; lower `-dcp` or raise `-tp` if you
hit it.

DeepSeek-R1 never needed this: 128 heads at `-tp 8` is 16 per rank, and the dcp8
gather lands on 128 exactly.

### Sparse DCP persistent attention and gqa=64

With an **fp8 Q and an fp8 KV cache**, aiter serves gqa=64 only from the
**persistent** decode kernel and aborts the process otherwise (`asm_mla.cu`:
*"fp8/fp8 with gqa_ratio=64 only supports persistent mode"*).

Native sparse MLA / DSA attention under DCP handles this on gfx950 by rebuilding
the persistent work/reduce metadata after every **full IndexShare layer**
compacts its rank-local top-k. The rebuild consumes that layer's
`dcp_sparse_kv_indptr`; following shared layers reuse the same indices, compact
indptr, and work plan. This makes the persistent descriptors and the actual
sparse regions agree without rebuilding metadata on shared layers.

The implementation is scoped to native, non-speculative serving on gfx950 with
page size 1: decode is q_len=1, while sparse prefill is represented as per-token
virtual q_len=1 rows. Unsupported paths (including gfx942 and plugin or
speculative sparse DCP paths without the per-layer rebuild) remain
non-persistent and round a gathered 64 up to **128**.

**GLM-5.2 is the model that benefits**: 64 query heads at `-tp 8` is 8 per
rank, so `-dcp 8` gathers exactly 64 and now dispatches the native persistent
gqa64 kernel instead of padding to 128. `-tp 4 -dcp 4` has the same gathered
width and uses the same persistent gqa64 path.

**Validated** (native ATOM, MI355 gfx950, GLM-5.2-MXFP4, fp8 KV, no
speculative decode): both `-tp 8 -dcp 8` and `-tp 4 -dcp 4` complete CUDA graph
capture, short decode, 7.7k-token sparse prefill/decode, and the full 1319
GSM8K 5-shot set. Both topologies score **flexible-extract 0.9689 /
strict-match 0.9666**. The TP8/DCP8 run also completes a 32-concurrent graph
smoke with no traceback, HIP error, or engine failure.

**Kimi-K3 validated** (ATOM server, 8×MI355 gfx950, `-tp 8 -dcp 8`, fp8 KV, full 1319
GSM8K 5-shot at 64 concurrency): **flexible-extract 0.9553 / strict-match
0.9553**, inside the [Kimi-K3 recipe](../recipes/Kimi-K3.md)'s
0.9538–0.9591 band. Prefix caching was **off** for that run, matching the
recipe — K3's KDA recurrent state cannot be reconstructed from the paged MLA
cache alone, and prefix caching combined with DCP on K3 is not part of the
validated configuration.

## DCP + Speculative Decode (MTP / DSpark)

DCP composes with two drafters: **MTP** (`--method mtp`) on dense MLA, and
**DSpark** (`--method dspark`) on Kimi-K3. Both verify several draft tokens per
step, so decode runs with query length `q > 1`. That is where DCP's round-robin
sharding starts to matter: whenever such a decode is **causal**, its mask has to
be expressed on **global** token positions rather than the rank-local ones the
kernel sees. MTP is causal and needs that treatment; DSpark's draft block is
bidirectional and skips it.

### MTP (dense MLA)

MTP verifies `q = num_speculative_tokens + 1` tokens per step under a causal
mask, so the intra-block mask has to be applied on global positions. This is
handled by a dedicated **round-robin CP (`cprr`) MLA kernel**, selected
automatically when DCP is on, `q > 1`, and the decode is causal.

> **Dense MLA only.** This applies to dense MLA (V3 / R1). **DSA / sparse MLA
> (V3.2-Exp) does not support MTP under DCP yet** — sparse decode with `q > 1` is
> rejected by an assert. Serve DSA + DCP without `--method mtp`.

**Support matrix:**

| | Supported |
|---|---|
| GPU arch | **gfx950 only** (the `cprr` kernel is persistent-only and ships for gfx950; gfx942 has no such kernel) |
| Method | `--method mtp` (`num_speculative_tokens` = 1, 2, or 3) |
| KV cache dtype | **bf16 and fp8** both work for all of `num_speculative_tokens` 1/2/3 |
| DCP size | dcp2 / dcp4 / dcp8 all validated (`tp8`) |

**Usage** (add MTP flags to any DCP command):

```bash
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    -tp 8 -dcp 8 --kv_cache_dtype fp8 \
    --method mtp --num-speculative-tokens 3
```

Plugin path (`vllm serve`): add `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.

> **Not on gfx942.** Speculative decode + DCP raises at startup on non-gfx950
> GPUs (`atom/config.py`). On gfx942 the non-persistent decode fallback ignores
> the `cprr` masking and would silently produce wrong output, so it is rejected
> rather than run — disable either DCP or speculative decode there.

Accuracy: gsm8k (DeepSeek-R1, tp8, 5-shot) matches the non-speculative DCP
baseline (≈0.95) across bf16/fp8 and `num_speculative_tokens` 1/2/3.

### DSpark (Kimi-K3)

[DSpark](../recipes/DSpark.md) drafts a whole block in one parallel backbone
pass instead of `k` serial passes. Two properties decide how it meets DCP:

- **The draft shares the target's paged pool and block tables**, so it inherits
  the round-robin sharding rather than choosing its own. The draft block pass
  therefore addresses and sizes itself in **local** terms: for a global position
  `p`, the owning rank is `p % dcp` and the slot on that rank is
  `block_table[p // (block_size × dcp)] × block_size + (p % (block_size × dcp)) // dcp`;
  positions this rank does not own are written as `-1` and dropped. A sequence of
  `L` global tokens contributes `ceil((L − rank) / dcp)` local rows.
- **The draft block is bidirectional**, not causal: every one of the `T` draft
  positions attends the whole block. So even though `q > 1`, there is no mask to
  place on global positions and the `cprr` kernel is not used — the plain decode
  path is correct. (The target's own verify pass rebuilds its metadata with
  `causal` back to its default, so this never leaks.)

**Support matrix:**

| | Supported |
|---|---|
| Target model | **Kimi-K3** (`--draft-model Inferact/Kimi-K3-DSpark`) |
| GPU arch | gfx950 |
| DCP size | dcp8 validated (`tp8`) |
| KV cache dtype | fp8 (per-tensor scale) |

**Validated** (8×MI355, `-tp 8 -dcp 8`, fp8 KV, `--num-speculative-tokens 2`,
full 1319 GSM8K 5-shot): **flexible-extract 0.9522 / strict-match 0.9522**, with
an **87.1% acceptance rate** (1.74 of 2 draft tokens accepted per step). The
per-step accepted-count distribution is `{0: 6.1%, 1: 13.7%, 2: 80.3%}` — 80% of
steps take the whole draft and only 6% take none, so the draft is genuinely
contributing under DCP rather than collapsing back to single-token decode.

## Constraints & Compatibility

| Constraint | Notes |
|-----------|-------|
| Models | MLA-bearing only: **dense** (DeepSeek-V3 / R1, …), **sparse / DSA** (DeepSeek-V3.2-Exp), and **hybrid KDA + MLA** (Kimi-K3), all prefill + decode |
| World size | `world = tp`, `tp % dcp == 0` (DCP does not add GPUs) |
| fp8 KV cache | Supported, **per-tensor scale only** (per-token / per-group not supported) |
| prefix caching / chunked prefill | Supported (dense and sparse / DSA). On **Kimi-K3** the validated configuration has prefix caching **off**, per its recipe |
| Kimi-K3 KDA layers | Not sharded — the KDA recurrent state is per-request, not paged, so DCP frees only the MLA share of attention memory |
| Kimi-K3 gathered head width | Padded to a natively dispatched MLA width (16 / 32 / 64 / 128); past 128 it falls back to the folded kernel with a warning — lower `-dcp` or raise `-tp` |
| gathered head width 64 + fp8 KV | Native sparse / DSA prefill and q_len=1 decode on gfx950 rebuild persistent metadata per full IndexShare layer and run gqa64 directly; unsupported non-persistent paths still pad to 128 — see [Sparse DCP persistent attention and gqa=64](#sparse-dcp-persistent-attention-and-gqa64) |
| speculative decode (MTP), dense MLA | Supported on **gfx950 only** (bf16/fp8, `num_speculative_tokens` 1–3); raises at startup on gfx942 |
| speculative decode (DSpark), Kimi-K3 | Supported on gfx950; validated at `tp8 -dcp 8` with `num_speculative_tokens 2` |
| speculative decode (MTP), sparse / DSA | **Not supported** — sparse decode with `q > 1` under DCP is rejected by an assert |
| vllm-atom plugin | Validated for dense MLA only; the sparse / DSA and Kimi-K3 plugin paths are not yet verified |
| DCP + PCP | Independent dimensions (different phases); combined use not validated here |

## Source Files

| File | Description |
|------|-------------|
| `atom/model_engine/arg_utils.py` | `--decode-context-parallel-size` / `-dcp` CLI |
| `atom/config.py` | `DCPConfig` (the four `--dcp-config` knobs) + their validation; `qrep_unsupported_reason` auto-gating; DCP validation (`tp % dcp == 0`); spec-decode + DCP arch gate (gfx950) |
| `atom/model_engine/block_manager.py` | Interleaved block allocation; prefix-cache virtual-block accounting |
| `atom/distributed/dcp_utils.py` | DCP distributed-access layer: `get_dcp_world_size` / `dcp_is_enabled` / `get_dcp_group` / `get_dcp_rank` |
| `atom/model_ops/dcp_ops.py` | Both merge backends -- `cp_lse_ag_out_rs` (AG+RS LSE-combine) and `cp_lse_a2a` (all-to-all pack / unpack-combine kernels); `reorg_kvcache`, local compressed-KV gather, `dcp_all_gather` / `dcp_all_gather_query_heads` (custom-collective AllGather) |
| `atom/model_ops/attention_mla.py` | Server-mode DCP decode + prefix-cache / chunked-prefill context; `mla_dcp_kernel_num_heads` / `mla_dcp_decode_is_persistent` gathered-head-width padding |
| `atom/model_ops/attentions/aiter_mla.py`, `attentions/backends.py` | DCP decode / prefill metadata (interleaved slot_mapping, local seq lens) |
| `atom/plugin/vllm/attention/layer_mla.py`, `attention/metadata.py` | vllm-atom plugin DCP decode + prefill context; persistent-metadata head sizing |
| `atom/models/kimi_k3.py` | Kimi-K3 hybrid backbone (KDA + MLA) served under DCP |
| `atom/spec_decode/eagle_proposer.py` | MTP draft loop: DCP round-robin slot for draft KV writes |
| `atom/spec_decode/dspark_proposer.py` | DSpark block draft: DCP-local slot mapping / context lengths, gathered-width work descriptors |
| `atom/model_ops/attentions/aiter_mla.py` (`prepare_mtp_decode`) | Per-draft-step DCP-local metadata rebuild; `cprr` decode selects the round-robin MLA kernel via `g_kv_indptr` |