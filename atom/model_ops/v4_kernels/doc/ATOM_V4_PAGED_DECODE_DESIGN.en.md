# DeepSeek-V4 Paged Decode: ATOM's Custom Design

This document describes ATOM's implementation of sparse decode attention for
DeepSeek-V4 — the design rationale, KV cache layout, index construction, and
CUDAGraph-friendliness constraints behind the unified paged kernel
`sparse_attn_v4_paged_decode` (page_size = 1). Audience: engineers extending
or maintaining the V4 attention backend.

> Reading guide: §1-§5 build the basic mental model (design goal, layer
> structure, KV layout, index translation, kernel interface). §6 covers
> per-forward index buffer construction. §7 covers the V4-Pro MTP-1
> per-token expansion. §8 onward is dispatch and CUDAGraph engineering
> details.

---

## 0. Differences from upstream V4 reference (ATOM-specific)

The official DeepSeek reference (`/data/DeepSeek-V4-Pro/inference/model.py`)
implements attention with one code path per ratio. SWA ring buffers and
compressor paged KV are kept in separate tensors, and each forward
materializes "the K's chosen by the indexer" into a temporary dense tensor
that is then fed to a dense attention kernel. That reference impl cannot be
captured by CUDAGraph because the temporary tensor's shape depends on
device-side data.

ATOM's implementation does 4 fundamental refactors with one goal:
**make V4 decode entirely free of host-sync and per-forward dynamic-shape
allocation, so the whole forward fits inside a CUDAGraph**:

1. **Unified KV pool (§3)**: every layer **physically merges** its SWA ring
   buffer and compressor paged KV into a single BF16 tensor. The kernel
   carries one base pointer; every index (whether from SWA, CSA Main, or
   HCA Main) is a row offset into that pool. Upstream uses two separate
   tensors and a per-source branch inside the kernel.
2. **Unified paged kernel (§5)**: a V4-specific Triton kernel
   `sparse_attn_v4_paged_decode`, page_size = 1, with an API aligned to
   ATOM's V3.2 `mla_decode_fwd` (`kv_indices` + `kv_indptr` packed cumsum
   + `attn_sink`). All three layer types (Dense / CSA / HCA) share **the
   same** kernel and only differ in indptr. Upstream's per-ratio dispatch
   is dissolved by this abstraction.
3. **CG-friendly index construction (§6)**: every per-forward index tensor
   (SWA uniform-stride buffer, CSA / HCA packed-cumsum kv_indptr and
   kv_indices) is computed in the metadata builder, with **all size
   bounds being metadata-time constants** (`max_num_reqs * (1 +
   max_spec_steps)` etc.) — no `.item()`, no device-data-dependent
   allocation. CSA per-layer translation uses a fixed-grid
   `csa_packed_write` Triton kernel (§6.4) instead of upstream's fancy
   scatter (fancy scatter is a dynamic-shape op and breaks CG).
4. **MTP-1 via packed-indptr physical replication (§7)**: MTP raises the
   per-request token count from 1 to `1 + max_spec_steps`. ATOM
   **physically replicates** the base- and draft-token ring slots / topk
   sections in the packed indptr instead of sharing them. Trading 6.5%
   buffer memory eliminates the bookkeeping complexity of "shared KV
   slots in indptr" — the kernel never has to be aware of MTP, and every
   per-token dimension simply uses
   `total_tokens = num_reqs * (1 + max_spec_steps)`.

### Why not just reuse V3.2's `mla_decode_fwd`

V3.2 sparse decode uses aiter `mla_decode_fwd`, which looks similar but has
3 irreconcilable differences:

| item       | V3.2 mla_decode_fwd  | V4 needs                                              |
|------------|----------------------|-------------------------------------------------------|
| attn_sink  | none                 | yes (per-head learnable softmax-denom bias)           |
| page_size  | 64                   | 1 (SWA / CSA / HCA all have different strides)        |
| KV source  | single paged KV cache| SWA ring + multiple compressor paged KV               |

Squeezing V4's multi-stride indices into a page_size=64 paged interface
would require per-layer KV repacking that costs more than writing a custom
kernel. So ATOM **keeps V3.2's naming convention** (`kv_indices` /
`kv_indptr` / `softmax_scale` / `nhead`) so both impls share reading mental
model, but the kernel is an independent V4-only implementation. Genuinely
new concepts get new names (`unified_kv`, `swa_pages`, `csa_block_capacity`,
`kv_idx_local`).

---

## 1. Design goal

On the decode path, every V4 attention layer does sparse attention: the
query gathers softmax-weighted values from a set of K's selected from the
SWA ring buffer / compressed KV pool. The K set varies in source and size
per layer (Dense / CSA / HCA), but in every case it can be abstracted as
"index-gather rows from a single BF16 pool".

`sparse_attn_v4_paged_decode` cements that abstraction:

- Each layer exposes **one** `unified_kv` BF16 pool (page_size = 1, i.e. one
  row = one KV slot).
- Each token's K set is described by an int32 index span (packed-cumsum-style
  `kv_indptr` + `kv_indices`).
- The kernel decides per-token how many K's to read at runtime — no
  per-forward dynamic-shape tensor allocation needed.

This design eliminates the "materialize selected K into a temporary
`kv_flat_sa`" intermediate step that requires per-forward variable-shape
allocation — one of the main causes of CUDAGraph capture failure.

---

## 2. V4 attention layer structure

V4 is hybrid attention; every layer's behavior is determined by
`compress_ratios[layer_id]` (a length-`num_layers` int list in the model
config):

| compress_ratio | name  | components                          | V4-Pro layer count (62 total) |
|----------------|-------|-------------------------------------|-------------------------------|
| 0              | Dense | SWA only                            | 1                             |
| 4              | CSA   | SWA + CSA Main Compressor + Indexer | 30                            |
| 128            | HCA   | SWA + HCA Main Compressor           | 31                            |

**Every layer has SWA** (sliding-window attention, fixed
`window_size = 128`, BF16, single KV head). Layers differ in the compress
component:

- **Dense**: no compressor, only attends the most recent `window_size` tokens.
- **CSA** (compressed selective attention): every 4 raw tokens are
  overlap-compressed into 1 K, then the Indexer picks
  top-`index_topk` by score.
- **HCA** (hierarchical coarse attention): every 128 raw tokens are
  compressed into 1 K (no overlap), all participating in attention (no
  selection).

Physical paged-KV block size is `block_size = lcm(4, 128) = 128` raw tokens,
so each block holds:

- CSA Main: `csa_block_capacity = block_size / 4 = 32` compressed K's.
- HCA Main: `hca_block_capacity = block_size / 128 = 1` compressed K.
- CSA Indexer: 32 FP8 score K's (written in sync with CSA Main, but
  independent dtype).

**`attn_sink`** is V4-specific — a per-head learnable bias added to the
softmax denominator (does not contribute to the numerator gather), equivalent
to a "virtual global K" that takes some probability mass.

V4-Pro also enables **MTP-1** (Multi-Token Prediction,
`num_nextn_predict_layers = 1`) — every forward computes 1 base token + 1
draft token per request (i.e. `1 + max_spec_steps = 2` tokens). This
affects the actual shape of every "per-token" tensor (see §7).

---

## 3. KV Cache Layout

### 3.1 unified_kv: per-layer BF16 pool

Each layer has one contiguous 2D tensor concatenating the SWA prefix and the
optional compress tail:

```
Dense layer  : unified_kv = [num_slots*window_size,                                  head_dim] BF16
CSA layer    : unified_kv = [num_slots*window_size + num_blocks*csa_block_capacity, head_dim] BF16
HCA layer    : unified_kv = [num_slots*window_size + num_blocks*hca_block_capacity, head_dim] BF16
                            └──────── SWA ────────┘ └─────────── compress ───────────┘
                                  swa_pages
```

The benefit of physically merging SWA and compress: the sparse attn kernel
needs only one base pointer; every index becomes a row offset into
`unified_kv`, and the kernel never has to disambiguate KV source.

#### Key quantities

- **`num_slots`** = `model_runner.max_per_req_cache_slots`
  - Number of per-request state-cache slots; each concurrent request
    occupies `slots_per_req` slots.
  - V4 has `slots_per_req = 1`, so `num_slots = max_num_seqs`.
  - Physical meaning: each slot holds one request's SWA ring buffer + the
    compressor's `kv_state` / `score_state`. Allocated when a request enters
    inference, released when it finishes.
  - Source: `atom/model_engine/model_runner.py:1157`.

- **`num_blocks`** = `model_runner.num_physical_kvcache_blocks`
  - Total physical paged-KV blocks; each block covers V4's `block_size = 128`
    raw tokens.
  - Auto-computed from GPU memory budget and `gpu_memory_utilization`.
  - Physical meaning: the global KV pool **shared** across all concurrent
    requests; BlockManager allocates blocks on demand to each request's
    paged KV table.
  - Source: `atom/model_engine/model_runner.py:1239`.

- **`window_size`** = 128 (V4 SWA window size).
- **`csa_block_capacity`** = `block_size / 4 = 32`.
- **`hca_block_capacity`** = `block_size / 128 = 1`.
- **`swa_pages`** = `num_slots * window_size` (boundary between SWA and
  compress in `unified_kv`; also the base for all subsequent compress
  offsets).
- **`head_dim`** = 512 (V4-Pro, globally shared MQA single KV head).

`num_slots` determines **concurrency** (number of simultaneously active
requests); `num_blocks` determines **total context** capacity (cumulative
KV across all requests). The two scale independently along their respective
axes.

### 3.2 Module view binding

`build_kv_cache_tensor` (in
`atom/model_ops/attentions/deepseek_v4_attn.py`'s
`DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor`) slices each
layer's `unified_kv` into the views the modules expect:

```python
attn.unified_kv     = unified_kv                                              # whole pool, for paged decode kernel
attn.swa_kv         = unified_kv[:swa_pages].view(num_slots, window_size, head_dim)
compressor.kv_cache = unified_kv[swa_pages:].view(num_blocks, block_capacity, head_dim)
```

`swa_write` / `Compressor.scatter` write paths need **no changes** — they
write to the same physical storage, just through different logical views.

### 3.3 Separate CSA Indexer FP8 pool

`v4_csa_idx_kv: [num_csa_layers, num_blocks, csa_block_capacity, aligned_index_dim]`
is **not** merged into `unified_kv`, because:

- Its dtype is FP8 + interleaved 4-byte fp32 scale (independent layout, can't
  share a pool with BF16).
- Written by `indexer_k_quant_and_cache`, read by
  `cp_gather_indexer_k_quant_cache` — the entire path is decoupled from the
  sparse attn kernel.
- `aligned_index_dim = ((index_head_dim + 4 + 15) // 16) * 16` — 16-byte
  alignment required by Inductor's unaligned-access optimization.

The CSA Indexer pool is only used to compute `topk_local` (which compressed
K's should be picked); it does not participate in the final attention
gather. Picked indices are translated to `unified_kv` offsets, then handed
to the paged decode kernel.

---

## 4. Index translation for the three KV sources

Each token's K set may come from SWA, CSA compress, or HCA compress. This
section defines the "logical position → unified_kv physical offset"
translation rules.

### 4.1 SWA (every layer)

**Write**: ring buffer; each sequence occupies one state slot:

```
swa_kv[state_slot, ring_offset, :] = kv_value
ring_offset = position % window_size
```

**Read**: a token at position `current_position` should attend the most
recent `window_size` tokens, i.e. positions
`[max(0, current_position - window_size + 1), current_position]`.

`_build_window_topk_batched` (`atom/model_ops/attentions/deepseek_v4_attn.py:79`)
produces `[total_tokens, window_size]` int32 from each sequence's
`start_pos` (the starting position when that sequence enters the current
forward). **The output semantics differ across three modes**:

| mode             | trigger                                 | output semantics            | formula                                                                                  |
|------------------|-----------------------------------------|-----------------------------|------------------------------------------------------------------------------------------|
| fresh prefill    | `start_pos == 0`                        | **absolute positions** (not ring) | `[max(0, current_position - window_size + 1), current_position]`, padded with `-1`       |
| prefix mode      | `0 < start_pos < window_size - 1`       | `ring_offset`               | `[0, 1, ..., start_pos, -1, ..., -1]` (first `start_pos+1` ring slots already written)   |
| steady cyclic    | `start_pos >= window_size - 1`          | `ring_offset`               | `[(start_pos + 1 + j) % window_size for j in 0..window_size)`, oldest → newest           |

Decode always lands in prefix or cyclic mode. In cyclic mode,
`current_position % window_size` is exactly the ring slot most recently
written (the last value in the array).

`-1` means a ring slot has not yet been written (early decode with fewer
than `window_size` tokens); the sparse attn kernel skips it.

**SWA → unified_kv physical offset** (**only valid for decode output** —
prefill mode outputs absolute positions, not `ring_offset`, and takes a
different path):

```
window_paged_offset[token_idx, window_idx] =
    state_slot[token_idx] * window_size + window_topk_batched[token_idx, window_idx]
        if window_topk_batched[token_idx, window_idx] >= 0
    -1
        otherwise
```

The prefill path uses case-A absolute positions to gather the freshly
computed KV slice from the current forward (layout
`[total_prefill_tokens, head_dim]`); it does not read the SWA ring buffer.
That's why paged_decode kernel only covers the decode path.

### 4.2 CSA Main compress (CSA layers)

**Write**: every 4 raw tokens are overlap-compressed into 1 K via
`Compressor.forward`, organized along V4's paged-KV blocks. Each physical
block holds `csa_block_capacity = 32` compressed K's (128 raw tokens),
written into `Compressor.kv_cache[physical_block_id, slot_in_block, :]`.

**Read**: the CSA Indexer computes scores, picks top-`index_topk`
sequence-local compressed positions
`compress_idx_logical ∈ [0, num_committed_csa)`, where
`num_committed_csa = ctx_len // 4`.

**CSA → unified_kv physical offset**:

```
block_idx_in_seq = compress_idx_logical // csa_block_capacity   # which paged block
slot_in_block    = compress_idx_logical %  csa_block_capacity   # offset within block
physical_block   = block_table[sequence_id, block_idx_in_seq]
paged_offset     = swa_pages
                 + physical_block * csa_block_capacity
                 + slot_in_block
```

`-1` sentinels (when the indexer outputs invalid positions) are kept as
`-1`.

### 4.3 HCA Main compress (HCA layers)

**Write**: every 128 raw tokens are compressed into 1 K (no overlap). Each
physical block holds `hca_block_capacity = 1` compressed K, written into
`kv_cache[physical_block_id, 0, :]`.

**Read**: HCA does not use the indexer; every token attends all
`num_committed_hca = ctx_len // 128` committed compressed K's (where
`ctx_len = batch.context_lens[seq]` is the total sequence token count,
post-extend).

**HCA → unified_kv physical offset** (simplifies because
`hca_block_capacity = 1`):

```
physical_block = block_table[sequence_id, compress_idx_logical]   # compress_idx_logical ∈ [0, num_committed_hca)
paged_offset   = swa_pages + physical_block * 1 + 0
               = swa_pages + physical_block
```

### 4.4 Translation rules summary

| KV type           | unified_kv physical offset                                                        | index source                          |
|-------------------|-----------------------------------------------------------------------------------|---------------------------------------|
| SWA               | `state_slot * window_size + ring_offset`                                          | `window_topk_batched`                 |
| CSA compress      | `swa_pages + physical_block * csa_block_capacity + slot_in_block`                 | indexer raw `topk_local`              |
| HCA compress      | `swa_pages + physical_block`                                                      | full `[0, num_committed_hca)`         |

---

## 5. Sparse Attention Kernel

### 5.1 Interface

Location: `atom/model_ops/v4_kernels/paged_decode.py`

```python
def sparse_attn_v4_paged_decode(
    q:                 torch.Tensor,  # [total_tokens, num_heads, head_dim]   BF16
    unified_kv:        torch.Tensor,  # [total_pages, head_dim]               BF16  ← single base ptr
    kv_indices:        torch.Tensor,  # [total_indices_in_batch]              int32 ← variable-length, packed
    kv_indptr:         torch.Tensor,  # [total_tokens + 1]                    int32 ← true prefix sum
    attn_sink:         torch.Tensor,  # [num_heads]
    softmax_scale:     float,
) -> torch.Tensor:                    # [total_tokens, num_heads, head_dim]
```

API style matches `aiter.mla.mla_decode_fwd`, page_size = 1 (one row of
`unified_kv` is one KV slot — no intra-block index needed).

### 5.2 Semantic conventions

- **Per-token valid K range** =
  `kv_indices[kv_indptr[token_idx] : kv_indptr[token_idx + 1]]`.
- **`-1` entries are auto-skipped**: inside the kernel,
  `valid = in_range & (slot >= 0)`; invalid slots contribute nothing to
  softmax (score → -inf, probability → 0).
- **Runtime trip count**: the kernel uses
  `tl.range(0, kv_len, BLOCK_K)` to decide per-token loop count, so short
  sequences don't pay for long-sequence worst-case work.
- No `K_MAX` constexpr; the kernel binary is shared across batch shapes.

### 5.3 Numerics

Online-softmax accumulation (fp32), with attn_sink merged into the
denominator at the end:

```
max_logit_final = max(max_logit_acc, attn_sink)
sum_exp_final   = sum_exp_acc * exp(max_logit_acc - max_logit_final)
                + exp(attn_sink - max_logit_final)
output          = output_acc / sum_exp_final
```

Kernel constants: `BLOCK_H = 16` (AMD MFMA minimum tile),
`BLOCK_D = next_pow2(head_dim)`, `BLOCK_K = 16` if `head_dim ≥ 256` else 32.

---

## 6. Per-Forward Index Construction

Each forward must construct `kv_indices` and `kv_indptr` before invoking
the sparse attn kernel. This section covers index buffer layout, indptr
design, the construction split, and the critical packed-write step.

### 6.1 Buffer layout

One independent buffer pair per ratio (SWA / CSA / HCA), preallocated in
`forward_vars` at CUDAGraph-stable addresses:

```
v4_kv_indices_swa : [max_num_reqs * (1 + max_spec_steps) * window_size]                            int32
v4_kv_indptr_swa  : [max_num_reqs * (1 + max_spec_steps) + 1]                                      int32

v4_kv_indices_csa : [max_num_reqs * (1 + max_spec_steps) * (window_size + index_topk)]             int32
v4_kv_indptr_csa  : [max_num_reqs * (1 + max_spec_steps) + 1]                                      int32

v4_kv_indices_hca : [max_num_reqs * (1 + max_spec_steps) * (window_size + max_num_committed_hca)]  int32
v4_kv_indptr_hca  : [max_num_reqs * (1 + max_spec_steps) + 1]                                      int32
```

`max_num_committed_hca = max_model_len // 128` (V4-Pro's 1M context = 8192).
Buffer capacity is sized for the worst case (packed-indptr requires each
token to occupy a complete index span; MTP draft tokens do not share
physical slots — overlapping content is physically replicated). Each
forward only uses the prefix.

The actual token count seen during CUDAGraph capture =
`max_capture_size` (typically set by `--cudagraph-capture-sizes`), well
below the buffer cap; the captured graph reads from fixed addresses.

#### Why SWA gets its own buffer rather than reusing the CSA/HCA window prefix

The three buffers' indptr strides differ (`window_size` vs
`window_size + index_topk` vs variable true prefix sum). Identical window
content must land at three different physical addresses for each indptr to
correctly index a contiguous span inside its own buffer. Dense layers use
`v4_kv_indices_swa`; CSA/HCA layers use their own buffers (whose window
prefix content is identical to the SWA buffer, just at a different
physical location).

The window contents in all 3 buffers are **identical**
(`state_slot * window_size + ring_offset`, layer-invariant). The builder
writes the same window data into the head of each of the 3 buffers in one
pass.

### 6.2 Construction split

| data                            | location                          | per-layer? | depends on                                     |
|---------------------------------|-----------------------------------|------------|------------------------------------------------|
| window paged indices            | metadata builder                  | no         | `state_slot`, `window_topk_batched`            |
| HCA compress paged indices      | metadata builder                  | no         | `block_tables`, `num_committed_hca`            |
| SWA / CSA / HCA kv_indptr       | metadata builder                  | no         | per-token kv_len of each ratio                 |
| **CSA compress paged indices**  | **V4Attention.forward CSA branch**| **yes**    | indexer raw `topk_local`                       |

CSA is the only per-layer work, because the indexer output differs per
layer. Everything else is layer-invariant — built once by the builder and
shared across all layers.

### 6.3 Three indptr forms

#### CSA: variable-length packed cumsum

Per-token actual kv_len = `window_size + min(n_committed_csa, index_topk)`.
In early decode (`ctx_len < 4 * index_topk = 4096`),
`n_committed_csa < index_topk`, so indptr automatically gives that token a
smaller buffer span.

```
kv_indptr_csa[token_idx + 1] =
    kv_indptr_csa[token_idx]
    + window_size + min(n_committed_csa[batch_id_per_token[token_idx]], index_topk)
```

`n_committed_csa = ctx_len // 4` is the number of CSA compressed K's the
sequence has committed.

**Example** (V4-Pro, 3 reqs × 1 token each, `ctx_len` = 200 / 400 / 5000):

```
n_committed_csa  = [50, 100, 1250]
valid_count      = min(n_committed_csa, 1024) = [50, 100, 1024]
per-token kv_len = 128 + valid_count          = [178, 228, 1152]

indptr  = [0, 178, 406, 1558]                            # 4 entries (T+1 = 4)
indices = [w0_0..w0_127,  c0_0..c0_49,                   # req0: 128 win + 50 compress
           w1_0..w1_127,  c1_0..c1_99,                   # req1: 128 win + 100 compress
           w2_0..w2_127,  c2_0..c2_1023]                 # req2: 128 win + 1024 compress (capped)
                                                          # 178+228+1152 = 1558 entries total
```

#### HCA: variable-length packed cumsum

Per-token actual kv_len = `window_size + num_committed_hca`; all committed
compressed K's participate (no sentinel waste).

```
kv_indptr_hca[token_idx + 1] =
    kv_indptr_hca[token_idx]
    + (window_size + num_committed_hca[batch_id_per_token[token_idx]])
```

For 1-token-per-sequence decode, `batch_id_per_token = arange(batch_size)`.

**Example** (V4-Pro, 2 reqs × 1 token each, `ctx_len` = 400 / 5000):

```
num_committed_hca = [400//128, 5000//128] = [3, 39]
per-token kv_len  = 128 + num_committed_hca = [131, 167]

indptr  = [0, 131, 298]                                  # 3 entries (T+1 = 3)
indices = [w0_0..w0_127,  h0_0, h0_1, h0_2,              # req0: 128 win + 3 compress
           w1_0..w1_127,  h1_0..h1_38]                   # req1: 128 win + 39 compress
                                                          # 131+167 = 298 entries total
```

#### SWA: uniform stride + sentinel

Per-token buffer capacity is constant `window_size`; indptr degenerates to
an arithmetic progression:

```
kv_indptr_swa[token_idx] = token_idx * window_size
```

In early decode (`current_position < window_size`), only
`current_position + 1` ring slots have been written; the rest hold `-1`
sentinels (generated by `_build_window_topk_batched` according to
`start_pos` mode — see §4.1). The kernel auto-skips `-1`, so actually
attended kv_len = `min(current_position + 1, window_size)`.

**Example** (V4-Pro, 3 reqs × 1 token each, `current_position` = 50 / 5000 / 100000):

```
indptr  = [0, 128, 256, 384]                             # 4 entries (T+1 = 4)
indices = [w0_0..w0_50,  -1 × 77,                        # req0 early: 51 valid + 77 sentinel
           w1_0..w1_127,                                  # req1 steady cyclic: 128 all valid
           w2_0..w2_127]                                  # req2 steady cyclic: 128 all valid
                                                          # 3 * 128 = 384 entries total
```

SWA uses uniform stride (not packed) deliberately: the `window_size = 128`
upper bound is small enough that the cumsum overhead (computing indptr in
the builder + packed write in the forward) outweighs the few sentinel slots
saved. The design is symmetric — switching SWA to packed is possible — but
keeping it uniform simplifies the implementation.

### 6.4 CSA per-layer translation + packed write

CSA is the only per-layer index work, because every layer's indexer
produces a different `topk_local`; the translation must happen inside that
layer's forward and write to `v4_kv_indices_csa`.

#### Translation formula

```python
# topk_local: [total_tokens, index_topk] int32
#   indexer raw output, sequence-local in [0, n_committed_csa)
#   first valid_count entries valid, rest -1
# block_tables_per_token: [total_tokens, max_blocks_per_seq] int32
#   prebuilt by builder = block_tables[batch_id_per_token]; shared by all layers

block_idx_in_seq = topk_local // self.csa_block_capacity   # [T, index_topk] int32
slot_in_block    = topk_local %  self.csa_block_capacity
physical_block   = block_tables_per_token.gather(1, block_idx_in_seq.long())
paged_compress   = (
    swa_pages
    + physical_block * self.csa_block_capacity
    + slot_in_block
)
```

`paged_compress[t, k]` is garbage for `k >= valid_count[t]` (the
corresponding `topk_local` slot is `-1`, so the translation result is
meaningless), but the next packed-write step only reads
`[0, valid_count[t])`, so the garbage is ignored.

#### Why packed write is necessary

Because CSA indptr is a packed cumsum, token t's buffer occupancy is
`[indptr[t], indptr[t+1])`, and the length differs per token. A simple
slice assignment like
`kv_indices[:, win:win+index_topk] = paged_compress` would overflow into
the next token's start offset — packed write is required.

`torch.scatter_` + bool mask is also unusable —
`tensor[bool_mask]` is a dynamic-shape selection, which causes CUDAGraph
capture to fail.

The CG-friendly packed write is **a fixed-grid Triton kernel**: each
`(token, k)` thread checks `k < valid_count[t]` and writes to
`kv_indices[indptr[t] + win + k]`:

```python
@triton.jit
def csa_packed_write_kernel(paged_compress_ptr, kv_indices_ptr,
                             kv_indptr_ptr, valid_count_ptr,
                             window_size: tl.constexpr, index_topk: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    valid_k = tl.load(valid_count_ptr + pid_t)
    if pid_k < valid_k:
        offset = tl.load(paged_compress_ptr + pid_t * index_topk + pid_k)
        write_pos = tl.load(kv_indptr_ptr + pid_t) + window_size + pid_k
        tl.store(kv_indices_ptr + write_pos, offset)
    # out-of-range thread is a no-op (the buffer slot may hold leftover
    # data from a previous forward, but indptr[t+1] bounds the attention
    # kernel away from reading there)
```

The builder precomputes `valid_count = min(n_committed_csa[batch_id],
index_topk)` and stages it into `forward_vars`, shared across all CSA
layers.

The whole path has zero H2D transfers, zero dynamic shapes, and a fixed
kernel grid — CUDAGraph-friendly.

### 6.5 Indexer output semantics

`indexer_score_topk` returns raw `[total_tokens, index_topk] int32`
`topk_local`, i.e. sequence-local compressed K positions (in
`[0, n_committed_csa)`), with invalid entries filled by `-1`. All
"translate to paged offset" logic lives in the caller
(`V4Attention.forward`), not polluting the indexer module's abstraction.

---

## 7. MTP (Multi-Token Prediction)

V4-Pro's config has `num_nextn_predict_layers = 1`. With MTP enabled, every
forward computes `(1 + max_spec_steps)` tokens per request:

- **base token**: at the current real position `start_pos`, outputs the
  logits that confirm the next token.
- **draft tokens**: `max_spec_steps` speculatively predicted tokens at
  positions `[start_pos + 1, start_pos + max_spec_steps]`, accepted/rejected
  by EagleProposer.

V4-Pro defaults to `max_spec_steps = 1` (MTP-1) — 2 tokens per request per
forward.

### 7.1 Per-forward token layout

```
positions  = [s0, s0+1, s1, s1+1, ..., s_{R-1}, s_{R-1}+1]   # R = num_reqs, 2R tokens total
cu_seqlens_q = [0, 2, 4, ..., 2R]                             # 2 tokens per req, contiguous
token_num_per_seq = [2, 2, ..., 2]                            # length R
total_tokens = R * (1 + max_spec_steps) = 2R                  # MTP-1
batch_id_per_token = [0, 0, 1, 1, ..., R-1, R-1]              # each req repeated (1+spec_steps) times
```

`prepare_decode` (`atom/model_ops/attentions/deepseek_v4_attn.py:1180`)
already expands this layout via `max_seqlen_q = batch.num_spec_step + 1`
and `np.tile + np.repeat`. Paged index construction code does not need MTP
special cases — just compute along the `total_tokens` axis.

### 7.2 Per-ratio behavior under MTP

#### SWA: adjacent draft tokens have heavily overlapping windows

Base at position p, draft at position p+1 — their attended ring slots only
differ by one trailing offset:

```
base  attended ring offsets: (p+1)%win, (p+2)%win, ..., p%win              # win entries
draft attended ring offsets: (p+2)%win, (p+3)%win, ..., (p+1)%win           # win entries
                                                                            # win-1 entries overlap
```

But packed-indptr requires per-token index spans to be contiguous and
non-overlapping in storage, so the buffer still gives base and draft each
their own `window_size` slots — duplicated logically, written independently
physically (see §6.1).

#### CSA: every draft token runs the indexer separately

Base and draft have different Q (different positions), each calls the
indexer to compute its own `topk_local`, and the results barely overlap.
The `csa_packed_write` kernel uses a `(total_tokens, index_topk)` grid;
each token does its translation + write independently.

`n_committed_csa[batch_id_per_token[token_idx]]` gives the same value to
both base and draft of one req (because `ctx_len` is sequence-level, not
per-token), so their `valid_count` matches.

#### HCA: base and draft almost always share num_committed_hca

`num_committed_hca = ctx_len // 128` only changes when ctx_len crosses a
multiple of 128. Since base and draft are 1 position apart in MTP-1, in 99%+
of cases their `num_committed_hca` is identical.

### 7.3 MTP-1 indptr examples

V4-Pro MTP-1, 2 reqs, `ctx_len` = 5000 / 5001 (note 5001 // 128 = 39 ==
5000 // 128, so both reqs have `num_committed_hca` = 39):

**HCA**:

```
total_tokens = 2 * 2 = 4
num_committed_hca per req = [39, 39]
per-token kv_len = 128 + 39 = 167 (same for all 4 tokens)

indptr  = [0, 167, 334, 501, 668]                            # T+1 = 5
indices = [w0_*,  h0_0..h0_38,                               # req0 base:  128 win + 39 compress
           w1_*,  h1_0..h1_38,                               # req0 draft: 128 win + 39 compress
           w2_*,  h2_0..h2_38,                               # req1 base:  128 win + 39 compress
           w3_*,  h3_0..h3_38]                               # req1 draft: 128 win + 39 compress
                                                              # 4 * 167 = 668 entries total
```

req0's base (`w0/h0`) and draft (`w1/h1`) compress sections point to
**identical** unified_kv slots (same `num_committed_hca`, same
`block_table`); their window sections point to the same ring with cyclic
ring_offsets shifted by 1. All these duplicates are physically replicated
in storage under the packed-indptr scheme.

**CSA** (same 2 reqs × 2 tokens, `ctx_len = 5000/5001`,
`n_committed_csa = ctx_len // 4 = 1250`, all ≥ `index_topk = 1024` so
capped):

```
valid_count = [1024, 1024, 1024, 1024]
per-token kv_len = 128 + 1024 = 1152

indptr  = [0, 1152, 2304, 3456, 4608]                        # T+1 = 5
indices = [w0_*, c0_0..c0_1023,                              # req0 base
           w1_*, c1_0..c1_1023,                              # req0 draft (different topk vs base!)
           w2_*, c2_0..c2_1023,                              # req1 base
           w3_*, c3_0..c3_1023]                              # req1 draft
                                                              # 4 * 1152 = 4608 entries total
```

Note that, unlike HCA, req0's base- and draft-CSA compress slots come from
their respective indexer runs and are not equal.

**SWA** (Dense layer, uniform stride, 2 reqs × 2 tokens,
`current_position = 5000 / 5001 / 7000 / 7001`):

```
indptr  = [0, 128, 256, 384, 512]                            # T+1 = 5
indices = [w0_0..w0_127,                                      # req0 base  (position 5000)
           w1_0..w1_127,                                      # req0 draft (position 5001, ring shifted 1)
           w2_0..w2_127,                                      # req1 base  (position 7000)
           w3_0..w3_127]                                      # req1 draft (position 7001)
                                                              # 4 * 128 = 512 entries total
```

### 7.4 Dispatch detection: is_pure_decode

The earlier `is_decode_only` predicate
(`(token_num_per_seq == 1).all()`) only holds for single-token decode; in
MTP mode `token_num_per_seq = (1 + max_spec_steps) > 1`.

To support MTP, the dispatch detection generalizes to "every sequence is
in decode mode":

```
is_pure_decode =
    (cu_seqlens_q[1:] - cu_seqlens_q[:-1] == 1 + max_spec_steps).all()
    and (start_pos_per_seq > 0).all()
```

Equivalently:
`scheduled_batch_size * (1 + max_spec_steps) == total_tokens` and no fresh
prefill (`start_pos == 0`) sequence is mixed into the batch.

---

## 8. Decode Dispatch

`V4Attention.forward` ends by branching on `is_pure_decode`:

```python
if is_pure_decode:
    # builder has already constructed window paged indices
    # + HCA compress paged indices + the three kv_indptr (SWA / CSA / HCA)
    kv_indices, kv_indptr = self._select_paged_buffers(compress_ratio)

    if compress_ratio == 4:
        # CSA layer: translate indexer raw topk_local, then packed-write
        self._fill_csa_compress_paged_indices(topk_local_raw, kv_indices, ...)

    output = sparse_attn_v4_paged_decode(
        q, self.unified_kv, kv_indices, kv_indptr,
        self.attn_sink, self.softmax_scale,
    )
else:
    # prefill or mixed batch: keep ragged_varlen kernel
    kv_sa, topk_flat = _v4_build_sparse_inputs_batched(...)
    output = sparse_attn_ragged_varlen(q, kv_sa, ..., topk_flat, ...)
```

`is_pure_decode` is detected by the metadata builder in
`_attach_v4_per_fwd_meta` (formula in §7.4), the result is attached to
`attn_metadata`, and all layers read it directly.

---

## 9. CUDAGraph friendliness

CUDAGraph capture is strict about ops — any per-forward dynamic-shape
allocation, host/device sync, or variable shape will break capture/replay.
Key constraints in this design:

| risk                                              | mitigation                                                  |
|---------------------------------------------------|-------------------------------------------------------------|
| `torch.empty(varying_size)` inside capture        | all indices/indptr buffers preallocated in `forward_vars`   |
| `torch.tensor(scalar)` H2D triggers hipError      | indptr built via cumsum + `fill_`; no scalar H2D            |
| kernel binary depending on `K_MAX` constexpr      | no `K_MAX`; runtime `tl.range(0, kv_len, BLOCK_K)`          |
| `kv_flat_sa` shape varies per forward             | variable removed entirely; kernel reads paged `unified_kv`  |
| `topk_flat` length varies per forward             | replaced with indptr+indices; indices buffer preallocated   |
| per-layer different indexer topk → multi buffer   | layers run sequentially; per-ratio buffer reused            |
| CSA packed write via `scatter_` + bool mask       | replaced with fixed-grid Triton kernel (§6.4)               |

---

## 10. Out of scope

- **Prefill path**: still uses `sparse_attn_ragged_varlen`. Prefill does not
  need CUDAGraph, and with both N and KV being large, packed
  materialization is actually efficient.
- **CSA Indexer FP8 cache layout**: still uses
  `cp_gather_indexer_k_quant_cache`, decoupled from the sparse attn
  kernel; not merged into `unified_kv` (different dtype).

---

## 11. File path quick reference

| content                                 | path                                                                       |
|-----------------------------------------|----------------------------------------------------------------------------|
| this design doc (Chinese)               | `atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_DECODE_DESIGN.zh.md`         |
| this design doc (English)               | `atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_DECODE_DESIGN.en.md`         |
| paged_decode kernel + wrapper           | `atom/model_ops/v4_kernels/paged_decode.py`                                |
| csa_packed_write kernel + wrapper       | `atom/model_ops/v4_kernels/csa_packed_write.py`                            |
| KV cache allocation + module view bind  | `atom/model_ops/attentions/deepseek_v4_attn.py` (`allocate_per_req_cache`, `build_kv_cache_tensor`) |
| paged index construction + indptr cumsum| `atom/model_ops/attentions/deepseek_v4_attn.py` (`_attach_v4_per_fwd_meta`) |
| V4 forward dispatch                     | `atom/models/deepseek_v4.py` (`DeepseekV4Attention.forward`)               |
| indexer output (`topk_local`)           | `atom/models/deepseek_v4.py` (`Indexer.indexer_score_topk`)                |
