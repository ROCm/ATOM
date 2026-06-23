# MiniMax-M3 fp8 KV cache + gluon PA — Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the fp8 KV-cache + gluon paged-attention path for MiniMax-M3 sparse attention from `origin/ganyi/shuffle_kv_cache_fp8_eagle` onto current `origin/main` as a **first-class citizen of main's attention framework** — driven by the framework's existing builder hooks and `config.kv_cache_dtype` switch, not an env-gated side path. Excludes eagle3 speculative-decoding and bf16-MoE work.

**Architecture:** Main's attention framework defines a strict builder protocol (`AttentionMetadataBuilder`/`CommonAttentionBuilder` in `backends.py`) that the runner drives: `compute_block_bytes` → `allocate_kv_cache_tensors` → `build_kv_cache_tensor` (which calls `module.bind_kv_cache`) → `prepare_prefill`/`prepare_decode`/`build_for_cudagraph_capture`. Sparse attention is flagged by `is_indexed_sparse_attention`, carries per-step state in `attn_metadata.sparse_attention_metadata` (built only via `make_sparse_{prefill,decode}_metadata`), and runs through the custom op `torch.ops.aiter.minimax_m3_sparse_attention_native(qkv, positions, layer_name, q_size)` whose cache mutation is a hidden side effect. Main already has fp8-capable Triton sparse kernels (`FP8_KV_CACHE` constexpr, `_is_fp8_kv_cache_tensor`) and gluon decode plumbing (`run_pa_decode_gluon`, `get_recommended_splits`). **This integration makes the fp8 path select automatically when `config.kv_cache_dtype == "fp8"`**, populating real `k_scale`/`v_scale` through `KVCacheTensor`, binding page-16 SHUFFLE views inside `build_kv_cache_tensor`, inserting KV through the quantized `reshape_and_cache`/`fused_qknorm_idxrqknorm` hook, and routing decode/prefill through the gluon runners — all behind the existing op signature and metadata factories so nothing about the framework's external contract changes.

**Tech Stack:** Python, Triton, PyTorch (ROCm), AITER gluon kernels. Source branch ref: `origin/ganyi/shuffle_kv_cache_fp8_eagle`. Merge-base: `72c4dca6ce208d269ed43a511768d0ed3ecaa7a9`.

## Global Constraints

- Target branch: detached `origin/main`. Do NOT pull in eagle3 spec-decode (`aux_hidden_state_layers`, `set_aux_hidden_state_layers`, `get_eagle3_aux_hidden_state_layers`, `aux_out=`, `MAX_Q`/`max_query_len` causal logic) or `MiniMaxM3Bf16Experts` MoE — those are out of scope.
- fp8 dtype is `aiter.dtypes.fp8` (`torch.float8_e4m3fn`, max 448.0). Per-token dynamic quant scales are fp32.
- **No env gate for correctness selection.** The fp8 vs bf16 choice is the framework's existing `config.kv_cache_dtype` switch (the same switch every other MHA backend uses). The gluon SHUFFLE decode path may keep the existing `ATOM_M3_SPARSE_USE_ASM_PA` env *only as a perf toggle for the bf16 path if main already exposes it*; it must NOT be the thing that turns fp8 on/off. When `kv_cache_dtype != "fp8"`, behavior is byte-identical to current main.
- The installed AITER already provides every required symbol (`pa_decode_gluon`/`run_pa_decode_gluon`, `get_recommended_splits`, `fused_qknorm_idxrqknorm` with `asm_layout`+`k_scale`/`v_scale`, `reshape_and_cache_with_pertoken_quant`). **No AITER changes.**
- Reference source code by `git show origin/ganyi/shuffle_kv_cache_fp8_eagle:<path>` rather than retyping kernels by hand — copy the kernel bodies verbatim, then adapt only the call-site naming to main.
- ALWAYS `AITER_LOG_LEVEL=WARNING` and clear `/root/.cache/atom/*` before any server/GPU run (per CLAUDE.md).
- Use `black . && ruff check .` before each commit.

## Framework contracts this integration MUST satisfy (compatibility checklist)

These are main's framework contracts (verified against `backends.py`, `aiter_attention.py`, `model_runner.py`, `attention_mha.py`). Every task is checked against them in Self-Review.

1. **Builder protocol unchanged.** Implement fp8 behavior *inside* the existing overrides (`compute_block_bytes` `aiter_attention.py:329`, `allocate_kv_cache_tensors` :425, `build_kv_cache_tensor` :488, `prepare_*`/`build_for_cudagraph_capture`). Do not add new builder methods or change signatures. `prepare_mtp_decode` is NOT on this (MHA) builder and must not be added (M3 raises on spec-decode at :921/:1132).
2. **KVCacheTensor carries the scales.** `build_kv_cache_tensor` for the sparse branch must return `KVCacheTensor(layer_num, k_cache, v_cache, k_scale, v_scale)` with **real fp32 scales when fp8** (today it returns `k_scale=None,v_scale=None` at :516 — that is the contract gap to close), mirroring the standard MHA path that slices `runner.kv_scale[0/1, attn_idx]` (:603).
3. **Binding through `bind_kv_cache`.** Page-16 SHUFFLE views + scale binding happen inside `build_kv_cache_tensor`/`module.bind_kv_cache(...)`, not via a hand-rolled `permute` in the model. After binding the module must expose `k_cache`/`v_cache`/`k_scale`/`v_scale`/`index_cache`/`max_model_len` and (for gluon) the 5D SHUFFLE views.
4. **Insert through the quantized hook.** fp8 KV insert uses `reshape_and_cache_with_pertoken_quant`/`fused_qknorm_idxrqknorm(asm_layout=True, k_scale=, v_scale=)` (same hook family as `attention_mha.py:345`), writing quantized KV + per-token scales — never a bf16 NHD `reshape_and_cache` for the fp8 case.
5. **Metadata only via factories.** All per-step state flows through `attn_metadata.sparse_attention_metadata` built by `make_sparse_{prefill,decode}_metadata`; the gluon runners read fields off that object. No new ad-hoc metadata channel.
6. **Op signature frozen.** Sparse forward stays `torch.ops.aiter.minimax_m3_sparse_attention_native(qkv, positions, layer_name, q_size) -> [tokens, q_size]` with cache mutation as a side effect (`module_dispatch_ops.py:154`). The gluon/fp8 path lives behind it.
7. **CUDAGraph-safe.** `build_for_cudagraph_capture` (:1104) must build metadata only from preallocated `forward_vars`; gluon scratch (`exp_sums`/`max_logits`/`temporary_output`) must be allocated-in-graph or hoisted to `forward_vars`, never `torch.empty` per-call inside a captured region.
8. **Byte accounting matches reality.** `compute_block_bytes` (:411) currently counts the index_cache at `config.torch_dtype` itemsize (bf16=2) and already budgets the standard fp32 kv_scale block for all layers. If the index_cache stays bf16, leave it; if any sparse cache/scale dtype changes, update this so the runner's budget cross-check (`model_runner.py:1608`) does not fire.
9. **gluon decode call shape.** `run_pa_decode_gluon` expects k_cache as `[num_blocks, num_kv_heads, head_dim//x, block_size, x]` (SHUFFLE), `k_scale.unsqueeze(-1)` when `numel>1`, `compute_type` = fp8|bf16, and fp32 `exp_sums/max_logits` of shape `(num_seqs, num_kv_heads, max_context_partition_num, query_group_size)` (`attention_mha.py:529-568`). The M3 gluon runner must match this exactly.

## Naming map (source → main) — apply this everywhere when grafting

| Source branch | Current main |
|---|---|
| `is_minimax_m3_sparse_attention` | `is_indexed_sparse_attention` |
| `minimax_m3_index_cache` (runner attr / cache key) | `sparse_attention_index_cache` |
| `_minimax_m3_sparse_cache_next` | `_sparse_attention_cache_next` |
| manual `module.kv_cache = ...permute(...)` binding | `module.bind_kv_cache(layer_kv_cache, index_cache, max_model_len)` (extend this method; do not bypass it) |
| `make_minimax_m3_sparse_{prefill,decode}_metadata` | `make_sparse_{prefill,decode}_metadata` |

## File Structure

| File | Responsibility / change |
|---|---|
| `atom/model_ops/minimax_m3/sparse_attn.py` | Add: fused page-16 SHUFFLE KV-insert kernel + host wrapper; page-16 sparse block-table builders (decode + prefill); gluon PA decode/prefill runners matching `run_pa_decode_gluon`'s contract. Add constants `ASM_PAGE_SIZE=16`, `PAGES_PER_SPARSE_BLOCK=8`. |
| `atom/model_ops/minimax_m3/index_topk.py` | Add fused sparse-block-table emission glue (`PAGES_PER_SPARSE_BLOCK`, `EMIT_SPARSE_BT` kernel branch, `emit_sparse_block_table` arg). Exclude `MAX_Q`. |
| `atom/model_ops/attentions/aiter_attention.py` | fp8 scale alloc in `allocate_kv_cache_tensors`; page-16 SHUFFLE views + real `k_scale`/`v_scale` returned from `build_kv_cache_tensor` (contract #2/#3); `compute_block_bytes` accounting check (contract #8); CUDAGraph scratch hoist in `build_for_cudagraph_capture` (contract #7). |
| `atom/models/minimax_m3.py` | Extend `bind_kv_cache` to expose SHUFFLE views + scales (contract #3); extend `_insert_kv` to use the quantized hook when fp8 (contract #4); route `_run_prefill_sparse`/`_run_decode_sparse` through gluon runners selected by `_is_fp8_kv_cache_tensor(self.kv_cache)` / cache dtype (contract #5/#6). No env gate for fp8 selection. |
| `tests/minimax_m3/test_m3_fp8_gluon_pa.py` (new) | Contract tests: scales present in KVCacheTensor for fp8; SHUFFLE view shapes; insert round-trip; gluon-vs-Triton parity; block-table builder; byte-accounting; bf16 path unchanged. |

---

### Task 1: Add ASM page constants + fused SHUFFLE KV-insert kernel to `sparse_attn.py`

**Files:**
- Modify: `atom/model_ops/minimax_m3/sparse_attn.py` (append new symbols; do not touch the existing Triton kernels / `make_sparse_*` factories)
- Test: `tests/minimax_m3/test_m3_fp8_gluon_pa.py`

**Interfaces:**
- Produces:
  - `ASM_PAGE_SIZE = 16`, `PAGES_PER_SPARSE_BLOCK = SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE  # 8`
  - `minimax_m3_fused_qknorm_rope_kv_insert_shuffle(qkv, q_norm_weight, k_norm_weight, cos_sin_cache, positions, num_heads, num_kv_heads, rotary_dim, eps, index_q_norm_weight, index_k_norm_weight, num_index_heads, slot_mapping, kv_cache_k, kv_cache_v, index_cache, q_out, index_q_out, idx_head_dim) -> None` (Triton fallback writer; matches `aiter.fused_qknorm_idxrqknorm` oracle math, writes page-16 SHUFFLE K/V + page-128 index cache).

- [ ] **Step 1: Copy the kernel + wrapper verbatim from source**

Run (inspect, then copy the bodies into the bottom of main's `sparse_attn.py`):

```bash
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/model_ops/minimax_m3/sparse_attn.py \
  | sed -n '180,420p'
```

Copy these source symbols into main's file: `_gemma_norm_rope_head` (helper, src ~187), `_fused_qknorm_rope_kv_insert_shuffle_kernel` (src 232–356), `minimax_m3_fused_qknorm_rope_kv_insert_shuffle` (src 360–419). Add `ASM_PAGE_SIZE`/`PAGES_PER_SPARSE_BLOCK` near the existing `SPARSE_BLOCK_SIZE` definition. Do NOT alter main's existing `make_sparse_prefill_metadata`/`make_sparse_decode_metadata` or the 5 existing Triton kernels.

- [ ] **Step 2: Write the failing oracle test**

```python
# tests/minimax_m3/test_m3_fp8_gluon_pa.py
import torch, pytest
from atom.model_ops.minimax_m3 import sparse_attn as S

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

def test_constants():
    assert S.ASM_PAGE_SIZE == 16
    assert S.PAGES_PER_SPARSE_BLOCK == S.SPARSE_BLOCK_SIZE // 16 == 8

def test_insert_shuffle_writes_index_cache():
    torch.manual_seed(0)
    nt, nh, nkv, hd, idx = 4, 8, 1, 128, 128
    row = nh*hd + 2*nkv*hd + nh*idx + idx  # q|k|v|iq|ik layout
    qkv = torch.randn(nt, row, device="cuda", dtype=torch.bfloat16)
    # minimal allocations; see source wrapper docstring for shapes
    # (this test asserts the index cache is written non-zero at mapped slots)
    # ... build q_norm/k_norm/iq_norm/ik_norm weights, cos_sin, slots ...
    # Assert: after the call, index_cache[mapped slot] != 0, unmapped == 0.
    assert hasattr(S, "minimax_m3_fused_qknorm_rope_kv_insert_shuffle")
```

- [ ] **Step 3: Run test to verify the constants/symbol exist**

Run: `python -m pytest tests/minimax_m3/test_m3_fp8_gluon_pa.py::test_constants -v`
Expected: PASS (constants), and `test_insert_shuffle_writes_index_cache` at minimum imports the symbol.

- [ ] **Step 4: Lint + commit**

```bash
black atom/model_ops/minimax_m3/sparse_attn.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
ruff check atom/model_ops/minimax_m3/sparse_attn.py
git add atom/model_ops/minimax_m3/sparse_attn.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
git commit -m "feat(minimax_m3): add page-16 SHUFFLE fused KV-insert kernel + ASM constants"
```

---

### Task 2: Add page-16 sparse block-table builders + `EMIT_SPARSE_BT` glue

**Files:**
- Modify: `atom/model_ops/minimax_m3/sparse_attn.py` (block-table builders)
- Modify: `atom/model_ops/minimax_m3/index_topk.py` (fused emission glue)
- Test: `tests/minimax_m3/test_m3_fp8_gluon_pa.py`

**Interfaces:**
- Consumes: `PAGES_PER_SPARSE_BLOCK`, `ASM_PAGE_SIZE` (Task 1).
- Produces:
  - `minimax_m3_build_sparse_block_table(topk_idx, ...) -> sparse_bt [total_q, topk*8]` (decode)
  - `minimax_m3_build_sparse_block_table_prefill(...) -> sparse_bt` (prefill)
  - `minimax_m3_index_topk_decode(..., emit_sparse_block_table: bool=False)` returning `(topk_idx, sparse_bt, sparse_ctx)` when `emit_sparse_block_table=True` (requires `num_idx_heads == 1`).

- [ ] **Step 1: Copy block-table builders from source**

```bash
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/model_ops/minimax_m3/sparse_attn.py | sed -n '976,1085p'   # decode builder + kernel
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/model_ops/minimax_m3/sparse_attn.py | sed -n '1224,1330p'  # prefill builder + kernel
```

Copy `_build_sparse_block_table_kernel` + `minimax_m3_build_sparse_block_table`, and `_build_sparse_block_table_prefill_kernel` + `minimax_m3_build_sparse_block_table_prefill` into main's `sparse_attn.py`.

- [ ] **Step 2: Port ONLY the `EMIT_SPARSE_BT` emission glue into `index_topk.py`**

```bash
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/model_ops/minimax_m3/index_topk.py | sed -n '700,870p'
```

Add `PAGES_PER_SPARSE_BLOCK` import/constant, the `EMIT_SPARSE_BT`/`MASK_INIT`/`MASK_LOCAL` constexpr branches in the merge kernel, and the `emit_sparse_block_table` arg + return-tuple branch in `minimax_m3_index_topk_decode`. **Do NOT port `MAX_Q`/`max_query_len`** (eagle3 — leave main's signature otherwise intact; if main's `minimax_m3_index_topk_decode` lacks `max_query_len`, keep it absent).

- [ ] **Step 3: Write the failing builder test**

```python
def test_build_sparse_block_table_expands_x8():
    import torch
    from atom.model_ops.minimax_m3 import sparse_attn as S
    topk = 3
    total_q = 2
    topk_idx = torch.tensor([[0, 2, -1], [1, -1, -1]], dtype=torch.int32, device="cuda")
    bt = S.minimax_m3_build_sparse_block_table(topk_idx, total_q=total_q, topk=topk)
    assert bt.shape == (total_q, topk * S.PAGES_PER_SPARSE_BLOCK)  # *8 page-16 expand
    # block 0 -> phys pages 0..7; -1 stays -1
    assert bt[0, 0].item() == 0 and bt[0, 7].item() == 7
    assert (bt[0, 16:] == -1).all()
```
(Adjust the exact call signature to the copied wrapper — read its docstring first.)

- [ ] **Step 4: Run + verify**

Run: `python -m pytest tests/minimax_m3/test_m3_fp8_gluon_pa.py::test_build_sparse_block_table_expands_x8 -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
black atom/model_ops/minimax_m3/sparse_attn.py atom/model_ops/minimax_m3/index_topk.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
ruff check atom/model_ops/minimax_m3/sparse_attn.py atom/model_ops/minimax_m3/index_topk.py
git add atom/model_ops/minimax_m3/sparse_attn.py atom/model_ops/minimax_m3/index_topk.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
git commit -m "feat(minimax_m3): add page-16 sparse block-table builders + topk EMIT_SPARSE_BT glue"
```

---

### Task 3: Add gluon PA decode + prefill runners to `sparse_attn.py`

**Files:**
- Modify: `atom/model_ops/minimax_m3/sparse_attn.py`
- Test: `tests/minimax_m3/test_m3_fp8_gluon_pa.py`

**Interfaces:**
- Consumes: block-table builders (Task 2); `run_pa_decode_gluon`, `get_recommended_splits` from `atom/model_ops/base_attention.py` (confirmed present on main) / `attention_mha.py` import pattern.
- Produces:
  - `minimax_m3_sparse_attn_decode_asm(q, kv_cache_k, kv_cache_v, sparse_bt, sparse_ctx, ..., k_scale=None, v_scale=None) -> out`
  - `minimax_m3_sparse_attn_prefill_asm(...) -> out` and helper `_run_prefill_fp8_gluon(...)`.
- **Contract #9 (gluon call shape):** the decode runner must call `run_pa_decode_gluon` exactly as `attention_mha.py:568` does — k_cache viewed as `[num_blocks, num_kv_heads, head_dim//x, block_size, x]`; `k_scale`/`v_scale` `unsqueeze(-1)` when `numel>1` and `None` when bf16; `compute_type` = fp8 vs bf16 by cache dtype; `max_context_partition_num = get_recommended_splits(num_seqs, num_kv_heads)`, `context_partition_size=256`; fp32 scratch `exp_sums`/`max_logits` shape `(num_seqs, num_kv_heads, max_context_partition_num, query_group_size)` and `temporary_output` with trailing `head_size`. When grafting, copy the source runner but reconcile any arg drift against main's `attention_mha.py` signature — main is the source of truth for the gluon ABI.

- [ ] **Step 1: Copy the gluon runners from source**

```bash
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/model_ops/minimax_m3/sparse_attn.py | sed -n '1087,1222p'  # decode_asm
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/model_ops/minimax_m3/sparse_attn.py | sed -n '1332,1495p'  # prefill_asm + _run_prefill_fp8_gluon
```

Copy `minimax_m3_sparse_attn_decode_asm`, `minimax_m3_sparse_attn_prefill_asm`, `_run_prefill_fp8_gluon`. Fix imports to main's locations: import `run_pa_decode_gluon` and `get_recommended_splits` the same way `atom/model_ops/attention_mha.py` does (`from atom.model_ops.base_attention import run_pa_decode_gluon`; `from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits`).

- [ ] **Step 2: Verify the gluon entry imports resolve**

```bash
python -c "from atom.model_ops.minimax_m3.sparse_attn import minimax_m3_sparse_attn_decode_asm, minimax_m3_sparse_attn_prefill_asm; print('ok')"
```
Expected: `ok` (no ImportError).

- [ ] **Step 3: Write a gluon-vs-Triton decode parity test (fp8 path)**

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_gluon_decode_matches_triton_bf16():
    # Build a small M3 sparse decode case; run both the existing Triton
    # decode (minimax_m3_sparse_attn_decode) and the new *_asm gluon runner
    # on the SAME bf16 caches; assert outputs close (atol/rtol per gluon prec).
    ...
    torch.testing.assert_close(out_gluon, out_triton, atol=2e-2, rtol=2e-2)
```
(If a full parity harness is too heavy here, assert the gluon runner produces finite output of the correct shape on a synthetic case; full parity is covered by the e2e smoke in Task 6.)

- [ ] **Step 4: Run + verify**

Run: `python -m pytest tests/minimax_m3/test_m3_fp8_gluon_pa.py -k gluon -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
black atom/model_ops/minimax_m3/sparse_attn.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
ruff check atom/model_ops/minimax_m3/sparse_attn.py
git add atom/model_ops/minimax_m3/sparse_attn.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
git commit -m "feat(minimax_m3): add gluon PA decode/prefill runners for sparse attention"
```

---

### Task 4: fp8 scales as a first-class builder output in `aiter_attention.py`

This task closes framework contracts #2 (KVCacheTensor carries scales), #3 (bind via `build_kv_cache_tensor`), #8 (byte accounting), and #7 (CUDAGraph scratch). The fp8 path activates purely off `config.kv_cache_dtype == "fp8"` — the same switch the standard MHA branch already uses.

**Files:**
- Modify: `atom/model_ops/attentions/aiter_attention.py` (`compute_block_bytes` 411–422; `allocate_kv_cache_tensors` sparse block ~472–486; `build_kv_cache_tensor` M3 branch 502–518; `build_for_cudagraph_capture` ~1104)
- Test: `tests/minimax_m3/test_m3_fp8_gluon_pa.py`

**Interfaces:**
- Consumes: `ASM_PAGE_SIZE`, `PAGES_PER_SPARSE_BLOCK` (Task 1), main's `module.bind_kv_cache(...)`.
- Produces: runner attrs `minimax_m3_k_scale`, `minimax_m3_v_scale` (fp32, only when fp8); the sparse branch of `build_kv_cache_tensor` returns `KVCacheTensor(..., k_scale, v_scale)` with real scales for fp8 (contract #2); module exposes `k_scale`/`v_scale` page-16 views (contract #3).

- [ ] **Step 1: Add scale allocation in `allocate_kv_cache_tensors`**

Inside the `if sparse_cfg:` block (after `sparse_attention_index_cache`), add (source lines 524–540; `num_kv_heads` already computed in this method on main):

```python
            if config.kv_cache_dtype == "fp8":
                tensors["minimax_m3_k_scale"] = torch.zeros(
                    hf_config.num_hidden_layers,
                    runner.num_physical_kvcache_blocks,
                    num_kv_heads,
                    runner.physical_block_size,
                    dtype=dtypes.fp32,
                    device="cuda",
                )
                tensors["minimax_m3_v_scale"] = torch.zeros(
                    hf_config.num_hidden_layers,
                    runner.num_physical_kvcache_blocks,
                    num_kv_heads,
                    runner.physical_block_size,
                    dtype=dtypes.fp32,
                    device="cuda",
                )
```

- [ ] **Step 2: Return real scales from the sparse branch of `build_kv_cache_tensor` (contract #2/#3)**

In the `is_indexed_sparse_attention` branch (502–518), after `key_cache, value_cache = module.bind_kv_cache(...)`, replace the `k_scale=None, v_scale=None` return with page-16 scale binding:

```python
            k_scale = v_scale = None
            if config.kv_cache_dtype == "fp8":
                from atom.model_ops.minimax_m3.sparse_attn import (
                    ASM_PAGE_SIZE,
                    PAGES_PER_SPARSE_BLOCK,
                )
                # page-128 scale buffer [blocks, nkv, 128] reinterpreted as
                # [blocks*8, nkv, 16] so a page-16 block id indexes the right
                # scale row (128 == 8*16, zero-copy). Mirrors the gluon reader.
                k_scale = runner.minimax_m3_k_scale[layer_id].view(
                    runner.num_physical_kvcache_blocks * PAGES_PER_SPARSE_BLOCK,
                    module.num_kv_heads,
                    ASM_PAGE_SIZE,
                )
                v_scale = runner.minimax_m3_v_scale[layer_id].view(
                    runner.num_physical_kvcache_blocks * PAGES_PER_SPARSE_BLOCK,
                    module.num_kv_heads,
                    ASM_PAGE_SIZE,
                )
            module.k_scale = k_scale
            module.v_scale = v_scale
            return KVCacheTensor(
                layer_num=layer_id, k_cache=key_cache, v_cache=value_cache,
                k_scale=k_scale, v_scale=v_scale,
            )
```
(Source lines 566–600. `module.bind_kv_cache` is extended in Task 5 to also expose the SHUFFLE views; this task only adds the scale half so the framework registers them.)

- [ ] **Step 3: Fix `compute_block_bytes` accounting (contract #8)**

Read main's `compute_block_bytes` sparse block (411–422). It already budgets the standard fp32 kv_scale block for all `num_hidden_layers` (404–410), which **covers the sparse layers' K/V scales** — so the new `minimax_m3_k_scale`/`v_scale` need no extra term (they reuse that budget; confirm shapes match `num_hidden_layers * num_kv_heads * physical_block_size * 4`). The index_cache term (417–421) uses `config.torch_dtype` itemsize — **leave it** since this plan keeps the index cache in `config.torch_dtype` (the fp8 work here is on the main K/V cache, not the indexer cache). Add a one-line comment asserting this invariant so a future indexer-fp8 change updates it:

```python
        # NOTE: sparse fp8 K/V dequant scales reuse the standard per-layer
        # fp32 kv_scale budget above; index_cache stays config.torch_dtype.
        # If the indexer cache is ever quantized, add its bytes here.
```

- [ ] **Step 4: CUDAGraph scratch must be graph-safe (contract #7)**

Inspect `build_for_cudagraph_capture` (~1104) and the gluon decode runner from Task 3. The gluon scratch (`exp_sums`/`max_logits`/`temporary_output`) must NOT be `torch.empty`-per-call inside a captured graph. Either (a) allocate them once and stash on the builder/module (preferred — sized by max capture bs), or (b) allocate inside the capture region so they become part of the graph. Document which approach the Task 3 runner uses and ensure decode-capture reuses the same buffers `prepare_decode` uses. If main's standard MHA gluon path (`attention_mha.py:548`) already `torch.empty`s these and is graph-captured successfully, follow that exact precedent.

- [ ] **Step 5: Write contract tests**

```python
def test_fp8_scale_buffers_allocated_only_for_fp8():
    # fake runner/config kv_cache_dtype="fp8": allocate_kv_cache_tensors returns
    # minimax_m3_k_scale/v_scale fp32 of the documented shape; "bf16" -> absent.
    ...

def test_build_kv_cache_tensor_sparse_returns_scales_for_fp8():
    # fp8: the is_indexed_sparse_attention branch returns KVCacheTensor with
    # non-None k_scale/v_scale and module.k_scale/.v_scale are page-16 views
    # [blocks*8, nkv, 16]. bf16: both None (byte-identical to current main).
    ...
```

- [ ] **Step 6: Run + verify**

Run: `python -m pytest tests/minimax_m3/test_m3_fp8_gluon_pa.py -k "scale or build_kv_cache" -v`
Expected: PASS. Regression: `python -m pytest tests/ -k "aiter_attention or kv_cache" -q` (bf16 path unchanged).

- [ ] **Step 7: Lint + commit**

```bash
black atom/model_ops/attentions/aiter_attention.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
ruff check atom/model_ops/attentions/aiter_attention.py
git add atom/model_ops/attentions/aiter_attention.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
git commit -m "feat(minimax_m3): fp8 KV dequant scales as first-class builder output"
```

---

### Task 5: Integrate fp8/gluon into `MiniMaxM3SparseAttention`, selected by cache dtype

This task closes contracts #3 (binding), #4 (quantized insert hook), #5 (metadata via factories), #6 (frozen op signature). **The fp8 path is chosen by the cache dtype, not an env gate** — `_is_fp8_kv_cache_tensor(self.kv_cache)` (already on main) is the selector, exactly as main's Triton sparse kernels already switch on `FP8_KV_CACHE`.

**Files:**
- Modify: `atom/models/minimax_m3.py` (`__init__` members; `bind_kv_cache` 545–558; `_insert_kv` 574+; `_run_prefill_sparse`/`_run_decode_sparse` dispatch)
- Test: `tests/minimax_m3/test_m3_fp8_gluon_pa.py` + e2e (Task 6)

**Interfaces:**
- Consumes: Task 1 insert wrapper, Task 3 `*_asm` gluon runners, Task 4 `module.k_scale`/`v_scale`, `_is_fp8_kv_cache_tensor` (main, `sparse_attn.py`), `aiter.reshape_and_cache_with_pertoken_quant`, `aiter.fused_qknorm_idxrqknorm`.
- Produces: the M3 sparse module transparently runs fp8 gluon when the bound cache is fp8, bf16 otherwise; the custom-op signature `(qkv, positions, layer_name, q_size)` is unchanged.

- [ ] **Step 1: Add SHUFFLE-view members in `__init__`**

After main's existing `self.index_cache = torch.tensor([])` add the view/scale holders (source lines ~564–575). **Do NOT add `_use_asm_pa` as the fp8 selector.** If main already has an `ATOM_M3_SPARSE_USE_ASM_PA` perf toggle for the bf16 gluon path, leave it as-is; otherwise omit it entirely.

```python
        self.kv_cache_k = torch.tensor([])   # page-16 SHUFFLE K view of self.kv_cache
        self.kv_cache_v = torch.tensor([])   # page-16 SHUFFLE V view of self.kv_cache
        self.k_scale = self.v_scale = None    # set by build_kv_cache_tensor (Task 4)
```

- [ ] **Step 2: Extend `bind_kv_cache` to expose SHUFFLE views (contract #3)**

```bash
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/models/minimax_m3.py | sed -n '595,705p'
```
Keep main's `bind_kv_cache` permute body, and at the end derive + store the page-16 SHUFFLE K/V views (`self.kv_cache_k`, `self.kv_cache_v`) as zero-copy reinterpretations of `self.kv_cache` (`x = 16 // itemsize`), so both `build_kv_cache_tensor` (which binds scales) and the gluon runners read a single source of truth. Port `_ensure_asm_shuffle_views` only if main's lazy-derivation pattern needs it; prefer deriving eagerly inside `bind_kv_cache` since the runner calls it once at bind time. Return `self.kv_cache.unbind(1)` as today.

- [ ] **Step 3: Extend `_insert_kv` to use the quantized hook when fp8 (contract #4)**

```bash
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/models/minimax_m3.py | sed -n '644,705p'
```
Select on cache dtype, not env:

```python
        if _is_fp8_kv_cache_tensor(self.kv_cache):
            # fp8: quantized per-token KV insert into page-16 SHUFFLE cache,
            # writing dequant scales into self.k_scale/self.v_scale.
            aiter.fused_qknorm_idxrqknorm(
                ..., k_scale=self.k_scale, v_scale=self.v_scale, asm_layout=True
            )   # or reshape_and_cache_with_pertoken_quant per source
        else:
            <main's existing bf16 _insert_kv body, verbatim>
```
The index_cache write stays as on main (this plan does not quantize the indexer cache).

- [ ] **Step 4: Route prefill/decode through gluon runners, selected by dtype (contract #5/#6)**

In `_run_prefill_sparse` / `_run_decode_sparse`, build the page-16 sparse block table (Task 2, via `make_sparse_*_metadata` fields — contract #5) and dispatch:

```python
        if _is_fp8_kv_cache_tensor(self.kv_cache):
            return minimax_m3_sparse_attn_decode_asm(   # / _prefill_asm
                q, self.kv_cache_k, self.kv_cache_v, sparse_bt, sparse_ctx,
                ..., k_scale=self.k_scale, v_scale=self.v_scale,
            )
        # bf16: existing path. If main exposes the bf16 gluon (ASM) runner under
        # its perf env, preserve that branch; else keep the native Triton call.
        return <main's existing bf16 sparse decode/prefill call>
```
`forward` keeps `torch.ops.aiter.minimax_m3_sparse_attention_native` unchanged (contract #6) — all selection happens inside the impl.

- [ ] **Step 5: Contract tests**

```python
def test_bf16_path_unchanged_when_not_fp8():
    # With a bf16-bound cache, _insert_kv / dispatch take the original main
    # branch (assert the fp8 hook is NOT called, e.g. via monkeypatch spy).
    ...

def test_fp8_dispatch_selected_by_cache_dtype():
    # With an fp8-bound cache + non-None k_scale/v_scale, _run_decode_sparse
    # routes to minimax_m3_sparse_attn_decode_asm (spy on the gluon runner).
    ...
```
Run the full M3 unit suite (bf16 unchanged): `python -m pytest tests/ -k minimax_m3 -q` — Expected: PASS.

- [ ] **Step 6: Lint + commit**

```bash
black atom/models/minimax_m3.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
ruff check atom/models/minimax_m3.py
git add atom/models/minimax_m3.py tests/minimax_m3/test_m3_fp8_gluon_pa.py
git commit -m "feat(minimax_m3): fp8 gluon sparse attention selected by cache dtype"
```

---

### Task 6: End-to-end GPU validation (bf16 gluon + fp8 gluon)

**Files:**
- Use: `/home/gyu_qle/serve_minimax.sh` (open in IDE) as the launch reference.

- [ ] **Step 1: bf16 baseline — confirm parity with current main**

```bash
rm -rf /root/.cache/atom/*
AITER_LOG_LEVEL=WARNING bash /home/gyu_qle/serve_minimax.sh   # default bf16 cache
```
Confirm server loads (check `rocm-smi --showmemuse` VRAM% > 0, not just `/health`), run a few prompts, record outputs. This is the bf16 path that must remain byte-identical to pre-change main.

- [ ] **Step 2: fp8 path — selected automatically by `--kv_cache_dtype fp8`**

```bash
rm -rf /root/.cache/atom/*
AITER_LOG_LEVEL=WARNING bash /home/gyu_qle/serve_minimax.sh --kv_cache_dtype fp8
```
No special env required — the dtype switch alone activates the fp8 gluon path (contracts #2/#4). Confirm: fp8 scales allocate (Task 4), the gluon decode/prefill runners execute (add a debug log or check a counter), and output is coherent. Verify VRAM is lower than the bf16 run (fp8 KV cache is half the bytes).

- [ ] **Step 3: confirm dtype is the sole selector**

Grep the running config / logs to confirm the fp8 path was entered because the bound cache is fp8 (`_is_fp8_kv_cache_tensor` true), not because of any env flag. If main carries an `ATOM_M3_SPARSE_USE_ASM_PA` perf toggle, run fp8 once with it unset to prove fp8 still activates from the dtype alone.

- [ ] **Step 4: Accuracy check**

Run the M3 `lm_eval` per `/ci-pr-guide` thresholds for fp8 vs bf16; record the delta.

- [ ] **Step 5: Commit results doc (optional)**

```bash
git add docs/  # if a results note is added
git commit -m "docs(minimax_m3): fp8 gluon PA validation results"
```

---

## Self-Review

- **Spec coverage:** fp8 cache scales (Task 4), gluon PA decode/prefill (Task 3), SHUFFLE KV insert (Task 1), block-table glue (Task 2), dtype-driven integration (Task 5), validation (Task 6). All covered.
- **Framework-contract coverage (the point of this revision):**
  - #1 builder protocol unchanged → Task 4 edits only existing overrides; no new methods. ✓
  - #2 KVCacheTensor carries scales → Task 4 Step 2. ✓
  - #3 bind via `build_kv_cache_tensor`/`bind_kv_cache` (no hand permute bypass) → Task 4 Step 2 + Task 5 Step 2. ✓
  - #4 quantized insert hook → Task 5 Step 3. ✓
  - #5 metadata via `make_sparse_*` factories → Task 5 Step 4. ✓
  - #6 frozen op signature → Task 5 Step 4 (forward unchanged). ✓
  - #7 CUDAGraph scratch graph-safe → Task 4 Step 4. ✓
  - #8 byte accounting → Task 4 Step 3. ✓
  - #9 gluon call shape matches `attention_mha.py` → Task 3 Interfaces + Step 1. ✓
- **No env gate for fp8:** selection is `config.kv_cache_dtype == "fp8"` / `_is_fp8_kv_cache_tensor`, mirroring main's existing `FP8_KV_CACHE` Triton switch. Validated in Task 6 Step 3. The bf16 path is byte-identical to current main (regression tests in Tasks 4–5).
- **Excluded per scope:** eagle3 (`aux_*`, `MAX_Q`), bf16-MoE experts — carved out in Task 2 and Global Constraints.
- **Naming consistency:** runner attrs `minimax_m3_k_scale`/`v_scale` reused identically in Tasks 4 & 5; main-side names per the naming map.
- **High-risk files:** `aiter_attention.py` (Task 4) and `minimax_m3.py` (Task 5) — main drifted structurally; both tasks extend main's API in place rather than copy source verbatim. Tasks 1–3 are clean additions.
- **Hard gate verified:** `run_pa_decode_gluon`/`get_recommended_splits` present on main; AITER symbols present. No cross-repo work.
- **Placeholder scan:** kernel bodies are pulled via `git show` (verbatim copy) rather than retyped; the `...` in test stubs mark fixtures the implementer fills, not missing logic. Acceptable for a port where the source is the reference.
