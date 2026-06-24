# MiniMax-M3 fp8 KV cache + gluon PA — Port Implementation Plan (subclass architecture)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the fp8 KV-cache + gluon paged-attention path for MiniMax-M3 sparse attention from `origin/ganyi/shuffle_kv_cache_fp8_eagle` onto current `origin/main`, **rebuilt as a first-class `PagedAttentionImpl` subclass** rather than a model-level custom op. The M3 model becomes a normal `Attention(...)` consumer that calls `self.attn(q, k, v, positions, **model_kwargs)` in standard atom style; all sparse/fp8/gluon behavior lives inside the impl subclass and the existing `AiterAttentionMetadataBuilder`. Excludes eagle3 speculative decoding and bf16-MoE.

**Architecture (the redirection):** Instead of M3 owning a bespoke custom op (`torch.ops.aiter.minimax_m3_sparse_attention_native`) plus a hand-rolled `bind_kv_cache` permute and a model-level prefill/decode dispatch, we:

1. **Reuse the generic per-layer custom op.** `torch.ops.aiter.unified_attention_with_output_base` (`base_attention.py:345`, `mark_spliting_op(is_custom=True)`) already provides the torch.compile boundary, functionalization-clone suppression for in-place KV writes, and Dynamo opacity for data-dependent sparse ops. It looks up `self = static_forward_context[layer_name]` and calls `self.impl.forward(query, key, value, position, q_scale, qkv)`. **M3 gets the compile boundary for free by becoming a normal `Attention(...).impl`** — so the bespoke `minimax_m3_sparse_attention_native` op is dropped entirely.
2. **New impl subclass `SparseMHAPagedAttentionImpl(PagedAttentionImpl)`** in `atom/model_ops/attention_mha.py`. It overrides:
   - `rope_cache(...)` → M3's fused norm+rope+KV-insert+index-insert (`aiter.fused_qknorm_idxrqknorm`, writing main's page-16 SHUFFLE KV + index cache + per-token fp32 scales). Produces the rotated `query` (parent's normal return) and **stashes `index_q` on `self`** for the dispatch step (the parent's fixed return tuple has no slot for index_q; the forward is single-threaded per layer so a `self.` attribute is safe).
   - `dispatch_backend(...)` (prefill + decode) → index top-k → page-16 sparse block-table → gluon PA runners (`run_pa_decode_gluon` for decode; sparse prefill runner). fp8 vs bf16 is selected by the cache dtype, not an env gate.
3. **All indexer state inside the subclass** (confirmed decision): `index_q_norm`/`index_k_norm`/`index_rotary_emb`/`index_q_size`/topk params/`index_cache` handle live on the impl instance, not the model.
4. **Reuse `AiterAttentionMetadataBuilder` unchanged in shape.** `build_kv_cache_tensor`'s sparse branch becomes **additive**: it runs the standard MHA binding (5D SHUFFLE `k_cache`/`v_cache`, real `k_scale`/`v_scale` from `runner.kv_scale`) *and also* binds `index_cache` onto the impl. `compute_block_bytes`/`allocate_kv_cache_tensors`/`prepare_*`/`build_for_cudagraph_capture` keep their signatures; sparse metadata still flows via `make_sparse_{prefill,decode}_metadata` on `attn_metadata.sparse_attention_metadata`.
5. **Standard `PagedAttentionImpl.forward`** (confirmed decision): no override of `forward`; the subclass plugs in only through `rope_cache` + `dispatch_backend`, which `forward_impl` already calls in sequence.

**Impl-class selection mechanism (decision in this plan):** `Attention.__init__` (`paged_attention.py:82`) sets `impl_cls = self.attn_backend.get_impl_cls()`. To plug in the subclass **without a whole new backend or touching every other model**, add an optional `impl_cls` kwarg to `Attention.__init__`: `impl_cls = impl_cls or self.attn_backend.get_impl_cls()`. M3 passes `impl_cls=SparseMHAPagedAttentionImpl`. This keeps `AiterAttentionMetadataBuilder` (and its `get_impl_cls`/`get_builder_cls`) as the backend, reused verbatim — only the impl is swapped. Rejected alternative: a dedicated `SparseAiterAttentionBackend` whose `get_impl_cls()` returns the subclass — heavier (new backend registration in the selector) for no benefit, since the builder is identical.

**Tech Stack:** Python, Triton, PyTorch (ROCm), AITER gluon kernels. Source branch ref: `origin/ganyi/shuffle_kv_cache_fp8_eagle`. Merge-base: `72c4dca6ce208d269ed43a511768d0ed3ecaa7a9`.

## Global Constraints

- Target branch: detached `origin/main`. Do NOT pull in eagle3 spec-decode (`aux_hidden_state_layers`, `set_aux_hidden_state_layers`, `get_eagle3_aux_hidden_state_layers`, `aux_out=`, `MAX_Q`/`max_query_len` causal logic) or `MiniMaxM3Bf16Experts` MoE — out of scope.
- fp8 dtype is `aiter.dtypes.fp8` (`torch.float8_e4m3fn`, max 448.0). Per-token dynamic quant scales are fp32.
- **fp8 vs bf16 selection is the framework's `config.kv_cache_dtype` switch / cache dtype**, exactly as standard MHA. No env gate decides correctness. When `kv_cache_dtype != "fp8"`, the sparse path must remain functional on bf16 (the gluon runner takes `compute_type=bf16`).
- **NEVER modify `@support_torch_compile`-decorated model code in a way that breaks Dynamo tracing** (CLAUDE.md). `MiniMaxM3Model` is decorated (`minimax_m3.py:862`). The model-file changes here are confined to the attention *module's* construction/forward (building `Attention(...)` and calling it) — the same pattern Qwen3 uses — and must not introduce data-dependent Python branches into the traced graph. The custom op boundary keeps the sparse internals opaque to Dynamo.
- The installed AITER already provides every required symbol (`pa_decode_gluon`/`run_pa_decode_gluon`, `get_recommended_splits`, `fused_qknorm_idxrqknorm` with `asm_layout`+`k_scale`/`v_scale`, `reshape_and_cache_with_pertoken_quant`). **No AITER changes.**
- Reference source code by `git show origin/ganyi/shuffle_kv_cache_fp8_eagle:<path>` rather than retyping kernels — copy kernel bodies verbatim, adapt only call-site naming to main.
- ALWAYS `AITER_LOG_LEVEL=WARNING` and clear `/root/.cache/atom/*` before any server/GPU run. `black . && ruff check .` before each commit.

## Framework contracts this integration MUST satisfy

Verified against `base_attention.py`, `paged_attention.py`, `attention_mha.py`, `attentions/aiter_attention.py`, `qwen3.py`.

1. **Generic op is the boundary.** Sparse forward routes through `unified_attention_with_output_base` → `self.impl.forward(query, key, value, position, q_scale, qkv)`. M3 passes `qkv=` (packed) via `model_kwargs` because the `rope_cache` override needs the packed tensor (GemmaRMSNorm branch, `attention_mha.py:222-261`). The bespoke `minimax_m3_sparse_attention_native` op is **deleted**.
2. **`forward`/`forward_impl` unchanged.** The subclass overrides only `rope_cache` and `dispatch_backend`. `forward_impl` (`attention_mha.py:163`) calls `rope_cache` (186) then `dispatch_backend` (190) — the subclass plugs into both, nothing else.
3. **Builder reused, binding additive.** `AiterAttentionMetadataBuilder` is the backend (selected by `get_attn_backend`). `build_kv_cache_tensor`'s `is_indexed_sparse_attention` branch (`aiter_attention.py:502-518`) becomes additive: run the standard MHA binding (5D SHUFFLE `k_cache`/`v_cache` + real `k_scale`/`v_scale`, as in :584-615) **and** bind `index_cache` (+ `max_model_len`) onto the impl. Returns `KVCacheTensor(layer_num, k_cache, v_cache, k_scale, v_scale)` with real fp32 scales when fp8 (closes the `k_scale=v_scale=None` gap at :516).
4. **Insert through the quantized hook.** fp8 KV insert uses `fused_qknorm_idxrqknorm(asm_layout=True, k_scale=, v_scale=)` / `reshape_and_cache_with_pertoken_quant` inside the `rope_cache` override (same hook family as `attention_mha.py:344-354`), writing quantized SHUFFLE KV + per-token scales + the index cache. Never a bf16 NHD insert for the fp8 case.
5. **Metadata only via factories.** Per-step sparse state flows through `attn_metadata.sparse_attention_metadata` built by `make_sparse_{prefill,decode}_metadata`; the override reads fields off it (block tables, context lens, qo_indptr, topk params). No new metadata channel.
6. **index_q threading.** `rope_cache` stashes `self._index_q` (and any index scale) for `dispatch_backend` to consume, then clears it at the end of `dispatch_backend`. Safe because per-layer forward is single-threaded and the op boundary serializes it.
7. **CUDAGraph-safe decode.** `build_for_cudagraph_capture` builds decode metadata from preallocated `forward_vars`. gluon scratch (`exp_sums`/`max_logits`/`temporary_output`) must be hoisted to `forward_vars` or allocated outside the captured region — never `torch.empty` per-call inside capture. (Source allocates per-call at `attention_mha.py:548-559`; for decode-capture this must be pre-hoisted.)
8. **Byte accounting matches reality.** `compute_block_bytes` (`aiter_attention.py:411-422`) must count: page-16 SHUFFLE KV at fp8 itemsize (1) when fp8, the fp32 kv_scale block, and the index_cache at its real dtype. Keep the runner budget cross-check (`model_runner.py`) from firing.
9. **gluon decode call shape.** `run_pa_decode_gluon` (`base_attention.py:78-126`) expects SHUFFLE `k_cache` `[num_blocks, num_kv_heads, head_dim//x, block_size, x]`, `k_scale.unsqueeze(-1)` when `numel>1`, `compute_type`=fp8|bf16, fp32 `exp_sums/max_logits` of shape `(num_seqs, num_kv_heads, max_context_partition_num, query_group_size)`. The override must match exactly.

## Naming map (source → main) — apply everywhere when grafting

| Source branch | Current main |
|---|---|
| `is_minimax_m3_sparse_attention` | `is_indexed_sparse_attention` |
| `minimax_m3_index_cache` (runner attr / cache key) | `sparse_attention_index_cache` |
| `_minimax_m3_sparse_cache_next` | `_sparse_attention_cache_next` |
| `torch.ops.aiter.minimax_m3_sparse_attention_native(qkv, positions, layer_name, q_size)` | **deleted** — replaced by `Attention(...).forward(q,k,v,positions, qkv=qkv)` → `unified_attention_with_output_base` → `self.impl.forward(...)` |
| `sparse_attention_forward_impl` (`module_dispatch_ops.py:154`) model-level dispatch | **deleted** — logic moves into `SparseMHAPagedAttentionImpl.rope_cache` + `.dispatch_backend` |
| manual `module.kv_cache = ...permute(...)` binding | `module.bind_kv_cache(...)` extended additively in `build_kv_cache_tensor` |
| `make_minimax_m3_sparse_{prefill,decode}_metadata` | `make_sparse_{prefill,decode}_metadata` |

## File Structure

| File | Responsibility / change |
|---|---|
| `atom/model_ops/attention_mha.py` | **NEW `SparseMHAPagedAttentionImpl(PagedAttentionImpl)`**: `__init__` stores indexer submodules + topk params + sparse constants; `rope_cache` override (fused norm+rope+SHUFFLE-KV-insert+index-insert via `fused_qknorm_idxrqknorm`, stash `self._index_q`); `dispatch_backend` override (index topk → page-16 sparse block table → gluon decode/prefill, fp8|bf16 by cache dtype). |
| `atom/model_ops/paged_attention.py` | Add optional `impl_cls` kwarg to `Attention.__init__`: `impl_cls = impl_cls or self.attn_backend.get_impl_cls()`. No other change. |
| `atom/model_ops/minimax_m3/sparse_attn.py` | Add: page-16 SHUFFLE constants (`ASM_PAGE_SIZE=16`, `PAGES_PER_SPARSE_BLOCK=8`); page-16 sparse block-table builders (decode + prefill); gluon PA decode/prefill runners matching `run_pa_decode_gluon`. (Kernels copied verbatim from source.) |
| `atom/model_ops/minimax_m3/index_topk.py` | Add `EMIT_SPARSE_BT` emission glue (`PAGES_PER_SPARSE_BLOCK`, kernel branch, `emit_sparse_block_table` arg + return-tuple branch). Exclude `MAX_Q`/`max_query_len`. |
| `atom/model_ops/attentions/aiter_attention.py` | `build_kv_cache_tensor` sparse branch → additive (standard binding + index_cache bind + real scales); `allocate_kv_cache_tensors` fp8 scale + index alloc; `compute_block_bytes` accounting (contract #8); `build_for_cudagraph_capture` gluon scratch hoist (contract #7). Builder shape unchanged. |
| `atom/models/minimax_m3.py` | Rewrite `MiniMaxM3SparseAttention` to construct `Attention(..., impl_cls=SparseMHAPagedAttentionImpl, rotary_emb=..., q_norm=, k_norm=, config=, prefix=, <indexer kwargs>)` and `forward` to call `o = self.attn(q, k, v, positions, qkv=qkv, **model_kwargs)` then `o_proj(o)` (Qwen3 pattern). Delete the bespoke op call, `minimax_m3_sparse_attention_native`, `_run_prefill_sparse`/`_run_decode_sparse`, and the manual permute in `bind_kv_cache`. |
| `tests/minimax_m3/test_m3_fp8_gluon_pa.py` (new) | Subclass selection (M3 `Attention.impl` is `SparseMHAPagedAttentionImpl`); scales present in `KVCacheTensor` for fp8; SHUFFLE view shapes; fused-insert round-trip; block-table x8 expand; gluon-vs-Triton parity; bf16 path unchanged. |

---

### Task 0: Pin source line ranges + write the subclass skeleton (no behavior yet)

**Files:** Modify `atom/model_ops/attention_mha.py`, `atom/model_ops/paged_attention.py`; new `tests/minimax_m3/test_m3_fp8_gluon_pa.py`.

- [ ] **Step 0.1: Capture exact source ranges** for every kernel/wrapper to be grafted, so later tasks copy verbatim:
```bash
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/model_ops/minimax_m3/sparse_attn.py | grep -n "def \|_kernel\|PAGES_PER_SPARSE_BLOCK\|ASM_PAGE_SIZE\|def run_pa\|build_sparse_block_table"
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/model_ops/minimax_m3/index_topk.py  | grep -n "EMIT_SPARSE_BT\|emit_sparse_block_table\|def minimax_m3_index_topk"
git show origin/ganyi/shuffle_kv_cache_fp8_eagle:atom/models/minimax_m3.py | grep -n "fused_qknorm_idxrqknorm\|reshape_and_cache\|run_pa_decode_gluon\|index_topk\|_run_prefill_sparse\|_run_decode_sparse\|bind_kv_cache\|_insert_kv"
```
Record the line ranges in a scratch comment block at the top of the test file.

- [ ] **Step 0.2: Add `impl_cls` kwarg to `Attention.__init__`** (`paged_attention.py`):
```python
def __init__(self, ..., impl_cls=None, **kwargs):
    ...
    impl_cls = impl_cls or self.attn_backend.get_impl_cls()
    self.impl = impl_cls(num_heads, head_dim, scale, num_kv_heads, alibi_slopes,
                         kv_cache_dtype, layer_num, mla_modules, sinks, sliding_window,
                         rotary_emb, dtype, q_norm, k_norm, **kwargs)
```
The `**kwargs` already flows through to the impl constructor — indexer kwargs ride along.

- [ ] **Step 0.3: Skeleton `SparseMHAPagedAttentionImpl(PagedAttentionImpl)`** in `attention_mha.py` — constructor pops indexer kwargs from `**kwargs` and stores them; `rope_cache`/`dispatch_backend` call `super()` (so the bf16 standard path works before the sparse logic lands). This compiles and is the TDD floor.

- [ ] **Step 0.4: Failing test — subclass is selected.** Construct an M3 `Attention(..., impl_cls=SparseMHAPagedAttentionImpl)` under mocked AITER/config and assert `isinstance(attn.impl, SparseMHAPagedAttentionImpl)`. RED → implement 0.2/0.3 → GREEN.

- [ ] **Step 0.5: Lint + commit** (`feat(attn): SparseMHAPagedAttentionImpl skeleton + Attention impl_cls override`).

---

### Task 1: Port page-16 constants + fused SHUFFLE KV-insert into `sparse_attn.py`

**Files:** Modify `atom/model_ops/minimax_m3/sparse_attn.py`; test `tests/minimax_m3/test_m3_fp8_gluon_pa.py`.

- [ ] Add `ASM_PAGE_SIZE = 16`, `PAGES_PER_SPARSE_BLOCK = SPARSE_BLOCK_SIZE // 16` (=8). Copy the fused page-16 SHUFFLE KV-insert kernel + host wrapper from source (range from Step 0.1). Adapt only naming.
- [ ] Test: insert a known K/V (fp8 + bf16), read back via `cp_mha_gather_cache(..., kv_cache_layout="SHUFFLE")`, assert round-trip within fp8 tolerance. RED→GREEN→lint→commit.

---

### Task 2: Port page-16 sparse block-table builders + topk EMIT glue

**Files:** Modify `sparse_attn.py`, `index_topk.py`; test file.

- [ ] Copy `_build_sparse_block_table_kernel` + `minimax_m3_build_sparse_block_table` (decode) and the prefill variant from source. Copy `EMIT_SPARSE_BT` constexpr branch + `emit_sparse_block_table` arg into `minimax_m3_index_topk_decode`/`minimax_m3_index_topk`. **Exclude `MAX_Q`/`max_query_len`.**
- [ ] Test `test_build_sparse_block_table_expands_x8` (x8 page-16 expansion, `-1` passthrough). RED→GREEN→lint→commit.

---

### Task 3: Port gluon PA decode + prefill runners into `sparse_attn.py`

**Files:** Modify `sparse_attn.py`; test file.

- [ ] Copy the gluon decode runner (`run_pa_decode_gluon` + `get_recommended_splits` call shape, scratch tensors) and the sparse prefill runner from source. Match contract #9 exactly (SHUFFLE k_cache, `k_scale.unsqueeze(-1)` when numel>1, `compute_type`, fp32 scratch shapes).
- [ ] Test: gluon decode output vs the existing Triton sparse decode reference on a small case, both fp8 and bf16 cache, parity within tolerance. RED→GREEN→lint→commit.

---

### Task 4: Implement `SparseMHAPagedAttentionImpl.rope_cache` override

**Files:** Modify `attention_mha.py`; test file.

- [ ] Override `rope_cache`: read `k_cache/v_cache/k_scale/v_scale/index_cache` off the impl (bound by build_kv_cache_tensor); call `aiter.fused_qknorm_idxrqknorm(asm_layout=True, k_scale=, v_scale=, ...)` to do norm+rope+SHUFFLE-KV-insert+index-insert in one shot; return rotated `query` (parent contract) and stash `self._index_q` (+ index scale). When `kv_cache_dtype != "fp8"`, fall through to the bf16 fused path (no quant args).
- [ ] Test: after `rope_cache`, the SHUFFLE KV cache + index cache + scales hold the expected quantized values; `self._index_q` is populated. RED→GREEN→lint→commit.

---

### Task 5: Implement `SparseMHAPagedAttentionImpl.dispatch_backend` override (prefill + decode)

**Files:** Modify `attention_mha.py`; test file.

- [ ] Override `dispatch_backend`: read `attn_metadata.sparse_attention_metadata`; for decode call `minimax_m3_index_topk_decode(emit_sparse_block_table=True)` → `minimax_m3_build_sparse_block_table` → `run_pa_decode_gluon`; for prefill the prefill counterparts. Consume `self._index_q`, then clear it. Select fp8|bf16 `compute_type` from cache dtype. Return the attention output in `[tokens, q_size]` shape the parent expects.
- [ ] Test: end-to-end impl forward (mocked metadata) produces correct-shaped output and matches the Triton sparse reference on a tiny case. RED→GREEN→lint→commit.

---

### Task 6: Make `AiterAttentionMetadataBuilder` binding additive + fp8 alloc/accounting

**Files:** Modify `atom/model_ops/attentions/aiter_attention.py`; test file.

- [ ] `build_kv_cache_tensor` sparse branch: run standard MHA binding (5D SHUFFLE k/v + real `k_scale`/`v_scale` from `runner.kv_scale`) AND bind `index_cache`/`max_model_len` onto the impl; return `KVCacheTensor(layer_num, k_cache, v_cache, k_scale, v_scale)` (real scales when fp8).
- [ ] `allocate_kv_cache_tensors`: ensure fp8 kv_scale + `sparse_attention_index_cache` allocations exist for the sparse layers.
- [ ] `compute_block_bytes`: count SHUFFLE KV at fp8 itemsize when fp8 + fp32 scale block + index_cache real dtype (contract #8).
- [ ] `build_for_cudagraph_capture`: hoist gluon scratch to `forward_vars` (contract #7).
- [ ] Tests: KVCacheTensor scales non-None for fp8; SHUFFLE shapes correct; impl exposes `index_cache`; byte accounting matches; bf16 path byte-identical to current main. RED→GREEN→lint→commit.

---

### Task 7: Rewrite `MiniMaxM3SparseAttention` to atom style + delete bespoke op

**Files:** Modify `atom/models/minimax_m3.py`, `atom/model_ops/module_dispatch_ops.py` (delete `sparse_attention_forward_impl`), and wherever `minimax_m3_sparse_attention_native` is registered.

- [ ] `__init__`: build `self.attn = Attention(num_heads, head_dim, scale, num_kv_heads, kv_cache_dtype, layer_num, use_mla=False, impl_cls=SparseMHAPagedAttentionImpl, rotary_emb=..., q_norm=..., k_norm=..., config=atom_config, prefix=..., index_q_norm=..., index_k_norm=..., index_rotary_emb=..., index_q_size=..., topk=..., init_blocks=..., local_blocks=...)`. Keep `is_indexed_sparse_attention=True` flag wherever the builder reads it.
- [ ] `forward`: `qkv = self.qkv_proj(hidden_states); o = self.attn(q, k, v, positions, qkv=qkv, **model_kwargs); return self.o_proj(o)` (Qwen3 pattern; q/k/v split only as the parent expects — pass `qkv=` for the rope_cache override).
- [ ] Delete `minimax_m3_sparse_attention_native`, `sparse_attention_forward_impl`, `_run_prefill_sparse`, `_run_decode_sparse`, and the manual permute in the model's `bind_kv_cache` (binding now happens in `build_kv_cache_tensor`).
- [ ] Verify `MiniMaxM3Model` `@support_torch_compile` still traces (no new data-dependent Python branch in the traced module). Lint→commit.

---

### Task 8: GPU smoke + accuracy validation

- [ ] `rm -rf /root/.cache/atom/*`; `AITER_LOG_LEVEL=WARNING python -m atom.examples.simple_inference --model <m3> --kv_cache_dtype fp8` — confirm VRAM via `rocm-smi --showmemuse`, sane output.
- [ ] Repeat with `--kv_cache_dtype bf16` to confirm the bf16 sparse path still works through the subclass.
- [ ] `lm_eval` per `/ci-pr-guide` thresholds (fp8 within tolerance of bf16 / source branch).
- [ ] Full `python -m pytest tests/` green; `black . && ruff check .`. Final commit.

---

## Self-Review checkpoints (run after each task)

- Does the change keep `forward`/`forward_impl`/builder signatures intact (only `rope_cache`+`dispatch_backend`+additive bind touched)? (contracts #2, #3)
- Is fp8 selected by cache dtype, never an env gate? (Global Constraints)
- Is `index_q` threaded via `self._index_q` and cleared after dispatch? (contract #6)
- Is the bespoke op fully deleted and the generic op the only boundary? (contract #1)
- Is the bf16 sparse path still functional and the non-sparse MHA path byte-identical? (Task 6/8)
- gluon decode call shape matches contract #9 exactly?
