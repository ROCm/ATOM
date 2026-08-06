# Kimi-K3 KDA prefill: kernel fusion design

**Date:** 2026-08-06
**Status:** approved (design); implementation pending
**Scope:** the KDA **prefill** branch only (`atom/models/kimi_k3.py:1118-1157`)
**Vendored from:** flash-linear-attention 0.5.2

## Problem

Profiling Kimi-K3 shows a cluster of small kernels around `chunk_kda` on the
prefill path that do no useful math: a state gather before the call, and a
scatter plus an output copy after it. The decode branch already avoids all of
this — `fused_sigmoid_gating_delta_rule_update`
(`atom/model_ops/fla_ops/fused_sigmoid_gating.py:196`) gathers the initial
state, writes the final state back inplace, and writes the recurrence output
straight into the caller's buffer, all in one launch. Prefill still calls stock
`fla.ops.kda.chunk_kda`, which supports none of that, so the model does it in
Python around the call.

### Inventory of removable work (all seven confirmed against the profile)

| # | What | Site | Cost |
|---|------|------|------|
| 1 | `gather_kda_initial_state` | `kimi_k3.py:1140` | 1 launch + `[N,HV,V,K]` fp32 alloc |
| 2 | `beta.float()` | `kimi_k3.py:1031` | d2d cast copy |
| 3 | `fused_beta_sigmoid` allocates fp32 regardless of input dtype | fla `ops/common/gate.py:59` | makes #2 redundant |
| 4 | `@input_guard` `.contiguous()` on every tensor arg | fla `utils/_decorators.py` | silent copies; q/k/v/g arrive as `rearrange` views |
| 5 | `o = torch.zeros_like(v_new)` | fla `ops/gla/chunk.py:982` | alloc + zero-fill |
| 6 | `ssm_state[state_indices] = last_state` | `kimi_k3.py:1156` | index_copy_ |
| 7 | `out.copy_(kda_out.squeeze(0))` | `kimi_k3.py:1157` | d2d copy |

Items 1 and 6 fold into the h-kernel's `h0` load / `ht` store as an indexed
gather and scatter. Items 5 and 7 fold into an `o=` out-parameter on the
output kernel. Items 2 and 3 collapse to nothing. Item 4 disappears with
`@input_guard`.

### Baseline: the forward chain as ATOM calls it

`chunk_kda` runs nine triton launches on ATOM's flag combination
(`use_qk_l2norm_in_kernel`, `use_gate_in_kernel`, `use_beta_sigmoid_in_kernel`,
`safe_gate`, `state_v_first`, `disable_recompute`, varlen `cu_seqlens`):

1. `l2norm_fwd(q)`
2. `l2norm_fwd(k)`
3. `fused_beta_sigmoid_fwd(beta)`
4. `kda_gate_chunk_cumsum`
5. `chunk_kda_fwd_kernel_intra_sub_chunk`
6. `chunk_kda_fwd_kernel_inter_solve_fused`
7. `recompute_w_u_fwd_kda_kernel` (`STORE_QG=True`, from `disable_recompute=True`)
8. `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (`STATE_V_FIRST=True`)
9. `chunk_gla_fwd_kernel_o` (`STATE_V_FIRST=True`)

Only 8 and 9 are modified by this design.

**No `@dispatch` backend is active on this platform.** `flash_kda` is not
installed and `tilelang` requires nvcc, so `@dispatch('kda')` falls through to
the default triton body. The triton path is the real path and is what we
vendor. (If `flash_kda` is ever installed, it would satisfy the verifier and
divert — see Risks.)

## Approach

Vendor only the files whose kernels change. Thread three new optional
arguments down to them. Everything else keeps importing from fla.

Rejected alternatives:

- **Deeper merging** (fold `l2norm` ×2 and `fused_beta_sigmoid` into the gate
  kernel) — saves 3 more launches, but they are elementwise ops over
  differently-shaped tensors with different tiling, so it means a hand-written
  fused kernel with no upstream counterpart to diff against. Revisit after this
  lands and the trace is re-measured.
- **Python-side buffer reuse only** — does not work. The launches are the cost,
  and `ssm_state[idx] = last_state` is a kernel regardless of allocation.

## Architecture

New package, deliberately separate from the existing GDN files in
`atom/model_ops/fla_ops/`:

```
atom/model_ops/fla_ops/kda/
  __init__.py          exports chunk_kda
  chunk.py             inference-only entry (no autograd / input_guard / dispatch)
  chunk_fwd.py         orchestrator; threads the three new args
  chunk_delta_h.py     h-kernel + indexed h0 gather & ht scatter   <- main change
  chunk_o_gk.py        gla output kernel + o= out-param            <- main change
```

Imported unmodified from fla: `l2norm_fwd`, `fused_beta_sigmoid`,
`prepare_chunk_indices`, `kda_gate_chunk_cumsum`, `chunk_kda_fwd_intra`,
`recompute_w_u_fwd`.

### Why a separate package, and not `chunk_delta_h_vk.py`

ATOM already vendors `atom/model_ops/fla_ops/chunk_delta_h_vk.py`, which is a
V-first h-kernel — superficially the exact thing KDA needs, since KDA runs
`state_v_first=True`. **It must not be reused.** It applies `exp` (base-e),
while KDA pre-scales its gate by `RCP_LN2` (fla `ops/kda/chunk_fwd.py:47-51`)
and requires `exp2`. Feeding KDA's gate to the base-e kernel computes
`decay^(1/ln2) = decay^1.4427` — wrong decay, no error raised, silently
degraded output.

Both files are internally consistent (ATOM's GDN caller at
`fla_ops/chunk_vk.py:47` does not scale by `RCP_LN2`, so `exp` is correct
*there*) but they are mutually incompatible. A separate `kda/` namespace plus
an explicit header comment on the new `chunk_delta_h.py` naming its base-e
sibling is the guard against a future reader swapping one for the other.

Other divergences found in `chunk_delta_h_vk.py` that reinforce not reusing it:
missing `.to(tl.float32)` on gate loads, `new_empty` instead of `new_zeros` for
the final state, int32 `i_t` in the `h` store offsets (overflow risk on long
sequences), and no `chunk_indices` / `cu_seqlens_cpu` / `state_v_first`
parameters.

## Component design

### `chunk.py` — inference-only entry

Signature mirrors `fla.ops.kda.chunk_kda` plus the new arguments. Differences
from upstream:

- No `torch.autograd.Function`. Forward-only; ATOM never backprops here.
- No `@input_guard` — this is what removes copy #4. Contiguity becomes an
  explicit precondition, asserted where it matters rather than silently forced.
- No `@dispatch` — no backend is active on this platform anyway, and the
  indirection would make the parity test ambiguous about what it compared.
- `@torch.compiler.disable` is retained. The mixer is already behind the opaque
  `kda_attention_with_output` splitting op (`kimi_k3.py:790`), but the
  decorator costs nothing and matches upstream.

New arguments, all defaulting to off:

| Arg | Type | Meaning |
|-----|------|---------|
| `h0_indices` | `Tensor \| None` | 1D int32 slot index per sequence. `None` → dense `i_nh` indexing (upstream behavior). |
| `has_initial_state` | `Tensor \| None` | 1D bool per sequence. False → that sequence starts from zeros. |
| `inplace_final_state` | `bool` | Write the final state back to the same indexed slots of `initial_state`. |
| `o` | `Tensor \| None` | Caller-provided output buffer. |

**Every new argument is `tl.constexpr`-gated, so with all of them at their
defaults the emitted kernels are identical to upstream.** This is what makes the
parity test meaningful — the same vendored file serves as both the fused
implementation and its own control.

Validation performed at the entry, before anything is launched:

- `h0_indices.ndim == 1`, else raise. Per the scope decision, 2D
  `spec_state_indices` is **not** supported on this path; it must fail loudly
  rather than mis-index. (The spec-decode branch uses the decode kernel, which
  does handle 2D.)
- `inplace_final_state` requires `h0_indices is not None` and
  `output_final_state`.
- `initial_state.dtype == torch.float32` (upstream already asserts this).
- If `o is not None`: shape equals the output shape, dtype matches, and
  `o.is_contiguous()`. Without `@input_guard` nothing will silently clone it,
  but an assert turns a wrong-strides bug into an error instead of corruption.

### `chunk_delta_h.py` — indexed gather and scatter

Upstream derives both state pointers from the dense sequence index
(`fla/ops/common/chunk_delta_h.py:105-108`):

```python
if USE_INITIAL_STATE: h0 = h0 + i_nh * K * V
if STORE_FINAL_STATE: ht = ht + i_nh * K * V
```

This becomes an indirection, following `fused_sigmoid_gating.py:108-126`:

- `i_n` is the sequence index; `slot = tl.load(h0_indices + i_n).to(tl.int64)`.
- Base offset `slot * stride_state_slot + i_hv * V * K`, where
  `stride_state_slot` is passed from `initial_state.stride(0)` at runtime rather
  than assumed to be `HV*V*K`.
- `slot < 0` (PAD_SLOT_ID) → skip the load; the accumulator stays zero.
- `has_initial_state[i_n]` false → skip the load. **This is what absorbs
  `gather_kda_initial_state`'s zero-masking**: a fresh sequence simply never
  loads, instead of loading and then being zeroed by a second pass.
- Epilogue: when `INPLACE_FINAL_STATE`, `ht` aliases `initial_state` and the
  store targets the same indexed slot, repeating the `slot >= 0` guard.

`INPLACE_FINAL_STATE` is an explicit `tl.constexpr`, not a heuristic, matching
`fused_sigmoid_gating.py:305`.

**No transpose is needed.** `ssm_state` is allocated `[slots, HV, V, K]` in
fp32 (`atom/model_ops/attentions/gdn_attn.py:211-215` for the shape,
`:218-230` for the Kimi-specific fp32 state dtype), which is exactly the
`state_v_first=True` layout this kernel already produces. The change is pointer
arithmetic only.

Retained from upstream and **not** to be regressed to the `_vk` variant's form:
`exp2`, `.to(tl.float32)` on gate loads, `new_zeros` for the final state, int64
`i_t` in `h` store offsets, and the autotune key including `HV` and
`STATE_V_FIRST`.

### `chunk_o_gk.py` — out-parameter

`o = torch.zeros_like(v_new)` (fla `ops/gla/chunk.py:982`) becomes an optional
caller buffer, removing the allocation, the zero-fill, and `out.copy_`.

The upstream zero-fill carries the comment *"Please ensure zeros, since vllm
will use padding v"*. Two consequences:

- The kernel must write every element it is responsible for. Any element it
  skips now surfaces as uninitialized memory rather than zero.
- The parity test must include a padded-`v` case (see Testing) to prove the
  guarantee still holds.

Follows the existing in-tree out-param discipline from
`atom/model_ops/fla_ops/chunk_o.py:151,176-186` and `chunk.py:282-298`,
including the no-op dtype-cast skip — `o.to(dtype)` returns a *new* tensor when
the dtype differs, which would silently break the inplace contract.

### Call site — `atom/models/kimi_k3.py`

The prefill branch collapses to one call:

```python
kda_out, _ = chunk_kda(
    q, k, v, gate, beta, ...,
    initial_state=ssm_state,
    h0_indices=state_indices,
    has_initial_state=gdn_metadata.has_initial_state,
    inplace_final_state=True,
    o=out,
)
```

- `beta.float()` (`kimi_k3.py:1031`) is dropped. `fused_beta_sigmoid_fwd`
  allocates fp32 output regardless of input dtype (fla `ops/common/gate.py:59`),
  so the widening is redundant with respect to the sigmoid's output. The
  accuracy note in the existing comment concerns the *sigmoid result* dtype,
  which is unchanged — this must be stated in the new comment so the gsm8k
  regression it references is not silently reintroduced.
- `gather_kda_initial_state` and `ssm_state[state_indices] = last_state` are
  removed; `out.copy_` is removed.
- The decode branch (`:1158-1200`) and spec-decode branch (`:1201-1247`) are
  **untouched**.
- `atom/model_ops/kimi_k3/kda_state.py` loses its only caller. Per CLAUDE.md's
  fix-then-sweep rule, grep for other callers repo-wide (including
  `atom/plugin/`) before deleting, and remove its re-export from
  `atom/model_ops/kimi_k3/__init__.py`.

## Data flow

```
in_proj -> mixed_qkv -> causal_conv1d_fn -> q,k,v (rearranged views)
                                              |
                                              v
        chunk_kda(..., initial_state=ssm_state, h0_indices=state_indices,
                       has_initial_state=..., inplace_final_state=True, o=out)
              |
              +-- l2norm(q), l2norm(k), beta_sigmoid, gate_cumsum   [unchanged, fla]
              +-- intra x2, recompute_w_u                            [unchanged, fla]
              +-- chunk_delta_h   : gathers h0 from ssm_state[slot]  [vendored]
              |                     scatters ht to ssm_state[slot]
              +-- chunk_o_gk      : writes into `out`                [vendored]
              |
              v
        out -> o_norm(out, out_gate) -> o_proj
```

Nothing between the conv and `o_norm` allocates or copies a state- or
output-sized tensor.

## Error handling

| Condition | Behavior |
|-----------|----------|
| `h0_indices` is 2D | Raise at entry. Not supported on the prefill path. |
| `slot < 0` (PAD_SLOT_ID) | Skip both load and store. Output for that sequence is produced from a zero state. |
| `has_initial_state[i]` false | Skip the load; accumulator starts at zero. |
| `inplace_final_state` without `h0_indices` / `output_final_state` | Raise at entry. |
| `o` wrong shape / dtype / non-contiguous | Raise at entry. |
| `initial_state` not fp32 | Raise at entry (matches upstream). |

## Testing

### Parity test — `tests/kernels/test_chunk_kda_fused.py`

Skipped when `torch.cuda.is_available()` is false. This is the repo's first GPU
kernel test (`atom/model_ops/fla_ops/` currently has zero coverage), so it also
establishes the pattern.

Reference is stock `fla.ops.kda.chunk_kda` on identical inputs. **Both `o` and
the final state are asserted** — a gather/scatter bug can leave the output
correct while corrupting state, which would only show up on a later token.

Cases:

1. **Fused off** — vendored path with all new args at defaults vs stock fla.
   Guards the vendoring itself, independent of the fusion.
2. **Indexed gather/scatter** — non-monotonic, non-contiguous slot indices, so a
   dense-indexing bug cannot pass by coincidence.
3. **Mixed `has_initial_state`** — some sequences fresh, some resuming.
4. **PAD_SLOT_ID** — a `-1` slot present; assert the untouched slots of
   `ssm_state` are bit-identical before and after.
5. **Varlen** — multiple sequences of differing lengths via `cu_seqlens`,
   including a length that is not a multiple of `chunk_size`.
6. **Padded `v` / out-buffer coverage** — pre-fill `o` with a sentinel value and
   assert no sentinel survives, proving the removed zero-fill is not needed.
7. **Kimi-K3 config** — K=V=128, bf16 q/k/v/g, fp32 state, the shape actually
   served.

Tolerance: the fused and reference paths run the *same* arithmetic in the same
order — the changes are pointer arithmetic and buffer ownership, not math — so
the expectation is **bitwise equality** (`torch.equal`). If any case cannot meet
that, the discrepancy is to be explained before a tolerance is substituted;
loosening a tolerance to make a test pass is a design change, not a test fix.

### End-to-end

Per `recipes/Kimi-K3.md` and CLAUDE.md's serving rules:

1. `rm -rf /root/.cache/atom/*` (stale compile cache causes silent failures
   after kernel changes).
2. `AITER_LOG_LEVEL=WARNING` before starting the server.
3. Confirm the model is actually loaded via `rocm-smi --showmemuse` (VRAM% > 0),
   not `curl /health`.
4. Capture a trace and confirm all seven items are gone from the KDA prefill
   region.
5. `lm_eval` against the CI threshold in `/ci-pr-guide`.

On any server or GPU error, `/debug-guide` first rather than retrying.

## Risks

| Risk | Mitigation |
|------|------------|
| Silent numeric drift (the `exp`/`exp2` class of bug) | Parity test asserts on output *and* state; separate namespace; header comment naming the base-e sibling. |
| fla version drift breaking the internal imports | Pin the validated fla version (currently 0.5.2) in the module header. The parity test imports stock `chunk_kda` and fails loudly on divergence. |
| `flash_kda` later installed, diverting stock `chunk_kda` | Only affects the *reference* side of the parity test, which would then compare against a different implementation. Test asserts the reference is the triton path, or is skipped with a clear message. |
| Partially-written `o` reintroducing the padding bug | Sentinel-fill case in the parity test; entry-point assertions. |
| Divergence from upstream accumulating untracked | Header records the upstream file, fla version, and an explicit list of ATOM changes — the convention already used at `fla_ops/chunk_delta_h_vk.py:13-15`. |

## Out of scope

- Decode and spec-decode branches (already fused).
- 2D `spec_state_indices` on the prefill path (asserted against).
- Merging `l2norm` / `fused_beta_sigmoid` into the gate kernel (possible
  follow-up, to be decided from the post-change trace).
- The vLLM/SGLang plugin KDA paths.
- Backward/training support.

## Conventions

Vendored files carry the header block used throughout `atom/model_ops/fla_ops/`
(Apache-2.0 + vLLM + the original fla MIT notice), plus `# ruff: noqa: E501`
where upstream long lines are preserved, plus an "Adapted for ATOM" note listing
the divergences. Plain `import triton` / `import triton.language as tl`; no
`vllm.triton_utils`. `black . && ruff check .` before commit.
