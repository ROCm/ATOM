# Kimi-K3 KDA Prefill Kernel Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove seven redundant kernel launches and device-to-device copies from the Kimi-K3 KDA prefill path by vendoring flash-linear-attention's `chunk_kda` into ATOM and adding an indexed state gather, an inplace state scatter, and an output out-parameter to its two innermost kernels.

**Architecture:** A new package `atom/model_ops/fla_ops/kda/` holds four files copied from fla 0.5.2. Two of them carry kernel modifications: `chunk_delta_h.py` learns to load `h0` and store `ht` through a slot-index indirection instead of a dense sequence index, and `chunk_o_gk.py` learns to write into a caller-supplied buffer instead of allocating one. `chunk_fwd.py` threads the new arguments down, and `chunk.py` is a forward-only entry point without autograd, `@input_guard`, or `@dispatch`. Every new argument is `tl.constexpr`-gated and defaults to off, so the vendored path with default arguments is bit-identical to upstream — which is what makes the parity test meaningful. The Kimi-K3 prefill call site then collapses from six statements to one call.

**Tech Stack:** Python 3.12, PyTorch, Triton, flash-linear-attention 0.5.2, AMD ROCm (gfx950), pytest, black, ruff.

## Global Constraints

- **fla version is pinned at 0.5.2.** Vendored files import fla internals; record this version in every vendored file's header.
- **`exp2`, not `exp`.** KDA pre-scales its gate by `RCP_LN2` (`fla/ops/kda/chunk_fwd.py:47-56`), so all decay math is base-2. ATOM's existing `atom/model_ops/fla_ops/chunk_delta_h_vk.py` uses base-e `exp` and is for GDN only. Never substitute one for the other; the error is silent.
- **Never modify `@support_torch_compile`-decorated model files' traced regions.** `KimiKDAAttention._forward_impl` is reached through the opaque custom op `torch.ops.aiter.kda_attention_with_output` (`atom/models/kimi_k3.py:1056`), so editing `_forward_impl` is safe. Do not touch `KimiLinearModel` or anything Dynamo traces.
- **Bitwise equality is the parity bar.** The fused and reference paths run the same arithmetic in the same order. Assert with `torch.equal`. If a case cannot meet it, stop and explain the discrepancy — do not substitute a tolerance. Loosening a tolerance to make a test pass is a design change, not a test fix.
- **Vendored file header** is the block already used across `atom/model_ops/fla_ops/` (Apache-2.0 + vLLM + original fla MIT notice), plus `# ruff: noqa: E501` where upstream long lines are preserved, plus an "Adapted for ATOM" note listing divergences. See `atom/model_ops/fla_ops/chunk_delta_h_vk.py:1-18` for the exact form.
- **Plain `import triton` / `import triton.language as tl`.** No `vllm.triton_utils`.
- **`black . && ruff check .` must pass** before every commit (CI enforces it).
- **Scope is the prefill branch only** (`atom/models/kimi_k3.py:1118-1157`). The decode branch (`:1158-1200`) and spec-decode branch (`:1201-1247`) are already fused and must not be touched.
- **1D `h0_indices` only.** 2D `spec_state_indices` must raise, not mis-index.
- **State layout is already correct.** `ssm_state` is `[num_slots, HV, V, K]` fp32 (`atom/model_ops/attentions/gdn_attn.py:207-215` shape, `:218-229` Kimi fp32 dtype), which is exactly `state_v_first=True`. No transpose anywhere in this work.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `atom/model_ops/fla_ops/kda/__init__.py` | Exports `chunk_kda`. |
| `atom/model_ops/fla_ops/kda/chunk_delta_h.py` | h-kernel. Indexed `h0` gather + indexed inplace `ht` scatter. **Modified.** |
| `atom/model_ops/fla_ops/kda/chunk_o_gk.py` | GLA output kernel wrapper. `o=` out-parameter. **Modified.** |
| `atom/model_ops/fla_ops/kda/chunk_fwd.py` | Orchestrator. Threads the new args to the two above; everything else imported from fla. |
| `atom/model_ops/fla_ops/kda/chunk.py` | Public entry. Validation, no autograd/input_guard/dispatch. |
| `tests/kernels/__init__.py` | New test package marker (empty). |
| `tests/kernels/test_chunk_kda_fused.py` | GPU parity test against stock `fla.ops.kda.chunk_kda`. |
| `atom/models/kimi_k3.py` | Prefill call site rewired; `_run_kda` gains the new args. **Modified.** |
| `atom/model_ops/kimi_k3/kda_state.py` | **Deleted** in the final task, after its only caller is gone. |
| `atom/model_ops/kimi_k3/__init__.py` | Re-export of `gather_kda_initial_state` removed. **Modified.** |

Task order is bottom-up: the two modified kernels first (each independently testable against fla), then the orchestrator and entry, then the parity test, then the call site, then cleanup.

---

### Task 1: Vendor the h-kernel with indexed gather and scatter

This is the largest task. The kernel file is ~340 lines copied verbatim from fla with a bounded set of edits. Copy first, verify the copy is faithful, then edit.

**Files:**
- Create: `atom/model_ops/fla_ops/kda/__init__.py`
- Create: `atom/model_ops/fla_ops/kda/chunk_delta_h.py`
- Source to copy from: `/opt/venv/lib/python3.12/site-packages/fla/ops/common/chunk_delta_h.py` (lines 1-338 for the forward kernel, 674-731 for the wrapper)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  ```python
  def chunk_gated_delta_rule_fwd_h(
      k: torch.Tensor,
      w: torch.Tensor,
      u: torch.Tensor,
      g: torch.Tensor | None = None,
      gk: torch.Tensor | None = None,
      initial_state: torch.Tensor | None = None,
      output_final_state: bool = False,
      chunk_size: int = 64,
      save_new_value: bool = True,
      state_v_first: bool = False,
      cu_seqlens: torch.LongTensor | None = None,
      cu_seqlens_cpu: torch.LongTensor | None = None,
      chunk_indices: torch.LongTensor | None = None,
      h0_indices: torch.Tensor | None = None,
      has_initial_state: torch.Tensor | None = None,
      inplace_final_state: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
      """Returns (h, v_new, final_state)."""
  ```
  When `inplace_final_state=True`, the returned `final_state` **is** the `initial_state` object (same storage), matching `fused_sigmoid_gating.py:254-256`.

- [ ] **Step 1: Create the package marker**

Create `atom/model_ops/fla_ops/kda/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2). The original source code was licensed under the MIT
# license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
"""Vendored KDA chunk forward, fused for ATOM's Kimi-K3 prefill path.

Upstream `fla.ops.kda.chunk_kda` allocates its own output and indexes the
recurrent state densely, which forces the caller to gather the initial state,
scatter the final state, and copy the output. This package threads a slot-index
indirection and an output buffer through the two innermost kernels so all three
happen inside them.

WARNING: these kernels are base-2 (`exp2`). KDA pre-scales its gate by RCP_LN2.
Do not interchange them with the base-e siblings in the parent package
(`fla_ops/chunk_delta_h_vk.py`, `fla_ops/chunk_o_vk.py`), which serve GDN and
do not pre-scale. The mismatch produces `decay ** 1.4427` and raises nothing.
"""

from .chunk import chunk_kda

__all__ = ["chunk_kda"]
```

Note: this imports `.chunk`, which does not exist until Task 3. That is expected; nothing imports this package until Task 4.

- [ ] **Step 2: Copy the h-kernel verbatim**

```bash
mkdir -p atom/model_ops/fla_ops/kda
python - <<'PY'
import pathlib
src = pathlib.Path("/opt/venv/lib/python3.12/site-packages/fla/ops/common/chunk_delta_h.py")
lines = src.read_text().splitlines(keepends=True)
# Forward kernel: lines 1-338 (1-indexed) -> [0:338]
# Forward wrapper: lines 674-731 -> [673:731]
out = "".join(lines[0:338]) + "\n\n" + "".join(lines[673:731])
pathlib.Path("atom/model_ops/fla_ops/kda/chunk_delta_h.py").write_text(out)
PY
grep -c "" atom/model_ops/fla_ops/kda/chunk_delta_h.py
```

Expected: 398 lines, starting with fla's copyright line and ending with
`return h, v_new, final_state`. The file currently will not import — it still has fla's header and imports. Fixed in the next step.

- [ ] **Step 3: Replace the header and imports**

Replace lines 1-28 of `atom/model_ops/fla_ops/kda/chunk_delta_h.py` (fla's copyright block through the `GATED_DELTA_RULE_FWD_H_NUM_WARPS` definition) with:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2, fla/ops/common/chunk_delta_h.py). The original source code
# was licensed under the MIT license and included the following copyright
# notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501
#
# Adapted for ATOM:
#   - Forward path only; the backward kernel and its wrapper are not copied.
#   - `h0` / `ht` are addressed through `h0_indices` (a per-sequence cache-slot
#     index) rather than the dense `i_nh`, so the caller no longer gathers the
#     initial state or scatters the final state. Follows the same shape as
#     `fla_ops/fused_sigmoid_gating.py:108-126`.
#   - `has_initial_state` skips the h0 load for fresh sequences, absorbing what
#     `model_ops/kimi_k3/kda_state.py` used to do in a separate pass.
#   - `inplace_final_state` lets `ht` alias `initial_state`.
#   - `@dispatch('common')` dropped from the wrapper: no backend is installed on
#     ROCm, and the indirection would make the parity test ambiguous.
#
# WARNING: base-2 (`exp2`). See this package's __init__ docstring before
# swapping anything here with `fla_ops/chunk_delta_h_vk.py`, which is base-e.

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import (
    IS_NVIDIA_BLACKWELL,
    IS_NVIDIA_HOPPER,
    autotune_cache_kwargs,
    check_shared_mem,
)

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]

# TODO: Triton mainline fixes a Blackwell tl.dot recurrence race.
# Keep this kernel on num_warps=2 for Blackwell until Triton 3.8 is released
# and we re-validate the wider config space.
GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]
```

Then verify the module imports and nothing else references `dispatch`:

```bash
grep -n "dispatch" atom/model_ops/fla_ops/kda/chunk_delta_h.py
python -c "import atom.model_ops.fla_ops.kda.chunk_delta_h as m; print(m.chunk_gated_delta_rule_fwd_h)"
```

Expected: `grep` prints the `@dispatch('common')` line above the wrapper (removed in Step 5); after Step 5 it prints nothing. The `python -c` will fail until Step 5 removes that decorator — that is fine, run it again there.

- [ ] **Step 4: Add the new kernel parameters and the indexed gather**

In `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, add to the `@triton.heuristics` dict (which currently ends with `'IS_VARLEN'`):

```python
    'USE_H0_INDICES': lambda args: args['h0_indices'] is not None,
    'USE_HAS_INITIAL_STATE': lambda args: args['has_initial_state'] is not None,
```

Add to the autotune `key` list so specializations do not collide — change

```python
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'],
```

to

```python
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST', 'USE_H0_INDICES', 'INPLACE_FINAL_STATE'],
```

Add these parameters to the kernel signature. Place the two pointers after `ht`, and the three `constexpr` flags after `STATE_V_FIRST`:

```python
    ht,
    h0_indices,
    has_initial_state,
    cu_seqlens,
    chunk_offsets,
    T,
    ...
    STATE_V_FIRST: tl.constexpr,
    USE_H0_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    stride_state_slot,
    IS_VARLEN: tl.constexpr,
```

`stride_state_slot` is a runtime (non-constexpr) argument — it comes from `initial_state.stride(0)`, matching `fused_sigmoid_gating.py:259`.

Now replace the dense offset block (originally lines 114-117):

```python
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V
```

with:

```python
    # Resolve the state slot. Upstream indexes densely by i_nh, which requires
    # the caller to gather ssm_state[state_indices] into a packed buffer first.
    # With h0_indices we read/write the cache slot directly.
    load_h0 = USE_INITIAL_STATE
    store_ht = STORE_FINAL_STATE
    if USE_H0_INDICES:
        i_slot = tl.load(h0_indices + i_n).to(tl.int64)
        # PAD_SLOT_ID (-1) marks an idle slot: no state to read, none to write.
        if i_slot < 0:
            load_h0 = False
            store_ht = False
        else:
            if USE_HAS_INITIAL_STATE:
                # A fresh sequence starts from zeros. Skipping the load is what
                # replaces the separate zero-fill pass.
                if tl.load(has_initial_state + i_n) == 0:
                    load_h0 = False
            i_state_base = i_slot * stride_state_slot + (i_h * K * V).to(tl.int64)
            if USE_INITIAL_STATE:
                h0 = h0 + i_state_base
            if STORE_FINAL_STATE:
                ht = ht + i_state_base
    else:
        if USE_INITIAL_STATE:
            h0 = h0 + i_nh * K*V
        if STORE_FINAL_STATE:
            ht = ht + i_nh * K*V
```

Then change the guard on the h0 load block (originally line 130) from

```python
    if USE_INITIAL_STATE:
```

to

```python
    if load_h0:
```

and the guard on the final-state store block (originally line 307) from

```python
    if STORE_FINAL_STATE:
```

to

```python
    if store_ht:
```

Leave the four nested `if K > 64:` / `if K > 128:` / `if K > 192:` sub-blocks inside both regions exactly as they are.

Note on `INPLACE_FINAL_STATE`: the kernel does not branch on it — when inplace, the wrapper simply passes the same tensor as both `h0` and `ht`, and both resolve to the same `i_state_base`. It is declared `constexpr` and included in the autotune key so the two modes get separate specializations, matching the explicit-flag discipline at `fused_sigmoid_gating.py:305`. The h0 load happens before the recurrence and the ht store after, so aliasing is safe within a program.

- [ ] **Step 5: Update the wrapper**

Remove the `@dispatch('common')` decorator above `chunk_gated_delta_rule_fwd_h` and the now-unused import. Add the three new parameters to the signature (after `chunk_indices`):

```python
    h0_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = False,
```

Replace the final-state allocation block (originally lines 702-707):

```python
    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None
```

with:

```python
    if inplace_final_state:
        if h0_indices is None:
            raise ValueError("inplace_final_state requires h0_indices.")
        if initial_state is None:
            raise ValueError("inplace_final_state requires initial_state.")
        if not output_final_state:
            raise ValueError("inplace_final_state requires output_final_state.")
    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
    else:
        h = k.new_empty(B, NT, HV, K, V)
    if not output_final_state:
        final_state = None
    elif inplace_final_state:
        # ht aliases the cache; the kernel writes the indexed slots in place.
        final_state = initial_state
    elif state_v_first:
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
```

Add before the launch:

```python
    if h0_indices is not None:
        if h0_indices.ndim != 1:
            raise ValueError(
                f"h0_indices must be 1D (one cache slot per sequence), got shape "
                f"{tuple(h0_indices.shape)}. 2D spec-decode indices are not "
                f"supported on the chunked prefill path."
            )
        if h0_indices.shape[0] != N:
            raise ValueError(
                f"h0_indices has {h0_indices.shape[0]} entries but there are {N} "
                f"sequences."
            )
    if has_initial_state is not None and h0_indices is None:
        raise ValueError("has_initial_state requires h0_indices.")
    stride_state_slot = initial_state.stride(0) if initial_state is not None else 0
```

And pass the new arguments in the kernel launch, after `ht=final_state`:

```python
        h0_indices=h0_indices,
        has_initial_state=has_initial_state,
        stride_state_slot=stride_state_slot,
        INPLACE_FINAL_STATE=inplace_final_state,
```

- [ ] **Step 6: Verify it imports and the default path is unchanged**

```bash
python -c "
import inspect
import atom.model_ops.fla_ops.kda.chunk_delta_h as m
sig = inspect.signature(m.chunk_gated_delta_rule_fwd_h)
for name in ('h0_indices', 'has_initial_state', 'inplace_final_state'):
    assert name in sig.parameters, name
    assert sig.parameters[name].default in (None, False), name
print('signature ok')
"
black atom/model_ops/fla_ops/kda/ && ruff check atom/model_ops/fla_ops/kda/
```

Expected: `signature ok`, then black reformats and ruff reports no errors. Behavioral verification is Task 4's parity test — this task's kernel is not reachable yet.

- [ ] **Step 7: Commit**

```bash
git add atom/model_ops/fla_ops/kda/__init__.py atom/model_ops/fla_ops/kda/chunk_delta_h.py
git commit -m "feat(kda): vendor h-kernel with indexed h0 gather and ht scatter

Copied from flash-linear-attention 0.5.2 fla/ops/common/chunk_delta_h.py,
forward path only. h0/ht are now addressed through a per-sequence cache-slot
index instead of the dense sequence index, with PAD_SLOT_ID and
has_initial_state handled in-kernel. All new arguments default to off, so the
emitted kernel is unchanged when they are unused.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Vendor the GLA output wrapper with an out-parameter

Only the Python wrapper changes; the Triton kernel is copied unmodified.

**Files:**
- Create: `atom/model_ops/fla_ops/kda/chunk_o_gk.py`
- Source to copy from: `/opt/venv/lib/python3.12/site-packages/fla/ops/gla/chunk.py` (kernel at lines 330-432, wrapper at lines 962-1002)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  ```python
  def chunk_gla_fwd_o_gk(
      q: torch.Tensor,
      v: torch.Tensor,
      g: torch.Tensor,
      A: torch.Tensor,
      h: torch.Tensor,
      scale: float,
      state_v_first: bool = False,
      cu_seqlens: torch.LongTensor | None = None,
      chunk_size: int = 64,
      chunk_indices: torch.LongTensor | None = None,
      o: torch.Tensor | None = None,
  ) -> torch.Tensor:
      """Returns the output tensor; `o` itself when provided."""
  ```

- [ ] **Step 1: Copy the kernel and wrapper**

```bash
python - <<'PY'
import pathlib
src = pathlib.Path("/opt/venv/lib/python3.12/site-packages/fla/ops/gla/chunk.py")
lines = src.read_text().splitlines(keepends=True)
# kernel: lines 330-432 -> [329:432]; wrapper: lines 962-1002 -> [961:1002]
out = "".join(lines[329:432]) + "\n\n" + "".join(lines[961:1002])
pathlib.Path("atom/model_ops/fla_ops/kda/chunk_o_gk.py").write_text(out)
PY
head -5 atom/model_ops/fla_ops/kda/chunk_o_gk.py
```

- [ ] **Step 2: Prepend the header and imports**

Insert at the very top of `atom/model_ops/fla_ops/kda/chunk_o_gk.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2, fla/ops/gla/chunk.py). The original source code was licensed
# under the MIT license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501
#
# Adapted for ATOM:
#   - Only chunk_gla_fwd_kernel_o and its wrapper are copied; the kernel itself
#     is unmodified.
#   - The wrapper accepts a caller-provided `o`, replacing
#     `o = torch.zeros_like(v)` and the caller's subsequent `out.copy_`.
#
# WARNING: base-2 (`exp2`). See this package's __init__ docstring.

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs
```

- [ ] **Step 3: Add the out-parameter to the wrapper**

Add `o: torch.Tensor | None = None,` as the last parameter of `chunk_gla_fwd_o_gk`, then replace:

```python
    # Please ensure zeros, since vllm will use padding v
    o = torch.zeros_like(v)
```

with:

```python
    if o is None:
        # Upstream zero-fills here ("Please ensure zeros, since vllm will use
        # padding v"). The kernel's store is masked to m_t & m_v, so padding
        # rows outside the sequence are never written and must already be zero.
        o = torch.zeros_like(v)
    else:
        # Same discipline as fla_ops/chunk_o.py:151,176-186: the kernel assumes
        # stride (HV*V, 1) on the (T, V) plane, so a non-contiguous or
        # wrong-dtype buffer would silently corrupt rather than fail.
        assert o.shape == v.shape, (
            f"chunk_gla_fwd_o_gk: caller-provided o.shape {tuple(o.shape)} != "
            f"v.shape {tuple(v.shape)}"
        )
        assert o.dtype == v.dtype, (
            f"chunk_gla_fwd_o_gk: caller-provided o.dtype {o.dtype} != v.dtype "
            f"{v.dtype}"
        )
        assert (
            o.is_contiguous()
        ), "chunk_gla_fwd_o_gk: caller-provided o must be contiguous"
```

The caller must zero any padding region it owns before the call. Task 5 records why the Kimi call site does not need to.

- [ ] **Step 4: Verify it imports**

```bash
python -c "
import inspect
import atom.model_ops.fla_ops.kda.chunk_o_gk as m
sig = inspect.signature(m.chunk_gla_fwd_o_gk)
assert sig.parameters['o'].default is None
print('signature ok')
"
black atom/model_ops/fla_ops/kda/ && ruff check atom/model_ops/fla_ops/kda/
```

Expected: `signature ok`, clean lint.

- [ ] **Step 5: Commit**

```bash
git add atom/model_ops/fla_ops/kda/chunk_o_gk.py
git commit -m "feat(kda): vendor GLA output wrapper with an o= out-parameter

Copied from flash-linear-attention 0.5.2 fla/ops/gla/chunk.py. The Triton
kernel is unmodified; the wrapper now accepts a caller buffer instead of
allocating and zero-filling one, which removes the caller's out.copy_.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Orchestrator and public entry point

**Files:**
- Create: `atom/model_ops/fla_ops/kda/chunk_fwd.py`
- Create: `atom/model_ops/fla_ops/kda/chunk.py`
- Source to mirror: `fla/ops/kda/chunk_fwd.py` and `fla/ops/kda/chunk.py`

**Interfaces:**
- Consumes: `chunk_gated_delta_rule_fwd_h` (Task 1) and `chunk_gla_fwd_o_gk` (Task 2), with the exact signatures given in those tasks.
- Produces:
  ```python
  def chunk_kda(
      q, k, v, g, beta,
      scale: float | None = None,
      initial_state: torch.Tensor | None = None,
      output_final_state: bool = False,
      use_qk_l2norm_in_kernel: bool = False,
      use_gate_in_kernel: bool = False,
      use_beta_sigmoid_in_kernel: bool = False,
      allow_neg_eigval: bool = False,
      safe_gate: bool = False,
      lower_bound: float | None = None,
      disable_recompute: bool = False,
      state_v_first: bool = False,
      cu_seqlens: torch.LongTensor | None = None,
      cu_seqlens_cpu: torch.LongTensor | None = None,
      A_log: torch.Tensor | None = None,
      dt_bias: torch.Tensor | None = None,
      chunk_size: int = 64,
      h0_indices: torch.Tensor | None = None,
      has_initial_state: torch.Tensor | None = None,
      inplace_final_state: bool = False,
      o: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
      """Returns (o, final_state)."""
  ```
  Note this entry takes `A_log` / `dt_bias` / `chunk_size` as **named parameters**, not `**kwargs` as upstream does, and does not accept the deprecated `transpose_state_layout`.

- [ ] **Step 1: Write `chunk_fwd.py`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2, fla/ops/kda/chunk_fwd.py). The original source code was
# licensed under the MIT license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted for ATOM:
#   - Forward only; context-parallel (cp_context) and the training-only
#     intermediate-state returns are dropped.
#   - Threads h0_indices / has_initial_state / inplace_final_state to the
#     h-kernel and `o` to the output kernel.
#   - Unmodified stages (gate cumsum, intra, recompute_w_u) are imported from
#     fla rather than copied.

import torch
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o_gk import chunk_gla_fwd_o_gk


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    h0_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = False,
    o: torch.Tensor | None = None,
):
    # RCP_LN2 puts the gate in the log2 domain; every downstream decay uses
    # exp2. Do not remove this scaling without changing the kernels too.
    if use_gate_in_kernel:
        g = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    w, u, _qg, kg, Aqk, _Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        h0_indices=h0_indices,
        has_initial_state=has_initial_state,
        inplace_final_state=inplace_final_state,
    )

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        o=o,
    )
    return o, final_state
```

- [ ] **Step 2: Write `chunk.py`**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2, fla/ops/kda/chunk.py). The original source code was licensed
# under the MIT license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted for ATOM:
#   - No torch.autograd.Function: inference only.
#   - No @input_guard: it calls .contiguous() on every tensor argument, which
#     silently clones the rearrange views this path passes and would clone a
#     caller-provided `o` out from under them. Contiguity is asserted instead.
#   - No @dispatch('kda'): flash_kda is not installed and tilelang needs nvcc,
#     so no backend is reachable on ROCm; the indirection would only make the
#     parity test ambiguous about what it compared.
#   - A_log / dt_bias / chunk_size are explicit parameters rather than **kwargs,
#     and the deprecated `transpose_state_layout` alias is not accepted.

import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.utils.index import prepare_chunk_indices

from .chunk_fwd import chunk_kda_fwd


@torch.compiler.disable
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    h0_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = False,
    o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked KDA forward, fused for ATOM's Kimi-K3 prefill path.

    Same semantics as ``fla.ops.kda.chunk_kda`` with these additions:

    ``h0_indices``
        1D per-sequence cache-slot index into ``initial_state``'s first
        dimension. When given, the kernel reads the initial state from
        ``initial_state[h0_indices[i]]`` instead of ``initial_state[i]``, so the
        caller does not gather. ``-1`` (PAD_SLOT_ID) skips both read and write.
    ``has_initial_state``
        1D per-sequence bool. False means the sequence starts from a zero state;
        the kernel skips the load rather than loading and zeroing afterwards.
    ``inplace_final_state``
        Write the final state back into the same indexed slots of
        ``initial_state``. The returned ``final_state`` *is* ``initial_state``.
    ``o``
        Caller-provided output buffer, written in place and returned.

    With all four at their defaults this is bit-identical to upstream.
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when "
            f"using `cu_seqlens`. Please flatten variable-length inputs first."
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError(
            f"initial_state must be float32, got {initial_state.dtype}."
        )
    if chunk_size not in (32, 64):
        raise ValueError(
            f"`chunk_size` must be either 32 or 64 for KDA, got {chunk_size}."
        )
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log must be provided when use_gate_in_kernel=True.")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError(
                "`lower_bound` must be specified when `safe_gate=True` and "
                "`use_gate_in_kernel=True`."
            )
        if not -5 <= lower_bound < 0:
            raise ValueError(
                f"`lower_bound` must be in the safe range [-5, 0), got "
                f"{lower_bound}."
            )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`."
        )

    B, T, H, K, HV = *q.shape, v.shape[2]
    if q.shape != k.shape:
        raise ValueError(f"q and k must match, got {q.shape} vs {k.shape}")
    if K > 256:
        raise ValueError(f"KDA supports key headdim <= 256, got {K}.")
    if HV % H != 0:
        raise ValueError(f"num_v_heads ({HV}) must be divisible by ({H}).")
    if tuple(g.shape) != (B, T, HV, K):
        raise ValueError(f"g must be {(B, T, HV, K)}, got {tuple(g.shape)}")
    if tuple(beta.shape) != (B, T, HV):
        raise ValueError(f"beta must be {(B, T, HV)}, got {tuple(beta.shape)}")

    if h0_indices is not None and h0_indices.ndim != 1:
        raise ValueError(
            f"h0_indices must be 1D, got shape {tuple(h0_indices.shape)}. 2D "
            f"spec-decode indices are not supported on the prefill path; the "
            f"decode kernel handles those."
        )
    if inplace_final_state and h0_indices is None:
        raise ValueError("inplace_final_state requires h0_indices.")
    if inplace_final_state and not output_final_state:
        raise ValueError("inplace_final_state requires output_final_state.")
    if has_initial_state is not None and h0_indices is None:
        raise ValueError("has_initial_state requires h0_indices.")
    if o is not None:
        # Without @input_guard nothing clones a bad buffer silently, so these
        # turn a wrong-strides bug into an error instead of corruption.
        if tuple(o.shape) != (B, T, HV, v.shape[-1]):
            raise ValueError(
                f"o must be {(B, T, HV, v.shape[-1])}, got {tuple(o.shape)}"
            )
        if o.dtype != v.dtype:
            raise ValueError(f"o.dtype {o.dtype} != v.dtype {v.dtype}")
        if not o.is_contiguous():
            raise ValueError("o must be contiguous")

    if scale is None:
        scale = K**-0.5

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    if use_beta_sigmoid_in_kernel:
        beta = fused_beta_sigmoid(beta, scale=2.0 if allow_neg_eigval else 1.0)

    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu
        )

    out, final_state = chunk_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=disable_recompute,
        h0_indices=h0_indices,
        has_initial_state=has_initial_state,
        inplace_final_state=inplace_final_state,
        o=o,
    )
    return out, final_state
```

Note: upstream returns `o.type_as(q)`. This entry does not, because with `o=` provided the buffer's dtype is already asserted equal to `v.dtype`, and a `.type_as` would allocate a new tensor and break the inplace contract. Task 4's "fused off" case verifies the dtypes agree without it.

- [ ] **Step 3: Verify the package imports end-to-end**

```bash
python -c "
from atom.model_ops.fla_ops.kda import chunk_kda
import inspect
sig = inspect.signature(chunk_kda)
for n in ('h0_indices', 'has_initial_state', 'inplace_final_state', 'o'):
    assert n in sig.parameters, n
print('chunk_kda importable, new args present')
"
black atom/model_ops/fla_ops/kda/ && ruff check atom/model_ops/fla_ops/kda/
```

Expected: `chunk_kda importable, new args present`, clean lint.

- [ ] **Step 4: Commit**

```bash
git add atom/model_ops/fla_ops/kda/chunk_fwd.py atom/model_ops/fla_ops/kda/chunk.py
git commit -m "feat(kda): forward-only chunk_kda entry threading the fused args

Orchestrator and public entry for the vendored KDA forward. No autograd, no
input_guard (it would clone the caller's o and the rearrange views), no
dispatch (no backend is reachable on ROCm). Unmodified stages still come from
fla.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: GPU parity test

This is the repo's first GPU kernel test, so it establishes the pattern: `tests/kernels/`, skipped without CUDA.

**Files:**
- Create: `tests/kernels/__init__.py` (empty)
- Create: `tests/kernels/test_chunk_kda_fused.py`

**Interfaces:**
- Consumes: `atom.model_ops.fla_ops.kda.chunk_kda` (Task 3).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/kernels/__init__.py` as an empty file. Then create `tests/kernels/test_chunk_kda_fused.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parity tests for ATOM's vendored, fused KDA chunk forward.

Reference is stock ``fla.ops.kda.chunk_kda``. Both the output and the final
state are asserted: a gather/scatter bug can leave the output correct while
corrupting the state, which would only surface on a later token.

The expectation is bitwise equality. The fused and reference paths run the same
arithmetic in the same order -- the changes are pointer arithmetic and buffer
ownership, not math. A discrepancy is a bug to explain, not a tolerance to
loosen.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KDA kernels require a GPU"
)

PAD_SLOT_ID = -1


def _make_inputs(seq_lens, hv=4, k_dim=128, v_dim=128, dtype=torch.bfloat16, seed=0):
    """Build a flattened varlen KDA input set on the GPU.

    Returns a dict of the arguments both paths take, plus ``cu_seqlens``.
    """
    torch.manual_seed(seed)
    dev = "cuda"
    total = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.long, device=dev
    )
    return {
        "q": torch.randn(1, total, hv, k_dim, dtype=dtype, device=dev),
        "k": torch.randn(1, total, hv, k_dim, dtype=dtype, device=dev),
        "v": torch.randn(1, total, hv, v_dim, dtype=dtype, device=dev),
        "g": torch.randn(1, total, hv, k_dim, dtype=dtype, device=dev),
        "beta": torch.randn(1, total, hv, dtype=torch.float32, device=dev),
        "A_log": torch.randn(hv, dtype=torch.float32, device=dev),
        "dt_bias": torch.randn(hv * k_dim, dtype=torch.float32, device=dev),
        "cu_seqlens": cu_seqlens,
    }


def _flags():
    """The exact flag set Kimi-K3's prefill path uses."""
    return {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": True,
        "lower_bound": -5.0,
        "state_v_first": True,
        "disable_recompute": True,
        "output_final_state": True,
    }


def _reference(inp, initial_state):
    """Stock fla, dense initial state, allocating its own output."""
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    return fla_chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=initial_state,
        cu_seqlens=inp["cu_seqlens"],
        **_flags(),
    )


def _fused(inp, initial_state, **extra):
    from atom.model_ops.fla_ops.kda import chunk_kda

    return chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=initial_state,
        cu_seqlens=inp["cu_seqlens"],
        **_flags(),
        **extra,
    )


def _dense_state(n, hv=4, k_dim=128, v_dim=128, seed=1):
    """[N, HV, V, K] fp32 -- the state_v_first layout ssm_state uses."""
    torch.manual_seed(seed)
    return torch.randn(n, hv, v_dim, k_dim, dtype=torch.float32, device="cuda")


def test_vendored_path_matches_fla_with_fusion_off():
    """Case 1: the vendoring itself, independent of any fusion."""
    inp = _make_inputs([64, 128])
    h0 = _dense_state(2)
    ref_o, ref_ht = _reference(inp, h0.clone())
    got_o, got_ht = _fused(inp, h0.clone())
    assert torch.equal(got_o, ref_o)
    assert torch.equal(got_ht, ref_ht)


def test_indexed_gather_and_scatter():
    """Case 2: non-monotonic, non-contiguous slots -- a dense-indexing bug
    cannot pass by coincidence."""
    inp = _make_inputs([64, 96, 128])
    slots = [5, 1, 3]
    cache = _dense_state(8)
    packed = torch.stack([cache[s] for s in slots]).contiguous()

    ref_o, ref_ht = _reference(inp, packed.clone())

    fused_cache = cache.clone()
    idx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    got_o, got_ht = _fused(
        inp, fused_cache, h0_indices=idx, inplace_final_state=True
    )
    assert got_ht.data_ptr() == fused_cache.data_ptr(), "inplace must alias"
    assert torch.equal(got_o, ref_o)
    for i, s in enumerate(slots):
        assert torch.equal(fused_cache[s], ref_ht[i]), f"slot {s}"
    # Slots nobody claimed must be untouched.
    for s in set(range(8)) - set(slots):
        assert torch.equal(fused_cache[s], cache[s]), f"slot {s} was clobbered"


def test_mixed_has_initial_state():
    """Case 3: some sequences fresh, some resuming."""
    inp = _make_inputs([64, 128, 64])
    slots = [2, 0, 6]
    cache = _dense_state(8)
    has_init = torch.tensor([True, False, True], device="cuda")

    # Reference: gather, then zero the fresh ones -- what the old path did.
    packed = torch.stack([cache[s] for s in slots]).contiguous()
    packed[~has_init] = 0
    ref_o, ref_ht = _reference(inp, packed)

    fused_cache = cache.clone()
    got_o, _ = _fused(
        inp,
        fused_cache,
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        has_initial_state=has_init,
        inplace_final_state=True,
    )
    assert torch.equal(got_o, ref_o)
    for i, s in enumerate(slots):
        assert torch.equal(fused_cache[s], ref_ht[i]), f"slot {s}"


def test_pad_slot_id_leaves_cache_untouched():
    """Case 4: a -1 slot reads nothing and writes nothing."""
    inp = _make_inputs([64, 64])
    cache = _dense_state(8)
    before = cache.clone()
    slots = [PAD_SLOT_ID, 4]

    got_o, _ = _fused(
        inp,
        cache,
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        inplace_final_state=True,
    )
    assert torch.isfinite(got_o).all()
    for s in set(range(8)) - {4}:
        assert torch.equal(cache[s], before[s]), f"slot {s} was clobbered"


def test_varlen_with_ragged_tail():
    """Case 5: a sequence length that is not a multiple of chunk_size (64)."""
    inp = _make_inputs([64, 100, 37])
    slots = [0, 1, 2]
    cache = _dense_state(4)
    packed = torch.stack([cache[s] for s in slots]).contiguous()

    ref_o, ref_ht = _reference(inp, packed.clone())
    fused_cache = cache.clone()
    got_o, _ = _fused(
        inp,
        fused_cache,
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        inplace_final_state=True,
    )
    assert torch.equal(got_o, ref_o)
    for i, s in enumerate(slots):
        assert torch.equal(fused_cache[s], ref_ht[i]), f"slot {s}"


def test_out_buffer_is_fully_written():
    """Case 6: prove the removed zero-fill is not load-bearing. Every element
    the kernel owns must be overwritten, so no sentinel survives."""
    inp = _make_inputs([64, 128])
    cache = _dense_state(4)
    slots = [0, 1]

    sentinel = torch.full_like(inp["v"], float("nan"))
    got_o, _ = _fused(
        inp,
        cache.clone(),
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        inplace_final_state=True,
        o=sentinel,
    )
    assert got_o.data_ptr() == sentinel.data_ptr(), "o= must be written in place"
    assert not torch.isnan(got_o).any(), "kernel left elements unwritten"

    ref_o, _ = _reference(inp, torch.stack([cache[s] for s in slots]).contiguous())
    assert torch.equal(got_o, ref_o)


def test_kimi_k3_shape():
    """Case 7: the shape actually served -- K=V=128, bf16 in, fp32 state."""
    inp = _make_inputs([512, 1024], hv=4, k_dim=128, v_dim=128)
    cache = _dense_state(16)
    slots = [11, 2]
    packed = torch.stack([cache[s] for s in slots]).contiguous()

    ref_o, ref_ht = _reference(inp, packed.clone())
    fused_cache = cache.clone()
    out = torch.empty_like(inp["v"])
    got_o, _ = _fused(
        inp,
        fused_cache,
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        inplace_final_state=True,
        o=out,
    )
    assert torch.equal(got_o, ref_o)
    for i, s in enumerate(slots):
        assert torch.equal(fused_cache[s], ref_ht[i]), f"slot {s}"


def test_2d_indices_rejected():
    """Spec-decode indices must fail loudly rather than mis-index."""
    from atom.model_ops.fla_ops.kda import chunk_kda

    inp = _make_inputs([64, 64])
    cache = _dense_state(4)
    with pytest.raises(ValueError, match="1D"):
        chunk_kda(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            g=inp["g"],
            beta=inp["beta"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            initial_state=cache,
            cu_seqlens=inp["cu_seqlens"],
            h0_indices=torch.zeros(2, 2, dtype=torch.int32, device="cuda"),
            **_flags(),
        )


def test_non_contiguous_out_rejected():
    from atom.model_ops.fla_ops.kda import chunk_kda

    inp = _make_inputs([64, 64])
    cache = _dense_state(4)
    bad = torch.empty(
        1, 128, 4, 256, dtype=inp["v"].dtype, device="cuda"
    )[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        chunk_kda(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            g=inp["g"],
            beta=inp["beta"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            initial_state=cache,
            cu_seqlens=inp["cu_seqlens"],
            h0_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            inplace_final_state=True,
            o=bad,
            **_flags(),
        )
```

- [ ] **Step 2: Run the test**

```bash
AITER_LOG_LEVEL=WARNING python -m pytest tests/kernels/test_chunk_kda_fused.py -v
```

Expected on the first run: some cases fail. That is the point of running now — this is the first execution of the Task 1-3 kernels. Read each failure before changing anything.

Triage guide, in the order failures are most likely:

- **All cases fail identically, including case 1 (`fusion off`)** → the vendoring diverged from upstream. Diff the copied regions against the fla source; do not start editing kernel math.
- **Case 1 passes, case 2 fails on state only** → the `ht` scatter offset is wrong. Check `stride_state_slot` is `initial_state.stride(0)` and that `i_state_base` uses `i_h`, not `i_nh`.
- **Case 2 fails on output too** → the `h0` gather is wrong; same offset check.
- **Case 3 fails only where `has_initial_state` is False** → the skip is inverted or the tensor is being read as the wrong dtype (it is `torch.bool`).
- **Case 4 clobbers slots** → the `i_slot < 0` guard is not suppressing the store.
- **Case 6 leaves NaNs** → the output kernel does not cover the whole buffer; the removed zero-fill *was* load-bearing. Stop and report this — it invalidates a design assumption rather than being a bug to patch.
- **Any case fails by a small numeric margin** → do not add a tolerance. The paths are supposed to be identical; a margin means something differs structurally. Report it.

- [ ] **Step 3: Fix until green, then re-run**

```bash
AITER_LOG_LEVEL=WARNING python -m pytest tests/kernels/test_chunk_kda_fused.py -v
```

Expected: 9 passed.

- [ ] **Step 4: Confirm the CPU suite is unaffected**

```bash
python -m pytest tests/ -q -x --ignore=tests/kernels
```

Expected: the existing suite passes exactly as before this branch.

- [ ] **Step 5: Commit**

```bash
black . && ruff check .
git add tests/kernels/__init__.py tests/kernels/test_chunk_kda_fused.py
git add -u atom/model_ops/fla_ops/kda/
git commit -m "test(kda): GPU parity tests for the fused chunk_kda

Nine cases against stock fla: vendoring with fusion off, indexed
gather/scatter with non-contiguous slots, mixed has_initial_state,
PAD_SLOT_ID, ragged varlen, out-buffer coverage via a NaN sentinel, the
Kimi-K3 shape, and two rejection cases. Output and final state are both
asserted bitwise; a gather bug can leave output correct while corrupting
state.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Rewire the Kimi-K3 prefill call site

**Files:**
- Modify: `atom/models/kimi_k3.py:1005-1045` (`_run_kda`)
- Modify: `atom/models/kimi_k3.py:1118-1157` (prefill branch)

**Interfaces:**
- Consumes: `chunk_kda` from Task 3, with the signature given there.
- Produces: nothing consumed by later tasks. Task 6 depends only on `gather_kda_initial_state` no longer being called from here.

Safety note: `_forward_impl` is reached through the opaque custom op `torch.ops.aiter.kda_attention_with_output` (`kimi_k3.py:1056`), which splits the Dynamo graph. Editing `_forward_impl` does not violate the `@support_torch_compile` rule. Do not touch `KimiLinearModel`.

- [ ] **Step 1: Rewrite `_run_kda`**

Replace the body of `_run_kda` (`kimi_k3.py:1005-1045`) with:

```python
    def _run_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        output_final_state: bool,
        h0_indices: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
        inplace_final_state: bool = False,
        o: torch.Tensor | None = None,
    ):
        from atom.model_ops.fla_ops.kda import chunk_kda

        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            # beta arrives as bf16 logits and is NOT widened here. The accuracy
            # constraint is on the sigmoid *result*: fused_beta_sigmoid_fwd
            # allocates its output fp32 regardless of input dtype
            # (fla/ops/common/gate.py:59), so the write strength is fp32 either
            # way and the .float() this used to do was a redundant d2d copy. The
            # gsm8k regression that motivated the original widening was a bf16
            # sigmoid *output*, which cannot occur on this path.
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self._kda_gate_lower_bound is not None,
            lower_bound=self._kda_gate_lower_bound,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
            # FLA's default KDA recompute specialization is non-deterministic
            # for long, packed gfx950 prefills and can emit extreme values.
            # disable_recompute selects its STORE_QG specialization, which is
            # stable and preserves the same chunk-KDA forward semantics.
            disable_recompute=True,
            h0_indices=h0_indices,
            has_initial_state=has_initial_state,
            inplace_final_state=inplace_final_state,
            o=o,
        )
```

Two renames come with this: the deprecated `transpose_state_layout=True` becomes `state_v_first=True` (the vendored entry does not accept the alias), and `beta.float()` is gone.

- [ ] **Step 2: Collapse the prefill branch**

Replace `kimi_k3.py:1132-1157` — everything from the three `rearrange` calls through `out.copy_(...)` — with:

```python
            q = rearrange(q, "t (h d) -> 1 t h d", d=self.head_dim)
            k = rearrange(k, "t (h d) -> 1 t h d", d=self.head_dim)
            v = rearrange(v, "t (h d) -> 1 t h d", d=self.head_dim)
            # One call, no surrounding copies: the kernel reads the initial
            # state from ssm_state[state_indices] (zero-started where
            # has_initial_state is False), writes the final state back to the
            # same slots inplace, and writes the recurrence output straight into
            # `out`. This is the same shape as the decode branch below.
            #
            # `out` needs no pre-zeroing: it is exactly [num_actual_tokens, H, D]
            # with no padding rows, and the output kernel's store covers every
            # (t, v) position in that range.
            self._run_kda(
                q,
                k,
                v,
                gate,
                beta,
                ssm_state,
                query_start_loc,
                True,
                h0_indices=state_indices,
                has_initial_state=gdn_metadata.has_initial_state,
                inplace_final_state=True,
                o=out.unsqueeze(0),
            )
```

`out` is `[num_actual_tokens, H, D]` (allocated at `:1110-1112`) and the kernel wants `[1, T, HV, V]`; `unsqueeze(0)` is a view, so writes land in `out`'s storage. `state_indices` is `int32` (`gdn_attn.py:135-139`) and `has_initial_state` is `torch.bool` (`gdn_attn.py:364`), matching what the kernel loads.

- [ ] **Step 3: Verify no dead references remain in this file**

```bash
grep -n "gather_kda_initial_state\|transpose_state_layout\|beta.float()\|out.copy_" atom/models/kimi_k3.py
python -c "import ast,sys; ast.parse(open('atom/models/kimi_k3.py').read()); print('parses')"
black atom/models/kimi_k3.py && ruff check atom/models/kimi_k3.py
```

Expected: `grep` prints nothing, then `parses`, then clean lint.

- [ ] **Step 4: Fix-then-sweep — check the same pattern elsewhere**

CLAUDE.md requires this after any fix. Look for other prefill sites doing a manual gather / scatter / output copy around a chunk call:

```bash
grep -rn "ssm_state\[.*\] = \|\.copy_(.*squeeze(0))" --include=*.py atom/ | grep -v __pycache__
```

Expected: nothing in `atom/models/kimi_k3.py`. If other models show the same shape, note them in the commit message as follow-ups — do **not** change them here; they use GDN (base-e) kernels and are out of this plan's scope.

- [ ] **Step 5: Commit**

```bash
git add atom/models/kimi_k3.py
git commit -m "perf(kimi-k3): fuse the KDA prefill gather, scatter, and output copy

The prefill branch now makes one chunk_kda call against the vendored kernels
instead of gathering the initial state, scattering the final state, and copying
the output around a stock call. Also drops beta.float(): fused_beta_sigmoid_fwd
allocates fp32 output regardless of input dtype, so the widening was a
redundant d2d copy and the sigmoid result -- which is what the gsm8k regression
was about -- is fp32 either way.

Removes from the profile: gather_kda_state_kernel, the beta cast, input_guard's
contiguous() copies, the output zeros_like, index_copy_, and out.copy_.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Delete the orphaned gather and validate end to end

**Files:**
- Delete: `atom/model_ops/kimi_k3/kda_state.py`
- Modify: `atom/model_ops/kimi_k3/__init__.py:11,15`

**Interfaces:**
- Consumes: Task 5 must be complete — `kimi_k3.py` must no longer import `gather_kda_initial_state`.
- Produces: nothing.

- [ ] **Step 1: Sweep for callers before deleting**

```bash
grep -rn "gather_kda_initial_state\|kda_state\|_gather_kda_state_kernel" \
  --include=*.py --include=*.md . | grep -v __pycache__
```

Expected: exactly two hits, both in `atom/model_ops/kimi_k3/__init__.py` (lines 11 and 15). If anything else appears — especially under `atom/plugin/` — stop and report it; the file has another consumer and this task's premise is wrong.

- [ ] **Step 2: Remove the re-export and delete the file**

In `atom/model_ops/kimi_k3/__init__.py`, delete the import line:

```python
from atom.model_ops.kimi_k3.kda_state import gather_kda_initial_state
```

and the `__all__` entry:

```python
    "gather_kda_initial_state",
```

Then:

```bash
git rm atom/model_ops/kimi_k3/kda_state.py
python -c "import atom.model_ops.kimi_k3 as m; print(m.__all__)"
```

Expected: `['apply_attn_res', 'rmsnorm_gated', 'situ_and_mul']`

- [ ] **Step 3: Full lint and test**

```bash
black . && ruff check .
python -m pytest tests/ -q --ignore=tests/kernels
AITER_LOG_LEVEL=WARNING python -m pytest tests/kernels/ -v
```

Expected: clean lint, the CPU suite passes, 9 GPU tests pass.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "refactor(kimi-k3): drop the now-unused KDA state gather

gather_kda_initial_state's only caller was the prefill branch, which now gets
the gather from the kernel itself. Swept the repo for other callers before
removing; the re-export goes with it.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 5: End-to-end serving validation**

Per CLAUDE.md's serving rules. Do each step; do not skip the VRAM check.

```bash
# Stale compile cache causes silent failures after kernel changes.
rm -rf /root/.cache/atom/*
export AITER_LOG_LEVEL=WARNING
```

Start the server per `recipes/Kimi-K3.md`. Then confirm the model is actually loaded — `curl /health` returns OK even when it is not:

```bash
rocm-smi --showmemuse
```

Expected: VRAM% > 0 on the TP ranks.

Then run a short generation and confirm the output is coherent. On any server or GPU error, run `/debug-guide` first — do not blindly retry.

- [ ] **Step 6: Confirm the profile actually improved**

Capture a trace of a prefill-heavy request and check the KDA region. All seven items from the spec's inventory should be absent:

1. `gather_kda_state_kernel`
2. the `beta` cast copy
3. (subsumed by 2)
4. `input_guard` contiguous copies
5. the output `zeros_like`
6. `index_copy_` into `ssm_state`
7. the `out.copy_` d2d

Expected: the KDA prefill region drops from nine launches plus copies to seven launches with no d2d copies between the conv and `o_norm`.

If any item is still present, report which one and what is still calling it rather than adjusting the plan silently.

- [ ] **Step 7: Accuracy check**

Run `lm_eval` against the CI threshold documented in `/ci-pr-guide`. gsm8k is the relevant task here — it is the benchmark the dropped `beta.float()` was originally added for, so a regression there is the specific signal that the dtype reasoning in Task 5 was wrong.

Expected: at or above the CI threshold. If gsm8k regresses, restore `beta.float()` in `_run_kda` as the first hypothesis and re-measure before investigating anything else.

---

## Self-Review

**Spec coverage.** Each spec section maps to a task:

| Spec section | Task |
|---|---|
| Inventory items 1, 6 (gather, index_copy) | 1, 5 |
| Inventory items 2, 3 (`beta.float()`, redundant fp32) | 5 |
| Inventory item 4 (`@input_guard`) | 3 |
| Inventory items 5, 7 (`zeros_like`, `out.copy_`) | 2, 5 |
| Architecture / new package layout | 1, 2, 3 |
| "Why not `chunk_delta_h_vk`" | Global Constraints + Task 1 Step 3 header |
| `chunk.py` entry validation table | 3 Step 2 |
| `chunk_delta_h.py` indexed gather/scatter | 1 Steps 4-5 |
| `chunk_o_gk.py` out-param | 2 Step 3 |
| Call site | 5 |
| `kda_state.py` deletion + sweep | 6 Steps 1-2 |
| Error-handling table (all six rows) | 3 Step 2, 1 Step 5, tested in 4 |
| Testing: 7 parity cases | 4 (plus two rejection cases) |
| Testing: end-to-end (5 substeps) | 6 Steps 5-7 |
| Risks: silent numeric drift | 4 (state asserted, not just output) |
| Risks: fla version drift | Global Constraints + headers in 1, 2, 3 |
| Risks: partially-written `o` | 4 case 6 |
| Conventions | Global Constraints |

One spec risk row has no dedicated task step: *"`flash_kda` later installed, diverting stock `chunk_kda`."* This is a latent hazard rather than work — the vendored path never dispatches, so only the parity test's reference side could change, and it would change loudly (the test asserts bitwise equality, so a different backend fails rather than silently passes). No task added.

**Placeholder scan.** No TBDs, no "add error handling," no "similar to Task N." Every code step carries the actual code. Task 4 Step 3 says "fix until green" without listing fixes, which is unavoidable for a first kernel bring-up — mitigated by the explicit triage guide in Step 2 that names the likely failure and its cause for each case.

**Type consistency.** Checked across tasks:
- `chunk_gated_delta_rule_fwd_h(..., h0_indices, has_initial_state, inplace_final_state)` — defined Task 1, called Task 3 with those exact names.
- `chunk_gla_fwd_o_gk(..., o=)` — defined Task 2, called Task 3.
- `chunk_kda(..., h0_indices, has_initial_state, inplace_final_state, o)` — defined Task 3, called Task 4 and Task 5.
- `state_v_first` (not `transpose_state_layout`) used consistently in Tasks 1, 2, 3, 5.
- `_run_kda`'s new keyword names match `chunk_kda`'s.
- Test helper `_flags()` passes `state_v_first=True` to both stock fla (which accepts it, `fla/ops/kda/chunk.py:195`) and the vendored entry.
