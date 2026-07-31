# SSM state cache — design & implementation notes

Prefix caching for mamba-like (linear-attention) models: Qwen3-Next, Qwen3.5,
Kimi K3 (KDA). Working document — enough context to resume on another machine.

**Status:** Qwen3.5 GPU-verified. Kimi K3 wired and kernel-verified, **not**
run end-to-end. Opt-in behind `--enable_ssm_state_cache` (default off).

**Branch:** `ganyi/state_cache`, rebased onto `d1aa2506`.
Commit `b0fdefa8 "support mamba prefix cache"` + uncommitted Kimi work.

---

## 1. The problem

A GDN/KDA layer keeps its state in a per-request slot outside the paged KV
pool. That state is a **fold over every token** — it cannot be reconstructed
from paged KV.

Take a KV hit of `P` tokens and the recurrence enters the uncached suffix
having never seen the first `P`. Wrong output, **silently and at full cache
speed** (measured: a repeated 18k prompt answered with a hallucinated value).

So without this feature such models must refuse every KV hit. The clamp that
makes hits safe:

```
KV hit is clamped to the deepest state checkpoint at or below it,
and is 0 when the state cache is disabled or misses.
```

Treat any path that widens a hit beyond a checkpoint as a correctness bug.

---

## 2. Design principles

**2.1 The pool stores copies, never live state.**
A request's live recurrence stays in its *runtime slot*
(`BlockManager.free_per_req_cache_groups`, one per concurrent request, never
evictable) for its whole life — prefill chunk 1 through the last decode step.
Checkpoints are read-once copy sources in a separate region of the same tensor.

Invariant: **losing a checkpoint costs performance, never correctness.**

An earlier design made chunk-to-chunk continuity depend on cache slots.
Simulated at 16 concurrent requests it gave `skipped_full=80` and sequences
losing their own resume point. Rejected. It's also why `pin_count` is a DMA
counter, not a refcount.

**2.2 Grid-aligned positions, content-addressed keys.**
`P` must be a multiple of `granularity` (default 64 = `lcm(kv_block_size,
FLA chunk)`). On `fla` 0.5.2, split-and-replay via `initial_state` is
**bit-exact** at multiples of 64, approximate off-grid.

The key is the chained block hash at `P` — the *same* chain the KV pool uses.
**The pool never hashes**: `BlockManager` owns hashing and hands the chain over
on the sequence.

**2.3 Two checkpoints per request.**

| position | serves |
|---|---|
| prompt end (grid-floored) | whole-prompt reuse (follow-up turn) |
| observed fork | mid-prefix branching (fixed system prompt, varying tail) |

Reserving only the fork was **worse than no fork detection** — the request
skipped its anchor *and*, if no forward landed exactly on the fork, wrote
nothing.

**2.4 Eviction: hit count first, recency as tiebreak.**
Most checkpoints sit inside content nobody will share, and they're always the
*most recently* written — so pure LRU keeps the junk and evicts the shared
branch points the cache exists for.

**2.5 All pool mutation in `allocate`, never `can_allocate`.**
`can_allocate` doubles as a read-only KV-pressure probe that may never admit.
It records observations on the seq and mutates nothing. Pinning there leaked
slots; crediting demand there made cold entries look hot.

---

## 3. Request lifecycle

```
can_allocate   (pure query)
  ├─ chain block hashes over the WHOLE prompt (anchor sits past the match)
  ├─ record fork_hit_blocks at the divergence, floored to the grid
  └─ bounded_hit: clamp the KV hit to the deepest published checkpoint

allocate       (every pool mutation)
  ├─ acquire_load  — re-validate identity, then pin
  ├─ credit_demand — a re-derived position is demand; eviction must see it
  └─ plan_save     — reserve fork + prompt-end

forward
  ├─ apply_state_cache_loads   checkpoint -> runtime slot (2 strided copies)
  ├─ chunk kernel              reads AND writes the runtime slot in place
  └─ write_state_checkpoints   one launch, both halves, both kinds

on_prefill_step_done
  ├─ commit_save   publish only positions the step actually REACHED
  └─ release_load  unpin the source
```

Publishing only on exact arrival matters: overshoot by one token and the next
request resumes mid-stream and diverges silently.

---

## 4. File map

| file | role |
|---|---|
| `atom/model_engine/state_cache.py` (610) | the pool. 4 members: `_free`, `_allocated`, `_pending`, `_tick` |
| `atom/model_engine/block_manager.py` | `can_allocate` clamp, `allocate` mutations, hash chain |
| `atom/model_engine/scheduler.py` | `state_load_slots`, `state_save_all` on `ScheduledBatch` |
| `atom/model_ops/fla_ops/chunk_delta_h.py` | paged kernel: `state_indices`, `dst_indices`, `h0_mask`, `state_v_first` |
| `atom/model_ops/fla_ops/state_checkpoint.py` (150) | one-launch checkpoint writer |
| `atom/model_ops/fla_ops/chunk_kda.py` (142) | **Kimi**: paged KDA forward |
| `atom/model_ops/attentions/gdn_attn.py` | metadata: `_checkpoint_targets`, `apply_state_cache_loads` |
| `atom/model_ops/attention_gdn.py` | GDN impl: paged call + checkpoint write |
| `atom/models/kimi_k3.py` | **Kimi**: `_run_kda_paged` + prefill branch |

---

## 5. Kernel notes

**5.1 `dst_indices` — paged write.** `dst` MAY equal `src`: each program owns
one `[K, BV]` column slice of one slot, loaded in the prologue and stored in
the epilogue, so no program reads a slice another writes. Bit-exact vs
gather/scatter at NT = 1, 2, 4, 8.

> **`restore_value=["h0"]` on the autotuner is required.** Autotuning re-runs
> the kernel per config; with `ht` aliasing `h0`, each trial consumes the
> previous trial's output. **Only shows on a cold `(H,K,V,BT,STATE_V_FIRST)`
> key** — repeats look fine — so tests must clear the autotune cache per case.
> An earlier version of the test passed with the fix removed.

**5.2 `state_checkpoint.py` — one launch, two sources.**
Split grid: `program_id(1) < NBLK_SSM` copies state, the rest copy conv
windows. Sources differ by position kind:

| target | source |
|---|---|
| interior (fork) | `h[chunk_offsets[row] + off/64]` |
| step end (anchor) | the **runtime slot** — *not in `h`* |

`h` holds chunk boundaries strictly *before* the end; index
`chunk_offsets[row] + T/64` is the **next sequence's** first chunk. `is_end`
tags which source to use.

It copies a flat `HKV` block via `stride(0)` and never interprets K vs V — so
it is **layout-agnostic** and needed no change for Kimi.

**5.3 `state_v_first` — Kimi's transposed state.**
Kimi's state is `[HV, V, K]`; GDN's is `[HV, K, V]`. ATOM branches only at
three endpoints — h0 load, ht store, `h` store — keeping the `[64, BV]`
accumulators and the whole recurrence layout-independent. fla instead flips
the accumulators, forcing the recurrence to branch.

The **`h` store must follow the same layout**: the state cache slices interior
checkpoints straight out of `h`, so a mismatch stores every Kimi fork
checkpoint transposed, and nothing else notices.

---

## 6. Kimi K3 wiring

**No fork needed.** `fla.ops.kda.chunk_kda_fwd` already calls
`fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_h` — the same function
ATOM forked for GDN (KDA passes `gk=g`, GDN `g=g`). So `chunk_kda.py` reuses
fla's gate/intra/output stages **unchanged** and swaps in ATOM's paged `fwd_h`
for the one call touching recurrent state. ~2900 lines avoided.

```
kimi_k3.py  _forward_impl (prefill branch)
  ckpt = gdn_metadata.ssm_checkpoints
  if ckpt is None:   -> _run_kda (unchanged: gather / fla chunk_kda / scatter)
  else:              -> _run_kda_paged  -> chunk_kda_paged
                        then write_state_checkpoints(...)
```

`dst_indices == state_indices` (a prefill chunk advances its own runtime slot).
`beta.float()` is preserved — a bf16 sigmoid erodes the delta-rule write
strength across 71 KDA layers (measured GSM8K regression upstream).
`disable_recompute=True` is preserved — fla 0.5.1's default recompute path is
non-deterministic for long packed gfx950 prefills.

Already true, no change needed: `kimi_linear` is in
`LLMEngine._per_req_cache_model_types()`, so `has_per_req_cache=True`;
`_state_shape` returns `(heads, head_v_dim, head_k_dim)` which is already
v-first; `_state_dtypes` returns fp32 for the SSM half; `linear_conv_kernel_dim`
is set in `_init_gdn_state`.

**Latent bug fixed along the way:** `_gdn_layer_index` assumed
`full_attention_interval`, which Kimi has no attribute for — it lists layers
explicitly in `kda_layers` with irregular spacing, so the interval formula
aliased two layers onto one cache row. Present on upstream too.

---

## 7. Bugs found during development

Each is a review hazard; most were invisible to single-sequence tests.

| bug | symptom |
|---|---|
| varlen indexing ignored per-sequence bases | one sequence's state written into another's checkpoint |
| prompt-end reserved, never written, **published anyway** | later request resumes from a previous tenant's bytes |
| `has_recurrent_state` false on a hit | kernel discards the checkpoint, restarts from zero — full speed, wrong answer |
| hash chain extended in place | inflated the reported match length |
| chain resumed from the failed block's hash | chain nothing reproduces; every lookup misses |
| pin taken in `can_allocate` | probe leaked a pin; entry never evictable |
| `summary()` referenced an undefined name | `NameError` in the scheduler log path |
| `last_h` captured on every call | pinned 33.5 MB for plugin callers that never pop it |
| `_gdn_layer_index` interval formula | aliases two Kimi layers onto one cache row |

---

## 8. Test inventory

107 tests. GPU-only ones skip on CPU CI.

| file | covers |
|---|---|
| `test_state_cache_pool.py` | slot alloc, reserve/publish/cancel, pinning |
| `test_state_cache_eviction.py` | hit-count-first, recency tiebreak, pinned-safe |
| `test_state_cache_forks.py` | fork observation, grid floor, demand crediting |
| `test_state_cache_blockmanager.py` | hit path, probe purity, hash chain |
| `test_state_cache_config.py` | validation, granularity |
| `test_gdn_has_initial_state.py` | seeding on hit |
| `test_gdn_paged_state.py` | paged read/write, aliasing, **v-first** |
| `test_state_checkpoint_kernels.py` | per-sequence bases, both halves, `is_end` |
| `test_kimi_kda_paged.py` | **Kimi**: vs fla, paging, `h` layout, checkpoint |

**Mutation-checked** (each fails a distinct subset when broken): per-sequence
bases, grid split, `is_end` source, `restore_value`, `state_v_first`, beta
sigmoid, h0/ht/`h` transposes.

Baseline: **1144 passed**; 78 failed, all pre-existing (47 unrelated +
31 sglang import pollution — identical on a clean `HEAD` worktree).

---

## 9. Verification status

**Qwen3.5-27B-FP8, TP8, 4096 slots, GPU:**

| | latency | needle recovered |
|---|---|---|
| cold | 7.08 s | yes |
| cached (×3) | 2.80 s | yes |

2.5× on a 62k-char prompt with a needle buried mid-prefix — a bad resume loses
the needle, so this tests correctness, not just speed.

**Kimi K3:** kernel bit-exact vs `fla.ops.kda.chunk_kda` (output + both state
slots, `maxdiff=0.0`); CPU-side scheduler path verified on a Kimi-shaped
config (whole-prompt reuse, fork, re-fork resume). **No end-to-end run** — no
K3 weights on the dev box.

### Not verified

- **GSM8K at a real hit rate.** The 20-shot run (0.8688 vs 0.8673 baseline) was
  measured *before* the prompt-end fix and at a **0.4% hit rate**. Treat it as
  "did not corrupt", not as validation. Re-run at 4096 slots.
- **Kimi end-to-end** — the needle probe is what caught the real bugs on the
  Qwen path.
- Spec decode, TP sizes other than 8, preemption mid-prefill.
- Pre-existing, unrelated: the conv kernel's `v` output is nondeterministic on
  the plain path (a 72-char prompt with no possible checkpoint also varies).

---

## 10. Resuming on another machine

```bash
git checkout ganyi/state_cache          # b0fdefa8 + uncommitted Kimi work
python -m pytest tests/test_state_cache_*.py tests/test_gdn_*.py \
                 tests/test_state_checkpoint_kernels.py \
                 tests/test_kimi_kda_paged.py -q      # expect 107 passed

# Qwen3.5 e2e
python -m atom.entrypoints.openai_server \
  --model <qwen3.5> --kv-cache-dtype fp8 -tp 8 \
  --enable_prefix_caching --enable_ssm_state_cache --ssm_state_cache_slots 4096
```

Note the flag is `--enable_ssm_state_cache` (**underscores**). Passing only
`--enable-prefix-caching` silently leaves the state cache off — check the
startup kwargs dump for `'enable_ssm_state_cache': True` before trusting any
measurement. Stats log every 100 requests.

### Next steps, in priority order

1. **GSM8K 20-shot at 4096 slots** — replaces the discredited number. ~10 min.
2. **Kimi K3 e2e** — needle probe first, then GSM8K.
3. Decide on `apply_state_cache_loads` (see below).
4. Spec decode / other TP sizes.

### Known open questions

- **`apply_state_cache_loads` could fold into the kernels** but probably
  shouldn't. It is 2 strided copies (~20 µs) **per cache hit**, not per layer —
  `[:, slot]` spans every layer at once. Removing it needs the conv kernel's
  2-D `cache_indices` APC path (verified working, but nothing in ATOM currently
  exercises it) kept in lockstep with the chunk kernel's `state_indices`.
- **Auto-sizing is `max_num_seqs // 2`**, deliberately a pure function of
  config: it must be identical across TP ranks. Deriving it from live free
  memory gave 2809/2810/2812 slots on 8 ranks and an `IndexError`.
- **DCP > 1 untested.** `block_manager.py` uses raw `block_size` for the fork
  position while neighbouring code uses `_hash_block_size()`. Self-consistent
  with `state_cache.py` today, but both would be wrong together under DCP.
