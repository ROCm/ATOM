# DSpark Implementation Notes (ATOM)

Developer-oriented map of how DSpark speculative decoding is wired into ATOM.
For *how to run it*, see `recipes/DeepSeek-V4-DSpark.md`. This doc is about the
*code paths*.

DSpark (DeepSeek-AI, 2026) is a **semi-autoregressive block drafter**. Unlike
serial MTP (drafts `k` tokens over `k` sequential passes), DSpark drafts a whole
block in **one parallel backbone pass**, then the target model verifies it. It
ships **inside the V4 checkpoint** under the `mtp.*` namespace but is a different
architecture from serial MTP, detected by the `dspark_block_size` config field.

---

## 1. Files

| File | Role |
|------|------|
| `atom/models/deepseek_v4_dspark.py` | DSpark **draft model** (parallel backbone + Markov head + confidence head). Read its module docstring first — it's the best single source. |
| `atom/spec_decode/eagle.py` | `EagleProposer`: the drafter driver. `use_dspark` branch routes to `_propose_dspark`. Also owns verify-metadata construction (`calc_spec_decode_metadata`) shared with MTP. |
| `atom/spec_decode/dspark_scheduler.py` | Hardware-Aware Prefix Scheduler (paper Alg. 1). Pure/GPU-free. Picks per-request verify length `ell`. Only used when `confidence_schedule=true`. |
| `atom/model_engine/model_runner.py` | Orchestration: aux-hidden capture hooks, q-bucket / ragged shrink, DP graph-shape sync, verify (`rejection_sampler`), draft proposal. |
| `atom/model_ops/rejection_sampler.py` | Greedy / relaxed rejection verify kernels. **Shared** by MTP and DSpark. |
| `atom/config.py` | `DSparkConfig` (the `--dspark-config` knobs). |

---

## 2. Config (`DSparkConfig`, `atom/config.py`)

Built once in the parent from `--dspark-config` (JSON), then
pickled into every worker as `config.dspark.*`. **No `os.environ` at read sites.**

| Field | Meaning |
|-------|---------|
| `confidence_schedule` | Use confidence head → per-request verify length `ell` (variable-length verify). Off = static full-block verify. |
| `ragged` | Per-request ragged verify (§5.2): each seq forwards its own `ell_r+1` tokens, no batch-level q padding. |
| `ragged_graph_sizes` | CUDA-graph query-length buckets for the ragged path (e.g. `"8"`). |
| `q_buckets` | CUDA-graph query-length buckets for the (older) batch-uniform q-bucket path. |
| `disable_sps_calib` | Skip SPS calibration; use synthetic SPS stub. |

Activation: `--method dspark --num-speculative-tokens K`. `arg_utils.py` builds a
`SpeculativeConfig(method="dspark", model=self.model, ...)`. Without `--method`,
no speculative_config is created and the model runs as **plain target** (the
`mtp.*` DSpark weights are simply not loaded as a drafter).

---

## 3. Draft model (`deepseek_v4_dspark.py`)

`DeepseekV4DSpark` mirrors `DeepseekV4MTP`'s wrapper contract:
- Loads `mtp.*` weights via standard `load_model(spec_decode=True)`.
- Shares `embed` / `head` with the target via `share_with_target`.

Two mechanisms (paper §3):
1. **Parallel backbone** (`DSparkLayer(Block)`): `dspark_block_size` V4 decoder
   layers with mHC + sliding-window attention over a **private rolling
   target-KV window**. Produces all base logits `U_1..U_gamma` in one pass.
   `DSparkLayer` inherits `Block` to reuse attn linears / MoE / mHC; only the
   attention *compute* is replaced (draft block attends its rolling window).
2. **Sequential Markov head** (`DSparkMarkovHead`): low-rank transition bias
   `B = W1 @ W2` (rank `dspark_markov_rank`). `forward_head` samples the block
   left-to-right, greedy argmax per position: `x_k = argmax(U_k + B(x_{k-1}))`.
   The bias is added *inside* a per-position softmax (local correction), so
   per-token probs stay exact → lossless verification.
3. **Confidence head** (`DSparkConfidenceHead`): `c_k = sigmoid(w^T[h_k; W1[x_{k-1}]])`,
   the survival prob consumed by the scheduler (only if `confidence_schedule`).

Entry point: `forward_spec(input_ids, main_hidden, positions, cache_indices,
num_draft)`:
- Injects target context: `main_proj(concat target hidden [58,59,60])` → norm.
- Builds draft block ids `[anchor, noise, noise, ...]` (`noise_token_id`).
- Runs all stages in one parallel pass → `base_logits [B,T,vocab]`.
- `forward_head` → sequential Markov sampling + confidence.

Rolling window KV lives in the **shared paged pool** (`self.attn.swa_kv`, a slice
of `unified_kv`), content-addressed by `swa_block_tables` — same paged-SWA
machinery as the V4 target (#1417). `precompute_context_kv` writes it via the
`swa_write` Triton kernel.

---

## 4. Target-side integration (`model_runner.py`)

DSpark needs target hidden states from `dspark_target_layer_ids` (e.g. [58,59,60]).
`V4ForCausalLM` is `@support_torch_compile` (must not be edited), so these are
captured via **forward hooks**:

- `_install_dspark_aux_hooks()` (gated on `use_dspark`, last PP rank): registers
  a **read-only** forward hook on each target layer that reduces the mHC residual
  (`mean(dim=1)`) into a fixed preallocated buffer (cudagraph-safe, no host sync).
- `_collect_dspark_aux(num_tokens)`: slices those buffers to the step's tokens,
  read out in `run_model` as `self._aux_hidden_states`.

**Only DSpark installs these hooks; serial MTP does not.**

---

## 5. Per-step decode flow

```
prepare_model(batch):
  _dspark_apply_q_bucket(batch)          # optional shrink (confidence_schedule)
    → ragged branch _dspark_apply_ragged if config.dspark.ragged
  [DP only] _dspark_local_shape + merged sync_dp_metadata all_gather
            + _apply_dspark_shape_max    # DP-max (q, bs, total_tokens), see §6
  prepare_input_ids(batch)               # build [anchor, draft...] per seq
  prepare_inputs(batch, ...)             # attn metadata, ForwardMode.decide

run_model():                             # target forward (q = num_spec_query_tokens)
  _collect_dspark_aux(num_tokens)        # gather target aux hidden

postprocess():
  rejection_sampler.forward(...)         # verify: accept draft==target argmax
  propose_draft_token_ids(...)           # → _propose_dspark → forward_spec
    record_dspark_ell(req_ids)           # async D2H of ell for NEXT step
```

Key single-source-of-truth: `batch.num_spec_query_tokens` = per-seq forward
length this step (= `mtp_k+1` for full, or the shrunk q-bucket). All length
consumers read it.

---

## 6. Verify (shared with MTP)

`calc_spec_decode_metadata` (eagle.py) builds, from the **actual forward layout**
(`num_sampled_tokens`), the indices verify needs:
- `num_draft_tokens = num_sampled_tokens - 1` (clipped to `[0, mtp_k]`)
- `target_logits_indices`, `bonus_logits_indices`, `cu_num_draft_tokens`
Deriving from `num_sampled_tokens` (not a separate `mtp_k` constant) keeps
draft/target/bonus indices consistent when the batch is q-shrunk.

`rejection_sampler.py` greedy kernel: for each request, walk draft positions;
accept while `draft_token == target_argmax`; on first mismatch, emit the target
argmax and stop (`rejected=True`). Output is **always the target-greedy token**
at accepted positions → lossless. Tail `[num_draft+1 .. num_spec_steps]` filled
with `-1` sentinel (downstream truncates on first `-1`). `num_bonus_tokens` =
accepted count, used to gather the next-step anchor.

---

## 7. Confidence scheduling (Phase 2, only if `confidence_schedule=true`)

- `_compute_schedule_ell` (eagle.py) → `schedule_prefix_lengths_tensor`
  (dspark_scheduler.py): given confidence + `sps_table`, pick per-request `ell`
  that maximizes `Theta = tau * SPS(B)` (greedy admission + early-stop). Kept
  sync-free for the hot path.
- `ell` is stashed by `record_dspark_ell` as `{req_id: ell}` (req_id-keyed —
  continuous batching reorders requests). The **next** step's
  `_dspark_apply_q_bucket` / `_dspark_apply_ragged` reads it to shrink verify
  length. `_dspark_ell_by_req` is an async-resolved property (D2H fired on a
  side stream, materialized lazily next step) to avoid stalling the pipeline.

Two shrink strategies (mutually exclusive, both gated on confidence_schedule):
- **q-bucket** (`q_buckets`): batch-uniform q = quantize_up(max ell + 1).
- **ragged** (`ragged`, `ragged_graph_sizes`): per-request `ell_r+1`, flat-packed;
  sets `num_spec_query_tokens_per_req`, which flips the V4 attention indexer to
  its ragged branch (`_score_topk_decode_ragged`, EAGER-only) and sets
  `dspark_ragged_lens_gpu` in attn metadata.

---

## 8. DP-attention specifics

Under DP, each rank picks its own verify length/bs from its local batch, which
diverges across ranks → the decode CUDA graph shape (bs, q) and the MoE
all_gather row count would mismatch → RCCL deadlock. Fix:
`_dspark_local_shape` computes local `(q, decode_bs, total_tokens)`; the DP-MAX is
taken via **`sync_dp_metadata`** (the packed TBO/DP all_gather — DSpark shape folded
in to avoid a second collective) and applied by `_apply_dspark_shape_max`. Raising
q/bs/total_tokens only enlarges graph capacity (real tokens stay flat-packed in
[0:total_tokens]),
so it is lossless. Run once in `prepare_model` (needs DP-max q before
`prepare_input_ids`) and reused in `prepare_inputs`.

---

## 9. Gotchas

- **Draft weights live under `mtp.*` but are NOT serial-MTP weights.** A V4
  checkpoint with DSpark lacks standard MTP tensors (`eh_proj`/`enorm`/`hnorm`);
  don't assume `--method mtp` gives a well-formed drafter on a DSpark checkpoint.
- **Draft correctness is irrelevant to output correctness.** A bad draft only
  lowers acceptance (speed); verify always emits the target-greedy token. Debug
  accuracy issues on the *verify / target-forward* side, not the draft.
- **q-bucket/ragged shrink is lossless**: dropped draft-suffix slots are
  re-drafted next step.
- Aux hooks are read-only; they do not mutate the target forward output.
