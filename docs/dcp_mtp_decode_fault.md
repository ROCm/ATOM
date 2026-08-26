# GLM-5.2 DCP decode + MTP: the guards were hiding a kernel-path switch

Measured 2026-08-26 on MI355X, GLM-5.2-MXFP4, PP4xTP1 prefill -> TP4xDCP4 decode
over mooncake PD, `ATOM_DCP_REPLICATE_INDEX_CACHE=1`, `--method mtp
--num-speculative-tokens 3`, `SPEC_ACCEPT_RATE=off`.

Context: `docs/dcp_decode_topk_bottleneck.md` showed the DCP decode trails the
DPA baseline on *tokens per step*, not time per step -- DPA ran MTP and the
replicated index cache forbade speculative decode outright. This lifts that.

Result: **GSM8K 1319 = 0.9303 +/- 0.007**, zero GPU faults, against 0.9295
(replicate=0, no MTP) and 0.9356 (replicate=1, no MTP) on the same harness.

## Root cause: MTP silently changed which attention kernel runs

Three `speculative_config is None` guards blocked MTP. Removing them is correct
on its own terms -- the stack comes up, PD transfer is unaffected, and a single
request decodes correctly. Under GSM8K-shaped load (5-shot ~1100-token prompts,
`num_concurrent=16`) the decode node faulted before emitting a token.

rocm-debug-agent, 382 wave dumps, one kernel in all of them:
`_fwd_kernel_stage2_asm` (`aiter/mla.py:177`), `MEMORY_VIOLATION`.

That kernel is the *intra-rank* split-KV merge -- one rank splits its own KV into
`num_kv_splits` chunks, scores them in parallel, and merges the partials into the
`final_lse` that then feeds DCP's cross-rank combine. It sits one level below the
DCP q/lse/output exchange, and the validated non-MTP config never runs it at all:

| config | `Sparse DCP persistent attention enabled` |
|---|---|
| non-MTP (validated 0.9356) | 4 (one per rank) |
| MTP (faulting) | 0 |

The chain is `speculative_config is not None` -> `sparse_dcp_metadata_rebuild`
False -> `use_persistent_mode` False -> aiter gets `work_meta_data=None` and takes
the non-persistent split-KV path.

Why that path cannot work here: it hands **every row the same number of split
slots** (`num_kv_splits_indptr` is a uniform `arange`), while the effective count
is `min(that, cdiv(cur_kv_seq_len, mgc))` with `cur_kv_seq_len` read from the
*compacted* DCP indptr. Under DCP each row holds only the top-k slots this rank
owns, so row lengths vary; the persistent path describes that with a real work
plan, the split-KV path cannot.

Note the variable row lengths are **not** MTP-specific -- non-MTP DCP rows are
compacted too. MTP's only role was flipping the kernel path.

## Fix

- Drop `speculative_config is None` from both copies of
  `sparse_dcp_metadata_rebuild` (`attention_mla.py`, `aiter_mla.py` -- the same
  predicate is spelled out twice).
- Pass `work_prefix="sparse_mtp_"` when rebuilding the persistent work plan under
  MTP. The `sparse_mtp_*` buffers were already built and already selected by
  `_forward_decode`; only the rebuild call was still writing the unprefixed set,
  and its empty-prefix branch asserts `max_seqlen_q == 1`.

DPA already ran MTP on the persistent path, so persistent+MTP was proven; only
persistent+MTP+DCP was untried.

### Also fixed: indexer/attention layout disagreement

Routing MTP through the DCP compacting filter means the indexer writes compacted
per-token regions with lengths in `dcp_sparse_kv_indptr_buffer`. Only the qlen=1
branch of `_forward_decode` read that buffer; the `max_seqlen_q > 1` branch still
used `sparse_kv_indptr` (uniform top-k stride) and walked past each written
region. `B` is a token count on both branches -- sparse layers in MTP verify run
at `max_q_len == 1` -- so one substitution covers them.

### Ruled out: the split count / `logits` transient

The non-persistent fallback sizes a transient fp32 `logits` as
`(total_s, num_kv_splits, nhead, v_head_dim)`, and MTP makes `total_s` a
per-draft-position count, so `16 // dcp_world_size` under-divides by
`max_seqlen_q`. Dividing it out does not help:

| `num_kv_splits` | conc 1 numerics | GSM8K conc 16 |
|---|---|---|
| 4 (`16 // dcp`) | correct | faults |
| 2 | correct | still faults |
| 1 | **garbage** | no fault |

At exactly 1, `FINAL_OUT` becomes true (`num_kv_splits_indptr[BATCH_NUM] == bs*N`
equals `BATCH_NUM == bs` only when `N == 1`) and the merge is skipped entirely --
which is why that row neither faults nor computes anything right. The clean run
at 1 is evidence the reduce was bypassed, not that the transient was too large.
The 16 is just aiter's search cap (`for i in range(1, 17)` in `get_meta_param`).

Buffer capacity was never it either: `_sparse_kv_indices_gpu` is
`max_num_batched_tokens * index_topk` = 16384 x 2048 int32 = 134 MB, against at
most `max_num_seqs * 4` = 2048 rows.

## Not yet done

- **Throughput unmeasured.** The whole point is tokens-per-step; accuracy was
  only the gate. 431 tok/s (no MTP) vs the DPA baseline's 797 is the number to
  move.
- The `recipes/mesh/GLM-5.2.md` eval (`local-chat-completions`,
  `--apply_chat_template --fewshot_as_multiturn`, threshold >= 0.93) aborted with
  `ServerDisconnectedError` -- `max_gen_toks=16384` outruns the client/proxy idle
  timeout. Server stayed healthy (no faults, no dead ranks). That harness is not
  comparable to the numbers above and needs its timeout raised before it can be
  used as a gate.

## Reproducing

```bash
docker exec -e ENABLE_MTP=1 -e SPEC_ACCEPT_RATE=off \
  -e PYTHONPATH=/it-share/yajizhan/code/ATOM \
  atom_pp4pd_dcp bash /it-share/yajizhan/code/ATOM/scripts/start_glm52_pp4pd_dcp.sh
docker exec atom_pp4pd_dcp bash /it-share/yajizhan/code/ATOM/scripts/eval_glm52_pp4pd_gsm8k.sh
```

`PYTHONPATH` is load-bearing whenever the launch changes cwd (e.g. running under
rocm-debug-agent, which must cd to a scratch dir to keep 4 MB code objects out of
the repo). The container's `WorkingDir` is the working tree, but the editable
install resolves to `/app/ATOM`. The tell-tale of getting this wrong is
`AssertionError: DCP + DeepSeek-V3.2 sparse indexer (DSA) currently supports
qlen=1 decode only`, which only the stale tree can reach.
