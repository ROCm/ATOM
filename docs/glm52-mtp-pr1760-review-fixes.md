# GLM-5.2 MTP: PR #1760 review fixes

This document tracks the correctness fixes requested during the final review of
PR #1760 and the validation evidence collected on this branch.

## Review findings and fixes

### GLM-5.1 decode CUDA-graph regression

Plain decode stopped resolving the fixed CUDA-graph metadata after the MTP
changes restricted that path to target verification. The eager fallback then
used the complete padded `out_cache_loc` graph buffer as `slot_mapping`, which
could feed invalid cache addresses to
`indexer_qk_rope_quant_and_cache_kernel`.

`forward_context.py` now resolves graph metadata for ordinary decode as well as
target verification. `glm52_dsa_bridge.py` limits eager `slot_mapping` to the
runtime batch width. This preserves fixed-buffer graph replay and makes the
fallback safe.

### PCP restoration

The review identified that PR #1760 had removed three necessary PCP pieces:

1. `register.py` again passes
   `prefill_context_model_parallel_size` to `aiter.init_dist_env`.
2. `glm52_dsa_bridge.py` reindexes sparse-prefill metadata to PCP-owned query
   rows, including its cumulative query offsets and request IDs.
3. `sparse_mla_indexer.py` applies the same dense per-query split to indexer
   request IDs and range metadata.

The metadata transformation was checked with a PCP=2 focused test. A full
PCP2 server needs TP4 × PCP2 (eight GPUs); only GPUs 4–7 were free for this
validation session, so a four-GPU GLM MTP test cannot prove PCP collectives.

### GLM and DeepSeek NextN separation

`_is_glm52_nextn_runner` previously treated `DeepseekV3ForCausalLMNextN` as
GLM. That sent DeepSeek V3/R1 through GLM-only graph staging and sub-step
paths. It now identifies only `GlmMoeDsaForCausalLMNextN`.

The generic GLM draft frontend marker is now set only by a GLM MTP wrapper.
DeepSeek NextN forwards therefore keep their normal DSA metadata construction.

The GLM draft graph staging predicate was also inverted: it now stages metadata
when the GLM generic draft frontend is active, rather than skipping that exact
path.

### SGLang draft-extend API compatibility

Newer SGLang releases replaced
`EagleDraftInput.prepare_for_extend_to_fill_draft_kvcache`. The patched DSV4
draft-extend path first checks for that legacy method; if absent it delegates
to SGLang's current implementation. This prevents the observed
`AttributeError` while retaining the plugin implementation for versions that
expose the older API.

### Runtime ownership and compatibility cleanup

* The attention-backend resolver now completes a partially supplied KV-pool
  pair from `full_attn_backend` instead of failing early.
* `typing.Self` was replaced with a forward reference so Python 3.10 can
  import `forward_context.py`.
* GLM hidden-state debug logging and unused indexer-context/accept-compaction
  helpers were removed.
* Standalone section-string expressions in `glm52_dsa_bridge.py` were converted
  to comments.
* SGLang speculative environment controls are registered in `atom.utils.envs`
  and documented in `docs/environment_variables.md`.

## Validation evidence

| Check | Result |
| --- | --- |
| Python compilation of modified plugin modules | Passed |
| Focused runtime-owner, graph-metadata, and MTP IndexShare tests | Passed: 20 tests |
| `git diff --check` | Passed |
| IDE diagnostics for edited files | No diagnostics |
| GLM-5.2 TP4 MTP3 server on GPUs 4–7 | Passed in `zhiwei_sgl_atom_0803` |
| Decode CUDA-graph capture | Passed: all 51 buckets through max batch size 256 captured |
| Draft-extend CUDA-graph capture | Passed: 6.32 s |
| `/generate`, 8192-token prefill + 1024-token decode | Passed: 5.99 s E2E, 48.7% speculative accept rate |
| Decode graph replay | Passed: runtime logs report `cuda graph: True` throughout decode |

The TP4 MTP recipe used `CUDA_VISIBLE_DEVICES=4,5,6,7`,
`--max-running-requests 256`, and `--cuda-graph-max-bs-decode 256` with the
`/workspace/shared/data/amd_int/models/GLM-5.2-MXFP4` checkpoint mounted in
`zhiwei_sgl_atom_0803`.

## Remaining validation gate

The code-level reviewer items above are addressed and the TP4 MTP graph path
has passed end-to-end validation.

## PCP2 validation status

A GLM-5.2 MXFP4 PCP2 attempt was made in `zhiwei_sgl_atom_0803` with all eight
GPUs free and the benchmark launch arguments:

```text
--tp-size 4 --enable-nsa-prefill-context-parallel --attn-cp-size 2
--nsa-prefill-cp-mode round-robin-split
```

The server reached application startup and produced `ATTN_CP0` through
`ATTN_CP3` workers, proving that the SGLang PCP launch path and restored plugin
distributed initialization execute. The container's `topk_v2` JIT first failed
because `cooperative_groups.h` is absent; setting `SGLANG_OPT_USE_TOPK_V2=false`
removed that unrelated environment failure.

The retry still fails during SGLang's warmup decode with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. Its synchronous stack is:

```text
deepseek_v2_moe_pcp_merge_forward
  -> DeepseekV2MoE._pcp_moe_merge_forward
  -> routed_expert_forward
  -> aiter.moe_forward
```

This occurs before an external request and after ordinary PCP process-group
startup, so PCP2 cannot yet be considered end-to-end supported on this SGLang
container. The server was stopped and GPUs released. The remaining fix is to
align the PCP MoE-merge dispatch with the SGLang NSA context-parallel layout
used by this version, then rerun the long-prefill request.
