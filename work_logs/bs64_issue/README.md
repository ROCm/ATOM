# DeepSeek-R1-FP4 v2 MTP3 bs=64 Accuracy Issue

## Summary

Nightly failure:

- Job: `SGLANG Model Accuracy (DeepSeek-R1-FP4 TP8 MTP3)`
- Model: `amd/DeepSeek-R1-0528-MXFP4-v2`
- Draft: `SGLang/DeepSeek-R1-NextN`
- Backend: ATOM SGLang
- Server args include:
  - `--tensor-parallel-size 8`
  - `--attention-backend aiter`
  - `--kv-cache-dtype fp8_e4m3`
  - `--speculative-algorithm NEXTN`
  - `--speculative-num-steps 3`
  - `--speculative-num-draft-tokens 4`
  - `--max-running-requests 256`
  - full `--cuda-graph-bs ... 64 ...`
- Eval: GSM8K, 3-shot, default `LM_EVAL_NUM_CONCURRENT=65`
- Failure score: `exact_match,flexible-extract = 0.026535253980288095`

Local reproduction:

- Container: `zhiwei_sgl_atom_0615`
- Image from terminal history: `docker.io/rocm/atom-dev:sglang-v0.5.12-nightly_20260615`
- Local full GSM8K with the same v2 MTP3 server reproduced the failure:
  - `limit=None`
  - `num_fewshot=3`
  - `num_concurrent=65`
  - `exact_match,flexible-extract = 0.0220`

Main conclusion so far:

- Not a CI-only issue.
- Not a plain target model issue: `DeepSeek-R1-FP4 TP8` without MTP passes on the same remote runner/model path.
- Not a generic MTP3 issue: `DeepSeek-R1-FP4-MTP-MoEFP4 TP8 MTP3` passes with the same draft model and MTP3 args.
- The failure is tied to `DeepSeek-R1-0528-MXFP4-v2 + SGLang/DeepSeek-R1-NextN` under high-batch SGLang spec-v2 target verify.
- The sharp cliff is around real running batch size 64.

## Source Logs

Files in `/shared/amdgpu/home/qichu_qle/zhiwei/dsv4`:

- `raw_log`: failing remote `DeepSeek-R1-FP4 TP8 MTP3`
  - score `0.026535253980288095`
  - result file `/tmp/atom_sglang_accuracy_results/20260615221347_DeepSeek-R1-FP4_TP8_MTP3.json`
- `raw_log_1`: passing remote `DeepSeek-R1-FP4 TP8`
  - score `0.9363153904473086`
- `raw_log_2`: passing remote `DeepSeek-R1-FP4-MTP-MoEFP4 TP8 MTP3`
  - score `0.9385898407884761`

Local useful logs:

- `log.dsr1_fp4_tp8_mtp3.serve.log`: latest server log in repo root. It is overwritten by each launch.
- `/tmp/v2_mtp3_*` inside container: lm_eval output directories from local experiments.

## Reproduction Scripts

Local launch script:

- `launch_dsr1_fp4_tp8_mtp3.sh`

Important local eval script:

- `val_gsm8k.sh`
- Current intended behavior:
  - model path: `/workspace/shared/data/amd_int/models/deepseek-ai/DeepSeek-R1-0528-MXFP4-v2`
  - `--num_fewshot 3`
  - optional limit: `bash val_gsm8k.sh 128`

Baseline launch:

```bash
podman exec -it zhiwei_sgl_atom_0615 bash
cd /home/qichu_qle/zhiwei/dsv4/atom
bash launch_dsr1_fp4_tp8_mtp3.sh
```

Eval examples:

```bash
# Full CI-like local run
bash val_gsm8k.sh

# Limit run with explicit concurrency
lm_eval --model local-completions \
  --model_args model=/workspace/shared/data/amd_int/models/deepseek-ai/DeepSeek-R1-0528-MXFP4-v2,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=1,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --num_fewshot 3 \
  --limit 128 \
  --output_path /tmp/v2_mtp3_c64_l128
```

When starting server, do not blind-wait. Watch the tail:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path("log.dsr1_fp4_tp8_mtp3.serve.log")
print("\n".join(p.read_text(errors="replace").splitlines()[-80:]))
PY
curl -fsS http://127.0.0.1:8000/v1/models
```

## Experiment Matrix

All rows below are local unless marked remote. Values are approximate from captured logs.

| Case | Limit | Concurrency | Extra | Result |
| --- | ---: | ---: | --- | ---: |
| v2 MTP3 | full | 65 | CI-like | `0.0220` |
| v2 MTP3 | 128 | 16 | normal | `0.9453` |
| v2 MTP3 | 128 | 32 | normal | `0.9297` |
| v2 MTP3 | 128 | 48 | normal | `0.9375` |
| v2 MTP3 | 128 | 56 | normal | `0.9453` |
| v2 MTP3 | 128 | 60 | normal | `0.9609` |
| v2 MTP3 | 128 | 62 | normal | `0.9297` |
| v2 MTP3 | 128 | 63 | normal | `0.9219` |
| v2 MTP3 | 128 | 64 | normal | `0.1094` |
| v2 MTP3 | 128 | 65 | normal | `0.1094` |
| v2 MTP3 | 128 | 64 | `--disable-cuda-graph` | `0.0469` |
| v2 MTP3 | 128 | 64 | `SGLANG_AITER_MLA_PERSIST=0` | `0.0` |
| v2 MTP1 | 128 | 64 | `steps=1,draft_tokens=2` | `0.6641` |
| v2 MTP3 | 128 | 64 | SGLang force reject experiment | `0.1094` |
| v2 MTP3 | 128 | 64 | `--disable-cuda-graph`, target_verify MLA chunked by req | `0.1094` |
| v2 MTP3 | 128 | 64 | `--disable-cuda-graph`, `ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0` | `0.1406` |
| v2 MTP3 | 128 | 64 | `--disable-cuda-graph --disable-overlap-schedule` | `0.1328` |
| v2 MTP3 | 128 | 64 | `--disable-cuda-graph`, `SGLANG_ENABLE_SPEC_V2=0` | `0.1406` |
| v2 MTP3 | 128 | 64 | `--disable-cuda-graph`, `SGLANG_AITER_FP8_PREFILL_ATTN=1` | `0.2422` |
| remote v2 TP8 | full | 65 | no MTP | `0.9363153904473086` |
| remote MTP-MoEFP4 MTP3 | full | 65 | same MTP3 args | `0.9385898407884761` |
| remote v2 MTP3 | full | 65 | failing nightly | `0.026535253980288095` |

Important nuance:

- `limit=32, concurrent=64` can pass because only 32 requests exist, so it does not necessarily create a true running batch of 64.
- With `limit=128, concurrent=64`, the server log shows real `Decode batch, #running-req: 64`.

## Observed Bad Output Shape

`--log_samples` on the bad c64 run shows outputs are genuinely wrong, not just evaluator extraction failures.

Examples:

- Repeats phrases like "hens lay eggs" or "download speed is 10MB/s".
- Often no final `####` answer.
- Questions and reasoning drift into unrelated content.

So the failure is model generation/state corruption, not GSM8K parsing.

## What Was Ruled Out

### CI-only issue

Ruled out. Local full run reproduced `0.0220`.

### Main v2 model without MTP

Ruled out. Remote `DeepSeek-R1-FP4 TP8` no MTP passes `0.9363`.

### SGLang/NextN draft model globally broken

Ruled out. Remote `DeepSeek-R1-FP4-MTP-MoEFP4 TP8 MTP3` passes `0.9386` with the same `SGLang/DeepSeek-R1-NextN` draft and same MTP3 args.

### CUDA graph replay alone

Ruled out. `--disable-cuda-graph` still fails badly at c64.

### MLA persistent kernel alone

Ruled out as sole cause. `SGLANG_AITER_MLA_PERSIST=0` still fails.

### CI-side model path mismatch

Likely ruled out by `raw_log_1`: same remote `/mnt/raid0/pretrained_model/amd/DeepSeek-R1-0528-MXFP4-v2` passes without MTP.

## Current Hypotheses

### Most likely

Bug is in SGLang spec-v2 target verify integration with ATOM DeepSeek MLA for `DeepSeek-R1-FP4-v2` under high running batch size around 64.

The relevant path:

```text
EAGLEWorkerV2.verify()
  -> EagleVerifyInputV2Mixin.prepare_for_v2_verify()
  -> target_worker.forward_batch_generation(is_verify=True)
  -> ATOM DeepSeek MLA SGLang plugin path
  -> EagleVerifyInputV2Mixin.sample()
  -> verify_tree_greedy_func / accept_index / accept_lens
  -> draft_extend_for_decode()
```

### Why v2 differs from MTP-MoEFP4

The two artifacts have different quantization metadata:

- `DeepSeek-R1-0528-MXFP4-v2`
  - 78 safetensors
  - config excludes most of `model.layers.61.*`
  - no `layer_quant_config` forcing `*self_attn*` to FP8
- `DeepSeek-R1-0528-MXFP4-MTP-MoEFP4`
  - 76 safetensors
  - only excludes a few layer-61/shared-head items
  - includes `layer_quant_config` for `*self_attn*` FP8

This makes v2 more likely to use MXFP4 self-attention / absorbed MLA projection paths during target verify, while MTP-MoEFP4 uses a safer FP8 self-attn path.

### Unproven details

Need to prove whether the bug is:

1. Incorrect target verify metadata (`qo_indptr`, `kv_indptr`, `kv_indices`, `out_cache_loc`, `retrieve_index`, etc.) at bs=64.
2. Correct metadata but wrong output from a specific kernel called by target verify.
3. Correct target verify output but wrong accept/cache commit after verify.

## Failed / Inconclusive Experiments

### `--attention-backend triton`

Server failed during graph capture:

```text
Unexpected absorbed weight shape for bmm fallback: (16, 128, 256) with in_dim=128, out_dim=512
```

So triton attention cannot currently be used as a clean A/B.

### Force target verify non-absorbed path

Patch attempted:

```python
use_non_absorbed = (
    forward_batch.forward_mode.is_extend_without_speculative()
    or forward_batch.forward_mode.is_target_verify()
)
```

With graph enabled, launch hit HIP illegal memory access during capture.
With graph disabled, server appeared to become unavailable during eval. Needs a cleaner retry if pursuing this path.

### ATOM native rejection sampler patch

Not useful for SGLang serving. SGLang spec-v2 uses its own `EagleVerifyInputV2Mixin.sample()` / `verify_tree_greedy_func`, not `atom/model_ops/rejection_sampler.py`.

### SGLang force-reject patch

Patch attempted in container-local `/app/sglang/python/sglang/srt/speculative/eagle_info_v2.py`:

```python
if os.getenv("SGLANG_DEBUG_FORCE_REJECT_ALL", "0") == "1":
    predict.copy_(target_predict.reshape(-1).to(torch.int32))
    accept_index.fill_(-1)
    accept_index[:, 0] = (
        torch.arange(bs, device=device, dtype=torch.int32)
        * self.draft_token_num
    )
    num_correct_drafts.zero_()
```

With `SGLANG_DEBUG_FORCE_REJECT_ALL=1`, `concurrent=64, limit=128` still failed around `0.1094`.

Important interpretation:

- This is stronger than the earlier ATOM native rejection-sampler patch because it sits in the actual SGLang spec-v2 verify path.
- It suggests that even when only the first target-verify token is accepted, the generation is still corrupted at high batch.
- This points either to target-verify logits already being wrong or to cache/sequence state being corrupted by target-verify preparation/commit even under forced single-token accept.

### MTP1 high-batch check

Launched v2 with:

```text
--speculative-num-steps 1
--speculative-num-draft-tokens 2
```

Result for `concurrent=64, limit=128`:

```text
exact_match,flexible-extract = 0.6641
```

This is much better than MTP3 c64, but still far below the non-MTP baseline and below c60 MTP3. Therefore the issue is not only the `draft_num=4 / M=256` target-verify shape; even `draft_num=2 / M=128` is affected under high concurrency.

### Triton attention A/B

Attempting `--attention-backend triton` failed during startup:

```text
Unexpected absorbed weight shape for bmm fallback: (16, 128, 256) with in_dim=128, out_dim=512
```

This does not give a valid accuracy A/B. It does show the fallback path for absorbed MLA is currently not compatible with this quantized weight layout.

### Disable-overlap experiment

Attempted to launch with `--disable-overlap-schedule`, but the test run that followed was invalid:

- Server was still loading / not ready.
- `lm_eval` failed with `ConnectionRefusedError`.

Do not count this experiment as evidence. It should be retried only after `/v1/models` is reachable.

### Follow-up on 2026-06-17

Additional local A/B runs after fixing the no-overlap startup-only `accept_length` field mismatch:

- `--disable-cuda-graph --disable-overlap-schedule` still failed (`0.1328`), so overlap scheduling is not the root cause.
- `SGLANG_ENABLE_SPEC_V2=0` with no graph still failed (`0.1406`), so the issue is not spec-v2-only.
- `ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0` with no graph still failed (`0.1406`), so QKNorm quant fusion is not the root cause.
- Chunking target_verify `mla_decode_fwd` by request with no graph still failed (`0.1094`), so the failure is not only the total `M=bs*draft_num=256` MLA decode launch shape.
- Replacing target_verify MLA decode with the causal prefill MLA path crashed in `mla_prefill_fwd` during warmup, so that path is not currently a drop-in fix.
- `SGLANG_AITER_FP8_PREFILL_ATTN=1` initially crashed because ATOM's `mla_reduce_v1` call was missing the newer `num_kv_splits` argument. After adding `prefill_ps_num_kv_splits`, no-graph serving ran and improved `c64/l128` to `0.2422`, but still did not recover normal accuracy. This means FP8 prefill affects the failure, but is not a full fix by itself.

Current best root-cause direction:

- TP=4 reproduces the same batch cliff on GPUs 0-3:
  - `concurrent=63, limit=128`: `0.9219`
  - `concurrent=64, limit=128`: `0.0156`
  - This rules out TP8-specific partitioning as the root cause.
- Rank-0 target-verify attention debug for c63/c64 showed `qo_indptr`, `kv_indptr`, `out_cache_loc`, `kv_last_page_len`, `num_kv_splits`, and MLA output ranges are structurally reasonable across the cliff. No obvious metadata blow-up was found in `ATOMAttnBackendForSgl._call_mla_decode_fwd()`.
- The first clear divergence happens before target verify:
  - `EagleVerifyInput.sample()` sees different `candidates` at c64 vs c63 before verification.
  - `EagleDraftWorker.draft_forward()` shows the c64 draft tokens are already different at `step=0 after_select`.
  - This moves the active suspect earlier than target-verify MLA: target prefill `next_token_ids` / initial draft-prefill `topk_index` / captured hidden state feeding the draft.
- Remaining strongest clue is still artifact-specific self-attention quantization: v2 uses MXFP4 self-attention while `DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` forces `*self_attn*` to FP8 and passes.
- Next useful step: instrument target prefill (`batch_output.next_token_ids`, capture hidden state summary, and initial draft-prefill topk) without destabilizing the runtime, then compare c63/c64 from the first prefill micro-batch.

Code note:

- A small compatibility fix was added in `ATOMAttnBackendForSgl`: draft-extend metadata now reads current SGLang's `num_accept_tokens` as a fallback when older `accept_length` is absent. This fixes the no-overlap startup crash:

```text
AttributeError: 'EagleDraftExtendInput' object has no attribute 'accept_length'
```

## Next Debug Steps

1. Run `--disable-overlap-schedule` with v2 MTP3 and `concurrent=64`.
   - If it passes: root cause is spec-v2/overlap integration.
   - If it fails: root cause is lower-level target verify / model path.
   - Previous attempt was invalid because eval started before server was ready.

2. Continue c63/c64 instrumentation, but move the dump point earlier:
   - target prefill `batch_output.next_token_ids`
   - target prefill hidden state summary used by MTP
   - `_draft_extend_for_prefill()` initial `topk_index/topk_p`
   - `draft_forward()` `step=0 after_select`

3. Compare v2 c64 against passing `DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` c64 at the same early dump points.

4. If target prefill diverges only in MTP mode, test target prefill with hidden capture disabled or reduced (`FULL` vs `LAST`/`NULL`) to isolate capture side effects.

## Cleanup Notes

Before continuing, verify:

```bash
podman exec zhiwei_sgl_atom_0615 bash -lc 'pgrep -af "sglang.launch_server|sglang::" || true'
podman exec zhiwei_sgl_atom_0615 bash -lc 'rocm-smi --showpids'
```

The temporary CI concurrency workaround was reverted. Do not reintroduce it as the final fix.

