# CI/PR Workflow Guide

## Pre-commit Checks (Required)

```bash
black .              # code formatting
ruff check .         # lint
python -m pytest tests/   # unit tests (no GPU needed)
```

CI enforces black + ruff in the `pre-checks` stage — failure skips all downstream jobs.

## CI Pipeline

```
PR/Push → pre-checks (black + ruff)
            ↓ pass
         build_atom_image (Docker image)
            ↓
         atom test (matrix accuracy tests)
            ↓
         check accuracy (compare against threshold)
```

- Triggers: push to main, PR to main, nightly schedule, manual dispatch
- `paths-ignore` in PR: `**/*.md`, `docs/**`, `LICENSE`, `.gitignore`
- Draft PRs do not trigger tests
- Concurrency: only the latest run per workflow+ref is kept

## CI Accuracy Test Matrix

| Model | Args | Threshold | PR trigger | Runner |
|-------|------|-----------|------------|--------|
| Llama-3-8B-Instruct | `--kv_cache_dtype fp8 --gpu-memory-utilization 0.3` | 0.73 | Yes | mi355-1 |
| Llama-3.3-70B MXFP4 | `--kv_cache_dtype fp8 --gpu-memory-utilization 0.3` | 0.88 | Yes | mi355-1 |
| DeepSeek-R1-0528 | `--kv_cache_dtype fp8 -tp 8` | 0.94 | Yes | 8gpu |
| DeepSeek-R1-0528 MTP | `--kv_cache_dtype fp8 -tp 8 --method mtp` | 0.94 | Yes | 8gpu |
| DeepSeek-R1-0528 MXFP4 | `--kv_cache_dtype fp8 -tp 8` | 0.93 | Yes | 8gpu |
| DeepSeek-R1-0528 MXFP4 MTP | `--kv_cache_dtype fp8 -tp 8 --method mtp` | 0.93 | Yes | 8gpu |
| gpt-oss-120b | `--kv_cache_dtype fp8 --gpu-memory-utilization 0.5` | 0.88 | Yes | mi355-1 |
| gpt-oss-120b (2 GPU) | `-tp 2 --enable-dp-attention --enable-expert-parallel` | 0.88 | No | mi355-4 |
| Qwen3-235B FP8 | `-tp 8 --enable-expert-parallel` + `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1` | 0.87 | Yes | 8gpu |
| Qwen3-235B MXFP4 | `-tp 8 --enable-expert-parallel` + `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1` | 0.87 | No | 8gpu |
| Qwen3-Next-80B-A3B | `-tp 8` | 0.65 | Yes | 8gpu |

- CI uses `num_fewshot=3`, `num_concurrent=16`
- Accuracy metric: GSM8K `flexible-extract`
- Test script: `.github/scripts/atom_test.sh accuracy`

> **Warning:** Local `lm_eval` typically uses `num_fewshot=5` and `num_concurrent=64`. CI and local results are NOT directly comparable due to different few-shot counts.

## PR Submission Checklist

1. [ ] `black .` and `ruff check .` pass
2. [ ] `python -m pytest tests/` passes
3. [ ] If model changes: confirm no `@support_torch_compile` decorated files were modified
4. [ ] If new model: CI matrix entry added
5. [ ] If accuracy-related changes: local `lm_eval` GSM8K validation done

## Related Files

- CI test workflow: `.github/workflows/atom-test.yaml`
- CI benchmark workflow: `.github/workflows/atom-benchmark.yaml`
- Pre-checks workflow: `.github/workflows/pre-checks.yaml`
- Test script: `.github/scripts/atom_test.sh`
