---
name: accuracy-debug
description: ATOM runs daily accuracy tests on all models with the latest ATOM and aiter. Because aiter and ATOM change every day, regressions can appear. When that happens, you will be given one known-good pair of ATOM and aiter commits and one known-bad pair; follow the steps below to find the commit that broke the model—it may be in ATOM or in aiter.
---

# User-editable configuration

right commit:
- ATOM: ee89271a
- aiter: d635dbf6

error commit:
- ATOM: ae873704
- aiter: 809b1db3

images: (pick the image that matches the good commit)
- rocm/atom-dev:vllm-v0.19.0-nightly_20260412

framework:
- vLLM-ATOM

model and path:
- Qwen3-235B-A22B-Instruct-2507-MXFP4
- /shared/data/amd_int/models/Qwen3-235B-A22B-Instruct-2507-MXFP4

## Stop condition

Run real accuracy tests until you identify the commit that causes accuracy to fall **clearly below** the model’s **`accuracy_threshold`** in `.github/benchmark/models_accuracy.json`.

### Pass/fail rules (aligned with CI)

- Compare **`accuracy_threshold`** to `exact_match,flexible-extract` on **gsm8k**, same as in workflows.
- **Scores clearly below the threshold count as failed accuracy.**
- **A valid bisect** requires that, under the **same image / same vLLM and plugin contract**, one side reaches at least **90% of `accuracy_threshold`** and the other side does not; otherwise **first reproduce a passing baseline** (image tag, ATOM/aiter, env vars matching CI), then bisect ATOM or aiter.

## Workflow

Copy this checklist and update progress:

```text
- [ ] 1) Follow workflows under `.github` to pull the image locally and start a container with podman or docker; mount the model path above—weights live on that path
- [ ] 2) Inside the container, clone ATOM and aiter at the desired commits and install; remove the existing ATOM and aiter install in the container before reinstalling
- [ ] 3) From the benchmark JSON, build the server and client scripts for this model and run the accuracy test
- [ ] 4) First verify the good commit pair and the bad commit pair
- [ ] 5) Then bisect or otherwise swap ATOM or aiter commits
- [ ] 6) Start server and client for each accuracy run, find the bad commit and analyze the failure; before each run, use a script to clean leftover processes
- [ ] 7) Finally write a test report for the whole run and remove the debug container and any temporary clone directories
```
