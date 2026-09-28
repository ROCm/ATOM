---
name: inferencex-sync
description: Compare the performance of the latest successful run from ATOM upstream benchmark(https://github.com/ROCm/ATOM/actions/workflows/atom-benchmark.yaml) and that of InferenceX(https://inferencex.semianalysis.com/api/v1/benchmarks?model=$MODEL) and report tput_per_gpu (Total token throughput / number of GPUs), report performance regression of each model on atom framework, create a PR if the performance of ATOM upstream benchamrk is better than that of InferenceX by updating docker image and atom serve arguments (python3 -m atom.entrypoints.openai_server)
memory: project
model: opus
---

# Compare ATOM upstream benchmark and InferneceX, and create a PR to InferenceX Instructions

Use this skill to compare the througput of ATOM upstream benchamrk and InferenceX and create a PR to InferenceX with the udpated docker image and atom serve arguments.

## Workflow

```text
- [ ] 1) Create a throughput table from InferenceX benchmark
- [ ] 2) Create a throughput table from ATOM upstream benchmark
- [ ] 3) Chek performance regression of ATOM upstream benchmark from InferenceX benchmark
- [ ] 4) Ask users what model needs a new PR
```

## 1) Create a throughput table from InferenceX benchmark

Check the InferenceX benchmark that is from these URLs and create a throughput performance table from ATOM upstream benchmark. 
- https://inferencex.semianalysis.com/api/v1/benchmarks?model=Kimi-K2.5
- https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4-Pro
- https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-R1-0528
- https://inferencex.semianalysis.com/api/v1/benchmarks?model=Qwen-3.5-397B-A17B
- https://inferencex.semianalysis.com/api/v1/benchmarks?model=GLM-5
- https://inferencex.semianalysis.com/api/v1/benchmarks?model=MiniMax-M2.5
- https://inferencex.semianalysis.com/api/v1/benchmarks?model=gpt-oss-120b

Rules:
- 1. Use curl to fetch data from URLs
- 2. Use only "hardware":"mi355x","framework":"atom","disagg":false,"is_multinode":false,"spec_method":"none"
- 3. If muliple data of "hardware":"mi355x","framework":"atom","precision":"fp4","disagg":false,"is_multinode":false exist for a same "model", use only the latest "date"
- 4. Create a throughput performance table of each "tput_per_gpu" of each "model","isl", "osl","conc","num_decode_gpus"(number of GPUs),"image","precision"

## 2) Create a throughput table from ATOM upstream benchmark

Check the ATOM upstream benchmark that is from the latest successful gh action runs at https://github.com/ROCm/ATOM/actions/workflows/atom-benchmark.yaml, which as all jobs passed. Append the throughput to the InferenceX performance table in Step 1) 

Rules:
- 1. Use main branch. Find the latest job of https://github.com/ROCm/ATOM/actions/workflows/atom-benchmark.yaml where all jobs passed    
- 2. "tput_per_gpu" (tput/GPU (tok/s)) is "Total token throughput" divided by TP(number of GPUs). TP(number of GPUs) is the value followed by -tp in https://github.com/ROCm/ATOM/blob/main/.github/benchmark/models.json
- 3. "conc" is "Max Concurrency". Compare same "conc" or "Max Concurrency" between InferenceX benchmark and Atom benchmark
- 4. Model name follows this mapping 

    ```python
    # inferenceX_model -> (atom_model, precision)
    MODEL_MAP = {
        "dsv4":        [("DeepSeek-V4-Pro",                   "fp4")],
        "dsr1":        [("DeepSeek-R1-0528",                  "fp8"),
                        ("DeepSeek-R1-0528-MXFP4",            "fp4")],
        "kimik2.5":    [("Kimi-K2.5-MXFP4",                   "fp4")],
        "qwen3.5":     [("Qwen3.5-397B-A17B-FP8",             "fp8"),
                        ("Qwen3.5-397B-A17B-MXFP4",           "fp4")],
        "glm5":        [("GLM-5-FP8",                         "fp8"),
                        ("GLM-5.1-MXFP4",                     "fp4")],
        "minimaxm2.5": [("MiniMax-M2.7",                      "fp8"),
                        ("MiniMax-M2.7-MXFP4",                "fp4")],
        "gptoss120b":  [("gpt-oss-120b",                      "fp4")],
    }
    ```

If the data is not available, stop and ask for any workaround

## 3) Chek performance regression of ATOM upstream benchmark from InferenceX benchmark

Update the performance table to show regression in % of ATOM upstream benchmark from InferenceX benchmark

## 4) Chek performance regression of ATOM upstream benchmark from InferenceX benchmark

Based on Step 3) results, ask user what models need a PR to https://github.com/SemiAnalysisAI/InferenceX/pulls and create a PR to each model.

Rules:
- 1. Find the docker image from the step 2) ATOM upstream benchmark gh action runs. for eample, "Docker:
  rocm/atom-dev:nightly_202605141645" from https://github.com/ROCm/ATOM/actions/runs/25875375446/job/76041129934#step:6:26 
- 2. Create a PR and check https://github.com/ROCm/ATOM/blob/main/.github/benchmark/models.json options

PR example: 
- 1. https://github.com/SemiAnalysisAI/InferenceX/pull/1311/changes
- 2. https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/dsv4_fp4_mi355x_atom.sh
- 3. https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/configs/amd-master.yaml#L1646
- 4. https://github.com/SemiAnalysisAI/InferenceX/blob/main/perf-changelog.yaml#L1824-L1833
