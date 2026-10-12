# MiniMax M3 C40 PD accuracy test

This manual CI case evaluates MiniMax-M3-MXFP4 on the C40 PD service configuration:

`minimax-m3-fp4-1p1d-tp4-eagle3-lmcache-gsm8k-only-c40`

The configuration is in `.github/benchmark/models_atomesh.yaml`, under
`MiniMax-M3-MXFP4.suites.accuracy`. It inherits the C40 AgentX service
configuration from reference commit `c02697c`.

The service also inherits the scoped pre-scatter AiTer M3 tuning override
used to work around the reference run's 32768-token startup fault. See the
[tuning provenance](../.github/benchmark/aiter/README.md) and
[failure investigation](../docs/analysis/minimax_m3_pd_c40_startup_failure_20261012_zh.md).
The workaround still needs validation on MI355X.

| Setting | Value |
| --- | --- |
| Placement | One node, 1 prefill worker and 1 decode worker |
| Prefill / Decode | TP4 on GPU 0–3 / TP4 on GPU 4–7 |
| Request concurrency / lm-eval batch size | 40 / 40 |
| Server max sequences | 80 on both roles |
| Speculation | EAGLE3 K=3, Inferact/MiniMax-M3-EAGLE3 |
| Fixed acceptance length | Disabled; use actual model verification |
| KV | fp8, block size 128, prefix caching enabled |
| GPU memory utilization / batched tokens | 0.95 / 32768 |
| CUDA graph | FULL on both roles |
| Indexer / mono | CP=1, index cache fp8 / mono disabled |
| Transfer | Mooncake RDMA |
| Prefill offload | LMCache DRAM, 256 GiB per rank, chunk 256, ATOM_SLRU |
| Thinking | enabled |
| Evaluation | Full GSM8K test split, 5-shot chat, few-shot as multiple turns |
| Maximum generation length | 16384 tokens |
| Pass criterion | exact_match,flexible-extract >= 0.93 |

The 0.93 gate comes from the existing M3 TP8 and M3 EAGLE3 TP8 entries in
`.github/benchmark/models_accuracy.json`. It is the initial acceptance target,
not a measured TP4 PD baseline.

`eval_only: true` starts fresh services and runs only GSM8K. The inherited
native context capacity is retained, but GSM8K uses its own prompt lengths;
this does not validate 1M-context accuracy or guarantee LMCache eviction and
reload coverage. The AgentX performance workload is not run.

## Run through GitHub Actions

The branch containing these changes must be pushed to the repository on which
you dispatch the workflow. Select **Actions → Atomesh Benchmark → Run workflow**,
then set:

| Input | Value |
| --- | --- |
| Branch | The branch containing this case, currently mengqing/test_m3_pd |
| suite | accuracy |
| case_names | minimax-m3-fp4-1p1d-tp4-eagle3-lmcache-gsm8k-only-c40 |
| run_all_models | false |
| run_model_benchmark | true (this switch also enables accuracy jobs) |
| atom_source | mounted |
| eval_concurrency | 40 |
| publish_dashboard | false |

Leave `model_names` and `benchmark_concurrency` empty. Select the appropriate
Slurm submit runner and node for your cluster. The example below uses TW MI355X
with automatic node selection. It needs one 8-GPU node with RDMA and enough
host memory for four 256 GiB LMCache allocations plus runtime overhead.

The existing runner setup must have the target and draft model directories:
`<ATOMESH_MODEL_ROOT>/amd/MiniMax-M3-MXFP4` and
`<ATOMESH_MODEL_ROOT>/Inferact/MiniMax-M3-EAGLE3`. On the default TW runner the
model root is `/mnt/models`.

Equivalent command using an authenticated GitHub CLI:

```bash
gh workflow run atomesh-benchmark.yaml \
  --repo ROCm/ATOM \
  --ref mengqing/test_m3_pd \
  -f suite=accuracy \
  -f case_names=minimax-m3-fp4-1p1d-tp4-eagle3-lmcache-gsm8k-only-c40 \
  -f run_all_models=false \
  -f run_model_benchmark=true \
  -f atom_source=mounted \
  -f atomesh_slurm_submit_runner=atomesh-cicd \
  -f atomesh_single_node=auto \
  -f eval_concurrency=40 \
  -f publish_dashboard=false
```

For a branch pushed to a fork, change `--repo` to that repository; its Actions
runner and model storage must be configured. For reproducible comparisons,
also set `atomesh_image` to the same immutable image tag or digest.

Find the submitted run and follow it:

```bash
gh run list --repo ROCm/ATOM \
  --workflow atomesh-benchmark.yaml --branch mengqing/test_m3_pd --limit 5
gh run watch RUN_ID --repo ROCm/ATOM --exit-status
```

## Results

The Slurm step must succeed and print:

```text
[eval] concurrent=40 exact_match,flexible-extract = ...
[eval] GSM8K threshold passed: ... >= 0.9300
```

A lower score, missing result, invalid score, or failing lm-eval command makes
the job fail. Other GSM8K cases that do not configure a threshold continue to
report valid scores without a numeric gate.

Download the `atomesh-model-benchmark-*` artifact for this case. It contains
`slurm_job-*/eval_results/*_gsm8k_1p1d_eagle3_c40/` with the lm-eval summary and
sample outputs, and `slurm_job-*/logs/eval/` with the service logs. The Slurm
log contains the client score and threshold result.

## CPU checks

From the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_atomesh_pd_matrix.py' -v
python3 -m unittest discover -s tests -p 'test_atomesh_gsm8k_eval.py' -v
python3 -m unittest discover -s tests -p 'test_atomesh_v4_workflow.py' -v
```

These checks verify configuration expansion, evaluation phase isolation,
client arguments, and score gating without loading model weights.
