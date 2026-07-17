# vLLM + Kimi-K2.6 Draft PR

## Scope

- vLLM + Kimi-K2.6

## ATOM Entry Reference

- `.github/benchmark/models_atomesh.yaml`

## Provided vLLM Disagg Command

```bash
IMAGE=vllm/vllm-openai-rocm:v0.23.0 MODEL_NAME=Kimi-K2.6-MXFP4 NODES=2 GPUS_PER_NODE=8 WIDE_EP_MODE=0 MORIIO_READ_MODE=0 RUN_AFTER_HEALTH=accuracy ROUTER_TYPE=vllm-router WAIT=1 SLURM_TIME_LIMIT=08:30:00 bash .buildkite/amd-disagg/run-slurm-disagg-test.sh &
```

## Details

- Model: `Kimi-K2.6-MXFP4`
- Image: `vllm/vllm-openai-rocm:v0.23.0`
- Router type: `vllm-router`
- `RUN_AFTER_HEALTH`: `accuracy`

## Note

Draft tracking PR; no runtime behavior change unless config integration is requested.
