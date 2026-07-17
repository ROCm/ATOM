# vLLM + MiniMax-M3 Draft PR

- Scope: vLLM + MiniMax-M3
- ATOM entry reference: `.github/benchmark/models_atomesh.yaml`
- Model: `MiniMax-M3-MXFP8`
- Image: `vllm/vllm-openai-rocm:v0.23.0`
- Router type: `proxy`
- RUN_AFTER_HEALTH: `accuracy`

## Provided vLLM disagg command

```bash
IMAGE=vllm/vllm-openai-rocm:v0.23.0 MODEL_NAME=MiniMax-M3-MXFP8 NODES=2 GPUS_PER_NODE=8 WIDE_EP_MODE=0 MORIIO_READ_MODE=0 RUN_AFTER_HEALTH=accuracy ROUTER_TYPE=proxy WAIT=1 SLURM_TIME_LIMIT=08:30:00 bash .buildkite/amd-disagg/run-slurm-disagg-test.sh &
```

## Note

Draft tracking PR; no runtime behavior change unless config integration is requested.
