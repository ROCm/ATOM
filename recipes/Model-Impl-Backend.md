# Model Impl Backend of vLLM and SGLang
ATOM can work as model implementation backend of popular framework, like vLLM and SGLang. The users can launch vLLM and SGLang server like before and specify an extra argument to enable ATOM, where the optimized implementation of the required target model will be provided to vLLM and SGLang to execute. When ATOM working under this mode, both framework-level features from vLLM/SGLang and latest model-level fusion kernels from ATOM/AITER can be combined together to achieve the competitive performance.

## Preparing environment for vLLM with ATOM model backend
Pull the latest docker from vLLM official nightly docker for ROCm https://hub.docker.com/r/rocm/vllm-dev/tags
```bash
docker pull rocm/vllm-dev:nightly
```
All the next operations will be executed inside the container.
Then the specific vLLM should be used because the PR to introduce the ATOM into vLLM has not been merged yet, so you need to:
```bash
pip uninstall -y vllm
git clone https://github.com/zejunchen-zejun/vllm.git
cd vllm
git checkout origin/zejun/model_impl
export PYTORCH_ROCM_ARCH="gfx950"
python3 setup.py develop 2>&1 | tee build.log
```
Then the ATOM should be installed
```bash
git clone https://github.com/zejunchen-zejun/ATOM.git
cd ATOM
git checkout origin/zejun/plugin_for_atom_1223
pip install -e . 2>&1 | tee build.log
```
For aiter, there is no specific requirement.

### Launching server of vLLM with ATOM model backend
You just need to deploy 2 code changes to your previous server launch command. The one is using CUSTOM vLLM attention backend, the other is a new argument of specifying the ATOM model impl backend. Here is the an example. From the example, the specific fusion kernels are used, which is not easy to use in vLLM side as vLLM has some heuristic to stipulate the boundary of ops and layers.
```bash
export VLLM_ATTENTION_BACKEND=CUSTOM

export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor

rm -rf /root/.cache/

model_path=<your model file path>

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --load-format fastsafetensors \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 18432 \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --model-impl atom \
    2>&1 | tee log.serve.log &
```
### Launching client for validating the accuracy
```bash
addr=localhost
port=8000
url=http://${addr}:${port}/v1/completions
model=<your model file path>
task=gsm8k
lm_eval --model local-completions \
        --model_args model=${model},base_url=${url},num_concurrent=65,max_retries=1,tokenized_requests=False \
        --tasks ${task} \
        --num_fewshot 3 \
        2>&1 | tee log.lmeval.log
```


## Preparing environment for SGLang with ATOM model backend
TODO

