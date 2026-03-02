# vLLM out-of-tree ATOM Plugin Backend
ATOM can work as the OOT plugin backend of vLLM. The OOT register mechanism is quite mature and most of accelerators have leveraged this design to register their devices into vLLM without any code changes in upper framework. ATOM follows this design and provide the layer/op and model implementations to vLLM. The frontend users can launch vLLM server like before and there is no need to specify any arguments. Meanwhile the ATOM platform can leverage most of the vLLM features and focus more on model- and kernel-level optimizations. For the overall design, here is a RFC to enable ATOM work as the OOT plugin platform of vLLM: https://github.com/ROCm/ATOM/issues/201

## Preparing environment for vLLM with ATOM model backend
Pull the vLLM official docker for ROCm. If you are using the vLLM nightly docker, there could be incompatible error because vLLM is changing its code and may break the class/module import in ATOM
```bash
docker pull rocm/vllm-dev:nightly_main_20260118
```

Then the ATOM should be installed. When the following PR merged, you can use ATOM main branch
```bash
git clone https://github.com/zejunchen-zejun/ATOM.git
cd ATOM
git checkout origin/zejun/plugin_for_atom_1223
pip install -e . 2>&1 | tee build.log
```
For AITER, you can build AITER with the latest main branch. The AITER version requirement for ATOM OOT is consistent with the AITER version requirement of the ATOM server mode.
Additionally, you may need to install some dependencies by:
```bash
pip install --upgrade triton
pip install transformers==5.0.0
pip install git+https://github.com/foundation-model-stack/fastsafetensors.git
```

### Launching server of vLLM with ATOM OOT Plugin Platform
There is no code change to vLLM side, so you can launch the vLLM server like before without any specific argument
```bash
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
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```

If you want to disable the ATOM OOT plugin platform and model register, you can use below env flags. The default value is 0
```bash
export ATOM_DISABLE_VLLM_PLUGIN=1
```
If you only want to disable the ATOM Attention Backend, you can use below env flags.  The default value is 0
```bash
export ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1
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

### Results for accuracy validation
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.9037|±  |0.0081|
|     |       |strict-match    |     3|exact_match|↑  |0.8832|±  |0.0088|

### Known Limitations
There are some known limitations for now:
- Only Qwen-Dense and Qwen-MoE family models are supported
- Only TP and EP are supported
