# Model Impl Backend of SGLang
ATOM can work as model implementation backend of popular framework, like SGLang. The users can launch the server like before and specify an extra argument to enable the ATOM model backend, where the optimized implementation of the required target model will be provided to SGLang to execute. When ATOM working under this mode, both framework-level features from SGLang and latest model-level fusion kernels from ATOM/AITER can be combined together to achieve the competitive performance.

- Here is a detailed design slide for this feature: https://amdcloud-my.sharepoint.com/:p:/g/personal/zejchen_amd_com/IQCFdvmEeLTWT7ysApmZv_hVAfw2nTo8iesJZGblHS0evqQ?e=hjnIDM
- Here is the RFC to introduce the ATOM as model impl backend into SGLang: TODO

## Preparing environment for SGLang with ATOM model backend
Here is the PR to introduce ATOM into SGLang: https://github.com/sgl-project/sglang/pull/16944, when this PR would be merged, the official SGLang can be used, but for now you need to use develop vllm branch
Pull the latest docker from SGLang official nightly docker for ROCm from https://hub.docker.com/r/rocm/sgl-dev/tags
```bash
docker pull rocm/sgl-dev:v0.5.8-rocm720-mi35x-20260130-preview
```
Launch the container as usual, then all the next operations will be executed inside the container
Then the specific SGLang should be used because the PR to introduce the ATOM into SGLang has not been merged yet, so you need to:
```bash
git clone https://github.com/zejunchen-zejun/sglang.git
git checkout remotes/origin/zejun/model_impl
pip uninstall sglang -y
pip uninstall sgl-kernel -y
cd sgl-kernel
python3 setup_rocm.py install
export PYTHONPATH=<sglang python folder path>
```
Then the ATOM should be installed
```bash
git clone https://github.com/zejunchen-zejun/ATOM.git
cd ATOM
git checkout origin/zejun/plugin_for_atom_1223
pip install -e . 2>&1 | tee build.log
```
For AITER, there is no specific requirements, however, if you find any latest fusion kernels are missing, you may need to upgrade the AITER

### Launching server of SGLang with ATOM model backend
You just need to deploy single code change, as add --model-impl atom to your SGLang server command. Here is an example:
```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

# quick allreduce
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8

python3 -m sglang.launch_server \
    --model-path $model_path \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --expert-parallel-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
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

### Known Limitations
There are some known limitations for now:
- Only Qwen-Dense and Qwen-MoE family models are supported
- Only TP and EP are supported
