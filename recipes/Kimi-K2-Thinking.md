# Kimi-K2 Usage Guide

[Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) is a powerful agent based on Kimi K2, developed by Moonshot AI. 
While it's already a native INT4 quantization model, this guide will focus on deploying the MXFP4 version of this model (refer to https://huggingface.co/amd/Kimi-K2-Thinking-MXFP4) on AMD GPUs with ATOM.

## Preparing environment
Pull the nightly docker from https://hub.docker.com/r/rocm/atom/.
All the operations in the next will be executed inside the container.

## Launching server
ATOM supports running the model with different parallelism, e.g., tensor parallel, expert parallel, data parallel.
Here we consider the parallelism of TP4 as an example. 

### Serving on 4xMI355 GPUs

```bash
#!/bin/bash
export HIP_VISIBLE_DEVICES=0,1,2,3

python -m atom.entrypoints.openai_server --model amd/Kimi-K2-Thinking-MXFP4 --trust-remote-code -tp 4 --kv_cache_dtype fp8
```

**Troubleshooting**:
- torch.compile reports an error with triton==3.5.1, which is a known issue of torch as detailed in [issue#161618](https://github.com/pytorch/pytorch/issues/161618). It has already been fixed by [commit 05eeb29](https://github.com/pytorch/pytorch/commit/05eeb29976fffd8331ea6c7d30960fdf48626294). A quick solution is to reinstall triton==3.4.0. 



## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=amd/Kimi-K2-Thinking-MXFP4 --backend=vllm --base-url=http://localhost:$PORT \
    --trust-remote-code --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 0.8 \
    --num-prompts=$(( $CONC * 10 )) \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir=${result_dir} --result-filename=$RESULT_FILENAME.json \
    --percentile-metrics="ttft,tpot,itl,e2el"
```
The performance number on 4 ranks is provided as a reference, with the following environment:
- docker image: rocm/atom:rocm_7.2_preview_gfx950_latest.
- ATOM: guanbao/kimi_k2_fp4 branch, commit 29e40d7.
- AITER: main branch, commit 3b2346f.


| ISL | OSL | Concurrency | Num Prompts | Mean TTFT (ms) | Mean TPOT (ms) | Output Throughput | Total Throughput |
| ------------ | ------------- | --------------- | ----------- | ------------ | ------------ | ------------------- | ------------------- |
| 1024         | 1024          | 4               | 40          | 117.68       | 11.65        | 329.18              | 661.59              |
| 1024         | 1024          | 8               | 80          | 134.22       | 13.08        | 592                 | 1179.62             |
| 1024         | 1024          | 16              | 160         | 173.87       | 15.46        | 1000.21             | 2011.07             |
| 1024         | 1024          | 32              | 320         | 273.36       | 18.81        | 1632.74             | 3260.31             |
| 1024         | 1024          | 64              | 640         | 319.94       | 25.13        | 2444.08             | 4889.26             |
| 1024         | 1024          | 128             | 1280        | 500.68       | 36.77        | 3349.54             | 6706.44             |


Here are the steps to reinstall ATOM/AITER in the docker, if you are trying to verify with other specific commits:
```bash
# uninstall existing ATOM/AITER
pip uninstall -y atom amd-aiter

cd PATH_TO_ATOM
# normally ATOM is already installed in develop mode
# you may just do checkout without reinstall
git checkout specific_branch_or_commit
pip install -e .

cd PATH_TO_AITER
rm -rf aiter/jit/build aiter/jit/*.so
git checkout specific_branch_or_commit
git submodule sync && git submodule update --init --recursive
python setup.py develop
```

### Accuracy test
We verified the lm_eval accuracy on gsm8k dataset with command:
```bash
lm_eval \
--model local-completions \
--model_args model=amd/Kimi-K2-Thinking-MXFP4,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True \
--tasks gsm8k \
--num_fewshot 3
```

Here is the reference value when deploying on 4 ranks:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.9363|±  |0.0067|
|     |       |strict-match    |     3|exact_match|↑  |0.9318|±  |0.0069|
```

### Serving on 8xMI355 GPUs
Running the model with parallelism of 8 ranks will fail for now, since some kernels are not applicable in this case, i.e., num_heads=8 for each rank (64 heads in total):
- get_mla_metadata_info_v1() assertion fail when num_heads=8.
- Existing MLA decode kernel doesn't support num_heads=8.