# (mock) 455-CI recipe, Qwen-35B-A3B server
HIP_VISIBLE_DEVICES=0 python3 -m atom.entrypoints.openai_server \
  --model /models/Qwen3.5-35B-A3B-MXFP4 --tensor-parallel-size 1 --server-port 8000
