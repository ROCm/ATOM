FROM rocm/atom-dev:nightly_202608041536

# transformers >= 5.16 is required for the glm5_next architecture (GLM-5.3-Flash).
# Every install below is --no-deps on purpose: the base image already ships ROCm
# torch + triton, and letting pip resolve `accelerate`/`kernels` dependencies pulls
# the CUDA torch wheel and uninstalls the ROCm one.
RUN pip install --no-cache-dir --no-deps \
      "transformers==5.16.1" \
      "tokenizers>=0.23.1,<0.24" \
      "accelerate" \
      "kernels==0.16.0" \
      "kernels-data" \
 && python3 -c "\
import torch, transformers;\
from transformers.models.glm5_next import modeling_glm5_next as m;\
print('torch', torch.__version__);\
print('transformers', transformers.__version__);\
print('glm5_next OK:', m.Glm5NextForConditionalGeneration.__name__)"

# The checkpoint is block-FP8, so every Linear goes through the hub-hosted
# `finegrained-fp8` Triton kernel. It is pure Triton and runs on ROCm, but it is
# fetched from the Hub on first use — warm it at build time so runs are repeatable
# and do not need network access.
ENV HF_HOME=/opt/hf_home
RUN python3 -c "\
from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel;\
print('finegrained-fp8 warmed:', type(load_finegrained_fp8_kernel()).__name__)"
