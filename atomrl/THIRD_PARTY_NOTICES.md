# AtomRL third-party notices

AtomRL and Lumen are distributed under the repository's MIT license. That
license does not relicense incorporated or vendored sources: each source keeps
the upstream license and attribution shown below. The Apache-2.0 text shipped
with this repository is `atom/mesh/LICENSE`; the DeepSeek MIT terms are
reproduced in the upstream `LICENSE` linked below.

## verl-derived sources

- Upstream: <https://github.com/volcengine/verl> (Apache-2.0)

- `atomrl/atomrl/algorithms/advantage_estimators.py`
- `atomrl/atomrl/algorithms/kl_controller.py`
- `atomrl/atomrl/algorithms/rollout_correction.py`
- `atomrl/atomrl/engine/training/base_engine.py`
- `atomrl/atomrl/engine/training/fsdp_engine.py`
- `atomrl/atomrl/data/sft_dataset.py`
- `atomrl/atomrl/rewards/math_reward.py`
- `atomrl/atomrl/engine/inference/bucketed_weight_transfer.py`
- `atomrl/atomrl/trainer/metric_utils.py`
- `atomrl/atomrl/utils/fsdp_utils.py`
- `atomrl/atomrl/utils/lr_scheduler.py`
- `atomrl/atomrl/utils/seqlen_balancing.py`
- `atomrl/atomrl/utils/torch_functional.py`
- `atomrl/atomrl/utils/ulysses.py`
- `atomrl/atomrl/utils/vocab_parallel.py`

These files preserve their upstream copyright and Apache-2.0 provenance
headers. The files adapted in this change now carry an SPDX header and an
explicit upstream source link.

## vLLM-compatible weight transfer

- Upstream project: <https://github.com/vllm-project/vllm> (Apache-2.0)
- `atomrl/atomrl/engine/inference/bucketed_weight_transfer.py`

The AtomRL implementation is self-contained and documents its direct `verl`
source path above. This entry records the compatible vLLM weight-transfer
interface; the file is not relicensed as MIT.

## DeepSeek-derived sources

- DeepSeek-V4 source:
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py>
  (MIT; copyright DeepSeek 2023):
  - `atomrl/atomrl/lumen/kernels/dsv4/quant/act_quant.py`
