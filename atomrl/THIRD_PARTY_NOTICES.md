# AtomRL third-party notices

AtomRL and Lumen are distributed under the repository's MIT license. That
license does not relicense incorporated or adapted sources. Apache-2.0 terms
are shipped in `LICENSE-APACHE`; each adapted file also carries its own SPDX,
upstream copyright, immutable source URL, and AMD modification notice.

## verl-derived sources

- Upstream: <https://github.com/verl-project/verl>
- Primary revision: `8357f1e0a7ce4c06dc970c2c637d49b33815f5cc`
- Historical source revisions retained for files removed upstream:
  - DAPO trainer: `246cac4ca6b0e3ce1009bc48aaa798689e0600af`
  - PPO Ray trainer: `ca439aa5afcfe913c075570a9cb3f0f83730904d`
- License: Apache-2.0

Files:

- `atomrl/atomrl/algorithms/advantage_estimators.py`
- `atomrl/atomrl/algorithms/dapo_sampling.py`
- `atomrl/atomrl/algorithms/kl_controller.py`
- `atomrl/atomrl/algorithms/loss_functions.py`
- `atomrl/atomrl/algorithms/rollout_correction.py`
- `atomrl/atomrl/data/sft_dataset.py`
- `atomrl/atomrl/engine/inference/bucketed_weight_transfer.py`
- `atomrl/atomrl/engine/inference/vllm_colocate_worker_ext.py`
- `atomrl/atomrl/engine/training/base_engine.py`
- `atomrl/atomrl/engine/training/fsdp_engine.py`
- `atomrl/atomrl/engine/training/vllm_colocate_worker_ext.py`
- `atomrl/atomrl/rewards/math_reward.py`
- `atomrl/atomrl/trainer/metric_utils.py`
- `atomrl/atomrl/utils/fsdp_utils.py`
- `atomrl/atomrl/utils/lr_scheduler.py`
- `atomrl/atomrl/utils/seqlen_balancing.py`
- `atomrl/atomrl/utils/torch_functional.py`
- `atomrl/atomrl/utils/ulysses.py`
- `atomrl/atomrl/utils/vocab_parallel.py`
- `atomrl/atomrl/workers/base_worker.py`

`atomrl/atomrl/quantization/rollout_correction.py` is an AMD-authored MIT
compatibility module that only re-exports the Apache-2.0 implementation; it
does not copy the implementation.

## vLLM-derived weight-transfer code

- Upstream: <https://github.com/vllm-project/vllm>
- Revision: `fad09e8a1f51b31eba1f42ff5d651256c77a734d`
- License: Apache-2.0
- Source:
  `examples/offline_inference/rlhf_utils.py`
- Local file:
  `atomrl/atomrl/engine/inference/bucketed_weight_transfer.py`

The file preserves both verl and vLLM provenance and is not relicensed as MIT.

## VeOmni-derived example configurations

- Upstream: <https://github.com/ByteDance-Seed/VeOmni>
- Revision: `573848a00fcd7329c2411346c6f4a983e9f67e3f`
- License: Apache-2.0

Files:

- `atomrl/lumen/examples/perf-veomni/qwen_image_1024.yaml`
- `atomrl/lumen/examples/perf-veomni/qwen_image_smoke.yaml`
- `atomrl/lumen/examples/perf-veomni/wan_video.yaml`

## SGLang-derived model-name remapping

- Upstream: <https://github.com/sgl-project/sglang>
- Revision: `88c10654aedd7d3cc8ae7f16699020aff53efee0`
- License: Apache-2.0
- Copyright: SGLang Team 2023-2024
- Source:
  `python/sglang/srt/models/deepseek_v4.py`
- Local file:
  `atomrl/lumen/examples/dsv4/tools/param_name_remap.py`

## MILES-derived launch arguments

- Upstream: <https://github.com/radixark/miles>
- Revision: `9437366e0aa3a25294720f70d18b081067595f85`
- License: Apache-2.0
- Copyright: RadixArk, Inc. 2025; includes software attributed to Zhipu AI 2025
- Source:
  `scripts/models/deepseek-v4-flash.sh`
- Local file:
  `atomrl/lumen/examples/dsv4/dsv4_flash_megatron_args.sh`

`atomrl/tests/lumen/models/dsv4/test_miles_qat_forward_semantics.py` is an
AMD-authored behavioral compatibility test. It inspects local source behavior
and does not copy MILES implementation code.

## DeepSeek-derived activation quantization

- Upstream:
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>
- Revision: `b5968e9190ef611bbf34a7229255be88a0e937c1`
- License: MIT
- Copyright: DeepSeek 2023
- Source:
  `inference/kernel.py`
- Local file:
  `atomrl/atomrl/lumen/kernels/dsv4/quant/act_quant.py`

## Container-only third-party distributions

`atomrl/lumen/examples/dsv4/prepare_bootstrap.sh` can copy installed Python
distributions from a user-selected bootstrap image. The package list is in
`atomrl/lumen/examples/dsv4/miles_overlay_packages.txt`. The copy operation
preserves each distribution's matching `.dist-info` directory, including its
license metadata, and writes `THIRD_PARTY_IMAGE_SOURCES` with the selected
image/container and copied distributions.

`atomrl/examples/GRPO/docker/Dockerfile` also copies bootstrap runtime
components from the image selected by `DSV4_BOOTSTRAP_IMAGE`. Those components
remain under their own licenses; the image build records the source image and
in-tree source revisions in `/workspace/SOURCE_VERSIONS`.
