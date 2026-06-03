# ATOM v0.1.3

## Highlights

This release features 294 merged PRs from 42 contributors (18 new)!

* **DeepSeek V4 maturity**: DeepSeek V4 received a major enablement and hardening pass this cycle — ATOM added an end-to-end inference path with Triton MoE (#650), DeepSeek V4-Pro recipe coverage (#713, #894), torch compile support (#705), MTP-K and MTP3 paths (#817, #894), TBO (#766), DP attention (#745), and PD disaggregation with prefill optimization (#812). A broad set of performance kernels and runtime fixes landed alongside it, including fused compressor and DualRMSNorm kernels (#723), sparse paged decode rewrite with optional FP8 KV (#936), fused attention / FFN norm (#967), unified AITER RMSNorm-quant plus Q-norm / WKV-gate fusions (#733), three-stream execution (#736), and prepare-decode optimization (#728).
* **ATOM matures for native high-performance inference**: Native paths for DeepSeek V4, Qwen3.5, Qwen3-Next, GLM, gpt-oss, Kimi, MiniMax, and MiMo are expanded with accuracy, quantization, MTP, KV cache, and scheduling fixes.
* **vLLM-ATOM coverage expands**: ATOM currently targets vLLM `0.19.0` and gains broader model, MTP, Sparse MLA, recipe, nightly, and benchmark coverage across DeepSeek V3.2 / V4, GLM, Kimi, Qwen3.5, Qwen3-Next, MiniMax, and DeepSeek-R1 FP4 (#483, #494, #557, #892, #938).
* **SGLang-ATOM support**: ATOM currently targets SGLang `0.5.10` and adds out-of-tree support, image release and validation workflows, Qwen3.5 / Qwen3-Next / DeepSeek coverage, MTP, DP attention, Specv2, Docker support, dashboard integration, and benchmark coverage (#355, #510, #532, #548, #743, #915, #926).
* **Dashboard, benchmark, and CI become release-grade**: This release redesigns the dashboard, adds regression and MTP trend tracking, improves Docker / model metadata, expands nightly accuracy and benchmark coverage, and hardens MI35x runner workflows (#492, #608, #621, #587, #444, #509).

## ATOM Server

### Model Support

* **DeepSeek V4**: Adds the first end-to-end ATOM path with Triton MoE, DeepSeek V4-Pro recipes, compressor state cache visualization, MTP-K skeleton, DP attention, TBO, PD disaggregation, sparse paged decode, and fused compressor / DualRMSNorm / norm-RoPE-quant paths (#650, #713, #714, #817, #745, #766, #812, #936, #899).
* **Qwen3.5 / Qwen3-Next**: Adds native Qwen3.5 support, MXFP4, MTP, multimodal support, fused GEMM, fused gated RMSNorm quant, shuffle-layout full attention, Qwen3-Next DualRMSNorm / QKVGParallelLinear, GDR fixes, and FlyDSL decode paths (#517, #576, #600, #831, #543, #421, #594, #615, #589, #895).
* **GLM**: Adds GLM-5.1 accuracy coverage and GLM MTP fixes (#519, #562, #709).
* **Kimi**: Fixes Kimi quant mapping and related Kimi-K2.5 recipe / loading paths (#732).
* **MiniMax**: Adds MiniMax-M2.5 loading support and upgrades MiniMax-M2.5 to M2.7 with reasoning parser fixes (#558, #775).
* **MiMo**: Adds MiMo-V2-Flash support and fixes loading / accuracy issues (#560, #910).
* **gpt-oss**: Adds transformers 4.57.6 / 5.2.0 compatibility, Quark GPT-OSS 120B loading, TP8 accuracy fixes, GPT-OSS MoE A4W4, int4 allreduce, and accuracy fixes (#419, #445, #449, #764, #776, #748).

### Engine Core

* Refactors ATOM KV cache ownership into attention builders and fixes prefix-cache metadata, scalar KV scales, FP8 KV cache, dead HBM prefix cache, and chunked-prefix workspace bounds (#659, #827, #793, #588, #939, #911).
* Expands MTP support across DeepSeek V4, Qwen3.5, Qwen3-Next, GLM, Kimi, and DeepSeek V3.2, including fixes for zero-token speculative mode, prefix-cache flags, hidden-state flow, and acceptance-rate tracking (#817, #600, #772, #722, #631, #835, #605, #824, #882).
* Adds TBO, DP attention, PrefillDelayer, chunked prefill, deferred-output fixes, and custom all-gather routing that survives CUDA graph capture (#515, #745, #930, #740, #982, #939, #952).
* Refactors OpenAI server support for tool calling, reasoning, debug logging, multimodal API performance, `enable_thinking=False`, and `n` support in completions / chat completions (#489, #971, #472, #647).

### Serving & Scale

* Adds PD disaggregation in ATOM, including Mooncake support and DeepSeek V4-specific P/D optimization (#253, #690, #812).
* Enables DP attention and high-concurrency DP-attention benchmark workloads (#745, #949, #986).
* Adds chunked prefill controls and auto-disable logic for DeepSeek V4 (#740, #982).
* Adds LMCache KV cache offload recipe for ATOM vLLM plugin usage (#890).

## Plugin Server

### vLLM-ATOM

#### Model Support

* Targets vLLM `0.19.0` and transformers `5.2.0` for the vLLM-ATOM plugin path (#483, #597).
* **DeepSeek V3.2**: Adds vLLM-ATOM plugin support, MTP support, FP8 KV cache and Sparse MLA recipes for TP4, plus benchmark / nightly coverage (#494, #557, #835, #892, #938).
* **DeepSeek V4 / DeepSeek-R1 FP4**: Expands plugin coverage for DeepSeek V4 and adds DeepSeek-R1 FP4 validation / recipes (#650, #614).
* **GLM**: Enables GLM-5 and GLM-4.7 support, including GLM-4.7 MTP in the vLLM-ATOM plugin (#399, #722, #805).
* **Kimi**: Adds Kimi-K2.5 plugin support and recipe coverage (#401).
* **Qwen3.5 / Qwen3-Next**: Adds plugin support for Qwen3.5 and Qwen3-Next, including Qwen3-Next MTP (#532, #772).
* **MiniMax**: Adds MiniMax support in ATOM vLLM plugin mode (#545).

#### Engine Core

* Enables Sparse MLA and GLM-5 for vLLM-ATOM, aligns plugin paged attention behavior, fixes KV cache dtype auto parsing, handles scalar KV scales, and passes FP8 dtype into Sparse MLA metadata (#399, #657, #804, #806, #793, #884).
* Improves plugin decode metadata with caching and tail-only fills (#970).
* Removes sequence-length D2H traffic for vLLM-ATOM MTP speculative decode (#933).
* Fixes trust-remote-code and config handling for Kimi-K2.5 and Qwen3.5 model configs (#751, #655).

### SGLang-ATOM

#### Model Support

* Adds SGLang out-of-tree support for ATOM, including image release and validation workflows (#355, #510).
* Adds SGLang-ATOM support for Qwen3.5, Qwen3-Next, DeepSeek V3 / V3.2 / R1 FP4, DeepSeek FP4 MTP, and SGLang MTP / DP attention paths (#532, #777, #643, #614, #834, #926).
* Adds Qwen3.5 MHA support, DeepSeek FP4 MTP benchmark cases, and DeepSeek FP4 DP4 / EP4 nightly accuracy and benchmark cases (#819, #834, #846).
* Adds SGLang-ATOM recipes for DeepSeek MTP and DeepSeek-R1 (#901, #942).

#### Engine Core

* Enables pure data parallel and DP attention in SGLang-ATOM (#665, #743).
* Adds MTP + DP attention support in SGLang-ATOM and Specv2 mode support (#926, #915).
* Enables DeepSeek V3 MTP and removes flattening from SGLang-ATOM MLA paths to align with ATOM vLLM MLA behavior (#643, #525).
* Adds non-absorb MLA prefill support and fused prefill attention preparation paths (#870, #876).

## Hardware & Performance

### ATOM Server

* DeepSeek V4 performance improves through fused compressor kernels, DualRMSNorm fusion, sparse paged decode rewrite, fused attention / FFN norm, unified AITER RMSNorm-quant + Q-norm / WKV-gate fusions, three-stream support, and prepare-decode optimization (#723, #936, #967, #733, #736, #728).
* DeepSeek V4 also gains V4-Pro paged prefill / decode metadata hot-path work, MTP3 direct kernels, Q+KV norm / RoPE consolidation, AITER `silu_and_mul` with in-kernel clamp, and Triton DSV4 fusion work (#894, #899, #731, #704).
* Qwen3.5 / Qwen3-Next gain fused GEMM, fused gated RMSNorm quant, shuffle-layout full attention, MXFP4, GDR fixes, in-place copy optimization, and chunk split removal (#543, #421, #594, #576, #589, #645, #682).
* Decode and attention preparation paths improve with dual-stream prepare decode, FlyDSL GDR decode integration, decode metadata stream parallelism, optimized attention metadata, and removal of unnecessary tensor contiguity before indexer QK RoPE quant/cache kernels (#499, #568, #547, #571, #873).
* Sparse MLA and MLA-related paths improve with Q/K norm-quant fusion, metadata dtype fixes, FP8 attention weight accuracy fixes, optimized fused indexer paths, chunked-prefix prefill workspace bounding, and non-blocking prefix-cache fixes (#528, #884, #670, #788, #911, #939).
* Quantization expands with online quantization, GPT-OSS MoE A4W4, generic MXFP4 shuffle layout alignment, robust MTP quant exclude logic, Quark model fixes, PTPC accuracy fixes, and a HIP replacement for Triton fused RMS FP8 group quant (#653, #764, #744, #646, #787, #670, #507).
* Load-time and runtime memory pressure are reduced by chunking shuffle-weight processing and tightening DeepSeek V4 chunked prefill behavior (#530, #740, #982).

### vLLM-ATOM

* Optimizes Sparse MLA and updates benchmark scope for key plugin workloads (#765, #739).
* Adds Q/K norm-quant fusion, DeepSeek V3.2 fused indexer path, FP8 dtype plumbing for Sparse MLA metadata, and auto KV-cache dtype handling for Sparse MLA (#528, #788, #884, #806).
* Adds FlyDSL GDR support for Qwen3-Next in benchmark and recipe paths, plus GDR decode layout fixes (#634, #895, #752).
* Improves plugin decode build cost with cached metadata, tail-only fills, and removal of sequence-length D2H traffic for MTP speculative decode (#970, #933).
* Enables and tunes GLM-4.7 fusions in the vLLM-ATOM plugin path (#940).
* MoE and communication paths improve with a fused routing-from-topk switch, configurable FP8 blockscale GEMM weight pre-shuffle, GPT-OSS int4 allreduce, MiniMax fused QKNorm+allreduce, and broader MiniMax communication fusion shapes (#725, #694, #776, #774, #820).
* Refines DeepSeek FP4 TP8 / EP8 and DeepSeek-R1 FP4 validation coverage (#639, #614).
* Adds DeepSeek V3.2 MTP TP4 benchmark / nightly cases and V4 DP benchmark coverage (#950, #949).

### SGLang-ATOM

* Adds Q/K norm-quant fusion with the ATOM plugin for DeepSeek and aligns the qknorm fuse operator with the AITER API (#528, #822).
* Adds fused op acceleration for prefill attention preparation and non-absorb MLA prefill support (#876, #870).
* Adds ATOM mesh benchmark cases for SGLang (#874).
* Fixes FP8 attention weight accuracy via PTPC quant recipes (#747).
* Removes MLA flattening in SGLang-ATOM to match the ATOM vLLM MLA path and reduce extra layout work (#525).
* Enables pure data parallel, DP attention, and MTP + DP attention paths for SGLang-ATOM (#665, #743, #926).
* Adds DeepSeek FP4 MTP benchmark coverage and DeepSeek FP4 DP4 / EP4 nightly accuracy and benchmark cases (#834, #846).

## Dashboard, Docs & Tooling

* Redesigns the dashboard around data visualization best practices, adding light mode, regression detection, MTP parsing, TP / GPU display, Docker metadata, separate nightly trend points, and SGLang benchmark integration (#492, #529, #608, #621, #587, #652, #548).
* Improves dashboard UX with sorting, total throughput display, column wrapping, scoped watermark, trend popovers, MI355X detection, and model-name normalization (#447, #461, #524, #498, #692, #600).
* Adds documentation and recipes for installation, model run guides, GPU support, GLM-5, Hermes Agent setup, DeepSeek V4, Kimi, DeepSeek-R1 FP4, vLLM-ATOM, and SGLang-ATOM (#505, #520, #454, #551, #713, #730, #614, #832).
* Adds `rocm-trace-lite` for GPU kernel profiling and improves profiler robustness / stream-parallel decode metadata (#535, #547).
* Adds automation skills for Llama workload enablement, benchmark comparison / PR creation, and InferenceX sync workflows (#467, #795, #802).

## Build & Dependencies

* Aligns the plugin stacks around vLLM `0.19.0`, transformers `5.2.0`, SGLang `0.5.10`, and matching Triton / AITER dependency handling (#483, #597, #458, #538, #823).
* Improves Docker release flows for ATOM, vLLM-ATOM, and SGLang-ATOM with nightly date-tagged images, GPU-machine builds, branch-aware releases, and podman-compatible benchmark paths (#503, #510, #561, #785, #601, #617).
* Hardens CI reliability with MI35x runner validation, AMD CI monitoring, lock-protected model downloads, model-cache reuse, longer server readiness waits, and pre-checkin gating (#444, #504, #509, #474, #635, #724).
* Expands nightly accuracy and benchmark coverage across GLM-5.1, MiniMax-M2.5, Qwen3.5 FP4 / MXFP4, DeepSeek V3.2 MTP, DeepSeek FP4, vLLM-ATOM, and SGLang-ATOM workloads (#519, #564, #593, #586, #938, #846).

## Contributors

Thanks to all contributors who made this release possible:

@sunway513, @asleepzzz, @dwiddows, @carlushuang, @ZhiweiYan-96, @XiaobingSuper, @xytpai, @kliuae, @benenzhu, @k50112113, @wanzhenchn, @inkcherry, @wuhuikx, @wufann, @ganyi1996ppo, @gyohuangxin, @andyluo7, @junhaha666, @functionstackx, @whx-sjtu, @ZLkanyo009, @zovonoir, @zhuyuhua-v, @valarLip, @amdfaa, @seungrokj, @amd-ruitang3, @thpereir, @junyyang-amd, @yzhou103, @ZhangLirong-amd, @zejunchen-zejun, @haoyangli0109, @kliuae-amd, @HaonanWang98, @gbyu-amd, @jiayyu, @qichu-yun, @PerryZhang01, @ChuanLi1101, @yhl-amd, @JiaoliangYu

## New Contributors

@dwiddows, @functionstackx, @asleepzzz, @benenzhu, @seungrokj, @ZLkanyo009, @whx-sjtu, @Copilot, @qichu-yun, @wufann, @zovonoir, @yhl-amd, @amdfaa, @yzhou103, @xytpai, @JiaoliangYu, @junyyang-amd, and @ZhiweiYan-96 made their first contributions in this release.

**Full Changelog**: v0.1.2...v0.1.3
