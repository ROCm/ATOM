window.BENCHMARK_DATA = {
  "lastUpdate": 1787243964262,
  "repoUrl": "https://github.com/ROCm/ATOM",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "wanzhenchn@gmail.com",
            "name": "wanzhenchn",
            "username": "wanzhenchn"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "18b17f4043ca381da8d1c8ec1beb409b44353b2a",
          "message": "ci(mesh): add Atomesh accuracy and benchmark workflows (#1159)\n\n* ci(mesh): add Atomesh accuracy and benchmark workflows\n\n- Validate standalone-mode accuracy via Atomesh entrypoints.\n- Mocker benchmark to PD routing scenarios with topology and consumer concurrency matrix.\n\n* [ci][mesh] add Atomesh mocker benchmark dashboard\n\n- Add a custom dashboard for Atomesh mocker benchmark results.\n- Show throughput, latency, detailed performance data, commit links, and CI run links.\n- Align the benchmark matrix with 1P1D, 2P1D, and 3P1D topologies across consumer concurrency levels.\n\n* [ci] Skip unrelated ATOM, vLLM, and SGLang CI for mesh-only PRs.\n\n* [ci][mesh] Enable mocker dashboard publishing workflow to run on zwan/feat-mesh-ci pushes.\n\n* Polish Atomesh mocker dashboard legends\n\n* [ci][mesh] fix atomesh standalone accuracy data source\n\n* Revert 'Enable mocker dashboard publishing workflow to run on zwan/feat-mesh-ci pushes.'\n\n* [ci][mesh] add logo and display theme for mesh mocker benchmark dashboard\n\n* [ci][mesh] Polish Atomesh dashboard and accuracy data flow",
          "timestamp": "2026-06-15T15:50:22+08:00",
          "tree_id": "6f4740956be82e7177ea5f44dd264b4cbcb4729f",
          "url": "https://github.com/ROCm/ATOM/commit/18b17f4043ca381da8d1c8ec1beb409b44353b2a"
        },
        "date": 1781511608187,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27531858784 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606141623 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7483,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27531858784 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606141623 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7491 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Hyukjoon Lee",
            "username": "hyukjlee",
            "email": "hyukjlee@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "368cd515d71a329031fc9f4d6f0f72065fe20717",
          "message": "Kimi k2.6 with claude code /v1/messages API support and  prompt caching enabled  (#1184)\n\n* feat(server): add Anthropic Messages API endpoint (/v1/messages)\n\nEnables Claude Code and other Anthropic-compatible tools to use ATOM\nas a backend. Translates between Anthropic Messages format and ATOM's\ninternal OpenAI format.\n\nSupports:\n- Non-streaming and streaming responses\n- System messages, multi-turn conversations\n- Thinking/reasoning content separation (via ReasoningFilter)\n- Anthropic SSE event format (message_start, content_block_delta, etc.)\n- Tool definitions translation (Anthropic → OpenAI format)\n\nUsage with Claude Code:\n  ANTHROPIC_BASE_URL=http://localhost:8000 \\\n  ANTHROPIC_AUTH_TOKEN=dummy \\\n  ANTHROPIC_MODEL=MiniMax-M2.7 \\\n  claude\n\n* fix(anthropic): fix streaming handler, reasoning filter, and Claude Code compat\n\n- Fix ToolCallStreamParser integration: consume (event_type, data) tuples\n  from process()/flush() instead of calling nonexistent get_content()/\n  get_tool_calls() methods\n- Fix cleanup_streaming_request() call with missing request_id argument\n- Fix _build_sampling_params() missing ignore_eos, None top_k/top_p\n- Init ReasoningFilter in state 1 when chat template ends with <think>,\n  so thinking models like K2.6 have reasoning properly hidden\n- Increase ReasoningFilter buffer threshold from 7 to 100 chars to avoid\n  prematurely emitting thinking as visible content\n- Add prompt truncation when input exceeds max_model_len\n- Add cache_creation_input_tokens and cache_read_input_tokens to usage\n\n* fix(anthropic): pass tool definitions to model via chat template\n\nClaude Code sends tool schemas (WebSearch, Bash, etc.) in every request,\nbut the /v1/messages handler was hardcoding tools=None. The model never\nsaw tool definitions and couldn't generate proper tool_use calls.\n\nNow converts and forwards request.tools via anthropic_to_openai_tools(),\nenabling the model to use WebSearch, WebFetch, and other Claude Code tools.\n\n* fix(anthropic): suppress thinking blocks, add signature support\n\n- Skip streaming thinking blocks entirely to avoid Claude Code's\n  signature verification rejection. Thinking still happens server-side\n  but only the final answer is sent to the client.\n- Add signature field to thinking content blocks and signature_delta\n  SSE events for compatibility with Claude Code 2.1.143+.\n- Add stream_signature_delta() helper function.\n\n* fix(anthropic): strip attribution header, use model tool IDs\n\n- Strip Claude Code's x-anthropic-billing-header from system prompt\n  server-side (matches vLLM behavior) to preserve prefix caching\n- Use model-native tool call IDs (functions.name:index) instead of\n  random UUIDs, matching vLLM's kimi_k2 parser for multi-turn compat\n- Remove unused uuid import from tool_parser\n- Add tests for attribution header stripping\n\n---------\n\nCo-authored-by: carlushuang <carlus.huang@amd.com>",
          "timestamp": "2026-06-15T14:17:10Z",
          "url": "https://github.com/ROCm/ATOM/commit/368cd515d71a329031fc9f4d6f0f72065fe20717"
        },
        "date": 1781547245171,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9447,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27565040307 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9409 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9553,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27565040307 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9568 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.5,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27565040307 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9568 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7536,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27565040307 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7521 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8901,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27565040307 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3601 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "103567126+valarLip@users.noreply.github.com",
            "name": "Lingpeng Jin",
            "username": "valarLip"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1b5bef7bb19bb7136b2b40a6841e56ad386f2d03",
          "message": "fix(v4): zero-init all-gather padding to stop uninitialized memory corrupting MoE (#1229)\n\n* fix(v4): zero-init all-gather padding to stop uninitialized memory corrupting MoE\n\npad_for_all_gather built the padding rows with torch.empty and never zeroed\nthem (the .zero_() was commented out), contradicting the function's own\ndocstring. Those uninitialized rows are all-gathered across DP ranks and fed\nstraight into the aiter fused-MoE expert GEMM, and the padded input_ids reach\ntid2eid[ids] for V4 hash routing. Garbage there leaks into real tokens'\noutputs.\n\nBecause the corruption is whatever happens to sit in freshly-allocated GPU\nmemory, the result is nondeterministic across machines/runs: locally it landed\nat GSM8K ~0.95, but CI on a different SKU dropped to 0.9007 (TBO+DPA conc1000,\nbelow the 0.93 threshold) and a local rerun crashed with a null-pointer GPU\nmemory access fault (garbage id -> out-of-range expert -> invalid weight ptr).\nRestoring the zero fixes all three: padding hidden is benign and padding ids\nroute to expert 0.\n\nWith the pad guaranteed zero, the _hash_topk clamp band-aid is replaced by an\nassert that input_ids length matches gating_output num_tokens, surfacing any\nreal DP-layout mismatch instead of silently masking it.\n\nAlso remove the _run_on_tbo_comm_stream side-stream helper: its only caller\n(MoE.combine_outputs TP all-reduce) now runs inline, matching the ids-gather\nwhich must stay inline to keep DP collective ordering aligned under TBO.\nRename compress_stream -> indexer_stream for accuracy.\n\nVerified: V4-Pro TBO+DPA conc1000 GSM8K 3-shot = 0.9515 (flexible) / 0.9522\n(strict), no GPU fault, drain clean.\n\n* ci: TEMP run only DeepSeek-V4-Pro TBO+DPA conc1000 (revert before merge)\n\nFlip every accuracy entry except the TBO+DPA conc1000 case to test_level\n\"off\" so any trigger (pr/push/dispatch/schedule) runs only this one job,\nto validate the pad zero-init fix in CI quickly.\n\nDO NOT MERGE this commit — drop it before merging the PR.\n\n* Fix TBO 1024c accurary issue by remove cpu yield in collective op\n\n(cherry picked from commit 9bf2d25c99e0c7ad03c61f9255d1b0d8edeebe45)\n\n* test(v4): disable pad zero-init for CI repro + print server cmd\n\n- moe.py: temporarily comment out pad_for_all_gather zero-init to reproduce\n  the uninitialized-padding behavior in CI (the CI gate already restricts the\n  run to the V4-Pro TBO+DPA conc1000 case).\n- deepseek_v4.py: restore the tid2eid[ids] clamp as a bounds guard for hash\n  routing.\n- atom_test.sh: print the full openai_server command line before launch so the\n  CI log shows the exact server args.\n\nExperiment on top of the pad zero-init fix — not for merge as-is.\n\n* ci: restore full accuracy matrix (undo temp single-case gate)\n\nReverts the test_level \"off\" gate from 3662ac00 — all accuracy cases are\nre-enabled at their original pr/main/nightly levels. The CI experiment that\nneeded only DeepSeek-V4-Pro TBO+DPA conc1000 is done.\n\n* ci: lower gpt-oss-120b accuracy threshold to 0.87\n\nBoth gpt-oss-120b entries (1-GPU and 2-GPU) drop from 0.88 to 0.87 to absorb\nrun-to-run GSM8K variance. Other models unchanged.\n\n* perf(v4): fuse _hash_topk into a single Triton kernel\n\nThe hash-routing custom_routing_function for V4's first layers ran\nsoftplus+sqrt over every routed expert (n_routed_experts ~256-384) but kept\nonly topk (~6) of them, plus separate clamp / tid2eid gather / score gather /\nrenorm / scale ops.\n\ntriton_hash_topk.py fuses all of it into one kernel (one program per token):\nid clamp, tid2eid[id] lookup, gating gather at the selected experts only,\nsqrt(softplus(.)), optional renorm and scaling. When shared experts are fused\nit writes directly into the first topk columns of the global topK buffer,\navoiding an extra copy.\n\nNumerics match the PyTorch path (max|dw| ~1e-7 fp32 / ~5e-7 bf16 across OOB\nids, bf16, renorm on/off, sliced-buffer write). V4-Pro TBO+DPA conc1000 GSM8K\n3-shot = 0.9522.\n\n* ci: print server cmd with [@] expansion to match actual invocation\n\nUse ${ARRAY[@]} instead of ${ARRAY[*]} in the debug echo so the printed\ncommand line reflects the same word-splitting/quoting as the real launch\nthat uses \"${ARRAY[@]}\" (addresses Copilot review).\n\n---------\n\nCo-authored-by: ZhangLirong-amd <Lirong.Zhang@amd.com>",
          "timestamp": "2026-06-16T23:01:23+08:00",
          "tree_id": "2726baca4aa6b8b962b93fe26548d65e29a11acd",
          "url": "https://github.com/ROCm/ATOM/commit/1b5bef7bb19bb7136b2b40a6841e56ad386f2d03"
        },
        "date": 1781624193878,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27627027719 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9439 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7491,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27627027719 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7483 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zejunchen-zejun",
            "username": "zejunchen-zejun",
            "email": "zejun.chen@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "e79fe6f51372e0d33efc48f0fe3e39548e3fe4dc",
          "message": "[atom-vllm benchmark MTP] refine benchmark command for atom-vllm MTP case (#1216)\n\n* [atom-vllm benchmark MTP] refine benchmark command for\natom-vllm MTP case\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>\n\n* add performance mode for glm4.7 mtp case and qwen3next mtp case\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>\n\n* add qwen3next mtp config\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>\n\n* remove perf mode because it is useless\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>\n\n* fix missing allreduce for glm4.7 mtp\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>\n\n* align atom-vllm acc test\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>\n\n* add mtp accept ratio check\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>\n\n---------\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>",
          "timestamp": "2026-06-16T15:17:58Z",
          "url": "https://github.com/ROCm/ATOM/commit/e79fe6f51372e0d33efc48f0fe3e39548e3fe4dc"
        },
        "date": 1781633842958,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9545,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27636884966 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27636884966 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9484 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.37,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27636884966 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9484 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7475,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27636884966 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7491 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8961,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27636884966 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202606151651 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3328 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "carlus.huang@amd.com",
            "name": "carlushuang",
            "username": "carlushuang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e66dc33f66925382fe28be22be8d6fb47d779c2c",
          "message": "glm_moe_dsa: support GLM-5.2 IndexShare (FP8) (#1260)\n\n* glm_moe_dsa: support GLM-5.2 IndexShare (FP8)\n\nGLM-5.2 (glm_moe_dsa) extends the DeepSeek-V3.2-style DSA stack with\nIndexShare: layers marked \"shared\" in `indexer_types` reuse the preceding\n\"full\" layer's indexer/topk and carry no indexer weights of their own in\nthe checkpoint.\n\n- models/deepseek_v2.py:\n  - Make `indexer_types` the authoritative source for the per-layer\n    indexer-skip decision (supersedes index_topk_pattern / index_topk_freq).\n  - Honor `index_skip_topk_offset` in the freq-based fallback (default 1\n    preserves existing DeepSeek behavior).\n  - Reuse the cached topk for the MTP layer when\n    `index_share_for_mtp_iteration` is set.\n  - Do not build indexer weights for \"shared\" layers; otherwise their\n    parameters load nothing from the checkpoint, stay at init values and\n    corrupt the indexer (the forward and the index-cache binding already\n    guard on `indexer is not None`).\n- config.py: auto-enable `use_index_cache` for glm_moe_dsa when the model\n  declares an IndexShare schedule, so serving works without passing an\n  --hf-overrides flag.\n- plugin/vllm/model_wrapper.py: re-apply the auto-enable after vLLM\n  replaces ATOM's hf_config.\n\nValidated on 8x MI355X (TP=8, FP8): native ATOM loads all weights with no\nunloaded params and generates correctly for 1k/1k and 8k/1k inputs.\n\n* docs: document GLM-5.2 (IndexShare) serving + add News entry\n\n- recipes/GLM-5.md: add a GLM-5.2 (IndexShare) section with the TP8 serve\n  command, configuration tips (bf16 KV, gpu-mem-util 0.8), and 8xMI355X\n  perf baselines for 1k/1k and 8k/1k; add a pointer from the intro.\n- README.md: add a News entry announcing GLM-5.2 FP8 support.\n\n* docs: note GLM-5.2 in README Supported Models table\n\n* style: black formatting for indexer_types skip return\n\n* style: condense GLM-5.2 code comments\n\n* refactor: move maybe_enable_glm_dsa_index_cache into deepseek_v2\n\nOwn the indexer-cache auto-enable in the model: call it once in\nDeepseekV2ForCausalLM.__init__ (covers native + vLLM plugin) instead of\nin config.get_hf_config and the vLLM wrapper.\n\n* refactor: inline index-cache enable into _should_skip_index_topk\n\nDrop maybe_enable_glm_dsa_index_cache; instead, when index_topk_freq > 1\n(IndexShare) turn on use_index_cache directly in _should_skip_index_topk.\nNo model_type gating needed.\n\n* refactor: gate index_topk_freq check under the use_index_cache branch\n\n* refactor: drop redundant 'or 1' guard on index_topk_freq\n\n* benchmark: add GLM-5.2-FP8 to dashboard (perf + accuracy)\n\nNative-engine catalog entries for the nightly dashboard:\n- models.json: TP8 FP8, kv_cache_dtype fp8, --gpu-memory-utilization 0.8\n  (DSA index cache OOMs at default 0.9), conc up to 256.\n- models_accuracy.json: gsm8k threshold 0.92 (measured 3-shot\n  flexible-extract 0.9447 on 8x MI355X).",
          "timestamp": "2026-06-17T21:35:46+08:00",
          "tree_id": "f9a0d69afe3773e3827fdc11b5f146fea9e77a27",
          "url": "https://github.com/ROCm/ATOM/commit/e66dc33f66925382fe28be22be8d6fb47d779c2c"
        },
        "date": 1781706065392,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9469,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27692965736 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606161823 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9447 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7491,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27692965736 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606161823 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7475 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "JiaoliangYu",
            "username": "JiaoliangYu",
            "email": "Jiaoliang.Yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0eac5ab5f828ad8ab29c1b8ba9e464257dc47313",
          "message": "benchmark: only cpu affinity (#1265)\n\nCo-authored-by: JiaoliangYu <jiaolyu@amd.com>",
          "timestamp": "2026-06-17T13:40:11Z",
          "url": "https://github.com/ROCm/ATOM/commit/0eac5ab5f828ad8ab29c1b8ba9e464257dc47313"
        },
        "date": 1781720237654,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27706397866 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606171607 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9545,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27706397866 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606171607 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9553 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.58,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27706397866 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606171607 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9553 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7559,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27706397866 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606171607 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7551 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8802,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27706397866 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202606171607 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.276 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "d7964d50be17a3910dec1d22cf1d4f6205764cb4",
          "message": "feat: fuse V4 decode SWA cache-write into qk_norm_rope_maybe_quant (#1272)\n\n* feat: fuse V4 decode SWA cache-write into qk_norm_rope_maybe_quant\n\nThread the SWA ring scatter through the qk_norm+rope bridge so the V4\ndecode path no longer launches a standalone swa_write per layer. When\nswa_kv is provided, the post-norm/rope KV row is written into\nswa_kv[slot, pos % cache_size, :] (slot = state_slot_mapping[\nbatch_id_per_token[t]]) inside the same kernel:\n\n- flydsl path: fuses the scatter into the qk_norm launch (no extra\n  kernel, no [T, D] KV HBM round-trip), via the new swa_kv /\n  state_slot_mapping / batch_id_per_token args on flydsl_qk_norm_rope_quant.\n- Triton fallback: emits the existing swa_write as a separate launch\n  (driven by swa_cu_seqlens_q + state_slot_mapping) so both backends have\n  identical side effects.\n\ndeepseek_v4.py decode deletes its standalone swa_write call and passes\nthe SWA args through the bridge instead; prefill is unchanged (still\nwrites its in-chunk SWA tail via swa_write after sparse_attn). BF16 only.\n\nRequires the matching aiter change (ROCm/aiter#3776) for the flydsl\nfused-scatter kernel support.\n\n* ci: drop GLM-5-FP8 from benchmark matrix to stay under 256 cells\n\nThe nightly atom-benchmark grid had grown to 264 fully-expanded matrix\ncells, exceeding GitHub Actions' hard limit of 256 configurations per\njob. Remove the GLM-5-FP8 benchmark variant (superseded by GLM-5.2-FP8,\nwhich is retained) and its workflow_dispatch checkbox (keeping it in sync\nwith the catalog prefixes). Matrix now resolves to 250 cells.\n\nAccuracy validation (models_accuracy.json) and the dashboard color map\nare left unchanged — GLM-5-FP8 stays covered there.\n\n* fix: standardize V4 batch_id_per_token on int32 for fused SWA scatter\n\nThe fused decode SWA scatter loads batch_id_per_token at int32 width\n(see ROCm/aiter#3793). The producers were int64, which raised\n\"batch_id_per_token must be 1-D int64\" on the V4-Pro MTP decode path\n(server failed to start -> accuracy job timed out).\n\nMake all batch_id_per_token producers int32:\n- v4_batch_id_per_token CpuGpuBuffer (model_runner path) int64 -> int32\n- batch_id numpy sources (per-fwd + MTP draft) int64 -> int32\n- sglang / vllm plugin bridge batch_id buffers + numpy sources -> int32\n\nint32 indices are accepted by torch advanced-indexing (indexer meta) and\nby the triton kernels (tl.load is dtype-agnostic); the explicit\n.to(torch.int64) casts in csa_translate_pack / sglang remain and tolerate\nint32 input. batch_id values are bounded by batch size, far below 2^31.\n\nValidated end-to-end: DeepSeek-V4-Pro MTP3 GSM8K (3-shot) flexible\n0.9477 / strict 0.9484, above the 0.94 CI threshold; decode drained\ncleanly with no TypeError.",
          "timestamp": "2026-06-18T14:06:23Z",
          "url": "https://github.com/ROCm/ATOM/commit/d7964d50be17a3910dec1d22cf1d4f6205764cb4"
        },
        "date": 1781805044075,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9462,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27776559944 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606181332 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9447 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27776559944 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606181332 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.42,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27776559944 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606181332 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7597,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27776559944 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606181332 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7582 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8908,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27776559944 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202606181332 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3124 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "d7964d50be17a3910dec1d22cf1d4f6205764cb4",
          "message": "feat: fuse V4 decode SWA cache-write into qk_norm_rope_maybe_quant (#1272)\n\n* feat: fuse V4 decode SWA cache-write into qk_norm_rope_maybe_quant\n\nThread the SWA ring scatter through the qk_norm+rope bridge so the V4\ndecode path no longer launches a standalone swa_write per layer. When\nswa_kv is provided, the post-norm/rope KV row is written into\nswa_kv[slot, pos % cache_size, :] (slot = state_slot_mapping[\nbatch_id_per_token[t]]) inside the same kernel:\n\n- flydsl path: fuses the scatter into the qk_norm launch (no extra\n  kernel, no [T, D] KV HBM round-trip), via the new swa_kv /\n  state_slot_mapping / batch_id_per_token args on flydsl_qk_norm_rope_quant.\n- Triton fallback: emits the existing swa_write as a separate launch\n  (driven by swa_cu_seqlens_q + state_slot_mapping) so both backends have\n  identical side effects.\n\ndeepseek_v4.py decode deletes its standalone swa_write call and passes\nthe SWA args through the bridge instead; prefill is unchanged (still\nwrites its in-chunk SWA tail via swa_write after sparse_attn). BF16 only.\n\nRequires the matching aiter change (ROCm/aiter#3776) for the flydsl\nfused-scatter kernel support.\n\n* ci: drop GLM-5-FP8 from benchmark matrix to stay under 256 cells\n\nThe nightly atom-benchmark grid had grown to 264 fully-expanded matrix\ncells, exceeding GitHub Actions' hard limit of 256 configurations per\njob. Remove the GLM-5-FP8 benchmark variant (superseded by GLM-5.2-FP8,\nwhich is retained) and its workflow_dispatch checkbox (keeping it in sync\nwith the catalog prefixes). Matrix now resolves to 250 cells.\n\nAccuracy validation (models_accuracy.json) and the dashboard color map\nare left unchanged — GLM-5-FP8 stays covered there.\n\n* fix: standardize V4 batch_id_per_token on int32 for fused SWA scatter\n\nThe fused decode SWA scatter loads batch_id_per_token at int32 width\n(see ROCm/aiter#3793). The producers were int64, which raised\n\"batch_id_per_token must be 1-D int64\" on the V4-Pro MTP decode path\n(server failed to start -> accuracy job timed out).\n\nMake all batch_id_per_token producers int32:\n- v4_batch_id_per_token CpuGpuBuffer (model_runner path) int64 -> int32\n- batch_id numpy sources (per-fwd + MTP draft) int64 -> int32\n- sglang / vllm plugin bridge batch_id buffers + numpy sources -> int32\n\nint32 indices are accepted by torch advanced-indexing (indexer meta) and\nby the triton kernels (tl.load is dtype-agnostic); the explicit\n.to(torch.int64) casts in csa_translate_pack / sglang remain and tolerate\nint32 input. batch_id values are bounded by batch size, far below 2^31.\n\nValidated end-to-end: DeepSeek-V4-Pro MTP3 GSM8K (3-shot) flexible\n0.9477 / strict 0.9484, above the 0.94 CI threshold; decode drained\ncleanly with no TypeError.",
          "timestamp": "2026-06-18T14:06:23Z",
          "url": "https://github.com/ROCm/ATOM/commit/d7964d50be17a3910dec1d22cf1d4f6205764cb4"
        },
        "date": 1781890106806,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27838449670 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606191602 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9469 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9545,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27838449670 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606191602 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.3,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27838449670 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606191602 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7483,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27838449670 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606191602 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7475 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.881,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27838449670 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202606191602 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3184 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ea08015c51aeaab40bd39b89eef009df9c148dc3",
          "message": "feat: fuse indexer Q fp8 quant into rope_rotate_activation (#1298)\n\nReplace the two-step indexer Q preparation (bf16 rope_rotate_activation +\nseparate get_hip_quant(per_1x128)) with the fused fp8 path: a single\nrope_rotate_activation call that applies RoPE + Hadamard-rotate and writes\nthe fp8-quantized Q with its per-(token, head) block scale via out_scale.\n\nThe bf16 rotated Q is never read back, so quantizing it in-kernel avoids\nmaterializing the intermediate. group_size = head_dim (128) => one scale\nper (token, head). The fused kernel's fp8 quant matches\ndynamic_per_group_scaled_quant_kernel.\n\nVerified on DeepSeek-V4-Pro: GSM8K 3-shot ~0.953-0.957 and 10-shot 0.9568\n(baseline 0.9522 +/- 0.0059, no regression); conc-16 throughput\n1644 tok/s (on par with baseline).",
          "timestamp": "2026-06-20T15:17:56Z",
          "url": "https://github.com/ROCm/ATOM/commit/ea08015c51aeaab40bd39b89eef009df9c148dc3"
        },
        "date": 1781976034169,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27877388597 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606201539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9469 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27877388597 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606201539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9538 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.63,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27877388597 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606201539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9538 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7513,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27877388597 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606201539 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7528 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8779,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27877388597 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202606201539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3207 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ea08015c51aeaab40bd39b89eef009df9c148dc3",
          "message": "feat: fuse indexer Q fp8 quant into rope_rotate_activation (#1298)\n\nReplace the two-step indexer Q preparation (bf16 rope_rotate_activation +\nseparate get_hip_quant(per_1x128)) with the fused fp8 path: a single\nrope_rotate_activation call that applies RoPE + Hadamard-rotate and writes\nthe fp8-quantized Q with its per-(token, head) block scale via out_scale.\n\nThe bf16 rotated Q is never read back, so quantizing it in-kernel avoids\nmaterializing the intermediate. group_size = head_dim (128) => one scale\nper (token, head). The fused kernel's fp8 quant matches\ndynamic_per_group_scaled_quant_kernel.\n\nVerified on DeepSeek-V4-Pro: GSM8K 3-shot ~0.953-0.957 and 10-shot 0.9568\n(baseline 0.9522 +/- 0.0059, no regression); conc-16 throughput\n1644 tok/s (on par with baseline).",
          "timestamp": "2026-06-20T15:17:56Z",
          "url": "https://github.com/ROCm/ATOM/commit/ea08015c51aeaab40bd39b89eef009df9c148dc3"
        },
        "date": 1782062235151,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9439,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27910812727 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606211542 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9416 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27910812727 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606211542 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.7,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27910812727 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3: 3 speculative tokens). Local full-eval (1319 samples, 3-shot) flexible-extract = 0.9560 ± 0.0056. | Docker: rocm/atom-dev:nightly_202606211542 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7491,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27910812727 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606211542 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7491 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8886,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/27910812727 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202606211542 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.4162 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "7058d7085ff2d1b3ca317d7ad1f49839e8348a0e",
          "message": "ci: native CI foundation de-inlining + unit-test gate (#1385)\n\n* ci: validate accuracy catalogs against JSON Schema in pre-checks\n\nAdd a JSON Schema for the flat accuracy catalogs (models_accuracy.json,\noot_models_accuracy.json, sglang_models_accuracy.json) plus a\nvalidate_catalog.py gate wired into the pre-checks (T0) workflow.\n\nadditionalProperties:false locks the current shape so typos / stray fields\nfail CI; a semantic rule requires each entry to declare exactly one pass-bar\nspelling (accuracy_threshold / accuracy_test_threshold). The existing\nextraArgs/extra_args and threshold-name drift is tolerated for now and will be\nnormalized separately. Documented in benchmark/README.md.\n\n* ci: extract docker login into reusable docker-auth composite action\n\nReplace the inline `echo $PASSWORD | docker login` steps in the ATOM-native\nworkflows (atom-test, atom-benchmark, atom-mmstar-ci, docker-release,\natomesh-accuracy-validation) with a shared .github/actions/docker-auth composite.\n\nCredentials are passed via env instead of being interpolated into the run\ncommand, removing the template-injection vector. The composite also supports an\nexplicit registry, image-derived registry, and a custom engine so the\nvllm/sglang call sites can reuse it in a follow-up.\n\n* ci: de-inline aiter wheel download into a shared script\n\nExtract the ~163-line aiter wheel resolve+download block (byte-identical in\natom-test and atomesh-accuracy-validation) into\n.github/scripts/download_aiter_wheel.sh; both workflows now call it\n(net -326 inline lines).\n\nLogic matches the previous inline block exactly. GITHUB_TOKEN is passed via env\ninstead of being interpolated into the run command, and the S3 / API /\nworkflow-id constants become overridable env defaults.\n\natom-mmstar-ci uses a simpler S3-only variant (no artifact fallback) and is\nleft for a follow-up.\n\n* ci: de-inline aiter wheel install into a shared script\n\nExtract the identical \"Install aiter from wheel\" block from atom-test and\natomesh-accuracy-validation into .github/scripts/install_aiter_wheel.sh.\n\nBehavior matches the previous inline block (no outer set -e, so a missing wheel\nstill hits the explicit error+ls path). CONTAINER_NAME comes from the job env;\nthe wheel dir is an overridable env default (/tmp/aiter-whl).\n\natom-mmstar-ci uses a --no-deps variant from a different dir and is left for a\nfollow-up.\n\n* ci: extract CI container startup into setup-gpu-container composite\n\nReplace the identical ~60-line \"Start CI container\" steps in atom-test and\natomesh-accuracy-validation with a shared .github/actions/setup-gpu-container\ncomposite. The three differences are inputs: network-host (atom-test sets host\nnetworking), extra-run-flags (atomesh adds USE_ATOMESH_ENTRYPOINTS/ATOM_SERVER_PORT),\nand the runner label that drives the --pull policy.\n\nThe assembled docker run command is byte-identical to the previous inline blocks\nfor both callers (verified with a stubbed docker). atom-mmstar-ci / docker-release\n/ gpu-load-test use more divergent startup blocks and are left for a follow-up.\n\n* ci: serialize gh-pages deploys with a shared concurrency group\n\nAll six workflows that push to the gh-pages branch (docs, deploy-pages,\natom-benchmark, atomesh-mocker-benchmark, atom-sglang-benchmark,\natom-vllm-benchmark) now run their deploy job under a shared concurrency group\n(gh-pages-deploy, cancel-in-progress: false).\n\nThis serializes the fetch/checkout/commit/push dance so concurrent runs can no\nlonger race on the branch and drop each other's updates. Job-level\nconcurrency is independent of the existing workflow-level groups, so redundant-run\ncancellation is unchanged.\n\n* ci: bump artifact actions off deprecated Node 20 (@v4 -> @v7/@v8)\n\nactions/upload-artifact@v4 and actions/download-artifact@v4 run on the\ndeprecated Node 20 runtime. Bump the remaining @v4 pins to the versions already\nused elsewhere in the repo (upload-artifact@v7, download-artifact@v8), which run\non Node 24.\n\nAll affected download steps fetch a single named artifact to an explicit path,\nso behavior is unchanged across the major bump; v4-v8 share the same artifact\nbackend.\n\n* test: align per-req-cache and connector-metadata tests with current behavior\n\nThe per-req-cache tests asserted a removed design where stateful requests\ndeducted 'equiv blocks' from the KV pool and were tracked in a\nper_req_cache_accounting dict. The current BlockManager sizes the state\ntensor separately and excludes it from num_kvcache_blocks, so admission only\nclaims a free slot index with no extra paged-block cost. Rewrite the seven\nstale tests to the slot-only model (can_allocate returns -1/hit-count, not\nFalse/bool) and rename two to match what they now verify.\n\nConnectorMetadata._build_req_meta parses transfer params leniently via\ndict.get, so a missing field yields None instead of raising KeyError. Update\nthe connector-metadata test accordingly.\n\n* test: make non-unit disaggregation tests skip visibly off the unit path\n\ntest_proxy gains importorskip guards for its optional msgpack/quart deps, so\nit runs where they are installed and skips with a reason otherwise instead of\nerroring at collection.\n\ntest_transfer_engine and test_kv_connector_scheduler import the\nkv_transfer_engine module that #690 split into the moriio subpackage; guard\nthem with importorskip so they skip visibly (with a reason pointing at the\nneeded path update) until the disaggregation owner refreshes them.\n\nDelete test_kimi_k25: it exec-loads the real atom/config.py at import time,\nwhich collides with conftest's atom package stub and cannot run under the\nshared unit harness.\n\n* test: remove obsolete mxfp4 swiglu source-introspection test\n\ntest_swiglu_branch_condition_no_bias_check asserted that\nMxfp4MoEMethod.process_weights_after_loading contains a literal\n'layer.activation == ActivationType.Swiglu:' branch. That function was\nrefactored to route via use_triton vs the AITER shuffle path, so the branch\nno longer exists in that form and the test had been @unittest.skip'd as\nobsolete. Drop it; the sibling test_swiglu_branch_does_not_couple_bias_and_shuffle\nstill guards against the original coupled-condition regression.\n\n* ci: add non-GPU unit test gate to pre-checks\n\nRun the native unit suite on ubuntu-latest as part of Pre Checkin, alongside\nblack/ruff/validate-catalog. .github/scripts/run_unit_tests.sh centralizes the\nscope: it runs tests/ minus tests/plugin (next-stage sglang/vllm/rtpllm work,\nwhich also installs import-time sys.modules stubs that would pollute native\ntests) and minus the GPU server integration test; P/D disaggregation tests\nself-skip via importorskip guards. The job installs CPU torch + base deps,\nemits a JUnit report, and uploads it as an artifact.\n\nLocally: 464 passed, 2 skipped, 0 failed.\n\n* test: fix unit gate failures on the non-GPU runner\n\nThe new pre-checks unit job failed on ubuntu (no aiter, no PIL) for two\nreasons, both now fixed:\n\n- test_api_server_helpers leaked stub modules. When the api_server import\n  fails (PIL absent), the except branch reset _injected_modules to [] before\n  the finally cleanup ran, so the injected stub for atom.model_engine.arg_utils\n  was never popped from sys.modules. It then shadowed the real EngineArgs for\n  test_arg_utils_spec (collected later), which failed with _StubEngineArgs /\n  missing SpeculativeConfig. Drop the reset so finally always tears the stubs\n  down, and pre-initialize _injected_modules so finally is safe if stub\n  installation itself raises. Verified by blocking PIL locally: arg_utils tests\n  pass, api_server tests skip cleanly.\n\n- test_mxfp4_moe_has_bias loads atom.config / atom.model_ops.moe, which import\n  the AITER GPU kernel library (no CPU build). Guard the module with\n  pytest.importorskip('aiter') so it skips visibly off the non-GPU gate and\n  runs in GPU CI.\n\n* ci: checkout repo in download_aiter_wheel jobs\n\nThe download_aiter_wheel jobs in atom-test and atomesh-accuracy-validation\nhave no checkout step — the original inline bash ran from the YAML directly.\nDe-inlining the logic into .github/scripts/download_aiter_wheel.sh introduced a\ndependency on the file being present on the runner, so the jobs failed with\n'No such file or directory' (exit 127). Add actions/checkout@v6 to both jobs.\n\n* ci: drop literal ${{ }} from docker-auth description\n\nGitHub evaluates ${{ }} expressions in an action's description field, and the\nsecrets context is not available to composite actions. The description quoted\nthe inline secret-interpolation form verbatim with braces, so loading the\ncomposite failed at runtime with 'Unrecognized named-value: secrets',\nshort-circuiting Docker Login in atom-test/atomesh. Reword without braces.\n\nactionlint does not evaluate description expressions, so this only surfaced on\na real runner.\n\n* ci: clone aiter with full history so its version isn't 0.0.0\n\nThe image build shallow-cloned aiter (git clone --depth 1), so its\nsetuptools_scm version fell back to 0.0.0 (no tags reachable), making the\nbaked-in aiter indistinguishable by version. Use --filter=blob:none instead:\nfull commit history + tags (so setuptools_scm computes a real version) while\ndeferring blob downloads to keep the clone fast. Submodule init is unaffected.\n\nNative workflows only (atom-test, atomesh-accuracy-validation); the sglang/vllm\nbenchmark workflows have the same shallow clone but are out of scope for now.\n\n* ci(benchmark): print the full benchmark command before running\n\nBuild the benchmark_serving invocation as a bash array and printf it right\nafter 'Running benchmark test', so the exact resolved command (model, ISL/OSL,\nconcurrency, extra args) is visible in the client log. Running the array\nguarantees the printed command matches what executes.\n\n* ci: notify Teams on nightly/release workflow failure\n\nAdd a workflow_run listener that posts a Teams message when a native scheduled\nworkflow fails (ATOM Test, ATOM Benchmark, Atomesh Accuracy Validation, Pre\nCheckin, Nightly Docker Release). Single listener instead of per-workflow steps\n— zero changes to the targets. Filtered to conclusion==failure and\nevent==schedule so only nightly/release runs notify, not PRs.\n\nPosts an Adaptive Card (built with jq; run metadata passed via env to avoid\ntemplate injection) to a Teams 'Post to a channel when a webhook request is\nreceived' Workflows webhook — classic O365 connector Incoming Webhooks were\nretired in 2026. Requires a TEAMS_WEBHOOK_URL repo secret; until it's set the\njob no-ops without failing. workflow_run fires from the default-branch copy, so\nit activates after merge.\n\n* fix(ci): unindent resolve_download_url python so the S3 fast-path works\n\nThe python3 -c body in download_aiter_wheel.sh indented its continuation lines\nto match the bash block, putting leading whitespace inside the single-quoted\nsource -> 'IndentationError: unexpected indent'. resolve_download_url is called\nunder a non-set-e context (download_from_s3_manifest), so the error was swallowed\nand the S3 manifest fast-path silently fell back to artifact enumeration every\nrun. Move the python body to column 0 (leading newline) so it parses.\n\n* ci: serialize native accuracy-dashboard gh-pages pushes\n\nThe gh-pages serialization added the gh-pages-deploy concurrency group to the\ndocs/benchmark deployers but missed two native jobs that also auto-push to\ngh-pages: atom-test 'Update accuracy dashboard' and atomesh 'Publish Atomesh\naccuracy data'. Add the same group so their auto-push can't race the serialized\ndeploys on the gh-pages branch.",
          "timestamp": "2026-06-28T16:32:45Z",
          "url": "https://github.com/ROCm/ATOM/commit/7058d7085ff2d1b3ca317d7ad1f49839e8348a0e"
        },
        "date": 1782668431704,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28328779853 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202606271512 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9469 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7475,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28328779853 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202606271512 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7475 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "04b120d3040e0dd1a6915e88a3a3c1f588e6684a",
          "message": "fix(dsv4): gate Triton batched_gemm_bf16 to gfx1250, einsum fallback elsewhere (#1433)\n\n* fix(dsv4): gate Triton batched_gemm_bf16 to gfx1250, einsum fallback elsewhere\n\nThe grouped output-LoRA BMM (PR #1270) unconditionally used the Triton\nbatched_gemm_bf16 kernel, which is only tuned/enabled on gfx1250. On other\narchs (e.g. gfx950 / MI355X) fall back to the original\ntorch.einsum(\"sgd,grd->sgr\") path.\n\n* fix: add missing get_gfx import (F821)",
          "timestamp": "2026-07-01T15:08:46Z",
          "url": "https://github.com/ROCm/ATOM/commit/04b120d3040e0dd1a6915e88a3a3c1f588e6684a"
        },
        "date": 1782927797107,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28533812486 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202607011530 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9439 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7498,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28533812486 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607011530 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7491 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zhangxinyuanliuhengyu",
            "username": "zhangxinyuanliuhengyu",
            "email": "xinyuazh@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "20c8096bb6b8fe294cdf97d6b0700ca6abb0c075",
          "message": "[fix][sgl-atom] avoid Qwen3-32B GSM8K truncation (#1489)\n\n* [fix][sgl-atom] avoid Qwen3-32B GSM8K truncation\n\n* fix accuracy catalog schema for lm-eval args\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: xinyuazh <xinyuazh@hjbgo-srdc-16.amd.com>\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-07-07T08:15:43Z",
          "url": "https://github.com/ROCm/ATOM/commit/20c8096bb6b8fe294cdf97d6b0700ca6abb0c075"
        },
        "date": 1783413462806,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.956,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28809207480 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607061543 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.956 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.81,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28809207480 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607061543 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.956 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7559,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28809207480 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607061543 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7544 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8825,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28809207480 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202607061543 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3806 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qichu@amd.com",
            "name": "qichu-yun",
            "username": "qichu-yun"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "760aef8ba706be8f79adb163d9b4d3be71bcd318",
          "message": "[Benchmark] Update Kimi2.7 recipe and workflow (#1478)",
          "timestamp": "2026-07-07T17:33:53+08:00",
          "tree_id": "3efc3e63eb561bac44f1cc95627ced8a2d79699f",
          "url": "https://github.com/ROCm/ATOM/commit/760aef8ba706be8f79adb163d9b4d3be71bcd318"
        },
        "date": 1783419244073,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28856306012 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202607061543 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9439 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7513,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28856306012 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607061543 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7483 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Wang, Yiting",
            "username": "yitingw1",
            "email": "yitiwang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ddb38b04ee8dbd9557fc3887b6d86041fc7d192d",
          "message": "[Fix] fix GLM5.2 n-shot100 accuracy (#1502)\n\n* [Fix] fix GLM5.2 n-shot100 accuracy\n\n* [Fix] preserve DeepSeek V3.2 neox indexer rope; clean up helper semantics\n\nThe prior fix routed the indexer rope through _is_neox_rope_style with a\ndefault of True. For configs that omit indexer_rope_interleave (DeepSeek\nV3.2), that resolved to is_neox_style=False, silently flipping V3.2's\nindexer from neox (its validated layout) to interleaved. GLM-5.x declares\nthe flag so it was unaffected, but V3.2 would regress at prompts longer\nthan index_topk (the only case that exercises the sparse indexer top-k).\n\nAlso make _is_neox_rope_style read directly:\n  - the default is now expressed in the same vocabulary as the return value\n    and the call site's assignment target: default_interleave -> default_is_neox\n    (keyword-only), so no reader has to mentally invert.\n  - the missing/null flag now early-returns default_is_neox, leaving the lone\n    `not bool(interleave)` to do exactly one thing: convert a present interleave\n    flag to is_neox_style. No `not` tangled with the default fallback.\n  - per-rope fallbacks (main=interleaved so default_is_neox=False; V3.2 indexer\n    =neox so default_is_neox=True) are passed inline with a comment on each.\n\nNet behavior: V3.2 indexer stays neox; GLM indexer stays interleaved.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: zejunchen-zejun <zejun.chen@amd.com>\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-07T13:54:27Z",
          "url": "https://github.com/ROCm/ATOM/commit/ddb38b04ee8dbd9557fc3887b6d86041fc7d192d"
        },
        "date": 1783446348656,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9447,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28883623174 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202607071606 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9447 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7506,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28883623174 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607071606 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7475 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8802,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28883623174 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202607071606 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3442 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yajizhan@amd.com",
            "name": "jasen",
            "username": "Jasen2201"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c069049670db6917f80d98b2e35bd7ad47313ce0",
          "message": " Remove legacy proxy, update docs, and enhance scripts (#1447)\n\n* refactor(pd): remove legacy Python proxy and ZMQ service discovery\n\nThe old `proxy.py` routing proxy is superseded by atomesh. Remove it\nalong with the `_service_discovery_ping` heartbeat threads in both\nmooncake and moriio connectors that registered with it. The\n`proxy_ip` / `proxy_ping_port` config keys are still accepted but\nsilently ignored, so existing kv-transfer-config payloads keep working.\n\n* chore(mesh): remove obsolete benchmark scripts\n\nKeep only the 7 scripts referenced by scripts/README.md (docker_start,\nstart_prefill, start_decode, start_router, run_gsm8k, run_benchmark,\nds_fp8_1p_tp4_1d_tp8_slurm). The rest are superseded by the weekly\nbenchmark harness and per-model slurm scripts outside the repo.\n\n* feat(mesh): add RDMA NIC auto-detection and SO mapping to docker_start.sh\n\nAuto-detect bnxt/ionic/mlx5 NICs and mount the correct host ibverbs\nprovider libraries into the container. Also adds /dev/infiniband,\n--shm-size 128G, and post-start TCP backlog tuning. Verified on\nionic (mia1-p02-g42).\n\n* docs: replace legacy proxy with atomesh in all PD guides\n\nUpdate kv-transfer-config examples to drop proxy_ip/proxy_ping_port\nand add atomesh router launch steps. Affected docs:\n- recipes/pd_disaggregation_guide.md\n- recipes/DeepSeek-V4.md\n- recipes/mesh/multi-node-atom.md\n- atom/kv_transfer/disaggregation/README.md\n\n* docs(pd): replace build-from-source with docker setup in PD guide\n\nRemove the Mooncake build-from-source section (67 lines) and replace\nwith docker pull + docker_start.sh which handles RDMA NIC detection\nand SO mounting automatically.\n\n* docs: add MiniMax-M3 PD disaggregation recipe\n\nSingle-node 1P+1D setup (GPU 0-3 prefill, GPU 4-7 decode, TP=4 each)\nwith atomesh router. Covers base MXFP4, EAGLE3 variant, GSM8K accuracy\nvalidation, and serving benchmark — all via the router endpoint.\n\n* docs: add MiniMax-M3 multi-node 2P+1D DPA+TBO recipe\n\n3-node setup: 2 prefill instances (TP=4, DPA + TBO) + 1 decode instance\n(TP=4, DPA, max-num-seqs=1024). Covers atomesh router with dual\n--prefill flags, GSM8K accuracy, and high-concurrency benchmark sweep.\n\n* docs: add online_quant_config to MiniMax-M3 PD recipes\n\nAdd --online_quant_config to all PD server commands (1P1D and 2P1D).\nInclude a reference table documenting the exclude_layer differences\nbetween MXFP4/MXFP8 and TP-only/DPA modes.\n\n* docs: fix online quant config section title to be mode-agnostic\n\n* refactor(pd): remove dead proxy_ip/proxy_ping_port attributes from connectors\n\nThese config reads were kept for backward compatibility but the only\nconsumer (_service_discovery_ping) was already removed. Old configs\nthat pass proxy_ip still work — dict.get() silently discards the value.\n\n* chore: rename default image to rocm/atom-dev:latest and container to atom_mesh\n\nReplace mesh-sglang-latest with rocm/atom-dev:latest as the default\ndocker image. List vllm-latest and sglang-latest as alternatives in\ncomments. Rename default container from atom_sglang_mesh to atom_mesh.\n\n* chore(mesh): remove obsolete ds_fp8 slurm script and its README section\n\nThe SLURM one-shot automation is now handled by per-model scripts\noutside the repo. Remove the generic ds_fp8_1p_tp4_1d_tp8_slurm.sh\nand its corresponding documentation from the scripts README.\n\n* docs: extract MiniMax-M3 PD recipes into recipes/mesh/MiniMax-M3.md\n\nMove all PD disaggregation sections (1P+1D, 2P+1D DPA, EAGLE3 variants,\nonline_quant_config reference) from recipes/MiniMax-M3.md into a dedicated\nmesh recipe. The original file now has a cross-reference link.\n\nThe new doc covers all 8 benchmark configurations (FP4/FP8 × 1P1D/2P1D ×\nplain/EAGLE3), distilled from the weekly_mesh_benchmark scripts.\n\n* docs: add DeepSeek-V4-Pro PD mesh recipe\n\nDistill the 5 weekly benchmark scripts (1P1D TP, 1P1D MTP, 2P1D DPA,\n2P1D MTP DPA, 2P1D MTP3 DPA) into a single user-facing recipe at\nrecipes/mesh/DeepSeek-V4.md covering all topologies with MTP as an\nadd-on section.\n\n* docs: drop MTP-3 DPA variant from DeepSeek-V4 mesh recipe\n\n* docs: replace inline PD section in DeepSeek-V4 recipe with mesh link\n\n* docs: replace obsolete ATOM_CPU_AFFINITY with ATOM_NUMA_BIND in DeepSeek-V4 recipe\n\nATOM_CPU_AFFINITY was the old NUMA-blind linear CPU slice, replaced by\nthe topology-aware ATOM_NUMA_BIND. The old var is not read by any code.",
          "timestamp": "2026-07-08T10:10:54+08:00",
          "tree_id": "d540d0cfb2e6de6311714bd6f57dcc61b210d162",
          "url": "https://github.com/ROCm/ATOM/commit/c069049670db6917f80d98b2e35bd7ad47313ce0"
        },
        "date": 1783478172877,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28912426945 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202607071606 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7604,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/28912426945 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607071606 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7604 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "dd64a5f1ae818fd81bec48d8b087d175c68f446f",
          "message": "fix(dsv4): align TBO v4_batch_id_per_token buffer to int32 (#1526)\n\nThe ubatch (TBO) allocation of v4_batch_id_per_token was int64 while the\nnon-ubatch allocation was already int32. The staged numpy array is int32\n(np.full(..., dtype=np.int32)) and _stage() asserts arr.dtype == buffer.dtype,\nso serving with --enable-tbo hit the dtype-mismatch assertion the moment it\nstaged batch_id_per_token. Make the ubatch buffer int32 to match the array,\nthe non-ubatch buffer, and the downstream int32 consumers.\n\nint32 is correct here: batch_id is used as a PyTorch advanced index\n(cu_committed_gpu[batch_id_per_token_gpu]), which accepts int32, and the\ndownstream kernels (deepgemm_fp8_paged_mqa_logits, top_k_per_row_decode,\nflydsl SWA scatter) all want / tolerate int32.\n\nAlso fix two stale docs that contradicted the code:\n- _stage docstring said \"Auto-casts dtype\" but the body asserts dtype equality.\n- A comment claimed \"int64 batch_id is mandated by PyTorch fancy indexing\".",
          "timestamp": "2026-07-09T13:59:56Z",
          "url": "https://github.com/ROCm/ATOM/commit/dd64a5f1ae818fd81bec48d8b087d175c68f446f"
        },
        "date": 1783617762999,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29035235790 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607091539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.7,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29035235790 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607091539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7278,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29035235790 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607091539 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7278 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8855,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29035235790 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202607091539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3586 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "yadaish",
            "username": "yadaish",
            "email": "yadai@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "945b3127e7bebc722ba6c6fe198fb1c1757a3ed2",
          "message": "update shuffle weight for gfx1250 (#1540)\n\n* update shufle weight for gfx1250\n\n* address review: remove redundant gfx1250 GUGU comment\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-11T12:27:54Z",
          "url": "https://github.com/ROCm/ATOM/commit/945b3127e7bebc722ba6c6fe198fb1c1757a3ed2"
        },
        "date": 1783789482716,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9477,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29159651649 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202607101554 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9416 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7619,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29159651649 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607101554 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7597 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "02db8a5a8f1e44080f8c21f15bc6af0f90ef42ec",
          "message": "ci: re-login to Docker Hub before each push in nightly release (#1567)\n\nThe nightly release logs in once at job start, then spends ~1h building\nbefore the first push. By the time a multi-GB image push finalizes, the\nDocker Hub auth token is stale and the manifest PUT fails with\n\"unauthorized: authentication required\" -- all layers upload fine, only the\nfinalize step 401s.\n\nRe-authenticate immediately before each of the three push steps (native,\nOOT, SGLang) so every push starts with a fresh token.\n\nAlso fold the duplicated push-gating booleans into a single \"Compute push\ngates\" step whose main/oot/sglang outputs every build/re-login/push/test\nstep references, so the conditions are defined once and can't drift.",
          "timestamp": "2026-07-12T16:22:25Z",
          "url": "https://github.com/ROCm/ATOM/commit/02db8a5a8f1e44080f8c21f15bc6af0f90ef42ec"
        },
        "date": 1783875114804,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29200061531 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607101554 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.54,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29200061531 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607101554 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7513,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29200061531 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607101554 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7498 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "honglie",
            "username": "yhl-amd",
            "email": "hyi@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2fc5091948eb16d2e5cd6013eb0a8f05b781abea",
          "message": "[Frontend] Abort engine request on client disconnect (free leaked KV) (#1562)\n\n* [Frontend] Abort engine request on client disconnect (free leaked KV)\n\nNon-streaming API handlers were not cancelled when the client hung up\n(Starlette only cancels StreamingResponse, not plain handlers), so the engine\nkept generating and the sequence's KV blocks leaked until it hit max_tokens.\n\nAdd a client-disconnect abort path:\n- engine: EngineCoreMgr.abort_request broadcasts an \"abort_request\" utility\n  command; _handle_abort_request marks the seq aborted; the scheduler finishes\n  it at the next step via the normal stop path (frees KV, emits a finished\n  RequestOutput). Adds Sequence.aborted.\n- api_server: _run_nonstream_with_disconnect runs generate_async in a task and\n  polls request.is_disconnected(); on disconnect it cancels the task, whose\n  teardown aborts + pops the request. Wired into /v1/chat/completions and\n  /v1/completions (return HTTP 499 on disconnect). generate_async /\n  generate_async_multimodal / generate_async_fanout / cleanup_streaming_request\n  abort + pop on early exit to avoid leaks.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* [Frontend] Detect client disconnect via ASGI event, not polling\n\nReplace the 0.5s is_disconnected() polling loop in\n_run_nonstream_with_disconnect with vLLM-style event-driven cancellation:\nrace the generator-collector task against a task that awaits the ASGI\nhttp.disconnect event (request.receive()), FIRST_COMPLETED wins.\n\nDetection is now immediate (0ms vs up to 500ms) and costs nothing while\nthe client stays connected (no periodic wakeups). The abort path is\nunchanged: cancelling the collector still propagates into generate_async's\nfinally -> abort_request + io_processor.requests.pop, freeing leaked KV.\n\nrequest.receive() is safe here because FastAPI parses the request body\ninto a pydantic model before the handler runs, so there is no unread body\nto race against.\n\nVerified on DeepSeek-V4-Pro tp8: curl --max-time drop -> immediate\n\"Client disconnected ... aborting request\" + abort_request found=True;\nnormal non-stream and streaming requests unaffected.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nSigned-off-by: yihonglie <hyi@amd.com>\n\n* [Frontend] Extend disconnect-abort to fan-out/multimodal; narrow cancel except\n\nAddress review of the client-disconnect abort:\n\n- Factor the disconnect race into `_race_disconnect(coro, ...)` and make\n  `_run_nonstream_with_disconnect` a thin wrapper that collects the\n  async-generator. This lets the fan-out path (whose `generate_async_fanout`\n  is a coroutine returning a list, not an async generator) reuse the same\n  cancellation machinery.\n\n- Wrap the previously-unguarded non-stream branches so an abandoned request\n  is aborted instead of running to max_tokens: multimodal (chat), n>1 fan-out\n  (chat, both multimodal and text), and n>1 fan-out (completions).\n  generate_async_fanout's try/finally aborts every sibling on cancel, so a\n  disconnect frees all n sibling seqs.\n\n- Narrow the post-cancel teardown handler from `except BaseException` to\n  `except asyncio.CancelledError` (expected) + `except Exception` (logged),\n  letting KeyboardInterrupt/SystemExit propagate.\n\nVerified on DeepSeek-V4-Flash tp4: plain and n=2 fan-out non-stream drops both\nlog \"Client disconnected ... aborting\" with abort_request found=True for every\nsibling (2/2 for n=2); normal plain, n=2 fan-out, and streaming unaffected.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nSigned-off-by: yihonglie <hyi@amd.com>\n\n* style: apply black formatting to api_server.py\n\nFixes the \"Check Code Style with Black\" CI check on the disconnect-abort\nchanges (extra blank lines, single-line logger call).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nSigned-off-by: yihonglie <hyi@amd.com>\n\n* [Frontend] Represent client-disconnect abort as SequenceStatus.ABORTED\n\nAddress review: fold the parallel `Sequence.aborted` boolean into the\nexisting `SequenceStatus` enum so a sequence's terminal intent has a\nsingle source of truth.\n\n- sequence: add SequenceStatus.ABORTED (kept distinct from FINISHED so an\n  aborted running seq still rides one cleanup pass; is_finished() stays\n  False until then); drop Sequence.aborted.\n- engine_utility: _handle_abort_request sets seq.status = ABORTED.\n- scheduler (running): finish check reads seq.status == ABORTED.\n- scheduler (waiting): intercept ABORTED seqs when popped from `waiting`,\n  BEFORE the waiting->running promotion overwrites status with RUNNING and\n  loses the abort intent. Such seqs hold no KV and need no forward pass, so\n  finish them outright via `_rejected` (mirrors the unschedulable exit).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nSigned-off-by: yihonglie <hyi@amd.com>\n\n---------\n\nSigned-off-by: yihonglie <hyi@amd.com>\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-13T12:41:04Z",
          "url": "https://github.com/ROCm/ATOM/commit/2fc5091948eb16d2e5cd6013eb0a8f05b781abea"
        },
        "date": 1783963219037,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29268178884 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607121715 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.33,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29268178884 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607121715 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7498,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29268178884 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607121715 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7498 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/ROCm/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784140948256,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7551,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29433095078 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607151539 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7544 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.881,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29433095078 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202607151539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3055 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Phi-C",
            "username": "Phi-C",
            "email": "chenxjhit@163.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "9bed95e4e48ac9a4eca2c0298e975298554c35a5",
          "message": "Fix: support agentic dataset benchmark under PD disaggregation mode (#1586)\n\n* Fix: support agentic dataset running under PD disaggregation mode\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* add recipe\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* fix pre-checkin\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* add this case to nightly ci\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* fix ci nightly case\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* unify an interface to get kv_connector's flag\n\nSigned-off-by: Phi-C <chenxjhit@163.com>\n\n* Revert agentic PD M3 CI changes\n\n---------\n\nSigned-off-by: Phi-C <chenxjhit@163.com>",
          "timestamp": "2026-07-16T15:26:58Z",
          "url": "https://github.com/ROCm/ATOM/commit/9bed95e4e48ac9a4eca2c0298e975298554c35a5"
        },
        "date": 1784222682156,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9462,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29516182338 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202607151539 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9416 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7536,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29516182338 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607151539 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7544 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "GnSight",
            "username": "ftyghome",
            "email": "ftyg@live.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2bc0c44bcf40a99d9ef59a23f300ec44e5e66eec",
          "message": "perf: optimize model loading speed (#1465)\n\n* perf: optimize model loading speed\n\n* chore: model load opt env var & cleanups\n\n* fix: eliminate redundant mem pin\n\n* docs: update expert parallel loading docs\n\n* style: remove unused var\n\n* ci: flip disable-mmap default to false\n\nSuggested-by: Lingpeng Jin <103567126+valarLip@users.noreply.github.com>\n\n* Revert \"ci: flip disable-mmap default to false\"\n\nThis reverts commit cb03a8058513f358380403a53fa1c2c990191367.\n\n---------\n\nCo-authored-by: Lingpeng Jin <103567126+valarLip@users.noreply.github.com>",
          "timestamp": "2026-07-17T16:35:54Z",
          "url": "https://github.com/ROCm/ATOM/commit/2bc0c44bcf40a99d9ef59a23f300ec44e5e66eec"
        },
        "date": 1784309304402,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29596527373 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607170233 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.95 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.64,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29596527373 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607170233 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.95 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7384,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29596527373 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607170233 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7339 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8886,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29596527373 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202607170233 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.4443 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zejun.chen@amd.com",
            "name": "zejunchen-zejun",
            "username": "zejunchen-zejun"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "01081aa3405a6004226823dc0724b0398b52baaf",
          "message": "[gptoss] remove AITER_USE_FLYDSL_MOE_SORTING (#1674)\n\n[glm5.2 fp8] change the online quant command\n\nSigned-off-by: zejunchen-zejun <zejun.chen@amd.com>",
          "timestamp": "2026-07-23T13:50:19+08:00",
          "tree_id": "3d74af006c4d41e40b474b667d6b94822a0ed9ea",
          "url": "https://github.com/ROCm/ATOM/commit/01081aa3405a6004226823dc0724b0398b52baaf"
        },
        "date": 1784787308889,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29983241024 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202607221602 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9439 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7574,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/29983241024 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607221602 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7574 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "PerryZhang01",
            "username": "PerryZhang01",
            "email": "Perry.Zhang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "493ddb2048a996efafe0dba2745c3af2014f4597",
          "message": "[fea](req): change transformers version to 5.12.1 (#1669)\n\n* [fea](req): change transformers version to 5.12.1\n\n* [fix](qwen): fix qwen config\n\n---------\n\nCo-authored-by: perzhang <perzhang@amd.com>",
          "timestamp": "2026-07-24T12:10:36Z",
          "url": "https://github.com/ROCm/ATOM/commit/493ddb2048a996efafe0dba2745c3af2014f4597"
        },
        "date": 1784913687723,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30109791889 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607241525 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.55,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30109791889 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202607241525 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7453,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30109791889 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202607241525 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7468 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8779,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30109791889 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202607241525 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3927 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "767212315a4b6de0e951f9e4c36974e66cd47be1",
          "message": "fix(engine): offline output thread dies on the first streamed token (#1768)\n\n* fix(engine): give both core managers one shared-state initialiser\n\n`DisaggCoreManager` spawns its engines differently and so cannot run\n`CoreManager.__init__`. It hand-copied the field block instead, and the copy\ndrifted: `_flush_stream_batch_fn` was added to the copy and to the API server\nthat assigns it, but never to the base class.\n\nNothing failed loudly. The server path assigns the hook before the first\nrequest and the disagg path had it from the copy, so only the offline\nentrypoint was left with neither. There the output thread raised\n\n    AttributeError: 'CoreManager' object has no attribute '_flush_stream_batch_fn'\n\non the first streamed token and died. The engine kept producing tokens that\nnobody collected, `simple_inference` blocked forever, and the nightly Docker\nsmoke test reported it a day later as a 60-minute step timeout.\n\nExtract `_init_shared_state()` and have both constructors call it, so there is\none place to add the next such field. Only `pp_*` (read solely by this class's\nspawn loop, and `DisaggCoreManager` overrides the one method that touches it)\nand the post-spawn tail stay per-class.\n\nThe tests pin the invariant rather than the single field: whatever\n`process_outputs_socket` reads off `self` -- discovered from the source, so a\nnewly added read is covered the day it lands -- both managers must provide.\n\n* test: import the real atom modules instead of hand-written stand-ins\n\n`tests/conftest.py` replaced `atom` and `atom.config` in `sys.modules` with\nhand-built module objects, so unit tests ran against a parallel API that\nnobody updated. It rotted: the stand-in lost `CompilationLevel`, and because\nthe modules that import it are guarded by a broad try/except that skips the\nwhole file, four test modules stopped running -- on every machine, not just\nCI -- while reporting a circular import that does not exist. One of them,\n`test_dspark_swa_fp8_2buff`, has two real tests that now run again.\n\nThe stand-ins existed because `import atom.config` was expensive: Python\nimports a package before its submodule, and `atom/__init__.py` eagerly\nimported `LLMEngine`, so reading a dataclass pulled in zmq, the model runner\nand AITER. Resolve `LLMEngine` on attribute access instead (PEP 562);\n`SamplingParams` and `prepare_model_for_sglang` stay eager, as both cost only\ndataclasses, logging and typing. `import atom.config` now loads no engine\nmodule at all, and `from atom import LLMEngine` is unchanged for callers.\n\nThat leaves AITER as the only thing a plain CPU runner lacks, and the config\nchain reaches exactly two of its attributes (`QuantType` and\n`utility.dtypes.d_dtypes`). Stub that -- an external boundary, conditionally,\nnext to the existing zmq and xxhash stubs -- and drop all four internal ones;\nthe `forward_context` and `custom_register` stand-ins turned out to be\nunnecessary too.\n\n* ci: run offline inference on the PR gate\n\nBoth the simple-inference step and its golden comparison were `if: false`, so\nthe offline entrypoint was exercised only by the nightly Docker release. That\nis how a bug that killed its output thread reached main: the accuracy test\ndrives the server over HTTP, and the two do not share the path that hands\ntokens back to the caller.\n\nEnable the inference step for every model in the matrix. Completing is the\nassertion -- the output is printed but not diffed, which is what the failure\nmode calls for (the bug hung, it did not produce wrong text) and what removes\nthe need for a per-model golden file. Only one such file was ever checked in,\nwhich is why the comparison was disabled wholesale when the matrix moved to\nmodels_accuracy.json; drop that step rather than leave it dead a second time.\n\nThe step timeout is what turns the hang this guards against into a failure\ninstead of a stuck runner.\n\n* refactor(config): drop the import-time AITER dependency, and every test stub\n\n`atom/config.py` reached AITER for two things: `QuantType.No` when a\ncheckpoint carries no quantization config, and one return annotation. Because\nPython imports a package before its submodule, that put a GPU kernel build\nbehind `import atom.<anything>` -- so the unit suite replaced `atom.config`\nwith a hand-written stand-in, which then drifted from the real class and\nsilently stopped four test modules from running anywhere.\n\n`atom.quant_spec` now resolves `QuantType` and `d_dtypes` on first use instead\nof at import. The handles return the genuine AITER objects, so values compare\nand behave exactly as before; only the lookup moves. Deferring the import is\nnot enough on its own -- three places evaluated those names while the module\nbody ran, and each had to move with it:\n\n  - the `LayerQuantConfig.quant_type` default (class creation) -> default_factory\n  - `_QSCHEME_TO_QUANT_TYPE` (module level) -> a cached function\n  - `GenericParser._QTYPE_PATTERNS` (class body) -> the same\n\n`config.py` then needs neither: the one runtime use has an exact equivalent in\n`LayerQuantConfig.no_quant()`, and the annotation moves under `TYPE_CHECKING`.\n\nWith that, `tests/conftest.py` fakes nothing at all. The zmq and xxhash stubs\nwent too -- both are declared dependencies in pyproject.toml, so their\n`find_spec(...) is None` guard could never fire.\n\n`test_quant_config.py` installed its AITER stand-ins only while it exec'd the\nmodule body, which no longer covers a first-use lookup; it now binds the\nhandles before dropping them. Deliberately not left in `sys.modules`: a\nlingering fake `aiter` satisfies `pytest.importorskip(\"aiter\")` in the other\ntest modules, and whether it did would depend on collection order.\n\nSimulating the CPU gate (aiter blocked at the meta-path):\n\n    before   839 passed, 16 skipped, 0 failed\n    after    846 passed, 14 skipped, 0 failed\n\nand on a GPU host, 895/6 -> 902/2. The 2 remaining failures\n(`test_mxfp4_moe_has_bias`) are pre-existing on main.\n\n* fix(ci): keep JSON in extraArgs intact for the simple-inference step\n\n`${{ matrix.extraArgs }}` was interpolated into a double-quoted\n`bash -lc \"…\"`, so the *host* shell stripped the inner quotes before the\ncontainer ever saw them: `--hf-overrides '{\"use_index_cache\": true}'` arrived\nas `'{use_index_cache: true}'`, which argparse rejects as JSON. Four of the 13\nPR-level models carry JSON there, and more of the full matrix do via\n--eplb-config, --dspark-config and --online_quant_config.\n\nPass it through an env var and pipe the command over stdin, which is what the\naccuracy step below already does for the same reason.\n\n* style: annotate GenericParser._DTYPE_PATTERNS as ClassVar\n\nRUF012. The dict is unchanged; moving _QTYPE_PATTERNS out from under it\nbrought it into the reviewdog diff context, which is what surfaced this.\n\n* fix(v4): import QuantType from aiter, not through atom.config\n\nMoving `QuantType` under `TYPE_CHECKING` in `atom/config.py` bound nothing at\nrun time, and `deepseek_v4.py` was the one module importing it *through* the\nconfig hub rather than from `aiter` directly:\n\n    ImportError: cannot import name 'QuantType' from 'atom.config'\n\nEvery other model file already does `from aiter import QuantType`, and this\none imports four other aiter modules anyway, so it matches them now.\n\nNothing caught this. `atom/models/*` needs the AITER build, so the CPU gate\nnever imports it, and a broken re-export surfaces only when someone loads that\nmodel on a GPU host. The new test closes that: it walks the tree for\n`from atom.config import X` and checks each name against what config.py binds\nat its top level -- `TYPE_CHECKING` blocks deliberately excluded, since that\nis the distinction that broke here. Pure AST, imports nothing, so it runs on\nthe gate that cannot import the modules it protects. It fails with\n`atom/models/deepseek_v4.py:52 imports 'QuantType'` against the broken tree.\n\n* Revert \"ci: run offline inference on the PR gate\" and its quoting fix\n\nWrong shape: it bolted simple_inference onto all 13 accuracy matrix jobs\nrather than guarding the docker-build path, which is where the smoke test\nthat actually caught the bug lives. Restores atom-test.yaml to main; the\nguard lands as its own job instead.\n\n* ci: smoke-test offline inference on every PR\n\nThe offline entrypoint was exercised only by the nightly Docker release, which\nruns it once after building the image. That is how a bug that killed its\noutput thread reached main: it surfaced a day later, as a 60-minute step\ntimeout. The accuracy matrix is not a substitute -- it drives the server over\nHTTP, and the two do not share the code that hands tokens back to the caller.\n\nA standalone job now runs `simple_inference` once, on\nMeta-Llama-3-8B-Instruct. It is deliberately outside `ci-gate`, which only\nopens on an approval or a ci:* label, so this really does run per PR; the\nmodel is the same one the nightly smoke test uses and the smallest in the\nmatrix.\n\nCompleting is the assertion -- the failure mode is a hang or a crash, not\nwrong text, so there is no golden file to maintain per model, and the step\ntimeout is what turns a hang into a failure instead of a stuck runner. One\nsmall model exercises this code path as well as thirteen would, and the job\nreuses the existing setup-gpu-container action rather than rebuilding the\nrelease image, which takes up to three hours.\n\n* ci: bound the smoke inference step at 15 minutes\n\nAn 8B model loads and answers a few prompts in a couple of minutes, and the\nfailure this step exists to catch is a hang. 15 minutes leaves ample headroom\nwhile keeping a hang from burning a GPU runner for half an hour.",
          "timestamp": "2026-08-01T15:07:22Z",
          "url": "https://github.com/ROCm/ATOM/commit/767212315a4b6de0e951f9e4c36974e66cd47be1"
        },
        "date": 1785603738910,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9424,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30708073611 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608011517 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9409 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7688,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30708073611 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608011517 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7657 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8878,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30708073611 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608011517 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3017 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "gbyu-amd",
            "username": "gbyu-amd",
            "email": "Guanbao.Yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2da53fb2abe1e415d229bfb62a7152f33cfc6b74",
          "message": "[fix] bunch of fixes to pass Kimi-K3 KVV  (#1727)\n\n* Port Kimi-K3 serving support onto main's registry parser framework\n\nRe-implement K3 OpenAI-serving support on top of main's refactored\ntool_parser registry and dialect-driven reasoning engine, replacing the\npre-refactor scaffolding.\n\nTool parsing:\n- Add KimiK3Parser (tool_parser/kimi_k3_tool_parser.py) implementing the\n  <|open|>call tool=\"...\" channel format. K3 carries per-argument\n  type=\"...\" on the wire, so it self-coerces without the request schema;\n  it buffers the whole output and parses at flush since K3 interleaves\n  think/response/tools sections.\n- Register K3 first in registry._DETECT_ORDER and sniff_stream (its\n  channel tokens are disjoint from every other format and it strips its\n  own framing from plain answers).\n\nReasoning:\n- Restore dialect-driven reasoning (reasoning.py + reasoning_dialects.py)\n  with CHANNEL_* tokens and ReasoningFilter.starts_thinking so a prompt\n  that begins mid-thought seeds the streaming filter's thinking state.\n\nProtocol / serving:\n- protocol.py: add response_format, reasoning_effort, thinking fields;\n  preserve dynamically-declared \"tools\" through to_template_dict.\n- serving_chat.py: normalize engine finish reasons (stop_<id> -> stop,\n  max_tokens -> length); honor tool_choice=\"none\" (drop tool calls) and\n  thread tool_choice / starts_thinking through streaming + non-streaming.\n- api_server.py: validate tools / tool_choice / response_format shape;\n  resolve thinking/effort (thinking > reasoning_effort precedence) and\n  forward response_format / tool_choice / thinking into the chat template.\n\n* style: apply black formatting to K3 serving changes\n\nCollapse over-conservative multi-line raise ValueError() calls in the\nchat request validation helpers to single lines (black, 88 cols).\n\n* style: satisfy ruff (pyupgrade) within PR diff context\n\nCI runs ruff via reviewdog in diff_context mode, which flags findings on\nchanged lines *and* their surrounding context. Modernize type annotations\n(UP006/UP007/UP035/UP045) in protocol.py and serving_chat.py so no ruff\nfinding lands inside the PR diff context, and suppress the intentional\nTRY004 on validator ValueErrors in api_server.py (ValueError is required\nso the handler maps to HTTP 400, not 500).\n\n* refactor: move K3 chat request validation/thinking out of api_server\n\napi_server.py is the ASGI wiring layer, not a home for chat-specific\nutilities. Move validate_chat_request / resolve_thinking (and their\nprivate helpers _validate_one_tool / _validate_tool_list and the\ntool-name/effort/tool-choice constants) into serving_chat.py, which\nalready owns the chat request/response helpers and imports both protocol\nand reasoning. api_server now imports the two public entry points and\ndrops its now-unused `re` import. Behavior is unchanged.\n\n* refactor: relocate constants out of serving_chat request-validation\n\nserving_chat.py is the generic chat handler and should not hardcode\nmodel-specific or spec-level constants:\n\n- _K3_TEMPLATE_EFFORTS (model-specific) becomes a per-dialect\n  ReasoningDialect.template_efforts field in reasoning_dialects.py, and\n  reasoning.py derives VALID_TEMPLATE_EFFORTS as the union across loaded\n  dialects (mirroring the existing marker-table derivation). This keeps\n  all model-specific knowledge in the dialect registry, matching the\n  module's \"add a model = add one entry\" design.\n- TOOL_CHOICE_VALUES and TOOL_NAME_RE (generic OpenAI spec, not\n  model-specific) move to protocol.py alongside the other request-spec\n  constants.\n\nserving_chat now imports these; its `re` import is dropped. Behavior\nunchanged.",
          "timestamp": "2026-08-03T16:54:13Z",
          "url": "https://github.com/ROCm/ATOM/commit/2da53fb2abe1e415d229bfb62a7152f33cfc6b74"
        },
        "date": 1785781654277,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30834224935 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608021516 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9538 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.55,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30834224935 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608021516 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9538 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7566,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30834224935 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608021516 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7551 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "junyyang-amd",
            "username": "junyyang-amd",
            "email": "junyyang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a7b1f5702afbb70e155e767a4ae59d509d1b28c7",
          "message": "[atomesh-benchmark] Remove pre-checkout cleanup (#1787)\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-04T09:03:50Z",
          "url": "https://github.com/ROCm/ATOM/commit/a7b1f5702afbb70e155e767a4ae59d509d1b28c7"
        },
        "date": 1785863107423,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9462,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30930338478 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608041536 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.63,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30930338478 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608041536 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7566,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30930338478 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608041536 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7582 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8787,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/30930338478 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608041536 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.2889 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Zhiwei",
            "username": "ZhiweiYan-96",
            "email": "yanzhw5@mail3.sysu.edu.cn"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "7e4ab0ce65fe48cba5ffe5bcf0cc1b8e2a4c31e4",
          "message": "[ATOM SGL][workflow] Add acceptance rate check (#1778)",
          "timestamp": "2026-08-05T09:42:11Z",
          "url": "https://github.com/ROCm/ATOM/commit/7e4ab0ce65fe48cba5ffe5bcf0cc1b8e2a4c31e4"
        },
        "date": 1785954952470,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7528,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31026179859 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608051535 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7521 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8817,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31026179859 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608051535 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3556 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jasen",
            "username": "Jasen2201",
            "email": "yajizhan@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "514ed7c8e5505ce72207f6f1c2a7eaaa0d36b460",
          "message": "feat(openai): accept Anthropic-style chat tools (#1810)\n\n* feat(openai): accept Anthropic-style chat tools\n\nNormalize Anthropic tool schemas at the OpenAI-compatible endpoint while preserving existing validation behavior.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* fix(ci): format chat test imports\n\nRemove the extra import-block spacing that triggers Ruff I001.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-07T15:38:06Z",
          "url": "https://github.com/ROCm/ATOM/commit/514ed7c8e5505ce72207f6f1c2a7eaaa0d36b460"
        },
        "date": 1786121582225,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197153518 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9401 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9545,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197153518 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9545 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.64,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197153518 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9545 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7528,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197153518 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X | VRAM: 252GB | ROCm: unknown | strict-match: 0.7536 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jasen",
            "username": "Jasen2201",
            "email": "yajizhan@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "514ed7c8e5505ce72207f6f1c2a7eaaa0d36b460",
          "message": "feat(openai): accept Anthropic-style chat tools (#1810)\n\n* feat(openai): accept Anthropic-style chat tools\n\nNormalize Anthropic tool schemas at the OpenAI-compatible endpoint while preserving existing validation behavior.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* fix(ci): format chat test imports\n\nRemove the extra import-block spacing that triggers Ruff I001.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-07T15:38:06Z",
          "url": "https://github.com/ROCm/ATOM/commit/514ed7c8e5505ce72207f6f1c2a7eaaa0d36b460"
        },
        "date": 1786207333271,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266294132 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9469 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266294132 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.73,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266294132 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.74,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266294132 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X | VRAM: 252GB | ROCm: unknown | strict-match: 0.7415 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jasen",
            "username": "Jasen2201",
            "email": "yajizhan@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "514ed7c8e5505ce72207f6f1c2a7eaaa0d36b460",
          "message": "feat(openai): accept Anthropic-style chat tools (#1810)\n\n* feat(openai): accept Anthropic-style chat tools\n\nNormalize Anthropic tool schemas at the OpenAI-compatible endpoint while preserving existing validation behavior.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* fix(ci): format chat test imports\n\nRemove the extra import-block spacing that triggers Ruff I001.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-07T15:38:06Z",
          "url": "https://github.com/ROCm/ATOM/commit/514ed7c8e5505ce72207f6f1c2a7eaaa0d36b460"
        },
        "date": 1786294143516,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31323129825 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608091507 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31323129825 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608091507 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.95 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.79,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31323129825 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608091507 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.95 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7491,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31323129825 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608091507 | GPU: AMD Instinct MI350X | VRAM: 252GB | ROCm: unknown | strict-match: 0.7483 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yuechao.guo@amd.com",
            "name": "Morpheus Guo",
            "username": "Yuechguo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2b61f9871e225730131e96ae0c4e21ef34653381",
          "message": "feat(dp_sticky) : add new dp_sticky policy for dp-aware routing. (#1699)\n\n* Add the dp_sticky policy to route requests with the same X-Session-ID to a\nconsistent healthy worker. New or reassigned sessions select the lowest-load\nworker, while requests without a session ID use the same load-balancing\nfallback.\n\n* feat(dp_sticky): reassign idle sessions to the lowest-load worker\n\n* feat(mesh): support dp_sticky CLI routing policy\n\n* fix(dp_sticky): preserve DP rank affinity and track worker load\n\n* fix(dp_sticky): avoid concurrent session reassignment races\n\n* update default session-id idle times\n\n* balance new sticky sessions\n\n---------\n\nCo-authored-by: yuechguo <yuechguo@amd.com>\nCo-authored-by: wanzhenchn <wanzhenchn@gmail.com>",
          "timestamp": "2026-08-10T16:50:56+08:00",
          "tree_id": "7459edde1029253ec1f4a77e3ba0d992df10c9f0",
          "url": "https://github.com/ROCm/ATOM/commit/2b61f9871e225730131e96ae0c4e21ef34653381"
        },
        "date": 1786353140107,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9447,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31371899680 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608091507 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9409 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.746,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31371899680 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608091507 | GPU: AMD Instinct MI350X | VRAM: 288GB | ROCm: unknown | strict-match: 0.7453 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "PerryZhang01",
            "username": "PerryZhang01",
            "email": "Perry.Zhang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "030b2ce9c058a441bd3ac45968a21e0c397d61ac",
          "message": "[fix](torch): fix torch errors for torch2.13 (#1846)\n\n* [fix](torch): fix torch errors for torch2.13\n\n* [fix](torch): add missing compatibility helpers and tests\n\nInclude the GPU architecture helper and regression coverage omitted from the initial Torch 2.13 fix, and satisfy Ruff import grouping.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* [fix](torch): satisfy Ruff import style\n\nUse Ruff's preferred direct torch.fx import so the Torch 2.13 compatibility change passes pre-check review.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: perzhang <perzhang@amd.com>\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-10T14:13:52Z",
          "url": "https://github.com/ROCm/ATOM/commit/030b2ce9c058a441bd3ac45968a21e0c397d61ac"
        },
        "date": 1786381516907,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31408481514 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608101525 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9431 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31408481514 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608101525 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9538 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.61,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31408481514 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608101525 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9538 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7566,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31408481514 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608101525 | GPU: AMD Instinct MI350X | VRAM: 288GB | ROCm: unknown | strict-match: 0.7559 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "la",
            "username": "junhaha666",
            "email": "junchen2@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "6a756fdb53e8fa77fcbc455c5418fd9bc9c6d4c8",
          "message": "[feat](qwen):Support qwen3.5x model (#1738)\n\n* support qwen3.x\n\n* fix ATOM_USE_UNIFIED_ATTN=1 need prepare_block_tables on prefill\n\n* style: fix ruff findings in qwen3_5 / qwen3_next\n\n- I001: sort the qwen3_next import block (ruff's own suggested order).\n- RUF100: drop the unused `# noqa: F401` on the model_ops.linear import;\n  every name in that block is actually used in the file.\n- RUF012: hoist the mutable class-attribute literals of\n  Qwen3_5ForConditionalGenerationTextOnly to module-level constants and\n  reference them, matching the existing _QWEN3_5_PACKED_MODULES_MAPPING\n  pattern. ClassVar would need a typing import, which drags the file's\n  unsorted import block into reviewdog's diff_context and trips I001.\n  weights_mapping, quant_exclude_name_mapping and skip_weight_prefixes are\n  all done together because diff_context spans them.\n\nNone of these attributes are mutated in place anywhere in atom/, so sharing\none module-level object is behaviour-preserving.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* update readme\n\n* fix\n\n---------\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-13T15:51:26Z",
          "url": "https://github.com/ROCm/ATOM/commit/6a756fdb53e8fa77fcbc455c5418fd9bc9c6d4c8"
        },
        "date": 1786641016495,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31720300787 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608131526 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9447 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31720300787 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608131526 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 65.98,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31720300787 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608131526 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7475,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31720300787 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608131526 | GPU: AMD Instinct MI350X | VRAM: 252GB | ROCm: unknown | strict-match: 0.743 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8855,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31720300787 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608131526 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.3897 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ba9ae20278965d06fe07fb0e50bd9b3c5a92ebbf",
          "message": "perf(benchmark): batch the tokenizer calls that build the random dataset (#1907)\n\nGenerating the dataset runs before a single request is sent, so it is dead\ntime on every benchmark, and it grows with --num-prompts. At the c=8192 shape\n(24576 prompts, input_len 1024) it was roughly two minutes of a single Python\nthread. Measured on DeepSeek-V4-Flash's tokenizer, 2000 prompts at input_len\n1024 took 9.74s, split:\n\n  retry loop (encode + re-decode)   6.05s   62.2%\n  the final encode                  2.71s   27.9%\n  first decode                      0.50s    5.2%\n  building the token ids            0.46s    4.8%\n\nThree changes, no behavioural difference in the dataset -- same length\ndistribution, same avg_token_mismatch of 0.00.\n\nThe tokenizer calls are now issued a batch at a time. A fast tokenizer\nparallelises a batch internally, and one call per prompt forfeits that plus a\nPython/Rust crossing each time. The round trip becomes: batch_decode\neverything, batch encode, keep only the prompts whose length did not converge,\nrepeat. This is the whole win -- 9.74s to 1.85s, 5.25x.\n\nThe final encode is deleted. When the retry loop broke it broke *because*\nencode(prompt) had already come back at the target length one line earlier, so\nre-encoding the same string could only produce the same number again. It is\nstill paid when a prompt never converged, or when a chat template rewrites the\nprompt afterwards, because in both of those cases the length really is unknown.\n\nBuilding the token ids uses numpy rather than a per-token Python comprehension\nwith a modulo. This is the smallest of the three: 4.8%, or 25M interpreter-level\noperations at the c=8192 shape.\n\nSeparately, the dataset now draws from its own np.random.default_rng(seed)\ninstead of the global numpy RNG. Sharing the global one makes the prompts a\nfunction of whatever else drew from it first, so re-running the same command\nneed not produce the same dataset -- which is not a property a benchmark should\nlack. The seed comes from the existing --seed. vLLM's RandomDataset isolates\nits generator for the same reason and says so in a comment.\n\nTwo notes for review. The typing modernisation across the file (List/Tuple/Dict\n-> builtins, AsyncGenerator/Callable -> collections.abc) is a formatter sweep,\nnot a deliberate change; the timezone fix on the result-file stamp is ruff\nDTZ005 on a line the sweep pulled into diff context, and astimezone() keeps the\nstamp in local time exactly as before.\n\nTests use a synthetic tokenizer so nothing is downloaded. It is deliberately\nnot injective -- decode joins ids with spaces, encode splits on them -- so the\nround-trip correction is actually exercised rather than skipped, including the\ncase where it never converges and the reported length has to be the real one\nrather than the target it missed.",
          "timestamp": "2026-08-15T15:42:39Z",
          "url": "https://github.com/ROCm/ATOM/commit/ba9ae20278965d06fe07fb0e50bd9b3c5a92ebbf"
        },
        "date": 1786813271222,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9447,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31894611888 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608151459 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9424 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7521,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31894611888 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608151459 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7491 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8855,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31894611888 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608151459 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.2813 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "17e3e0771b11b2ed74d313f571327b481aa03efb",
          "message": "perf(frontend+server): stop the event loop stalling between request waves (#1912)\n\n* fix(benchmark): report requests that failed instead of hiding them\n\nEvery metric the run prints is computed over the requests that succeeded, so\na run that lost some of them reported the survivors' throughput as if nothing\nhad gone wrong -- at concurrency 8192 that silently hid thousands of dropped\nrequests behind a healthy-looking number.\n\nCount the failures by the last line of their traceback, which is the part\nthat identifies them, and print the total with a breakdown of the top\nreasons. Nothing else about the output changes, so the existing consumers of\nthis stdout keep working.\n\n* feat(server): expose uvicorn's keep-alive window and access log\n\nBoth were hardcoded to uvicorn's defaults and both matter at high\nconcurrency, so make them settable without changing what they default to.\n\n--timeout-keep-alive: a pooling client keeps an idle connection far longer\nthan the server does (aiohttp defaults to 15s against uvicorn's 5s), and only\nfinds out the server closed it by reading EOF from a request it already sent.\nRaising the server past the caller's idle window avoids that; requests here\nrun for minutes, so 5s is short.\n\n--disable-uvicorn-access-log: the access log copies a LogRecord and writes to\nthe same stdout the engine logs to, on the event loop, and says less than the\nengine's own per-request line.\n\n* perf(frontend): give each stream's detokenizer to the callback that feeds it\n\nDropping a request's detokenizer state meant finding it first, and the state\nlived in a dict shared by the engine's output threads and the event loop.\nFinding it was a scan of every live stream -- O(live) per cleanup, so draining\nN concurrent streams cost O(N^2): 894 ms at N=8192, four times that per\ndoubling. Keeping it findable by index instead is worse than it looks: a\nfan-out has several streams under one request id, so the index is a\nrequest-to-keys map that two threads maintain without a lock, and teardown has\nto remember to clear it or leak a detokenizer whose token list is unbounded.\n\nThere is no need to find it. The engine callback that produces a stream's\nchunks is already per-stream -- stream_callback for n=1, make_callback(i) for\neach fan-out sibling -- so let the closure hold the detokenizer and send it\nalong with the chunk. The dispatcher keeps no per-stream state at all, the\nper-chunk lookup is gone from the flush path, and the lifetime takes care of\nitself: when the engine drops a finished stream's callback the detokenizer\ngoes with it.\n\nTeardown splits the same way. cleanup_streaming_request took a seq id and a\nrequest id together, and a fan-out ran the whole thing once per sibling --\nincluding the two per-request pops, n-1 of them no-ops. cleanup_stream(seq_id)\nand cleanup_request(request_id) each take only what they need, and the\nfan-out finally reads as what the comment there always said it did: clean up\nevery sibling, then the request.\n\nAt concurrency 8192, 16384/16384 requests complete with no server-side\ntraceback, and GPU busy / idle / worst wave-boundary gap all land inside the\nrange of the pre-change runs.\n\n* perf(server): drop per-request admission logging to debug\n\nThese two lines run once per request, on the event loop, and logging takes a\nlock the engine's output threads are contending for on the same handler. A\nloop-stall watchdog at concurrency 8192 caught 26 stalls of 0.4s or more over\none run; 9 of them were sitting on the single-sequence line, up to 3.3 s each.\nA stall that long is not a latency blip -- the server accepts no new request\nfor its duration, so the engine drains its queue and the GPUs sit idle waiting\nfor work that is stuck behind a log write.\n\nMoving both to debug removed that whole class: stalls dropped to 18 and the\nworst to 1.45 s, with none of them landing in logging any more. What is left\nis h11/starlette protocol work and waking the SSE generators, which is\nstructural.\n\nuvicorn's own access log costs the same way (13 of the same 26 stalls) but is\nleft alone here -- it already has --disable-uvicorn-access-log, and turning it\noff by default is a visible change to server behaviour.",
          "timestamp": "2026-08-16T15:25:19Z",
          "url": "https://github.com/ROCm/ATOM/commit/17e3e0771b11b2ed74d313f571327b481aa03efb"
        },
        "date": 1786900067180,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9477,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31957679972 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608161502 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7528,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31957679972 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608161502 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7506 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "7229049cb9591c9463e5982cf692bc3cfa99543a",
          "message": "perf(v4): a smaller PAGE checkpoint image, a 60x cheaper copy, and no reserve floor (#1943)\n\n* perf(v4): hold only the part of a slot a resumer reads in a checkpoint\n\nA PAGE-backed checkpoint copied the whole Active Slot. Most of it is dead\nat the boundary the checkpoint sits on. DeepSeek-V4's HCA compressor pools\n`ratio` tokens with no overlap, so the first pool at or after a boundary P\ncovers `[P, P + 128)` — every row of it written by the very forward that\nreads it — and a checkpoint is aligned to `hash_block_size`, a multiple of\n128. Its two fields are 51% of a slot and a resumer never reads a byte of\nthem. The sliding windows sharing the slot are a sliding window and stay\nwhole; the padding between the two halves belongs to neither.\n\n`StateField.in_checkpoint` lets a field say it is not carried,\n`checkpoint_ranges_for` turns the declaration into byte ranges, and the\nDSV4 builder composes those with the window rows.\n`PagedStateCheckpointSpec.image_bytes` prices the result, so on\nV4-Flash-DSpark a checkpoint costs 7 PAGE units instead of 14 and displaces\nhalf the KV history it used to.\n\nThe rule holds only while every compress ratio divides the block size a\ncheckpoint is aligned to. `_assert_ratios_divide_block` refuses a build\nwhere it does not, because that failure is silent — a resumer reading stale\nKV for its first pool, which costs a little accuracy and nothing else. Two\nworkers disagreeing about the rule would read one image at two layouts, so\n`layout_id` names what is dropped and takes a new version.\n\nThe offset walk that `entry_bytes_for`, the arena's field offsets and the\nnew ranges all need is now written once, in `field_extents`; the three have\nto agree and previously each carried its own copy.\n\nGates on V4-Flash-DSpark tp2, bf16 KV, DSpark-5, interval 256. Arms were\nalternated one fresh server each: this box has a per-process spread that\nsequential arms read as a regression, twice.\n\n  units_per_checkpoint   14 -> 7, image 10,288,128 B = 48.6% of a slot\n  #1417 coherence        0/3 collapsed\n  throughput      n=3    39,000 vs 37,538 tok/s\n  acceptance      n=3    42.93% vs 41.97%, overlapping\n  GSM8K flexible  n=3    0.9510 vs 0.9530, overlapping, both inside\n                         the 0.9522 +/- 0.0059 band\n\nCopy cost is descriptor-bound, not bandwidth-bound: halving the bytes left\nthe measured 2.7 ms per op (0.87 plan + 1.85 launch, 135 spans) unchanged,\nwhich is what caps the gain at 3.9%.\n\n* fix(kv-cache): stop a checkpoint store from taking the pool's last units\n\nA PAGE-backed store called `ensure_free_units(units_per_checkpoint)` and\ntook whatever it got. Its units are then unreclaimable until the next\n`complete_inflight` publishes the record: `ensure_free_units` only evicts\nREADY ones, and an in-flight store is COPYING. A burst of requests crossing\na rung together therefore drains the free list, and the raise lands not on\nthe checkpoint — which is best-effort and would have been happy to be\ndropped — but on whichever live request calls `_fresh_block` next, where it\nis an AssertionError with no path back.\n\nA store now has to leave `reserve_units` behind and returns None when it\ncannot, which routes it into `checkpoints_dropped`, the same answer the\npool already gives when nothing can be evicted. The floor is one batch's\nworth of new blocks — a chunk of prefill plus at most one append per\nrunning sequence — because one batch is exactly how long a store's units\nstay unreclaimable. On V4-Flash-DSpark that derives to 320 of 103,809\nunits, 0.31% of the pool, and it is logged at startup: the number comes\nfrom the batch shape rather than a flag, so a zero would quietly mean there\nis no floor at all.\n\nNot a quota on how much checkpoints may hold in total. They can already be\nreclaimed on demand — `_ensure_page_units` evicts READY records for live KV\n— so the steady state is self-correcting and only the in-flight window\nneeded protecting.\n\nVerified on V4-Flash-DSpark tp2 bf16 DSpark-5, interval 256: the floor\nchanges nothing at this scale, `checkpoints_kept` 81 / `dropped` 0 /\n`evicted` 0 / hit 83.2%, all identical to the run without it.\n\n* perf(v4): describe a checkpoint copy in spans, not in 4 KiB tiles\n\nCopying a checkpoint cost 0.92 ms per op against 0.018 ms of kernel. The\nother 98% was describing the copy: 22 throwaway tensor views per PAGE unit\nto learn addresses, then every span expanded to 4 KiB tiles on the host and\nshipped as three Python lists through three pageable transfers — 2,632\ntiles where there were 135 spans.\n\nBoth go.\n\n`_page_unit_regions` works out `(base, num_bytes)` per region once. Blocks\nsit back to back in every pool, so a block's address is `base + id * bytes`\nand slicing the tensors to find it was buying one multiplication with a\nview. `tensor_segment`'s contiguity check comes along, asked once of the\nlayout instead of every time of a slice.\n\n`launch_copy_spans` uploads one descriptor row per span and lets the grid's\nsecond axis cut the tiles. That axis has to be as tall as the widest span,\nso where spans differ most programs find nothing to do — 94% of them here,\nwhose spans run from 8 KiB to 1.4 MB. Measured before being believed: an\nempty program is cheap enough that the trade is not close, and the kernel\nis unchanged at 0.018 ms.\n\n  per op, DSV4 narrowed image     before    after\n  region addresses                 0.260    0.030\n  plan_segmented_copy              0.061    0.061\n  launch_copy_spans                0.600    0.040\n  total                             0.92     0.13     7.1x\n\nEnd to end, arms alternated three fresh servers each: TTFT 202 -> 172 ms,\n3/3 rounds lower, ranges disjoint. Throughput +2.0% and acceptance -0.4 pp\nboth overlap and are not claimed — this workload runs OSL 64, so its total\nis prefill-bound and a decode-side instance spread of 7% sits on top.\n\nCorrectness is pinned by an oracle rather than by inspection: the addresses\nare asserted equal to what the tensor slicing produced, and the new kernel's\nbytes equal to what the tile kernel wrote. Both were checked to fail —\nswapping the layer and block strides, shortening a region, filling the\ndescriptor's destination column from its source, and dropping a byte per\nspan each turn a test red.\n\n* perf(v4): cut a checkpoint copy once, and stop carrying rows nothing reads\n\nTwo changes to the PAGE-backed checkpoint path: describing a copy no longer\ncosts per span, and an image no longer carries the entry's interleave\npadding. The first is what makes the second free.\n\nDescribing a copy cost about 0.53 us a span -- one Python loop to intersect\nthe slot's ranges with the image's PAGE regions, another to read three\nfields off each span into the descriptor. On the DeepSeek-V4 image that is\n0.107 ms an op against 0.018 ms of actually copying, and it is what decided\nwhether a finer image was affordable: every byte saved cost host time to\ndescribe.\n\nNone of it has to be paid per op. Which source segment meets which\ndestination segment, at what offset into each and for how many bytes,\nfollows from the two streams' *sizes*; addresses enter only when a copy is\nissued. Both streams are geometry -- the slot's ranges come from the\nlayout, and every image is `units_per_checkpoint` units of identical region\nsizes, which `_validate_paged_state_op` already insisted on. So the\nintersection is walked once for the life of the pool and an op becomes two\ngathers and two adds over precomputed offset arrays.\n\n`plan_segmented_copy` now takes sizes and returns a `SegmentedCopyPlan`:\nfive parallel int64 arrays, no addresses. `write_descriptor` fills a\n`(spans, 3)` block from one base per segment of each stream, which is where\nthe caller's geometry enters. A store and a restore are the same\nintersection read opposite ways, so `forward=False` reuses the plan rather\nthan cutting a second one, and every op of a batch shares one descriptor\nand one launch. `ByteSegment`, `CopySpan`, `tensor_segment` and\n`launch_copy_spans` go with it; V4's three segment builders collapse into\n`_checkpoint_slot_bases` (a `[group, segment]` matrix built once, so the\nper-op source side is a row lookup) and `_page_unit_bases` (one outer\nproduct).\n\nWith a span costing nothing to describe, the image can drop the rows the\nrow space only holds so that one index formula can serve every layer of a\ncompress class. The interleave runs by ring *position*, not by layer: rows\n`[c*run_rows, (c+1)*run_rows)` hold every layer's positions for run `c`, so\nthe rows the construction skips are reachable by no `(layer, position)`\npair at all -- nothing writes or reads them, and a checkpoint image, which\nis only ever gathered back into a slot and never read by an attention\nkernel, does not owe them. `ClassLayout.entry_row_runs` enumerates what is\nreachable; on the DSpark configuration the rest is 17.3% of the entry.\n\nThis keeps every ring position, so it needs no phase input and the range\nlist is still computed once. `layout_id` goes to v3 because an image is no\nlonger a subsequence of the slot's rows -- a v2 reader would gather every\nwindow row shifted.\n\n  image        10,288,128 -> 9,060,352 B (48.6% -> 42.8% of a slot)\n  PAGE units            7 -> 6\n  per op            0.107 -> 0.044 ms\n\nGates on DeepSeek-V4-Flash-DSpark bf16 tp2, `--num-speculative-tokens 5`,\n`--state-checkpoint-interval-tokens 256`:\n\n  - 1483 unit tests, including `entry_row_runs` checked against\n    `ring_offset_for` over the whole (layers, stride, ring_slots) product\n    and against the real Flash geometry's `ring_row`.\n  - The descriptor is asserted equal row for row to the walk it replaces,\n    on four image shapes, before any timing.\n  - #1417 coherence probe 0/27 collapsed, hits at 256 and 512 -- the\n    resumer reading window rows only the gather can have given it. A\n    control arm with packing disabled and nothing else changed: also 0/27.\n  - Acceptance rate, the probe this change would show up in first, on a\n    workload with 83% prefix hits: 42.28 packed against 42.28 unpacked\n    over four alternating instances per arm. On GSM8K with a\n    counterbalanced order: 64.460 against 64.465, and the same 0.96125\n    score.\n  - GSM8K 1319-question, three runs: 0.9484 / 0.9553 / 0.9484, inside the\n    0.9522 +/- 0.0059 band this configuration has held all along.\n\nThroughput came out 3.2% lower in the packed arm (p = 0.29, arm ranges\noverlapping, and the two position-matched pairs disagree in sign). Reading\nit as an effect would need many more samples than it is worth: there is no\nmechanism -- the image is smaller, takes fewer PAGE units, and costs\n0.044 ms an op against 0.042 -- and the metric that would see a real\nregression first shows nothing.\n\n* fix(kv-cache): test the checkpoint floor instead of evicting towards it\n\nThe floor added for live KV was handed to `ensure_free_units` as part of the\ncount it must reach. That reads like a request and behaves like a demand:\n`ensure_free_units` gives up only after it has evicted every READY\ncheckpoint it can, so whenever live KV held the rest of the pool a single\n`begin_store` emptied the entire cache and still returned `None` -- and did\nit again on the next batch, and the next. Reproduced on the real classes: a\n100-unit pool with 50 READY checkpoints and a drained free list loses all\n50 to one dropped store.\n\nThe units it freed did go to live KV, but `_fresh_block` already takes\nthose on demand, one at a time, at the moment they are actually needed.\nNothing was gained for the cost of the cache.\n\nAsk whether the floor is reachable before asking to reach it:\n`reclaimable_units` is the ceiling eviction could raise the free list to, so\na store that cannot clear the floor now refuses without evicting anything.\nRecycling is unaffected -- a store still takes the oldest checkpoint's units\nwhen the floor allows it. The victim predicate `ensure_free_units` walked\ninline becomes `_evictable`, so the two cannot come to disagree about what\nis evictable.\n\nThe rationale for the floor was also wrong, which is what made its size look\nindefensible. It said the hazard was a burst of stores holding `COPYING`\nunits that live KV cannot reclaim. That window does not exist on the normal\npath: `schedule` calls `complete_previous_state_batch` before it allocates\nanything, so an allocating `_fresh_block` always sees the previous batch's\nstores as READY.\n\nThe reachable hazard is the other unevictable state. `BlockManager.allocate`\npins a restore and then asks for fresh blocks *in the same pass*, and the\npin holds until the next `complete_previous_state_batch` -- so one pass of\nprefix hits can pin every checkpoint it resumes from and then find nothing\nleft to evict. A floor of one pass's worth of new blocks\n(`ceil(max_num_batched_tokens / block_size)` + `max_num_seqs`) is exactly\nwhat keeps that pass from ever having to evict, which makes how much of the\ncache it pinned stop mattering. It is sized against the pass, not against\nthe unevictable set. `test_the_floor_survives_a_pass_that_pins_the_whole_cache`\npins that invariant, which had no coverage at all; it fails with the floor\nset to zero.\n\nAlso: the derived-reserve log no longer fires when the coordinator ends up\ndisabled, where it reported a floor nothing would ever consult; and the\nfour address caches on the V4 builder carry the constraint that makes them\nsafe, since a reallocating pool would turn a stale one into a copy to the\nwrong address rather than a crash.\n\n* fix(kv-cache): stop the ladder cutting chunks for stores the floor refuses\n\nA demand is an instruction to cut a prefill chunk onto a rung, and that cut\ncosts the request a forward -- the same forward the interval grid exists to\namortize, and the one this guide measures at 17.5% of throughput when spent\nunconditionally. `begin_store` then drops the checkpoint whenever taking it\nwould leave the pool under the floor, so under pool pressure the ladder was\nbuying forwards for stores that were already going to be refused.\n\nThe ladder now asks the question the store will ask, through the same\nexpression: `has_room_for_store` is what `begin_store` refuses on, so the\ntwo cannot come to disagree about what is affordable. The gate goes on\n`_record_checkpoint_demand` rather than on either reader, because\n`checkpoint_cut` and `checkpointers_at` have to agree position for position\nand both read the one field -- gating the field keeps that free.\n\nWhat is deliberately NOT suppressed is the attribution.\n`num_wanted_hit_blocks`, and hence `Lost-to-checkpoint`, still say the reuse\nwas declined for want of a checkpoint, because it was; only the instruction\nto act on it is withheld. `demands_declined_no_room` joins the funnel so the\ndifference is visible, which keeps \"the ladder is quiet because there is no\ndemand\" apart from \"the ladder is quiet because the pool is tight\" -- the\nwhole reason that funnel is assembled stage by stage.\n\nThe fork path is untouched: a fork checkpoint costs the paged pool nothing,\nso `_checkpoint_has_room` is trivially true where there is no PAGE-backed\ncoordinator at all.\n\nAlso corrects the comment on `reserve_units` itself, which still described\nthe floor as protecting in-flight `COPYING` units. That window does not\nexist on the normal path -- `schedule` publishes the previous batch's stores\nbefore it allocates anything. The reachable one is a pass that pins every\ncheckpoint it resumes from, which is what the floor is sized against.\n\n`test_a_demand_the_floor_would_refuse_is_not_recorded` fails with the gate\nshort-circuited, and asserts the attribution survives it.\n\n* perf(v4): charge a checkpoint copy for its bytes, and stop reserving against it\n\nTwo halves, both of them subtraction.\n\n## The copy path\n\n`launch_copy_descriptor` opened a rectangular grid, `(spans, ceil(widest /\nTILE))`, which gives every span as many programs as the *widest* one needs. On\na DeepSeek-V4 image, whose spans run 8 KiB to 1.4 MB, that is 46,364 programs\nto do 2,631 tiles of work. One op could afford the waste; `execute_paged_state\n_copies` batches every op of a step into one launch, and a batch cannot. The\ngrid is now one program per tile that exists, from a tiling that is a pure\nfunction of the plan's geometry -- computed once when the plan is cut, resident\non the device from first use, shared by every op. That also retires `widest`,\na parameter whose only failure mode was silent: pass one too small and every\nlonger span was truncated, byte-correct on its prefix and stale on its tail.\n\nThe descriptor is then built for the whole batch in one pass instead of one\ncopy at a time. `write_descriptor` takes `(copies, segments)` base arrays and\n`_page_unit_bases` grew an image axis to match; store and restore are batched\napart because they read the same intersection in opposite directions. At these\nsizes a numpy call is nearly all call overhead -- a span table is a few hundred\nentries -- so paying it per copy was what made describing a batch a quarter of\nthe path once the kernel stopped being the bottleneck.\n\nFinally the kernel runs on four warps rather than eight. Its speed turns out to\nbe set by the width of one lane's access, `TILE / (num_warps * 64)`: three\nunrelated (TILE, warps) pairs that land on sixteen bytes measured within 0.5%\nof each other, while eight warps halves the width and costs 12%. This is\nrecorded in a comment because it reads like a knob to turn up, and turning it\nup makes it slower.\n\nAt 256 ops the path measures 1.29 ms against 15 ms, and the kernel is 88% of\nwhat is left. Every step was checked against the previous implementation as an\noracle, byte for byte, before it was timed.\n\n## The reserve\n\n`reserve_units` is deleted rather than resized. It had two stated purposes and\nneither survived being measured.\n\nThe first was to keep `_fresh_block` from raising. It cannot be reached: a\nREADY unpinned checkpoint is *already* available to live KV, since\n`has_available_units` counts it and `ensure_free_units` will spend it, so the\nsize of the cache is not the variable. What competes is the unevictable set --\n`COPYING`, or held by a restore pin -- and that set is confined to one pass.\n`schedule` publishes the previous batch's stores and releases its pins before\nit allocates anything, and this batch's stores are taken at batch construction,\nafter every allocation. The one overlap is `allocate`, which pins a restore and\nthen asks for fresh blocks in the same pass -- and its own `can_allocate`\ncounted that pin. `may_append` never overlaps at all, because the decode loop\nruns only in a pass that scheduled no prefill. Driven through the real\n`BlockManager` across a grid of pool shapes, thirty-eight gated runs never\nreached the raise while bypassing the gate reached it in fifteen of nineteen.\nUnder contention the reachable outcome is a refused admission, which the next\npass retries.\n\nThe second was to make that gate never refuse -- \"sized so the set is never\nconsulted\". It cannot do that either: the floor is a chunk's worth of blocks,\n`ceil(max_num_batched_tokens / block_size)`, while `allocate` takes a whole\nprompt's block table, up to `max_model_len` of them. A single 128K admission\nasks for more than the entire floor. No reserved quantity can promise live KV a\nblock, because live KV's demand is unbounded and legitimate, and that is now\nsaid in the code so the next reader stops looking for one.\n\nWhat the floor did do was evict. `begin_store` asked `ensure_free_units` for\n`needed + reserve`, so one accepted store spent up to fifty-five checkpoints\nbuilding a cushion for a hazard that is not there. It now asks for `needed`:\nfree units first, and at most one image's worth of eviction for the shortfall.\nTwo silent failure modes go with it -- a reserve larger than the pool would\nhave disabled checkpoints permanently behind a healthy-looking startup line,\nand the floor's prefill term used `block_size` where DCP wants\n`hash_block_size`.\n\nEviction eligibility and eviction policy are separated on the way past.\n`_is_evictable` says whether a checkpoint may be spent and is the single rule\n`has_available_units` and `ensure_free_units` share; `_next_victim` says which\none to spend first and is the only place the policy lives, least recently used\ntoday. `has_available_units` stops at the shortfall rather than totalling the\ncache, which is both the faster answer and the reason a future policy cannot\nmove the gate: the eligible set decides it, not the order it is walked in. That\nwalk is per-sequence per-pass, and a warm pool holds `num_kvcache_blocks /\nunits_per_checkpoint` checkpoints -- ten thousand of them here, previously\nsummed in full for an answer one or two settle.\n\n## Also\n\n`_assert_ratios_divide_block` is now `_assert_ratios_divide_the_alignment`,\nbecause it stopped asking about `block_size` when it started asking about\n`kv_cache_block_size * decode_context_parallel_size`. `_invalidate_pool_caches`\ngives the five address caches on the V4 builder one place to be dropped from,\nwhich whoever wires an elastic pool has to call -- a stale one is a copy to the\nwrong slot, not a crash. The vLLM bridge's HCA fields carry `in_checkpoint\n=False` like the native list, since `layout_id` is derived from the native list\nalone and cannot fence a disagreement between the two. And\n`checkpoint_bytes_for` goes, along with the duplicated flattening that had left\nit with no production caller.\n\n## Gates\n\nUnit tests 1497 (from 1443 on main); the nine mutations the new assertions\nexist to catch were each confirmed to fail them. The #1417 prefix-hit gate is\n0/6 collapsed with a hit at position 512, which the resumer never wrote and can\nonly have gathered from its image. Startup geometry is unchanged to the byte --\n`image_bytes=9060352`, `units_per_checkpoint=6`, layout v3 -- and the reserve\nline is gone. `demands_declined_no_room` and `checkpoints_dropped` are both\nzero. Six counterbalanced server instances put GSM8K, throughput, TTFT and MTP\nacceptance all in overlapping intervals, with each arm's own spread several\ntimes the difference between them.\n\n* fix(v4): ask the checkpoint demand gate about the pool its admission leaves\n\nThe gate asked whether an image fits, and then the very admission that asked\ntook its block table. A pool with room for an image but not for the request\n*and* an image answered yes, `begin_store` refused many forwards later, and\nthe prefill chunk the gate exists to withhold had already been bought -- with\n`demands_declined_no_room` at zero, so the funnel showed nothing. It now asks\n`num_new_blocks + units_per_checkpoint`, with the same `protected_hash`\n`can_allocate` passes to `_has_page_units` on the next line, so the two gates\nof one pass agree on what eviction could reclaim. It is asked afresh on every\nattempt, because a demand affordable when it was recorded is not still\naffordable once it is not; the sequence carries a marker per counter so only\nthe counting stays once per admission. It remains a sample even so, and the\ncomment says which loss it removes and which `checkpoints_dropped` still owns.\n\nThe reachability refusal moves from `begin_store` into `ensure_free_units`.\nThe bare loop gives up only after evicting everything it can, so an\nunreachable count destroyed the cache on the way to saying no -- and only\n`_fresh_block` asking for a single unit kept the other caller from needing it.\n\nStaging. `launch_copy_descriptor` now takes a descriptor already resident.\nA pageable `torch.from_numpy(x).to(dev)` issued from `build()` synchronizes\nthe current stream, so the host waits out the whole enqueued forward rather\nthan the 800 KB: measured 2.9 ms behind 4 ms of work against 0.1 ms through a\npinned `CpuGpuBuffer`, a cost the transfer's own size says nothing about. The\nkernel's tile and span counts stop being `tl.constexpr` -- specialising on\nthem keyed the compiled kernel to a pool geometry, so every image shape missed\nthe on-disk cache to save a divide the copy does not notice.\n\nNew `AttentionMetadataBuilder.warmup_per_req_cache`, called once after the\npools are installed. Everything the copy path builds lazily -- the plan, the\nslot views, the slot base table, the tiling's upload, the pinned buffer, the\nTriton JIT -- otherwise lands inside the batch of whichever request first\ncrosses a rung.\n\nGuards that could not fire, or fired on the wrong thing:\n\n- `_assert_ratios_divide_block` compared `CSA_RATIO`/`HCA_RATIO` against an\n  alignment `config.py` pins to 256 for every `DeepseekV4*`, so it could only\n  restate the config. Renamed `_assert_ratios_divide_the_alignment` and aimed\n  at `hf_config.compress_ratios`, which is what a variant is free to change.\n  A non-positive alignment gets its own refusal, since every ratio divides\n  zero and the ratio check would otherwise accuse an empty list.\n- `merge_abutting` silently merged unordered runs: `[(0,400),(256,64),\n  (320,64)]` returned `[(0,400),(256,128)]`, double-counting 128 bytes into\n  every consumer of `checkpoint_image_bytes`, its own cross-check included.\n- `write_descriptor` let numpy broadcast a short `dst_bases`, aiming every\n  copy of a batch at the first image's addresses, and accepted a flat one\n  that failed two lines later about an array the caller never passed.\n- `_page_unit_regions` keys on the addresses it was built from. Half of them\n  come from pools `_invalidate_pool_caches` does not own, so that hook could\n  never have been the invariant.\n- `checkpoint_ranges_for` no longer emits zero-length ranges, which\n  `plan_segmented_copy` refuses on the first copy -- after sizing, the\n  cross-check and startup had all passed.\n\n`has_room_for_store` and `reclaimable_units` are gone; neither had a\nproduction caller left. The first survives in the tests as\n`an_image_fits_on_its_own`, which is the contrast the new gate is read against.\n\n1522 passed / 52 skipped. Every new assertion was checked against a mutation\nrestoring the behaviour it describes.\n\n* test(v4): skip the checkpoint slot-copy module where aiter cannot load\n\nThe non-GPU CI runner collected it as an error: every class in the file reads\nunbound methods off `DeepseekV4AttentionMetadataBuilder`, and that module does\n`from aiter import dtypes` at load. One error in 1321 collected items, and the\nonly test module in the repo importing that chain without a guard.\n\nTwo things the obvious guard gets wrong, both found by rebuilding the runner's\nshape locally (an empty `aiter` namespace package shadowing the real one,\nwhich reproduces the message verbatim):\n\n`importorskip(\"aiter\")`, which is what the neighbouring V4 kernel tests use,\nis not the right question here. The failure reads \"cannot import name\n'dtypes' from 'aiter' (unknown location)\" rather than \"no module named\", so\n`aiter` is resolving as a namespace package and a guard on it can succeed and\nleave the real import to fail anyway. Asked of the module actually needed.\n\n`exc_type=ImportError` is required, not tidiness. The module *is* found, so a\nbare `importorskip` treats an ImportError out of it as the caller's mistake:\na deprecation warning on pytest 9.0, which is what this box has, and an error\nfrom 9.1, which is what CI runs -- so the unqualified form would have left CI\nred. Naming the type also keeps the skip narrow: anything that is not an\nImportError still fails.\n\nVerified both ways, because a guard that skips everywhere is not a fix: under\nthe runner's shape with 9.1 semantics the module skips where it used to error,\nand against the real aiter on this box its 26 tests still run.\n\nThose 26 assertions are now CI-skipped, as every V4-kernel-adjacent module\nalready is. Splitting the file would not recover any of them -- there is no\nclass in it that does not go through the builder.",
          "timestamp": "2026-08-18T14:13:32Z",
          "url": "https://github.com/ROCm/ATOM/commit/7229049cb9591c9463e5982cf692bc3cfa99543a"
        },
        "date": 1787073306763,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9477,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32158836903 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608181633 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9454 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9515,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32158836903 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:latest@sha256:e67fefb45f059e6d13629f7a0d0e2af3628ff254f7d1f3dc52c9a8fe6daea5ed | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.12,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32158836903 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:latest@sha256:e67fefb45f059e6d13629f7a0d0e2af3628ff254f7d1f3dc52c9a8fe6daea5ed | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7513,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32158836903 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:latest@sha256:e67fefb45f059e6d13629f7a0d0e2af3628ff254f7d1f3dc52c9a8fe6daea5ed | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7528 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8734,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32158836903 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:latest@sha256:e67fefb45f059e6d13629f7a0d0e2af3628ff254f7d1f3dc52c9a8fe6daea5ed | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3374 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "PerryZhang01",
            "username": "PerryZhang01",
            "email": "Perry.Zhang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "1e7659fde32eeaa0d9aa868c3e90847e5e46a51c",
          "message": "[feat](k3): support dcp on vllm-atom kimi k3 (#1951)\n\n* [feat](k3): support dcp on vllm-atom kimi k3\n\n* fix(k3): preserve non-DCP vLLM execution\n\nOnly bind and validate the aiter TP group when decode context parallelism is actually enabled.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* refactor(k3): reuse shared DCP query gather\n\nKeep the vLLM Kimi-K3 path aligned with the common DCP communication implementation introduced by #1930.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* fix(vllm): defer DCP patch to platform registration\n\nKeep native ATOM imports independent of vLLM while still installing the ROCm DCP full-graph patch for actual vLLM plugin runs.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: perzhang <perzhang@amd.com>\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-20T14:00:36Z",
          "url": "https://github.com/ROCm/ATOM/commit/1e7659fde32eeaa0d9aa868c3e90847e5e46a51c"
        },
        "date": 1787243924213,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32390560122 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608201458 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7362,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32390560122 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608201458 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7377 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.884,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32390560122 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608201458 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3904 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      }
    ]
  }
}