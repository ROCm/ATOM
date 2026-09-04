window.BENCHMARK_DATA = {
  "lastUpdate": 1788555100250,
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
      },
      {
        "commit": {
          "author": {
            "name": "haoyangli0109",
            "username": "haoyangli0109",
            "email": "lihaoyang0109@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "fa5efe5cbbcdbd8502170fe6dd5fbefd47ed8b15",
          "message": "[Feature] Quantize weights online when loading weights (#1749)\n\n* support streaming onine quantization\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>\n\n* fix ep mode\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>\n\n* comment\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>\n\n* fix test\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>\n\n* for ci error\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>\n\n* fix(loader): isolate expert staging from quant streaming\n\nKeep expert batching enabled for parameters not managed by the online quant streamer, preventing excluded MoE experts from falling back to thousands of small H2D copies. Add tests covering mixed streamer and expert staging paths.\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>\n\n* fix(loader): defer child quantization until parent post-processing for kimi-k3\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>\n\n* fix compile err\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>\n\n---------\n\nSigned-off-by: Haoyang Li <lihaoyang0109@gmail.com>",
          "timestamp": "2026-08-21T10:00:44Z",
          "url": "https://github.com/ROCm/ATOM/commit/fa5efe5cbbcdbd8502170fe6dd5fbefd47ed8b15"
        },
        "date": 1787331744696,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9469,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32501597679 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608211514 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9439 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32501597679 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608211514 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.13,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32501597679 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608211514 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7483,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32501597679 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608211514 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7468 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.887,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32501597679 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608211514 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3184 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "ZhangLirong",
            "username": "ZhangLirong-amd",
            "email": "lirzhang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ffdf3bb5ddf8a15a64d521a2eeb3bc7438f9ac21",
          "message": "Fix dpa spec middle chunk collective align for agentic (#1987)\n\n* Fix DP-attention deadlock on an all-middle-chunk prefill step\n\nA batch that produces no output (every seq mid-prompt in a chunked\nprefill) returns from ModelRunner.forward before postprocess, so it never\nreaches propose_draft_token_ids. propose() carries DP collectives -- the\ndrafter's DPMetadata.make all_reduce and the draft block's MoE comm --\nso under DP attention that rank silently drops out of the step while its\npeers (real decode, or the dummy decode dummy_execution hands an idle\nrank) block in that all_reduce forever. All workers spin at 99% CPU with\nthe GPUs at 0% and the log frozen after \"Scheduled prefill batch\".\n\nNeeds all three of --enable-dp-attention, a drafter (--method mtp/dspark)\nand a prompt over max-num-batched-tokens; _refresh_dp_metadata returns\nearly at data_parallel_size <= 1, which is why TP-only never hangs.\n\nRun propose for its collectives and drop the ids. A middle chunk sharing\na batch with a final-chunk seq already goes through propose, anchored by\nthe scheduler's successor token in batch.next_token_ids, so this is the\nexisting path for this case -- only the all-middle batch took the\nshortcut. Aligning by making the peers skip instead would strand a\nreal-decode rank without drafts and desync the deferred-output pipeline.\n\nVerified on DeepSeek-V4-Pro-DSpark, 8xMI355X, TP8/DP8, --level 3\n--cudagraph-mode FULL --method mtp -k 3, with and without --enable-tbo:\n19805-token (2 chunks) and 49405-token (4 chunks) prompts both deadlocked\nbefore and now answer in 2.7s / 5.5s, single-chunk prompts unaffected,\n8-way concurrent mixed lengths all complete.\n\n* Hold draft passes to a count the drafter declares, under DP attention\n\nThe all-middle-chunk fix left EAGLE running two collective-carrying draft\npasses where its DP peers run one. `precompute_context_kv` is a full draft\nmodel forward, and it fires on exactly the batch that now also runs\n`propose(align_only=True)` -- same rows, same anchors, which is why its own\ndocstring calls that case duplicate work. `dummy_execution` mirrors only\npropose, so the peers deadlock on the extra pass: one step later than the\nfirst deadlock and parked in recv_async_output_draft rather than the\ndrafter's all_reduce, but the same divergence.\n\nSkip the context pass on the aligning step, for drafters where it\nduplicates propose (`precompute_duplicates_propose`). DSpark keeps it:\n`write_context_kv` is local projections plus a Triton scatter, carries no\ncollectives, and propose only reads what it writes -- which is also why\nDSpark never hung here.\n\nThen close the class rather than the instance. Every draft site decided on\nits own whether to run, from data (`is_dummy_run`, `produces_output()`, an\nanchor of -1) rather than from what the peers would do, and that cost two\ndeadlocks in one code path. Drafters now declare\n`draft_passes_per_forward` (EAGLE mtp_k, DSpark 1), count each launch, and\nModelRunner.forward verifies the total before returning -- a local error\nnaming the rank and the counts, instead of eight workers at 99% CPU with\nthe GPUs idle and nothing in the log. The base property raises rather than\ndefaulting, so a new drafter has to state its count. Enforced only under DP\nattention, where the count is a contract at all: off lockstep an\noutput-less batch legitimately drafts nothing.\n\nDrops `_run_dummy_drafter`, an earlier unfinished pass at the same job --\nuncalled, and hardcoding mtp_k, which is wrong for DSpark's single block\npass.\n\nUnit tests cover the accounting and both deadlock shapes. Not GPU-verified\nhere; the Flash (EAGLE) agentic run that reproduced the second hang now\ncompletes.\n\n* Trim the draft-lockstep comments to what the code does not say\n\nThe prose around the pass accounting restated the mechanism the names and\nthe code already carry. Keep what a reader cannot derive: why DSpark's\ncontext pass stays, why count_draft_pass() is called before the launch,\nwhy the check is DP-only, and that the zeros passed to align_only are\nplaceholders the anchor override replaces rather than fake inputs.\n\nComments only -- no behavior change.\n\n* Drop the draft-pass accounting",
          "timestamp": "2026-08-22T13:31:44Z",
          "url": "https://github.com/ROCm/ATOM/commit/ffdf3bb5ddf8a15a64d521a2eeb3bc7438f9ac21"
        },
        "date": 1787417103045,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9538,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32583726138 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32583726138 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.12,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32583726138 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7491,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32583726138 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7445 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8832,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32583726138 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3397 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
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
          "id": "56d5565bdf2f5ccbfd5cf2f5ed9e43ee52e050b6",
          "message": "refactor(openai): one reader per wire format, for both delivery modes (#1992)\n\n* test(openai): generate the streaming-marker properties from the registries\n\nFive places in the streaming path answer one question -- given the text so\nfar, how much can be released without splitting a marker that has to be\nintercepted -- and they answer it four different ways. Three of those are\nwrong, and none of it is visible to the tests that exist: a change can leave\nthe wire behaviour identical and still go green.\n\nThis is the oracle, not the fix. Two properties, and they are not nested:\n\n  chunk-invariance  the same text split differently must produce the same\n                    reasoning, content, tool events and finish reason\n  bounded withhold  text must not be held back longer than a marker could\n                    justify\n\nThe stall is chunk-invariant -- everything comes out at flush no matter how\nthe input was split -- so the first property cannot see it, and on main it\npasses on every shape while the stall is sitting there. The second is what\ncatches it. Both are needed; an earlier draft claimed the first subsumed the\nrest and this suite disproved it.\n\nBounded withhold is judged against the same text with its trigger characters\nneutralised, not against a fixed budget. A flat budget failed the control\ntoo, because state 0 deliberately buffers 100 characters before releasing\nanything and because holding until `</think>` is correct rather than a stall.\nIt is also scoped to shapes carrying no marker the pipeline is entitled to\nhonour.\n\nThe 22 known-broken cases are `xfail(strict=True)`, conditional on the\ngenerated text containing the character the broken rule keys on -- so MiniMax,\nwhose ns_token opens with ']' and is genuinely unaffected, is not marked as if\nit were. Strict, so that fixing the rule turns these into failures and forces\nthe markers off rather than leaving them to rot.\n\nThe corpus is generated, not written: DIALECTS x (_DETECT_ORDER + KimiParser)\nx four text shapes built from each pair's own markers, 69 cases from two\ndialects and six formats. Registering a new model family or wire format adds\ncoverage by itself. `ToolCallParser.MARKERS` is the declaration that makes\nthat possible; it is inert here, read only by the tests, and Layer 1 of the\nrefactor is what will consume it.\n\nVerified by registering a temporary seventh parser: declared, its eleven\ncases joined on their own (47 -> 58 passed); declaration forgotten, one hard\nfailure. That second arm did not work at first -- the stub subclassed\nGlmParser and inherited its markers, so a missing declaration read as present\nand the new format was covered against another format's tags. The meta-test\nasks `vars(cls)` for that reason: a registered format declares its own, even\nwhen the tuple repeats.\n\n47 passed / 22 xfailed. Full suite 2350 passed against main's 2303, with the\nsame six pre-existing failures (EPLB dispatch and dspark, both unrelated).\nruff unchanged at 1134.\n\n* fix(openai): one rule for how much of a stream is safe to release\n\nA '<' in an ordinary answer withheld everything after it until the stream\nended. `if (a < b)` was enough. Measured on the real two-stage path, a\n1082-character answer produced **zero** content frames while streaming and\narrived in one frame at EOS; the first byte now leaves at input offset 4.\n\nThe rule was \"hold once a marker's first character appears anywhere in the\nbuffer\". Right idea, wrong scope: the check ran over the whole accumulated\nbuffer instead of a bounded tail, and the buffer was never cleared while the\ncheck held. That one scoping error produced both the stall and an O(n^2)\nscan -- 515 ms of host CPU at 64 KB of answer, against 6 ms now, and the\ngrowth is 2.0x per doubling where it was up to 11x.\n\n`MarkerScanner` is the rule, and the only copy of it: release everything\nexcept the longest suffix of the buffer that is a prefix of some marker. The\n'<' test survives as its fast path, applied to a buffer that cannot exceed\nthe longest marker, which is what makes it both correct and cheaper than what\nit replaces. `assert len(self._buf) < self._longest` is the invariant -- a\nstall is not something to test for, it is something that cannot be\nrepresented.\n\nFive sites answered this question four ways. All five now go through the\nprimitive:\n\n  reasoning.py state 0        `len(buf) > 100 and \"<\" not in buf`\n  reasoning.py _drain_thinking  already correct; now delegates\n  registry.py sniff_stream    `\"<\" not in buf`, over a buffer WAIT never cleared\n  BufferedMarkerParser        `rfind` with no branch for a hit at index 0, so\n                              after one emission it never emitted again\n  kimi_tool_parser            `\"<|tool\" not in buf and len(buf) > 30`\n\n`EMIT_CONTENT` is gone. It and `WAIT` conflated \"which format is this\" with\n\"may this text be sent\", and that conflation was the stall; `WAIT` now means\nonly the first, and the scanner answers the second before it is asked.\n`HOLDBACK_CHARS` is gone too -- a hand-kept copy of the first characters of\n`START_MARKERS`, which the scanner derives.\n\nState 0 no longer honours a `</think>` it never saw opened, and no longer\nbuffers 100 characters hoping for one. Both were the same guess -- that the\ntemplate had injected the opener -- made at run time about something the\nprompt answers. Bounded first-byte latency, honouring an unopened end marker,\nand chunk-invariance are three properties of which an implementation gets\ntwo; vLLM's streaming path drops the second and buffers nothing, SGLang drops\nit and names the consequence in a test, and this drops it too. The guess also\nlet pre-`</think>` text reach the tool-call sniffer, which is how reasoning\nthat merely mentions `<tool_call>` came to be emitted as a tool call.\n\nWhat replaces it is the seed, so the seed has to be right everywhere: ten\nsites now pass `starts_thinking`, including the Anthropic `/v1/messages`\nendpoint the stall was reported against, which hand-rolled\n`prompt.rstrip().endswith(\"<think>\")` -- blind to Kimi-K3's\n`<|open|>think<|sep|>` -- and assigned `.state = 1`, bypassing\n`__post_init__`. `separate_reasoning` takes the same seed, because it did not\nand the two disagreed: a trace truncated at `max_tokens` came back as\n`content` non-streamed and `reasoning_content` streamed, so a client reading\n`delta.content` saw an empty completion.\n\nAlso here, found while doing the above:\n\n- `tool_choice=\"none\"` was ignored by the `n>1` streaming path. Four emission\n  sites, none gated, while the single-choice path gated all of its.\n- Every format declares `START_MARKERS` and nothing else. A second list\n  existed briefly and was hand-copied wrong twice -- DSML's real openers use\n  full-width pipes, and K3's detection keys on five literals where the copy\n  had four. `<arg_key>` lives next to the cascade that discriminates by it,\n  with the reason it cannot be a `START_MARKER`: `find_start` would take it\n  for the region's beginning and parse from the wrong offset.\n- `docs/serving_benchmarking_guide.md` claimed \"a token is never delivered\n  later than it otherwise would have been\". It can be, by a bounded amount,\n  and it used to be by an unbounded one. Says so now.\n\nCoverage that extends itself, so the next format or endpoint is not covered\nby someone remembering: the property corpus generates from `DIALECTS` x\n`_DETECT_ORDER` and fails if a parser declares no `START_MARKERS`; a source\nscan fails on any `ReasoningFilter(...)` or `separate_reasoning(...)` without\n`starts_thinking`; another fails on any tool event emitted without checking\n`tool_choice`. All three armed by breaking them.\n\n297 passed in tests/entrypoints, no xfails left -- the 22 the previous commit\nrecorded were this. Full suite 2434 passed against main's 2303, same six\npre-existing failures. ruff unchanged at 1134.\n\n* feat(openai): decide a model's tool-call format at startup, not from its output\n\nThe format was sniffed from the model's own text, which means deciding from a\nprefix: a discriminator may not have arrived yet, so the answer needed a\n\"cannot tell yet\" state. `--tool-call-parser` defaults to `auto`, which\nrenders the chat template with a tools payload -- the template's own\ninstructions for calling one -- and runs the same `_DETECT_ORDER` cascade\n`parse_tool_calls` already runs on a complete output. No new table and no new\nrules: `MiniMaxParser.detect` still requires its ns_token, so it does not\nclaim a Qwen prompt merely because both use `<tool_call>`.\n\nEvery model on this box, through the shipped path:\n\n  DeepSeek-V4-Pro / Flash / Flash-DSpark   dsml\n  Kimi-K3                                  kimi_k3\n  Qwen3.5-27B / 4B                         qwen\n  MiniMax-M3                               minimax\n  Qwen3-8B                                 glm\n  gpt-oss-120b, DeepSeek-R1-0528           none\n\nThe V4 models resolve through `<model>/encoding/encoding_dsv4.py`, a Python\nmessage encoder ATOM already loads at startup; `tokenizer.chat_template` is\n`None` for them, and rendering is what makes the difference. A first attempt\nat this searched that attribute for each format's literals and concluded most\nmodels here have no template -- wrong twice over, since the V4 template is not\na Jinja string and the literals that do exist are interpolated\n(`\"<{dsml_token}invoke ...>\"`).\n\nUnresolved is a real answer and it is said out loud: one line naming the model\nand stating that tool calls will be delivered as plain text. It does not fall\nback to reading the output, because that fallback is the silent failure being\nremoved -- a Hermes-style `<tool_call>{\"name\": ...}` is claimed by the GLM\nparser, which delivers the whole JSON blob as the tool's *name* and `{}` as\nits arguments, and nothing about that reaches a log. (That is pre-existing:\nthe old sniffer picks GLM for it too. It wants its own issue.) An unknown\n`--tool-call-parser` name raises rather than quietly disabling tool parsing,\nand a template that cannot render logs and resolves to nothing.\n\nGone: `sniff_stream`, `WAIT`, `all_markers`, `_SNIFF_ONLY`. With the format\nknown before the first token there is nothing to wait for, and the read-ahead\nnarrows from every format's markers to the chosen format's -- 14 to 4 on\nDeepSeek-V4. `parse_tool_calls` takes the same resolved format, so the\nstreaming and non-streaming paths cannot answer differently; ten call sites\nacross both servers pass it.\n\nResolution first went into a module of its own, to keep `registry.py` free of\nexternal imports. It needed exactly one, `apply_chat_template`, which every\nconsumer of the registry already pulls in, and the module held nothing but\nregistry business: the probe payload, the name lookup, the logging. Folded\nback next to the cascade it runs and the map it looks names up in.\n\n`test_tool_call_resolution.py` covers the override, the template, the\nunrecognised template, a template that raises, and that the probe really\ncarries a tool -- a template that only mentions tools when given some would\notherwise render its plain-chat branch and every model would resolve to\nnothing. One case is parametrised over `PARSERS_BY_NAME`, so a format that\njoins the registry without a usable command-line name fails here.\n\n298 passed in tests/entrypoints. Full suite 2435 passed against main's 2303,\nsame six pre-existing failures. ruff unchanged at 1134.\n\n* fix(openai): stop the Anthropic stream dropping a reasoning block, silently\n\nA model that answers, opens a `<think>` block and answers again lost the\nreasoning entirely on `/v1/messages`. Measured on that exact shape, 29\ncharacters went nowhere: no error, no log, no frame.\n\nAnthropic frames a response as indexed blocks of a kind — text, thinking,\ntool_use — and a change of kind is a close and an open. Those transitions were\nwritten out at each of the four places a segment could arrive (content and\ntool_call_start, in the main loop and again in the flush), each covering the\nsubset its author needed. The one nobody needed, text -> thinking, was missing:\n\n    if not started_thinking and not started_text: ... started_thinking = True\n    if started_thinking: yield delta(...)\n\nWith `started_text` set and `started_thinking` clear, neither branch fires and\nthe segment falls off the end of the loop.\n\n`AnthropicBlocks` asks for the transition instead of writing it out: `delta`\nsays which kind this text belongs to, and switching is the class's problem.\nThere is no branch left to fall off. Four copies collapse into one, and the\n`started_text`/`started_thinking`/`block_index` bookkeeping goes with them.\n\nTwo-arm check on the failing shape: booleans deliver `thinking=''`, blocks\ndeliver the whole 29 characters, and text+thinking now reconstitute the input\nminus its markers, 62 of 62. `test_anthropic_blocks.py` covers all nine\norderings of the three kinds, that a block is closed before the next opens,\nthat indices stay unique and ascending, and that a thinking block signs off\nbefore it stops — all eight of the relevant cases fail against the old rule.\n\nThe bug that this fix introduced on the way is the second half of the commit.\nA `yield from` inside an `async def` generator is a syntax error, and\n`api_server.py` did not parse. The suite reported `279 passed, 40 skipped`:\n`test_api_server_helpers.py` guards its import with `except Exception` and\nskips the module, so nineteen tests went quiet and nothing failed. A syntax\nerror in a shipped module read as green.\n\n`tests/import_guard.py` asks the question once. An `ImportError` naming a\nthird-party module is an environment that cannot run the test; anything else\n— a `SyntaxError`, a `NameError`, an `ImportError` naming one of ours — is a\nbug that has to be seen, and is re-raised. `ModuleNotFoundError` is a subclass\nof `ImportError`, so both \"no module named aiter\" and \"cannot import name\n'dtypes' from 'aiter'\" still skip; the second is what a namespace package on\nthe path produces and is how the non-GPU CI actually fails.\n\nEight modules used the same swallowing guard and all eight now use the shared\none. Armed both ways: a syntax error in `api_server.py` turns 19 silent skips\ninto one collection error, and with an empty `aiter` on the path the\ndependency-gated modules still skip.\n\nIncidental: `ruff --fix tests/` was run over the tree and its cleanups kept\nrather than reverted to preserve a count. All of it is import ordering and\ngrouping, no semantics. `tests/plugin/` is `--ignore`d by the CI runner, so\nits files were checked separately — 5 collection errors, identical to main.\n\nFull suite 2451 passed, same six pre-existing failures. ruff 1134 -> 1100.\n\n* feat(openai): make a stalled stream visible while it is stalled\n\nThe stall this line of work started from is fixed, so nothing here is\njustified by it. What justifies it is the shape of the report: ten minutes of\nsilence on a streaming request with every metric looking healthy. The next\ncause will be different and will present the same way, and there was no\ntimeout anywhere on the SSE path -- the only one in the server is uvicorn's\n`timeout_keep_alive`, which does not apply mid-response.\n\n`StreamOutputCollector.get` is the single await both the single-choice and\nfan-out streams pass through, so recording when each wait began covers both\nin one place. `atom:stream_longest_silence_seconds` reports the age of the\noldest: zero when nothing waits, non-zero and growing when a response has\nstopped delivering while its client waits. A wait that ends after more than\n30 seconds also logs a line naming the request, because by scrape time the\ngauge cannot see a stall that already recovered. Neither alone is enough.\n\nNot `asyncio.wait_for`, and the measurement is the reason. This runs once per\ntoken per stream; arming a timer costs 1.38 us against 0.07 us for a\ntimestamp and a dict entry, 20x, and the timestamp needs no background task\nto own or shut down. The gauge is read live rather than from the engine's\nmetrics snapshot: a stream starved by the engine is exactly the case where\nthat snapshot may also be late, and this question is answered by the event\nloop serving the stalled request.\n\nSeven tests: the gauge while a stream waits, that it clears on delivery, that\nthe oldest wait is the one reported, that a cancelled stream leaves nothing\nbehind -- an abandoned request pinning the gauge high forever would make the\nsignal useless from then on -- and both sides of the log, fired and quiet.\nThree mutations caught: not recording the wait, not clearing it, never\nlogging.\n\nOne of those mutations is worth writing down. Written first as\n`except BaseException: pass`, it swallowed the `CancelledError` the\ndisconnect test depends on and hung the harness rather than failing it.\nMutation arms need their own timeout; a hang is a result too.\n\nFull suite 2458 passed, same six pre-existing failures. ruff unchanged\nat 1100.\n\n* fix(openai): stop the streaming path deleting text it decided not to parse\n\nA code review of the four commits before this one found that pinning the tool\nformat at startup had introduced a silent deletion, and that the property\nsuite written to catch exactly that class was running with the tool parser\nswitched off. Both are fixed here, along with four more the same review found.\n\n**Text after a quoted marker was dropped.** With the format known up front, a\nbare start marker opens the region, and an answer explaining that \"the model\nwrites `<tool_call>` followed by the name\" opens one that never closes.\nReproduced: 82 characters in, 32 delivered, no event, no error,\n`finish_reason` still `stop` — and under GLM a fabricated tool call named\n` followed by the name. Hope that helps!`. On main the sniffer returned WAIT\nand the whole buffer came back as content.\n\nFour sites, not one: `BufferedMarkerParser.flush` discarded the region when\n`parse` found no calls; `KimiParser.flush` drained `self.buf` while the held\ntail lives in the scanner now; its state-1 dropped an empty section; and\n`KimiK3Parser.parse` truncated at its tools token whether or not a call\nfollowed. A start marker is not a promise, and all four now say so.\n\n**GLM accepted any prose as a tool name**, which is the deeper cause: its\nunterminated branch takes everything after `<tool_call>` as the name. A\nHermes-style `<tool_call>{\"name\": ...}` therefore arrived as a call named\nafter the whole JSON object — verbatim the failure the previous commit\nclaimed to have retired, now pinned for the process lifetime because the\nformat is resolved once. A tool name is an identifier, which is what the\nmodel was given; requiring that turns a misresolved format from \"fabricated\ncall, answer swallowed\" into \"delivered as text\", which does not depend on\nresolving correctly.\n\n**DeepSeek-R1 lost its reasoning entirely.** Gating the bare-`</think>` split\non `starts_thinking` was justified by \"vLLM's streaming path drops it\", and\nthat was a misreading: vLLM registers `DeepSeekR1ReasoningParser` for R1\nprecisely to override that branch. It is a property of the model family,\ndecided at startup, not a universal rule. Derived here rather than listed —\na chat template that closes a reasoning block it never opens describes that\nshape. On this box: R1 true; Qwen3-8B and Qwen3.5 false (they open their own);\nMiniMax-M3, Kimi-K3, gpt-oss, DeepSeek-V4 false.\n\nThree more from the same review:\n\n- Kimi-K2 could never resolve: `_DETECT_ORDER` omits `KimiParser` because it\n  is the terminal fallback there, so a K2 deployment parsed nothing while the\n  non-streaming path still fell through to it and returned the calls.\n- `parse_tool_calls(parser_cls=...)` ran the format's `parse` unconditionally,\n  and formats strip. A code-block answer came back without its trailing\n  newline when `stream=false` and verbatim when `stream=true`. It now returns\n  the input untouched when no call was found.\n- atomesh had no `tool_choice` at all: four emission sites ungated and\n  `tools=` never passed, so `tool_choice=\"none\"` was ignored and argument\n  coercion was off, on the entrypoint whose neighbouring lines the previous\n  commits had just edited.\n\n**The watchdog was measuring queueing.** Both SSE generators await `get()` as\nthe first statement of their loop, before admission and prefill, and at the\nconcurrency this server is benchmarked at that first wait routinely outlives\nthe threshold — a log line per backlogged request onto the event loop, and a\ngauge duplicating `atom:requests_waiting`. It now measures silence only after\na stream has delivered once, which is what a stall is.\n\nThree test defects, all of which let the above through:\n\n- `TestChunkInvariance` called `drive(text, chunks)` without its parametrized\n  parser, so all 42 cases ran with the tool parser disabled: 0 tool events\n  across the corpus where wiring it gives 96. The docstring claimed the\n  property checked tool events. Caused by a `.replace()` with no assertion —\n  black had already reflowed the line, so it matched nothing and said so\n  nowhere.\n- Wiring it back still did not catch the deletion, because deleting the same\n  bytes under every chunking is perfectly chunk-invariant. Two properties\n  added: what follows a marker in an answer that calls nothing must arrive,\n  and a partial marker held at EOS must be released. Two generated shapes to\n  reach them — every marker quoted, not just the first, and an answer ending\n  mid-marker — and the shape filter changed from a hardcoded list to a prefix\n  match, since the list is what excluded the generated shapes.\n- Both AST lints were vacuous in the ways that mattered: the tool_choice one\n  passed with every gate inverted to `== \"none\"` and matched nothing after a\n  plausible rename; the seeding one accepted a hardcoded\n  `starts_thinking=False` and rejected a correct positional call. Both now\n  check the shape of the answer, and both have a control asserting they can\n  still fail.\n\nSome states are unreachable through the facade — a parser is only built once\na complete marker has arrived, so its own pre-region path cannot be driven\nfrom there. That is why the property suite could not see the Kimi loss, and\nwhy those three cases are now direct-parser tests.\n\nEvery fix here was checked against a mutation restoring the behaviour it\nreplaces; the watchdog's `except BaseException` arm hung rather than failed,\nso the harness times each arm out.\n\nFull suite 2568 passed, same six pre-existing failures. ruff 1100 -> 1070,\nits own auto-fixes kept rather than reverted.\n\n* chore(openai): make the tool-call flag reachable, and stop docs describing the old design\n\nThe last of the review's findings, all of the kind that leaves the code\nlooking audited when it is not.\n\n`--tool-call-parser` existed only on the OpenAI server. The atomesh entrypoint\nnever registered it and `AtomStandaloneService` hardcoded `override=None`, so\nthe documented escape hatch for a misresolved format did not exist there — on\nthe entrypoint whose neighbouring lines three of these commits had just\nedited. Registered now, threaded through `StandaloneArgs`, and its help text\nlives once in `registry.py` next to the names it accepts, because two copies\nis how the flag came to exist on one server and not the other.\n\nThe flag was also documented in the `EngineArgs` table, where there is no\nsuch argument — it is a server flag, registered in `api_server.main`. Moved\nto the server-specific table.\n\n`tool_parser/__init__.py` still told the next author to register a format in\n`sniff_stream`, which no longer exists, and described formats as\n\"auto-detected\" from output. One entry in `_DETECT_ORDER` is now the whole\njob: startup resolution, the streaming read-ahead, the flag's accepted names\nand the property corpus are all derived from it, and saying so is the point\nof the paragraph.\n\n`tests/test_eplb_module_c.py` was the last guard still catching `Exception`\nand skipping. Its five siblings importing the same module were converted;\nthis one was missed, which is precisely the failure mode — a run that looks\naudited. Swept, and no swallowing guard remains: the fifteen other modules\nusing `allow_module_level=True` reach it through `importorskip` or a plain\nGPU skip, neither of which hides a broken import.\n\nLast, the one the review could not reproduce end to end, now reproduced. When\n`/v1/messages` is called without `thinking`, reasoning segments are dropped.\nThat was harmless while an unseeded filter sent most output down the content\nchannel; with seeding correct, a reasoning model stopped at `max_tokens`\nproduces nothing else, and the client got 20 pings, zero delta frames, an\nempty text block and `stop_reason=end_turn` — a wholly empty message where\nmain delivered the text. The withheld reasoning is kept and sent as text if\nit turns out to be all there was. Answering with the model's working out is\nworse than answering properly and better than answering nothing.\n\nThree tests pin the shape that decision reads: an untouched `AnthropicBlocks`\nreports `index == 0`, a delivered block advances it, and opening a block\nwithout delivering does not — the endpoint opens a trailing text block before\nit checks, so the third is the one that matters.\n\nFull suite 2571 passed, same six pre-existing failures. ruff unchanged\nat 1070.\n\n* fix(anthropic): do not separate a reasoning channel nobody asked for\n\n`/v1/messages` without `thinking` built a seeded reasoning filter anyway,\nsplit the output into two channels, and discarded the reasoning one. That was\ninvisible while an unseeded filter sent most output down the content channel.\nWith seeding correct, a reasoning model stopped at `max_tokens` produces only\nreasoning: the client got 20 pings, zero delta frames, an empty text block and\n`stop_reason: end_turn` — a wholly empty message where main delivered text.\n\nThe previous commit answered this by keeping the discarded reasoning and\nsending it as `text` when nothing else had been delivered. That was wrong in\nthe way the request is explicit about: `thinking` off means the client does\nnot want a chain of thought, and handing it one relabelled as the answer\nhonours neither the request nor the format.\n\nSo nothing separates. When `thinking` is off no filter is built, no marker is\nwatched for, and the model's output is the text — markers and all. Three\nproblems go with the premise rather than being handled: nothing is discarded,\nso there is no empty message; nothing is looked for, so nothing is held back;\nand there is no `</think>` detection, so nothing is truncated at one.\n`_ANTHROPIC_PING_FRAME` and its branch are gone.\n\nThe decision now lives in `anthropic_reasoning_split` and\n`anthropic_reasoning_filter`, called by the non-streaming and streaming paths\nrespectively. Not for tidiness: the endpoint body is an async generator inside\na route handler that no unit test reaches, and an earlier version of these\ntests re-implemented the branch in order to test it — and passed while the\nreal one did the opposite. A mutation putting `separate_reasoning` back inline\nstill passed every behavioural test after extraction, so there is also a\nsource check that each helper is called and not merely defined. Both mutations\nnow fail.\n\n`stop_reason` is separate and pre-existing: it was the constant `end_turn` on\nthe streaming path and the engine's own reason was never read, so a response\ncut off at `max_tokens` reported a normal ending. The vocabularies already\nline up — `eos`, `max_tokens`, `stop_sequence` — and are now connected;\n`aborted` has no counterpart and keeps the default, the client having already\ngone.\n\nTwo function-local imports of `ReasoningFilter` and `separate_reasoning`\nhoisted to module scope, which the helpers need and which this repo bans\nanyway.\n\nFull suite 2586 passed, same six pre-existing failures. ruff unchanged\nat 1070.\n\n* fix(openai): answer a request's `thinking` in the prompt, and eight review findings\n\nA second review of this branch found fifteen issues. Twelve reproduced when\nrun against the branch; two were wrong about the fix and one was pre-existing.\nThis is all of them, plus the sweep each implied.\n\n**The one that matters, and it undoes the previous commit's answer.**\n`dbea888e4` decided that `thinking` off means the output is delivered as text,\nmarkers and all. It is: the reasoning is then still *in* that text, and the\ntool parser is a second reader of the same bytes. Reproduced -- a model\nreasoning aloud about the syntax (\"...the syntax is `<tool_call>` then\n`<function=NAME>`...\") before making a real call produced a call to a\nnonexistent tool `NAME`, the real call gone, the answer truncated to the\ninside of its own thought.\n\nTying the tool parse to `thinking` would have fixed that and cost more than it\nsaved: `thinking` is absent on most requests, so `tool_use` would have stopped\nworking for most traffic. What both upstreams do instead is answer the field\nwhere it costs nothing -- in the prompt. SGLang's `apply_reasoning_enabled`\nwrites the chat template's own switch so the model emits no reasoning; vLLM has\nno such request field at all, runs its reasoning parser unconditionally, and\nlets `include_reasoning` suppress only the result after the split. Either way\nthe tool parser never sees a chain of thought.\n\nSo: `thinking: disabled` now sets the template switch, and everything\ndownstream is unconditional, exactly as on `/v1/chat/completions`. Three\nproblems go with the premise -- nothing is generated, so nothing is discarded\n(no empty message), nothing is relabelled (no reasoning smuggled as `text`),\nand nothing unseparated reaches the tool parser. The Anthropic path had been\ndropping the request's `thinking` on the floor when rendering the prompt; that\nis the actual root cause the last three commits were each working around.\n\nAn *absent* `thinking` leaves the model's default alone rather than switching\nreasoning off, which is what SGLang keys on too: Anthropic's default is off,\nbut reading absence as \"switch this model's reasoning off\" would silently\nchange what every existing caller gets back.\n\n**Which kwarg carries the switch is asked, not listed.** A template silently\nignores a kwarg it does not read, so a hardcoded name is a no-op that looks\nlike a feature -- and the chat path had one. `merged_kwargs[\"thinking\"]` is\nright for Kimi-K3, which is what its comment says, and a no-op for the entire\nQwen family, whose templates read `enable_thinking`; measured, `thinking=False`\nleft Qwen3.5's `<think>` prefill exactly where it was. `resolve_reasoning_toggle`\nrenders twice and compares, because \"the prompt changed\" is the evidence the\ntemplate read it. Verified on ten models: Qwen3/Qwen3.5 `enable_thinking`,\nKimi-K3 `thinking`, MiniMax-M3 `thinking_mode=\"disabled\"`, DeepSeek-V4\n`thinking_mode=\"chat\"`, and for all six that begin inside the reasoning channel\nthe pair takes the rendered prompt back out of it. gpt-oss and DeepSeek-R1 have\nno switch and are named in the startup log.\n\nCandidates are (name, value) pairs and a refusal continues the search, because\n`thinking_mode` is read by two families with disjoint vocabularies -- MiniMax\nwants \"disabled\" and DeepSeek-V4's Python encoder asserts on anything outside\n{\"chat\", \"thinking\"}. An earlier probe of mine caught that assertion in a bare\n`except Exception` and concluded V4 had no switch at all.\n\n**The seven others.**\n\n- `parse_tool_calls` still ran the detection cascade over the *output* when no\n  format had been resolved, so an answer quoting `<|tool_calls_section_begin|>`\n  lost 30 characters with `stream=false` and arrived whole with `stream=true`.\n  gpt-oss and DeepSeek-R1 both resolve to `None` on this box. The cascade is\n  deleted rather than gated: `_DETECT_ORDER` now has one consumer, the prompt,\n  which is what the docs already claimed.\n\n- A format's `parse` rewrote answers it found no call in. Every one ended in\n  `.strip()`, which the streaming path has nothing to match, so a code-block\n  answer lost its trailing newline on one path only. The rule is now stated on\n  `ToolCallParser.parse` and generated from the registry by the property suite,\n  so a format added later is bound by it without a new case. Kimi-K3 needed a\n  different fix: its channel framing *should* go, but `_strip_k3_framing` also\n  trimmed whitespace, which is position-dependent -- applied to a region when\n  streaming and to the whole answer when not, turning `writes <tok> to` into\n  `writes to` on one path and `writes  to` on the other.\n\n- GLM's tool-name check was narrower than OpenAI's own grammar and ASCII-only,\n  so `7z_extract` and any CJK name were dropped by a bare `continue`, on a\n  Chinese model family.\n\n- Kimi-K3 tested \"did a call parse\" to decide whether a tools section had\n  opened, so a call truncated at `max_tokens` parsed to nothing and its\n  half-written payload was kept and shipped as the answer. A call *prefix*\n  after the marker separates the two cases; a truncated call still has one.\n\n- `template_opens_reasoning_implicitly` read the raw `tokenizer.chat_template`,\n  which is `None` for every model shipping a Python encoder and a `dict` for\n  multi-template tokenizers, where `\"</think>\" in <dict>` silently tests the\n  keys. The review's fix -- render like the sibling probe does -- is wrong, and\n  measurably: an end marker is what a template does with a *reply*, so it never\n  reaches a fresh prompt. Qwen3.5's source carries both markers, its rendered\n  prompt only the opener, and Qwen3-8B's neither. Reading a render would answer\n  False for every model alive. `chat_template_source` reads the source and\n  handles all three shapes.\n\n- `--tool-call-parser` on atomesh shadowed the mesh router's flag of the same\n  name: registering it made `parse_known_args` swallow an argument that had\n  been passing straight through to the Rust CLI, which declares its own with\n  its own vocabulary. Both consumers get it now. The name is also validated\n  before `initialize_engine` rather than inside the service constructor, so a\n  typo no longer costs a full model load.\n\n- `stop_reason` was wired into the streaming generator only, so the same\n  response reported `max_tokens` with `stream=true` and `end_turn` with\n  `stream=false`. Not routed through `_normalize_finish_reason`, which maps\n  into OpenAI's vocabulary and shares no member with these keys; chaining them\n  would send every reason to the default. `stop_<token_id>` is its own case --\n  a stop token fired, which is Anthropic's `stop_sequence`.\n\n- `MarkerScanner` ran `frozenset.intersection(buf)`, which hashes every\n  character, where `any(c in buf ...)` is a C substring search: 2.9 us against\n  0.12 us on 900 chars, once per token per stream. The suffix sweep re-sliced\n  every marker at every length. Both fixed by a per-marker-set plan computed\n  once and cached, and `feed` is 1.2-5.3x faster with `__init__` *cheaper* than\n  before (864 ns -> 300 ns) because the sort is cached too. `reasoning`'s\n  `_hold_back_len` had taken the sweep and none of the first-character reject;\n  it delegates now.\n\n**Two corrections to the review.** `tool_choice=\"none\"` does silently delete a\ntool-call span, but `git blame` puts it in #489/#1727 -- pre-existing, and left\nfor its own change. And the atomesh flag's blast radius was overstated: the\n`--tool-call-parser deepseekv4` in `recipes/` and `.github/` are SGLang's own\nlaunch args and never reach `parse_standalone_args`.\n\nEvery fix here was checked against a mutation restoring the behaviour it\nreplaces, 17 of them, re-run after formatting because a reflow turns a\n`.replace()` into a silent no-op.\n\nFull suite 2739 passed, same six pre-existing failures (EPLB dispatch x4,\ndspark x2). Entrypoints 533 -> 602. ruff unchanged at 1070.\n\nKnown gap, not addressed: atomesh's standalone service renders chat templates\nwithout ever threading `thinking` through. That is a missing feature rather\nthan a regression -- it never had the mechanism.\n\n* perf(openai): one copy of the read-ahead loop, and one of the tool-event dispatch\n\nA read of this branch's own diff for the two things it kept telling other code\nto do: say a thing once, and measure before believing it.\n\n**Four copies removed, all of them mine or made reachable by mine.**\n\n`marker_scanner`'s module docstring says asking this question in more than one\nplace is how the answers drift apart, and then the module asked it twice --\n`held_suffix_len` for callers holding their own buffer, and a method on the\nscanner for its own. One `_suffix_len` now, called by both.\n\nThe Anthropic streaming endpoint dispatched tool-parser events to blocks in\ntwenty-two lines, written out once for `process` and once for `flush`. That is\nthe hazard `AnthropicBlocks` was extracted to remove one level up, and it was\nalso untestable where it sat: the endpoint body is an async generator inside a\nroute handler no unit test reaches. `tool_event_frames` is a plain generator\nbecause `yield from` is a syntax error inside an async one, and whether a call\nstarted is read off the events by `starts_a_tool_call` for the same reason.\n\n`KimiParser` had grown its own lazy scanner build with the marker written out\na second time instead of read from `START_MARKERS` -- the hand-kept copy this\nbranch removed everywhere else. `_scanner` moves to the base class, which is\nwhere the declaration it reads already lives.\n\n`reasoning._hold_back_len` had become a one-line pass-through that only\nrenamed `held_suffix_len`. The rename was the thing that let the two sides\ndrift; the call sites use the real name.\n\n**Faster, measured on a whole response and not on one function.**\n\nAlternating arms, three rounds each, reasoning filter into tool parser at\nfour-character tokens: 1579 -> 1383 ns/token on plain prose, 1652 -> 1537 on\nan answer ending in a tool call. Zero overlap between arms.\n\nThree changes get there. `_plan_for` is cached on the spelling as well as the\nset, because the callers that ask once per token always pass the same module\nconstant and `sorted(set(...))` was 148 ns of the 193 it took to answer.\n`_suffix_len` rejects on one substring search over the last `longest - 1`\nbytes, since nothing can be held unless the tail *starts* a marker -- and that\nreject is a bet, not a free win: 2.4x when it fires against 1.3x slower when\nit does not, stated in the docstring because a reader deciding whether to keep\nit needs both halves. And the reasoning path gets the reject at all, which it\nhad gone without while running once per token on every stream.\n\nThe method matters more than the numbers. A single-function micro-benchmark\nsaid this change was 3.2x faster; one cross-process run of the real pipeline\nsaid it was 4.6x *slower*; alternating the arms in one process showed a\nreproducible 7-12% win. The first two were both measuring the wrong thing, and\nonly the third has arms that do not overlap.\n\n**A gap in the tests, older than this branch.**\n\nEvery shape in the streaming property corpus is a *non*-call -- text that\nmerely looks like one. So nothing ever drove a real tool call through the\nstreaming facade, and pointing every parser's scanner at another format's\nmarker broke tool calls on four formats with all 606 tests still green. That\nmutation is now caught.\n\n`REAL_CALLS` holds one real call per registered format in that format's own\nsyntax, and `test_every_format_has_a_call` fails when a format joins the\nregistry without one. Written out rather than generated: a call's payload is\nthe one thing each format spells differently, so this is auto-*detecting*\nrather than auto-generating, the same shape as\n`test_every_registered_parser_declares_its_markers` next to it.\n\nConsidered and not done: the six copies of `(content.strip() if tool_calls\nelse text)`. Collapsing them into a base-class template method needs an opt-out\nflag for Kimi-K3, which legitimately rewrites a no-call answer, plus a rename\nof `parse` to `_parse` in six files -- an exception flag and six renames to\nsave six short expressions. The rule they implement is stated once on\n`ToolCallParser.parse` and enforced across the registry by the property suite,\nso the \"a new format will forget\" argument is already covered.\n\nFull suite 2762 passed, same six pre-existing failures. 19 mutations, all\ncaught, including two new ones for the dispatch and the scanner. ruff\nunchanged at 1070.\n\n* perf(openai): send what the client can act on as soon as it exists\n\nThree things, one question: when does a streaming client actually hear\nsomething. The correctness of this pipeline was settled over the previous\ncommits; none of the below changes what is delivered, only when.\n\n**Kimi-K3 was not streaming at all.** Every K3 answer opens with\n`<|open|>response<|sep|>`, which is one of the five markers it declares, and\nthe facade reads any declared marker as \"the region has opened, the rest\nbelongs to the format\". So the whole body was buffered to end of stream --\nmeasured, 324 of 324 characters in one frame. That is the common path for the\nmodel, not an edge case.\n\n`START_MARKERS` had been answering two questions. \"Which literals must not be\nsplit across a chunk boundary\" is one; \"which of them hand the stream over\" is\nthe other, and for five formats they have the same answer, which is why nobody\nnoticed. `opens_region` asks the second one separately; only K3 overrides it,\nand only for the three markers that are channel framing rather than a call.\n\n**The silence watchdog could not see the silence it was built for.**\n`StreamOutputCollector.get` is where a stream waits for the engine, but the\nreasoning read-ahead and the tool-call read-ahead sit between it and the\nsocket, and while either withholds, `get` keeps waking on every token.\nMeasured: an answer quoting a tool marker fed 126 tokens, sent the client 6\nframes, and the gauge read zero.\n\nMoved to the frame -- `_client_stream` times the gap before each yield. That\nalso retired the \"ignore the first wait\" rule, which existed because at the\ncollector the first wait is admission and prefill and would have made the\ngauge a queue-depth proxy `atom:requests_waiting` already provides; out here\nthe opening frame goes out immediately, so every gap after it is real. And it\ncovers the Anthropic endpoint, which `_logged_stream` never wrapped: the name\nwas the bug, since it wrapped what wanted logging rather than what wanted\nwatching.\n\n**The tool's name no longer waits for its arguments.** A region is buffered\nuntil it closes, so on a 20 KB file write the client learned *which* tool was\nbeing called after 5030 of 5040 tokens. Every format carries the name in its\nopener; `peek_name` is one regex each, and across all six the name now lands\nat chunk 11-21 instead of 225-248, independent of payload size.\n\nOnly for a name the request declared in `tools`. That check is what makes an\nearly name safe -- it cannot be retracted, and an answer quoting\n`<tool_call><function=NAME>` opens a region too. SGLang's cursor parsers\nannounce with no such check and will emit a call named after whatever follows\nthe tag; this branch spent a commit removing exactly that.\n\nArguments still wait for the region to close. SGLang streams those as JSON\nfragments and a truncated response then leaves the client holding an\nunterminated object. Taking only the name keeps one new failure mode instead\nof two, and it is the part a client can act on.\n\nThat one failure mode had to be defused first, and it is why\n`starts_a_tool_call` is now `completes_a_tool_call`. `finish_reason` came from\nthe *name* event, so a response cut off mid-call would have told the client it\nhad a tool to run. A call is made when its arguments exist; six consumers now\nkey on that. The change is inert until something announces early -- every\nparser emits name and arguments together otherwise -- which is what makes it\nsafe to land in the same commit.\n\n**Two of these were found by a mutation that the tests let through, and one\nguard here prevented nothing.**\n\nWiring Kimi to announce produced two `tool_call_start` events for one call.\nThe cause was single: `_drain_entries` builds that event inline rather than\nthrough `_emit_call`, because its id and index come off the wire, so the\n\"already announced?\" check never ran -- one more hand-kept copy of a thing\nthis branch keeps consolidating. It is `_start_event` now, shared.\n\nWhile diagnosing that I also added an `_announce_used` flag against a second\ndouble-announce, wrote a comment stating that failure as fact, and did not go\nback to check once the real cause was fixed. Reverting the flag: 708 tests\npass. Constructing the shape it claimed to prevent -- two calls in one Kimi\nsection, where the buffer could still show a name after one has been drained\n-- gives byte-identical output either way, because `_drain_entries` consumes\nthe entry from the buffer. The flag is gone. A guard is justified by its\nfailure being reachable, not by nothing breaking after it is added.\n\nThe property tests needed two corrections of the same kind. \"The name arrived\nat flush\" is not \"the name was not announced early\": Kimi emits a call when\nits entry closes, which is before flush, so that judged a correct parser\nwrong. \"The name and the arguments are adjacent in the event list\" is not it\neither: the payload between them produces no events, so they are adjacent on\nboth arms and every case passed. Chunk index is the only thing that can see\nthe difference.\n\nAlso here: a real tool call per registered format, generated into the\nstreaming corpus, because every shape in it was a *non*-call. Pointing every\nparser's scanner at another format's marker broke tool calls on four formats\nwith all 606 tests green.\n\nFull suite 2845 passed, same six pre-existing failures. Entrypoints 602 -> 708.\n28 mutations, all caught. ruff unchanged at 1070.\n\n* fix(openai): announce a tool's name early only when it is safe to\n\nFifteen findings from a review of this branch, verified by running each\none before fixing it. Together in one commit because the first two are\nwhat the rest are: the early-name announcement introduced in\n`0aad7beff` was under-gated, and the formats disagreed about what\n`START_MARKERS` meant.\n\nTwo of them broke ordinary answers:\n\n  - GLM's peek regex required `<arg_key>`, so a zero-argument call was\n    skipped and the *second* call's name was announced for the first.\n    The parse then disagreed with the announcement and `_start_event`\n    raised, killing the stream mid-answer. The regex now accepts both\n    call shapes.\n  - Kimi's state 1 dropped the section body and moved to a state that\n    swallowed everything after it: a 4-character-chunked answer\n    delivered 26 of 135 characters. It now keeps what it did not parse\n    and returns to state 0.\n\nTwo were quadratic or unbounded: `announce` re-ran the format's regex\nover the whole region on every chunk (3.0 -> 9.8 -> 36 -> 137 ms across\n2k/4k/8k/16k tokens -- the shape `marker_scanner` exists to retire, put\nback one layer up), now bounded to a 256-byte window with an exhaustion\nlatch, 8.2 ms at 16k; and `chat_template_source` read ATOM's own source\nrather than the model's 27852-byte encoder.\n\nThe rest, each with a test that fails without the fix:\n\n  - Prose that merely quotes `<function=NAME>` was salvaged into a\n    phantom call. The unclosed branch now requires both a declared name\n    and the format's own next structural token.\n  - Kimi-K2 announced every call at index 0 -- its index and id come off\n    the wire, so it cannot peek. It no longer announces.\n  - Anthropic emitted a `tool_use` block with no id and no name when\n    arguments arrived without a start.\n  - Kimi-K3 leaked four framing markers into content, and parsed its\n    arguments twice because `parse`'s cut and `opens_region` disagreed\n    about which marker opens a region.\n  - A zero-argument tool reported `finish_reason: stop`.\n  - The silence watchdog timed queueing again, contradicting the commit\n    that removed it. `FrameWait` now takes `armed`.\n  - `glm.detect` claimed Hermes templates; it now keys on `<arg_key>`,\n    and Qwen3-8B honestly resolves to no parser.\n  - The reasoning toggle was resolved for \"off\" and hardcoded for \"on\".\n    The table is now `(name, off_value, on_value)` and both directions\n    go through the resolved name.\n  - atomesh forwarded unknown parser names to the router and dropped\n    ATOM's own; the two vocabularies now coexist.\n  - `RUF059` at test_tool_parser.py:118 (CI-blocking).\n\nTwo of the guard tests written for earlier fixes were vacuous -- they\nasserted a source literal was absent, and the mutation that restores\nthe bug does not restore the literal. Both are now behavioural, with\nan AST check for the shape rather than the spelling.\n\n836 entrypoints tests (from 708); 2973 passed overall with the six\npre-existing GPU failures; ruff 1069, one below the baseline.\n\n* fix(openai): one rule per question, where two were being answered separately\n\nFive findings from the last review, and three problems that predate it. They\nare one commit because they are one shape: a rule written down in one place\nand then written again, differently, somewhere else.\n\n**`tool_choice: \"none\"` deleted the answer instead of suppressing the call.**\nEnforced at the twelve places an event is *sent*, across two endpoints, while\nthe parser went on consuming the region -- so a 95-character answer reached\nthe client as its first six, no event, `finish_reason: stop`. The rule moves\nto the one place the parser is chosen, which is also the correct reading: the\nrequest said this cannot be a call, so it is prose. `/v1/messages` reads the\nfield too now; it parsed `tool_choice` off the request and used it nowhere, so\na client that forbade tool calls got `tool_use` blocks anyway.\n\n**`peek_name` and `parse` disagreed about what a call looks like.** Each\nformat wrote the rule twice -- a follower set in a peek regex, a truncation\ntest in `parse` -- and four of five drifted. Qwen's peek accepted\n`</tool_call>`, which closes the *outer* wrapper and leaves the `<function=`\nblock open, so `parse` read the same bytes as prose and a name went out for a\ncall that never came: `arguments: \"\"` on the OpenAI path, which every agent\nloop hands to `json.loads`, and on `/v1/messages` a syntactically complete\n`tool_use` block a client cannot tell from a real zero-argument call. Each\nformat now keeps one tuple of what may follow the name and both callers test\nagainst it. `peek_name` takes `tools` for the same reason `parse` does:\nMiniMax names a parameter by its own tag, so `<city>` and `<br>` are the same\nshape without the schema.\n\n**The reasoning split diverged from the stream, twice.** `</think>` was\nmatched only at position 0, so a model that answers, opens a `<think>` block\nand answers again had it extracted when streamed and handed over as literal\ntags with the chain of thought in `content` when not. And both halves were\n`.strip()`ed -- which is the trailing-newline bug `ToolCallParser.parse`\nalready documents, one stage earlier. A model writes `</think>\\n\\nThe answer.`\nand `stream=true` delivers `\"\\n\\nThe answer.\"` at every real chunk size while\n`stream=false` delivered `\"The answer.\"`. Measured over 12544 (dialect, shape,\nchunking) comparisons: 50% byte-agreement on content before, 100% after. The\nfilter's own `lstrip(\"\\n\")` went with it -- it saw only what happened to be\nbuffered when the marker arrived, so there was no chunk-invariant behaviour\nthere for the other path to match.\n\n**A marker that is a prefix of another can change the handover.** Longest-first\nsettles a tie only between markers already complete in the buffer, so which of\na prefix pair fires depends on where the chunk boundary landed -- and `_plan`\nclaimed otherwise. Harmless while both halves agree about opening a region,\nwhich every pair today does; with a synthetic pair that disagrees, one text\nproduced two different answers across six chunk sizes. Guarded at the registry\nrather than fixed in the scanner: the stronger rule costs every such marker a\nwait, and no format needs it yet.\n\n**`--request-log` broke the last frame of every OpenAI stream.** `serving_chat`\ncoalesces finish + usage + `[DONE]` into one send on purpose, and the logger\nran `json.loads` over the whole send. `Extra data:` came out of the generator,\nso `[DONE]` never reached the client. Anthropic frames put `event:` on the\nline above their data and were not logged at all. Frames are split now, and a\npayload that will not parse is logged as text: this is the diagnostic path and\nmust not be why a response fails.\n\n**A buffered region cost what it was squared.** `self.buf += text` on an\n*attribute* is quadratic in CPython -- the instance dict holds a reference, so\nthe in-place path never applies and every chunk copies the whole buffer. A\n128 KB tool call at four characters a chunk: 25.3 ms of event-loop CPU, and\nper-KB cost still climbing. A list and a join, plus a bounded head so\n`announce` never needs the region materialised: 9.8 ms, and flat per KB. The\nsame loop over a *local* string is linear, which is why measuring `s += x`\nfinds nothing.\n\nEvery fix has a test that fails without it, and the mutation harness in\n`/app/logs_claude/` runs 72 arms green. Three guards written along the way\nwere vacuous and had to be rebuilt: two derived their corpus from the code\nunder test, and one keyed on \"did `parse` hand the region back unchanged\",\nwhich is unsound for Kimi-K3 -- it rewrites the content of every answer.\n\n963 entrypoints tests (from 903); 3100 passing overall with the six\npre-existing GPU failures; ruff one below baseline.\n\n* fix(openai): the two halves of a rule, told apart by when they run\n\nFifteen findings from a review of the previous commit. Six are that commit's\nown; nine predate it and sit in code it rewrote.\n\n**A follower has to have arrived.** `peek_name` and `parse` share the test for\n\"is this this format's own next token\", and it accepted a *prefix* -- so `<`,\none character into `<br>`, satisfied `<parameter=`. The same prose announced a\ntool at chunk sizes 1 and 2 and stayed silent at 5, and an announcement is\ndelivered as a dispatchable zero-argument call. The two callers want opposite\nthings and now say which: `parse` runs at end of stream, where a token cut\noff part-way is all there will ever be; `peek_name` runs mid-stream, where a\nprefix means \"not yet\". One function, one `arrived` flag, four copies gone.\n\n**GLM's peek searched past its own first call.** `parse` reads the unclosed\ncase from the first `<tool_call>` to end-of-string, so when that one carries\nno usable name it produces nothing at all, while the peek slid forward and\nannounced a later opener's name -- at every chunk size. The peek is anchored\nto the same opener. The docstring calling GLM \"the one format whose peek and\n`parse` already agreed\" was wrong.\n\n**One call arrived as two `tool_use` blocks with one id.** Carrying the open\ncall across a block close let `delta` re-open, and the first block had no\ninput -- a client iterating content blocks runs `get_weather({})` and then\n`get_weather({\"city\": \"Paris\"})`. Nothing closes a block between the name and\nthe arguments, so surviving a *batch* was all that was ever needed; the\ninterleaved shape is unreachable from any registered parser, measured over\nevery format's real call and every chunking.\n\n**atomesh discarded events it had no room for.** The drain takes `max_items`\nand the build loop `break`ed at that count, after the parser had yielded --\nso at `max_items=1` a tool call lost its arguments permanently, and once\n`has_tool_calls` moved onto the argument event it also reported\n`finish_reason: stop`. Overflow is queued now. Fixing that surfaced a second\none the tests caught: the early return skipped the final chunks, leaving a\ndrained stream with no finish reason, no usage and no `[DONE]`.\n\n**An effort is not an opt-in.** `resolve_thinking` collapsed \"the request said\nnothing\" to `True`, which was harmless while the caller wrote a key no\ntemplate read. Writing the template's real switch -- merged after the server\ndefaults and after the client's own `chat_template_kwargs` -- meant a request\ncarrying only `reasoning_effort` re-enabled reasoning over an operator's\n`--default-chat-template-kwargs '{\"enable_thinking\": false}'`.\n\n**Withheld reasoning now sends a ping.** The suppression branch was a bare\n`continue`, so the socket was silent for the whole chain of thought: on an\nR1-shaped 5019-character trace the first client-visible frame arrived after\n5016 of them.\n\nThe nine that predate it: three streaming paths hardcoded `finish_reason:\n\"stop\"` while `stream=false` reported `length` for the same generation;\n`_split_channel` was three branches where one rule belongs, and the middle one\ndiscarded everything before `<|open|>response<|sep|>` -- a single byte between\nthe two markers deleted the chain of thought from both fields -- while none of\nthe three asked whether a channel had been opened, so any model *quoting* one\nof those tokens lost text; the streaming filter knew only the explicit channel\nclose, so K3's usual answer streamed entirely as `reasoning_content` with\nempty `content`; K3 stripped four token shapes it never declared, MiniMax cut\ncontent at a token its own calls do not contain and did not declare\n`<invoke name=\"`, and Kimi's `parse` dropped the text after a section and the\nmarker an answer ended on.\n\nTwo new properties. Framing agreement is swept per format from **the module's\nown string constants**, not from `START_MARKERS` -- a corpus built from the\ndeclared list cannot contain a token the format strips but never declared,\nwhich is the drift being looked for, and the first version of this guard was\ngreen for exactly that reason. And a follower must have arrived, across chunk\nsizes.\n\nTwo more vacuous guards written and rebuilt along the way, both the same\nmistake as before: one drew its corpus from the code under test, the other\nmutated a shared constant to \"create\" a disagreement that by construction\ncannot exist.\n\n1066 entrypoints tests (from 963); 3203 passing overall with the six\npre-existing GPU failures; ruff at the previous commit's baseline; the\nmutation harness in `/app/logs_claude/` runs 86 arms green.\n\n* perf(openai): stop Kimi paying for its buffer twice on every chunk\n\nThe sweep that put the buffered formats on a list accumulator missed this\none: `KimiParser` is not a `BufferedMarkerParser` -- its token format is\nself-delimiting, so it drains per completed entry and implements `process`\nitself -- and it kept `self.buf += text`. On an attribute that is quadratic;\nit also re-scanned the whole buffer for a section end on every chunk, which\nis a second factor on top.\n\nMeasured at four characters a chunk, against Qwen's shape for the same\npayload:\n\n     payload    before     after     qwen\n       16 KB     2.9 ms    2.0 ms    1.3 ms\n       32 KB    12.4       3.8       2.5\n       64 KB    97.0       7.3       4.9\n      128 KB   427.8      15.0       9.9\n       ms/KB   0.181 -> 3.342     0.117 flat\n\n428 ms is the event loop, so it is every concurrent stream, not one.\n\nTwo changes, and the second is what removes the rescan: the region goes into\n`Region`, and it is materialised only when a chunk carries one of the two\nmarkers that can end an entry or the section -- with a tail so a marker split\nacross the boundary still counts. Until one arrives there is nothing in the\nbuffer to look at. `flush` takes whatever `process` never examined, because\nat end of stream there is nothing more coming.\n\nThe complexity property now covers both accumulators. Its first Kimi arm did\nnot catch the regression: the mutation left the lines that hand the remainder\nback, so the buffer never actually grew -- a revert that reverts nothing. The\narm that does catch it removes the laziness instead.\n\nTwo things this review looked at and left alone, with the numbers, so the\nnext person does not re-derive them:\n\n`ReasoningFilter`'s two `self.buf += text` sites read like the same bug and\nare not -- both branches drain to at most a marker's length every chunk, so\nthe concatenation is on a short string. Measured flat at 0.202 and 0.222\nms/KB from 16 KB to 128 KB. `+=` on an attribute is quadratic when the buffer\n*grows*, which is the part a grep cannot see.\n\nA fast path for plain prose in state 0 -- the common case allocates three\ntimes where once would do -- measures 0.197 -> 0.166 ms/KB, 16% of that stage\nand 8% of the pipeline. Declined: the pipeline is ~1.5 us/token against a\nfrontend whose wave-boundary CPU is measured to go elsewhere, so this is\n0.1% of it, and it buys that by adding a third place that knows what a marker\nlooks like.\n\n1068 entrypoints tests; 3205 passing overall with the six pre-existing GPU\nfailures; 87 mutation arms green.\n\n* refactor(openai): one reader per wire format, for both delivery modes\n\nThree rounds of review on this branch found fifteen defects each, and about\nforty percent of every round was introduced by the round before it. That rate\nis the finding. The cause is that a tool-call format was read twice -- a\nclassmethod `parse` over a complete output, and a `process`/`flush` state\nmachine of its own -- and both had to decide where content ends, whether an\nunclosed tag is a call, what a region that parses to nothing means, and which\nbytes are framing. Six formats, two copies, four rules: forty-eight places for\none answer, and the reviews kept finding them disagreeing.\n\nSo the second reader is gone rather than corrected. `stream=false` is\n`read_whole`, which is the streaming engine over a single chunk. The two modes\ncannot answer differently because there is nothing left to differ.\n\nA format now declares only what is particular to it -- `START_MARKERS`,\n`opens_region`, `CALL_OPENERS`/`CALL_CLOSERS`/`CALL_FILLERS`,\n`REGION_END_MARKERS`, `RECOGNISES_A_CALL_IN_PROGRESS` -- and implements one\nmethod, `parse_region(region, tools, *, at_end) -> RegionParse(calls, begins,\nconsumed)`. It holds no state; the engine does. `begins` and `consumed`\nbracket the markup, so a region is accounted for byte by byte and neither the\nsentence before a call nor the one after it can be swallowed by a format that\nforgot to return it. `ToolCallStreamParser` owns everything else: reading\nahead, releasing content, dropping framing, \"a start marker is not a promise\",\nstamping call indices, and handing back the answer that follows.\n\n`peek_name` is gone too. The early name is the first call of `parse_region`\nover the bytes so far, with `at_end=False` -- same function, same enumeration,\nthe second run seeing a superset of the first's bytes. `at_end` is the only\ndifference between the two moments and it is now the only place that\ndifference is written. Five formats had a peek regex of their own and four\ndisagreed with their own parse; the last one skipped a self-closing\n`<invoke name=\"x\"/>` its parse returned first, putting three tool calls on the\n`/v1/messages` wire for a response containing two.\n\nThe reasoning half is the same shape. `resolve_dialect` picks one dialect at\nstartup, from the chat template's source *and* its rendered probe, and\n`ReasoningChannel` carries it with `starts_open` and one accessor per delivery\nmode. It used to be decided twice per response and differently each time: the\nnon-streaming split tried every dialect in order, the streaming filter carried\nnone and closed on the union of all their end markers.\n\nWhat that fixes, each measured before and after:\n\n  text written after a tool call            deleted on both paths, five of six formats\n  a second Kimi section                     lost, and its raw tokens shown as content\n  Kimi's per-section wire index             two calls merged into one by any index-keyed client\n  MiniMax prose mentioning `<invoke`        37 characters deleted, stream=false only\n  `tool_choice: \"none\"`                     K3's channel tokens leaked to the client\n  a `<think>` model quoting a K3 token      24 characters of the answer filed as reasoning\n  a thinking block inside a tool call       `tool_use` with `input: {}`, `stop_reason: tool_use`\n  an announced name on /v1/messages         a complete zero-argument call the model never made\n  a thinking-only Anthropic reply           no text block at all\n  atomesh's queued deltas                   unreachable past the terminal chunks\n  atomesh completions                       `stop` for a response cut off at `max_tokens`\n  a template control on a Python encoder    HTTP 500 on DeepSeek-V4 and Kimi-K3\n  `_json_safe`                              every non-streaming atomesh response, HTTP 500\n  Kimi-K3's reasoning channel               whole answer as `reasoning_content`, `content` empty\n  a request that turned thinking off        an ordinary answer returned as reasoning\n  a quoted call opener before a real call   one call named after the placeholder, the real one lost\n  a call cut off at `max_tokens`            `tool_calls`, so the client ran it with half its arguments\n\nSweeping GLM's non-nesting guard to the other four also removed a pre-existing\nquadratic: a `write_file` whose content is HTML costs 1078 ms at 32 KB before\nand 1.4 ms after, in the request coroutine.\n\nOne thing came out that went in last round. `_gave_up_on_region` released a\nregion that had produced nothing after 1 KB, to stop an answer *about* tool\ncalls being withheld to end of stream. It rests on acceptance being monotone\nin how many bytes have arrived, and that is false: MiniMax gates its\nin-progress test on the first tag being in the declared schema, and DSML's\nwrapper-less and direct-JSON branches match no prefix at all -- so real calls\nover 1 KB were delivered as raw text with `finish_reason: stop` on three of\nsix formats. It was also quadratic,…",
          "timestamp": "2026-08-23T14:34:11Z",
          "url": "https://github.com/ROCm/ATOM/commit/56d5565bdf2f5ccbfd5cf2f5ed9e43ee52e050b6"
        },
        "date": 1787503137789,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32650699543 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32650699543 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.18,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32650699543 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.99,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7589,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32650699543 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7566 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8832,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32650699543 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608221457 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.348 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
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
          "id": "b8dcfe09535b0e02752409730a3a638e13d47f7d",
          "message": "fix(engine): include DP dummies in staging lifetime (#1993)\n\n* fix(offload): support DSV4 FP4 indexer pages\n\n* fix(engine): include DP dummies in staging lifetime",
          "timestamp": "2026-08-24T15:42:16Z",
          "url": "https://github.com/ROCm/ATOM/commit/b8dcfe09535b0e02752409730a3a638e13d47f7d"
        },
        "date": 1787589952925,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32749651947 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608241610 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32749651947 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608241610 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.12,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32749651947 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608241610 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7536,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32749651947 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608241610 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7521 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8939,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32749651947 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608241610 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3821 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
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
          "id": "4e0848dcdf94f3d4be0c1e3ce51db99bc93f8b8c",
          "message": "[fix](dspark): keep Kimi-K3 KDA out of the piecewise cudagraph (#2004)\n\nServing Kimi-K3 with a DSpark drafter died during capture with \"HIP error:\noperation not permitted when stream is capturing\".\n\nDSpark makes a capture batch carry 1 + num_spec tokens per request, which\ndecode_threshold=1 reads as prefill, so capture takes the chunked KDA path.\nThat path builds chunk_indices from the batch's own cu_seqlens via a host\nreadback -- illegal mid-capture, and frozen into every replay if the read is\ndodged. Plain K3 has one token per request, is classified as decode, and\nnever reaches it.\n\nMark the mixer with @eager_break_during_capture so it runs eagerly and is\nre-executed per replay against live metadata, matching vLLM's own Kimi GDN\nlinear attention. The decorator rejects ops returning a fresh tensor -- its\naddress moves each replay and breaks downstream segments -- so the plugin\nregisters its own write-into-output op rather than widening the shared one,\nleaving the native ATOM and SGLang paths untouched.\n\nFull decode graphs are unaffected: they reach the builder through\nbuild_for_cudagraph_capture, which synthesises the draft counts and lands on\nthe fused spec path, so KDA stays inside those graphs.\n\nAlso drops DCP from the Kimi-K3 CI entry -- vLLM rejects it outright for this\nmodel in config/speculative.py, so the entry could never start -- and moves\nextra_args after the harness defaults so an entry can override them.\n\ngsm8k 5-shot, TP8, N=7: 0.9522 flexible / 0.9515 strict, against 0.9553 /\n0.9560 for plain K3; draft acceptance 48.5%.\n\nCo-authored-by: perzhang <perzhang@amd.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-25T09:19:20Z",
          "url": "https://github.com/ROCm/ATOM/commit/4e0848dcdf94f3d4be0c1e3ce51db99bc93f8b8c"
        },
        "date": 1787676150239,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9477,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32870878654 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608251555 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9469 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32870878654 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608251555 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.18,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32870878654 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608251555 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.99,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8764,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32870878654 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608251555 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3662 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "carlushuang",
            "username": "carlushuang",
            "email": "carlus.huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ae38bd9ca40e3d92c293cc421d72bd9f81cbc0c2",
          "message": "[atom] Add diffusion subsystem and MiniMax-H3 video+audio generation (#1836)\n\n* [atom] Add diffusion subsystem and MiniMax-H3 video+audio generation\n\nATOM has been an autoregressive LLM engine: registry keyed on *ForCausalLM,\nprefill/decode with a KV cache and a BlockManager. A denoise loop shares\nalmost none of that -- no KV cache, one fixed N-step loop per job instead of\ncontinuous batching, minutes per request instead of milliseconds per token,\nfour heterogeneous networks instead of one, and sequence parallelism across a\n*single* request rather than tensor parallelism over a replicated batch.\n\nSo this adds `atom/diffusion/` as a sibling subsystem rather than extending\n`atom/model_engine/`, with MiniMax-H3 as its first model.\n\n    atom/diffusion/\n      config.py request.py            DiffusionConfig / DiffusionJob\n      stages/ pipelines/              stage contracts + ComposedPipeline\n      distributed/ulysses.py          sequence-parallel all-to-all\n      layers/attention/               asm | triton | sdpa varlen FMHA\n      models/                         DiT, VAEs, text encoder, scheduler\n      engine/                         scheduler, runner, ZMQ workers, engine\n      entrypoints/                    async /v1/videos server\n      postprocess/mux.py              H.264 + AAC stereo MP4\n\nEverything outside `atom/diffusion/` is additive: a `[diffusion]` extra for\nPyAV/Pillow/torchaudio, those same test deps in pre-checks CI, the subsystem in\nCLAUDE.md's index, and recipes/MiniMax-H3.md. No existing behaviour changes.\n\nAll three MiniMax-H3 tasks are validated against the upstream sglang reference\non 4x MI308X (gfx942), same seed, Triton attention:\n\n    task     packed seq   conditioning                    PSNR      SSIM\n    t2va         37,760   --                            41.48 dB   0.963\n    fl2va        39,808   1,008 keyframe rows           40.66 dB   0.970\n    ref2va       52,544   7,168 image + 414 audio rows  41.52 dB   0.969\n\nComponent-level, all measured rather than asserted: DiT forward at steps 0 and\n45 bit-exact (max_rel_err 0.000e+00), 535/535 weights, packed layouts\nvalue-exact for all three tasks including the fp64 position grid, 45 steps of\nthe full loop bit-exact, fl2va conditioning rows to mean |diff| 5.4e-7. The\n~41 dB residual is H.264 re-encode loss between two independently encoded\nvideos.\n\nServing is asynchronous by contract, because the work is: POST /v1/videos\nreturns a job id in milliseconds while the generation runs for minutes.\nWorkers are spawned per rank behind ZMQ; a request fans out to every rank\nbecause Ulysses is collective, and only rank 0 -- which decodes and muxes --\nreplies. Admission rejects with 429 past the queue cap rather than queueing:\nwith 4-minute jobs an unbounded queue is an unbounded invisible wait. t2va and\nfl2va were both driven end to end over HTTP on 4 GPUs, including clean shutdown\nwith no stranded VRAM.\n\nMemory is the one place the design had to change on contact: the 67 GB Qwen3-VL\ntext encoder is staged in host memory and moved onto the device only for the\nencode it performs once per request. Left resident alongside the DiT it\noverflows a 192 GB card -- the first served request died with 182 GiB\nallocated.\n\nFour ROCm-specific traps worth naming, all found the hard way:\n\n* `tensor.is_cuda` is True for HIP tensors. Upstream gates three CUDA-only JIT\n  kernels on it and each is fatal under hipcc even though the correct eager\n  fallback is the next statement.\n* Attention must dispatch on `q.device.type`, not on whether aiter imports --\n  aiter imports fine in a CPU-only process and then dies inside the kernel.\n* The video VAE emits ImageNet-normalized pixels; decode must finish with the\n  checkpoint's transform_rev. Skipping it is invisible to every structural\n  check and costs ~22 dB.\n* Do not copy upstream's USE_AITER_GFX942 Triton fallback as a default: the\n  ASM varlen path matches the tuned fixed-length kernel here (124.0 vs 123.9\n  TFLOP/s), so the workaround costs ~20% for nothing. Triton stays selectable\n  because it is the only backend that reproduces the reference bit-for-bit.\n\n247 CPU-only tests (no GPU, no AITER), black and ruff clean. Recipe in\nrecipes/MiniMax-H3.md.\n\n* [atom] Condense diffusion docstrings and comments\n\nThe subsystem's total prose volume was in line with the repo (0.30 vs 0.25\nprose/code) but its composition was not: 2.7 docstring lines per comment line\nagainst the repo's 0.9, i.e. long narrative docstrings explaining rationale and\nhistory where a contract would do.\n\nCuts 211 prose lines, almost all from module docstrings and multi-line comment\nblocks, bringing prose/code to 0.25. Kept: measured constants, the ROCm traps,\nthe RNG contracts, and anything a reader would otherwise have to re-derive.\n\nAlso fixes a docstring that had gone stale -- pipelines/minimax_h3 still said\nref2va was refused rather than implemented.\n\n* [atom] Restructure atom/diffusion to be model-major\n\nThe layout mirrored sglang's `multimodal_gen/runtime/`, which groups by\ncomponent type (dits/, vaes/, encoders/, schedulers/). That is right for a repo\nhosting ~30 models that share components; with one model it only fragments.\nMeasured before this change:\n\n* 65% of the subsystem (4,433 of 6,845 lines) was MiniMax-H3, spread across\n  seven directories, so adding a second model would touch all seven;\n* `stages/` contained **zero** PipelineStage subclasses -- all eight lived in\n  `pipelines/minimax_h3.py`, and the 2,155 lines under `stages/minimax_h3/`\n  were a pure-function library. The name was simply false;\n* eight of sixteen directories held exactly one file, and `layers/` held none\n  (a directory containing a directory containing one file);\n* `models/loader.py` was fake-generic: 99 lines, one H3 function, importing\n  from `models/dits/minimax_h3` -- a shared layer depending on a model.\n\nNow the framework sits at the top level and each model is one package:\n\n    atom/diffusion/\n      config.py request.py pipeline.py attention.py ulysses.py mux.py\n      engine/ entrypoints/ examples/\n      models/minimax_h3/   arch, dit, vae, text_encoder, scheduler, loader,\n                           pipeline (with its stages), and the H3 helpers\n\nSixteen directories become six, every single-file directory is gone, the stages\nlive with the pipeline that runs them, and the loader is honestly H3-specific.\nThis also matches ATOM's own convention -- `atom/models/` is one file per model\nfamily, not a split by layer type.\n\nThe cost is losing path-parity with sglang, which made side-by-side diffing of\nported files easy. Judged not worth seven directories per model.\n\nPure relocation: no behaviour change, no signature change beyond\n`stages/base.py` and `pipelines/base.py` merging into `pipeline.py` (they were\nmutually dependent). 247 tests pass and the server's pipeline path resolves and\nconstructs.\n\nA component graduates out of a model package into the shared layer when a\nsecond model uses it, not in anticipation.\n\n* [atom] Decode the video VAE in bf16 (3.3x faster, parity unchanged)\n\nDecode took 91.8 s against 486 s of denoise -- a sixth of the request, never\nnoticed because every parity run measured pixels rather than time.\n\nThe first guess was wrong. Poor MIOpen Conv3d is a known ROCm problem, so\nCausalConv3d looked like the obvious suspect; profiling says convolution is\n**0.0%** of decode. H3's video VAE is\ntransformer-based, not a CNN: 39.7% of decode is `aten::addmm` and 6.3% is\nattention. The checkpoint ships fp32 weights, so we were running 69 s of fp32\nGEMMs.\n\nMeasured at 1344x768x124f:\n\n    fp32   88.4 s\n    bf16   24.4 s          agreeing to 51.4 dB, mean |diff| 0.49/255\n\n51 dB is an order of magnitude inside the 41 dB bar the pipeline is validated\nat -- that bar is set by H.264 re-encode loss between two independently encoded\nvideos, not by the VAE.\n\nEnd to end on the captured t2va state, Triton attention, against the same\ngolden:\n\n                    decode     PSNR       SSIM\n    fp32 (before)   91.8 s     41.48 dB   0.963\n    bf16 (after)    27.8 s     41.47 dB   0.9633\n\nSo 64 seconds off a 585 s served request -- 11% -- for no measurable quality\nchange. Encode still runs fp32: that is the reference's recipe and the\nposterior is sampled there.\n\n* [atom] Stop copying the text encoder back to host (12.7 s -> 1.0 s per request)\n\nStaging the 50 GiB encoder off the GPU is what makes a 4-GPU replica fit, but\nthe way it was written cost 12.7 s of every request: 5.0 s to upload and 7.7 s\nto copy back, both through pageable memory.\n\nThe copy back is pure waste. Weights are read-only under inference, so the host\ncopy is already authoritative -- releasing only has to drop the device copy and\nrestore the pointers. Pinning the host side at load then makes the upload a DMA.\n\n    before   H2D 5.01 s + D2H 7.71 s  = 12.72 s per request\n    after    upload 0.99 s + release 0.02 s = 1.02 s per request\n\nPinning (~11 s) happens once in load_components rather than lazily, so no\nrequest pays it. Verified the rows are bit-identical across three stagings and\nthe encoder returns to the host each time.\n\nt2va at Ulysses-4 is now ~476 s end to end: ~23 s text encode, 1.0 s\nstaging, 424.7 s denoise, 27.7 s decode.\n\n* [atom] Skip the padding segment in attention (-30 s, output identical)\n\nThe packed sequence rounds up to a 64-row boundary and hands the leftover rows\nto attention as a second segment: `cu_seqlens = [0, 37736, 37760]`, where the\ntail is 24 rows of pure alignment padding whose output is discarded.\n\nThat is not free. rocprofv3 gives the ASM kernel's grid as\n`(heads, num_segments, ceil(max_seqlen / 256))` -- 512 threads per workgroup,\nBLOCK_M 256:\n\n    1 segment    grid (7168, 1, 148)  -> 14 x 1 x 148 = 2,072 workgroups\n    2 segments   grid (7168, 2, 148)  -> 14 x 2 x 148 = 4,144 workgroups\n\nZ is sized from `max_seqlen`, not from each segment's own length, so the 24-row\nsegment gets a full plane of 2,072 workgroups of which exactly one has work.\nThe grid doubles; the work does not.\n\n    1 seg  37,736            80.87 ms   125.9 TFLOP/s\n    2 segs 37,736 + 24       93.00 ms   109.7 TFLOP/s\n    2 segs balanced x2       40.60 ms   125.5 TFLOP/s   (full speed)\n\nThe balanced case is what identifies the cause: it is not the segment count,\nit is `max_seqlen x num_segments` overshooting the real work.\n\nFix: pass `pad_from` and drop those rows from the kernel call, writing through\naiter's `out=` into a full-length buffer so nothing is copied. Bit-exact by\nconstruction -- the padding already sat in its own segment, so no real token\nattended to it either way.\n\nMeasured, t2va at Ulysses-4 with ASM:\n\n    attention   93.00 -> 80.87 ms per layer   (-13.0%)\n    denoise     424.7 -> 394.6 s              (-30.1 s, -7.1%)\n    output      PSNR inf, mean|diff| 0.000, SSIM 1.0000 against the same\n                configuration without the fix -- pixel-identical\n\nAlso verified: the kernel's real rows are bit-identical under torch.equal, the\n5-step bisect against the reference capture is unchanged (video cos 0.99999994,\naudio 1.00000000, max|rel| 1.847e-04), and the padded tail is zeroed rather\nthan left as uninitialised memory -- this codebase has already produced one\nintermittent-NaN bug from a torch.empty buffer.\n\nASM only: Triton measures 104.78 vs 104.66 ms, no benefit.\n\n* [atom] Distribute video VAE decode across ranks (27.5 s -> 4.1 s)\n\nDecode was flat at ~27 s whether the replica had 4 GPUs or 8, because\nDecodeStage was MAIN_RANK_ONLY -- rank 0 decoded alone while the others\nidled.\n\nThe fix turned out to be almost entirely wiring. The checkpoint's bundled VAE\n*already* implements tiled decode and the rank sharding for it --\n`_local_tile_indices(num_tiles, sp_rank, sp_size)` and\n`_all_gather_tiled_results` -- and `from_pretrained` enables it from the\nconfig. What it cannot discover is our process group, so it seeds a\nsingle-process state:\n\n    sp_size = 1, sp_rank = 0, sp_process_group = None\n\nwhich tiles the work and then runs every tile on one rank. Pointing that state\nat the Ulysses group, loading the video VAE on every rank (~5 GB in bf16), and\nmaking DecodeStage REPLICATED so no rank skips the gather is the whole change.\n\nMeasured, t2va at Ulysses-8:\n\n    video decode   27.5 s -> 4.1 s     (6.7x)\n    output         PSNR inf, mean|diff| 0.000, SSIM 1.0000 against the serial\n                   decode -- pixel-identical\n\nDecodeStage is REPLICATED rather than MAIN_RANK_ONLY because the tiled decode\nis a collective: a rank that skips it hangs the rest in the all-gather. Audio\nstays serial on rank 0 -- 3 s of work on a 0.6 GB model, and\n`components.get(\"audio_vae\")` is already None off the main rank.\n\n* diffusion: warm the DiT at load, not inside the first request\n\nThe first DiT forward in a fresh process costs far more than the rest. On\ngfx950 at Ulysses-8 the token refiner's first attention forward is 8.9 s of\naiter kernel JIT and the first denoise step is 1,373 ms against 563 ms for\nevery later one -- roughly 10 s the first generation paid for no model work.\nMeasured, paired, same harness:\n\n                              --no-warmup    default\n  warmup at load                      --      10.8 s\n  rope + refine_prompt_embeds      8.9 s       0.0 s\n  denoise step 1                  1373 ms      552 ms\n  first denoise block             13.0 s       3.4 s\n\nComposedPipeline.warmup() is opt-in per pipeline -- a generic \"run the\npipeline once\" would need a real prompt and would decode and mux a throwaway\nfile. MiniMaxH3Pipeline runs one step of the real denoise loop on zeros at the\nreleased 1344x768 / 5.17 s geometry, so what gets warmed is what the first\nrequest will actually call rather than a synthetic forward that drifts from it.\n\nImplementations must be collective and identical on every rank: this runs\ninside the denoise process group, so a rank that skips hangs the ones that do\nnot. A failure is logged rather than raised -- the same work reruns on the\nfirst request, where the error is attributable to a job instead of killing a\nreplica that just spent minutes loading.\n\nThis reverses the pre-PR audit, which removed warmup() as an unkept promise\n(\"called by nothing; the server has no geometry to warm at\"). That reasoning\nheld on gfx942, where the cost was buried under 6x slower kernels; on gfx950\nit is the largest single item left in a first request.\n\n* diffusion: consolidate the model package from 17 files to 7\n\nAdding a second model meant adding another 17 files, most of them 100-200\nlines. Group them by role instead, so a model is seven files:\n\n  arch.py          architecture config\n  dit.py           the network\n  components.py    both VAEs, text encoder, weight loading\n  layout.py        geometry, packed sequence, patchify, initial latents\n  conditioning.py  keyframes, references, noise aug, presentation\n  denoise.py       the loop and its rectified-flow sampler\n  pipeline.py      the 8 stages and the pipeline\n\nMechanical: bodies move unchanged, imports are unioned, callers are rewritten.\nNo behaviour change and no new abstractions.\n\nThe alternative -- promoting the generic-looking pieces to a shared layer\nahead of a second model -- does not survive inspection. Weight sharding sits\nin the same file as H3's grouped-QKV reorder, patchify beside H3's packed\nscatter, rectified-flow Euler under minimax_h3_* names. Each would have to be\nsplit first, and the result would be a shared API with exactly one caller.\nThey graduate when a second model uses them.\n\nAlso drops the vestigial config surface: ComponentConfig, the components list\nand tp_size were parsed and never read (the H3 loader hardcodes its subfolders\nand the DiT runs at tp_size 1), so a second model author would have had to\nreason about all three for nothing. ulysses_degree now simply has to equal\nnum_gpus, which is what the code already assumed.\n\nComment density is down to 0.21 prose lines per code line, the lowest of any\nATOM subsystem (model_engine 0.34, models and model_ops 0.26).\n\n253 diffusion tests pass; the five that covered the removed config surface are\ngone with it.\n\n* diffusion: trim the test suite\n\n253 tests for 7.3k lines was 0.51 test:code against 0.14 for the rest of ATOM\n-- 3.6x the repo norm, and the first thing a reviewer sees.\n\nTwo changes:\n\nDropped 31 exception guards that can only be triggered by our own code:\npatchify on an indivisible grid, denormalize with mismatched channels, an\nunknown ref2va block kind. Guards reachable from a request are kept -- task\nvalidation, conditions without a uri, unaligned resolutions, reference media\nwith degenerate dimensions -- as are the framework stage contracts a second\nmodel will lean on.\n\nFolded two genuinely uniform families into parametrized tables: backend\nresolution (5 -> 1) and reference image shape (4 -> 1). The second is\nstronger than what it replaced, since every invariant now runs against every\ninput rather than one property per case.\n\n223 tests, 3447 LOC, ratio 0.47. Short of where a mechanical count-reduction\nwould land, because most of the remaining short tests are not a family --\nthey check different functions with different assertions, and collapsing them\ninto a table of lambdas trades clarity for a smaller number.\n\nEvery regression test for a bug that actually happened is untouched: the\nuninitialised RoPE buffer that emitted NaN on ~1 run in 15, the queued job\nresurrected after a rank died, the ImageNet de-normalisation worth 22 dB.\n\n* diffusion: warm the VAEs as well as the DiT\n\nBenchmarking at the cookbook's 209-frame geometry showed decode taking 14.5 s,\nwhich did not survive a second look: decoding the same latents three times on\n8x gfx950 gives\n\n    video   4.524 s, 1.375 s, 1.370 s\n    audio   5.676 s, 0.078 s, 0.077 s\n\nSo ~9 s of that 14.5 s was first-call kernel setup, not work -- the same thing\nthe DiT warmup was added for, and proportionally worse. Warm decode is 1.45 s.\n\nwarmup() now runs both decoders once after the denoise step. The canvas and\nrow count come from the geometry it is given rather than from WARMUP_GEOMETRY,\nso a caller that warms a different shape gets a decode that matches it.\n\nWorth noting for anyone reading a diffusion benchmark: a run that decodes once\nreports setup as if it were throughput. Ours did, until it was checked.\n\n* diffusion: clear requires_grad when placing components\n\nRunning the scaling matrix at Ulysses-1 crashed in aiter:\n\n    File \"aiter/ops/mha.py\", line 2916, in forward\n        assert return_lse\n\naiter refuses a varlen attention whose inputs require grad. `.eval()` does not\nclear the flag, and the loaded weights carry it, so every q/k/v in the model\nrequires grad. At Ulysses >= 2 the all-to-all writes into a fresh `torch.empty`\nand launders the flag away before attention ever sees it -- which is why every\nvalidated topology (2, 4, 8) missed this and degree 1 fails immediately.\n\nplace_components() now calls requires_grad_(False), which is what an inference\nrunner should have been doing regardless.\n\nReproduced in isolation: one Linear + one aiter varlen call asserts with the\nflag set and passes without it.\n\n* diffusion: fuse RMSNorm with the indexed AdaLN modulation\n\nEach DiT block normalises, then applies a per-token affine gathered from a\nsmall modulation table, twice. Unfused that is: write the normalised\nactivation and read it straight back, and materialise both table gathers at\n[tokens, hidden] -- 680 MB each at a 209-frame request.\n\natom.diffusion now routes both call sites through aiter's\nfused_rmsnorm_indexed_adaln, which reads the activation once and writes the\nresult once, with the table served from cache. On 63,232 x 5376 bf16:\n\n    eager (this path before)   1.708 ms\n    fused                      0.329 ms      5.2x\n\nEnd to end at 209 frames, 1344x768, 50 steps on MI355X:\n\n    Ulysses-4   111.52 -> 110.11 s\n    Ulysses-1   387.42 -> 381.35 s\n\nNumerics move by bf16 rounding, not more: against the eager path on real\nweights, one full step gives cos 0.9999999 and max_rel 1.0e-4 on both the\nvideo and audio predictions. The fused path keeps the row in fp32 across the\nnorm and the affine, so it is nearer the fp32 result than the eager path is,\nnot further -- the same category of difference as the ASM/Triton attention\nchoice, which is documented as different-but-equally-valid.\n\nFalls back to the eager expression when aiter is absent or the input is not\nCUDA bf16, so CPU tests and non-ROCm environments are unaffected.\n\n* diffusion: fuse QK-Norm with 3-D RoPE\n\nThe aten-level trace of a Ulysses-1 step showed the remaining elementwise work\nwas not mostly AdaLN -- it was QK-Norm and RoPE:\n\n    aten::mul  175.5 ms   aten::cat  157.0 ms   aten::add  70.7 ms\n    aten::neg   22.3 ms   aten::_fused_rms_norm  290.3 ms\n\nUnfused, RoPE broadcasts the token's cos/sin row into a [tokens, heads, 128]\ntemporary, then spends a slice, a negate, two concatenates and two multiplies\nper tensor, on top of a separate per-head norm.\n\natom.diffusion now routes both through aiter's fused_qk_norm_rope_cached, one\nprogram per token: a token's heads are contiguous, so the [H, D] tile is one\ncoalesced run and the cos/sin row is read once for all 56 heads. It writes\nthrough the qkv projection's own storage, so the split views are never\nmaterialised either.\n\n    eager   11.690 ms      fused   1.262 ms      9.3x   (63,232 x 56 x 128)\n\nEnd to end at 209 frames, 1344x768, 50 steps on MI355X:\n\n    Ulysses-1   381.35 -> 355.07 s\n    Ulysses-4   110.11 -> 103.13 s\n\nWith both fusions active, one full step against the fully eager path on real\nweights gives cos 1.0000000 with max_rel 1.3e-4 -- bf16 rounding, and on the\naccurate side of it, since the fused path holds the row in fp32 across the\nnorm and the rotation.\n\nThe existing aiter rope ops do not cover this shape: they assume the rotated\nsubspace is the whole head or half of it, and H3 rotates 96 of 128. The\ncache-write variants additionally want a paged KV cache, which a diffusion\nmodel does not have.\n\n* diffusion: consolidate duplicated MiniMax-H3 layout and conditioning helpers\n\nThe H3 package accumulated near-duplicate code as the t2va and ref2va paths\nwere added side by side. Both packed-sequence builders re-derived the\nalignment arithmetic, the per-frame spatial grid and the audio w-axis pinning;\nboth visual conditioning paths repeated the same seeded fp32 posterior-sample\nencode; and dit.py still carried four single-use RoPE and QK-norm wrappers\nfrom before the aiter fusions replaced them.\n\nFactor the shared pieces into one definition each -- align_packed_length,\nspatial_grid, _pin_audio_w and _text_tags in layout.py, _encode_visual_latent\nin conditioning.py, _rope_qk in dit.py -- and hoist the duplicated constants to\nthe module top.\n\nAlso drop three unreferenced leftovers: frame_count_from_video_latent_t (an\ninverse nothing inverts), the build_packed_sequence_t2va keyword shim, and\naudio_cond_noise_aug_rows. The last is the only judgement call -- it was a\nno-op at the released default of noise_aug = 1.0 and had no caller, so it was\nuntested-in-anger code standing in for a configuration the model does not\nship. The visual equivalent, imgvid_cond_noise_aug_rows, stays because it does\nrun.\n\n191 diffusion tests pass. The three tests that exercised the removed helpers\ngo with them; test_video_latent_t_roundtrips is replaced by\ntest_video_latent_t_is_five_per_seventeen_frames, which checks the forward\nmapping the code actually uses.\n\n* diffusion: select the pipeline from the checkpoint, as the LLM side does\n\nATOM resolves an LLM checkpoint to an implementation by looking up\n`hf_config.architectures[0]` in a dict of dotted paths -- support_model_arch_dict\nin model_runner, and now _MULTIMODAL_ARCH_TO_MODEL in model_engine/multimodal.py\nfor the Kimi-K3 vision tower. Diffusion had no equivalent: diffusion_server\nhardcoded DEFAULT_PIPELINE = MiniMaxH3Pipeline, so every checkpoint got the H3\nstages whether or not it was an H3 checkpoint, and a second model would have\nmeant a second default to get wrong.\n\nDiffusion checkpoints carry the same information under a different key --\nmodel_index.json names its producer in `_class_name`. Add\natom/diffusion/registry.py mapping that to a pipeline qualname, so\n`--model /path/to/FL2VA` identifies the pipeline on its own and an unrecognised\ncheckpoint fails naming what is supported instead of running the wrong stages.\n`--pipeline` still wins, which is what out-of-tree pipelines and the tests use.\n\nDrop engine_core.resolve_pipeline_class's hand-rolled importlib body for\natom.utils.resolve_obj_by_qualname, the helper multimodal.py resolves its\nregistry with; the name is re-exported from engine_core, where its callers and\ntests already import it from.\n\nFour serving tests passed paths with no checkpoint on them and now name a\npipeline explicitly. One of those, test_a_bad_ulysses_degree_is_refused_at_\nconfig_time, had started passing for the wrong reason -- it asserts only that\nValueError is raised, and pipeline resolution raises one too.\n\n* diffusion: declare component placement instead of deriving it per pipeline\n\nWhich ranks hold which component was stated three times in three forms and\nnowhere as data: an early return inside load_components, a verify_components\noverride restating the same rule, and the reader's own head. The override\nasked only for the transformer off rank 0, so a video VAE that failed to load\non rank 3 surfaced as a hang in collective decode rather than an error at load.\n\nAdd ComponentPlacement (ALL_RANKS / MAIN_RANK) and a class-level\ncomponent_placement dict, the weights-side sibling of StageParallelism.\nComposedPipeline derives both guards from it: register_component refuses a\ncomponent this rank should not hold, and verify_components checks each rank\nagainst its own expected set. MiniMaxH3Pipeline declares the four components\nand its override goes away. host_staged_components is untouched -- that is\nhost-or-device, a different axis.\n\nAlso register MiniMaxH3Pipeline as a checkpoint architecture. A partition\nmanifest under the checkpoint root declares _class_name MiniMaxH3Pipeline\nwhere the root declares MiniMaxH3ModularPipeline, and --model <root>/FL2VA is\nthe documented invocation, so the manifest the server actually reads was the\nunregistered one.\n\nValidated on 8x MI355X: a probe running the real load_components confirms\nrank 0 holds all four and ranks 1-7 hold transformer + video_vae, and the\nserver generates 124 frames at 1344x768 with audio at both Ulysses-8 and\nUlysses-1. 235 diffusion tests pass.\n\n* tests: move diffusion tests into tests/diffusion/, 14 files into 6\n\nThe branch was adding 14 files flat at the top of tests/, where main has 77\nentries and two subdirectories (plugin/, entrypoints/) that this work ignored.\nTop-level tests/ now gains one directory instead of fourteen files.\n\nGrouped by subject rather than by source module: framework, serving,\nattention, then H3's geometry, networks and pipelines. Merged sources keep\ntheir own docstrings under a heavy rule, so the seams stay readable.\n\nPure reorganisation -- all 224 test functions survive with identical names\n(verified by diffing the collected sets), 235 tests pass. Four collisions\nresolved by renaming, none of which changes a test: FRAME_ROWS split into\nFL2VA_/REF2VA_ (they differed, 1008 vs 24), build() into\nbuild_t2va_pipeline()/build_ref2va_sequence(), and packed_sequence's OBS_SEQ /\nOBS_USED dropped as duplicates of geometry's identical values.\n\nThe recipe's test count was stale at 247; it is 235.",
          "timestamp": "2026-08-26T13:56:02Z",
          "url": "https://github.com/ROCm/ATOM/commit/ae38bd9ca40e3d92c293cc421d72bd9f81cbc0c2"
        },
        "date": 1787763047430,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32988856087 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608251555 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9454 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7506,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32988856087 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608251555 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7491 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8832,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/32988856087 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608251555 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.345 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "ZhangLirong",
            "username": "ZhangLirong-amd",
            "email": "lirzhang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f9127a88e755cbfa3709aed5d5f93009982d4763",
          "message": "Reconstruction piecewise core, only piecewise attn compressor (#2026)\n\n* refactor(v4): split the piecewise attention by granularity\n\nOne graph per granularity. The two BATCH-shaped kernels -- the compressor\n(grid `graph_bs * per_seq_bound`) and the DSpark ragged indexer top-k\n(`bs * full_q`) -- are the split op, keyed `(layer, num_tokens, bs, q_eff)`.\nEverything token-shaped moves into the dense pieces, keyed on num_tokens:\nQK-norm/RoPE before, CSA pack + the paged attention itself after. The paged\nattention only ever sat in the eager core because torch.compile could not\ntrace it, which a regular (opaque, non-splitting) custom op solves.\n\nDrops the split boundary from nine tensors to four and leaves nothing on\n`copy_per_step`. PIECEWISE and AF_PIECEWISE now share this structure exactly;\nthe only difference is whether the split op is captured, which is `capture`\ninside `piecewise_core`. Plain PIECEWISE gets the token-shaped work out of\nits eager gap as a side effect.\n\nAlso: publish `kv_indices_{swa,csa,hca}` whole instead of slicing to\n`indptr[T]` -- that length is a per-token cumsum, varies at fixed num_tokens,\nand gets baked at capture while neither consumer reads it.\n\nKNOWN BROKEN: `plugin/vllm/models/deepseek_v4.py` overrides `_attn_core` to\nreconcile padded-vs-real token counts, and that body no longer exists. Left to\nfail at import rather than compute wrong shapes; the header says what the fix\nneeds. Not exercised by any test here.\n\nAccuracy verified on V4-Pro-DSpark tp8 AF_PIECEWISE. PIECEWISE unverified,\nand neither mode has been timed.\n\n* fix(v4): the 2buff packed width is a constant, not head_dim - rope_head_dim\n\n`_qk_norm_rope_out` derived the fp8 2buff Q/K width as `head_dim -\nrope_head_dim` = 448. That is `V4_DIM_NOPE`; the packed row is the NoPE fp8\nplus its inline e8m0 scale plus padding, `V4_DIM_QK_PACKED` = 512. The fake\nimpl and the body share this helper, so both were 448 while the real kernel\nproduced 512 -- an `assert_size_stride` inside the compiled graph. Only on\n`--kv_cache_dtype fp8`; bf16 runs take the other branch and never saw it.\n\nTake both widths from `v4_quant`, the same constants\n`sparse_attn_v4_paged_decode` asserts its Q against.\n\nThe existing fake-vs-body test could not catch this: both sides come from the\none helper, so it was comparing the bug to itself. The new test checks the\nshapes against the consumer's constants instead, and covers both kv_fp8\nbranches -- the fp8 one had no coverage at all.\n\n* refactor(v4): the core is the compressor alone, and prefill is what it skips\n\nTwo changes, both narrowing what the captured core is for.\n\nThe indexer top-k leaves it. On the FP8 path it could not --\n`_score_topk_decode_ragged` pads into a `bs * full_q` rectangle -- but FP4 is\nthe default indexer (`index_cache_dtype` resolves to fp4 off gfx942) and its\nvarqlen path is token-shaped throughout: `padded_tokens = q_fp4.size(0)`, a\nconstant `n_ctas` grid, windows read from fixed-address buffers the builder\nrefreshes every decode. So it belongs in a dense piece, and joins the CSA pack\nin `_attn_paged_core`. The compressor is now the only thing left whose launch\nis sized by the batch, which leaves the split op at one tensor in and a\none-element ordering token out -- the token is what keeps the compressor ahead\nof the top-k that reads what it wrote.\n\nCapture eligibility asks whether the step is a DECODE instead of bounding rows\nat 512. Prefill is what must not be captured; the row bound was trying to say\nthat in the wrong units, and at DSpark q=6 it also cut every decode above\nbs~85, silently disabling AF for the three largest buckets. `max_tokens`\nsurvives as an optional memory cap, off by default -- the capture pool measured\n8.37GB before the granularity split and 1.62GB after.\n\nAccuracy verified on V4-Pro-DSpark tp8. Latency still unmeasured, and the newly\neligible bs=128/256/512 decode buckets are the open question there.\n\n* docs(v4): say why the split op returns a one-element tensor\n\nInlines the `_compress_done` helper -- one line at two call sites did not\nneed a name -- and moves the explanation to `v4_attn_compress`, which is\nwhere the question gets asked.\n\nThe token reads like a workaround and is not one. A custom op's registered\ncontract IS its signature: the body is opaque on purpose, so \"returns nothing,\nmutates nothing\" is the op telling the compiler it does nothing, and deleting\nit is correct. Measured under inductor: the no-return no-mutation form is\nDCE'd and the compressor never runs at all. The only other way to declare an\neffect is a mutated argument, and every buffer the compressor really writes is\neither None or a placeholder at trace time -- the builder setattr-replaces\n`kv_cache`/`kv_state`/`score_state` in `build_kv_cache_tensor`, after warmup\nhas traced. So one element is the cheapest available declaration, and on the\ncaptured path it costs nothing: the allocation is recorded once, not replayed.\n\nAlso drops a fabricated figure from an earlier draft of that docstring: it\nclaimed declaring `x` mutated buys a defensive clone, measured at +60MB. It\ndoes not. A 256MB buffer declared mutated showed a 0.00MB peak increase. The\nobjection to that route is that `x` is not mutated, not that it costs.\n\n* chore(v4): one test file, and trim the docstrings\n\nFolds the kv_indices length-invariance tests into tests/test_dspark.py --\n47 tests, one file. Cuts the docstrings on the three attention halves and\ntheir ops roughly in half; the reasoning that earned its place stays, the\nessays do not.\n\n* fix(v4-vllm): move the padded/real reconciliation to the paged half\n\nThe plugin overrode `_attn_core` to clip every per-token input from the padded\nbucket width down to the real token count and pad the output back, because the\nnarrow split calls the core directly and `forward_impl`'s clip never runs.\nThe granularity split left that override with no body to wrap.\n\nIt belongs on `_attn_paged_core` now -- of the three halves that is the one\nreading the real-sized metadata (`batch_id_per_token`, `kv_indptr_*`), and it\nis the one whose padded output the graphed `_attn_post` piece downstream was\ncaptured at. Simpler than before, too: `_attn_paged_core` carries no\n`@piecewise_core`, so the override no longer has to re-apply the decorator or\nreach through `__wrapped__` to keep the output-slot bookkeeping at the padded\nwidth -- the decorated half now returns a one-element token whose slot is\nwidth-independent.\n\n`x` is left unclipped for the compressor: its plan is built from the real\nlengths and addresses rows through it, so padded rows are never read. Stated in\na comment rather than assumed silently -- this path has no test coverage here\nand was not run.\n\nAlso trims the kv_indices comment in the metadata builder.\n\n* refactor(v4): the split op returns nothing\n\n`v4_attn_compress` was returning a one-element tensor purely so something\ndownstream would consume it. It does not need to, and the reasoning that said\nit did was measured in the wrong place: an effect-free custom op is dropped by\nAOT autograd (survives `backend=\"eager\"`, gone under `aot_eager` and\n`inductor`), and a split op's submodule is exactly the piece the backend leaves\nuncompiled -- `submod_names_to_compile` excludes `is_splitting_graph` -- so it\nnever reaches that layer. Ordering comes from `split_graph`'s\n`keep_original_order=True` and the sequential submodule calls it generates.\n\nSo the op is `-> ()`, `_attn_compress` returns None, and `StableOutputs` drops\nout of V4 with it -- there is no result left to stabilise for the downstream\npiece. `piecewise_core` and `CudagraphCaptureRunner` now take a void core:\nnothing to size a slot from, nothing to copy into one, nothing for replay to\nhand back.\n\nThe new test pins both facts this rests on, because getting either wrong stops\nthe compressor silently: that AOT is the layer that DCEs, and that the backend\nexcludes split-op submodules from compilation.\n\n* fix(test): drop a dead dict the op-body scan never used\n\nCI's ruff caught F841. It was a first-draft leftover -- the scan builds\n`registered` from the decorator/registration calls and never looked at it.\n\n* chore(v4): shorten two comments in _attn_pre\n\n11 lines to 4 and 8 to 3. The facts survive; the argument for each does not\nneed to be written out at the call site.\n\n* chore(v4): shorten the qk_norm_rope op docstrings\n\n* chore(v4): name the two QKNormRopeOut helpers after what they do\n\nNeither has anything to do with rope. `_qk_norm_rope_list` takes a\nQKNormRopeOut and returns a list -- it is the custom-op boundary's flattening,\nnamed after its caller. `_qk_norm_rope_out` builds a blank one. Now\n`_qkn_to_list` and `_qkn_blank`, plus the shortened docstrings on the op.\n\n* chore(v4): _qkn_blank -> _qkn_placeholder\n\n'blank' said empty but not what for. It is the stand-in the op's fake impl and\nthe dummy_run short-circuit both build, and the docstring now leads with why\none function serves both: they have to agree on shape, or the compiled graph\nfails on assert_size_stride.\n\n* refactor(v4): lift the piecewise attention path into its own method\n\nPull the narrow-split branch out of the attn dispatch into\n_forward_piecewise_attention; the dispatch now just picks piecewise vs the\nwide eager op. Rename the carried activation to `hidden` and trim the stale\ntwo-kernel-split comments -- the split op is the compressor alone.\n\n* refactor(v4): rename _attn_paged_core -> _sparse_attention\n\nRename the op family (method, custom op, fake impl, op name, call sites) and\nits doc/comment mentions to say what it is -- the sparse attention -- across\nthe model, the vllm override, and the tests. No behavior change.\n\n* fix(v4): size the sparse attention output on the Q, not on positions\n\n`_attn_paged_core` took its row count from `positions.shape[0]`. The\nattention output feeds the mHC residual stream, which is sized by the\nhidden state -- and the Q descends from the hidden state through\n`wqkv_a`/`wq_b`, so it carries that token count. `positions` does not have\nto: under `--enable-dp-attention`, a step where any rank is prefilling\ntakes the variable-length path (`model_runner.py`, `dp_uniform_decode`),\neach rank keeps its own token count, and the two diverge.\n\nIt surfaced as an aiter assert deep inside a compiled piece, far from the\ncause: `mhc_fused_post_pre residual_in shape mismatch: expected (6,4,7168),\ngot (5,4,7168)`.\n\nAnchor both the body and `_v4_attn_paged_core_fake` on the Q, and move the\n\"did the QK-norm/RoPE run upstream\" assert ahead of the dummy-run guard so\nthat path is anchored too. Fake and body disagreeing is its own class of\nfailure, so the regression test pins both with T_q != T_positions.\n\nSigned-off-by: ZhangLirong-amd <Lirong.Zhang@amd.com>\n\n---------\n\nSigned-off-by: ZhangLirong-amd <Lirong.Zhang@amd.com>",
          "timestamp": "2026-08-27T14:20:08Z",
          "url": "https://github.com/ROCm/ATOM/commit/f9127a88e755cbfa3709aed5d5f93009982d4763"
        },
        "date": 1787867845031,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9447,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33109303754 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608271830 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33109303754 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608271830 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 65.99,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33109303754 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608271830 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
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
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33109303754 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608271830 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7453 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8726,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33109303754 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608271830 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3222 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
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
          "id": "a5e3958eb3ce9e1d0e4bda65aff6346aec5176af",
          "message": "[feat](vllm-atom k3): support DSpark speculative decoding under DCP (#2033)\n\n* [feat](k3): support DSpark speculative decoding under DCP\n\nLifts the MLA-DSpark DCP config guard, localizes the draft's KV slots and\nseq lens to the DCP rank, routes causal multi-token decode through aiter's\nround-robin (cprr) kernel, and restores the KDA reorder_batch_threshold that\nvLLM otherwise clamps to 1 under DCP.\n\nTP8 + DCP8 + N=7 measures gsm8k 5-shot 0.9538, matching the no-DCP baseline\n(0.9538), with ~52% draft acceptance. Needs --gpu-memory-utilization 0.75:\nFULL cudagraph capture OOMs on a 9.35 GiB scratch alloc at 0.85.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* [ci](k3): run the Kimi-K3 DSpark accuracy job under DCP8\n\ngpu-memory-utilization drops 0.93 -> 0.75 because it has to: DCP8 makes every\nrank compute all 128 MLA heads over its 1/8 KV shard, and with DSpark's 8-token\nqueries the FULL cudagraph capture needs ~10 GiB of scratch on top of a 38 GiB\ngraph pool. At 0.93 capture deadlocks, at 0.85 it OOMs on a 9.35 GiB alloc.\n\nMeasured at 0.75: gsm8k 5-shot 0.9538 / 0.9530, matching the no-DCP baseline.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* [ci](k3): lower Kimi-K3 DSpark DCP8 gpu-memory-utilization to 0.7\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: perzhang <perzhang@amd.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-28T13:44:57Z",
          "url": "https://github.com/ROCm/ATOM/commit/a5e3958eb3ce9e1d0e4bda65aff6346aec5176af"
        },
        "date": 1787952361807,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9469,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33206557484 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608281849 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9439 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9515,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33206557484 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608281849 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.07,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33206557484 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608281849 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7491,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33206557484 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608281849 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7506 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8787,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33206557484 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608281849 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3033 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "00760297ef69af7ab5d345af9c8fc6da00f5314d",
          "message": "[CI] Prune stale Docker data on TW runners (#2084)\n\n* ci: prune stale Docker data on TW runners\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n* ci: prune stale Docker data on TW runners\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-08-29T13:17:36Z",
          "url": "https://github.com/ROCm/ATOM/commit/00760297ef69af7ab5d345af9c8fc6da00f5314d"
        },
        "date": 1788024534380,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33262501395 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608291452 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33262501395 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608291452 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9538 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.11,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33262501395 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608291452 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9538 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7528,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33262501395 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608291452 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7536 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8772,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33262501395 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608291452 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3351 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "00760297ef69af7ab5d345af9c8fc6da00f5314d",
          "message": "[CI] Prune stale Docker data on TW runners (#2084)\n\n* ci: prune stale Docker data on TW runners\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n* ci: prune stale Docker data on TW runners\n\n* Potential fix for pull request finding\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-08-29T13:17:36Z",
          "url": "https://github.com/ROCm/ATOM/commit/00760297ef69af7ab5d345af9c8fc6da00f5314d"
        },
        "date": 1788108689526,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33321974827 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7521,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33321974827 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7506 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8779,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33321974827 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3427 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "Jiaoliang.Yu@amd.com",
            "name": "JiaoliangYu",
            "username": "JiaoliangYu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "79ccec285785cac11878f59843532e62459d30d2",
          "message": "bench: add DeepSeek-V4-Pro EPLB + MegaMoE case at c=512/4096 (#2002)\n\n* bench: add DeepSeek-V4-Pro EPLB + MegaMoE case at c=512/4096\n\nAdds a `deepseek-v4-pro-eplb` catalog entry running MegaMoE on top of\nEP + EPLB (r0, naive placement) at 8k1k, concurrency 512 and 4096. It\nis a distinct server configuration -- expert parallel with EPLB\nenabled and --moe-backend mega -- so it gets its own entry and\nworkflow toggle rather than a variant of the base model.\n\nc=4096 needs client-side sizing the runner cannot infer: the default\nconc*10 prompts is 40960 requests, roughly 90 minutes at the measured\nthroughput, well past the 80-minute benchmark step timeout. Scenarios\nmay now carry `num_prompts` / `num_warmups`, plumbed\ncatalog -> matrix config -> benchmark-tmpl -> atom_test.sh. Empty\nmeans unset, so every existing cell keeps the conc*10 / conc*2\ndefaults; `NUM_PROMPTS_OVERRIDE` already existed for the regression\npath and `NUM_WARMUPS_OVERRIDE` mirrors it.\n\nTwo scenarios can now share (isl, osl, ratio) and differ only in that\nsizing, so both the config grouping key and the\none-config-per-server-key test include it -- otherwise the two bands\nsilently merge and one override is dropped.\n\nVerified: the 317 pre-existing cells expand byte-identically, and the\ncatalog test suite passes.\n\n* bench: scale the EPLB rebalance interval with the request count\n\nThe two cells differ 4x in request count (5120 vs 20480), so a shared\nrebalance_interval would give them very different numbers of rebalance\nevents over a run and make the two points hard to read together. Scale\nit with the load instead: 200 at c=512, 800 at c=4096.\n\nServer args resolve per variant, not per scenario, so the eplb-config\nmoves out of the shared model config and the entry splits into two\nvariants, each pinned to its own concurrency. They keep the same\n`-mega` suffix; every uniqueness key that matters (result filename,\ncell identity, config grouping) already includes the concurrency.\n\n* bench: cap c=4096 client load via bench_args instead of new catalog keys\n\nUse the existing variant bench_args path for --num-prompts/--num-warmups\noverrides and drop the num_prompts/num_warmups scenario plumbing.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* bench: hoist EPLB scenarios to model level\n\nDeclare the 8k1k c=512/4096 grid once on the model entry and let each\nvariant's conc band select its cell, instead of duplicating scenarios.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* change interval size\n\n* Bench/dsv4 eplb mega c512 c4096 (#2093)\n\n* bench: add DeepSeek-V4-Pro EPLB + MegaMoE case at c=512/4096\n\nAdds a `deepseek-v4-pro-eplb` catalog entry running MegaMoE on top of\nEP + EPLB (r0, naive placement) at 8k1k, concurrency 512 and 4096. It\nis a distinct server configuration -- expert parallel with EPLB\nenabled and --moe-backend mega -- so it gets its own entry and\nworkflow toggle rather than a variant of the base model.\n\nc=4096 needs client-side sizing the runner cannot infer: the default\nconc*10 prompts is 40960 requests, roughly 90 minutes at the measured\nthroughput, well past the 80-minute benchmark step timeout. Scenarios\nmay now carry `num_prompts` / `num_warmups`, plumbed\ncatalog -> matrix config -> benchmark-tmpl -> atom_test.sh. Empty\nmeans unset, so every existing cell keeps the conc*10 / conc*2\ndefaults; `NUM_PROMPTS_OVERRIDE` already existed for the regression\npath and `NUM_WARMUPS_OVERRIDE` mirrors it.\n\nTwo scenarios can now share (isl, osl, ratio) and differ only in that\nsizing, so both the config grouping key and the\none-config-per-server-key test include it -- otherwise the two bands\nsilently merge and one override is dropped.\n\nVerified: the 317 pre-existing cells expand byte-identically, and the\ncatalog test suite passes.\n\n* bench: scale the EPLB rebalance interval with the request count\n\nThe two cells differ 4x in request count (5120 vs 20480), so a shared\nrebalance_interval would give them very different numbers of rebalance\nevents over a run and make the two points hard to read together. Scale\nit with the load instead: 200 at c=512, 800 at c=4096.\n\nServer args resolve per variant, not per scenario, so the eplb-config\nmoves out of the shared model config and the entry splits into two\nvariants, each pinned to its own concurrency. They keep the same\n`-mega` suffix; every uniqueness key that matters (result filename,\ncell identity, config grouping) already includes the concurrency.\n\n* bench: cap c=4096 client load via bench_args instead of new catalog keys\n\nUse the existing variant bench_args path for --num-prompts/--num-warmups\noverrides and drop the num_prompts/num_warmups scenario plumbing.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* bench: hoist EPLB scenarios to model level\n\nDeclare the 8k1k c=512/4096 grid once on the model entry and let each\nvariant's conc band select its cell, instead of duplicating scenarios.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* change interval size\n\n---------\n\nCo-authored-by: JiaoliangYu <jiaolyu@amd.com>\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* bench: use catalog scenarios when dispatch param_lists is empty\n\nManual runs defaulted to 1024,1024,128,0.8 which overrides model.scenarios\nand drops EPLB cells (conc bands 512/4096). Empty param_lists now matches\nnightly behaviour so custom-grid models run without extra inputs.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* Revert \"bench: use catalog scenarios when dispatch param_lists is empty\"\n\nThis reverts commit 1bed2aa592289fbb394f363d9a80809ce4360a3c.\n\n* Bench/dsv4 eplb mega c512 c4096 (#2094)\n\n* bench: add DeepSeek-V4-Pro EPLB + MegaMoE case at c=512/4096\n\nAdds a `deepseek-v4-pro-eplb` catalog entry running MegaMoE on top of\nEP + EPLB (r0, naive placement) at 8k1k, concurrency 512 and 4096. It\nis a distinct server configuration -- expert parallel with EPLB\nenabled and --moe-backend mega -- so it gets its own entry and\nworkflow toggle rather than a variant of the base model.\n\nc=4096 needs client-side sizing the runner cannot infer: the default\nconc*10 prompts is 40960 requests, roughly 90 minutes at the measured\nthroughput, well past the 80-minute benchmark step timeout. Scenarios\nmay now carry `num_prompts` / `num_warmups`, plumbed\ncatalog -> matrix config -> benchmark-tmpl -> atom_test.sh. Empty\nmeans unset, so every existing cell keeps the conc*10 / conc*2\ndefaults; `NUM_PROMPTS_OVERRIDE` already existed for the regression\npath and `NUM_WARMUPS_OVERRIDE` mirrors it.\n\nTwo scenarios can now share (isl, osl, ratio) and differ only in that\nsizing, so both the config grouping key and the\none-config-per-server-key test include it -- otherwise the two bands\nsilently merge and one override is dropped.\n\nVerified: the 317 pre-existing cells expand byte-identically, and the\ncatalog test suite passes.\n\n* bench: scale the EPLB rebalance interval with the request count\n\nThe two cells differ 4x in request count (5120 vs 20480), so a shared\nrebalance_interval would give them very different numbers of rebalance\nevents over a run and make the two points hard to read together. Scale\nit with the load instead: 200 at c=512, 800 at c=4096.\n\nServer args resolve per variant, not per scenario, so the eplb-config\nmoves out of the shared model config and the entry splits into two\nvariants, each pinned to its own concurrency. They keep the same\n`-mega` suffix; every uniqueness key that matters (result filename,\ncell identity, config grouping) already includes the concurrency.\n\n* bench: cap c=4096 client load via bench_args instead of new catalog keys\n\nUse the existing variant bench_args path for --num-prompts/--num-warmups\noverrides and drop the num_prompts/num_warmups scenario plumbing.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* bench: hoist EPLB scenarios to model level\n\nDeclare the 8k1k c=512/4096 grid once on the model entry and let each\nvariant's conc band select its cell, instead of duplicating scenarios.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* change interval size\n\n* bench: use catalog scenarios when dispatch param_lists is empty\n\nManual runs defaulted to 1024,1024,128,0.8 which overrides model.scenarios\nand drops EPLB cells (conc bands 512/4096). Empty param_lists now matches\nnightly behaviour so custom-grid models run without extra inputs.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* Revert \"bench: use catalog scenarios when dispatch param_lists is empty\"\n\nThis reverts commit 1bed2aa592289fbb394f363d9a80809ce4360a3c.\n\n---------\n\nCo-authored-by: JiaoliangYu <jiaolyu@amd.com>\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* ci: add 10-min hang window for EPLB MegaMoE benchmark case\n\nRead ATOM_BENCHMARK_STUCK_POLLS from models.json env_vars (60 polls =\n10 min) and keep the global default at 18 polls for other benchmarks.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: JiaoliangYu <jiaolyu@amd.com>\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-31T20:21:16+08:00",
          "tree_id": "065f2a6621431c4d28ceed73267f3f03a2dd9cab",
          "url": "https://github.com/ROCm/ATOM/commit/79ccec285785cac11878f59843532e62459d30d2"
        },
        "date": 1788180687592,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33391340897 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7415,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33391340897 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7392 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
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
          "id": "171de4553944a8bee86c587928e4afa1211d0448",
          "message": "Log the KV transfer hot path at debug, not info (#2092)\n\nEvery per-request and per-step step of a PD KV transfer was logged at\ninfo: the consumer's write_request, the producer's receipt of it, the\nRDMA write itself, the per-rank and all-stage write-done notifications,\nstart_load_kv, get_finished, build_connector_meta, and the per-seq\nPD transition/first-decode/decode lines in the scheduler. At 48-way\nconcurrency and pp=4 that is ~36 lines per request on the consumer and\n~16 on the producer, which buries the startup and failure lines that\nactually need reading.\n\nMove those to debug. A one-hour 4842-request run drops from 403k to 25k\ndecode lines and from 599k to 228k prefill lines (of which 196k are\nLMCache's own pin-timeout and allocation warnings, not ours).\n\nStartup and registration lines stay at info, as do every warning, error\nand exception on the transfer paths. PD backpressure also stays at info\n— it reports a real stall — but now prints every 1000 schedule ticks\ninstead of every 100.",
          "timestamp": "2026-08-31T13:47:56Z",
          "url": "https://github.com/ROCm/ATOM/commit/171de4553944a8bee86c587928e4afa1211d0448"
        },
        "date": 1788195534925,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413428249 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9484 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413428249 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.16,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413428249 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8749,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413428249 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3071 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Hexiang Wang",
            "username": "whx-sjtu",
            "email": "56632993+whx-sjtu@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0093d6c8bb07abaf18c3f0e3e74fcaa1e42ee775",
          "message": "fuse prefill dcp (#2113)\n\nSigned-off-by: whx-sjtu <xiaowang990929@gmail.com>",
          "timestamp": "2026-09-02T12:58:51Z",
          "url": "https://github.com/ROCm/ATOM/commit/0093d6c8bb07abaf18c3f0e3e74fcaa1e42ee775"
        },
        "date": 1788369325883,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9416,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33654285662 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9393 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33654285662 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 65.76,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33654285662 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7612,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33654285662 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7582 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8878,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33654285662 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3609 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
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
          "id": "d3b2a3de79e5da2a64ec7eb6857146867c510d2a",
          "message": "[Model] GLM-5.3-Flash (glm5_next) — text path on MI355X (#2051)\n\n* [atom] GLM-5.3-Flash bring-up: k-pool DSA indexer + reference harness\n\nFirst step of GLM-5.3-Flash (`glm5_next`) support. The model is not servable\nunder ATOM yet; this lands the one genuinely new op plus the reference oracle\nthe rest of the port will be built and validated against.\n\natom/model_ops/kpool_indexer.py\n  k-pool compressed DSA indexer. GLM-5.3 scores *pools* of `index_kpool`\n  consecutive keys rather than individual keys, so with index_topk=2048 and\n  kpool=4 only 512 candidates are ranked to cover 2048 tokens. Closest existing\n  ATOM code is DeepSeek-V4's Compressor, which pools the same way at\n  compress_ratio=4 with an `ape` term but is overlapping and RoPE'd; GLM's is\n  non-overlapping and NoPE (the whole text model is NoPE, qk_rope_head_dim == 0).\n\n  Verified to select byte-identical token indices to transformers'\n  Glm5NextTextIndexer on real layer-3 weights at seq 7/64/300/2048/3000 and with\n  5 and 17 tokens of left padding. seq=3000 exceeds index_topk, so real sparse\n  pool selection is exercised. This is the dense form; the paged/ragged variant\n  the scheduler wants is the follow-up, and this is its correctness oracle.\n\nrecipes/GLM-5.3-Flash.md\n  Architecture breakdown, the mapping of each component onto existing ATOM code\n  (KDA -> KimiKDAAttention + aiter kimi_delta_attn; mHC -> hc_split_sinkhorn,\n  which is a math-exact match down to the checkpoint tensor names; MoE, block\n  FP8, MTP, vision), the checkpoint->model weight remap, and remaining work.\n\n  Also records two upstream bugs found during bring-up:\n  - transformers mis-quantizes the KDA forget gate. modules_to_not_convert names\n    it model.layers.N.self_attn.f_a_proj, but the keys are under\n    model.language_model.layers.N and the glm5_next conversion mapping renames\n    those tensors to self_attn.forget_gate.f_a_proj before the FP8 quantizer\n    runs. All 68 forget-gate linears end up wrapped in FP8Linear while holding\n    BF16 weights with a freshly-initialised weight_scale_inv.\n  - kernels-community/finegrained-fp8 does not compile on gfx950 (LLVM\n    iota_range assert). Every block-FP8 Linear and the MoE experts route through\n    it, so no FP8 glm5_next forward runs on MI355X without a substitute. ATOM\n    uses its own aiter block-FP8 GEMMs and is unaffected.\n\nrecipes/glm5_3_flash/\n  Reference harness: ROCm torch 2.10 + transformers 5.16.1 image, the loader\n  that dumps oracle logits, a torch-only stand-in for the broken FP8 kernel, and\n  the parity check. Loads in ~131s on 4x MI355X and generates coherent text.\n\ntests/model_ops/test_kpool_indexer.py\n  CPU tests for causality, padding, whole-pool expansion, tail coverage and the\n  token budget. No GPU or checkpoint needed.\n\n* [atom] GLM-5.3-Flash: run the reference on aiter block-FP8, fix multi-GPU launch bug\n\nThe reference path now uses ATOM's own block-FP8 GEMM instead of the hub Triton\nkernel that does not compile on gfx950, so GLM-5.3-Flash runs end to end on\n4x MI355X through aiter kernels: 4.25 tok/s vs 2.68 for the torch dequant bundle,\nwith the top-4 next tokens identical to the torch oracle (the small logit deltas\nare FP8 activation quant, which is what the checkpoint was trained for).\n\nfp8_aiter_backend.py routes matmul / batched_matmul / grouped_matmul to\naiter.gemm_a8w8_blockscale. aiter_fp8_check.py pins it against a torch dequant\nreference on a real GLM weight (cos 0.9997 at M=1/21/256).\n\nThird bug found while wiring this up, now in the recipe as §4c: aiter kernels\nlaunch on the *current* CUDA device, not the tensor's device. With a multi-GPU\ndevice_map, accelerate moves tensors to cuda:1..3 without touching the CUDA\ncontext, so aiter reads and writes GPU 0's memory and returns NaN. Every input\nwas finite and well-scaled; only the output was garbage, and only past the first\ndevice boundary -- so the model loads, runs, and silently emits nonsense.\ntransformers guards DeepGEMM against exactly this; aiter has no such guard.\nFixed by pinning the context with torch.cuda.device(t.device) around every call.\nATOM proper is unaffected (one device per rank) but any multi-device\nsingle-process user of aiter is exposed.\n\nref_run.py gains GLM53_FP8_BACKEND / GLM53_MAX_NEW_TOKENS / GLM53_FP8_VERIFY.\nThe verify mode runs both backends on every call and reports the first\ndivergence with shapes and devices -- that is what localised the bug.\n\n* [atom] GLM-5.3-Flash runs end to end under ATOM on MI355X\n\nAdds atom/models/glm5_next.py and the engine wiring behind it. GLM-5.3-Flash\n(320B total / 18B active) now loads and generates under ATOM on 4x MI355X:\nevery one of its parameters loads, and greedy decoding produces coherent text\nthat opens identically to the transformers oracle (\"The user is asking why the\nsky appears blue ...\"). 21-token prompt, TP4, --enforce-eager: TTFT 3.12s,\nTPOT 45ms (~22 tok/s decode) -- 5.2x the 4.25 tok/s transformers reference,\nbefore CUDA graphs or MTP.\n\nThe model is assembly of components ATOM already had:\n  KDA            -> kimi_k3.KimiKDAAttention + aiter kimi_delta_attn\n  mHC            -> sparse_attn_v4.hc_split_sinkhorn + aiter mhc_pre/mhc_post\n  MLA            -> attention_mla via MLAModules\n  MoE            -> FusedMoE (sigmoid / noaux_tc), swiglu_limit=10\n  clamped SwiGLU -> swiglu_oai_split(alpha=1, beta=0, limit=10), which is\n                    exactly GLM's silu(clamp(gate)) * clamp(up)\n\nThree decisions worth knowing about, each exact rather than approximate:\n\n- DSA layers run DENSE. index_topk=2048 with index_kpool=4 selects 512 pools\n  covering 2048 tokens, so at or below 2048 tokens the indexer selects\n  everything and dense causal MLA is numerically identical. Beyond that this\n  model is not yet correct. The indexer weights are loaded (Glm5NextIndexer)\n  but unused; model_ops/kpool_indexer.py already implements and validates the\n  selection maths the sparse path will need.\n- NoPE runs on a zero-width rope slice with an identity rotary. Padding those\n  lanes with zeros would also have been exact but is impossible: qk_nope_head_dim\n  is already 256 and the CK prefill kernel caps head dims at 256.\n- The KDA output gate is folded at load. GLM factorises it (g_b_proj @ g_a_proj)\n  where Kimi-K3 has a single g_proj; both are linear with nothing between, so\n  materialising the product once lets KimiKDAAttention -- and all of its state\n  cache, TP and CUDA-graph integration -- be reused unchanged.\n\nEngine wiring, all following existing precedent:\n- config.py: glm5_next joins the multimodal text-config extraction, and\n  glm5_next_text the set of text configs this image's transformers has no class\n  for (loaded as a bare PretrainedConfig, like kimi_linear).\n- model_runner.py: registers the arch, and is_kimi_linear() -- really \"hybrid\n  MLA + KDA\", which selects KimiMLAGDNBackend -- now covers glm5_next_text.\n- llm_engine.py: glm5_next_text added to per_req_cache_model_types, as its own\n  startup assert demands for any model with a recurrent state pool.\n- gdn_attn.py: the KDA layer lists are read 0-based for GLM (Kimi numbers them\n  1-based), and the temporal state is fp32 for both, which aiter's\n  chunk_kimi_delta_attn requires and reads back verbatim.\n\nTwo things that cost real debugging time and are worth recording:\n- packed_modules_mapping must be a class attribute. model_runner remaps the\n  quant config from the class before the model exists; doing it a second time\n  from __init__ to add per-layer entries corrupts the layer pattern specs and\n  silently marks every attention projection quantized. GLM needs no per-layer\n  mapping anyway -- its MLA layers use q_a/q_b/kv_*, so a layer-agnostic\n  .q_proj/.k_proj/.v_proj -> .in_proj mapping is unambiguous.\n- Parameter names come from the module attribute path, not the `prefix` string.\n  The mHC parameters are flat on the layer in the checkpoint (hc_attn_fn), so\n  they are declared flat here; nesting them in a submodule renamed them to\n  attn_hc.fn and left them at their init values. ATOM's loader caught this\n  immediately (\"1189/1190 model parameters were NOT loaded\") -- that report is\n  worth its weight.\n\nRequires an aiter with chunk_kimi_delta_attn and mla_decode_fwd(causal=...);\nrocm/atom-dev:nightly_202608270231 or newer.\n\n* [atom] GLM-5.3-Flash: validate the ATOM port against the reference layer by layer\n\nAdds the validation harness and records what it says. Headline: ATOM's forward\ntracks the transformers reference to cosine >= 0.9997 at every one of the 45\nlayers, and does not degrade with depth -- which is the property that separates\na sound port from a mis-wired one.\n\n  layer   0     3     7    11    19    27    35    43    44\n  cosine  .99972 .99976 .99970 .99999 1.00003 .99999 1.00002 .99998 .99998\n\ncompare_layers.py drives that: it hooks the reference's decoder layers and diffs\neach against ATOM's ATOM_FWD_DUMP output. Two traps it now handles, both of which\nproduce convincing-looking nonsense:\n\n- ATOM_FWD_DUMP_ONE_SHOT defaults on and writes only the FIRST call, which is the\n  warmup forward over 16384 dummy tokens. Comparing against that reports cosine\n  0.08 at layer 0 and reads as a catastrophic bug. Dump with ONE_SHOT=0 and pick\n  the call whose row count matches the prompt.\n- Teacher-forced rank-1 agreement stops meaning much once two implementations\n  fork. score_atom_tokens.py scores ATOM's tokens under the reference and gets\n  67% rank-1 / mean p 0.58, versus a 98.4% / 0.86 baseline from rescoring the\n  reference's own torch-FP8 output with the aiter-FP8 backend. That gap is mostly\n  an artifact: the baseline never forks, while ATOM forks at position 4 on\n  ' why' (p 0.63) vs ' about' (p 0.11) -- a genuinely split distribution -- after\n  which the reference is grading a prefix it would not have written. The\n  per-layer cosines are the load-bearing evidence; this number is not.\n\nOne real numerical gap found and measured, not yet fixed: GLM clamps its expert\nSwiGLU at +-swiglu_limit (10.0), but only ATOM's flydsl and mori MoE paths plumb\nswiglu_limit -- the default standard/CK path drops it. Instrumenting the\nreference shows the clamp binds on under 0.001% of elements (max |gate| 19.6,\nmax |up| 14.2), so it is small but real. Dense layers 0-2 are unaffected; they go\nthrough swiglu_oai_split.\n\nAlso: GLM53_DISABLE_FUSED_MHC=1 forces the torch mHC path for bisecting (it\nproduces the same output as the fused aiter kernels, at 2.8x the decode cost),\nand atom_run.py gained --logprobs plus a dump of its generated ids.\n\nrecipes/GLM-5.3-Flash.md is rewritten for the new status: how to run it, the\nmeasured numbers, and what is left (contexts past index_topk, swiglu_limit, MTP,\nvision, CUDA graphs).\n\n* [atom] GLM-5.3-Flash: vision tower\n\nAdds atom/models/glm5_next_vl.py -- the 24-layer ViT, 2-D rope, packed\nvariable-length attention, 2x2 spatial merge and projection into the language\nmodel's 4096-wide embedding space -- and wires it into\nGlm5NextForConditionalGeneration behind the engine's existing multimodal\ncontract (get_vision_embeddings / merge_multimodal_embeddings).\n\nValidated against transformers' Glm5NextVisionModel with the real\nmodel.visual.* weights, on the merged [n_tokens, 4096] output that gets\nscattered onto image placeholders:\n\n  fp32 + SDPA   bit-exact on all four grids (max|d| = 0.00000)\n  bf16 + aiter  cos .9941 - .9998\n\nThe fp32 row is the assertion: same kernel as the reference and no BF16\nrounding, so it isolates the maths -- patch embed, the block-major 2-D rope\nlayout, 24 blocks, clamped SwiGLU, the merge, downsample conv and merger. The\nbf16 row is the serving path, reported not asserted; it carries aiter\npacked-varlen vs SDPA plus ATOM's single fused [gate|up] GEMM vs the\nreference's two, compounded over 24 blocks on random patches far worse\nconditioned than real image input.\n\nTwo things worth recording:\n\n- config.py: when this image's transformers has no class for the architecture,\n  get_hf_config fell through to _multimodal_config = None, which left the model\n  with no vision_config and no way to build a tower. It now rebuilds the\n  multimodal config from the raw dict, promoting sub-configs to\n  PretrainedConfig so attribute access works either way.\n- The model's packed_modules_mapping rewrites .gate_proj / .up_proj to\n  .gate_up_proj by substring, so it hits the vision tower too. Leaving the\n  tower's projections unfused silently dropped 98 tensors (24 blocks x 4, plus\n  the merger's 2) -- the load report caught it. They are fused here, with a\n  plain nn.Linear carrying a (param, tensor, shard_id) weight_loader rather\n  than MergedReplicatedLinear, so the tower stays usable without an initialised\n  TP group (the parity harness has none).\n\nLoad is clean: all 1614 parameters load, no checkpoint tensor dropped, and the\ntext path is unchanged (TTFT 3.15s, TPOT 46ms).\n\nStill missing for actually serving an image: Glm5NextProcessor only exists in\ntransformers >= 5.16 while ATOM pins 5.12.1, so nothing turns an image into\npixel_values + image_grid_thw, and model_engine/multimodal.py has no glm5_next\nbranch. Recorded in recipes/GLM-5.3-Flash.md section 7.\n\n* [atom] GLM-5.3-Flash: move the bring-up harness to tools/, add a raw-prompt runner\n\nThe harness is tooling, not a recipe, so it moves from recipes/glm5_3_flash/ to\ntools/glm5_3_flash/ alongside the other standalone scripts. It does not belong\nin tests/ either: everything there runs on CPU with aiter mocked, while these\nneed four GPUs and the 306 GiB checkpoint. recipes/GLM-5.3-Flash.md stays where\nrecipe docs live and now points at the new location.\n\nAdds raw_prompt_offline.py: feeds an un-templated prompt through the offline\nengine. atom_run.py applies the chat template, but GSM8K is scored on\n/v1/completions with a plain few-shot prompt, so this isolates whether a\nserved-path failure is in the model or in serving -- same prompt, same weights,\none request, no scheduler.\n\n* Fix GLM-5.3-Flash accuracy: pad the NoPE rope block to 64\n\nGLM-5.3-Flash is NoPE (`qk_rope_head_dim == 0`), and this port expressed that\nas a zero-WIDTH rope slice. Measured over all 1319 gsm8k questions that scored\nflexible-extract 0.0099 / strict-match 0.0000; it now scores 0.9682 / 0.9689.\n\nThe zero-width representation fails in two independent ways:\n\n* The paged MLA entry is sized `kv_lora_rank + qk_rope_head_dim`, so it came out\n  512 while aiter's asm decode kernel is built for a 576-wide query. That kernel\n  asserts the 576 only on the gfx1250 path and `cfg_mla_asm` never dispatches on\n  head_size, so on gfx950 the mismatch is computed rather than rejected. 69% of\n  replies came back empty and, of the rest, an extraction-free audit found the\n  gold value in only 8.8% (98.0% after this change). Prefill was unaffected --\n  it goes through head-dim-generic `flash_attn_varlen` -- which is why a\n  prefill-only per-layer cosine check reads 0.9997 on a broken model.\n* `KV_PeDim == 0` turns every `tl.arange(0, KV_PeDim)` into `arange(0, 0)`,\n  which Triton rejects at compile time, and upstream aiter's three\n  `gather_kv_b_proj` kernels carry no guard for it. It fires only on the\n  cached-prefix prefill path, so a single-prompt run passes and any concurrent\n  one dies ~15 s in with `NameError('kv_pe_data is not defined')`.\n\nSo materialize the rope block at `_ROPE_PAD = 64` lanes and hold it at zero.\nThe objection that ruled this out before -- `qk_nope_head_dim` is already 256\nand CK caps head_dim at 256 -- conflates two constraints that apply to\ndifferent tensors: the latent/cache/decode side wants 576, the per-head\nqk/prefill side wants <= 256. `MLAModules.rope_is_zero_pad` makes the MLA drop\nthe zero lanes at every `flash_attn_varlen_func` site, which is exact because\nthe lanes are identically zero, so both widths are satisfied at once. All five\nsites are patched: prefill has plain, cached-single-pass and chunked\ncontext/suffix variants, and a short-prompt smoke test only reaches one.\n\nThe padded width has to be declared to the cache allocator as\n`config.mla_kv_entry_dim`. `KimiMLAGDNBackend` shadows the plain MLA allocator\nand sizes the pool from the raw config, so without it the pool is built 512\nwide under a 576-wide write and the server dies at startup. `aiter_mla` grows a\n`mla_kv_entry_dim()` helper that prefers the declared width and falls back to\nthe old expression, leaving every other MLA model unchanged.\n\n`_ZeroRopePad` appends the pad at call time and is deliberately not an\n`nn.Module`: wrapping `q_b_proj` in one inserts a level into the parameter path\nand the weights then silently never load.\n\nAlso widens `TritonMLAMetadataBuilder.set_mla_persistent_worker_buffers` to take\n`**kwargs`. That is a pre-existing bug on main, unrelated to this model: the\naiter caller passes `is_cp_round_robin`, so the Triton MLA path dies at init for\nany hybrid model and this port had no working fallback backend.\n\nAdds the `GLM-5.3-Flash` CI accuracy entry (baseline 0.9682, threshold 0.94) so\nthe number is guarded, and records it in the recipe. Verified with the standard\nCI config -- TP8, bf16 KV, max-num-batched-tokens 2048, 32 concurrent -- which\nis the configuration that used to crash.\n\n* Add the k-pool sparse indexer so GLM-5.3-Flash serves past 2048 tokens\n\nContexts at or below `index_topk` (2048) were already exact: top-k selects\nevery pool, so the expansion yields every token position and dense causal MLA\nis numerically identical. Past that threshold the pooled selection decides what\nthe model attends to, and it was not implemented -- so long context was\nsilently wrong rather than refused.\n\nThis ports the measured implementation from the parallel GLM-5.3-Flash port:\n\n* `model_ops/glm5_next/kpool.py` -- the pooled ops. One cached entry per\n  `index_kpool` tokens: a per-dimension softmax over the pool weighted by\n  `index_kpool_compress_gate` plus the per-slot `index_kpool_compress_ape`\n  bias, Hadamard-rotated and FP8-quantized. `model_ops/kpool_indexer.py` stays\n  as the dense `transformers`-parity oracle it was written to be, off the\n  serving path.\n* `Glm5NextIndexer` becomes a real `Indexer` subclass driving that path, with\n  the NoPE override (no rope component to split or rotate) and a guard that\n  REFUSES the regime where pooled and token-granular top-k genuinely differ\n  rather than quietly returning the fallback.\n* `KimiMLAGDNBackend` grows the pooled index cache sizing and the per-request\n  tail slots. The tail rides the KDA state pool rather than a paged cache.\n* `glm5_kpool_block_size()` sets the KV block to `index_kpool * 16 = 64`.\n  `deepgemm_fp8_paged_mqa_logits` is correct only in its preshuffled layout,\n  which needs the per-block index row count to be a multiple of 16; a 16-token\n  block gives 4 rows and forces the index cache back to one row per token.\n\nThree wiring points were needed beyond the ops themselves, and each failed in a\nway worth recording:\n\n* `is_deepseek_v32` -- which is what binds the index cache -- was gated on\n  `use_mla`, but this model is served by the Kimi MLA+GDN hybrid backend and so\n  sets `use_kimi_mla` instead. A sparse indexer is orthogonal to that choice.\n  Without it `sparse_kv_indptr` stays None and decode dies in\n  `mla_decode_stage1_asm_fwd`.\n* MLA does not call the indexer; the model's forward has to drive it. Missing\n  that call is silent: `is_sparse=True` makes MLA read a selection buffer that\n  nothing ever fills, so short contexts stay perfect and only long ones break.\n* The `sparse_attn_indexer_kpool` custom op and its `mutates_args` registration\n  must be present, or inductor is free to hoist the MLA read above the in-place\n  pooled cache write.\n\nMeasured, all 1319 gsm8k questions, chat + fewshot_as_multiturn, TP8, bf16 KV:\n\n    3-shot  @2048  0.9682 / 0.9689   (dense, unchanged by this commit)\n    16-shot @8192  0.9659 / 0.9666   (pooled)\n\nOnly the 16-shot row exercises pooling: 3-shot prompts are ~389 tokens, so they\nnever reach `index_topk` and the pooled selection is computed and discarded. A\nshort-context score shows only that the pooled writes did no harm. Long-context\nretrieval additionally checked with a control at 6166 tokens -- two different\nsecrets in the same prompt shape, each recovered exactly, so it is retrieval\nrather than a guess.\n\nAdds the `GLM-5.3-Flash-kpool-16shot` CI entry (baseline 0.9659, threshold 0.94)\nbeside the 2048 one, which does not cover this path.\n\n* Gate the pooled path in PR CI, and fix claims the last two commits falsified\n\n`GLM-5.3-Flash-kpool-16shot` moves from `nightly` to `pr`. The pooled indexer is\nthe part of this port with the least static coverage -- there is no bit-exact\nunit test for `model_ops/glm5_next/kpool.py` on this branch -- so it is the part\nthat most needs an end-to-end gate on every PR. The dense `GLM-5.3-Flash` entry\nstays nightly; note that this leaves the dense-only regime (context at or below\n`index_topk`) uncovered by PR CI.\n\nStale claims, each left false by the two commits before this one, and each of\nthem the kind that reads as fact to the next person:\n\n* `atom_run.py` said \"v1 of this model is only exact at or below index_topk\n  (2048) tokens\". The k-pool path lifted that.\n* The module docstring said the vision tower was \"not yet wired\". It is built\n  and loaded from `model.visual.*`; what is missing is upstream of it -- nothing\n  turns an image into `pixel_values` -- so the tower cannot be reached at\n  serving time. Naming the actual gap is the useful form.\n* §2's mapping table pointed at `model_ops/kpool_indexer.py` as the k-pool\n  implementation. That is the dense `transformers`-parity oracle; the serving\n  path is `model_ops/glm5_next/kpool.py`.\n* §7's preamble pinned `--max-model-len 2048` while the table below it lists an\n  8192 row.\n\nAlso states the coverage gap plainly in §3, because it is not visible to a\nreviewer: the k-pool code that serves has no bit-exact test here, and a gsm8k\nscore cannot pin what one would -- the pooling softmax being per-dimension\nrather than per-slot, the Hadamard being orthonormal, and the pool/tail slot\narithmetic all have wrong spellings that still produce plausible keys and a\nscore in the right range.\n\nDocumentation and CI config only; no behaviour change.\n\n* Silence BLE001 on the multimodal-config fallback so PR ruff passes\n\nThe ruff job runs reviewdog with `-filter-mode=diff_context -fail-on-error=true`,\nso any finding on a line inside this PR's hunks fails it -- including lines the\nPR only touches as context. `get_hf_config`'s broad `except Exception` is one:\nunchanged itself, but sitting in the hunk that rebuilds `_multimodal_config`.\n\nThe catch is deliberate and stays. `AutoConfig.from_pretrained` raises whatever\nthe architecture's own code raises when this image has no class for it, so\nenumerating the types would be guesswork that silently stops handling one. Marked\nwith the reason, matching the `# noqa: BLE001 - <why>` convention already used\nacross `atom/diffusion/`.\n\n* Give a PAGE unit the index-cache regions it was priced with\n\nPR CI's GLM-5.3-Flash-kpool-16shot job never reaches the eval: every\nrank dies in `allocate_kv_cache`, before a single request.\n\n    RuntimeError: a PAGE unit is 836352 B but this pool gives a logical\n    block 11 rows x 73728 B = 811008 B; the two disagree about block\n    granularity\n\nNeither side is wrong alone. This branch put GLM-5.3's sparse indexer\nkey cache in the paged pool, so `sub_pool_specs` prices a block with it\nin -- 11 indexer layers x 16 pooled rows x 144 B = 25,344 B on top of\nthe 11 MLA rows, which is the 836352/811008 gap exactly. Meanwhile main\ngained `_page_unit_regions`, the destination side of K3's PAGE-backed\nKDA checkpoints, and it describes a unit as the MLA rows alone -- true\nfor K3, which has no indexer. The two only meet in the merge, and the\nequality between them is the assertion that fires.\n\nNaming the shorter list is not a smaller, safer copy:\n`units_per_checkpoint` is `ceil(image_bytes / page_unit_bytes)`, priced\nagainst the whole block, so regions covering 811,008 of every 836,352\npriced bytes leave the tail of every image with nowhere to land. So the\nindex cache is named here too, the way the DSV4 builder names its\nindexer pools. Its block axis is the logical block a unit id already\ncarries -- no `block_ratio` conversion, unlike the MLA side -- so a unit\nowns one contiguous region per indexer layer, the layers a fixed stride\napart: the same affine answer the MLA rows give. The region list now\nsums to the priced 836,352 B exactly.\n\nThe check stays, since it is the one relation that cannot hold if the\ngranularity is wrong, and gains the index term in its message so the\nnext mismatch says which pool is short.\n\nTests: ten cases over a stub whose indexer shares the pool -- addresses\nagainst what slicing gives, the region sum against the price, a unit\npriced without the index cache refused, a model without an indexer left\nwith the MLA regions alone, both pools in the cache key, and a\nstore/restore round trip whose image genuinely spills into the index\nregions and leaves every block past it untouched.\n\n* Read the dense last-page lengths only where a prefill has them\n\n`_forward_prefill_mla` derived its sequence count from\n`attn_metadata.kv_last_page_lens` before splitting dense from sparse. That\narray is per SEQUENCE and the prefill builder fills it only for a drafter or a\ncached prefix; a sparse (DSA) prefill has neither in the common case and\ncarries its own per-QUERY `sparse_kv_last_page_lens` instead, which the branch\nthree lines below substitutes for exactly that reason. So the read landed on\n`None` and took down all eight ranks:\n\n    n_seqs = attn_metadata.kv_last_page_lens.shape[0]\n    AttributeError: NoneType object has no attribute shape\n\nGLM-5.3-Flash is the first model in CI to reach it. It is sparse, has no MTP\ndrafter yet, and serves with prefix caching off (its KDA state is\nper-request), so every request past `index_topk` = 2048 tokens crashed the\nengine -- which is every 16-shot GSM8K prompt, on the first one.\n\nThe value it computed was dead on that path anyway: the only use is cutting\n`cu_seqlens_q`, and sparse replaces that with `sparse_cu_seqlens_q` regardless.\nMove both into the dense branch, where the array is present because the kernel\ncall itself needs it.\n\nReproduced and verified on 8x MI355X, GLM-5.3-Flash TP8, the CI accuracy\nconfig verbatim: gsm8k 16-shot pooled over all 1319 questions, previously dead\nbefore the first response, now flexible-extract 0.9583 / strict-match 0.9591\nagainst a 0.94 threshold.\n\n* Drop the bring-up harness and the test report that rode in with it\n\n`tools/glm5_3_flash/` was scaffolding for building a transformers oracle to\ndiff this port against. That job is finished and its results are in the recipe;\nwhat would ship is 1409 lines that nothing in ATOM imports, no test exercises,\nand CI never runs.\n\nIt could not be run from ATOM anyway. The harness needs `transformers>=5.16`\nfor `glm5_next` while ATOM pins 5.12.1, so its own Dockerfile builds a separate\nimage; five of the eleven files exist only to work around `finegrained-fp8`\nfailing to compile on gfx950 (recipe section 4b), and `vision_parity.py` covers\na tower that no input path reaches yet. `atom_run.py` and `raw_prompt_offline.py`\nduplicate `atom.examples.simple_inference` and the offline engine, differing\nonly in using the oracle prompt. Code that cannot run in this repository's\nenvironment and has no caller does not get maintained; it gets believed, and\nthen found stale.\n\n`unit-report.xml` is a 745 KB pytest JUnit artifact. `run_unit_tests.sh` writes\nit into the workspace root, and it was committed by accident in 545593a76 with\nfour failures recorded inside it. #1662 removed one of these before, so this\nadds a .gitignore rule rather than relying on care.\n\nThe recipe keeps every measurement and now stands on its own: section 5 says\nwhat rebuilding the oracle takes, including the two non-obvious parts (the FP8\nkernel replacement, and that having both backends is what exposed the aiter\nmulti-GPU bug of section 4c); sections 3 and 6 say which results are bring-up\nfindings and which are the standing CI guard. The quickstart now runs\n`atom.examples.simple_inference`, verified on GLM-5.3-Flash at TP4.\n\n* Fix GLM-5.3 k-pool state consistency\n\nPreserve partial pool state across checkpoint, relocation, and chunk boundaries so long-context requests cannot consume stale cross-request data. Align configuration and quantization edge cases with their existing contracts.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* Format GLM-5.3 review fixes\n\nApply the repository's Black formatting so the correctness fixes pass PR pre-checkin.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* Fix GLM-5.3 second-pass review findings\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* Format GLM-5.3 environment settings\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* Refactor GLM-5.3 model into ATOM style\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* Size the chunked-prefill K workspace from mla_kv_entry_dim\n\nThe has_cached chunked-prefill workspace was still sized from the raw HF\nconfig, `qk_nope_head_dim + qk_rope_head_dim`. That is the one place the\nNoPE widening was not swept into: the pool allocator and every\nflash-attention call site already go through `mla_kv_entry_dim` /\n`rope_is_zero_pad`, and this buffer is fed to the gather at three call\nsites, so it was the last raw-config geometry left on the MLA path.\n\nFor GLM-5.3-Flash the config deliberately holds `qk_rope_head_dim` at its\ntrue 0 -- the indexer has to stay NoPE -- and the rope block is widened to\na 64-lane zero pad on the MLA side only. The raw sum therefore gives 256\nwhere the kernels want 320.\n\nIt does not fail as a size mismatch. `gather_kv_b_proj` takes the rope\nwidth from its DESTINATION buffer (`qk_nope_pe_dim = k_prefix.shape[-1]`)\nand the nope width from the kv_b_proj weight, so a 256-wide workspace\nagainst a 256-wide nope half makes `KV_PeDim = 0` and Triton rejects the\nresulting `tl.arange(0, 0)` at compile time -- exactly the failure the\n`_ROPE_PAD` note in glm5_next.py warns about, one kernel below where the\nwidth is chosen. The line right after the gather already calls\n`_drop_rope_pad` on this buffer, so the surrounding code was written for\nthe padded width; only the allocation disagreed.\n\n`mla_qk_head_dim` derives the rope width from `mla_kv_entry_dim` instead,\nwhich is the declared padded width. For a real-rope model that reduces to\nthe sum it replaces, so no shipped model changes shape.\n\nThe line dates to #911 and only became wrong when a NoPE model arrived, so\nit is not reachable on main today. Note also that it needs a CACHED prefix\nto fire, which `--no-enable_prefix_caching` in the CI entry suppresses --\nCI would not have caught this.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Run the GLM MoE router in fp32\n\nATOM had no reference to `moe_router_dtype` anywhere, so the router ran at\nthe model dtype. That is wrong for every `glm_moe_dsa` model, and the\ndamage is not in the gate logits but in the `noaux_tc` correction bias.\n\nIts 256 values sit in a narrow band riding a large offset -- measured on\nGLM-5.3-MXFP4 layer 10: min 6.817, max 7.063, 238 distinct values. bf16's\nULP at ~7 is 1/64, four orders of magnitude coarser than the ~6e-6 median\nspacing, so loading that tensor into a bf16 parameter collapses 238\ndistinct values onto 8 and discards almost all of the selection signal it\nexists to carry. Every GLM-5.x checkpoint on disk ships it as fp32, and\nGLM-5, 5.1, 5.2 and 5.3 all share the same [6.02, 8.11] range.\n\nvLLM forces fp32 routing for this model_type unconditionally, including\nfor GLM-5/5.1/5.2, whose configs predate `moe_router_dtype` and so cannot\nask for it; keying on model_type rather than the config key matches that\nand keeps those generations correct. ATOM already did this for `glm4_moe`\nand `glm5_next`, both of which build the gate and the bias in fp32 --\n`deepseek_v2.py` was the one GLM path that missed it.\n\nThe dtype must be set at parameter CREATION. The loader casts the\ncheckpoint tensor into whatever the parameter holds, so an fp32 bias\nlanding in a bf16 parameter is rounded once, on load, and the existing\n`.to(torch.float32)` calls in the MoE backends then widen a number whose\nlow bits are already gone.\n\nGate output and bias are not independently choosable: aiter's\n`biased_grouped_topk` dispatches on `gating_output.dtype()` and then\nreinterpret_casts `correction_bias` to that same `scalar_t`, so an fp32\ngate with a bf16 bias would read the bias buffer at double width. One\ndtype governs both, and `ReplicatedLinear.forward` already accepts\n`otype`. Non-GLM models resolve to None and are untouched, keeping both\n`torch.empty` and the gate call byte-identical.\n\nMeasured on GSM8K, and it is NOT a scores change: paired McNemar over all\n1319 questions gives b=7/c=6 (p=1.00) in chat 5-shot and b=52/c=56\n(p=0.77) in 3-shot completions. Two runs of the UNFIXED code scored 0.9719\nand 0.9773 in chat, a wider gap than the change produces, so aggregate\nrates cannot speak to this either way. This is a faithfulness fix; GSM8K\nsimply does not probe routing quality. It costs +0.14-0.29 ms per decode\nstep across 75 MoE layers, inflated because aiter carries no tuned config\nfor the fp32-output gate shape and falls back to a torch path.\n\nGLM-5/5.2 CI baselines were measured on the old routing numerics and\nshould be re-confirmed.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Gate GLM-5.3 accuracy in CI, and drop the redundant Flash entry\n\nGLM-5.3 (`glm_moe_dsa`) ran on ATOM but had no accuracy entry and no\nrecipe, and its MXFP4 checkpoint ships with no evaluation on its model\ncard at all. Adds GLM-5.3-FP8 (tp8, baseline 0.9719) and GLM-5.3-MXFP4\n(tp4 + ptpc_fp8 online quant, baseline 0.9765), both over all 1319\nquestions on 8xMI355X.\n\nBoth must run the chat client, not the default raw-completions one.\nGLM-5.3's template ends `<|assistant|><think>` and the server reports that\nit has no switch for reasoning, so few-shot completions suppress the\nreasoning the model is trained to do: the same checkpoints score 0.9136\n(MXFP4) and 0.9234 (FP8) that way, which would fail this threshold on a\nhealthy model. Chat is also far steadier -- two runs of identical code\ndisagree on 13 of 1319 questions in chat versus 108 in 3-shot completions.\n\nMXFP4 tracks its FP8 source (0.9765 vs 0.9719), so the quantization costs\nnothing detectable here.\n\nThe short-context GLM-5.3-Flash entry is removed in favour of the 16-shot\nkpool one. Below index_topk the indexer selects every token and\nattention_mla runs dense, so that entry exercised a strict subset of the\n16-shot path while costing a second 8-GPU nightly.\n\nThresholds are set to 0.96 as requested. This is comfortable for the two\nGLM-5.3 entries (~1.6pp of headroom below the worst observed run) but\nTIGHT for GLM-5.3-Flash-kpool-16shot: its baseline is 0.9659 and repeat\nruns of identical code have landed at 0.9613/0.9629/0.9644, so the worst\nobserved run clears 0.96 by 0.0013 and that entry should be expected to\nflake. It was 0.94 before.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Modernize an annotation this PR pulled into ruff's diff context\n\nCI runs ruff through reviewdog with `-filter-mode=diff_context` and\n`-fail-on-error=true`, so a finding fails the PR whenever it lands on a line\ninside the diff -- including lines this PR never edited. Adding\n`_moe_router_dtype` directly above `_can_fuse_indexer_wk_weights_proj` brought\nthat function's pre-existing `Optional[QuantizationConfig]` into the context\nwindow, and UP045 fired on it.\n\n`Optional` still has 71 other uses in this file, so the import stays live and\nnothing else in the diff moves.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Fix GLM-5.3 third-pass review findings\n\nFinding 1 (raw-config chunk workspace) was already fixed; this covers 2-5.\n\n2. Bound the decode index conversion to the request's own output region.\n   `_convert_req_index_to_global_index_kernel` gated its store on\n   `indice_id < kv_len`, which counts entries on the INPUT side and is not the\n   width of the output region -- that is `page_kv_indptr[b+1] - [b]`, since the\n   output is packed by that indptr. A pooled selection makes the two diverge\n   (the row is padded to `round_up(index_topk + kpool - 1, 128)` = 2176 columns\n   while the region holds at most 2051) and a long context pushes kv_len past\n   both, so every column stored. The surplus columns are top-k padding, -1,\n   which `tl.where(out_val >= 0, ...)` maps to cache slot 0 -- written over the\n   START of request b+1. `OUT_NUMEL` only caught the final request, so it was\n   silent for every other one. Now masked by `out_offset < out_kv_end`, which is\n   a no-op wherever the two widths already agree.\n\n3. Widen `_is_kda()` to both KDA models. Its two siblings in the same file\n   (`gdn_attn.py:129`, `:331`) were widened for GLM-5.3-Flash and this one was\n   not, so it answered \"not KDA\" for a KDA model and\n   `_replayssm_buffer_shapes()` sized the ReplaySSM record buffers for the wrong\n   head geometry -- an out-of-bounds write reachable with\n   `ATOM_ENABLE_REPLAYSSM=1`. Swept the tree: no `== \"kimi_linear\"` test remains.\n\n4. Let `--load_dummy` start. The KDA shard-coverage check exists to catch a\n   checkpoint that supplied only part of q/k/v, but dummy loading never runs a\n   weight_loader, so the set is empty by design and all 34 KDA layers refused to\n   start. Skipped when `load_dummy` is set; the gate fold still runs, so the\n   layer keeps the shape and dtype it would otherwise have had.\n\n5. Guard both sparsity-off knobs the same way, at startup. `ATOM_GLM5_KPOOL=0`\n   raised `NotImplementedError` from inside `forward_impl`, which takes the\n   engine down mid-batch rather than refusing a request the way `envs.py` and\n   `docs/environment_variables.md` both describe; `ATOM_GLM5_FORCE_DENSE_MLA`\n   had no length guard at all despite being documented as short-context only, so\n   the two failed in opposite directions. Both are now validated in\n   `Config.__post_init__` against `max_model_len`, which already bounds\n   `max_seqlen_k`: the launch fails cleanly with a message naming the fix, and a\n   per-forward branch leaves a 45-layer hot path.\n\nUnit suite unchanged at 4831 passed / 14 failed; those 14 fail identically on\nplain origin/main (they arrived with #1965 / #2113, after the review's measured\n4902/0). black and the diff-context ruff gate are clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Keep the new unit tests off the AITER import path so CI can run them\n\nThe Pre Checkin unit-test job failed with pytest exit code 2 -- a COLLECTION\nerror, not test failures. It runs on a plain CPU runner with no AITER, and the\ntwo tests added for the round-3 fixes imported their subjects from modules that\npull AITER at module scope (`aiter_mla.py:11`, `deepseek_v2.py:30`), so both\nfiles errored during collection and took the whole run down.\n\nReproduced locally by blocking `aiter` on the meta path: \"Interrupted: 2 errors\nduring collection\", exit 2, matching CI.\n\n`importorskip` would have turned the job green while leaving the tests unrun,\nwhich is the complaint the last two reviews already made about this PR's test\nsurface. Moved the subjects instead, into modules that need nothing but the\nconfig:\n\n* `atom/model_ops/mla_geometry.py` -- `mla_kv_entry_dim`, `mla_qk_head_dim`,\n  `aligned_index_cache_dim`. Pure integer arithmetic on config fields.\n* `atom/models/moe_router.py` -- `moe_router_dtype`. Needs torch and the\n  config, nothing else.\n\nBoth follow the reasoning already written down in `atom/model_ops/__init__.py`,\nwhich resolves its attention frontend lazily for exactly this purpose.\n`kimi_mla_gdn_attn` now imports the geometry helpers straight from the light\nmodule rather than through `aiter_mla`, so no re-export is left dangling.\n\nVerified with AITER blocked: both modules import with zero `aiter.*` in\nsys.modules, and all 8 tests run and pass. Local full suite is unchanged at\n4831 passed / 14 failed -- those 14 fail identically on plain origin/main.\nblack clean, diff-context ruff clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Guard the new unit tests with importorskip(\"aiter\") instead of relocating them\n\nReverts the module split from 6b0e5d464 and takes the simpler route requested:\nthe two tests added for the round-3 fixes now guard on `pytest.importorskip\n(\"aiter\")` before importing their subjects, matching\n`tests/plugin/test_vllm_mha_backend.py`.\n\nThe Pre Checkin unit-test job runs on a CPU-only runner with no AITER, and\n`aiter_mla.py` and `deepseek_v2.py` both import AITER at module scope, so\ncollecting these files raised `ModuleNotFoundError` and pytest exited 2 --\na collection interrupt, which fails the whole job rather than one test.\n\n`mla_geometry.py` / `moe_router.py` are gone again; `mla_kv_entry_dim`,\n`mla_qk_head_dim`, `aligned_index_cache_dim` and `_moe_router_dtype` sit back\nwhere they were, and `kimi_mla_gdn_attn` imports the geometry helpers from\n`aiter_mla` as before, so the production import graph is exactly what it was\ntwo commits ago.\n\nTrade-off, recorded because it is the known cost rather than an oversight:\nthese 8 assertions now SKIP on the CPU runner instead of executing, so they\nonly ever run on a machine with AITER. Verified both directions -- with AITER\nblocked they report as 2 skips and the run exits cleanly; with AITER present\nall 8 pass.\n\nblack clean (682 files), diff-context ruff clean, no `noqa` needed: the repo's\nruff config does not enable E402, so `importorskip` ahead of the imports is\naccepted as-is.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Fix GLM-5.3 fourth-pass review findings\n\nRound 4 was taken at a883ea8b1, so three of its items were already fixed by\nb01d4232a / cf023234d: `_is_kda()` widening, `_assert_kpool_regime` raising\nfrom inside the forward, and finding 1 (the two new test files aborting the\nwhole CI unit run -- now guarded with `importorskip(\"aiter\")`). The rest:\n\n2. `_moe_router_dtype` now matches the sibling predicates in this file and\n   accepts the MTP draft. `SpeculativeConfig._MTP_TYPE_MAP` rewrites\n   `glm_moe_dsa` to `deepseek_mtp` while keeping the GLM-only\n   `index_share_for_mtp_iteration` marker, so a model_type test alone left the\n   draft router in bf16 against an fp32 target -- different experts selected on\n   each side and a lower acceptance rate, which shows up as a throughput\n   regression while task accuracy stays clean.\n\n4. Fake-EPLB's synthetic router table is cast to the dtype of the logits it\n   replaces. It is built once at the default dtype, and this PR's own docstring\n   states the invariant it broke: `biased_grouped_topk` dispatches on\n   `gating_output.dtype()` and reinterpret_casts `correction_bias` to that\n   scalar_t, so a bf16 table against an fp32 bias read that buffer at half\n   width. Stating an invariant and leaving a path unswept was the actual\n   mistake.\n\n5. `hybrid_mla_layers` no longer treats an empty list as the hybrid layout.\n   `[] is not None`, so a config without `full_attn_layers` took the hybrid\n   branch with zero MLA layers and every index-cache region was numbered on top\n   of the KV regions instead of after them -- a P/D transfer would write index\n   bytes into KV rows. The adjacent length check cannot catch it because both\n   sides derive from the same empty list.\n\n6. Not a defect, and now recorded so it is not re-raised. `_drop_rope_pad`\n   hands non-contiguous slices to five flash-attention sites; measured on\n   gfx950, the output is bit-identical to a `.contiguous()` copy (max abs diff\n   0.0), so the kernel takes its row pitch from the stride, and the call is\n   ~1.5% slower at 4096 tokens -- far cheaper than materializing K per chunk\n   per layer. Written into the docstring with the numbers.\n\n8. The accuracy manifest. Reverted the `0.90 -> 0.9` reformat on gpt-oss-120b\n   (x2) and Llama-3.3-70B: that was a `json.load`/`json.dump` round-trip in my\n   own edit script, and those three entries are now byte-identical to main\n   again. Separately, `GLM-5.3-Flash-kpool-16shot` keeps threshold 0.94 rather\n   than the 0.96 asked for earlier: it is a `pr`-level gate, and repeat runs of\n   IDENTICAL code measured 0.9613/0.9629/0.9644/0.9659, so 0.96 clears by\n   0.0013 in the worst case and would fail roughly one unrelated PR in four.\n   Baseline moved to the 0.9613 measured on this tree.\n\nAlso modernized one annotation that finding 2's edit pulled into ruff's\ndiff_context window, the same way a883ea8b1 did.\n\nLocal suite unchanged at 4831 passed / 14 failed; those 14 fail identically on\nplain origin/main. black clean (682 files), diff-context ruff clean, catalog\nschema valid.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Modernize three more annotations ruff 0.16.5 flags in the diff context\n\nThe Ruff check failed on d77ef46fd while my local replay of the same gate\nreported clean. Cause: CI does `pip3 install ruff` and got 0.16.5, this\ncontainer had 0.16.3, and 0.16.5 reports UP045 on annotations the older build\nleft alone. Reproduced by installing 0.16.5 into a scratch venv and re-running\nthe gate, which surfaced the same finding CI did.\n\nThree `Optional[...]` annotations in `DeepseekV2MoE` are now `X | None`. All\nthree are pre-existing lines that this PR's edits pulled into ruff's\n`diff_context` window -- the same mechanism as a883ea8b1, which is why the\ngate fails on lines the PR never meant to touch. `Optional` still has 68 uses\nin the file, so the import stays live.\n\nChecked with a +/-10 line window as well as CI's +/-3, so a slightly wider\ncontext in reviewdog will not surface another one.\n\nblack clean, catalog valid, local suite unchanged at 4831 passed / 14 failed.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: perzhang <perzhang@amd.com>\nCo-authored-by: Cursor <cursoragent@cursor.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-09-03T11:19:53+08:00",
          "tree_id": "4210fe59cc2b85a182df3242f7d341f3a30ea43a",
          "url": "https://github.com/ROCm/ATOM/commit/d3b2a3de79e5da2a64ec7eb6857146867c510d2a"
        },
        "date": 1788407595273,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9477,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33710971199 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9424 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7491,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33710971199 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7506 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "junyyang@amd.com",
            "name": "junyyang-amd",
            "username": "junyyang-amd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "db725f422c39cb5d0dfcc4ff90416ef2f7c4b9c2",
          "message": "ci: prevent benchmark dashboard history loss (#2099)\n\n* ci: prevent benchmark dashboard history loss\n\nValidate and normalize existing dashboard data before every benchmark action so parser failures cannot replace history with a single result.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* fix(ci): resolve dashboard helper lint failures\n\nUse Python 3.10 type annotations, correct type exceptions, mark the helper executable, and replace the deprecated reviewdog failure option.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* fix(ci): make dashboard publishing race-safe\n\nSynchronize gh-pages from the remote, validate data before publishing, and reject history shrinkage before any dashboard push.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* style: format dashboard data test\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* ci: kill leftover GPU occupants in vLLM/SGLang plugin preflight\n\nAccuracy jobs were failing HIP smoke tests on leftover VRAM; opt-in occupant kill for plugin CI only, leaving ATOM native preflight unchanged.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* Revert \"ci: kill leftover GPU occupants in vLLM/SGLang plugin preflight\"\n\nPID kill could not signal leftover KFD processes across namespaces, so revert it in favor of docker-level cleanup.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* ci: kill leftover docker containers in vLLM/SGLang plugin preflight\n\nReplace host PID signaling with docker kill of other running containers, keeping the current job container and leaving ATOM native CI unchanged.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\nCo-authored-by: Junyan Yang <junyyang@crs-m2m-cpu-spur-v2-009.us-east2-a.compute.internal>",
          "timestamp": "2026-09-03T14:29:45+08:00",
          "tree_id": "1dcd6cd7c63c6512e9129e02ce78bbd0c949e0b2",
          "url": "https://github.com/ROCm/ATOM/commit/db725f422c39cb5d0dfcc4ff90416ef2f7c4b9c2"
        },
        "date": 1788421862160,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33723474050 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7089,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33723474050 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202609021444 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.6619 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "ZhangLirong",
            "username": "ZhangLirong-amd",
            "email": "lirzhang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "412f5bfe5241e2f15801b5bc6f46c82fbdaff529",
          "message": "feat(dp): DP-vocab-sharded greedy argmax for speculative drafting (#2128)\n\n* feat(dp): DP-vocab-sharded greedy argmax for speculative drafting\n\nMTP/eagle draft compute_argmax_token runs a replicated full-vocab GEMM per DP\nrank ([N, V]) -- weight-read bound at the tiny draft M, re-reading the whole\n[H, V] lm_head every draft step. Shard the vocab across the DP group so each\nrank reads only [H, V/dp]; DP ranks own distinct rows, so hidden is gathered\nfirst, each rank reduces its shard to a packed (max, global_id), and only the\n[Sigma-rows, 2] pack is all-gathered (never the O(vocab) logits) before the\nglobal argmax is sliced back to this rank's rows. Token-identical to the\nreplicated argmax, tie-break included.\n\nShares the gather/pad/project front half with the decode head: the argmax is a\nthird mode of _dp_sharded_logits (alongside all2all / allgather), so a draft\nstep calls it with mode='argmax' and gets ids back.\n\nOn by default (ATOM_DP_DRAFT_ARGMAX); engages only on pure DP for a unified\n(rectangular) draft step, and only while the GEMM is weight-read bound -- past\nATOM_DP_DRAFT_ARGMAX_MAX_ROWS (default 256) rows the hidden gather outweighs the\nread it saves, so it falls back to the replicated argmax. Every gate value is\nDP-agreed, so the verdict stays consistent across ranks.\n\n* fix(dp): harden the draft argmax gate (review)\n\nAddress valarLip's review at 41a3fa1: _can_use_dp_sharded_argmax was the decode\ngate with its safety removed, reachable from contexts the sibling excludes.\n\n- Total predicate: get_dp_group() (which asserts the group exists) now runs\n  last, behind the dp_metadata check -- so an unbuilt DP group is a False, not\n  an AssertionError. The unit-test monkeypatch that worked around the raise is\n  dropped.\n- dp_metadata is not None: the proof running_tokens was DP-reduced. Absent under\n  SGLang dp-attention (aiter DP group >1 while ATOM data_parallel_size ==1) and\n  single-GPU, where the fixed-size all_gather would otherwise post mismatched\n  sizes and deadlock.\n- Skip plugin mode (both the dispatch and the gate): its caller issues one\n  collective per per-rank chunk, so ranks disagree on the count -> DP deadlock.\n- Exclude is_prefill, matching the decode gate.\n\nTests (CPU, no DP group): tests/test_dp_sharded_argmax.py covers the gate\nverdict table, the total-predicate no-raise case, and the rank-major offset\narithmetic of the argmax exchange.\n\nDocstring: the DP path reshapes the GEMM ([M,V]->[dp*M,V/dp]), so its logits are\nnot bitwise-identical to the replicated path -- only the TP path is. Softened\nthe identity claim accordingly.",
          "timestamp": "2026-09-03T15:10:38Z",
          "url": "https://github.com/ROCm/ATOM/commit/412f5bfe5241e2f15801b5bc6f46c82fbdaff529"
        },
        "date": 1788456660084,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33777866079 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202609031453 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9447 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33777866079 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202609031453 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.1,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33777866079 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202609031453 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7574,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33777866079 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202609031453 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7597 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8795,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33777866079 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202609031453 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.351 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
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
          "id": "2ce5f68f83639967d6f228d416a070ac1d1b1c3a",
          "message": "refactor(attn): derive a prefill step's token indices without a Python loop (#2139)\n\n* fix(dspark): let ragged verify shrink, and stop re-deriving the step's spans\n\n`_dspark_apply_ragged` handed `ragged_verify_len` the batch-MAX bonus count, so\nevery request took the largest one's floor. At any real concurrency some request\naccepts all of its drafts on nearly every step, which makes that floor `full_q`\n-- the ragged path had never once shrunk anything, and so had never been\nexercised. Measured from the `decode[bs= tok=]` trace labels: 0/146 decode steps\nshrank before, 122/146 after, with `tok` down to 29 against a rectangular 96 at\nbs=16.\n\n`ragged_verify_lens` now takes the counts as an array, so there is no scalar for\na caller to reach for, and runs the rule vectorized rather than per request\n(14400-case exhaustive parity against the scalar version).\n\nThree defects surfaced only once the path started firing:\n\n* A newly admitted request never got its draft columns staged.\n  `fill_deferred_decode_ids` skips it (`src < 0`) and the host-side staging was\n  gated on `ragged_lens is None`, so it verified its own committed history as\n  drafts -- rejected on sight, so the output stayed correct and only the\n  acceptance rate showed it.\n* The drafter anchor rebuilt `num_bonus` as `mtp_k - num_reject_tokens`, an\n  identity that holds only where `num_reject_tokens` was DEFINED as its\n  complement. On a step that scored no drafts it is a zero buffer, and the\n  offset went negative. The sampler's own count is threaded through instead.\n* `make_compress_plans` derived each token's position from the unreduced\n  `context_lens`, tail-anchoring the plan against attention's span-head anchor.\n\nThe anchor itself collapses to one expression. `prepare_inputs` counted back\nfrom each segment's END, which needs the segment length -- and ragged makes that\nper-request, which is why this carried three special cases (`1 + num_reject`,\nthe q-bucket's `q - num_bonus`, and a ragged branch). `cu[1:]` cancels out of\n`cu[1:] - (len - num_bonus)`, so none of them had to be asked: the anchor is\n`segment start + accepted drafts`, read forward.\n\nOne representation per quantity, which is what let the two disagree:\n\n* `cu_seqlens_q` gets a single publisher (`CommonAttentionBuilder`, which\n  declares the buffer) and a single H2D per step; `prepare_prefill` and the\n  MLA/aiter decode paths cross-check or take views instead of re-uploading.\n  The private `decode_cu` copy is gone -- it was sized by\n  `forward_mode.max_seqlen_q`, which `ForwardMode.decide` raises to the DP-wide\n  maximum, so under dp>1 the ids were written at wider offsets than attention\n  read positions from.\n* `decode_spans()` is the one derivation of `(bs, per-request lengths, prefix\n  sum)`, replacing three copies.\n* `dynamic_spec_query_tokens_per_req`, `last_token_offset`, `draft_ragged_lens`\n  and the `ragged_lens`/`ragged_extend` staging buffers are deleted:\n  `num_scheduled_tokens` states the lengths and every reader reads it, so a\n  shrunk step is that array with smaller entries rather than a second shape.\n\nValidation: unit suite unchanged against HEAD (13 failed / 4818 passed -- the 13\nare a pre-existing aiter signature drift in test_dcp_topk). GSM8K, each against\nits own clean-HEAD control: V4-Flash-DSpark ragged n=6 mean 0.9508 vs ragged-off\nn=3 mean 0.9500 (indistinguishable, ranges overlap); Qwen3.5-27B 0.8867 vs\n0.8733; GLM-5.2-MXFP4 0.9242 vs 0.9265 with strict-match identical. Zero faults\nand zero assertion failures across all runs.\n\n* perf(v4): derive each step's metadata once instead of rebuilding it\n\nThree places built a quantity the step already had:\n\n`make_compress_plans` re-derives every token's position from\n`(ctx - extend) + j`, so its `ctx` must be the seq length THROUGH the last\ntoken this forward writes. The caller was computing that as\n`ctx - (full_q - len_i)` -- a second anchor, correct only as long as it\nstays in step with the one `positions` was built on. `plan_context_lens`\nreads the length off `positions` itself, so the two cannot drift; the\nubatch path derives its own the same way rather than slicing the\nstep-level array, and the parameter that carried it disappears.\n\nThe fp8 decode `qo_indptr` allocated a temporary, filled its head with a\nfresh `arange`, and handed it to `_stage` to copy again -- while the\nconstant it needs is already precomputed and budgeted for exactly this.\nWrite the head slice and the flat pad tail straight into the staging\nbuffer. `_stage`'s capacity check is owed here instead, so it is asserted\nlocally. `prepare_mtp_decode` and the vLLM bridge were already this shape.\n\nThree of the four compress-plan columns do not depend on the ratio, yet\nthe whole `[T, 4]` row block was stacked and cast once per ratio (twice on\nV4). Build it once and rewrite only `window_len`. The cast is not dropped\nbut absorbed: an int32 destination truncates exactly as `.astype` did, so\ncallers that pass int64 lengths -- the plugin bridges, which we cannot run\nhere -- keep their behavior, and the dtype dependency goes away entirely.\n\nSharing the row block across ratios is only safe because both selections\nare boolean fancy-indexing, which numpy always materializes as a copy. The\nexisting ten tests all check counts and none check row content, so that\nproperty had no coverage; one is added. Verified against HEAD over 22\nshapes (eager / decode-CG / verify-cap / empty / single-ratio / ragged /\nint64 inputs), byte-identical on every staged buffer, GPU view, count and\npublished plan. Hoisting `window_len` out of the loop -- the mistake this\nrefactor invites -- drops that to 4/22 and fails the new test.\n\n* refactor(attn): derive a prefill step's token indices without a Python loop\n\nThe per-token loop in `CommonAttentionBuilder.prepare_prefill` built\n`positions` and `slot_mapping` one token at a time, walking each sequence's\nown ragged block list. Both are now numpy over the whole token axis, reading\nthe 2-D table `prepare_block_tables` already packs. That makes the marshal a\nonce-per-step thing serving both the slot arithmetic and the upload, instead\nof the same rows being packed again by every caller that only wanted the\ntable on the device.\n\nThe cached prefix is subtracted out of `context_lens` and the scheduler's\n`num_scheduled_tokens` rather than read from `num_cached_tokens`, so it cannot\ndisagree with the query lengths the rest of the step uses -- and is then\nasserted elementwise against `num_cached_tokens` anyway, because that\ncomparison is the one thing the derivation costs. `context_lens` is\n`cached + scheduled` for a PREFILL row and `seq.num_tokens` for a DECODE one,\nso a decode row among the first `scheduled_bs` shows up there even in the\ncases where the totals agree and the cu_seqlens_q assert lets it through.\n\n26x at the shapes a step runs -- bs=16 x 1024 tokens: 821us -> 31us; bs=32 x\n1024, the 32768-token envelope: 1652us -> 59us -- and slower on a 16-token\nstep, where numpy's dispatch costs more than the loop it replaces. Measure\nthis kind of thing with the arms alternating: timing one to completion and\nthen the other lets each run with numpy's free list holding its own\nallocation pattern, a state no real step ever sees, and on the slot mapping\nthe two harnesses disagree by 2.6x.\n\nWhere things live changed with it. `attentions/` was 14 flat files mixing\nbackends with helpers, so the helpers moved into two packages named for the\nquestion they answer: `pool_layout/` (where a byte lives in the cache pools,\na function of the config) and `token_layout/` (where this step's tokens go, a\nfunction of the batch). Both also import neither aiter nor the rest of atom,\nwhich is what lets a runner with no AITER build check the arithmetic against\na naive reference; `tests/test_layout_packages.py` holds them to that,\nbecause CI is such a runner and one import failure there aborts collection\nfor every test rather than one. The invariant is a property the packages\nhave, not the reason they are separate -- membership is decided by the axis.\n\nThe DCP slot mapping moved to `dcp_ops`, beside the `dcp_owner_rank` /\n`dcp_local_index` pair whose docstring already spelled out the formula it was\nduplicating; the generic builder no longer needs to know what interleaving\nis. Nothing in this tree runs dcp_size > 1, so the move is backed by a\ntranscription of the pre-move body checked over 630 (block_size, W, S, rank,\nshape) combinations rather than by a test.\n\n`num_cached_tokens` becomes a resident mirror like every other per-sequence\narray. It had been a fresh pinned tensor per step whose lifetime was never\nfenced against its own async H2D.\n\nGSM8K on DeepSeek-V4-Flash-DSpark tp2 fp8, 1319 questions, n=3 with a fresh\nserver each: 0.9522 / 0.9553 / 0.9530, mean 0.9535, against 0.9492 for this\nsame branch before the reorganization. The arms overlap, so read it as\nunchanged. DSpark acceptance 64.99-65.54% at 4.25-4.28 tokens/forward, flat.\n\n* refactor(attn): move the page-unit geometry into the package for it\n\n`page_unit_geometry` landed on main while this branch was splitting the flat\n`attentions/` directory into packages named for the question each answers. It\nbelongs in `pool_layout/`, and says so itself: its docstring opens \"Where a\ncheckpoint image's bytes live in the MLA paged pool\" -- the package's topic --\nand justifies its own existence by being reachable without aiter so the\ngeometry deciding where every checkpoint byte goes can be tested on a GPU-less\nrunner, which is the package's invariant. It imports neither aiter nor atom.\n\nThe module no longer restates that invariant; `pool_layout/__init__.py` is\nwhere it is written, and `tests/test_layout_packages.py` now enforces it over\nsix members rather than five.\n\n* refactor(attn): one derivation per quantity on the metadata path\n\n`token_layout` had one module and a prefill-shaped name on a function both\nsides of the forward want. It now splits by what actually differs -- `prefill`\nand `decode` shape the token axis, ragged and rectangular -- from what does\nnot: `slots` maps a position to a KV slot and `batch_ids` maps a token to its\nsequence, and neither cares which side shaped the axis. Both were transcribed\nper caller before, and both go wrong in silence: a slot lands a KV write on\nanother request's row, a batch id reads one.\n\nThe decode slot mapping was a per-token Python loop over each sequence's ragged\nblock-table row, held byte-identically by the MHA and MLA builders. It is now\nthe same gather prefill uses, over the table `prepare_block_tables` packs --\n13-15x at the shapes a speculative decode step runs (bs=256 x 6 tokens:\n94 -> 7 us), and unchanged at block_size 1, 16 and 256, which the ragged loop\nwas checked against. The marshal moves ahead of the slots to make that\npossible; it already ran on the same step, and in the MLA builder it ran twice.\nA caller that trimmed its own rows for rejected drafts may still hand over the\nuntrimmed table, because a token's block index stays under its own row's\ntrimmed length.\n\n`decode_positions` writes the rectangle as a rectangle instead of a tile plus a\nrepeat, which leaves the token axis touched once rather than three times\n(1.3-2.2x). The MLA sparse path stops walking that rectangle a second time for\n`per_token_kv_lens`, which is the position it already has, plus one.\n\nFour transcriptions of the token -> sequence map become one module, and the two\ncallers that built it before calling `_attach_v4_per_fwd_meta` no longer have\nit built again inside. That function now takes the map rather than the\ningredients and the width to pad it to: taking both would be two ways in, and\nthe ingredients were dead the moment the map arrived.\n\nOne name per quantity on the decode path. `running_tokens` is how many token\nrows a forward runs -- previously also `sum_scheduled_tokens_padded`,\n`padded_total_tokens`, `sum_tokens`, `padded_tok_count` and `running_bs *\nmax_q_len` recomputed in four places. `scheduled_tokens` is how many tokens\nwere scheduled onto it -- previously also `sum_scheduled_tokens` and, where it\nmeant this, `total_tokens`. Names left alone where the quantity differs: an\nMLA chunk's KV length, a PCP rank's shard, a TBO ubatch's share.\n\nCUDAGraph capture pads nothing: its per-sequence lengths already sum to the row\ncount, so the metadata is sized to that rather than to `running_bs * max_q_len`\nas the old argument resolved to. On the ragged AF_PIECEWISE path those differ,\nand the wider one disagreed with the `Context` the same function returns. This\nnarrows the buffers there; the capture path is not runnable on the machine this\nwas developed on.\n\nStatic checks for the two defect shapes this refactor kept producing by hand --\nan expression recomputed when a local already holds it, and one name assigned\nto another -- are in `/app/logs_claude/tool/find_redundant_derivations.py`. It\nfound three more in the functions being edited, which are fixed here; it\nreports zero over the new package.\n\nUnit tests 5278 passed; the 22 failures are `test_dcp_topk` (an aiter signature\ndrift) and `test_deepseek_v4_wo_a_dequant` (a test double behind its subject),\nboth present on main. GSM8K on DeepSeek-V4-Flash-DSpark tp2 fp8, 1319\nquestions, n=3 with a fresh server each: 0.9515 / 0.9538 / 0.9492, mean 0.9515\nagainst 0.9535 before -- overlapping arms, read as unchanged; DSpark acceptance\n65.18-65.37% at 4.26 tokens/forward, flat. That model uses the V4 builder, so\nit exercises `backends` and `deepseek_v4_attn` but NOT the two aiter decode\npaths this changes most; those have the unit corpus and the byte-for-byte\ncomparison against the loop they replaced, and want an MLA model to gate.\n\n* style(tests): restore the import-group blank line the formatter took\n\nMoving `page_unit_geometry` into `pool_layout` turned a one-line import into a\nparenthesised one, and reformatting it dropped the blank line between the\nthird-party group and `atom`. The `I001` that follows was already standing on\nmain, so a whole-repo ruff count against main shows nothing -- CI runs ruff\nthrough reviewdog with `-filter-mode=diff_context`, which reports a finding\nonly when it lands on a touched line, and this branch had just touched a line\ninside that import block.\n\n`/app/logs_claude/tool/ruff_in_diff_context.py` answers the question the count\ncannot: which findings CI will actually report. Note it has to intersect a\nfinding's whole RANGE with the diff, not its start row -- `I001` is reported at\nthe top of a block it spans, which is why its first version answered 0 here\nwhile CI answered 1.\n\n* perf(v4): stop building a per-token block table the FP4 indexer never reads\n\n`_attach_v4_paged_decode_meta` expanded `block_tables` to one row per query\ntoken on every decode forward. Its only reader is `deepgemm_fp8_paged_mqa_logits`\non the FP8 path, whose schedule gives one id per CTA that has to serve as both\nthe q row and the block-table row -- so the two are made the same tensor. The FP4\nscorer's schedule carries `row_id` and `batch_id` as separate fields and indexes\n`block_tables` as it stands, and has done since a19cac998 routed decode through\n`flydsl_pa_mqa_logits_fp4_prefill`. Under that indexer the gather has been\nwritten and never read.\n\nIt is not small: `[max_num_seqs * (1 + spec steps), block_table_cols]` int32 is\n24 MiB written per decode step at 256 x DSpark-5 with a 1M context, plus the\nsame again resident.\n\nThe buffer is gated too, not just the gather, so a reader added back without\nungating raises on a missing key instead of reading rows nobody wrote.\n\nBoth plugin bridges keep theirs: `grep -ci fp4` is 0 in each, so their\n`block_tables_per_token` is live.\n\nValidated on DeepSeek-V4-Flash-DSpark tp2, fp8 KV, FP4 indexer confirmed active\nin the server log: GSM8K 1319 questions flexible 0.9522 / strict 0.9530 against\na 0.948-0.950 baseline, MTP acceptance 65.43% at 4.27 toks/fwd against 65.2% /\n4.26. Unit suite 5278 passed / 22 failed, the 22 being test_dcp_topk and\ntest_deepseek_v4_wo_a_dequant, both already failing on main.",
          "timestamp": "2026-09-04T14:28:51Z",
          "url": "https://github.com/ROCm/ATOM/commit/2ce5f68f83639967d6f228d416a070ac1d1b1c3a"
        },
        "date": 1788555099488,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOMesh::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33894404452 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202609041453 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33894404452 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202609041453 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.14,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33894404452 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202609041453 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOMesh::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOMesh::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8886,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33894404452 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202609041453 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3063 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
          }
        ]
      }
    ]
  }
}