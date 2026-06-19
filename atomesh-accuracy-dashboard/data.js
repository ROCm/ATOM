window.BENCHMARK_DATA = {
  "lastUpdate": 1781890115738,
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
      }
    ]
  }
}