window.BENCHMARK_DATA = {
  "lastUpdate": 1781547253269,
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
      }
    ]
  }
}