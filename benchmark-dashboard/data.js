window.BENCHMARK_DATA = {
  "lastUpdate": 1788217324715,
  "repoUrl": "https://github.com/ROCm/ATOM",
  "entries": {
    "Benchmark": [
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
          "id": "171de4553944a8bee86c587928e4afa1211d0448",
          "message": "Log the KV transfer hot path at debug, not info (#2092)\n\nEvery per-request and per-step step of a PD KV transfer was logged at\ninfo: the consumer's write_request, the producer's receipt of it, the\nRDMA write itself, the per-rank and all-stage write-done notifications,\nstart_load_kv, get_finished, build_connector_meta, and the per-seq\nPD transition/first-decode/decode lines in the scheduler. At 48-way\nconcurrency and pp=4 that is ~36 lines per request on the consumer and\n~16 on the producer, which buries the startup and failure lines that\nactually need reading.\n\nMove those to debug. A one-hour 4842-request run drops from 403k to 25k\ndecode lines and from 599k to 228k prefill lines (of which 196k are\nLMCache's own pin-timeout and allocation warnings, not ours).\n\nStartup and registration lines stay at info, as do every warning, error\nand exception on the transfer paths. PD backpressure also stays at info\n— it reports a real stall — but now prints every 1000 schedule ticks\ninstead of every 100.",
          "timestamp": "2026-08-31T21:47:56+08:00",
          "tree_id": "ba5bb683ce43977d3c2e4388844eb77483b86ea2",
          "url": "https://github.com/ROCm/ATOM/commit/171de4553944a8bee86c587928e4afa1211d0448"
        },
        "date": 1788188385258,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP accuracy (GSM8K)",
            "value": 0.9431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Same base model as DeepSeek-R1-0528 FP8 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9401 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP MTP acceptance (%)",
            "value": 67.31,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Same base model as DeepSeek-R1-0528 FP8 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9401 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP avg toks/fwd (tok/fwd)",
            "value": 3.02,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 accuracy (GSM8K)",
            "value": 0.9371,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.931 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP accuracy (GSM8K)",
            "value": 0.9393,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9348 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP MTP acceptance (%)",
            "value": 64.45,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9348 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Full-eval (1319 samples) 3-shot flexible-extract = 0.9522 ± 0.0059 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark MTP acceptance (%)",
            "value": 45,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark avg toks/fwd (tok/fwd)",
            "value": 4.15,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9462,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.08,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Kimi-K2.7-Code-MXFP4 accuracy (GSM8K)",
            "value": 0.9439,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.92 | Baseline: 0.9409 | BaselineModel: moonshotai/Kimi-K2.7-Code | BaselineNote: Kimi-K2.7-Code-MXFP4 native ATOM coverage; threshold inherited from Kimi-K2.5-MXFP4 until CI baseline is refreshed. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9416 | fewshot: 3 | Model: /models/amd/Kimi-K2.7-Code-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K3 accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 | BaselineNote: Kimi-K3 (kimi_linear KDA+MLA, MXFP4 MoE) native ATOM FP8 kv-cache, TP8 (GSM8K 3-shot flexible-extract). Baseline 0.95; threshold 0.94 leaves ~1pp headroom. Refresh after the first CI run. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark MTP acceptance (%)",
            "value": 50.93,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark avg toks/fwd (tok/fwd)",
            "value": 4.57,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4-Preview accuracy (GSM8K)",
            "value": 0.9083,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: meta-llama/Llama-3.3-70B-Instruct | BaselineNote: HF page inaccessible; needs CI measurement of baseline | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.6194 | fewshot: 3 | Model: /models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview"
          },
          {
            "name": "ATOM::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7513,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Instinct MI355X | VRAM: 252GB | ROCm: 7.2.4 | strict-match: 0.7498 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOM::MiniMax-M3-MXFP4 accuracy (GSM8K)",
            "value": 0.9348,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.93 | Baseline: 0.9363 | BaselineModel: amd/MiniMax-M3-MXFP4 | BaselineNote: FP4 M3 tp8. GSM8K 5-shot chat (apply_chat_template + fewshot_as_multiturn, num_concurrent=32, max_gen_toks=16384) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9348 | fewshot: 5 | Model: /models/amd/MiniMax-M3-MXFP4"
          },
          {
            "name": "ATOM::Qwen3-235B-A22B-Instruct-2507-FP8 accuracy (GSM8K)",
            "value": 0.8939,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.87 | Baseline: 0.909 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: HF: amd/Qwen3-235B-A22B-Instruct-2507-MXFP4 card shows baseline=0.909 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8832 | fewshot: 3 | Model: /models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
          },
          {
            "name": "ATOM::Qwen3-Next-80B-A3B-Thinking accuracy (GSM8K)",
            "value": 0.6763,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.65 | Baseline: 0.69 | BaselineModel: Qwen/Qwen3-Next-80B-A3B-Thinking | BaselineNote: No public GSM8K baseline; HF card has no GSM8K | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7938 | fewshot: 3 | Model: /models/Qwen/Qwen3-Next-80B-A3B-Thinking"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-FP8 MTP accuracy (GSM8K)",
            "value": 0.8764,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.85 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: Same base model as Qwen3.5-397B-A17B-FP8; MTP3 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8597 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-FP8 MTP MTP acceptance (%)",
            "value": 84.55,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.85 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: Same base model as Qwen3.5-397B-A17B-FP8; MTP3 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8597 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-FP8 MTP avg toks/fwd (tok/fwd)",
            "value": 3.54,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-MXFP4 accuracy (GSM8K)",
            "value": 0.8537,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.835 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: CI baseline=0.8605. HF card reports 0.9538 but uses chat API with reasoning_parser | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8355 | fewshot: 3 | Model: /models/amd/Qwen3.5-397B-A17B-MXFP4"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-MXFP4 MTP accuracy (GSM8K)",
            "value": 0.8537,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.835 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: CI baseline=0.8605. HF card reports 0.9538 but uses chat API with reasoning_parser | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8393 | fewshot: 3 | Model: /models/amd/Qwen3.5-397B-A17B-MXFP4"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-MXFP4 MTP MTP acceptance (%)",
            "value": 84.59,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.835 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: CI baseline=0.8605. HF card reports 0.9538 but uses chat API with reasoning_parser | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8393 | fewshot: 3 | Model: /models/amd/Qwen3.5-397B-A17B-MXFP4"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-MXFP4 MTP avg toks/fwd (tok/fwd)",
            "value": 3.54,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8901,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33398946975 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.4594 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
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
        "date": 1788199247787,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM::DeepSeek-R1-0528 accuracy (GSM8K)",
            "value": 0.9477,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (GSM8K 3-shot flexible-extract) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9454 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP accuracy (GSM8K)",
            "value": 0.9439,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Same base model as DeepSeek-R1-0528 FP8 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9416 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP MTP acceptance (%)",
            "value": 67.07,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Same base model as DeepSeek-R1-0528 FP8 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9416 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP avg toks/fwd (tok/fwd)",
            "value": 3.01,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant accuracy (GSM8K)",
            "value": 0.9409,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Online quantization on top of DeepSeek-R1-0528 MTP (FP8 native): global ptpc_fp8 + expert layers mxfp4, excluding lm_head and *.gate.*. Threshold set to 0.93 (same headroom as DeepSeek-R1-0528-FP4 MTP) as a conservative placeholder for the MoE-MXFP4 accuracy drop. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9348 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant MTP acceptance (%)",
            "value": 64.32,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Online quantization on top of DeepSeek-R1-0528 MTP (FP8 native): global ptpc_fp8 + expert layers mxfp4, excluding lm_head and *.gate.*. Threshold set to 0.93 (same headroom as DeepSeek-R1-0528-FP4 MTP) as a conservative placeholder for the MoE-MXFP4 accuracy drop. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9348 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Streaming Online-Quant accuracy (GSM8K)",
            "value": 0.9462,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Nightly streaming variant of DeepSeek-R1-0528 MTP online quantization. Covers streaming load with its default settings, mixed FP8/MXFP4 quantization, inference accuracy, and MTP acceptance. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Streaming Online-Quant MTP acceptance (%)",
            "value": 64.26,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Nightly streaming variant of DeepSeek-R1-0528 MTP online quantization. Covers streaming load with its default settings, mixed FP8/MXFP4 quantization, inference accuracy, and MTP acceptance. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Streaming Online-Quant avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 accuracy (GSM8K)",
            "value": 0.9416,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9386 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP accuracy (GSM8K)",
            "value": 0.9393,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9348 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP MTP acceptance (%)",
            "value": 64.46,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9348 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Full-eval (1319 samples) 3-shot flexible-extract = 0.9522 ± 0.0059 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark accuracy (GSM8K)",
            "value": 0.9568,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9568 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark MTP acceptance (%)",
            "value": 45.32,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9568 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark avg toks/fwd (tok/fwd)",
            "value": 4.17,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro EPLB r0 accuracy (GSM8K)",
            "value": 0.9545,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.956 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: EP+DPA, EPLB pure rearrangement (num_redundant_experts=0, no extra memory), rebalance_interval=200. g64 8xMI355X measured GSM8K 5-shot flexible/strict = 0.9560/0.9568 (2026-07-20), 4 rebalances during the eval, 0 crashes. Guards the num_redundant>0 startup-OOM/migration-deadlock fixes (redundant=0 doesn't hit them, but shares the rebalance/migration code path). | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9553 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro EPLB r64 biased accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.955 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: EP+DPA, EPLB biased placement (64 redundant physical experts = top-8 hottest fully replicated to all 8 GPUs), rebalance_interval=200. Exercises fill_redundant init + runtime rebalance/migration end-to-end, guarding the num_redundant>0 startup-OOM/migration-deadlock fixes. g64 8xMI355X measured GSM8K 5-shot flexible/strict = 0.9553/0.9560 (2026-07-20), 4 rebalances including migration during the eval, 0 crashes. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro EPLB r64 naive accuracy (GSM8K)",
            "value": 0.9515,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.956 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: EP+DPA, EPLB naive placement (64 redundant physical experts spread thinly via balanced_packing), rebalance_interval=200. Exercises fill_redundant init + runtime rebalance/migration end-to-end, guarding the num_redundant>0 startup-OOM/migration-deadlock fixes. g64 8xMI355X measured GSM8K 5-shot = 0.956 (2026-07-20), 4 rebalances including migration during the eval, 0 crashes. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9545,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9553 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 66.06,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9553 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO+DPA conc1000 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.95 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: TBO + dp-attention at conc=1000. Local 1319-sample GSM8K 3-shot, 4 runs = 0.9439/0.9484/0.9538/0.9530 (mean ~0.950, 2026-06-14, after TBO ids-gather + pad_for_all_gather fixes). Baseline 0.95; threshold 0.93 (~1.4pp below lowest 0.9439, conc=1000 variance). | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.95 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::GLM-5-FP8 accuracy (GSM8K)",
            "value": 0.9401,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9545 | BaselineModel: zai-org/GLM-5 | BaselineNote: HF: amd/GLM-5-MXFP4 card shows GLM-5 baseline=0.9545 (5-shot) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/zai-org/GLM-5-FP8"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 accuracy (GSM8K)",
            "value": 0.9393,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: ATOM native FP8 gsm8k 3-shot flexible-extract=0.9447 (5-shot=0.9416); --gpu-memory-utilization 0.8 needed since the DSA index cache OOMs at default 0.9. Threshold 0.92 leaves ~2.5pp headroom. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9409 | fewshot: 3 | Model: /models/zai-org/GLM-5.2-FP8"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 accuracy (GSM8K)",
            "value": 0.9181,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9189 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP accuracy (GSM8K)",
            "value": 0.9212,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9219 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP MTP acceptance (%)",
            "value": 75.27,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9219 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP avg toks/fwd (tok/fwd)",
            "value": 3.26,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Kimi-K2.5-MXFP4 Eagle3 accuracy (GSM8K)",
            "value": 0.9416,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.91 | Baseline: 0.9257 | BaselineModel: amd/Kimi-K2.5-MXFP4 + lightseekorg/kimi-k2.5-eagle3 | BaselineNote: Eagle3 spec decode on Kimi-K2.5-MXFP4. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9409 | fewshot: 3 | Model: /models/amd/Kimi-K2.5-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K2.5-MXFP4 Eagle3 MTP acceptance (%)",
            "value": 68.92,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.91 | Baseline: 0.9257 | BaselineModel: amd/Kimi-K2.5-MXFP4 + lightseekorg/kimi-k2.5-eagle3 | BaselineNote: Eagle3 spec decode on Kimi-K2.5-MXFP4. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9409 | fewshot: 3 | Model: /models/amd/Kimi-K2.5-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K2.5-MXFP4 Eagle3 avg toks/fwd (tok/fwd)",
            "value": 3.07,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Kimi-K2.7-Code-MXFP4 accuracy (GSM8K)",
            "value": 0.9477,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.92 | Baseline: 0.9409 | BaselineModel: moonshotai/Kimi-K2.7-Code | BaselineNote: Kimi-K2.7-Code-MXFP4 native ATOM coverage; threshold inherited from Kimi-K2.5-MXFP4 until CI baseline is refreshed. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9477 | fewshot: 3 | Model: /models/amd/Kimi-K2.7-Code-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K3 accuracy (GSM8K)",
            "value": 0.9553,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 | BaselineNote: Kimi-K3 (kimi_linear KDA+MLA, MXFP4 MoE) native ATOM FP8 kv-cache, TP8 (GSM8K 3-shot flexible-extract). Baseline 0.95; threshold 0.94 leaves ~1pp headroom. Refresh after the first CI run. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9553 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark MTP acceptance (%)",
            "value": 51.17,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark avg toks/fwd (tok/fwd)",
            "value": 4.58,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4-Preview accuracy (GSM8K)",
            "value": 0.9151,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: meta-llama/Llama-3.3-70B-Instruct | BaselineNote: HF page inaccessible; needs CI measurement of baseline | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.6346 | fewshot: 3 | Model: /models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview"
          },
          {
            "name": "ATOM::MiMo-V2-Flash accuracy (GSM8K)",
            "value": 0.79,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.778 | Baseline: 0.79 | BaselineModel: XiaomiMiMo/MiMo-V2-Flash | BaselineNote: CI GSM8K 3-shot. First stable run base=0.8082 (run 26410931088, commit 24e4367b). Baseline 0.79 sits ~1.5pp below to absorb run-to-run noise (stderr ±0.011); threshold 0.778 leaves ~1.1σ headroom. tp pinned to 4 to match the MTP entry. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7824 | fewshot: 3 | Model: /models/XiaomiMiMo/MiMo-V2-Flash"
          },
          {
            "name": "ATOM::MiMo-V2-Flash MTP accuracy (GSM8K)",
            "value": 0.8021,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.778 | Baseline: 0.79 | BaselineModel: XiaomiMiMo/MiMo-V2-Flash | BaselineNote: CI GSM8K 3-shot MTP1=0.7983 (run 26410931088). Baseline 0.79; threshold 0.778 (~1.1σ). tp MUST=4, num-speculative-tokens MUST=1: ATOM builds only MTP layer 0 (vLLM _MIMO_V2_FLASH_NUM_MTP_LAYERS=1); more spec → layers 1/2 KV unpopulated, accept craters. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7976 | fewshot: 3 | Model: /models/XiaomiMiMo/MiMo-V2-Flash"
          },
          {
            "name": "ATOM::MiMo-V2-Flash MTP MTP acceptance (%)",
            "value": 93.69,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.778 | Baseline: 0.79 | BaselineModel: XiaomiMiMo/MiMo-V2-Flash | BaselineNote: CI GSM8K 3-shot MTP1=0.7983 (run 26410931088). Baseline 0.79; threshold 0.778 (~1.1σ). tp MUST=4, num-speculative-tokens MUST=1: ATOM builds only MTP layer 0 (vLLM _MIMO_V2_FLASH_NUM_MTP_LAYERS=1); more spec → layers 1/2 KV unpopulated, accept craters. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7976 | fewshot: 3 | Model: /models/XiaomiMiMo/MiMo-V2-Flash"
          },
          {
            "name": "ATOM::MiMo-V2-Flash MTP avg toks/fwd (tok/fwd)",
            "value": 1.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::MiniMax-M2.7 accuracy (GSM8K)",
            "value": 0.8961,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.8872 | Baseline: 0.9022 | BaselineModel: MiniMaxAI/MiniMax-M2.7 | BaselineNote: ATOM CI measured: 0.9022 (gsm8k 3-shot flexible-extract). Threshold = baseline - 0.015. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9333 | fewshot: 3 | Model: /models/MiniMaxAI/MiniMax-M2.7"
          },
          {
            "name": "ATOM::MiniMax-M3-MXFP4 accuracy (GSM8K)",
            "value": 0.9431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9363 | BaselineModel: amd/MiniMax-M3-MXFP4 | BaselineNote: FP4 M3 tp8. GSM8K 5-shot chat (apply_chat_template + fewshot_as_multiturn, num_concurrent=32, max_gen_toks=16384) | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9439 | fewshot: 5 | Model: /models/amd/MiniMax-M3-MXFP4"
          },
          {
            "name": "ATOM::MiniMax-M3-MXFP4 Eagle3 accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9469 | BaselineModel: amd/MiniMax-M3-MXFP4 + Inferact/MiniMax-M3-EAGLE3 | BaselineNote: FP4 M3 + EAGLE3 draft (tp8), lossless vs greedy target. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 5 | Model: /models/amd/MiniMax-M3-MXFP4"
          },
          {
            "name": "ATOM::MiniMax-M3-MXFP4 Eagle3 MTP acceptance (%)",
            "value": 73.42,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.93 | Baseline: 0.9469 | BaselineModel: amd/MiniMax-M3-MXFP4 + Inferact/MiniMax-M3-EAGLE3 | BaselineNote: FP4 M3 + EAGLE3 draft (tp8), lossless vs greedy target. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 5 | Model: /models/amd/MiniMax-M3-MXFP4"
          },
          {
            "name": "ATOM::MiniMax-M3-MXFP4 Eagle3 avg toks/fwd (tok/fwd)",
            "value": 3.2,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Qwen3-235B-A22B-Instruct-2507-FP8 accuracy (GSM8K)",
            "value": 0.8999,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.87 | Baseline: 0.909 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: HF: amd/Qwen3-235B-A22B-Instruct-2507-MXFP4 card shows baseline=0.909 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.884 | fewshot: 3 | Model: /models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
          },
          {
            "name": "ATOM::Qwen3-235B-A22B-Instruct-2507-MXFP4 accuracy (GSM8K)",
            "value": 0.8946,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.87 | Baseline: 0.909 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: HF: amd/Qwen3-235B-A22B-Instruct-2507-MXFP4 card shows baseline=0.909 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8749 | fewshot: 3 | Model: /models/amd/Qwen3-235B-A22B-Instruct-2507-MXFP4"
          },
          {
            "name": "ATOM::Qwen3-Next-80B-A3B-Thinking accuracy (GSM8K)",
            "value": 0.6778,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.65 | Baseline: 0.69 | BaselineModel: Qwen/Qwen3-Next-80B-A3B-Thinking | BaselineNote: No public GSM8K baseline; HF card has no GSM8K | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7923 | fewshot: 3 | Model: /models/Qwen/Qwen3-Next-80B-A3B-Thinking"
          },
          {
            "name": "ATOM::Qwen3.5-35B-A3B TP2 accuracy (GSM8K)",
            "value": 0.8605,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.83 | Baseline: 0.85 | BaselineModel: Qwen/Qwen3.5-35B-A3B | BaselineNote: Mean of first 4 valid CI runs (0.8226 / 0.8529 / 0.8620 / 0.8628). Threshold 0.83 from sglang nightly retained. | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8431 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-35B-A3B"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-FP8 accuracy (GSM8K)",
            "value": 0.8704,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.85 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: CI baseline=0.8605. HF card reports 0.9538 but uses chat API with reasoning_parser | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8438 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-FP8 MTP accuracy (GSM8K)",
            "value": 0.8628,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.85 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: Same base model as Qwen3.5-397B-A17B-FP8; MTP3 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8484 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-FP8 MTP MTP acceptance (%)",
            "value": 84.53,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.85 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: Same base model as Qwen3.5-397B-A17B-FP8; MTP3 | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8484 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-FP8 MTP avg toks/fwd (tok/fwd)",
            "value": 3.54,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-MXFP4 accuracy (GSM8K)",
            "value": 0.8506,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.835 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: CI baseline=0.8605. HF card reports 0.9538 but uses chat API with reasoning_parser | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8378 | fewshot: 3 | Model: /models/amd/Qwen3.5-397B-A17B-MXFP4"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-MXFP4 MTP accuracy (GSM8K)",
            "value": 0.8499,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.835 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: CI baseline=0.8605. HF card reports 0.9538 but uses chat API with reasoning_parser | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8362 | fewshot: 3 | Model: /models/amd/Qwen3.5-397B-A17B-MXFP4"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-MXFP4 MTP MTP acceptance (%)",
            "value": 84.66,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.835 | Baseline: 0.9538 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: CI baseline=0.8605. HF card reports 0.9538 but uses chat API with reasoning_parser | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8362 | fewshot: 3 | Model: /models/amd/Qwen3.5-397B-A17B-MXFP4"
          },
          {
            "name": "ATOM::Qwen3.5-397B-A17B-MXFP4 MTP avg toks/fwd (tok/fwd)",
            "value": 3.54,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::gpt-oss-120b accuracy (GSM8K)",
            "value": 0.8886,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33413111541 | Threshold: 0.87 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | BaselineNote: No public GSM8K baseline available | Docker: rocm/atom-dev:nightly_202608301440 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3487 | fewshot: 3 | Model: /models/openai/gpt-oss-120b"
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
        "date": 1788206708111,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM-SGLang::MI308 Qwen3-32B-FP8 TP8 accuracy (GSM8K)",
            "value": 0.8795,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33321479754 | Threshold: 0.8 | BaselineModel: Qwen/Qwen3-32B-FP8 | BaselineNote: Adds max_gen_toks=1024 for the MI308 CI gsm8k path to avoid truncating Qwen3-32B reasoning output. | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8939 | fewshot: 3 | Model: /models/Qwen/Qwen3-32B-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3.5-35B-A3B-FP8 TP1 accuracy (GSM8K)",
            "value": 0.8431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33321479754 | Threshold: 0.76 | BaselineModel: Qwen/Qwen3.5-35B-A3B-FP8 | BaselineNote: Threshold aligned with the SGLANG accuracy validation workflow target for gsm8k. | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8294 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-35B-A3B-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3.5-35B-A3B-PTPC-FP8 TP1 accuracy (GSM8K)",
            "value": 0.8446,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33321479754 | Threshold: 0.76 | BaselineModel: amd/Qwen3.5-35B-A3B-PTPC-FP8 | BaselineNote: Threshold aligned with the SGLANG accuracy validation workflow target for gsm8k. | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8279 | fewshot: 3 | Model: /models/amd/Qwen3.5-35B-A3B-PTPC-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3.5-397B-A17B-FP8 TP4 accuracy (GSM8K)",
            "value": 0.8779,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33321479754 | Threshold: 0.83 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: Threshold aligned with the SGLANG accuracy validation workflow target for gsm8k. | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8658 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3.5-397B-A17B-FP8 TP8 accuracy (GSM8K)",
            "value": 0.8681,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33321479754 | Threshold: 0.83 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: Threshold aligned with the SGLANG accuracy validation workflow target for gsm8k. | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8506 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
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
        "date": 1788217294574,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=16 throughput (tok/s)",
            "value": 335.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=16 Total Tput (tok/s)",
            "value": 675.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=16 TTFT (ms)",
            "value": 1001.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=16 TPOT (ms)",
            "value": 45.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=16 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=32 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=32 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=32 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=32 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=32 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=4 throughput (tok/s)",
            "value": 97.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=4 Total Tput (tok/s)",
            "value": 196.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=4 TTFT (ms)",
            "value": 560.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=4 TPOT (ms)",
            "value": 39.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=4 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=64 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=64 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=64 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=64 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=64 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=8 throughput (tok/s)",
            "value": 187.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=8 Total Tput (tok/s)",
            "value": 374.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=8 TTFT (ms)",
            "value": 708.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=8 TPOT (ms)",
            "value": 40.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 1024/1024 c=8 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 61440/600 c=64 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 61440/600 c=64 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 61440/600 c=64 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 61440/600 c=64 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 61440/600 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 61440/600 c=64 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=16 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=16 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=16 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=16 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=16 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=32 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=32 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=32 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=32 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=32 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=4 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=4 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=4 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=4 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=4 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=64 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=64 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=64 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=64 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=64 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=8 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=8 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=8 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=8 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 MI308 8192/1024 c=8 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=16 throughput (tok/s)",
            "value": 940.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1890.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=16 TTFT (ms)",
            "value": 248.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=16 TPOT (ms)",
            "value": 16.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=32 throughput (tok/s)",
            "value": 1646.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3288.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=32 TTFT (ms)",
            "value": 353.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=32 TPOT (ms)",
            "value": 18.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=4 throughput (tok/s)",
            "value": 260.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=4 Total Tput (tok/s)",
            "value": 524,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=4 TTFT (ms)",
            "value": 164.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=4 TPOT (ms)",
            "value": 14.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=64 throughput (tok/s)",
            "value": 2686.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5374.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=64 TTFT (ms)",
            "value": 501.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=64 TPOT (ms)",
            "value": 22.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=8 throughput (tok/s)",
            "value": 508.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1012.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=8 TTFT (ms)",
            "value": 199.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=8 TPOT (ms)",
            "value": 15.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 1024/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 61440/600 c=64 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 61440/600 c=64 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 61440/600 c=64 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 61440/600 c=64 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 61440/600 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 61440/600 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=16 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=16 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=16 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=16 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=32 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=32 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=32 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=32 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=4 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=4 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=4 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=4 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=64 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=64 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=64 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=64 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=8 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=8 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=8 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=8 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/33407555396 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:sglang-v0.5.17-nightly_20260830@sha256:7d7a174f2565b22a993893573ca75c75b52b75b430c06fd50b478f45ee09e2be"
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-SGLang::Qwen3-32B-FP8 TP8 MI308 8192/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          }
        ]
      }
    ]
  }
}