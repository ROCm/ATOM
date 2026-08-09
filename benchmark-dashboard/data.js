window.BENCHMARK_DATA = {
  "lastUpdate": 1786239178764,
  "repoUrl": "https://github.com/ROCm/ATOM",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "xinyuazh@amd.com",
            "name": "zhangxinyuanliuhengyu",
            "username": "zhangxinyuanliuhengyu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "024a171ceda358dce2b041195aa24ec503639b62",
          "message": "[sgl+atom]ci(sglang): reduce benchmark matrix job outputs (#1825)\n\n* ci(sglang): reduce benchmark matrix job outputs\n\nAvoid publishing the duplicated full SGLang benchmark matrix as a job output so full-suite runs stay under GitHub's output size limit.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-07T16:42:52+08:00",
          "tree_id": "5341d0175ac35fe7f417aac3541d1252aa087f63",
          "url": "https://github.com/ROCm/ATOM/commit/024a171ceda358dce2b041195aa24ec503639b62"
        },
        "date": 1786097627794,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Same base model as DeepSeek-R1-0528 FP8 | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9439 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP MTP acceptance (%)",
            "value": 67.14,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Same base model as DeepSeek-R1-0528 FP8 | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9439 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP avg toks/fwd (tok/fwd)",
            "value": 3.01,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP accuracy (GSM8K)",
            "value": 0.9431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9393 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP MTP acceptance (%)",
            "value": 64.4,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9393 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Full-eval (1319 samples) 3-shot flexible-extract = 0.9522 ± 0.0059 | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark MTP acceptance (%)",
            "value": 45.47,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark avg toks/fwd (tok/fwd)",
            "value": 4.18,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP accuracy (GSM8K)",
            "value": 0.9287,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9272 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP MTP acceptance (%)",
            "value": 75.79,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9272 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP avg toks/fwd (tok/fwd)",
            "value": 3.27,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Kimi-K2.7-Code-MXFP4 accuracy (GSM8K)",
            "value": 0.9447,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31162851792 | Threshold: 0.92 | Baseline: 0.9409 | BaselineModel: moonshotai/Kimi-K2.7-Code | BaselineNote: Kimi-K2.7-Code-MXFP4 native ATOM coverage; threshold inherited from Kimi-K2.5-MXFP4 until CI baseline is refreshed. | Docker: rocm/atom-dev:nightly_202608070936 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9439 | fewshot: 3 | Model: /models/amd/Kimi-K2.7-Code-MXFP4"
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
          "id": "024a171ceda358dce2b041195aa24ec503639b62",
          "message": "[sgl+atom]ci(sglang): reduce benchmark matrix job outputs (#1825)\n\n* ci(sglang): reduce benchmark matrix job outputs\n\nAvoid publishing the duplicated full SGLang benchmark matrix as a job output so full-suite runs stay under GitHub's output size limit.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-07T08:42:52Z",
          "url": "https://github.com/ROCm/ATOM/commit/024a171ceda358dce2b041195aa24ec503639b62"
        },
        "date": 1786102427678,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_dpa_tp8 8192/1024 c=256 perf point",
            "value": 0,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-DPA-TP8 | random_range_ratio=0.8 | perf_point=%7B%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A0%2C%22concurrency%22%3A256%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_dpa_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Atrue%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A18079.5423%2C%22e2el_ms%22%3A0.0%2C%22e2el_p99%22%3A0.0%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22isl%22%3A8192%2C%22itl_ms%22%3A0.0%2C%22median_e2el_ms%22%3A0.0%2C%22median_itl_ms%22%3A0.0%2C%22median_tpot_ms%22%3A0.0%2C%22median_ttft_ms%22%3A0.0%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A2560%2C%22osl%22%3A1024%2C%22output_tput%22%3A0.0%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Atrue%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A0.0%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d_dpa-isl8192-osl1024-conc256-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786091305000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A0.0%2C%22tpot_ms%22%3A0.0%2C%22tpot_p99%22%3A0.0%2C%22ttft_ms%22%3A0.0%2C%22ttft_p99%22%3A0.0%7D"
          },
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_tp8 8192/1024 c=1 perf point",
            "value": 8.0215,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-TP8 | random_range_ratio=0.8 | perf_point=%7B%22accuracy_fewshot%22%3A3%2C%22accuracy_metric%22%3A%22flexible-extract%22%2C%22accuracy_score%22%3A0.953%2C%22accuracy_score_raw%22%3A%220.9530%22%2C%22accuracy_strict%22%3A0.9522%2C%22accuracy_task%22%3A%22gsm8k%22%2C%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A10%2C%22concurrency%22%3A1%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Afalse%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A144.9706%2C%22e2el_ms%22%3A14496.6548%2C%22e2el_p99%22%3A15716.4244%2C%22gsm8k%22%3A0.953%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22input_tput%22%3A510.2619%2C%22input_tput_per_gpu%22%3A63.7827%2C%22interactivity%22%3A65.9556%2C%22isl%22%3A8192%2C%22itl_ms%22%3A15.1632%2C%22median_e2el_ms%22%3A14144.7315%2C%22median_itl_ms%22%3A15.1735%2C%22median_tpot_ms%22%3A15.1617%2C%22median_ttft_ms%22%3A407.4514%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A10%2C%22osl%22%3A1024%2C%22output_tput%22%3A64.1716%2C%22output_tput_per_gpu%22%3A8.0215%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Afalse%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A0.069%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d-isl8192-osl1024-conc1-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786064986000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A574.4335%2C%22tpot_ms%22%3A15.1632%2C%22tpot_p99%22%3A15.1772%2C%22tput_per_gpu%22%3A35.9021%2C%22ttft_ms%22%3A405.4697%2C%22ttft_p99%22%3A424.8222%7D"
          },
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_tp8 8192/1024 c=128 perf point",
            "value": 389.1732,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-TP8 | random_range_ratio=0.8 | perf_point=%7B%22accuracy_fewshot%22%3A3%2C%22accuracy_metric%22%3A%22flexible-extract%22%2C%22accuracy_score%22%3A0.9439%2C%22accuracy_score_raw%22%3A%220.9439%22%2C%22accuracy_strict%22%3A0.9431%2C%22accuracy_task%22%3A%22gsm8k%22%2C%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A1280%2C%22concurrency%22%3A128%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Afalse%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A378.2815%2C%22e2el_ms%22%3A36380.0799%2C%22e2el_p99%22%3A58011.9826%2C%22gsm8k%22%3A0.9439%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22input_tput%22%3A24994.5134%2C%22input_tput_per_gpu%22%3A3124.3142%2C%22interactivity%22%3A36.0336%2C%22isl%22%3A8192%2C%22itl_ms%22%3A27.5025%2C%22median_e2el_ms%22%3A35675.8436%2C%22median_itl_ms%22%3A27.6738%2C%22median_tpot_ms%22%3A27.7519%2C%22median_ttft_ms%22%3A10193.158%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A1280%2C%22osl%22%3A1024%2C%22output_tput%22%3A3113.3855%2C%22output_tput_per_gpu%22%3A389.1732%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Afalse%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A3.3837%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d-isl8192-osl1024-conc128-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786066866000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A28107.8988%2C%22tpot_ms%22%3A27.5055%2C%22tpot_p99%22%3A29.726%2C%22tput_per_gpu%22%3A1756.7437%2C%22ttft_ms%22%3A11102.3334%2C%22ttft_p99%22%3A32545.1998%7D"
          },
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_tp8 8192/1024 c=16 perf point",
            "value": 102.0825,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-TP8 | random_range_ratio=0.8 | perf_point=%7B%22accuracy_fewshot%22%3A3%2C%22accuracy_metric%22%3A%22flexible-extract%22%2C%22accuracy_score%22%3A0.9447%2C%22accuracy_score_raw%22%3A%220.9447%22%2C%22accuracy_strict%22%3A0.9447%2C%22accuracy_task%22%3A%22gsm8k%22%2C%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A160%2C%22concurrency%22%3A16%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Afalse%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A179.2986%2C%22e2el_ms%22%3A17237.5481%2C%22e2el_p99%22%3A20731.7783%2C%22gsm8k%22%3A0.9447%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22input_tput%22%3A6553.4941%2C%22input_tput_per_gpu%22%3A819.1868%2C%22interactivity%22%3A55.827%2C%22isl%22%3A8192%2C%22itl_ms%22%3A17.8879%2C%22median_e2el_ms%22%3A17167.7807%2C%22median_itl_ms%22%3A17.92%2C%22median_tpot_ms%22%3A17.9125%2C%22median_ttft_ms%22%3A536.3142%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A160%2C%22osl%22%3A1024%2C%22output_tput%22%3A816.6603%2C%22output_tput_per_gpu%22%3A102.0825%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Afalse%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A0.8924%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d-isl8192-osl1024-conc16-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786065807000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A7370.1544%2C%22tpot_ms%22%3A17.8878%2C%22tpot_p99%22%3A18.095%2C%22tput_per_gpu%22%3A460.6346%2C%22ttft_ms%22%3A885.129%2C%22ttft_p99%22%3A4584.4385%7D"
          },
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_tp8 8192/1024 c=2 perf point",
            "value": 15.4306,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-TP8 | random_range_ratio=0.8 | perf_point=%7B%22accuracy_fewshot%22%3A3%2C%22accuracy_metric%22%3A%22flexible-extract%22%2C%22accuracy_score%22%3A0.95%2C%22accuracy_score_raw%22%3A%220.9500%22%2C%22accuracy_strict%22%3A0.9507%2C%22accuracy_task%22%3A%22gsm8k%22%2C%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A20%2C%22concurrency%22%3A2%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Afalse%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A151.5257%2C%22e2el_ms%22%3A15103.7972%2C%22e2el_p99%22%3A16324.9352%2C%22gsm8k%22%3A0.95%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22input_tput%22%3A976.6267%2C%22input_tput_per_gpu%22%3A122.0783%2C%22interactivity%22%3A63.7376%2C%22isl%22%3A8192%2C%22itl_ms%22%3A15.6945%2C%22median_e2el_ms%22%3A15263.4902%2C%22median_itl_ms%22%3A15.7154%2C%22median_tpot_ms%22%3A15.6893%2C%22median_ttft_ms%22%3A425.2923%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A20%2C%22osl%22%3A1024%2C%22output_tput%22%3A123.4444%2C%22output_tput_per_gpu%22%3A15.4306%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Afalse%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A0.132%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d-isl8192-osl1024-conc2-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786065172000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A1100.0711%2C%22tpot_ms%22%3A15.6946%2C%22tpot_p99%22%3A15.7418%2C%22tput_per_gpu%22%3A68.7544%2C%22ttft_ms%22%3A441.2289%2C%22ttft_p99%22%3A677.9234%7D"
          },
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_tp8 8192/1024 c=32 perf point",
            "value": 180.2722,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-TP8 | random_range_ratio=0.8 | perf_point=%7B%22accuracy_fewshot%22%3A3%2C%22accuracy_metric%22%3A%22flexible-extract%22%2C%22accuracy_score%22%3A0.9492%2C%22accuracy_score_raw%22%3A%220.9492%22%2C%22accuracy_strict%22%3A0.95%2C%22accuracy_task%22%3A%22gsm8k%22%2C%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A320%2C%22concurrency%22%3A32%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Afalse%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A205.3444%2C%22e2el_ms%22%3A19742.294%2C%22e2el_p99%22%3A26811.1167%2C%22gsm8k%22%3A0.9492%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22input_tput%22%3A11454.2314%2C%22input_tput_per_gpu%22%3A1431.7789%2C%22interactivity%22%3A50.4337%2C%22isl%22%3A8192%2C%22itl_ms%22%3A19.783%2C%22median_e2el_ms%22%3A19420.7226%2C%22median_itl_ms%22%3A19.82%2C%22median_tpot_ms%22%3A19.828%2C%22median_ttft_ms%22%3A721.9221%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A320%2C%22osl%22%3A1024%2C%22output_tput%22%3A1442.1773%2C%22output_tput_per_gpu%22%3A180.2722%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Afalse%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A1.5584%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d-isl8192-osl1024-conc32-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786066064000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A12896.4088%2C%22tpot_ms%22%3A19.784%2C%22tpot_p99%22%3A21.0858%2C%22tput_per_gpu%22%3A806.0255%2C%22ttft_ms%22%3A1453.9994%2C%22ttft_p99%22%3A8895.8664%7D"
          },
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_tp8 8192/1024 c=4 perf point",
            "value": 29.3029,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-TP8 | random_range_ratio=0.8 | perf_point=%7B%22accuracy_fewshot%22%3A3%2C%22accuracy_metric%22%3A%22flexible-extract%22%2C%22accuracy_score%22%3A0.9515%2C%22accuracy_score_raw%22%3A%220.9515%22%2C%22accuracy_strict%22%3A0.9515%2C%22accuracy_task%22%3A%22gsm8k%22%2C%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A40%2C%22concurrency%22%3A4%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Afalse%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A157.0024%2C%22e2el_ms%22%3A15267.6664%2C%22e2el_p99%22%3A17355.7773%2C%22gsm8k%22%3A0.9515%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22input_tput%22%3A1872.9011%2C%22input_tput_per_gpu%22%3A234.1126%2C%22interactivity%22%3A62.2936%2C%22isl%22%3A8192%2C%22itl_ms%22%3A16.0442%2C%22median_e2el_ms%22%3A15302.0508%2C%22median_itl_ms%22%3A16.0614%2C%22median_tpot_ms%22%3A16.053%2C%22median_ttft_ms%22%3A426.2269%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A40%2C%22osl%22%3A1024%2C%22output_tput%22%3A234.4231%2C%22output_tput_per_gpu%22%3A29.3029%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Afalse%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A0.2548%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d-isl8192-osl1024-conc4-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786065369000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A2107.3242%2C%22tpot_ms%22%3A16.0444%2C%22tpot_p99%22%3A16.1426%2C%22tput_per_gpu%22%3A131.7078%2C%22ttft_ms%22%3A520.9986%2C%22ttft_p99%22%3A1262.0204%7D"
          },
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_tp8 8192/1024 c=64 perf point",
            "value": 294.9957,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-TP8 | random_range_ratio=0.8 | perf_point=%7B%22accuracy_fewshot%22%3A3%2C%22accuracy_metric%22%3A%22flexible-extract%22%2C%22accuracy_score%22%3A0.9484%2C%22accuracy_score_raw%22%3A%220.9484%22%2C%22accuracy_strict%22%3A0.9484%2C%22accuracy_task%22%3A%22gsm8k%22%2C%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A640%2C%22concurrency%22%3A64%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Afalse%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A249.9727%2C%22e2el_ms%22%3A24005.5242%2C%22e2el_p99%22%3A39874.1372%2C%22gsm8k%22%3A0.9484%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22input_tput%22%3A18912.2403%2C%22input_tput_per_gpu%22%3A2364.03%2C%22interactivity%22%3A42.1512%2C%22isl%22%3A8192%2C%22itl_ms%22%3A23.5752%2C%22median_e2el_ms%22%3A23404.503%2C%22median_itl_ms%22%3A23.7028%2C%22median_tpot_ms%22%3A23.7241%2C%22median_ttft_ms%22%3A1255.7451%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A640%2C%22osl%22%3A1024%2C%22output_tput%22%3A2359.9656%2C%22output_tput_per_gpu%22%3A294.9957%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Afalse%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A2.5603%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d-isl8192-osl1024-conc64-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786066384000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A21272.2059%2C%22tpot_ms%22%3A23.5798%2C%22tpot_p99%22%3A24.0511%2C%22tput_per_gpu%22%3A1329.5129%2C%22ttft_ms%22%3A2298.3842%2C%22ttft_p99%22%3A16924.6082%7D"
          },
          {
            "name": "Atomesh::DeepSeek-V4-Pro mi350x_atomesh-vllm_fp4_1p1d_tp8 8192/1024 c=8 perf point",
            "value": 55.3191,
            "unit": "point",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31121070276 | docker_image=rocm/atom-dev:latest | precision=FP4 | display_topology=1P1D-TP8 | random_range_ratio=0.8 | perf_point=%7B%22accuracy_fewshot%22%3A3%2C%22accuracy_metric%22%3A%22flexible-extract%22%2C%22accuracy_score%22%3A0.95%2C%22accuracy_score_raw%22%3A%220.9500%22%2C%22accuracy_strict%22%3A0.95%2C%22accuracy_task%22%3A%22gsm8k%22%2C%22backend%22%3A%22atomesh-vllm%22%2C%22chart_group%22%3A%22atomesh-model-performance%22%2C%22chart_label%22%3A%22MI350X%20%28atomesh-vllm%20FP4%29%22%2C%22client_bench%22%3A%22inferencemax%20bench%22%2C%22completed%22%3A80%2C%22concurrency%22%3A8%2C%22config_label%22%3A%22mi350x_atomesh-vllm_fp4_1p1d_tp8%22%2C%22date%22%3A%222026-08-07%22%2C%22decode_dpa%22%3Afalse%2C%22decode_tp%22%3A8%2C%22decode_workers%22%3A1%2C%22duration%22%3A167.3675%2C%22e2el_ms%22%3A16384.621%2C%22e2el_p99%22%3A19332.3995%2C%22gsm8k%22%3A0.95%2C%22hardware%22%3A%22mi350x%22%2C%22image%22%3A%22rocm%2Fatom-dev%3Alatest%22%2C%22input_tput%22%3A3494.1605%2C%22input_tput_per_gpu%22%3A436.7701%2C%22interactivity%22%3A58.977%2C%22isl%22%3A8192%2C%22itl_ms%22%3A16.945%2C%22median_e2el_ms%22%3A16375.6098%2C%22median_itl_ms%22%3A16.9597%2C%22median_tpot_ms%22%3A16.9558%2C%22median_ttft_ms%22%3A442.4241%2C%22model%22%3A%22DeepSeek-V4-Pro%22%2C%22num_decode_gpu%22%3A8%2C%22num_prefill_gpu%22%3A8%2C%22num_prompts%22%3A80%2C%22osl%22%3A1024%2C%22output_tput%22%3A442.553%2C%22output_tput_per_gpu%22%3A55.3191%2C%22precision%22%3A%22fp4%22%2C%22prefill_dpa%22%3Afalse%2C%22prefill_tp%22%3A8%2C%22prefill_workers%22%3A1%2C%22ratio%22%3A0.8%2C%22req_tput%22%3A0.478%2C%22rocm%22%3A%22%22%2C%22run_id%22%3A%22pd-atom-DeepSeek-V4-Pro-1p1d-isl8192-osl1024-conc8-0.8%22%2C%22run_url%22%3A%22https%3A%2F%2Fgithub.com%2FROCm%2FATOM%2Factions%2Fruns%2F31121070276%22%2C%22slurm_job%22%3A%22%22%2C%22source%22%3A%22ATOMesh%22%2C%22timestamp%22%3A1786065578000%2C%22total_gpu%22%3A16%2C%22total_tput%22%3A3936.7135%2C%22tpot_ms%22%3A16.9451%2C%22tpot_p99%22%3A17.1584%2C%22tput_per_gpu%22%3A246.0446%2C%22ttft_ms%22%3A712.7798%2C%22ttft_p99%22%3A2337.8799%7D"
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
          "id": "514ed7c8e5505ce72207f6f1c2a7eaaa0d36b460",
          "message": "feat(openai): accept Anthropic-style chat tools (#1810)\n\n* feat(openai): accept Anthropic-style chat tools\n\nNormalize Anthropic tool schemas at the OpenAI-compatible endpoint while preserving existing validation behavior.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n* fix(ci): format chat test imports\n\nRemove the extra import-block spacing that triggers Ruff I001.\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n---------\n\nCo-authored-by: Cursor <cursoragent@cursor.com>",
          "timestamp": "2026-08-07T23:38:06+08:00",
          "tree_id": "99618f334e3f0c1fce7aed403e88be1b8ccd3f27",
          "url": "https://github.com/ROCm/ATOM/commit/514ed7c8e5505ce72207f6f1c2a7eaaa0d36b460"
        },
        "date": 1786120288116,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 accuracy (GSM8K)",
            "value": 0.9333,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.931 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP accuracy (GSM8K)",
            "value": 0.9401,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9363 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP MTP acceptance (%)",
            "value": 64.25,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9363 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Full-eval (1319 samples) 3-shot flexible-extract = 0.9522 ± 0.0059 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.953 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark accuracy (GSM8K)",
            "value": 0.887,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.8848 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark MTP acceptance (%)",
            "value": 0.03,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.8848 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark avg toks/fwd (tok/fwd)",
            "value": 1,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9575,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9583 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.66,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9583 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP accuracy (GSM8K)",
            "value": 0.9204,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9227 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP MTP acceptance (%)",
            "value": 75.53,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9227 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP avg toks/fwd (tok/fwd)",
            "value": 3.27,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Kimi-K2.7-Code-MXFP4 accuracy (GSM8K)",
            "value": 0.9538,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.92 | Baseline: 0.9409 | BaselineModel: moonshotai/Kimi-K2.7-Code | BaselineNote: Kimi-K2.7-Code-MXFP4 native ATOM coverage; threshold inherited from Kimi-K2.5-MXFP4 until CI baseline is refreshed. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9538 | fewshot: 3 | Model: /models/amd/Kimi-K2.7-Code-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K3 accuracy (GSM8K)",
            "value": 0.9545,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 | BaselineNote: Kimi-K3 (kimi_linear KDA+MLA, MXFP4 MoE) native ATOM FP8 kv-cache, TP8 (GSM8K 3-shot flexible-extract). Baseline 0.95; threshold 0.94 leaves ~1pp headroom. Refresh after the first CI run. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9545 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark accuracy (GSM8K)",
            "value": 0.9545,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9545 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark MTP acceptance (%)",
            "value": 47.48,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9545 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark avg toks/fwd (tok/fwd)",
            "value": 4.32,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4-Preview accuracy (GSM8K)",
            "value": 0.9098,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: meta-llama/Llama-3.3-70B-Instruct | BaselineNote: HF page inaccessible; needs CI measurement of baseline | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.6042 | fewshot: 3 | Model: /models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview"
          },
          {
            "name": "ATOM::Qwen3-235B-A22B-Instruct-2507-FP8 accuracy (GSM8K)",
            "value": 0.8969,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31193602091 | Threshold: 0.87 | Baseline: 0.909 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: HF: amd/Qwen3-235B-A22B-Instruct-2507-MXFP4 card shows baseline=0.909 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.8772 | fewshot: 3 | Model: /models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
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
        "date": 1786123624044,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant accuracy (GSM8K)",
            "value": 0.9431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Online quantization on top of DeepSeek-R1-0528 MTP (FP8 native): global ptpc_fp8 + expert layers mxfp4, excluding lm_head and *.gate.*. Threshold set to 0.93 (same headroom as DeepSeek-R1-0528-FP4 MTP) as a conservative placeholder for the MoE-MXFP4 accuracy drop. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9409 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant MTP acceptance (%)",
            "value": 64.32,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Online quantization on top of DeepSeek-R1-0528 MTP (FP8 native): global ptpc_fp8 + expert layers mxfp4, excluding lm_head and *.gate.*. Threshold set to 0.93 (same headroom as DeepSeek-R1-0528-FP4 MTP) as a conservative placeholder for the MoE-MXFP4 accuracy drop. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9409 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 accuracy (GSM8K)",
            "value": 0.9348,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9325 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP accuracy (GSM8K)",
            "value": 0.9401,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9371 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP MTP acceptance (%)",
            "value": 64.55,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9371 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Full-eval (1319 samples) 3-shot flexible-extract = 0.9522 ± 0.0059 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark accuracy (GSM8K)",
            "value": 0.9462,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9469 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark MTP acceptance (%)",
            "value": 45.13,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9469 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark avg toks/fwd (tok/fwd)",
            "value": 4.16,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro EPLB r0 accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.956 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: EP+DPA, EPLB pure rearrangement (num_redundant_experts=0, no extra memory), rebalance_interval=200. g64 8xMI355X measured GSM8K 5-shot flexible/strict = 0.9560/0.9568 (2026-07-20), 4 rebalances during the eval, 0 crashes. Guards the num_redundant>0 startup-OOM/migration-deadlock fixes (redundant=0 doesn't hit them, but shares the rebalance/migration code path). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro EPLB r64 biased accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.955 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: EP+DPA, EPLB biased placement (64 redundant physical experts = top-8 hottest fully replicated to all 8 GPUs), rebalance_interval=200. Exercises fill_redundant init + runtime rebalance/migration end-to-end, guarding the num_redundant>0 startup-OOM/migration-deadlock fixes. g64 8xMI355X measured GSM8K 5-shot flexible/strict = 0.9553/0.9560 (2026-07-20), 4 rebalances including migration during the eval, 0 crashes. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro EPLB r64 naive accuracy (GSM8K)",
            "value": 0.9538,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.956 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: EP+DPA, EPLB naive placement (64 redundant physical experts spread thinly via balanced_packing), rebalance_interval=200. Exercises fill_redundant init + runtime rebalance/migration end-to-end, guarding the num_redundant>0 startup-OOM/migration-deadlock fixes. g64 8xMI355X measured GSM8K 5-shot = 0.956 (2026-07-20), 4 rebalances including migration during the eval, 0 crashes. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9545 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9591,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9591 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.66,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9591 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO+DPA conc1000 accuracy (GSM8K)",
            "value": 0.9416,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.95 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: TBO + dp-attention at conc=1000. Local 1319-sample GSM8K 3-shot, 4 runs = 0.9439/0.9484/0.9538/0.9530 (mean ~0.950, 2026-06-14, after TBO ids-gather + pad_for_all_gather fixes). Baseline 0.95; threshold 0.93 (~1.4pp below lowest 0.9439, conc=1000 variance). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9431 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::GLM-5-FP8 accuracy (GSM8K)",
            "value": 0.9424,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.93 | Baseline: 0.9545 | BaselineModel: zai-org/GLM-5 | BaselineNote: HF: amd/GLM-5-MXFP4 card shows GLM-5 baseline=0.9545 (5-shot) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9484 | fewshot: 3 | Model: /models/zai-org/GLM-5-FP8"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 accuracy (GSM8K)",
            "value": 0.9431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: ATOM native FP8 gsm8k 3-shot flexible-extract=0.9447 (5-shot=0.9416); --gpu-memory-utilization 0.8 needed since the DSA index cache OOMs at default 0.9. Threshold 0.92 leaves ~2.5pp headroom. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9439 | fewshot: 3 | Model: /models/zai-org/GLM-5.2-FP8"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 accuracy (GSM8K)",
            "value": 0.9234,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9234 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP accuracy (GSM8K)",
            "value": 0.9143,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9143 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP MTP acceptance (%)",
            "value": 75.75,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9143 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP avg toks/fwd (tok/fwd)",
            "value": 3.27,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Kimi-K2.5-MXFP4 Eagle3 accuracy (GSM8K)",
            "value": 0.9371,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.91 | Baseline: 0.9257 | BaselineModel: amd/Kimi-K2.5-MXFP4 + lightseekorg/kimi-k2.5-eagle3 | BaselineNote: Eagle3 spec decode on Kimi-K2.5-MXFP4. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9356 | fewshot: 3 | Model: /models/amd/Kimi-K2.5-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K2.5-MXFP4 Eagle3 MTP acceptance (%)",
            "value": 68.94,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.91 | Baseline: 0.9257 | BaselineModel: amd/Kimi-K2.5-MXFP4 + lightseekorg/kimi-k2.5-eagle3 | BaselineNote: Eagle3 spec decode on Kimi-K2.5-MXFP4. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9356 | fewshot: 3 | Model: /models/amd/Kimi-K2.5-MXFP4"
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
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.92 | Baseline: 0.9409 | BaselineModel: moonshotai/Kimi-K2.7-Code | BaselineNote: Kimi-K2.7-Code-MXFP4 native ATOM coverage; threshold inherited from Kimi-K2.5-MXFP4 until CI baseline is refreshed. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9484 | fewshot: 3 | Model: /models/amd/Kimi-K2.7-Code-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K3 accuracy (GSM8K)",
            "value": 0.9515,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 | BaselineNote: Kimi-K3 (kimi_linear KDA+MLA, MXFP4 MoE) native ATOM FP8 kv-cache, TP8 (GSM8K 3-shot flexible-extract). Baseline 0.95; threshold 0.94 leaves ~1pp headroom. Refresh after the first CI run. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9515 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9515 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark MTP acceptance (%)",
            "value": 47.58,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9515 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark avg toks/fwd (tok/fwd)",
            "value": 4.33,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4-Preview accuracy (GSM8K)",
            "value": 0.9158,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: meta-llama/Llama-3.3-70B-Instruct | BaselineNote: HF page inaccessible; needs CI measurement of baseline | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.6293 | fewshot: 3 | Model: /models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview"
          },
          {
            "name": "ATOM::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7468,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X | VRAM: 252GB | ROCm: unknown | strict-match: 0.7453 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOM::MiniMax-M2.7 accuracy (GSM8K)",
            "value": 0.8984,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.8872 | Baseline: 0.9022 | BaselineModel: MiniMaxAI/MiniMax-M2.7 | BaselineNote: ATOM CI measured: 0.9022 (gsm8k 3-shot flexible-extract). Threshold = baseline - 0.015. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.928 | fewshot: 3 | Model: /models/MiniMaxAI/MiniMax-M2.7"
          },
          {
            "name": "ATOM::Qwen3-235B-A22B-Instruct-2507-FP8 accuracy (GSM8K)",
            "value": 0.8931,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.87 | Baseline: 0.909 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: HF: amd/Qwen3-235B-A22B-Instruct-2507-MXFP4 card shows baseline=0.909 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.8764 | fewshot: 3 | Model: /models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
          },
          {
            "name": "ATOM::Qwen3-235B-A22B-Instruct-2507-MXFP4 accuracy (GSM8K)",
            "value": 0.8848,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197025348 | Threshold: 0.87 | Baseline: 0.909 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: HF: amd/Qwen3-235B-A22B-Instruct-2507-MXFP4 card shows baseline=0.909 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.8719 | fewshot: 3 | Model: /models/amd/Qwen3-235B-A22B-Instruct-2507-MXFP4"
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
        "date": 1786147818279,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM-vLLM::DeepSeek-R1-0528-MXFP4 TP8 accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.93 | Baseline: 0.93 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-R1-FP8 TP8 accuracy (GSM8K)",
            "value": 0.9462,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.93 | Baseline: 0.93 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9409 | fewshot: 3 | Model: deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8 MTP TP4 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.93 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 20 | Model: deepseek-ai/DeepSeek-V3.2"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8 PTPC TP4 accuracy (GSM8K)",
            "value": 0.95,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.93 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 20 | Model: amd/DeepSeek-V3.2-mtp-ptpc"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8 TP4 accuracy (GSM8K)",
            "value": 0.9515,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.93 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9515 | fewshot: 20 | Model: deepseek-ai/DeepSeek-V3.2"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8 TP8 accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.93 | Baseline: 0.956 | BaselineModel: deepseek-ai/DeepSeek-V3.2 | BaselineNote: 20-shot gsm8k reference from DeepSeek-V3.2 usage docs; nightly uses 20-shot to exercise sparse MLA. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9507 | fewshot: 20 | Model: deepseek-ai/DeepSeek-V3.2"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro TP8 accuracy (GSM8K)",
            "value": 0.8203,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.94 | Baseline: 0.94 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: 20-shot GSM8K local-completions coverage aligned with launch.sh/lm_eval.sh. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.721 | fewshot: 20 | Model: deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM-vLLM::GLM-4.7-FP8 MTP TP4 accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: zai-org/GLM-4.7-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-4.7-FP8 MTP TP8 accuracy (GSM8K)",
            "value": 0.9439,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9409 | fewshot: 3 | Model: zai-org/GLM-4.7-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-4.7-FP8 TP4 accuracy (GSM8K)",
            "value": 0.9469,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9447 | fewshot: 3 | Model: zai-org/GLM-4.7-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-4.7-FP8 TP8 accuracy (GSM8K)",
            "value": 0.9333,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Baseline: 0.9386 | BaselineModel: zai-org/GLM-4.7-FP8 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.928 | fewshot: 3 | Model: zai-org/GLM-4.7-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-5.1-FP8 TP8 accuracy (GSM8K)",
            "value": 0.9431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.88 | Baseline: 0.9545 | BaselineModel: zai-org/GLM-5.1 | BaselineNote: CI uses 3-shot, not comparable to HF 5-shot baseline | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 20 | Model: zai-org/GLM-5.1-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-FP8 TP4 accuracy (GSM8K)",
            "value": 0.9393,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: 20-shot GSM8K local-completions coverage for GLM-5.2-FP8 IndexShare; threshold follows the existing GLM-5.2 nightly gate until FP8 CI baseline is recalibrated. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9401 | fewshot: 20 | Model: zai-org/GLM-5.2-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4 MTP TP4 accuracy (GSM8K)",
            "value": 0.9469,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: 20-shot GSM8K is lossless for MTP; threshold follows GLM-5.2-FP8 until MXFP4 MTP-specific CI baseline is calibrated. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9462 | fewshot: 20 | Model: amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4 TP4 accuracy (GSM8K)",
            "value": 0.931,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: 20-shot GSM8K local-completions coverage for GLM-5.2-MXFP4 IndexShare; threshold/baseline follow GLM-5.2-FP8 until MXFP4 CI baseline is calibrated. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.931 | fewshot: 20 | Model: amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2-Thinking-MXFP4 TP4 accuracy (GSM8K)",
            "value": 0.928,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.9 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9265 | fewshot: 3 | Model: amd/Kimi-K2-Thinking-MXFP4-AttnFP8"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2-Thinking-MXFP4 TP8 accuracy (GSM8K)",
            "value": 0.9325,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.9 | Baseline: 0.9 | BaselineModel: amd/Kimi-K2-Thinking-MXFP4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9287 | fewshot: 3 | Model: amd/Kimi-K2-Thinking-MXFP4-AttnFP8"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.5-MXFP4 TP4 accuracy (GSM8K)",
            "value": 0.9333,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.931 | fewshot: 3 | Model: amd/Kimi-K2.5-MXFP4-AttnFP8"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.5-MXFP4 TP8 accuracy (GSM8K)",
            "value": 0.9242,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.93 | Baseline: 0.93 | BaselineModel: amd/Kimi-K2.5-MXFP4 | BaselineNote: Reference value from recipes/atom_vllm/Kimi-K2.5.md | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9249 | fewshot: 3 | Model: amd/Kimi-K2.5-MXFP4-AttnFP8"
          },
          {
            "name": "ATOM-vLLM::Llama-3.1-8B-Instruct TP1 accuracy (GSM8K)",
            "value": 0.7544,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Llama-3.1-8B-Instruct | BaselineNote: Threshold aligned with existing 8B Llama baseline used in CI (3-shot GSM8K). | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.6596 | fewshot: 3 | Model: meta-llama/Llama-3.1-8B-Instruct"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M2.5 TP2 accuracy (GSM8K)",
            "value": 0.9356,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9318 | fewshot: 3 | Model: MiniMaxAI/MiniMax-M2.5"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M2.5 TP4 accuracy (GSM8K)",
            "value": 0.9249,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9227 | fewshot: 3 | Model: MiniMaxAI/MiniMax-M2.5"
          },
          {
            "name": "ATOM-vLLM::Qwen3-235B-A22B-Instruct-2507-FP8 TP8+EP8 accuracy (GSM8K)",
            "value": 0.8969,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.87 | Baseline: 0.87 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8795 | fewshot: 3 | Model: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8 TP1 accuracy (GSM8K)",
            "value": 0.7998,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.81 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7142 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8 TP2 accuracy (GSM8K)",
            "value": 0.8029,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.81 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.721 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8 TP4 accuracy (GSM8K)",
            "value": 0.0697,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.81 | Baseline: 0.76 | BaselineModel: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.0644 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8-MTP TP1 accuracy (GSM8K)",
            "value": 0.8082,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.8 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.718 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8-MTP TP4 accuracy (GSM8K)",
            "value": 0.0614,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.8 | Baseline: 0.81 | BaselineModel: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | BaselineNote: Qwen3-Next-80B-A3B-Instruct-FP8 baseline with TP4 (no MTP) as proxy; needs CI measurement for MTP-specific baseline | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.0599 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B TP8 accuracy (GSM8K)",
            "value": 0.8537,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.83 | Baseline: 0.83 | BaselineModel: Qwen/Qwen3.5-397B-A17B | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8362 | fewshot: 3 | Model: Qwen/Qwen3.5-397B-A17B"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-FP8 TP4 accuracy (GSM8K)",
            "value": 0.8795,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.83 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8575 | fewshot: 3 | Model: Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-FP8 TP8 accuracy (GSM8K)",
            "value": 0.8628,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.83 | Baseline: 0.83 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8491 | fewshot: 3 | Model: Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4 TP4 accuracy (GSM8K)",
            "value": 0.8575,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.83 | Baseline: 0.82 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: Using Qwen3-235B baseline as proxy; needs CI measurement for Qwen3.5 specific baseline | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8415 | fewshot: 3 | Model: amd/Qwen3.5-397B-A17B-MXFP4"
          },
          {
            "name": "ATOM-vLLM::gpt-oss-120b TP1 accuracy (GSM8K)",
            "value": 0.8976,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3017 | fewshot: 3 | Model: openai/gpt-oss-120b"
          },
          {
            "name": "ATOM-vLLM::gpt-oss-120b TP2 accuracy (GSM8K)",
            "value": 0.8916,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.489 | fewshot: 3 | Model: openai/gpt-oss-120b"
          },
          {
            "name": "ATOM-vLLM::gpt-oss-120b TP8 accuracy (GSM8K)",
            "value": 0.8893,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197009756 | Threshold: 0.88 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.1706 | fewshot: 3 | Model: openai/gpt-oss-120b"
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
        "date": 1786149015906,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 throughput (tok/s)",
            "value": 2839.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5684.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 TTFT (ms)",
            "value": 449.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 TPOT (ms)",
            "value": 41.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 throughput (tok/s)",
            "value": 759.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1527.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 TTFT (ms)",
            "value": 237.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 TPOT (ms)",
            "value": 20.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 throughput (tok/s)",
            "value": 4423.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8843.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 TTFT (ms)",
            "value": 690.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 TPOT (ms)",
            "value": 55.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 throughput (tok/s)",
            "value": 1284.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2565.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 TTFT (ms)",
            "value": 259.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 TPOT (ms)",
            "value": 23.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 throughput (tok/s)",
            "value": 222.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 Total Tput (tok/s)",
            "value": 446.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 TTFT (ms)",
            "value": 238.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 TPOT (ms)",
            "value": 17.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 throughput (tok/s)",
            "value": 2035.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4072.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 TTFT (ms)",
            "value": 385.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 TPOT (ms)",
            "value": 30.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 throughput (tok/s)",
            "value": 414.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 Total Tput (tok/s)",
            "value": 826.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 TTFT (ms)",
            "value": 192.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 TPOT (ms)",
            "value": 18.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 throughput (tok/s)",
            "value": 1938.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 Total Tput (tok/s)",
            "value": 17499.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 TTFT (ms)",
            "value": 2171.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 TPOT (ms)",
            "value": 62.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 throughput (tok/s)",
            "value": 613.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 Total Tput (tok/s)",
            "value": 5535.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 TTFT (ms)",
            "value": 630.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 TPOT (ms)",
            "value": 24.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 throughput (tok/s)",
            "value": 2330.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 Total Tput (tok/s)",
            "value": 20969.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 TTFT (ms)",
            "value": 4258.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 TPOT (ms)",
            "value": 103.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 throughput (tok/s)",
            "value": 1048.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 Total Tput (tok/s)",
            "value": 9378.05,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 TTFT (ms)",
            "value": 806.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 TPOT (ms)",
            "value": 28.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 throughput (tok/s)",
            "value": 192.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 Total Tput (tok/s)",
            "value": 1730.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 TTFT (ms)",
            "value": 390.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 TPOT (ms)",
            "value": 19.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 throughput (tok/s)",
            "value": 1490.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 Total Tput (tok/s)",
            "value": 13438.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 TTFT (ms)",
            "value": 1282.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 TPOT (ms)",
            "value": 40.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 throughput (tok/s)",
            "value": 385.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3433.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 TTFT (ms)",
            "value": 473.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 TPOT (ms)",
            "value": 19.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 throughput (tok/s)",
            "value": 3606.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 Total Tput (tok/s)",
            "value": 7220.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 TTFT (ms)",
            "value": 4524.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 TPOT (ms)",
            "value": 29.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 throughput (tok/s)",
            "value": 5620.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 Total Tput (tok/s)",
            "value": 11234.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 TTFT (ms)",
            "value": 4430.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 TPOT (ms)",
            "value": 38.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 throughput (tok/s)",
            "value": 9161.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 Total Tput (tok/s)",
            "value": 18315.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 TTFT (ms)",
            "value": 4577.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 TPOT (ms)",
            "value": 48.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 throughput (tok/s)",
            "value": 1983.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3966.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 TTFT (ms)",
            "value": 2999.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 TPOT (ms)",
            "value": 26.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 throughput (tok/s)",
            "value": 4600.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 Total Tput (tok/s)",
            "value": 41394.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 TTFT (ms)",
            "value": 17226.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 TPOT (ms)",
            "value": 200.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 throughput (tok/s)",
            "value": 2400.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 Total Tput (tok/s)",
            "value": 21675.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 TTFT (ms)",
            "value": 5634.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 TPOT (ms)",
            "value": 45.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 throughput (tok/s)",
            "value": 3331.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 Total Tput (tok/s)",
            "value": 29975.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 TTFT (ms)",
            "value": 7054,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 TPOT (ms)",
            "value": 67.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 throughput (tok/s)",
            "value": 3939.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 Total Tput (tok/s)",
            "value": 35451.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 TTFT (ms)",
            "value": 10496.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 TPOT (ms)",
            "value": 114.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 throughput (tok/s)",
            "value": 1516.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 Total Tput (tok/s)",
            "value": 13672.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 TTFT (ms)",
            "value": 5039.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 TPOT (ms)",
            "value": 35.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 throughput (tok/s)",
            "value": 3680.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 Total Tput (tok/s)",
            "value": 7353.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 TTFT (ms)",
            "value": 9466.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 TPOT (ms)",
            "value": 53.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 1.99,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 Acceptance Rate (%)",
            "value": 14.21,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 throughput (tok/s)",
            "value": 8563.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 Total Tput (tok/s)",
            "value": 17104.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 TTFT (ms)",
            "value": 8392.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 TPOT (ms)",
            "value": 46.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 Accept Length (tok/fwd)",
            "value": 2.68,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 Acceptance Rate (%)",
            "value": 23.99,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 throughput (tok/s)",
            "value": 1520.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3039.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 TTFT (ms)",
            "value": 6657.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 TPOT (ms)",
            "value": 29.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.1,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 Acceptance Rate (%)",
            "value": 15.73,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 throughput (tok/s)",
            "value": 4375.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 Total Tput (tok/s)",
            "value": 39285.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 TTFT (ms)",
            "value": 11210.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 TPOT (ms)",
            "value": 101.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 Accept Length (tok/fwd)",
            "value": 2.65,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 Acceptance Rate (%)",
            "value": 23.56,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 throughput (tok/s)",
            "value": 4112.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8231.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 TTFT (ms)",
            "value": 3934.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 TPOT (ms)",
            "value": 25.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.24,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 Acceptance Rate (%)",
            "value": 41.19,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 throughput (tok/s)",
            "value": 6363.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 12719.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 TTFT (ms)",
            "value": 6461.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 TPOT (ms)",
            "value": 31.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.3,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 Acceptance Rate (%)",
            "value": 43.47,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 throughput (tok/s)",
            "value": 9276.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 Total Tput (tok/s)",
            "value": 18540.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 TTFT (ms)",
            "value": 6967.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 TPOT (ms)",
            "value": 45.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 Accept Length (tok/fwd)",
            "value": 2.29,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 Acceptance Rate (%)",
            "value": 42.91,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 throughput (tok/s)",
            "value": 2828.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5658.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 TTFT (ms)",
            "value": 3657.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 TPOT (ms)",
            "value": 17.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.17,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 Acceptance Rate (%)",
            "value": 38.94,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 throughput (tok/s)",
            "value": 4807.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 Total Tput (tok/s)",
            "value": 43222.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 TTFT (ms)",
            "value": 15141.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 TPOT (ms)",
            "value": 192.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 Accept Length (tok/fwd)",
            "value": 2.52,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 Acceptance Rate (%)",
            "value": 50.62,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 throughput (tok/s)",
            "value": 2918.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 26327.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 TTFT (ms)",
            "value": 6778.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 TPOT (ms)",
            "value": 34.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.51,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 Acceptance Rate (%)",
            "value": 50.46,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 throughput (tok/s)",
            "value": 3533.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 Total Tput (tok/s)",
            "value": 31758.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 TTFT (ms)",
            "value": 10549.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 TPOT (ms)",
            "value": 59.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.55,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 Acceptance Rate (%)",
            "value": 51.56,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 throughput (tok/s)",
            "value": 4212.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 Total Tput (tok/s)",
            "value": 37877.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 TTFT (ms)",
            "value": 12476.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 TPOT (ms)",
            "value": 105.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 Accept Length (tok/fwd)",
            "value": 2.56,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 Acceptance Rate (%)",
            "value": 52.04,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 throughput (tok/s)",
            "value": 1938.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 17454.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 TTFT (ms)",
            "value": 6131.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 TPOT (ms)",
            "value": 25.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.5,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 Acceptance Rate (%)",
            "value": 49.91,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 throughput (tok/s)",
            "value": 5298.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 Total Tput (tok/s)",
            "value": 10592.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 TTFT (ms)",
            "value": 4164.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 TPOT (ms)",
            "value": 41.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 throughput (tok/s)",
            "value": 7675.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 Total Tput (tok/s)",
            "value": 15344.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 TTFT (ms)",
            "value": 4663.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 TPOT (ms)",
            "value": 56.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 throughput (tok/s)",
            "value": 5114.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 Total Tput (tok/s)",
            "value": 46019.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 TTFT (ms)",
            "value": 12977.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 TPOT (ms)",
            "value": 182.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 throughput (tok/s)",
            "value": 3239.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 Total Tput (tok/s)",
            "value": 29145.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 TTFT (ms)",
            "value": 6278.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 TPOT (ms)",
            "value": 69.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 throughput (tok/s)",
            "value": 4369.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 Total Tput (tok/s)",
            "value": 39321.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 TTFT (ms)",
            "value": 9041.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 TPOT (ms)",
            "value": 104.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 throughput (tok/s)",
            "value": 3098.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6202.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 TTFT (ms)",
            "value": 540.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 TPOT (ms)",
            "value": 39.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.24,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 Acceptance Rate (%)",
            "value": 41.19,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 throughput (tok/s)",
            "value": 967.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1944.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 TTFT (ms)",
            "value": 270.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 TPOT (ms)",
            "value": 15.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.24,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 Acceptance Rate (%)",
            "value": 41.47,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 throughput (tok/s)",
            "value": 3945.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 7885.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 TTFT (ms)",
            "value": 792.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 TPOT (ms)",
            "value": 62.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.29,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 Acceptance Rate (%)",
            "value": 43.08,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 throughput (tok/s)",
            "value": 1677.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3347.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 TTFT (ms)",
            "value": 324.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 TPOT (ms)",
            "value": 18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.19,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 Acceptance Rate (%)",
            "value": 39.83,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 throughput (tok/s)",
            "value": 348.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 700.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 TTFT (ms)",
            "value": 227.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 TPOT (ms)",
            "value": 10.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.26,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 Acceptance Rate (%)",
            "value": 41.96,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 throughput (tok/s)",
            "value": 2295.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4592.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 TTFT (ms)",
            "value": 378.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 TPOT (ms)",
            "value": 26.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.16,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 Acceptance Rate (%)",
            "value": 38.83,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 throughput (tok/s)",
            "value": 711.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1417.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 TTFT (ms)",
            "value": 249.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 TPOT (ms)",
            "value": 10.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.32,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 Acceptance Rate (%)",
            "value": 43.87,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 throughput (tok/s)",
            "value": 2079.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 18759.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 TTFT (ms)",
            "value": 2276.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 TPOT (ms)",
            "value": 57.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.51,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 Acceptance Rate (%)",
            "value": 50.26,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 throughput (tok/s)",
            "value": 953.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 8596.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 TTFT (ms)",
            "value": 597.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 TPOT (ms)",
            "value": 15.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.47,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 Acceptance Rate (%)",
            "value": 49.08,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 throughput (tok/s)",
            "value": 2339.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 Total Tput (tok/s)",
            "value": 21031.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 TTFT (ms)",
            "value": 4142.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 TPOT (ms)",
            "value": 103.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.55,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 Acceptance Rate (%)",
            "value": 51.59,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 throughput (tok/s)",
            "value": 1235.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 11035.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 TTFT (ms)",
            "value": 915.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 TPOT (ms)",
            "value": 24.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.46,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 Acceptance Rate (%)",
            "value": 48.59,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 throughput (tok/s)",
            "value": 412.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 3701.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 TTFT (ms)",
            "value": 419.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 TPOT (ms)",
            "value": 9.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.53,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 Acceptance Rate (%)",
            "value": 51.07,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 throughput (tok/s)",
            "value": 1637.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 14744.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 TTFT (ms)",
            "value": 1439.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 TPOT (ms)",
            "value": 36.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.52,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 Acceptance Rate (%)",
            "value": 50.51,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=8 throughput (tok/s)",
            "value": 617.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 5490.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=8 TTFT (ms)",
            "value": 474.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=8 TPOT (ms)",
            "value": 12.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.58,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=8 Acceptance Rate (%)",
            "value": 52.56,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 throughput (tok/s)",
            "value": 2560.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5127.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 TTFT (ms)",
            "value": 475.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 TPOT (ms)",
            "value": 47.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 throughput (tok/s)",
            "value": 4100.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8196.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 TTFT (ms)",
            "value": 670.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 TPOT (ms)",
            "value": 59.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 throughput (tok/s)",
            "value": 1713.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3428.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 TTFT (ms)",
            "value": 321.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 TPOT (ms)",
            "value": 35.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 throughput (tok/s)",
            "value": 1617.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 Total Tput (tok/s)",
            "value": 14599.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 TTFT (ms)",
            "value": 2075.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 TPOT (ms)",
            "value": 75.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 throughput (tok/s)",
            "value": 2265.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 Total Tput (tok/s)",
            "value": 20382.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 TTFT (ms)",
            "value": 3767.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 TPOT (ms)",
            "value": 107.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 throughput (tok/s)",
            "value": 1223.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 Total Tput (tok/s)",
            "value": 11024.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 TTFT (ms)",
            "value": 1868.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 TPOT (ms)",
            "value": 48.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2474.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 4954.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 434.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 50.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 throughput (tok/s)",
            "value": 666.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1340.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 TTFT (ms)",
            "value": 170.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 TPOT (ms)",
            "value": 23.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 throughput (tok/s)",
            "value": 3534.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 Total Tput (tok/s)",
            "value": 7065.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 TTFT (ms)",
            "value": 619.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 TPOT (ms)",
            "value": 70.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 throughput (tok/s)",
            "value": 1080.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2157.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 TTFT (ms)",
            "value": 253.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 TPOT (ms)",
            "value": 28.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 throughput (tok/s)",
            "value": 251.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 Total Tput (tok/s)",
            "value": 505.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 TTFT (ms)",
            "value": 114.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 TPOT (ms)",
            "value": 15.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 throughput (tok/s)",
            "value": 1614.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3228.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 TTFT (ms)",
            "value": 266.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 TPOT (ms)",
            "value": 38.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 throughput (tok/s)",
            "value": 429.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 Total Tput (tok/s)",
            "value": 854.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 TTFT (ms)",
            "value": 133.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 TPOT (ms)",
            "value": 18.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 throughput (tok/s)",
            "value": 1289.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 Total Tput (tok/s)",
            "value": 11639.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 TTFT (ms)",
            "value": 10704.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 TPOT (ms)",
            "value": 85.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 throughput (tok/s)",
            "value": 557.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 Total Tput (tok/s)",
            "value": 5035.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 TTFT (ms)",
            "value": 710.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 TPOT (ms)",
            "value": 27.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 throughput (tok/s)",
            "value": 1306.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 Total Tput (tok/s)",
            "value": 11756.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 TTFT (ms)",
            "value": 95907.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 TPOT (ms)",
            "value": 85.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 throughput (tok/s)",
            "value": 826.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 Total Tput (tok/s)",
            "value": 7392.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 TTFT (ms)",
            "value": 1017.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 TPOT (ms)",
            "value": 36.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 throughput (tok/s)",
            "value": 227.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2042.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 TTFT (ms)",
            "value": 469.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 TPOT (ms)",
            "value": 16.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 throughput (tok/s)",
            "value": 1108.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 Total Tput (tok/s)",
            "value": 9995.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 TTFT (ms)",
            "value": 1480.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 TPOT (ms)",
            "value": 55.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 throughput (tok/s)",
            "value": 377.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3355.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 TTFT (ms)",
            "value": 505.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 TPOT (ms)",
            "value": 20.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 throughput (tok/s)",
            "value": 3730.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 7467.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 TTFT (ms)",
            "value": 474.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 TPOT (ms)",
            "value": 32.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 Acceptance Rate (%)",
            "value": 65.93,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 throughput (tok/s)",
            "value": 1182.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2375.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 TTFT (ms)",
            "value": 217.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 TPOT (ms)",
            "value": 12.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.91,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 Acceptance Rate (%)",
            "value": 63.54,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 throughput (tok/s)",
            "value": 4992.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 9980.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 TTFT (ms)",
            "value": 750.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 TPOT (ms)",
            "value": 49.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 3.02,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 Acceptance Rate (%)",
            "value": 67.24,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 throughput (tok/s)",
            "value": 1782.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3556.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 TTFT (ms)",
            "value": 261.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 TPOT (ms)",
            "value": 17.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.96,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 Acceptance Rate (%)",
            "value": 65.43,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 throughput (tok/s)",
            "value": 526.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1058.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 TTFT (ms)",
            "value": 151.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 TPOT (ms)",
            "value": 7.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 Accept Length (tok/fwd)",
            "value": 3.01,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 Acceptance Rate (%)",
            "value": 67.02,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 throughput (tok/s)",
            "value": 2599.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5200.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 TTFT (ms)",
            "value": 344.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 TPOT (ms)",
            "value": 23.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 Acceptance Rate (%)",
            "value": 65.73,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 throughput (tok/s)",
            "value": 794.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1583.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 TTFT (ms)",
            "value": 176.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 TPOT (ms)",
            "value": 9.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 Accept Length (tok/fwd)",
            "value": 3.01,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 Acceptance Rate (%)",
            "value": 66.96,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 throughput (tok/s)",
            "value": 1589.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 14341.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 TTFT (ms)",
            "value": 14737.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 TPOT (ms)",
            "value": 63.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 Acceptance Rate (%)",
            "value": 64.75,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 throughput (tok/s)",
            "value": 847.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 7639.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 TTFT (ms)",
            "value": 701.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 TPOT (ms)",
            "value": 17.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 Acceptance Rate (%)",
            "value": 65.57,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 throughput (tok/s)",
            "value": 1583.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 Total Tput (tok/s)",
            "value": 14233.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 TTFT (ms)",
            "value": 85155.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 TPOT (ms)",
            "value": 65.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 Acceptance Rate (%)",
            "value": 65.78,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 throughput (tok/s)",
            "value": 1210.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 10815.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 TTFT (ms)",
            "value": 1010.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 TPOT (ms)",
            "value": 24.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 Acceptance Rate (%)",
            "value": 65.52,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 throughput (tok/s)",
            "value": 414.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 3717.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 TTFT (ms)",
            "value": 458.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 TPOT (ms)",
            "value": 8.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 Accept Length (tok/fwd)",
            "value": 3.04,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 Acceptance Rate (%)",
            "value": 68.11,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 throughput (tok/s)",
            "value": 1515.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 13644.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 TTFT (ms)",
            "value": 1624.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 TPOT (ms)",
            "value": 39.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.95,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 Acceptance Rate (%)",
            "value": 64.96,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 throughput (tok/s)",
            "value": 586.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 5215.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 TTFT (ms)",
            "value": 622.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 TPOT (ms)",
            "value": 12.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 Acceptance Rate (%)",
            "value": 64.36,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 throughput (tok/s)",
            "value": 3221.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6450.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 TTFT (ms)",
            "value": 354.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 TPOT (ms)",
            "value": 38.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 throughput (tok/s)",
            "value": 811.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1632.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 TTFT (ms)",
            "value": 145.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 TPOT (ms)",
            "value": 19.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 throughput (tok/s)",
            "value": 4376.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8747.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 TTFT (ms)",
            "value": 547.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 TPOT (ms)",
            "value": 56.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 throughput (tok/s)",
            "value": 1245.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2486.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 TTFT (ms)",
            "value": 224.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 TPOT (ms)",
            "value": 24.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 throughput (tok/s)",
            "value": 288.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 579.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 TTFT (ms)",
            "value": 116.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 TPOT (ms)",
            "value": 13.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 throughput (tok/s)",
            "value": 2001.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4004.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 TTFT (ms)",
            "value": 245.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 TPOT (ms)",
            "value": 30.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 throughput (tok/s)",
            "value": 502.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1001.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 TTFT (ms)",
            "value": 123.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 TPOT (ms)",
            "value": 15.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 throughput (tok/s)",
            "value": 1773,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 16006.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 TTFT (ms)",
            "value": 2366.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 TPOT (ms)",
            "value": 68.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 throughput (tok/s)",
            "value": 676.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 6106.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 TTFT (ms)",
            "value": 590.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 TPOT (ms)",
            "value": 22.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 throughput (tok/s)",
            "value": 2113.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 19017.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 TTFT (ms)",
            "value": 4178.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 TPOT (ms)",
            "value": 115.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 throughput (tok/s)",
            "value": 942.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 8427.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 TTFT (ms)",
            "value": 803.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 TPOT (ms)",
            "value": 32.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 throughput (tok/s)",
            "value": 256.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2305.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 TTFT (ms)",
            "value": 375.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 TPOT (ms)",
            "value": 14.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 throughput (tok/s)",
            "value": 1314.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 11844.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 TTFT (ms)",
            "value": 1362.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 TPOT (ms)",
            "value": 46.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 throughput (tok/s)",
            "value": 432.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3845.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 TTFT (ms)",
            "value": 457.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 TPOT (ms)",
            "value": 17.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 throughput (tok/s)",
            "value": 4321.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8651.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 TTFT (ms)",
            "value": 419.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 TPOT (ms)",
            "value": 28.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.9,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 Acceptance Rate (%)",
            "value": 63.49,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 throughput (tok/s)",
            "value": 1383.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2777.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 TTFT (ms)",
            "value": 196.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 TPOT (ms)",
            "value": 11.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.89,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 Acceptance Rate (%)",
            "value": 62.93,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 throughput (tok/s)",
            "value": 5667.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 11328.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 TTFT (ms)",
            "value": 642.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 TPOT (ms)",
            "value": 43.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 Acceptance Rate (%)",
            "value": 64.21,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 throughput (tok/s)",
            "value": 2196.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 4382.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 TTFT (ms)",
            "value": 389.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 TPOT (ms)",
            "value": 13.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.89,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 Acceptance Rate (%)",
            "value": 63.04,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 throughput (tok/s)",
            "value": 563.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1133.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 TTFT (ms)",
            "value": 145.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 TPOT (ms)",
            "value": 6.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.85,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 Acceptance Rate (%)",
            "value": 61.82,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 throughput (tok/s)",
            "value": 3092.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 6185.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 TTFT (ms)",
            "value": 307.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 TPOT (ms)",
            "value": 19.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.91,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 Acceptance Rate (%)",
            "value": 63.67,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 throughput (tok/s)",
            "value": 871.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1737.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 TTFT (ms)",
            "value": 150.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 TPOT (ms)",
            "value": 8.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.88,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 Acceptance Rate (%)",
            "value": 62.52,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 throughput (tok/s)",
            "value": 2092.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 18874.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 TTFT (ms)",
            "value": 2458.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 TPOT (ms)",
            "value": 57.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.85,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 Acceptance Rate (%)",
            "value": 61.64,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 throughput (tok/s)",
            "value": 999.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 9015.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 TTFT (ms)",
            "value": 660.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 TPOT (ms)",
            "value": 14.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.88,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 Acceptance Rate (%)",
            "value": 62.81,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 throughput (tok/s)",
            "value": 2314.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 Total Tput (tok/s)",
            "value": 20798.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 TTFT (ms)",
            "value": 4385.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 TPOT (ms)",
            "value": 104.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.88,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 Acceptance Rate (%)",
            "value": 62.82,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 throughput (tok/s)",
            "value": 1464.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 13081.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 TTFT (ms)",
            "value": 911.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 TPOT (ms)",
            "value": 20.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.89,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 Acceptance Rate (%)",
            "value": 62.84,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 throughput (tok/s)",
            "value": 472.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 4239.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 TTFT (ms)",
            "value": 444.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 TPOT (ms)",
            "value": 7.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.81,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 Acceptance Rate (%)",
            "value": 60.43,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 throughput (tok/s)",
            "value": 1793.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 16147.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 TTFT (ms)",
            "value": 1424.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 TPOT (ms)",
            "value": 33.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.89,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 Acceptance Rate (%)",
            "value": 62.87,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 throughput (tok/s)",
            "value": 686.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 6099.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 TTFT (ms)",
            "value": 566.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 TPOT (ms)",
            "value": 10.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.87,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 Acceptance Rate (%)",
            "value": 62.48,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 throughput (tok/s)",
            "value": 4768.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 9547.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 TTFT (ms)",
            "value": 408.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 TPOT (ms)",
            "value": 25.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 throughput (tok/s)",
            "value": 996.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2004.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 TTFT (ms)",
            "value": 138.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 TPOT (ms)",
            "value": 15.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 throughput (tok/s)",
            "value": 6160.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 12315.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 TTFT (ms)",
            "value": 663.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 TPOT (ms)",
            "value": 39.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 throughput (tok/s)",
            "value": 1790.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3575.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 TTFT (ms)",
            "value": 206.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 TPOT (ms)",
            "value": 17.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 throughput (tok/s)",
            "value": 261.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 526.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 TTFT (ms)",
            "value": 104,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 TPOT (ms)",
            "value": 14.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 throughput (tok/s)",
            "value": 2919.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5839.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 TTFT (ms)",
            "value": 299.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 TPOT (ms)",
            "value": 20.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 throughput (tok/s)",
            "value": 520.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1037.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 TTFT (ms)",
            "value": 111.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 TPOT (ms)",
            "value": 14.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 throughput (tok/s)",
            "value": 1477.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 13337.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 TTFT (ms)",
            "value": 2584.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 TPOT (ms)",
            "value": 82.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 throughput (tok/s)",
            "value": 654.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 5910.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 TTFT (ms)",
            "value": 701.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 TPOT (ms)",
            "value": 22.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 throughput (tok/s)",
            "value": 1429.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 12861.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 TTFT (ms)",
            "value": 55852.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 TPOT (ms)",
            "value": 114.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 throughput (tok/s)",
            "value": 979.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 8759.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 TTFT (ms)",
            "value": 955.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 TPOT (ms)",
            "value": 30.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 throughput (tok/s)",
            "value": 212.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 1911.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 TTFT (ms)",
            "value": 451.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 TPOT (ms)",
            "value": 17.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 throughput (tok/s)",
            "value": 1253.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 11302.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 TTFT (ms)",
            "value": 1540.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 TPOT (ms)",
            "value": 48.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 throughput (tok/s)",
            "value": 394.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3506.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 TTFT (ms)",
            "value": 536.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 TPOT (ms)",
            "value": 19.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31197825471 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 _tp",
            "value": 1,
            "unit": ""
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
        "date": 1786161721794,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 throughput (tok/s)",
            "value": 4472.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8954.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 TTFT (ms)",
            "value": 664.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 TPOT (ms)",
            "value": 27.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 throughput (tok/s)",
            "value": 1511.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 3038.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 TTFT (ms)",
            "value": 216.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 TPOT (ms)",
            "value": 10.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 throughput (tok/s)",
            "value": 6013.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 12022,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 TTFT (ms)",
            "value": 1090.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 TPOT (ms)",
            "value": 40.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 throughput (tok/s)",
            "value": 2329.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 4651.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 TTFT (ms)",
            "value": 323.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 TPOT (ms)",
            "value": 13.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 throughput (tok/s)",
            "value": 570.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1146.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 TTFT (ms)",
            "value": 203.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 TPOT (ms)",
            "value": 6.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 throughput (tok/s)",
            "value": 7183.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 Total Tput (tok/s)",
            "value": 14360.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 TTFT (ms)",
            "value": 1910.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 TPOT (ms)",
            "value": 68.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 throughput (tok/s)",
            "value": 3131.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 6264.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 TTFT (ms)",
            "value": 446.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 TPOT (ms)",
            "value": 19.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 throughput (tok/s)",
            "value": 917.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1827.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 TTFT (ms)",
            "value": 173.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 TPOT (ms)",
            "value": 8.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 throughput (tok/s)",
            "value": 5520.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 Total Tput (tok/s)",
            "value": 6211.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 TTFT (ms)",
            "value": 566.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 TPOT (ms)",
            "value": 22.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 throughput (tok/s)",
            "value": 1574.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1772.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 TTFT (ms)",
            "value": 207.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 TPOT (ms)",
            "value": 9.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 throughput (tok/s)",
            "value": 7572.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 Total Tput (tok/s)",
            "value": 8518.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 TTFT (ms)",
            "value": 899.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 TPOT (ms)",
            "value": 32.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 throughput (tok/s)",
            "value": 2754.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 Total Tput (tok/s)",
            "value": 3096.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 TTFT (ms)",
            "value": 268.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 TPOT (ms)",
            "value": 11.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 throughput (tok/s)",
            "value": 655.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 Total Tput (tok/s)",
            "value": 739.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 TTFT (ms)",
            "value": 145.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 TPOT (ms)",
            "value": 6.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 throughput (tok/s)",
            "value": 7183.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 Total Tput (tok/s)",
            "value": 8082.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 TTFT (ms)",
            "value": 44711.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 TPOT (ms)",
            "value": 63.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 throughput (tok/s)",
            "value": 4236.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 Total Tput (tok/s)",
            "value": 4765.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 TTFT (ms)",
            "value": 362.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 TPOT (ms)",
            "value": 14.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 throughput (tok/s)",
            "value": 977.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1100.05,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 TTFT (ms)",
            "value": 174.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 TPOT (ms)",
            "value": 8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 throughput (tok/s)",
            "value": 1826.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 16486.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 TTFT (ms)",
            "value": 3656.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 TPOT (ms)",
            "value": 65.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 throughput (tok/s)",
            "value": 996.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 8989.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 TTFT (ms)",
            "value": 837.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 TPOT (ms)",
            "value": 14.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 throughput (tok/s)",
            "value": 1965.05,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 17680.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 TTFT (ms)",
            "value": 11061.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 TPOT (ms)",
            "value": 116.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 throughput (tok/s)",
            "value": 1309.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 11713.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 TTFT (ms)",
            "value": 1394.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 TPOT (ms)",
            "value": 22.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 throughput (tok/s)",
            "value": 499.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 4489.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 TTFT (ms)",
            "value": 562.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 TPOT (ms)",
            "value": 7.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 throughput (tok/s)",
            "value": 1973.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 Total Tput (tok/s)",
            "value": 17761.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 TTFT (ms)",
            "value": 124056.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 TPOT (ms)",
            "value": 119.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 throughput (tok/s)",
            "value": 1610.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 14512.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 TTFT (ms)",
            "value": 2068.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 TPOT (ms)",
            "value": 37.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 throughput (tok/s)",
            "value": 697.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 6207.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 TTFT (ms)",
            "value": 657.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 TPOT (ms)",
            "value": 10.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 throughput (tok/s)",
            "value": 3019.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6045.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 TTFT (ms)",
            "value": 453.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 TPOT (ms)",
            "value": 40.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 throughput (tok/s)",
            "value": 802.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1613.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 TTFT (ms)",
            "value": 176.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 TPOT (ms)",
            "value": 19.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 throughput (tok/s)",
            "value": 4248.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8493.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 TTFT (ms)",
            "value": 760.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 TPOT (ms)",
            "value": 58.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 throughput (tok/s)",
            "value": 1197.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2391.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 TTFT (ms)",
            "value": 222.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 TPOT (ms)",
            "value": 25.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 throughput (tok/s)",
            "value": 292.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 588.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 TTFT (ms)",
            "value": 125.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 TPOT (ms)",
            "value": 13.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 throughput (tok/s)",
            "value": 5484.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 Total Tput (tok/s)",
            "value": 10964.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 TTFT (ms)",
            "value": 1367.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 TPOT (ms)",
            "value": 90.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 throughput (tok/s)",
            "value": 1978.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3957.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 TTFT (ms)",
            "value": 304.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 TPOT (ms)",
            "value": 31.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 throughput (tok/s)",
            "value": 504.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1004.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 TTFT (ms)",
            "value": 153.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 TPOT (ms)",
            "value": 15.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 throughput (tok/s)",
            "value": 3436.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 Total Tput (tok/s)",
            "value": 3866.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 TTFT (ms)",
            "value": 426.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 TPOT (ms)",
            "value": 36.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 throughput (tok/s)",
            "value": 831.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 Total Tput (tok/s)",
            "value": 936.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 TTFT (ms)",
            "value": 173.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 TPOT (ms)",
            "value": 18.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 throughput (tok/s)",
            "value": 5158.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 Total Tput (tok/s)",
            "value": 5802.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 TTFT (ms)",
            "value": 707.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 TPOT (ms)",
            "value": 48.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 throughput (tok/s)",
            "value": 1239.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1393.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 TTFT (ms)",
            "value": 214.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 TPOT (ms)",
            "value": 25.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 throughput (tok/s)",
            "value": 300.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 Total Tput (tok/s)",
            "value": 339.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 TTFT (ms)",
            "value": 126.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 TPOT (ms)",
            "value": 13.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 throughput (tok/s)",
            "value": 5728.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 Total Tput (tok/s)",
            "value": 6444.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 TTFT (ms)",
            "value": 31376.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 TPOT (ms)",
            "value": 82.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 throughput (tok/s)",
            "value": 2163.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 Total Tput (tok/s)",
            "value": 2434.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 TTFT (ms)",
            "value": 284.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 TPOT (ms)",
            "value": 28.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 throughput (tok/s)",
            "value": 517.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 Total Tput (tok/s)",
            "value": 581.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 TTFT (ms)",
            "value": 140.41,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 TPOT (ms)",
            "value": 15.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 1024/8192 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 throughput (tok/s)",
            "value": 1499.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 13538.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 TTFT (ms)",
            "value": 2929.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 TPOT (ms)",
            "value": 81.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 throughput (tok/s)",
            "value": 615.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 5556.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 TTFT (ms)",
            "value": 771.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 TPOT (ms)",
            "value": 24.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 throughput (tok/s)",
            "value": 1790.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 16111.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 TTFT (ms)",
            "value": 5368.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 TPOT (ms)",
            "value": 135.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 throughput (tok/s)",
            "value": 842.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 7530.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 TTFT (ms)",
            "value": 1065.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 TPOT (ms)",
            "value": 36.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 throughput (tok/s)",
            "value": 267.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2401.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 TTFT (ms)",
            "value": 491.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 TPOT (ms)",
            "value": 14.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 throughput (tok/s)",
            "value": 1672.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 Total Tput (tok/s)",
            "value": 15050.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 TTFT (ms)",
            "value": 120024.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 TPOT (ms)",
            "value": 169.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 throughput (tok/s)",
            "value": 1177.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 10616.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 TTFT (ms)",
            "value": 1691.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 TPOT (ms)",
            "value": 51.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 throughput (tok/s)",
            "value": 424.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3774.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 TTFT (ms)",
            "value": 601.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 TPOT (ms)",
            "value": 17.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8-PTPC-tp4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=128 throughput (tok/s)",
            "value": 3057.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6121.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=128 TTFT (ms)",
            "value": 481.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=128 TPOT (ms)",
            "value": 40.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=16 throughput (tok/s)",
            "value": 786.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1580.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=16 TTFT (ms)",
            "value": 684.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=16 TPOT (ms)",
            "value": 19.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=256 throughput (tok/s)",
            "value": 4636.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=256 Total Tput (tok/s)",
            "value": 9268.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=256 TTFT (ms)",
            "value": 715.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=256 TPOT (ms)",
            "value": 53.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=32 throughput (tok/s)",
            "value": 1342.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2680.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=32 TTFT (ms)",
            "value": 284.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=32 TPOT (ms)",
            "value": 22.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=4 throughput (tok/s)",
            "value": 238.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=4 Total Tput (tok/s)",
            "value": 479.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=4 TTFT (ms)",
            "value": 211.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=4 TPOT (ms)",
            "value": 16.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=512 throughput (tok/s)",
            "value": 6619.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=512 Total Tput (tok/s)",
            "value": 13233.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=512 TTFT (ms)",
            "value": 1228.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=512 TPOT (ms)",
            "value": 74.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=64 throughput (tok/s)",
            "value": 2089.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4179.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=64 TTFT (ms)",
            "value": 326.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=64 TPOT (ms)",
            "value": 29.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=8 throughput (tok/s)",
            "value": 450.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=8 Total Tput (tok/s)",
            "value": 898.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=8 TTFT (ms)",
            "value": 300.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=8 TPOT (ms)",
            "value": 17.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 1024/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=128 throughput (tok/s)",
            "value": 1971.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=128 Total Tput (tok/s)",
            "value": 17797.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=128 TTFT (ms)",
            "value": 2315.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=128 TPOT (ms)",
            "value": 61.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=16 throughput (tok/s)",
            "value": 695.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=16 Total Tput (tok/s)",
            "value": 6279.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=16 TTFT (ms)",
            "value": 576.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=16 TPOT (ms)",
            "value": 21.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=256 throughput (tok/s)",
            "value": 2422.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=256 Total Tput (tok/s)",
            "value": 21799.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=256 TTFT (ms)",
            "value": 4224.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=256 TPOT (ms)",
            "value": 99.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=32 throughput (tok/s)",
            "value": 1078.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=32 Total Tput (tok/s)",
            "value": 9648.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=32 TTFT (ms)",
            "value": 822.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=32 TPOT (ms)",
            "value": 27.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=4 throughput (tok/s)",
            "value": 227.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2045.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=4 TTFT (ms)",
            "value": 384.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=4 TPOT (ms)",
            "value": 16.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=512 throughput (tok/s)",
            "value": 2690.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=512 Total Tput (tok/s)",
            "value": 24218.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=512 TTFT (ms)",
            "value": 8670.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=512 TPOT (ms)",
            "value": 179.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=64 throughput (tok/s)",
            "value": 1516.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=64 Total Tput (tok/s)",
            "value": 13673.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=64 TTFT (ms)",
            "value": 1327.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=64 TPOT (ms)",
            "value": 39.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=8 throughput (tok/s)",
            "value": 419.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3729.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=8 TTFT (ms)",
            "value": 467.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=8 TPOT (ms)",
            "value": 18.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro 8192/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=128 throughput (tok/s)",
            "value": 5537,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 11086.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=128 TTFT (ms)",
            "value": 496.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=128 TPOT (ms)",
            "value": 21.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=16 throughput (tok/s)",
            "value": 1618.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 3254.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=16 TTFT (ms)",
            "value": 212.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=16 TPOT (ms)",
            "value": 9.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=256 throughput (tok/s)",
            "value": 6573.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 13141.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=256 TTFT (ms)",
            "value": 837.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=256 TPOT (ms)",
            "value": 36.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=32 throughput (tok/s)",
            "value": 2602.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 5197.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=32 TTFT (ms)",
            "value": 250.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=32 TPOT (ms)",
            "value": 11.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=4 throughput (tok/s)",
            "value": 668.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1343.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=4 TTFT (ms)",
            "value": 157.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=4 TPOT (ms)",
            "value": 5.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=64 throughput (tok/s)",
            "value": 3738.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 7478.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=64 TTFT (ms)",
            "value": 338.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=64 TPOT (ms)",
            "value": 15.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=8 throughput (tok/s)",
            "value": 1036.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 2065.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=8 TTFT (ms)",
            "value": 180.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=8 TPOT (ms)",
            "value": 7.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=128 throughput (tok/s)",
            "value": 7437.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=128 Total Tput (tok/s)",
            "value": 8368.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=128 TTFT (ms)",
            "value": 437.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=128 TPOT (ms)",
            "value": 16.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=16 throughput (tok/s)",
            "value": 1886.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2122.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=16 TTFT (ms)",
            "value": 204.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=16 TPOT (ms)",
            "value": 8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=256 throughput (tok/s)",
            "value": 8668.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=256 Total Tput (tok/s)",
            "value": 9750.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=256 TTFT (ms)",
            "value": 757.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=256 TPOT (ms)",
            "value": 27.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=32 throughput (tok/s)",
            "value": 3203.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=32 Total Tput (tok/s)",
            "value": 3601.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=32 TTFT (ms)",
            "value": 237.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=32 TPOT (ms)",
            "value": 9.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=4 throughput (tok/s)",
            "value": 757.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=4 Total Tput (tok/s)",
            "value": 854.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=4 TTFT (ms)",
            "value": 158.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=4 TPOT (ms)",
            "value": 5.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=512 throughput (tok/s)",
            "value": 10505.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=512 Total Tput (tok/s)",
            "value": 11819.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=512 TTFT (ms)",
            "value": 4283.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=512 TPOT (ms)",
            "value": 45.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=64 throughput (tok/s)",
            "value": 4656.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=64 Total Tput (tok/s)",
            "value": 5237.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=64 TTFT (ms)",
            "value": 313.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=64 TPOT (ms)",
            "value": 12.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=8 throughput (tok/s)",
            "value": 1164.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1310.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=8 TTFT (ms)",
            "value": 174.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=8 TPOT (ms)",
            "value": 6.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 1024/8192 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=128 throughput (tok/s)",
            "value": 2230.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 20133.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=128 TTFT (ms)",
            "value": 2703.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=128 TPOT (ms)",
            "value": 53.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=16 throughput (tok/s)",
            "value": 1168.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 10548.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=16 TTFT (ms)",
            "value": 653.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=16 TPOT (ms)",
            "value": 12.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=256 throughput (tok/s)",
            "value": 2495.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 22456.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=256 TTFT (ms)",
            "value": 4985.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=256 TPOT (ms)",
            "value": 96.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=32 throughput (tok/s)",
            "value": 1524.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 13633.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=32 TTFT (ms)",
            "value": 967.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=32 TPOT (ms)",
            "value": 19.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=4 throughput (tok/s)",
            "value": 558.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 5021.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=4 TTFT (ms)",
            "value": 417.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=4 TPOT (ms)",
            "value": 6.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=512 throughput (tok/s)",
            "value": 2557.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=512 Total Tput (tok/s)",
            "value": 23018.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=512 TTFT (ms)",
            "value": 60251.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=512 TPOT (ms)",
            "value": 131.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=64 throughput (tok/s)",
            "value": 1917.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 17280.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=64 TTFT (ms)",
            "value": 1568.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=64 TPOT (ms)",
            "value": 31.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=8 throughput (tok/s)",
            "value": 806.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 7178.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=8 TTFT (ms)",
            "value": 506.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=8 TPOT (ms)",
            "value": 9.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-MTP-tp4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=128 throughput (tok/s)",
            "value": 3281.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6569.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=128 TTFT (ms)",
            "value": 411.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=128 TPOT (ms)",
            "value": 37.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=16 throughput (tok/s)",
            "value": 884.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1779.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=16 TTFT (ms)",
            "value": 170.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=16 TPOT (ms)",
            "value": 17.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=256 throughput (tok/s)",
            "value": 4911.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 9818.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=256 TTFT (ms)",
            "value": 636.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=256 TPOT (ms)",
            "value": 50.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=32 throughput (tok/s)",
            "value": 1301.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2598.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=32 TTFT (ms)",
            "value": 209.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=32 TPOT (ms)",
            "value": 23.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=4 throughput (tok/s)",
            "value": 310.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 623.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=4 TTFT (ms)",
            "value": 127.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=4 TPOT (ms)",
            "value": 12.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=512 throughput (tok/s)",
            "value": 6649.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=512 Total Tput (tok/s)",
            "value": 13292.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=512 TTFT (ms)",
            "value": 1125.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=512 TPOT (ms)",
            "value": 74.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=64 throughput (tok/s)",
            "value": 2074.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4150.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=64 TTFT (ms)",
            "value": 277.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=64 TPOT (ms)",
            "value": 29.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=8 throughput (tok/s)",
            "value": 526.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1049.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=8 TTFT (ms)",
            "value": 144.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=8 TPOT (ms)",
            "value": 14.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=128 throughput (tok/s)",
            "value": 3812.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=128 Total Tput (tok/s)",
            "value": 4289.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=128 TTFT (ms)",
            "value": 373.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=128 TPOT (ms)",
            "value": 32.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=16 throughput (tok/s)",
            "value": 913.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1027.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=16 TTFT (ms)",
            "value": 169.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=16 TPOT (ms)",
            "value": 17.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=256 throughput (tok/s)",
            "value": 5924,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=256 Total Tput (tok/s)",
            "value": 6663.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=256 TTFT (ms)",
            "value": 592.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=256 TPOT (ms)",
            "value": 41.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=32 throughput (tok/s)",
            "value": 1388.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1560.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=32 TTFT (ms)",
            "value": 209.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=32 TPOT (ms)",
            "value": 22.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=4 throughput (tok/s)",
            "value": 319.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=4 Total Tput (tok/s)",
            "value": 360.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=4 TTFT (ms)",
            "value": 134.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=4 TPOT (ms)",
            "value": 12.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=512 throughput (tok/s)",
            "value": 7099.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=512 Total Tput (tok/s)",
            "value": 7987.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=512 TTFT (ms)",
            "value": 20679.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=512 TPOT (ms)",
            "value": 66.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=64 throughput (tok/s)",
            "value": 2304.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=64 Total Tput (tok/s)",
            "value": 2592.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=64 TTFT (ms)",
            "value": 264.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=64 TPOT (ms)",
            "value": 26.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=8 throughput (tok/s)",
            "value": 534.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=8 Total Tput (tok/s)",
            "value": 601.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=8 TTFT (ms)",
            "value": 147.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=8 TPOT (ms)",
            "value": 14.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 1024/8192 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=128 throughput (tok/s)",
            "value": 1738.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 15696.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=128 TTFT (ms)",
            "value": 2458.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=128 TPOT (ms)",
            "value": 69.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=16 throughput (tok/s)",
            "value": 697.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 6295.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=16 TTFT (ms)",
            "value": 621.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=16 TPOT (ms)",
            "value": 21.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=256 throughput (tok/s)",
            "value": 2105.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 18941.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=256 TTFT (ms)",
            "value": 4551.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=256 TPOT (ms)",
            "value": 115.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=32 throughput (tok/s)",
            "value": 955.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 8542.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=32 TTFT (ms)",
            "value": 878.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=32 TPOT (ms)",
            "value": 31.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=4 throughput (tok/s)",
            "value": 263.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2366.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=4 TTFT (ms)",
            "value": 397.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=4 TPOT (ms)",
            "value": 14.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=512 throughput (tok/s)",
            "value": 2031.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=512 Total Tput (tok/s)",
            "value": 18280.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=512 TTFT (ms)",
            "value": 94146.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=512 TPOT (ms)",
            "value": 145.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=64 throughput (tok/s)",
            "value": 1334.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 12027.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=64 TTFT (ms)",
            "value": 1415.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=64 TPOT (ms)",
            "value": 45.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=8 throughput (tok/s)",
            "value": 458.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 4076.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=8 TTFT (ms)",
            "value": 494.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=8 TPOT (ms)",
            "value": 16.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4-tp4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=128 throughput (tok/s)",
            "value": 3741.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 7490.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=128 TTFT (ms)",
            "value": 286.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=128 TPOT (ms)",
            "value": 33.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=16 throughput (tok/s)",
            "value": 1141.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2296.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=16 TTFT (ms)",
            "value": 121.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=16 TPOT (ms)",
            "value": 13.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=256 throughput (tok/s)",
            "value": 5468.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 10931.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=256 TTFT (ms)",
            "value": 454.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=256 TPOT (ms)",
            "value": 45.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=32 throughput (tok/s)",
            "value": 1701.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3397.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=32 TTFT (ms)",
            "value": 144.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=32 TPOT (ms)",
            "value": 18.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=4 throughput (tok/s)",
            "value": 378.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 760.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=4 TTFT (ms)",
            "value": 100.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=4 TPOT (ms)",
            "value": 10.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=512 throughput (tok/s)",
            "value": 7658.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=512 Total Tput (tok/s)",
            "value": 15309.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=512 TTFT (ms)",
            "value": 770.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=512 TPOT (ms)",
            "value": 64.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=64 throughput (tok/s)",
            "value": 2530.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5061.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=64 TTFT (ms)",
            "value": 199.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=64 TPOT (ms)",
            "value": 24.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=8 throughput (tok/s)",
            "value": 720.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1435.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=8 TTFT (ms)",
            "value": 104.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=8 TPOT (ms)",
            "value": 10.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=128 throughput (tok/s)",
            "value": 3930.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=128 Total Tput (tok/s)",
            "value": 4422.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=128 TTFT (ms)",
            "value": 260.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=128 TPOT (ms)",
            "value": 31.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=16 throughput (tok/s)",
            "value": 1153.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1297.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=16 TTFT (ms)",
            "value": 115.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=16 TPOT (ms)",
            "value": 13.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=256 throughput (tok/s)",
            "value": 5639.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=256 Total Tput (tok/s)",
            "value": 6343.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=256 TTFT (ms)",
            "value": 438,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=256 TPOT (ms)",
            "value": 44.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=32 throughput (tok/s)",
            "value": 1777.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1997.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=32 TTFT (ms)",
            "value": 133.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=32 TPOT (ms)",
            "value": 17.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=4 throughput (tok/s)",
            "value": 400.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=4 Total Tput (tok/s)",
            "value": 451.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=4 TTFT (ms)",
            "value": 94.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=4 TPOT (ms)",
            "value": 9.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=512 throughput (tok/s)",
            "value": 7564.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=512 Total Tput (tok/s)",
            "value": 8511.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=512 TTFT (ms)",
            "value": 822.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=512 TPOT (ms)",
            "value": 66.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=64 throughput (tok/s)",
            "value": 2636.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=64 Total Tput (tok/s)",
            "value": 2965.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=64 TTFT (ms)",
            "value": 184.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=64 TPOT (ms)",
            "value": 23.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=8 throughput (tok/s)",
            "value": 712.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=8 Total Tput (tok/s)",
            "value": 801.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=8 TTFT (ms)",
            "value": 98.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=8 TPOT (ms)",
            "value": 10.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 1024/8192 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=128 throughput (tok/s)",
            "value": 2332.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 21061.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=128 TTFT (ms)",
            "value": 1390.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=128 TPOT (ms)",
            "value": 52.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=16 throughput (tok/s)",
            "value": 950.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 8579.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=16 TTFT (ms)",
            "value": 360.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=16 TPOT (ms)",
            "value": 15.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=256 throughput (tok/s)",
            "value": 2923.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 26305.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=256 TTFT (ms)",
            "value": 2548.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=256 TPOT (ms)",
            "value": 83.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=32 throughput (tok/s)",
            "value": 1339.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 11980.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=32 TTFT (ms)",
            "value": 497.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=32 TPOT (ms)",
            "value": 22.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=4 throughput (tok/s)",
            "value": 374.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 3365.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=4 TTFT (ms)",
            "value": 233.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=4 TPOT (ms)",
            "value": 10.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=512 throughput (tok/s)",
            "value": 3077.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=512 Total Tput (tok/s)",
            "value": 27697.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=512 TTFT (ms)",
            "value": 17765.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=512 TPOT (ms)",
            "value": 144.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=64 throughput (tok/s)",
            "value": 1806.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 16280.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=64 TTFT (ms)",
            "value": 802.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=64 TPOT (ms)",
            "value": 33.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=8 throughput (tok/s)",
            "value": 596.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 5308.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=8 TTFT (ms)",
            "value": 282.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=8 TPOT (ms)",
            "value": 12.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.7-MXFP4-tp4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=128 throughput (tok/s)",
            "value": 4735.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 9481.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=128 TTFT (ms)",
            "value": 278.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=128 TPOT (ms)",
            "value": 26.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=16 throughput (tok/s)",
            "value": 1307.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2629.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=16 TTFT (ms)",
            "value": 116.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=16 TPOT (ms)",
            "value": 11.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=256 throughput (tok/s)",
            "value": 4746.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 9489.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=256 TTFT (ms)",
            "value": 23607.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=256 TPOT (ms)",
            "value": 26.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=32 throughput (tok/s)",
            "value": 2087.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 4168.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=32 TTFT (ms)",
            "value": 250.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=32 TPOT (ms)",
            "value": 14.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=4 throughput (tok/s)",
            "value": 437.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 879.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=4 TTFT (ms)",
            "value": 269.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=4 TPOT (ms)",
            "value": 8.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=512 throughput (tok/s)",
            "value": 4757.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=512 Total Tput (tok/s)",
            "value": 9511.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=512 TTFT (ms)",
            "value": 70548.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=512 TPOT (ms)",
            "value": 26.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=64 throughput (tok/s)",
            "value": 3267.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 6536.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=64 TTFT (ms)",
            "value": 218.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=64 TPOT (ms)",
            "value": 18.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=8 throughput (tok/s)",
            "value": 781.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1557.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=8 TTFT (ms)",
            "value": 194.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=8 TPOT (ms)",
            "value": 9.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=128 throughput (tok/s)",
            "value": 5589,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=128 Total Tput (tok/s)",
            "value": 6288.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=128 TTFT (ms)",
            "value": 315.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=128 TPOT (ms)",
            "value": 22.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=16 throughput (tok/s)",
            "value": 1399.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1574.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=16 TTFT (ms)",
            "value": 118.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=16 TPOT (ms)",
            "value": 11.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=256 throughput (tok/s)",
            "value": 5605.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=256 Total Tput (tok/s)",
            "value": 6305.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=256 TTFT (ms)",
            "value": 158071.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=256 TPOT (ms)",
            "value": 22.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=32 throughput (tok/s)",
            "value": 2306.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=32 Total Tput (tok/s)",
            "value": 2592.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=32 TTFT (ms)",
            "value": 343.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=32 TPOT (ms)",
            "value": 13.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=4 throughput (tok/s)",
            "value": 452.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=4 Total Tput (tok/s)",
            "value": 510.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=4 TTFT (ms)",
            "value": 151.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=4 TPOT (ms)",
            "value": 8.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=64 throughput (tok/s)",
            "value": 3753.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=64 Total Tput (tok/s)",
            "value": 4221.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=64 TTFT (ms)",
            "value": 156.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=64 TPOT (ms)",
            "value": 16.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=8 throughput (tok/s)",
            "value": 804.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=8 Total Tput (tok/s)",
            "value": 905.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=8 TTFT (ms)",
            "value": 119.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=8 TPOT (ms)",
            "value": 9.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 1024/8192 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=128 throughput (tok/s)",
            "value": 3109.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 28069.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=128 TTFT (ms)",
            "value": 1220.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=128 TPOT (ms)",
            "value": 39.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=16 throughput (tok/s)",
            "value": 1145.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 10341.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=16 TTFT (ms)",
            "value": 313.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=16 TPOT (ms)",
            "value": 13.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=256 throughput (tok/s)",
            "value": 3108.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 27964.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=256 TTFT (ms)",
            "value": 36931.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=256 TPOT (ms)",
            "value": 40.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=32 throughput (tok/s)",
            "value": 1714.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 15332.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=32 TTFT (ms)",
            "value": 444.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=32 TPOT (ms)",
            "value": 17.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=4 throughput (tok/s)",
            "value": 409.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 3683.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=4 TTFT (ms)",
            "value": 261.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=4 TPOT (ms)",
            "value": 9.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=512 throughput (tok/s)",
            "value": 3145,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=512 Total Tput (tok/s)",
            "value": 28305.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=512 TTFT (ms)",
            "value": 107604.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=512 TPOT (ms)",
            "value": 40.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=64 throughput (tok/s)",
            "value": 2426.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 21873.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=64 TTFT (ms)",
            "value": 719.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=64 TPOT (ms)",
            "value": 25.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=8 throughput (tok/s)",
            "value": 711.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 6327.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=8 TTFT (ms)",
            "value": 270.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=8 TPOT (ms)",
            "value": 10.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::MiniMax-M3-MXFP4-tp4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=128 throughput (tok/s)",
            "value": 4730.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 9471.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=128 TTFT (ms)",
            "value": 323.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=128 TPOT (ms)",
            "value": 26.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=16 throughput (tok/s)",
            "value": 1405.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2826.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=16 TTFT (ms)",
            "value": 137.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=16 TPOT (ms)",
            "value": 11.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=256 throughput (tok/s)",
            "value": 6708.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 13410.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=256 TTFT (ms)",
            "value": 525.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=256 TPOT (ms)",
            "value": 36.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=32 throughput (tok/s)",
            "value": 2199.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 4391.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=32 TTFT (ms)",
            "value": 168.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=32 TPOT (ms)",
            "value": 14.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=4 throughput (tok/s)",
            "value": 494.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 994.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=4 TTFT (ms)",
            "value": 120.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=4 TPOT (ms)",
            "value": 7.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=512 throughput (tok/s)",
            "value": 8653.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=512 Total Tput (tok/s)",
            "value": 17298.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=512 TTFT (ms)",
            "value": 946.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=512 TPOT (ms)",
            "value": 57.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=64 throughput (tok/s)",
            "value": 3249.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 6500.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=64 TTFT (ms)",
            "value": 218.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=64 TPOT (ms)",
            "value": 18.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=8 throughput (tok/s)",
            "value": 824.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1643.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=8 TTFT (ms)",
            "value": 396.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=8 TPOT (ms)",
            "value": 9.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=128 throughput (tok/s)",
            "value": 2714.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 24504.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=128 TTFT (ms)",
            "value": 2002.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=128 TPOT (ms)",
            "value": 44.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=16 throughput (tok/s)",
            "value": 1177.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 10630.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=16 TTFT (ms)",
            "value": 398.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=16 TPOT (ms)",
            "value": 12.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=256 throughput (tok/s)",
            "value": 3348.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 30128.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=256 TTFT (ms)",
            "value": 3260.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=256 TPOT (ms)",
            "value": 72.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=32 throughput (tok/s)",
            "value": 1740.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 15560.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=32 TTFT (ms)",
            "value": 579.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=32 TPOT (ms)",
            "value": 17.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=4 throughput (tok/s)",
            "value": 417.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 3755.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=4 TTFT (ms)",
            "value": 225.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=4 TPOT (ms)",
            "value": 9.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=512 throughput (tok/s)",
            "value": 3408.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=512 Total Tput (tok/s)",
            "value": 30678.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=512 TTFT (ms)",
            "value": 7493.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=512 TPOT (ms)",
            "value": 141.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=512 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=512 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=64 throughput (tok/s)",
            "value": 2257.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 20350.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=64 TTFT (ms)",
            "value": 1079.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=64 TPOT (ms)",
            "value": 26.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=8 throughput (tok/s)",
            "value": 821.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 7309.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=8 TTFT (ms)",
            "value": 293.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=8 TPOT (ms)",
            "value": 9.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31194345723 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: 7.2.4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4-tp4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
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
        "date": 1786208642472,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Same base model as DeepSeek-R1-0528 FP8 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9484 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP MTP acceptance (%)",
            "value": 67.37,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Same base model as DeepSeek-R1-0528 FP8 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9484 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP avg toks/fwd (tok/fwd)",
            "value": 3.02,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant accuracy (GSM8K)",
            "value": 0.9363,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Online quantization on top of DeepSeek-R1-0528 MTP (FP8 native): global ptpc_fp8 + expert layers mxfp4, excluding lm_head and *.gate.*. Threshold set to 0.93 (same headroom as DeepSeek-R1-0528-FP4 MTP) as a conservative placeholder for the MoE-MXFP4 accuracy drop. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9333 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant MTP acceptance (%)",
            "value": 64.19,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: Online quantization on top of DeepSeek-R1-0528 MTP (FP8 native): global ptpc_fp8 + expert layers mxfp4, excluding lm_head and *.gate.*. Threshold set to 0.93 (same headroom as DeepSeek-R1-0528-FP4 MTP) as a conservative placeholder for the MoE-MXFP4 accuracy drop. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9333 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528 MTP Online-Quant avg toks/fwd (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 accuracy (GSM8K)",
            "value": 0.9363,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.934 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP accuracy (GSM8K)",
            "value": 0.9401,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9348 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP MTP acceptance (%)",
            "value": 64.62,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.9553 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | BaselineNote: CI measured FP8 baseline (deepseek-ai/DeepSeek-R1-0528 is natively FP8) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9348 | fewshot: 3 | Model: /models/amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM::DeepSeek-R1-0528-FP4 MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro accuracy (GSM8K)",
            "value": 0.956,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Full-eval (1319 samples) 3-shot flexible-extract = 0.9522 ± 0.0059 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9568 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark accuracy (GSM8K)",
            "value": 0.9515,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark MTP acceptance (%)",
            "value": 46.25,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: DSpark spec-decode (7 tokens, dp-attention, PIECEWISE cudagraph) on the DeepSeek-V4-Pro-DSpark checkpoint. Spec-decode is lossless w.r.t. the target, so baseline reuses the DeepSeek-V4-Pro FP8 base (0.96); threshold 0.93 leaves ~3pp headroom for spec-decode / dp-attention run-to-run variance. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate — add it once measured to guard draft-head regressions. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9522 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro-DSpark"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DSpark avg toks/fwd (tok/fwd)",
            "value": 4.24,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro EPLB r0 accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.956 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: EP+DPA, EPLB pure rearrangement (num_redundant_experts=0, no extra memory), rebalance_interval=200. g64 8xMI355X measured GSM8K 5-shot flexible/strict = 0.9560/0.9568 (2026-07-20), 4 rebalances during the eval, 0 crashes. Guards the num_redundant>0 startup-OOM/migration-deadlock fixes (redundant=0 doesn't hit them, but shares the rebalance/migration code path). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9507 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro EPLB r64 biased accuracy (GSM8K)",
            "value": 0.9477,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.955 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: EP+DPA, EPLB biased placement (64 redundant physical experts = top-8 hottest fully replicated to all 8 GPUs), rebalance_interval=200. Exercises fill_redundant init + runtime rebalance/migration end-to-end, guarding the num_redundant>0 startup-OOM/migration-deadlock fixes. g64 8xMI355X measured GSM8K 5-shot flexible/strict = 0.9553/0.9560 (2026-07-20), 4 rebalances including migration during the eval, 0 crashes. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9477 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP accuracy (GSM8K)",
            "value": 0.9492,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.95 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP MTP acceptance (%)",
            "value": 64.58,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.96 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: Same base model as DeepSeek-V4-Pro FP8 (MTP-3). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.95 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP avg toks/fwd (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO+DPA conc1000 accuracy (GSM8K)",
            "value": 0.9507,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.95 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: TBO + dp-attention at conc=1000. Local 1319-sample GSM8K 3-shot, 4 runs = 0.9439/0.9484/0.9538/0.9530 (mean ~0.950, 2026-06-14, after TBO ids-gather + pad_for_all_gather fixes). Baseline 0.95; threshold 0.93 (~1.4pp below lowest 0.9439, conc=1000 variance). | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9515 | fewshot: 3 | Model: /models/deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM::GLM-5-FP8 accuracy (GSM8K)",
            "value": 0.9439,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.93 | Baseline: 0.9545 | BaselineModel: zai-org/GLM-5 | BaselineNote: HF: amd/GLM-5-MXFP4 card shows GLM-5 baseline=0.9545 (5-shot) | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9492 | fewshot: 3 | Model: /models/zai-org/GLM-5-FP8"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 accuracy (GSM8K)",
            "value": 0.9431,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: ATOM native FP8 gsm8k 3-shot flexible-extract=0.9447 (5-shot=0.9416); --gpu-memory-utilization 0.8 needed since the DSA index cache OOMs at default 0.9. Threshold 0.92 leaves ~2.5pp headroom. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9424 | fewshot: 3 | Model: /models/zai-org/GLM-5.2-FP8"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 accuracy (GSM8K)",
            "value": 0.9234,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9234 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP accuracy (GSM8K)",
            "value": 0.928,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9287 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP MTP acceptance (%)",
            "value": 75.63,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: Initial GLM-5.2-MXFP4 MTP online-quant native accuracy case. Threshold/baseline follow GLM-5.2-FP8 until MXFP4 MTP CI baseline is calibrated. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9287 | fewshot: 3 | Model: /models/amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP avg toks/fwd (tok/fwd)",
            "value": 3.27,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Kimi-K2.5-MXFP4 Eagle3 accuracy (GSM8K)",
            "value": 0.9348,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.91 | Baseline: 0.9257 | BaselineModel: amd/Kimi-K2.5-MXFP4 + lightseekorg/kimi-k2.5-eagle3 | BaselineNote: Eagle3 spec decode on Kimi-K2.5-MXFP4. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9348 | fewshot: 3 | Model: /models/amd/Kimi-K2.5-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K2.5-MXFP4 Eagle3 MTP acceptance (%)",
            "value": 68.94,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.91 | Baseline: 0.9257 | BaselineModel: amd/Kimi-K2.5-MXFP4 + lightseekorg/kimi-k2.5-eagle3 | BaselineNote: Eagle3 spec decode on Kimi-K2.5-MXFP4. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9348 | fewshot: 3 | Model: /models/amd/Kimi-K2.5-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K2.5-MXFP4 Eagle3 avg toks/fwd (tok/fwd)",
            "value": 3.07,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Kimi-K2.7-Code-MXFP4 accuracy (GSM8K)",
            "value": 0.9538,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.92 | Baseline: 0.9409 | BaselineModel: moonshotai/Kimi-K2.7-Code | BaselineNote: Kimi-K2.7-Code-MXFP4 native ATOM coverage; threshold inherited from Kimi-K2.5-MXFP4 until CI baseline is refreshed. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9545 | fewshot: 3 | Model: /models/amd/Kimi-K2.7-Code-MXFP4"
          },
          {
            "name": "ATOM::Kimi-K3 accuracy (GSM8K)",
            "value": 0.9439,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 | BaselineNote: Kimi-K3 (kimi_linear KDA+MLA, MXFP4 MoE) native ATOM FP8 kv-cache, TP8 (GSM8K 3-shot flexible-extract). Baseline 0.95; threshold 0.94 leaves ~1pp headroom. Refresh after the first CI run. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9431 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark accuracy (GSM8K)",
            "value": 0.9575,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9583 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark MTP acceptance (%)",
            "value": 47.63,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.94 | Baseline: 0.95 | BaselineModel: moonshotai/Kimi-K3 + Inferact/Kimi-K3-DSpark | BaselineNote: Kimi-K3 DSpark block spec-decode (7 tokens) on the Kimi-K3 target with the Inferact/Kimi-K3-DSpark draft. Spec-decode is lossless w.r.t. the target, so baseline reuses the Kimi-K3 FP8 base (0.95); threshold 0.94 matches the target. mtp_accept_threshold intentionally omitted until the first CI run reports the DSpark acceptance rate -- add it once measured. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.9583 | fewshot: 3 | Model: /models/moonshotai/Kimi-K3"
          },
          {
            "name": "ATOM::Kimi-K3 DSpark avg toks/fwd (tok/fwd)",
            "value": 4.33,
            "unit": "tok/fwd"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4-Preview accuracy (GSM8K)",
            "value": 0.9204,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: meta-llama/Llama-3.3-70B-Instruct | BaselineNote: HF page inaccessible; needs CI measurement of baseline | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.6497 | fewshot: 3 | Model: /models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview"
          },
          {
            "name": "ATOM::Meta-Llama-3-8B-Instruct accuracy (GSM8K)",
            "value": 0.7468,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Meta-Llama-3-8B-Instruct | BaselineNote: HF reports 0.796 but 8-shot CoT; CI uses 3-shot, not comparable | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X | VRAM: 252GB | ROCm: unknown | strict-match: 0.7475 | fewshot: 3 | Model: /models/meta-llama/Meta-Llama-3-8B-Instruct"
          },
          {
            "name": "ATOM::MiniMax-M2.7 accuracy (GSM8K)",
            "value": 0.8863,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.8872 | Baseline: 0.9022 | BaselineModel: MiniMaxAI/MiniMax-M2.7 | BaselineNote: ATOM CI measured: 0.9022 (gsm8k 3-shot flexible-extract). Threshold = baseline - 0.015. | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI350X VF | VRAM: 288GB | ROCm: unknown | strict-match: 0.9227 | fewshot: 3 | Model: /models/MiniMaxAI/MiniMax-M2.7"
          },
          {
            "name": "ATOM::Qwen3-235B-A22B-Instruct-2507-FP8 accuracy (GSM8K)",
            "value": 0.8992,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.87 | Baseline: 0.909 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: HF: amd/Qwen3-235B-A22B-Instruct-2507-MXFP4 card shows baseline=0.909 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.8772 | fewshot: 3 | Model: /models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
          },
          {
            "name": "ATOM::Qwen3-235B-A22B-Instruct-2507-MXFP4 accuracy (GSM8K)",
            "value": 0.8848,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266229831 | Threshold: 0.87 | Baseline: 0.909 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: HF: amd/Qwen3-235B-A22B-Instruct-2507-MXFP4 card shows baseline=0.909 | Docker: rocm/atom-dev:nightly_202608071513 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | strict-match: 0.8719 | fewshot: 3 | Model: /models/amd/Qwen3-235B-A22B-Instruct-2507-MXFP4"
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
        "date": 1786209099233,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM-SGLang::MI308 DeepSeek-V4-Flash accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31196803555 | Threshold: 0.94 | BaselineModel: sgl-project/DeepSeek-V4-Flash-FP8 | BaselineNote: MI308 SGLang DeepSeek-V4-Flash coverage uses the FP8 checkpoint to avoid the MXFP4 MoE loading path on gfx942; refresh baseline after nightly measurements land. | Docker: rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260805 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.9492 | fewshot: 5 | Model: /models/sgl-project/DeepSeek-V4-Flash-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3-32B-FP8 TP8 accuracy (GSM8K)",
            "value": 0.8795,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31196803555 | Threshold: 0.8 | BaselineModel: Qwen/Qwen3-32B-FP8 | BaselineNote: Adds max_gen_toks=1024 for the MI308 CI gsm8k path to avoid truncating Qwen3-32B reasoning output. | Docker: rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260805 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8939 | fewshot: 3 | Model: /models/Qwen/Qwen3-32B-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3.5-35B-A3B-FP8 TP1 accuracy (GSM8K)",
            "value": 0.8461,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31196803555 | Threshold: 0.76 | BaselineModel: Qwen/Qwen3.5-35B-A3B-FP8 | BaselineNote: Threshold aligned with the SGLANG accuracy validation workflow target for gsm8k. | Docker: rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260805 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8271 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-35B-A3B-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3.5-35B-A3B-PTPC-FP8 TP1 accuracy (GSM8K)",
            "value": 0.8461,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31196803555 | Threshold: 0.76 | BaselineModel: amd/Qwen3.5-35B-A3B-PTPC-FP8 | BaselineNote: Threshold aligned with the SGLANG accuracy validation workflow target for gsm8k. | Docker: rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260805 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8317 | fewshot: 3 | Model: /models/amd/Qwen3.5-35B-A3B-PTPC-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3.5-397B-A17B-FP8 TP4 accuracy (GSM8K)",
            "value": 0.8696,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31196803555 | Threshold: 0.83 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: Threshold aligned with the SGLANG accuracy validation workflow target for gsm8k. | Docker: rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260805 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.8552 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM-SGLang::MI308 Qwen3.5-397B-A17B-FP8 TP8 accuracy (GSM8K)",
            "value": 0.8795,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31196803555 | Threshold: 0.83 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | BaselineNote: Threshold aligned with the SGLANG accuracy validation workflow target for gsm8k. | Docker: rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260805 | GPU: AMD Instinct MI308X | VRAM: 192GB | ROCm: 7.2.4 | strict-match: 0.862 | fewshot: 3 | Model: /models/Qwen/Qwen3.5-397B-A17B-FP8"
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
        "date": 1786235282699,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 throughput (tok/s)",
            "value": 3008.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6022.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 TTFT (ms)",
            "value": 427.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 TPOT (ms)",
            "value": 41,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 throughput (tok/s)",
            "value": 759.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1527.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 TTFT (ms)",
            "value": 202.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 TPOT (ms)",
            "value": 20.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 throughput (tok/s)",
            "value": 4397.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8791.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 TTFT (ms)",
            "value": 724.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 TPOT (ms)",
            "value": 56.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 throughput (tok/s)",
            "value": 1263.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2523.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 TTFT (ms)",
            "value": 280.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 TPOT (ms)",
            "value": 24.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 throughput (tok/s)",
            "value": 191.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 Total Tput (tok/s)",
            "value": 383.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 TTFT (ms)",
            "value": 272.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 TPOT (ms)",
            "value": 19.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 throughput (tok/s)",
            "value": 2037.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4076.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 TTFT (ms)",
            "value": 301.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 TPOT (ms)",
            "value": 30.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 throughput (tok/s)",
            "value": 370.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 Total Tput (tok/s)",
            "value": 738.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 TTFT (ms)",
            "value": 192.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 TPOT (ms)",
            "value": 20.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 1024/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 throughput (tok/s)",
            "value": 1938.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 Total Tput (tok/s)",
            "value": 17498.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 TTFT (ms)",
            "value": 2204.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 TPOT (ms)",
            "value": 62.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 throughput (tok/s)",
            "value": 580.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 Total Tput (tok/s)",
            "value": 5239.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 TTFT (ms)",
            "value": 677.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 TPOT (ms)",
            "value": 25.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 throughput (tok/s)",
            "value": 2216.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 Total Tput (tok/s)",
            "value": 19940.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 TTFT (ms)",
            "value": 6491.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 TPOT (ms)",
            "value": 106.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 throughput (tok/s)",
            "value": 1047.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 Total Tput (tok/s)",
            "value": 9371.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 TTFT (ms)",
            "value": 822.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 TPOT (ms)",
            "value": 28.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 throughput (tok/s)",
            "value": 182.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 Total Tput (tok/s)",
            "value": 1639.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 TTFT (ms)",
            "value": 445.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 TPOT (ms)",
            "value": 20.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 throughput (tok/s)",
            "value": 1390.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 Total Tput (tok/s)",
            "value": 12530.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 TTFT (ms)",
            "value": 1336.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 TPOT (ms)",
            "value": 42.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 throughput (tok/s)",
            "value": 363.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3230.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 TTFT (ms)",
            "value": 453.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 TPOT (ms)",
            "value": 21.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro 8192/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 throughput (tok/s)",
            "value": 3046.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6099.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 TTFT (ms)",
            "value": 4699.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 TPOT (ms)",
            "value": 35.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 throughput (tok/s)",
            "value": 5603.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 Total Tput (tok/s)",
            "value": 11200.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 TTFT (ms)",
            "value": 3960.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 TPOT (ms)",
            "value": 39.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 throughput (tok/s)",
            "value": 9131.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 Total Tput (tok/s)",
            "value": 18254.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 TTFT (ms)",
            "value": 4743.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 TPOT (ms)",
            "value": 48.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 throughput (tok/s)",
            "value": 2024.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4050.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 TTFT (ms)",
            "value": 2801.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 TPOT (ms)",
            "value": 26.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 throughput (tok/s)",
            "value": 4378.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 Total Tput (tok/s)",
            "value": 39398.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 TTFT (ms)",
            "value": 66350.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 TPOT (ms)",
            "value": 162.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=1024 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 throughput (tok/s)",
            "value": 2248.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 Total Tput (tok/s)",
            "value": 20301.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 TTFT (ms)",
            "value": 6020.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 TPOT (ms)",
            "value": 48.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 throughput (tok/s)",
            "value": 3329.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 Total Tput (tok/s)",
            "value": 29956.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 TTFT (ms)",
            "value": 7040.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 TPOT (ms)",
            "value": 67.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 throughput (tok/s)",
            "value": 3966.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 Total Tput (tok/s)",
            "value": 35700.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 TTFT (ms)",
            "value": 10165,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 TPOT (ms)",
            "value": 114.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 throughput (tok/s)",
            "value": 1498.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 Total Tput (tok/s)",
            "value": 13510.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 TTFT (ms)",
            "value": 4760.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 TPOT (ms)",
            "value": 35.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 throughput (tok/s)",
            "value": 5828.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 Total Tput (tok/s)",
            "value": 11640.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 TTFT (ms)",
            "value": 8312.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 TPOT (ms)",
            "value": 32.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 3.28,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 Acceptance Rate (%)",
            "value": 32.56,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 throughput (tok/s)",
            "value": 8710.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 Total Tput (tok/s)",
            "value": 17396.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 TTFT (ms)",
            "value": 8308.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 TPOT (ms)",
            "value": 46.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 Accept Length (tok/fwd)",
            "value": 3.19,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 Acceptance Rate (%)",
            "value": 31.24,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 throughput (tok/s)",
            "value": 2271.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4540.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 TTFT (ms)",
            "value": 4959.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 TPOT (ms)",
            "value": 20.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 Acceptance Rate (%)",
            "value": 28.11,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=128 throughput (tok/s)",
            "value": 3020.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=128 Total Tput (tok/s)",
            "value": 27202.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=128 TTFT (ms)",
            "value": 7467.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=128 TPOT (ms)",
            "value": 31.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 4.04,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=128 Acceptance Rate (%)",
            "value": 43.4,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=256 throughput (tok/s)",
            "value": 3808.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=256 Total Tput (tok/s)",
            "value": 34176.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=256 TTFT (ms)",
            "value": 8731.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=256 TPOT (ms)",
            "value": 54.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 4.16,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=256 Acceptance Rate (%)",
            "value": 45.2,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 throughput (tok/s)",
            "value": 4364.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 Total Tput (tok/s)",
            "value": 39182.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 TTFT (ms)",
            "value": 11404.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 TPOT (ms)",
            "value": 100.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 Accept Length (tok/fwd)",
            "value": 3.18,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 Acceptance Rate (%)",
            "value": 31.12,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=64 throughput (tok/s)",
            "value": 2038.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=64 Total Tput (tok/s)",
            "value": 18327.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=64 TTFT (ms)",
            "value": 6806.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=64 TPOT (ms)",
            "value": 22.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 4.1,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=64 Acceptance Rate (%)",
            "value": 44.27,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA DSpark 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 throughput (tok/s)",
            "value": 3979.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 7965.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 TTFT (ms)",
            "value": 4704.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 TPOT (ms)",
            "value": 25.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.25,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 Acceptance Rate (%)",
            "value": 41.69,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 throughput (tok/s)",
            "value": 7208.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 14407.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 TTFT (ms)",
            "value": 5765.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 TPOT (ms)",
            "value": 27.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.29,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 Acceptance Rate (%)",
            "value": 42.96,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 throughput (tok/s)",
            "value": 9741.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 Total Tput (tok/s)",
            "value": 19470.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 TTFT (ms)",
            "value": 6945.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 TPOT (ms)",
            "value": 42.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 Accept Length (tok/fwd)",
            "value": 2.29,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 Acceptance Rate (%)",
            "value": 42.93,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 throughput (tok/s)",
            "value": 2854.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5710.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 TTFT (ms)",
            "value": 3288.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 TPOT (ms)",
            "value": 17.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.17,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 Acceptance Rate (%)",
            "value": 38.88,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 throughput (tok/s)",
            "value": 4732.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 Total Tput (tok/s)",
            "value": 42549.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 TTFT (ms)",
            "value": 18736.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 TPOT (ms)",
            "value": 192.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 Accept Length (tok/fwd)",
            "value": 2.53,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 Acceptance Rate (%)",
            "value": 51.09,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=1024 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 throughput (tok/s)",
            "value": 2725.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 24586.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 TTFT (ms)",
            "value": 7179.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 TPOT (ms)",
            "value": 37.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.51,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 Acceptance Rate (%)",
            "value": 50.47,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 throughput (tok/s)",
            "value": 3563.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 Total Tput (tok/s)",
            "value": 32035.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 TTFT (ms)",
            "value": 9575.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 TPOT (ms)",
            "value": 59.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.56,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 Acceptance Rate (%)",
            "value": 52.06,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 throughput (tok/s)",
            "value": 4188.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 Total Tput (tok/s)",
            "value": 37663.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 TTFT (ms)",
            "value": 12844.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 TPOT (ms)",
            "value": 105.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 Accept Length (tok/fwd)",
            "value": 2.56,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 Acceptance Rate (%)",
            "value": 51.94,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 throughput (tok/s)",
            "value": 2143.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 19302.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 TTFT (ms)",
            "value": 5440.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 TPOT (ms)",
            "value": 23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.48,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 Acceptance Rate (%)",
            "value": 49.37,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA MTP3 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 throughput (tok/s)",
            "value": 4812.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 Total Tput (tok/s)",
            "value": 9621.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 TTFT (ms)",
            "value": 4308.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 TPOT (ms)",
            "value": 45.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 throughput (tok/s)",
            "value": 7948.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 Total Tput (tok/s)",
            "value": 15890.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 TTFT (ms)",
            "value": 4803.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 TPOT (ms)",
            "value": 54.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 1024/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 throughput (tok/s)",
            "value": 5091.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 Total Tput (tok/s)",
            "value": 45812.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 TTFT (ms)",
            "value": 12892.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 TPOT (ms)",
            "value": 183.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=1024 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 throughput (tok/s)",
            "value": 3045.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 Total Tput (tok/s)",
            "value": 27400.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 TTFT (ms)",
            "value": 6968.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 TPOT (ms)",
            "value": 72.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 throughput (tok/s)",
            "value": 4382.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 Total Tput (tok/s)",
            "value": 39445.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 TTFT (ms)",
            "value": 8556.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 TPOT (ms)",
            "value": 104.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro DPA TBO 8192/1024 c=512 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 throughput (tok/s)",
            "value": 3299.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6605.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 TTFT (ms)",
            "value": 527.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 TPOT (ms)",
            "value": 37.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.26,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 Acceptance Rate (%)",
            "value": 42.11,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 throughput (tok/s)",
            "value": 877.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1762.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 TTFT (ms)",
            "value": 304.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 TPOT (ms)",
            "value": 17.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.26,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 Acceptance Rate (%)",
            "value": 41.85,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 throughput (tok/s)",
            "value": 4036.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8068.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 TTFT (ms)",
            "value": 1310.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 TPOT (ms)",
            "value": 59.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.29,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 Acceptance Rate (%)",
            "value": 43.06,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 throughput (tok/s)",
            "value": 1672.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3338.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 TTFT (ms)",
            "value": 317.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 TPOT (ms)",
            "value": 18.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.22,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 Acceptance Rate (%)",
            "value": 40.55,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 throughput (tok/s)",
            "value": 361.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 726.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 TTFT (ms)",
            "value": 278.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 TPOT (ms)",
            "value": 10.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.26,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 Acceptance Rate (%)",
            "value": 41.98,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 throughput (tok/s)",
            "value": 2254.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 4509.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 TTFT (ms)",
            "value": 403.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 TPOT (ms)",
            "value": 27.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.15,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 Acceptance Rate (%)",
            "value": 38.32,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 throughput (tok/s)",
            "value": 690.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1376.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 TTFT (ms)",
            "value": 233.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 TPOT (ms)",
            "value": 10.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.28,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 Acceptance Rate (%)",
            "value": 42.59,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 1024/1024 c=8 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 throughput (tok/s)",
            "value": 2099.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 18937.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 TTFT (ms)",
            "value": 2282.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 TPOT (ms)",
            "value": 57.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.53,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 Acceptance Rate (%)",
            "value": 50.95,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 throughput (tok/s)",
            "value": 989.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 8924.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 TTFT (ms)",
            "value": 636.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 TPOT (ms)",
            "value": 15.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.51,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 Acceptance Rate (%)",
            "value": 50.32,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=16 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 throughput (tok/s)",
            "value": 2343.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 Total Tput (tok/s)",
            "value": 21068.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 TTFT (ms)",
            "value": 4171.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 TPOT (ms)",
            "value": 102.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.55,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 Acceptance Rate (%)",
            "value": 51.77,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 throughput (tok/s)",
            "value": 1364.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 12188.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 TTFT (ms)",
            "value": 855.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 TPOT (ms)",
            "value": 21.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.46,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 Acceptance Rate (%)",
            "value": 48.8,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=32 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 throughput (tok/s)",
            "value": 436.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 3924.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 TTFT (ms)",
            "value": 419.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 TPOT (ms)",
            "value": 8.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.65,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 Acceptance Rate (%)",
            "value": 54.99,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=4 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 throughput (tok/s)",
            "value": 1764.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 15887.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 TTFT (ms)",
            "value": 1339.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 TPOT (ms)",
            "value": 34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.47,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 Acceptance Rate (%)",
            "value": 48.98,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro MTP3 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 throughput (tok/s)",
            "value": 2549.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5104.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 TTFT (ms)",
            "value": 492.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 TPOT (ms)",
            "value": 48.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 throughput (tok/s)",
            "value": 4072,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8140.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 TTFT (ms)",
            "value": 1033.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 TPOT (ms)",
            "value": 59.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 throughput (tok/s)",
            "value": 1586.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3173.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 TTFT (ms)",
            "value": 408.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 TPOT (ms)",
            "value": 37.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 1024/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 throughput (tok/s)",
            "value": 1681.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 Total Tput (tok/s)",
            "value": 15181.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 TTFT (ms)",
            "value": 2127.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 TPOT (ms)",
            "value": 72.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=128 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 throughput (tok/s)",
            "value": 1857.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 Total Tput (tok/s)",
            "value": 16709.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 TTFT (ms)",
            "value": 6245.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 TPOT (ms)",
            "value": 129.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=256 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 throughput (tok/s)",
            "value": 1140.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 Total Tput (tok/s)",
            "value": 10282.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 TTFT (ms)",
            "value": 1923.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 TPOT (ms)",
            "value": 50.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::DeepSeek-V4-Pro TBO 8192/1024 c=64 _tp",
            "value": 8,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2364.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 4734.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 446.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 52.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 throughput (tok/s)",
            "value": 724.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1456.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 TTFT (ms)",
            "value": 156.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 TPOT (ms)",
            "value": 21.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 throughput (tok/s)",
            "value": 3550.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 Total Tput (tok/s)",
            "value": 7098.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 TTFT (ms)",
            "value": 614.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 TPOT (ms)",
            "value": 69.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 throughput (tok/s)",
            "value": 1079.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2155.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 TTFT (ms)",
            "value": 252.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 TPOT (ms)",
            "value": 28.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 throughput (tok/s)",
            "value": 248.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 Total Tput (tok/s)",
            "value": 499.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 TTFT (ms)",
            "value": 116.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 TPOT (ms)",
            "value": 15.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 throughput (tok/s)",
            "value": 1628.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3258.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 TTFT (ms)",
            "value": 345.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 TPOT (ms)",
            "value": 37.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 throughput (tok/s)",
            "value": 415.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 Total Tput (tok/s)",
            "value": 828.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 TTFT (ms)",
            "value": 179.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 TPOT (ms)",
            "value": 18.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 throughput (tok/s)",
            "value": 1289.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 Total Tput (tok/s)",
            "value": 11643.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 TTFT (ms)",
            "value": 10688.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 TPOT (ms)",
            "value": 85.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 throughput (tok/s)",
            "value": 584.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 Total Tput (tok/s)",
            "value": 5278.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 TTFT (ms)",
            "value": 655.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 TPOT (ms)",
            "value": 25.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 throughput (tok/s)",
            "value": 1299.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 Total Tput (tok/s)",
            "value": 11695.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 TTFT (ms)",
            "value": 96398.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 TPOT (ms)",
            "value": 86.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 throughput (tok/s)",
            "value": 824.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 Total Tput (tok/s)",
            "value": 7369.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 TTFT (ms)",
            "value": 919.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 TPOT (ms)",
            "value": 36.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 throughput (tok/s)",
            "value": 213.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 Total Tput (tok/s)",
            "value": 1922.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 TTFT (ms)",
            "value": 426.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 TPOT (ms)",
            "value": 17.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 throughput (tok/s)",
            "value": 1115.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 Total Tput (tok/s)",
            "value": 10058.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 TTFT (ms)",
            "value": 1482.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 TPOT (ms)",
            "value": 54.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 throughput (tok/s)",
            "value": 377.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3359.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 TTFT (ms)",
            "value": 513.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 TPOT (ms)",
            "value": 20.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 throughput (tok/s)",
            "value": 3720.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 7447.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 TTFT (ms)",
            "value": 482.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 TPOT (ms)",
            "value": 32.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 Acceptance Rate (%)",
            "value": 65.61,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 throughput (tok/s)",
            "value": 1100.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2209.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 TTFT (ms)",
            "value": 231.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 TPOT (ms)",
            "value": 13.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.92,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 Acceptance Rate (%)",
            "value": 63.92,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 throughput (tok/s)",
            "value": 5034.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 10062.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 TTFT (ms)",
            "value": 798.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 TPOT (ms)",
            "value": 48.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 3.01,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 Acceptance Rate (%)",
            "value": 66.94,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 throughput (tok/s)",
            "value": 1764.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3520.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 TTFT (ms)",
            "value": 414.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 TPOT (ms)",
            "value": 17.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.96,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 Acceptance Rate (%)",
            "value": 65.32,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 throughput (tok/s)",
            "value": 492.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 990.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 TTFT (ms)",
            "value": 158.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 TPOT (ms)",
            "value": 7.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 Acceptance Rate (%)",
            "value": 65.92,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 throughput (tok/s)",
            "value": 2603.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5208.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 TTFT (ms)",
            "value": 337.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 TPOT (ms)",
            "value": 23.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 Acceptance Rate (%)",
            "value": 65.57,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 throughput (tok/s)",
            "value": 740.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1475.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 TTFT (ms)",
            "value": 188.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 TPOT (ms)",
            "value": 10.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 Accept Length (tok/fwd)",
            "value": 3.02,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 Acceptance Rate (%)",
            "value": 67.18,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 throughput (tok/s)",
            "value": 1591.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 14356.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 TTFT (ms)",
            "value": 14510.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 TPOT (ms)",
            "value": 63.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.93,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 Acceptance Rate (%)",
            "value": 64.39,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 throughput (tok/s)",
            "value": 837.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 7555.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 TTFT (ms)",
            "value": 704.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 TPOT (ms)",
            "value": 17.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.95,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 Acceptance Rate (%)",
            "value": 64.86,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 throughput (tok/s)",
            "value": 1607.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 Total Tput (tok/s)",
            "value": 14450.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 TTFT (ms)",
            "value": 83967.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 TPOT (ms)",
            "value": 64.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 Acceptance Rate (%)",
            "value": 66,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 throughput (tok/s)",
            "value": 1195.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 10679.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 TTFT (ms)",
            "value": 1014.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 TPOT (ms)",
            "value": 25.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 Acceptance Rate (%)",
            "value": 64.79,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 throughput (tok/s)",
            "value": 446.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 4009.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 TTFT (ms)",
            "value": 451.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 TPOT (ms)",
            "value": 8.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.98,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 Acceptance Rate (%)",
            "value": 65.99,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 throughput (tok/s)",
            "value": 1499.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 13498.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 TTFT (ms)",
            "value": 1640.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 TPOT (ms)",
            "value": 40.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.97,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 Acceptance Rate (%)",
            "value": 65.61,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 throughput (tok/s)",
            "value": 569.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 5062.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 TTFT (ms)",
            "value": 630.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 TPOT (ms)",
            "value": 13.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.9,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 Acceptance Rate (%)",
            "value": 63.18,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-FP8 MTP3 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 throughput (tok/s)",
            "value": 3201.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 6409.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 TTFT (ms)",
            "value": 353.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 TPOT (ms)",
            "value": 38.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 throughput (tok/s)",
            "value": 834.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1678,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 TTFT (ms)",
            "value": 143.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 TPOT (ms)",
            "value": 18.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 throughput (tok/s)",
            "value": 4469.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8934.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 TTFT (ms)",
            "value": 570.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 TPOT (ms)",
            "value": 55.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 throughput (tok/s)",
            "value": 1237.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2471.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 TTFT (ms)",
            "value": 394.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 TPOT (ms)",
            "value": 24.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 throughput (tok/s)",
            "value": 301.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 604.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 TTFT (ms)",
            "value": 131.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 TPOT (ms)",
            "value": 12.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 throughput (tok/s)",
            "value": 1980.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3962.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 TTFT (ms)",
            "value": 283.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 TPOT (ms)",
            "value": 31.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 throughput (tok/s)",
            "value": 479.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 955.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 TTFT (ms)",
            "value": 144.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 TPOT (ms)",
            "value": 16.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 throughput (tok/s)",
            "value": 1776.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 16036.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 TTFT (ms)",
            "value": 2297.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 TPOT (ms)",
            "value": 68.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 throughput (tok/s)",
            "value": 639.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 5775.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 TTFT (ms)",
            "value": 599.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 TPOT (ms)",
            "value": 23.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 throughput (tok/s)",
            "value": 2091.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 18816.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 TTFT (ms)",
            "value": 4016.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 TPOT (ms)",
            "value": 116.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 throughput (tok/s)",
            "value": 921,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 8235.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 TTFT (ms)",
            "value": 811.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 TPOT (ms)",
            "value": 33.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 throughput (tok/s)",
            "value": 265.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2389.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 TTFT (ms)",
            "value": 394.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 TPOT (ms)",
            "value": 14.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 throughput (tok/s)",
            "value": 1322.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 11924.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 TTFT (ms)",
            "value": 1319.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 TPOT (ms)",
            "value": 46.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 throughput (tok/s)",
            "value": 405.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3609.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 TTFT (ms)",
            "value": 474.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 TPOT (ms)",
            "value": 18.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 throughput (tok/s)",
            "value": 4264.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8537.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 TTFT (ms)",
            "value": 426.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 TPOT (ms)",
            "value": 28.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.92,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 Acceptance Rate (%)",
            "value": 64.03,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 throughput (tok/s)",
            "value": 1286.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2583.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 TTFT (ms)",
            "value": 237.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 TPOT (ms)",
            "value": 11.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.87,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 Acceptance Rate (%)",
            "value": 62.22,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 throughput (tok/s)",
            "value": 5599.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 11193.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 TTFT (ms)",
            "value": 640.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 TPOT (ms)",
            "value": 44.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.94,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 Acceptance Rate (%)",
            "value": 64.57,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 throughput (tok/s)",
            "value": 2237.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 4464.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 TTFT (ms)",
            "value": 261.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 TPOT (ms)",
            "value": 13.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.89,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 Acceptance Rate (%)",
            "value": 63.13,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 throughput (tok/s)",
            "value": 583.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1172.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 TTFT (ms)",
            "value": 136.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 TPOT (ms)",
            "value": 6.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.95,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 Acceptance Rate (%)",
            "value": 64.93,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 throughput (tok/s)",
            "value": 3098.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 6198.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 TTFT (ms)",
            "value": 313.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 TPOT (ms)",
            "value": 19.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.9,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 Acceptance Rate (%)",
            "value": 63.46,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 throughput (tok/s)",
            "value": 875.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1744.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 TTFT (ms)",
            "value": 151.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 TPOT (ms)",
            "value": 8.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.88,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 Acceptance Rate (%)",
            "value": 62.78,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 1024/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 throughput (tok/s)",
            "value": 2105.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 18987.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 TTFT (ms)",
            "value": 2388.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 TPOT (ms)",
            "value": 57.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 Accept Length (tok/fwd)",
            "value": 2.85,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 Acceptance Rate (%)",
            "value": 61.59,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=128 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 throughput (tok/s)",
            "value": 1038.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 9367.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 TTFT (ms)",
            "value": 609.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 TPOT (ms)",
            "value": 14.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 Accept Length (tok/fwd)",
            "value": 2.9,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 Acceptance Rate (%)",
            "value": 63.27,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=16 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 throughput (tok/s)",
            "value": 1950.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 Total Tput (tok/s)",
            "value": 17527.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 TTFT (ms)",
            "value": 4467.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 TPOT (ms)",
            "value": 124.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 Accept Length (tok/fwd)",
            "value": 2.89,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 Acceptance Rate (%)",
            "value": 63.07,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=256 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 throughput (tok/s)",
            "value": 1478.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 13211.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 TTFT (ms)",
            "value": 950.41,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 TPOT (ms)",
            "value": 20.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 Accept Length (tok/fwd)",
            "value": 2.89,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 Acceptance Rate (%)",
            "value": 63.13,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=32 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 throughput (tok/s)",
            "value": 493.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 4427.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 TTFT (ms)",
            "value": 394.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 TPOT (ms)",
            "value": 7.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 Accept Length (tok/fwd)",
            "value": 2.83,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 Acceptance Rate (%)",
            "value": 60.96,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=4 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 throughput (tok/s)",
            "value": 1798.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 16195.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 TTFT (ms)",
            "value": 1413.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 TPOT (ms)",
            "value": 33.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 Accept Length (tok/fwd)",
            "value": 2.88,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 Acceptance Rate (%)",
            "value": 62.73,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=64 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 throughput (tok/s)",
            "value": 678.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 6027.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 TTFT (ms)",
            "value": 534.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 TPOT (ms)",
            "value": 10.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 Accept Length (tok/fwd)",
            "value": 2.88,
            "unit": "tok/fwd",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 Acceptance Rate (%)",
            "value": 62.55,
            "unit": "%",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 _gpu_count",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::GLM-5.2-MXFP4 MTP3 8192/1024 c=8 _tp",
            "value": 4,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 throughput (tok/s)",
            "value": 4793.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 Total Tput (tok/s)",
            "value": 9596.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 TTFT (ms)",
            "value": 415.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 TPOT (ms)",
            "value": 25.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=128 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 throughput (tok/s)",
            "value": 984.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1980.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 TTFT (ms)",
            "value": 146.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 TPOT (ms)",
            "value": 15.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=16 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 throughput (tok/s)",
            "value": 6111.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 Total Tput (tok/s)",
            "value": 12217.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 TTFT (ms)",
            "value": 662.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 TPOT (ms)",
            "value": 40.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=256 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 throughput (tok/s)",
            "value": 1781.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3557.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 TTFT (ms)",
            "value": 214.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 TPOT (ms)",
            "value": 17.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=32 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 throughput (tok/s)",
            "value": 266.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 Total Tput (tok/s)",
            "value": 535.05,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 TTFT (ms)",
            "value": 113.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 TPOT (ms)",
            "value": 14.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=4 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 throughput (tok/s)",
            "value": 3049.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 Total Tput (tok/s)",
            "value": 6100.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 TTFT (ms)",
            "value": 290.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 TPOT (ms)",
            "value": 19.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=64 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 throughput (tok/s)",
            "value": 520.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1036.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 TTFT (ms)",
            "value": 111.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 TPOT (ms)",
            "value": 14.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 1024/1024 c=8 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 throughput (tok/s)",
            "value": 1453.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 Total Tput (tok/s)",
            "value": 13117.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 TTFT (ms)",
            "value": 2681.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 TPOT (ms)",
            "value": 84.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=128 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 throughput (tok/s)",
            "value": 656.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 Total Tput (tok/s)",
            "value": 5928.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 TTFT (ms)",
            "value": 669.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 TPOT (ms)",
            "value": 22.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=16 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 throughput (tok/s)",
            "value": 1399.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 Total Tput (tok/s)",
            "value": 12593.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 TTFT (ms)",
            "value": 57156.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 TPOT (ms)",
            "value": 117,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=256 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 throughput (tok/s)",
            "value": 961.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 Total Tput (tok/s)",
            "value": 8597.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 TTFT (ms)",
            "value": 985.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 TPOT (ms)",
            "value": 31.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=32 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 throughput (tok/s)",
            "value": 212.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 Total Tput (tok/s)",
            "value": 1908.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 TTFT (ms)",
            "value": 451.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 TPOT (ms)",
            "value": 17.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=4 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 throughput (tok/s)",
            "value": 1249.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 Total Tput (tok/s)",
            "value": 11265.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 TTFT (ms)",
            "value": 1586.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 TPOT (ms)",
            "value": 48.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=64 _tp",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 throughput (tok/s)",
            "value": 391.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 Total Tput (tok/s)",
            "value": 3483.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 TTFT (ms)",
            "value": 554.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 TPOT (ms)",
            "value": 19.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266739829 | GPU: AMD Instinct MI355X | VRAM: 288GB | ROCm: unknown | Docker: rocm/atom-dev:nightly_202608071513"
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "ATOM::Llama-3.3-70B-Instruct-MXFP4 8192/1024 c=8 _tp",
            "value": 1,
            "unit": ""
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
        "date": 1786239147878,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "ATOM-vLLM::DeepSeek-R1-0528-MXFP4 TP8 accuracy (GSM8K)",
            "value": 0.9424,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.93 | Baseline: 0.93 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9378 | fewshot: 3 | Model: amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-R1-FP8 TP8 accuracy (GSM8K)",
            "value": 0.9462,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.93 | Baseline: 0.93 | BaselineModel: deepseek-ai/DeepSeek-R1-0528 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 3 | Model: deepseek-ai/DeepSeek-R1-0528"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8 MTP TP4 accuracy (GSM8K)",
            "value": 0.9484,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.93 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9469 | fewshot: 20 | Model: deepseek-ai/DeepSeek-V3.2"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8 PTPC TP4 accuracy (GSM8K)",
            "value": 0.9416,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.93 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9416 | fewshot: 20 | Model: amd/DeepSeek-V3.2-mtp-ptpc"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8 TP4 accuracy (GSM8K)",
            "value": 0.9522,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.93 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9522 | fewshot: 20 | Model: deepseek-ai/DeepSeek-V3.2"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V3.2-FP8 TP8 accuracy (GSM8K)",
            "value": 0.953,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.93 | Baseline: 0.956 | BaselineModel: deepseek-ai/DeepSeek-V3.2 | BaselineNote: 20-shot gsm8k reference from DeepSeek-V3.2 usage docs; nightly uses 20-shot to exercise sparse MLA. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9538 | fewshot: 20 | Model: deepseek-ai/DeepSeek-V3.2"
          },
          {
            "name": "ATOM-vLLM::DeepSeek-V4-Pro TP8 accuracy (GSM8K)",
            "value": 0.8355,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.94 | Baseline: 0.94 | BaselineModel: deepseek-ai/DeepSeek-V4-Pro | BaselineNote: 20-shot GSM8K local-completions coverage aligned with launch.sh/lm_eval.sh. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7483 | fewshot: 20 | Model: deepseek-ai/DeepSeek-V4-Pro"
          },
          {
            "name": "ATOM-vLLM::GLM-4.7-FP8 MTP TP4 accuracy (GSM8K)",
            "value": 0.9409,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9409 | fewshot: 3 | Model: zai-org/GLM-4.7-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-4.7-FP8 MTP TP8 accuracy (GSM8K)",
            "value": 0.9333,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9249 | fewshot: 3 | Model: zai-org/GLM-4.7-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-4.7-FP8 TP4 accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9424 | fewshot: 3 | Model: zai-org/GLM-4.7-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-4.7-FP8 TP8 accuracy (GSM8K)",
            "value": 0.9378,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Baseline: 0.9386 | BaselineModel: zai-org/GLM-4.7-FP8 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9318 | fewshot: 3 | Model: zai-org/GLM-4.7-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-5.1-FP8 TP8 accuracy (GSM8K)",
            "value": 0.9454,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.88 | Baseline: 0.9545 | BaselineModel: zai-org/GLM-5.1 | BaselineNote: CI uses 3-shot, not comparable to HF 5-shot baseline | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9431 | fewshot: 20 | Model: zai-org/GLM-5.1-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-FP8 TP4 accuracy (GSM8K)",
            "value": 0.9386,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: 20-shot GSM8K local-completions coverage for GLM-5.2-FP8 IndexShare; threshold follows the existing GLM-5.2 nightly gate until FP8 CI baseline is recalibrated. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9393 | fewshot: 20 | Model: zai-org/GLM-5.2-FP8"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4 MTP TP4 accuracy (GSM8K)",
            "value": 0.9303,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: 20-shot GSM8K is lossless for MTP; threshold follows GLM-5.2-FP8 until MXFP4 MTP-specific CI baseline is calibrated. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9295 | fewshot: 20 | Model: amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM-vLLM::GLM-5.2-MXFP4 TP4 accuracy (GSM8K)",
            "value": 0.9257,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Baseline: 0.9447 | BaselineModel: zai-org/GLM-5.2-FP8 | BaselineNote: 20-shot GSM8K local-completions coverage for GLM-5.2-MXFP4 IndexShare; threshold/baseline follow GLM-5.2-FP8 until MXFP4 CI baseline is calibrated. | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9265 | fewshot: 20 | Model: amd/GLM-5.2-MXFP4"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2-Thinking-MXFP4 TP4 accuracy (GSM8K)",
            "value": 0.9325,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.9 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9303 | fewshot: 3 | Model: amd/Kimi-K2-Thinking-MXFP4-AttnFP8"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2-Thinking-MXFP4 TP8 accuracy (GSM8K)",
            "value": 0.931,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.9 | Baseline: 0.9 | BaselineModel: amd/Kimi-K2-Thinking-MXFP4 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.931 | fewshot: 3 | Model: amd/Kimi-K2-Thinking-MXFP4-AttnFP8"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.5-MXFP4 TP4 accuracy (GSM8K)",
            "value": 0.9386,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9378 | fewshot: 3 | Model: amd/Kimi-K2.5-MXFP4-AttnFP8"
          },
          {
            "name": "ATOM-vLLM::Kimi-K2.5-MXFP4 TP8 accuracy (GSM8K)",
            "value": 0.928,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.93 | Baseline: 0.93 | BaselineModel: amd/Kimi-K2.5-MXFP4 | BaselineNote: Reference value from recipes/atom_vllm/Kimi-K2.5.md | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9287 | fewshot: 3 | Model: amd/Kimi-K2.5-MXFP4-AttnFP8"
          },
          {
            "name": "ATOM-vLLM::Llama-3.1-8B-Instruct TP1 accuracy (GSM8K)",
            "value": 0.7468,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.73 | Baseline: 0.75 | BaselineModel: meta-llama/Llama-3.1-8B-Instruct | BaselineNote: Threshold aligned with existing 8B Llama baseline used in CI (3-shot GSM8K). | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.6619 | fewshot: 3 | Model: meta-llama/Llama-3.1-8B-Instruct"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M2.5 TP2 accuracy (GSM8K)",
            "value": 0.934,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.9325 | fewshot: 3 | Model: MiniMaxAI/MiniMax-M2.5"
          },
          {
            "name": "ATOM-vLLM::MiniMax-M2.5 TP4 accuracy (GSM8K)",
            "value": 0.931,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.92 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.928 | fewshot: 3 | Model: MiniMaxAI/MiniMax-M2.5"
          },
          {
            "name": "ATOM-vLLM::Qwen3-235B-A22B-Instruct-2507-FP8 TP8+EP8 accuracy (GSM8K)",
            "value": 0.9022,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.87 | Baseline: 0.87 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8855 | fewshot: 3 | Model: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8 TP1 accuracy (GSM8K)",
            "value": 0.8059,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.81 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7225 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8 TP2 accuracy (GSM8K)",
            "value": 0.8165,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.81 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.7255 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8 TP4 accuracy (GSM8K)",
            "value": 0.0697,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.81 | Baseline: 0.76 | BaselineModel: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.0705 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8-MTP TP1 accuracy (GSM8K)",
            "value": 0.7991,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.8 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.721 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3-Next-80B-A3B-Instruct-FP8-MTP TP4 accuracy (GSM8K)",
            "value": 0.0652,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.8 | Baseline: 0.81 | BaselineModel: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | BaselineNote: Qwen3-Next-80B-A3B-Instruct-FP8 baseline with TP4 (no MTP) as proxy; needs CI measurement for MTP-specific baseline | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.066 | fewshot: 3 | Model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B TP8 accuracy (GSM8K)",
            "value": 0.8529,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.83 | Baseline: 0.83 | BaselineModel: Qwen/Qwen3.5-397B-A17B | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.84 | fewshot: 3 | Model: Qwen/Qwen3.5-397B-A17B"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-FP8 TP4 accuracy (GSM8K)",
            "value": 0.8658,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.83 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8484 | fewshot: 3 | Model: Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-FP8 TP8 accuracy (GSM8K)",
            "value": 0.8575,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.83 | Baseline: 0.83 | BaselineModel: Qwen/Qwen3.5-397B-A17B-FP8 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.8476 | fewshot: 3 | Model: Qwen/Qwen3.5-397B-A17B-FP8"
          },
          {
            "name": "ATOM-vLLM::Qwen3.5-397B-A17B-MXFP4 TP4 accuracy (GSM8K)",
            "value": 0.856,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.83 | Baseline: 0.82 | BaselineModel: Qwen/Qwen3-235B-A22B-Instruct-2507 | BaselineNote: Using Qwen3-235B baseline as proxy; needs CI measurement for Qwen3.5 specific baseline | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.84 | fewshot: 3 | Model: amd/Qwen3.5-397B-A17B-MXFP4"
          },
          {
            "name": "ATOM-vLLM::gpt-oss-120b TP1 accuracy (GSM8K)",
            "value": 0.8855,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.3139 | fewshot: 3 | Model: openai/gpt-oss-120b"
          },
          {
            "name": "ATOM-vLLM::gpt-oss-120b TP2 accuracy (GSM8K)",
            "value": 0.8893,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.88 | Baseline: 0.9 | BaselineModel: openai/gpt-oss-120b | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.4716 | fewshot: 3 | Model: openai/gpt-oss-120b"
          },
          {
            "name": "ATOM-vLLM::gpt-oss-120b TP8 accuracy (GSM8K)",
            "value": 0.8984,
            "unit": "score",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/31266228818 | Threshold: 0.88 | Docker: rocm/atom-dev:vllm-v0.25.1-nightly_20260805 | GPU: AMD Radeon Graphics | VRAM: 288GB | ROCm: 7.2.4 | strict-match: 0.1751 | fewshot: 3 | Model: openai/gpt-oss-120b"
          }
        ]
      }
    ]
  }
}