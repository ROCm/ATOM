window.BENCHMARK_DATA = {
  "lastUpdate": 1786149044052,
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
      }
    ]
  }
}