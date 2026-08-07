window.BENCHMARK_DATA = {
  "lastUpdate": 1786102457714,
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
      }
    ]
  }
}