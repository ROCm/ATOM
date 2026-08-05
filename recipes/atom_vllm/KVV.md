# KVV Run — Commands

Env: local vLLM for Kimi-K3 at `http://localhost:8000/v1`.
Two Python envs: `/opt/venv` (pytest + BEAM), project `.venv` (inspect-ai benchmarks).

Per-suite flags follow the **authoritative README examples**: only
`tests/tool_call_json_schema/` passes `--thinking` (and runs parallel with
retries); `params`/`k3_features`/`prompt_tokens` do **not** pass `--thinking`.
Several `k3_features` and `tool_call_json_schema` tests use
`@pytest.mark.flaky(reruns=...)`, so `pytest-rerunfailures` must be installed
or those retries are silently skipped.

### Prerequisites (in `/opt/venv`)
```bash
/opt/venv/bin/pip install tblib pytest-xdist pytest-rerunfailures
```
- `tblib` — required by `tests/conftest.py` (import fails without it).
- `pytest-xdist` — provides `-n 4` (parallel) for the tool-call schema suite.
- `pytest-rerunfailures` — provides `--reruns` and activates the in-repo
  `@pytest.mark.flaky` decorators.

## Clone KVV

```bash
git clone git@github.com:MoonshotAI/Kimi-Vendor-Verifier.git
```

## Server
Check server launch commands in recipe.

## 1. tests/params/

```bash
cd ~/Kimi-Vendor-Verifier
/opt/venv/bin/pytest tests/params/ \
  --base-url http://localhost:8000/v1 --api-key EMPTY \
  --smoke-model "/workspace/shared/data/amd_int/models/Kimi-K3" \
  --think-mode opensource
```

## 2. tests/k3_features/

```bash
cd ~/Kimi-Vendor-Verifier
/opt/venv/bin/pytest tests/k3_features/ \
  --base-url http://localhost:8000/v1 --api-key EMPTY \
  --smoke-model "/workspace/shared/data/amd_int/models/Kimi-K3" \
  --think-mode opensource
```

## 3. tests/prompt_tokens/

```bash
cd ~/Kimi-Vendor-Verifier
/opt/venv/bin/pytest tests/prompt_tokens/ \
  --base-url http://localhost:8000/v1 --api-key EMPTY \
  --smoke-model "/workspace/shared/data/amd_int/models/Kimi-K3" \
  --think-mode opensource
```

## 4. tests/tool_call_json_schema/
Authoritative README invocation: thinking on, parallel (`-n 4`), and retries
(`--reruns 3`) to absorb stochastic tool-emission flakiness.

```bash
cd ~/Kimi-Vendor-Verifier
/opt/venv/bin/pytest -n 4 tests/tool_call_json_schema/ \
  --base-url http://localhost:8000/v1 --api-key EMPTY \
  --smoke-model "/workspace/shared/data/amd_int/models/Kimi-K3" \
  --think-mode opensource \
  --thinking \
  --reruns 3 --reruns-delay 2 \
  --tool-json-report=reports/rerun_50228/tool-call-schema-report.json \
  -ra
```

## 5. OCRBench

```bash
cd ~/Kimi-Vendor-Verifier
KIMI_BASE_URL=http://localhost:8000/v1 KIMI_API_KEY=EMPTY \
.venv/bin/python eval.py ocrbench \
  --model "opensource//workspace/shared/data/amd_int/models/Kimi-K3" \
  --max-tokens 16384 --thinking --think-mode opensource --stream \
  --max-connections 50 --temperature 1.0 --top-p 0.95 --thinking-effort high
```

## 6. MMMU Pro Vision

```bash
cd ~/Kimi-Vendor-Verifier
KIMI_BASE_URL=http://localhost:8000/v1 KIMI_API_KEY=EMPTY \
.venv/bin/python eval.py mmmu \
  --model "opensource//workspace/shared/data/amd_int/models/Kimi-K3" \
  --max-tokens 98304 --thinking --think-mode opensource --stream \
  --max-connections 50 --temperature 1.0 --top-p 0.95 --thinking-effort high
```

BEAM and Judge are low priority tests. Let's ignore them for now.

## 7. BEAM 1M (smoke)

Generate:

```bash
cd ~/Kimi-Vendor-Verifier/beam
/opt/venv/bin/python beam_generate.py \
  --model /workspace/shared/data/amd_int/models/Kimi-K3 \
  --base-url http://localhost:8000/v1 --api-key EMPTY \
  --temperature 1.0 --top-p 0.95 --max-tokens 32768 \
  --thinking-json '{"chat_template_kwargs": {"thinking": true, "preserve_thinking": true, "thinking_effort": "high"}}' \
  --concurrency 4 --max-retries 3 --limit 8 \
  --output answers_smoke.jsonl
```

Judge (self-judge via same local server):

```bash
cd ~/Kimi-Vendor-Verifier/beam
/opt/venv/bin/python beam_judge.py \
  --answers answers_smoke.jsonl \
  --judge-model /workspace/shared/data/amd_int/models/Kimi-K3 \
  --judge-base-url http://localhost:8000/v1 --judge-api-key EMPTY \
  --judge-max-tokens 16384 --judge-temperature 0.3 \
  --concurrency 16 \
  --output scores_smoke.jsonl
```
