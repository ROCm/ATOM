# MiniMax-M3 OAI Conformance: ATOM native status & fixes

> **Test suite**: `m3_format_check/m3_text_tests.py` from the external
> `MiniMax-Provider-Verifier` repo. It validates the OpenAI
> `/v1/chat/completions` **protocol contract** (not model quality); text
> modality is ~138 cases.
>
> **System under test (SUT)**: ATOM native OpenAI server
> (`python -m atom.entrypoints.openai_server`, code under
> `atom/entrypoints/openai/`) — **not** the vllm-atom / sglang-atom plugin paths.
>
> **Current status**: on text, ATOM native passes roughly **55%** of cases.
> Failures cluster into **7 root causes**; the single largest is that ATOM does
> not parse MiniMax's tool-call format (fixing it alone recovers ~40 cases).
> This document records, for each root cause: symptom → where it lives in the
> code → fix direction, so that ATOM native can pass MiniMax's external check.

---

## 0. Reproducing the test

### 0.1 Start the ATOM native server (MiniMax-M3-MXFP4, per `recipes/MiniMax-M3.md`)

```bash
model_path=/shared/data/amd_int/models/MiniMax-M3-MXFP4   # or amd/MiniMax-M3-MXFP4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8010 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","vision_tower","multi_modal_projector","patch_merge_mlp","*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
```

> Two of these flags directly affect the results (see Root Cause 2 / 3):
> - `--max-model-len 32768` → only a 32K context window; long-context cases are rejected.
> - `--no-enable_prefix_caching` → prefix cache disabled; `cached_tokens` is always 0.

### 0.2 Run the tests

```bash
# from MiniMax-Provider-Verifier/m3_format_check
pip install -r requirements.txt

export M3_BASE_URL="http://localhost:8010"   # without the /v1/chat/completions suffix
export M3_AUTH_TYPE="none"                     # ATOM native has no auth; MUST be "none" or conftest raises UsageError
export M3_MODEL="/shared/data/amd_int/models/MiniMax-M3-MXFP4"   # must match the server --model exactly

pytest m3_text_tests.py                        # all text cases
pytest m3_text_tests.py -k TestToolCallBasic -v  # a single module
```

- The first stdout line prints the jsonl log path: `[m3_api_test] run log → .../logs/run_<UTC>.jsonl`.
- The jsonl records the full request + response + status + trace_id per call — check it first when triaging failures.

> Per-case descriptions live in the suite's `m3_format_check/docs/m3_text_cases_en.md`.

---

## 1. Module overview & expected results on ATOM native

| Module | Theme | Cases | Expected on ATOM native | Root cause |
|:---:|:---|:---:|:---|:---:|
| 01 basic_text | Basic non-stream | 3 | ✅ all pass | — |
| 02 sse_stream | SSE stream fields | 6 | 5 pass, **02_04 fails** | 6 |
| 03 multiturn | Multi-turn context | 2 | ✅ all pass | — |
| 04 thinking | thinking toggle | 4 | 3 pass, **04_01 fails** | 5 |
| 05 sampling | temperature/top_p/seed | 3 | ✅ all pass | — |
| 06 max_tokens | max_tokens bounds | 13 | 10 pass, **06_08 / 06_09×2 fail** | 4 / 2 |
| 07 message_format | Message-format edges | 12 | ✅ all pass | — |
| 08 model_compat | Model-name compat | 1 | ✅ pass (mini xfail) | — |
| 09 response_format | JSON output | 3 | — suite-side SKIP | — |
| 10 usage_field | usage semantics/math/cache | 8 | 6 pass, **10_04 / 10_05 fail** | 3 / 1 |
| 11 role_root | role=root & identity | 8 | ~2 pass, **11_02 / 11_04 fail** | 7 |
| 12 text_semantic | Text semantic adherence | 6 | ✅ all pass | — |
| 13 tool_call_basic | Tool-call basics | 12 | ❌ nearly all fail | 1 |
| 14 tool_call_schema | Tool schema | ~8 | ❌ all fail | 1 |
| 15 tool_call_combo | Tool combinations | ~11 | 2–3 pass, rest fail | 1 |
| 16 tool_call_edge | Tool edge/errors | ~24 | ~14 pass, ~10 fail | 1 / 4 |
| 17 param_stress | Long conv / long context | 16 | 8 pass, 8 fail | 2 |
| 18 reasoning_split | reasoning_split | 2 | ✅ pass | — |
| 19 finish_reason | finish_reason | 2 | 1 pass, **19_01 fails** | 1 |
| 20 error_codes | Error codes | 8 | 3 pass, **5 fail** | 4 |

---

## 2. Seven root causes: symptom → location → fix

Each section is one failure class. **Code-level** = requires an ATOM code change;
**Config-level** = a launch-flag change is enough.

### Root cause 1 — MiniMax tool-call format is not parsed (largest, ~40 cases) [code-level, top priority]

**Symptom**: modules 13/14 almost entirely fail, 15/16 fail about half, plus
`10_05` and `19_01`. In the logs the model does emit a tool call, but the
response's `message.tool_calls` is empty, `finish_reason` is `stop` instead of
`tool_calls`, and the raw tool text is left inside `content`.

**Location**: MiniMax-M3's chat template makes the model emit tool calls as:
```
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Beijing</parameter>
</invoke>
</minimax:tool_call>
```
But `atom/entrypoints/openai/tool_parser.py` only recognizes two formats:
- Qwen: `<function=...>` (`_is_qwen_xml`, `tool_parser.py:74`)
- Kimi: `<|tool_calls_section_begin|>` (`tool_parser.py:206`)

`<minimax:tool_call>` / `<invoke name=` matches neither, so
`parse_tool_calls()` returns the whole span as plain content and `tool_calls`
is never populated.

In addition, `tool_choice` is accepted by `protocol.py:115` but never used in
`atom/entrypoints/openai/api_server.py`, so `13_08`'s `required`/`auto`
enforcement cannot be satisfied.

**Fix direction**:
1. Add the MiniMax format to `tool_parser.py`:
   - Detection: text contains `<minimax:tool_call>` or `<invoke name=`.
   - Non-stream parse: extract each `<invoke name="X">` block, assemble its
     `<parameter name="k">v</parameter>` entries into arguments, and coerce each
     value to the JSON-Schema type declared in the request `tools` (reuse the
     existing `_build_param_types` / `_coerce_param_value` — it is nearly
     isomorphic to the Qwen XML handling).
   - Stream: mirror the existing `_process_qwen` / `_flush_qwen` to add a
     MiniMax branch in `ToolCallStreamParser` (buffer the
     `<minimax:tool_call>...</minimax:tool_call>` block, and on close emit
     `tool_call_start` / `tool_call_args` / `tool_call_end`).
2. Make `api_server.py` honor `request.tool_choice`: `none` skips tool parsing;
   `required` / a named function constrains accordingly.
3. Following the `CLAUDE.md` **Fix-then-sweep** rule, grep to ensure all three
   paths are covered (`serving_chat.py` and the anthropic branch in
   `api_server.py` share the same parser).

> Highest leverage: this one fix recovers ~40 cases (module 13, 14, most of 15,
> 16_11, 10_05, 19_01).

---

### Root cause 2 — 32K context window rejects long-context cases [config-level]

**Symptom**:
- `17_03` (512k, strict 200) → 400 → FAIL ×4
- `17_05` (~523k, strict 200), `17_07` (~1.02M, strict 200) → 400 → FAIL ×4
- `06_09` (max_tokens=512000/524288, strict 200) → 400 → FAIL ×2
- `17_04` (~553k) / `17_06` (~1M) are lenient (only require "no 5xx"), 400 < 500 → ✅ PASS

**Location**: the recipe uses `--max-model-len 32768`;
`api_server.py:_validate_context_length` (around `api_server.py:181`) raises
`ValueError → 400` when `prompt_tokens + max_tokens > max_model_len`. The tests
assert against M3's advertised 512k/1M window.

**Fix direction**:
- If memory allows, raise `--max-model-len` to ≥ `1048576` (1M) and re-run.
- If the hardware genuinely caps at 32K, these strict cases are a deployment
  choice rather than a contract defect; mark them as "known N/A under
  max-model-len" in the report.

---

### Root cause 3 — prefix caching off → cached_tokens always 0 [config-level]

**Symptom**: `10_04` (send the same prompt twice, assert the second call's
`cached_tokens > 0`) → always 0 → FAIL. (`10_02`'s `cached <= prompt` passes.)

**Location**: the recipe passes `--no-enable_prefix_caching`. `cached_tokens`
is forwarded from `RequestOutput.num_cached_tokens`
(`api_server.py:_send_stream_chunk_direct` / `generate_async`); with the cache
off it stays 0.

**Fix direction**: drop `--no-enable_prefix_caching` and re-run `10_04`. After
enabling, also run 01/02/12 to confirm no regression on the quantization /
sparse-attention paths.

---

### Root cause 4 — missing error-code / input validation (half of module 20, part of 16) [code-level]

The suite's `assert_error` requires an **exact** status match (400 must be 400;
422/500 count as failures).

| Case | Expected | ATOM native actual | Result | Fix direction |
|:---|:---|:---|:---:|:---|
| 20_05 no Authorization | 401 | **no auth on the server** → 200 | ❌ | add optional API-key auth |
| 20_07 invalid API key | 401 | no auth → 200 | ❌ | same |
| 20_03 temperature=5.0 | 400 | `SamplingParams` does **not** validate temperature (`sampling_params.py:24` checks only top_p/top_k/n) → 200 | ❌ | add a range check (0–2) |
| 06_08 max_tokens=-1 | 400/422 | no negative check (`max(0,-1)` bypasses `_validate_context_length`) → 200 | ❌ | validate `max_tokens >= 1` |
| 20_01 empty messages | 400 | template still yields a generation prompt → 200 | ❌ | make `get_messages()` raise ValueError on empty |
| 20_06 invalid role | 400 | template silently drops unknown roles → 200 | ❌ | validate role against a whitelist |
| 16_01/16_07 tool_result=object | 400 | pydantic validation fails → **422** (not 400) | ❌ | custom handler, or relax the expectation |
| 16_08 tool_call_id mismatch | 400 | no validation → 200/500 | ❌ | validate tool-message structure |
| 16_09 partial tool reply | 400 | no validation → 200 | ❌ | same |
| 16_12 invalid JSON arguments | 400 | template calls `.items()` on a string → 500 | ❌ | validate arguments are parseable up front |
| 20_02 invalid model | 400/404 | `validate_model()` → **400** | ✅ | — |
| 20_04 top_p out of range | 400 | `SamplingParams.__post_init__` raises ValueError → **400** | ✅ | — |
| 20_08 content moderation | 400/200 | 200 | ✅ | — |

**Fix direction**: add an input-validation layer at the `api_server.py` request
entry that maps invalid input to 400/401, rather than letting it reach the
template/engine and become 200/422/500. Note:
- FastAPI/pydantic validation failures return **422** by default; a custom
  `RequestValidationError` handler is needed to match OAI's 400.
- Make auth an optional API-key toggle (only enforce when a key is configured)
  so no-auth deployments are unaffected.
- If a deployment does not require OAI-strict error codes, this group can be
  marked "leniently tolerated" instead of changing code.

---

### Root cause 5 — the `thinking` extension field is ignored [code-level]

**Symptom**: `04_01` (`thinking:{type:disabled}` → assert no thinking signal) →
reasoning_content is still produced → FAIL. `04_02/03/04` and `18_01` pass
because they only require 200/400.

**Location**: `ChatCompletionRequest` uses `model_config={"extra":"ignore"}`
(`protocol.py:98`), so M3's `thinking` field is dropped; the M3 template always
appends `<think>`, so the model always thinks.

**Fix direction**: map the request's `thinking.type` to the chat-template toggle
in `api_server.py` (merge into `merged_kwargs`/`chat_template_kwargs`); when
`disabled`, do not inject `<think>` (the exact kwarg name depends on the M3
template). A temporary workaround is launching with
`--default-chat-template-kwargs '{"enable_thinking": false}'` (if the template
supports that kwarg), but that is a global default and cannot be toggled
per-request.

---

### Root cause 6 — stream usage chunk lacks a `choices` field [code-level, smallest change]

**Symptom**: `02_04` (assert every non-`[DONE]` chunk carries `id/choices/object`)
→ FAIL. The usage chunk is emitted **unconditionally**, so even a plain stream
(without `include_usage`) fails 02_04.

**Location**: the trailing usage-only chunk in `serving_chat.py` carries only
`{id, object, model, usage}` — no `choices` key (near `serving_chat.py:165`).
The OpenAI spec expects this terminal chunk to carry `choices: []`.

**Fix direction**: add `"choices": []` when building `usage_chunk` in both
`stream_chat_response` and `stream_chat_response_fanout` in `serving_chat.py`.
Zero-risk, and it also makes 02_02/05/06 more spec-compliant.

---

### Root cause 7 — role=root not honored by the template (module 11 identity cases) [code-level / template-dependent]

**Symptom**: `11_02` (root overrides system, strict full-string match on
`MiniMax-M3-taoxi`) and `11_04` (root-only identity) → FAIL; `11_01` (only
requires that root is accepted + 200) passes; `11_03` (system-only identity)
depends on whether the model repeats the identity string verbatim — at risk.

**Location**: the MiniMax template's role branches cover only
`system/user/assistant/tool`; an unknown `root` role is silently dropped, so the
identity instruction carried in root is lost.

**Fix direction**: confirm whether the loaded M3 chat template supports the
`root` role; if not, map `root` to a high-priority system message before/inside
`ChatMessage.to_template_dict()` (this is a protocol extension — align with the
M3 spec before changing it).

---

## 3. Result summary (text, ~138 cases)

- ✅ **Passing (~84)**: all of 01, 03, 05, 07, 08, 12; 02 (except 02_04),
  04 (except 04_01), 06 (except 06_08/06_09), 10 (except 10_04/10_05), the
  "200-only" cases in 16, the lenient cases in 17, 18, 19_02, and 20_02/04/08.
- ❌ **Failing (~65)**, by root cause:
  - **RC1 tool parsing**: all of 13, all of 14, most of 15, 16_11, 10_05, 19_01 — **~40 cases**.
  - **RC2 32K window**: 17_03×4, 17_05×2, 17_07×2, 06_09×2.
  - **RC3 no cache**: 10_04.
  - **RC4 error codes / validation**: 20_01/03/05/06/07, 06_08, 16_01/07/08/09/12.
  - **RC5 thinking**: 04_01.
  - **RC6 stream usage**: 02_04.
  - **RC7 role=root**: 11_02, 11_04 (11_03 at risk).

---

## 4. Suggested fix priority

**Code-level (by leverage):**
1. **RC1**: add MiniMax parsing to `tool_parser.py` + make `tool_choice` effective — recovers ~40 cases, top priority.
2. **RC6**: add `choices: []` to the usage chunk — smallest change.
3. **RC5**: pass `thinking` through to chat_template_kwargs.
4. **RC4**: unified input validation at the request entry + optional API-key auth.
5. **RC7**: map role=root (align with the M3 spec).

**Config-level (no code change):**
6. **RC2**: raise `--max-model-len` to ≥ 1.05M (when memory allows).
7. **RC3**: drop `--no-enable_prefix_caching`.

---

## 5. Triage checklist (start here when a run fails)

1. Is the server actually up? `curl /v1/models` returns; `rocm-smi --showmemuse`
   shows VRAM > 0 (a `/health` 200 alone is not enough).
2. Is `M3_AUTH_TYPE=none` set? (Without it, conftest raises UsageError and
   nothing runs.)
3. Does `M3_MODEL` exactly match the server `--model`? (Mismatch → all 400.)
4. Open this run's `logs/run_*.jsonl`, look up the failing case's `case_id`, and
   inspect `request/response/status/trace_id` — ~90% map directly to one of the
   7 root causes above.
5. Tool-call failures: check whether the response `content` still contains raw
   `<minimax:tool_call>` text → confirms RC1.
6. Long-context failures: check whether the status is 400 with a body mentioning
   `maximum context length` → confirms RC2.

---

> This document is based on static analysis of ATOM native
> `atom/entrypoints/openai/` plus a check of the MiniMax-M3 chat template's
> tool-call format; exact numbers should be confirmed against a live jsonl log.
> Line numbers reflect HEAD at writing time and may drift after refactors.
