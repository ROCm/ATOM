# Qwen3.8-Flash-Next on SGLang-ATOM

Flash uses `qwen4_exp` / `Qwen4ExpForConditionalGeneration`. Target and MTP
compute run through Native ATOM `Qwen4Exp` and `Qwen4ExpMTP`; the plugin
adapts SGLang scheduling, QSA KV pools, PLE state and HC hidden storage.
It requires the Native Flash implementation in this tree and SGLang 0.5.17.
The recognition patch supplies Flash config registration for that version.

## Text serving with MTP

The following configuration is tested on MI308X with TP2/EP2 and the
Qwen3.8-Flash-Next-PTPC-FP8 checkpoint. Set `MODEL_PATH` to the local checkpoint.
Target verification uses CUDA graphs and overlap scheduling. Draft graphs
remain disabled because Native Flash MTP uses mRoPE.

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-PTPC-FP8
export CUDA_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export SGLANG_AITER_KV_CACHE_LAYOUT=nhd
export AITER_MOE_PADDING_SIZE=128

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --tp 2 --ep-size 2 --attention-backend aiter \
  --kv-cache-dtype bf16 --page-size 64 \
  --context-length 8192 --max-running-requests 8 \
  --mem-fraction-static 0.70 --disable-radix-cache \
  --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2 --speculative-eagle-topk 1 \
  --sampling-defaults openai
```

SGLang and ATOM must both be installed or available on `PYTHONPATH`.
The target and draft share the same checkpoint, including its `mtp.*` weights.
Flash MTP supports one QSA draft layer and top-k 1; separate draft checkpoints
and other speculative algorithms are rejected. For eager comparison, replace
the two CUDA graph flags with `--disable-cuda-graph` and add
`--disable-overlap-schedule`. To serve without MTP, omit the speculative flags.

A deterministic smoke request:

```bash
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The result of 17 + 25 is", "sampling_params":{"temperature":0,"top_k":1,"top_p":1,"max_new_tokens":64}}'
```

## Boundaries

- Use BF16 KV and page size 64. TP greater than 1 requires EP because the
  expert intermediate width is 640.
- Leave PLE embeddings in Native ATOM; do not enable PLE embedding offload.
  The memory fraction above leaves room for plugin-owned QSA indexer caches.
- Validation covers text requests, concurrent mixed lengths, long prompts,
  slot reuse, and eager fallback above the largest captured batch followed by
  graph replay. It does not establish
  multimodal MTP, radix caching, performance gains, or correctness at other
  parallel sizes and speculative horizons.
- On a SGLang upgrade, review recognition and MTP EntryClass mapping separately.
  Upstream Flash recognition does not replace Native ATOM compute or the
  plugin's graph padding and post-draft metadata refresh contracts.
