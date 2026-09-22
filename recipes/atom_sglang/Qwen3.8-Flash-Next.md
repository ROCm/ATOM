# Qwen3.8-Flash-Next on SGLang-ATOM (MI308, first knife)

Text-only eager serving. Native compute from ATOM PR **#2048**. GDN / graph
sentinel from **#2067**. This plugin only translates `ForwardBatch` → Native
QSA + PLE metadata.

**Not Qwen3.5.** `Qwen4ExpForConditionalGeneration` / `qwen4_exp` is a third
line. Do not hang Flash on `Qwen3_5*` EntryClass.

## Checkpoint

```
/data/pretrained_model/Qwen/Qwen3.8-Flash-Next-FP8
```

Architecture: `Qwen4ExpForConditionalGeneration`. Quant: FP8. KV must be **bf16**.

## MI308 (this machine)

- 8x gfx942, **~192 GB** HBM each (not 288 GB MI355).
- Do **not** copy the 355 TP1 `--gpu-memory-utilization 0.98` command.
- TP>1 **must** use expert parallel (`moe_intermediate_size=640`).
- `page-size` / `block-size` divisible by `indexer_compress_ratio` (4).
- Leave PLE in Native. Do **not** pass `--ple-offload-embedding` (avoids ~102 GB double alloc).
- gfx950 FP8 prefill MHA from #2067 will not run here; ignore it.

Suggested first launch (2-GPU, short context):

```bash
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

python -m sglang.launch_server \
  --model-path /data/pretrained_model/Qwen/Qwen3.8-Flash-Next-FP8 \
  --tp 2 --enable-expert-parallel \
  --kv-cache-dtype bf16 \
  --page-size 64 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --mem-fraction-static 0.75 \
  --disable-cuda-graph \
  --trust-remote-code
```

Native-only smoke (no SGLang plugin), after #2048 is on the tree:

```bash
python -m atom.entrypoints.openai_server \
  --model /data/pretrained_model/Qwen/Qwen3.8-Flash-Next-FP8 \
  -tp 2 --enable-expert-parallel --kv-cache-dtype bf16 \
  --max-model-len 2048 --gpu-memory-utilization 0.75
```

## Out of scope (first knife)

MTP, VLM, radix, speculative, tc_piecewise, CUDA graph, gfx942 kernel rewrites.
