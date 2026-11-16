# MiniMax-M3 (428B): GSM8K 87.3%, 7.4 points below the known-good 94.7% (455-CI 2026-11-14)

## Environment

- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114`
- ATOM `main@9fec99786b088168aec7729858c1176e99d818e6`, aiter `main@b6f1ff6b7f3754aca8cc8c403b5fcd7a258eba47`; the first 455-CI run of this model
- Machine: 4 x MI455X (TP4); weights: `amd/MiniMax-M3-MXFP4` with the EAGLE3 draft `Inferact/MiniMax-M3-EAGLE3-GQA`

## Steps

1. Start the test container as in `recipes/455-CI-recipe.md`.
2. Start the server: [server_cmd.sh](server_cmd.sh) (TP4, EAGLE3 with 3 draft tokens, PTPC FP8 outside the MoE, FP8 KV cache).
3. Run GSM8K: [gsm8k_cmd.sh](gsm8k_cmd.sh) (lm_eval `local-completions`, 3-shot).

## Expected vs actual

- Expected: 94.7% flexible-extract / 94.8% strict-match, the validated MXFP4 + EAGLE3 TP4 result (`fp4_eagle_tp4`) in [recipes/MiniMax-M3.md](https://github.com/ROCm/ATOM/blob/main/recipes/MiniMax-M3.md) on ATOM main.
- Actual: 87.3% flexible-extract / 87.0% strict-match ([results](gsm8k/results_2026-11-14T14-25-16.684808.json)), one run.

## What we checked

- The known-good run used lm_eval `local-chat-completions`, 5-shot with `--fewshot_as_multiturn` and up to 16,384 new tokens; 455-CI uses `local-completions`, 3-shot, with lm_eval's default generation length. The method moves other models too, but less: GLM-5.2 scored 92.2% here against 96.4% (chat 5-shot) in its recipe, 4.2 points. So the 7.4 points here are probably not all method.
- Open and recently merged PRs in ROCm/ATOM and ROCm/aiter for "MiniMax-M3 accuracy", "MiniMax M3 gsm8k" and "minimax gfx1250": none about this.

## Next step

Run the recipe's own GSM8K (5-shot chat, the command in recipes/MiniMax-M3.md) against the same image and server. Near 94.7%: the gap is the eval method, and 455-CI needs a known-good value measured its own way. Well below it: a real accuracy regression on main or on gfx1250.
