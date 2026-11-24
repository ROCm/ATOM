> [Examples README](../README.md) > Running from the release image

# 8. Running the eight examples from the release image

> 中文版：[08-release_cn.md](08-release_cn.md)

This chapter runs the eight examples in §8.2 from the **release container image**:
the software stack is pinned and the aiter kernels are already compiled, so each
example is a single command with nothing to install. Chapters
[1](01-env-setup.md)–[4](04-launching.md) are the other path — building the
environment from source — which is what you need in order to swap models or run two
nodes (example 8, see [chapter 7](07-disaggregated-rdma.md)).

> ⚠️ **This image supports AMD gfx950 only** (Instinct MI350X / MI355X) and requires
> 8 cards. See §8.3.1.

```bash
git clone https://github.com/ROCm/ATOM.git
cd ATOM/atomrl
docker pull rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119
export DATA_ROOT=/path/to/data
bash examples/run.sh prepare 1        # models and data, once (§8.4.3)
bash examples/run.sh 1                # run example 1 and judge it against §8.5.1
```

The image is built from this repository (§8.4.2 builds the same one locally). To switch
examples, change the final digit; `bash examples/run.sh suite --tier t1`
runs all eight one after another. `examples/run.sh` is the launcher for every example
in this repository; [harness/README.md](../harness/README.md) describes it in full.

The image supplies the environment; the checkout you just cloned supplies the code. The
launcher mounts it, so editing `atomrl/` and running the same command again runs the
edit — no rebuild, no new tag. `--atomrl-src /other/checkout` runs code from elsewhere.

---

## 8.1 What is in the image

| | |
|---|---|
| Task | DAPO math RL (GRPO-style, per-uid group normalization) |
| Models | Qwen3-8B-Base (dense), Qwen3-30B-A3B-Base (MoE, 128 experts) |
| Training backends | Lumen FSDP2 (BF16 / FP8 blockwise2d), Megatron-Native (EP=8) |
| Rollout engines | vLLM 0.31.1.dev0+gdb9527a46.d20261119.rocm724 (BF16 / `fp8_per_block`), ATOM (BF16 / `per_block_fp8`) |
| Topology | 8 training actors + 8 co-located rollout replicas (TP=1) inside one Ray driver |
| Weight sync | ZMQ CUDA-IPC, same-device transfer, with coverage assertions |
| Hardware | **AMD gfx950 only (MI350X / MI355X), 8 cards** |

On the algorithm side: clip-higher + dual-clip + token-mean policy loss, dynamic
sampling (`filter_groups`), an overlong reward buffer, and TIS rollout correction.

The image carries the kernels its build warms up already compiled; the others compile
on first use, in the first minutes of the first run in a fresh container:

```bash
docker run --rm --entrypoint /bin/bash rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119 \
  -lc 'echo $ATOM_COMMIT; ls /opt/atomrl/aiter-jit/*.so | wc -l'
```

`ATOM_COMMIT` is the commit of this repository the image was built from. ATOM, AtomRL
and Lumen sit under `/opt/atomrl/ATOM`, while AITER is under `/app/aiter-test`;
`run.sh` mounts your checkout over `/opt/atomrl/ATOM/atomrl`.

The references in §8.5.1 were first measured on a legacy Lumen-RL release
image. That private image is retained only as historical provenance and is not
part of the public reproduction path.

### 8.1.1 Pinned versions

The release build copies ATOM, AtomRL, and Lumen from the current checkout. AITER and
MORI come from their official ROCm repositories at the exact commits in
`release/versions.env`. A result is therefore identified by the image digest and the
current checkout commit. Every run prints both.

| Component | Repository | Branch | Commit |
|---|---|---|---|
| ATOM + AtomRL | current repository | current branch | your checkout |
| Lumen | `atomrl/lumen` | in-tree | current checkout |
| AITER | `ROCm/aiter` | detached | `c813348adc36` |
| MORI | `rocm/atom-dev:vllm-latest` | installed package | `amd_mori_nightly 1.2.6.dev20261119` |

Both external dependencies are checked out by SHA before their submodules are
initialized. Docker build arguments may override the repositories or SHAs explicitly;
the defaults never follow mutable `main` or `HEAD`.

⚠️ **Swapping ATOM yourself needs two settings that AtomRL supplies rather than
inherit**:

- `compilation_config.cudagraph_mode=FULL` (engine kwarg, from `_pin_cudagraph_mode`).
  Without it a no-eager ATOM rollout aborts on the first CUDA graph replay.
- `ATOM_FORCE_ATTN_TRITON=1` (environment variable; `run_dapo.sh` defaults it to 1 for
  every `atom*` `MODE`, and `0` turns it off). ATOM's assembly paged-decode kernel
  returns **finite but wrong** output when a sequence's context occupies exactly 16 pages
  with a partial last one, so a handful of tokens per run come back with an arbitrary
  logprob. It **hides from `abs_diff`** — the number and average size of disagreeing
  tokens do not move — and shows only in the quadratic `chi2_token`: example 4 step 3
  measured 5761.69 against 0.0138 the step before, while `abs_diff` went 0.0370 → 0.0473.
  With the variable set, the highest `chi2_token` over three runs and nine steps of the
  same config is 5.39.
  ⚠️ **Do not expect it to improve `k3_kl`.** At a 4k response the defect fires rarely —
  once in those nine steps — and the reference is read at **step 1**, while example 4's
  event was at step 3, so this table's metrics structurally cannot see it. Judge whether
  the variable is working by `chi2_token`. ATOM's own fix is not in the pinned commit
  yet; drop this once it is.

Both are inert on ATOM builds that predate the corresponding field. What the failures
look like and why is in the comments of
[`release/versions.env`](../../release/versions.env).

Sleep is **not** pinned: it follows ATOM's own default, which releases the rollout's
weights, graphs and KV pool. AtomRL used to force `sleep_keeps_memory_resident=true`
here; that pin was removed on 2026-09-16 after the failure it guarded against stopped
reproducing. Releasing costs one graph recapture per step, about 1 s (2–3%), and does
not move the published metrics.

The base image is the published `rocm/atom-dev:vllm-latest`, with PyTorch
`2.10.0+rocm7.2.4` and vLLM
`0.31.1.dev0+gdb9527a46.d20261119.rocm724`. The validated additions are
Megatron commit `8dd45f5a`, ROCm Apex `daed8525`, and ROCm
TransformerEngine `6e541a10`.
The full list is in [`release/versions.env`](../../release/versions.env).

The container prints the HEAD of all four source trees at startup. To verify the
software stack:

```bash
docker exec atomrl-ex-release bash -lc 'python3 -c "
import aiter, atomrl, atomrl.lumen, vllm, flydsl, transformers
print(vllm.__version__, flydsl.__version__, transformers.__version__)
print(aiter.__file__)"'
```

Expect vLLM `0.31.1.dev0+gdb9527a46.d20261119.rocm724`, with `aiter`
resolving under `/app/aiter-test/`.

### 8.1.2 MORI-EP requires the Megatron ROCm fork

The image's `megatron-core` is NVIDIA upstream and has no MORI backend. The eight
examples do not need it — they all use the default `alltoall` dispatcher. For
`moe_token_dispatcher_type=flex` with `mori`, pin the fork instead:

```bash
git clone --depth 1 -b core_r0.18.0_rocm \
  https://github.com/ROCm/Megatron-LM.git "$DATA_ROOT/megatron-rocm"

bash examples/run.sh 7 --env MEGATRON_PATH=$DATA_ROOT/megatron-rocm
```

`MEGATRON_PATH` is prepended to `PYTHONPATH`, so it must be visible inside the
container.

---

## 8.2 The eight examples

All eight run training *and* inference on the same 8 cards.

### 8.2.1 Overview

| # | Example | Training | Rollout | Command (alias of the id in `DAPO/examples.yaml`) |
|---|---------|----------|---------|---------|
| 1 | 8B BF16 baseline | FSDP2 BF16 | vLLM BF16 | `bash examples/run.sh 1` |
| 2 | 8B FP8 rollout | FSDP2 BF16 | vLLM `fp8_per_block` | `bash examples/run.sh 2` |
| 3 | 8B FP8 end-to-end | FSDP2 **FP8 blockwise2d** | vLLM `fp8_per_block` | `bash examples/run.sh 3` |
| 4 | 8B ATOM FP8 | FSDP2 **FP8 blockwise2d** | **ATOM** `per_block_fp8` | `bash examples/run.sh 4` |
| 5 | 8B ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash examples/run.sh 5` |
| 6 | MoE FSDP2 | FSDP2 BF16 | vLLM BF16 | `bash examples/run.sh 6` |
| 7 | MoE Megatron EP=8 | **Megatron** TP=PP=CP=1, EP=8, DP=8 | vLLM BF16 | `bash examples/run.sh 7` |
| 9 | MoE ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash examples/run.sh 9` |

**Example 8 is not in this chapter.** It is the two-node disaggregated RDMA
deployment of [chapter 7](07-disaggregated-rdma.md); `bash examples/run.sh 8` runs it
when the site file names two hosts (see [harness/README.md](../harness/README.md#sites)). The
numbering is shared across the whole examples set, so this chapter runs 1–7 and 9.

- Examples 2 and 3 share one config and differ only in `TRAIN_FP8`: `0` quantizes the
  rollout only, `1` puts the training forward pass on FP8 as well.
- Example 5 is example 4's BF16 control: the same ATOM engine with the rollout online
  quantization and the training-side FP8 both switched off.
- Example 7 is example 6's Megatron twin: the two configs are field-for-field identical
  apart from `training_backend` and `megatron_cfg`, and EP=8 gives DP=8 to match FSDP2,
  so the two metric sets can be subtracted and the difference is the training backend.
- Example 9 is example 6's ATOM twin, and the pair answers "what does switching the
  rollout engine cost?": same model, same training config, `generation_backend` vllm
  → atom plus an `atom_cfg` block, nothing else. It is also the only example that
  exercises MoE expert weights over the ATOM weight-sync path — see §8.5.2 on why
  `skipped` is a health criterion.

### 8.2.2 Full parameters per example

This table is what `examples/DAPO/examples.yaml` passes for each example. When using the
manual command (§8.4.5), every column on the row has to be supplied.

| # | `MODE` | `TRAIN_FP8` | `CONFIG_OVERRIDE` (all under `examples/DAPO/configs/`) | `STEPS` | `max_response_length` | Model | Extra env |
|---|---|---|---|---|---|---|---|
| 1 | `bf16` | `0` | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 2 | `fp8` | `0` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 3 | `fp8` | `1` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 4 | `atomfp8` | `1` | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | Qwen3-8B-Base | `ATOM_FORCE_ATTN_TRITON=1` |
| 5 | `atombf16` | `0` | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | Qwen3-8B-Base | `ATOM_FORCE_ATTN_TRITON=1` |
| 6 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `ATOMRL_FP32_MOE_ROUTER=0` |
| 7 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `ATOMRL_FP32_MOE_ROUTER=0` |
| 9 | `atombf16` | `0` | `dapo_qwen3moe_a3b_ray_atom_bf16_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `ATOMRL_FP32_MOE_ROUTER=0` `ATOM_FORCE_ATTN_TRITON=1` |

> ⚠️ **`MODE` and `CONFIG_OVERRIDE` must be given as a pair.** Besides selecting
> environment variables, `MODE` **appends a set of Hydra overrides**, and
> `CONFIG_OVERRIDE` only replaces the config file without cancelling them. The typical
> consequence of a mismatch: `MODE=atomfp8` unconditionally appends
> `compilation_config.level=3`, and combining that with a vLLM config yields
> `RuntimeError: aot_compile is not supported by the current configuration`.
> `examples/run.sh` already pairs them correctly, so this is not a concern when using it.

All eight configs are `logger.wandb_enabled: false`, so **no wandb account is needed**
(see §8.4.6). `STEPS` is the command-line override for `num_training_steps`. None of
the eight smoke configs writes a checkpoint, so the examples can be run back to back in
any order.

---

## 8.3 Requirements

### 8.3.1 Hardware and drivers

- **8x AMD gfx950** (Instinct MI350X or MI355X), all cards idle
- Host ROCm 7.2, with `/dev/kfd` and `/dev/dri` accessible
- Docker (if your user is not in the docker group, set `docker: sudo docker` in the site file, §8.4.4)

> ⚠️ **This image only runs on gfx950.** TransformerEngine and Apex are compiled with
> `NVTE_ROCM_ARCH=gfx950` / `PYTORCH_ROCM_ARCH=gfx950`, and the 25 aiter kernels
> compiled into the image were built on gfx950. Those JIT artifacts do not carry an
> architecture tag in their filenames, so on gfx942 (MI300X / MI308X / MI325X) they are
> loaded as-is and fail at runtime instead of being rebuilt. For gfx942, build
> separately: `PYTORCH_ROCM_ARCH=gfx942 bash release/build_image.sh`.

### 8.3.2 Disk

| Item | Measured |
|---|---|
| Image download size (compressed layers, summed from the registry manifest) | **13.4 GB** |
| Image unpacked on disk | **50.8 GB** |
| Lumen-RL image `260921b`: download / unpacked | 11.8 GB / 47.3 GB |

**Recommended budget**: about 65 GB for the image (unpacked plus the compressed layers
retained in the content store) plus 74 GB of models and data, so about **140 GB**; a
local build needs its build cache on top. All eight examples are smokes and write no checkpoints. A long run
(`--full`) needs checkpoint space on top — a single 30B-A3B FSDP2 checkpoint
(fp32 weights plus optimizer) is about 342 GB, and `save_total_limit` decides how many
are kept.

### 8.3.3 Models and data

The following must exist under `$DATA_ROOT`. This is exactly the list `examples/run.sh`
preflights (the `requires` of each example):

| Path (relative to `$DATA_ROOT`) | Size | Needed by |
|---|---|---|
| `models/Qwen3-8B-Base/` | 16 GB | examples 1–5, plus the tokenizer for all of them |
| `models/Qwen3-30B-A3B-Base/` | 57 GB | examples 6, 7, 9 |
| `data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet` | 1.02 GB | all (train) |
| `data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet` | 892 KB | all (val) |
| `logs/` | — | created by `examples/run.sh` |

To prepare them from scratch, see §8.4.3.

---

## 8.4 Usage

### 8.4.1 Confirm the cards are idle before starting

```bash
docker ps -a                                    # is someone else's container holding cards
rocm-smi --showmeminfo vram | grep -i used      # works directly on the host
```

All eight cards should sit at the **idle baseline of about 298 MB**
(297766912–297832448 B measured on MI355X). Anything above that means a co-tenant or an
orphan process from a previous run. `examples/run.sh` makes this a hard gate: it refuses
to start if any card is above 2 GB and says so; `--force` skips the gate.

### 8.4.2 Get the image

Pull it:

```bash
docker pull rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119
```

That is what `run.sh` uses. Or build the same image from this repository:

```bash
git clone -b main https://github.com/ROCm/ATOM.git
cd ATOM/atomrl
TAG=atomrl:release-base bash release/build_image.sh       # 45-60 min, mostly TransformerEngine
TAG=atomrl:release-base OUT_TAG=atomrl:release bash release/precompile_kernels.sh
```

and pass `--image release=atomrl:release` to `run.sh` (or set
`images: {release: atomrl:release}` in a site file).

`precompile_kernels.sh` needs a GPU: aiter kernels are only compiled on first use and
`docker build` has no devices, so they have to be compiled in a container with cards
attached and committed into the image. That script's synthetic warmup covers 5 of the
25 kernels; see its header for how to cover all 25.

### 8.4.3 Prepare the data

```bash
export DATA_ROOT=/path/to/data
bash examples/run.sh prepare 1 6       # 8B and 30B-A3B models, both datasets, filtered
```

`prepare` runs `examples/DAPO/tools/prepare_data.py` in the image. Models are reused only
when their `.download_complete` marker matches the immutable revision configured by that
script; legacy or mismatched directories are refreshed. `run.sh <N> --prepare` does the
same before a run when something in §8.3.3 is missing. What it does, step by step, inside
the container:

```bash
# 1) models and raw datasets
docker exec -e DATA_ROOT="$DATA_ROOT" atomrl-ex-release bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
from pathlib import Path
import os; D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  revision="49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
Path(f"{D}/models/Qwen3-8B-Base/.download_complete").write_text(
    "hf@49e3418fbbbca6ecbdf9608b4d22e5a407081db4")
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY'

# extra for examples 6, 7 and 9 (about 57 GB)
docker exec -e DATA_ROOT="$DATA_ROOT" atomrl-ex-release bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --revision 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8
printf %s hf@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 \
  > "$DATA_ROOT/models/Qwen3-30B-A3B-Base/.download_complete"'
```

```bash
# 2) filter out prompts longer than 1024 tokens, producing the two parquet files in §8.3.3
docker exec -e DATA_ROOT="$DATA_ROOT" atomrl-ex-release bash -lc '
python3 - <<PY
import os, glob, datasets
from transformers import AutoTokenizer
D = os.environ["DATA_ROOT"]; MAXLEN = 1024
OUT = f"{D}/data_cached/qwen3-8b-maxprompt1024"
tok = AutoTokenizer.from_pretrained(f"{D}/models/Qwen3-8B-Base")
def first(g): return sorted(glob.glob(g, recursive=True))[0]
jobs = [(first(f"{D}/raw/DAPO-Math-17k/**/*.parquet"), f"{OUT}/dapo-math-17k.filtered.parquet"),
        (first(f"{D}/raw/AIME-2024/**/*.parquet"),     f"{OUT}/aime-2024.filtered.parquet")]
os.makedirs(OUT, exist_ok=True)
nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
for src, dst in jobs:
    ds = datasets.Dataset.from_parquet(src); n0 = len(ds)
    ds = ds.filter(lambda d: len(tok.apply_chat_template(d["prompt"], add_generation_prompt=True,
                                                        tokenize=True)) <= MAXLEN, num_proc=nproc)
    ds.to_parquet(dst); print(src, "->", dst, n0, "->", len(ds))
PY'
```

> The data only has to be filtered once and is shared by all eight examples: the two
> models have identical `tokenizer.json` / `vocab.json` / `merges.txt` (vocab 151936),
> so a filter computed with the 8B tokenizer is valid for the MoE model too.
>
> **The MoE model must be the Base variant.** The instruct / thinking Qwen3-30B-A3B does
> not close `</think>` within `max_response_length`, so every sample is truncated,
> reward is stuck at -1, `filter_groups` comes up empty for 10 rounds, and the run
> raises `RuntimeError: filter_groups collected no valid groups`.
>
> For ModelScope mirrors (same repo IDs, same local paths) see
> [`03-data.md`](03-data.md).

### 8.4.4 The launcher

`examples/run.sh` runs on the GPU host (or on any machine that reaches the hosts named
in a site file). It checks that the image is there and the cards are idle, manages the
container, assembles every environment variable from `examples/DAPO/examples.yaml`,
follows the log, and judges the result against §8.5.1.

```bash
bash examples/run.sh list                 # every example, tier and hardware
bash examples/run.sh <id|1..7|9> [options]
bash examples/run.sh --help
```

The container is named `atomrl-ex-release` and created once; before every run the
launcher restarts it, because after a finished run each card may still hold about
90.9 GB (see §8.5.2), and it stops it afterwards. A container whose mounts or image no
longer match is recreated. The checkout running `run.sh` is mounted over the copy of
AtomRL in the image; leave that out (§8.4.5) and the container runs the copy baked into
the image instead, which is a different commit as soon as you change anything.

Logs are written under the data root and copied into the results directory:

```
$DATA_ROOT/logs/atomrl-<id>-<timestamp>.log            # trainer log
$DATA_ROOT/logs/atomrl-<id>-<timestamp>.launcher.log   # wrapper output and exit code
atomrl/output/regression/<timestamp>-<id>/             # summary.json, summary.md, junit.xml, logs
```

| Option | Effect |
|---|---|
| `--steps N` | override the number of training steps |
| `--full` | the example's long-run config instead of the smoke (see §8.4.6) |
| `--no-check` | judge by exit code only |
| `--dry-run` | print the commands that would be issued, run nothing (`show <id>` does the same with a description) |
| `--prepare` | prepare missing models and data first |
| `--force` | run on busy cards and remove other containers of the same image |
| `--keep-cache` | do not clear compile caches between ATOM examples (see §8.6) |
| `--keep-container` | leave the container running afterwards |
| `--verbose` | print the full log rather than the highlights |
| `--override K=V` | extra config override |
| `--env K=V` | extra container environment, e.g. `MEGATRON_PATH` (§8.1.2) |
| `--image release=REF` | image to use instead of the published release image (`@key` swaps in another image definition) |
| `--atomrl-src PATH` | AtomRL checkout to mount instead of this one |
| `--atom-src PATH` | ATOM checkout to mount over the image's ATOM |
| `--site FILE` | hosts, data root, docker CLI (`examples/site.example.yaml`) |

`DATA_ROOT` comes from the environment unless the site file sets `data_root`.

The other trees in the image are editable installs too. `--atom-src` mounts an ATOM
checkout over the image's; unlike AtomRL, ATOM, Lumen and aiter are what the reference
values are pinned to, so a result from a swapped tree is no longer comparable with
§8.5.1.

### 8.4.5 The manual command, without the launcher

`bash examples/run.sh show <N>` prints the complete command sequence for any example:
the `docker run` that creates the container and the `docker exec` that runs it. For
example 1 it is, in essence:

```bash
export DATA_ROOT=/path/to/data

docker run -d --name atomrl-ex-release \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$DATA_ROOT":"$DATA_ROOT" -e DATA_ROOT="$DATA_ROOT" \
  -v "$PWD":/opt/atomrl/ATOM/atomrl \
  rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119 sleep infinity

docker exec \
  -e RL_ROOT=/opt/atomrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base \
  -e LOG=$DATA_ROOT/logs/example-1.log \
  atomrl-ex-release bash -lc 'bash /opt/atomrl/ATOM/atomrl/examples/DAPO/run_dapo.sh'
```

`$PWD` is the root of this checkout (`atomrl/`). For the other examples, replace `MODE`,
`TRAIN_FP8`, `CONFIG_OVERRIDE`, `STEPS` and `MODEL_PATH` per the table in §8.2.2, then
add the environment variables in that table's last column.

The log does not go to stdout; `run_dapo.sh` writes straight to `$LOG`. Follow it with
`tail -f "$LOG"`, judge it with `bash examples/run.sh check <N> --log "$LOG"`, and run
`docker restart atomrl-ex-release` between examples; also clear the compile caches
between ATOM examples that differ in precision or model (§8.6).

Four things that are easy to miss:

- `CONFIG_OVERRIDE` is **relative to `$RL_ROOT/ATOM/atomrl`**; an absolute path is not found.
- Without `CONFIG_OVERRIDE`, `MODE` selects the **longrun** config
  (`wandb_enabled: true`, `max_response_length: 20480`), not the smoke one.
- `MODEL_PATH` defaults to the 8B model, so examples 6, 7 and 9 silently run the wrong
  model if it is not given.
- The empty value after `PYTORCH_CUDA_ALLOC_CONF=` is not a typo: only an explicitly
  empty string turns off `expandable_segments`.

### 8.4.6 wandb

| | smoke configs (the eight in §8.2.2) | longrun configs (`--full`) |
|---|---|---|
| `logger.wandb_enabled` | `false` | `true` |
| Account needed | **no** | yes, `WANDB_API_KEY` |
| `max_response_length` | 512 / 4096 | 20480 (4096 for example 7) |

So the eight examples in §8.2 need no wandb account. Only `--full` uses it:

```bash
WANDB_API_KEY=xxxx bash examples/run.sh 1 --full

# without a key the launcher appends logger.wandb_enabled=false itself
bash examples/run.sh 1 --full
```

A long run takes hours to days in the foreground; run it under `tmux` or `nohup`.

> The Hydra key is `logger.wandb_enabled`, not a top-level `wandb_enabled`; getting it
> wrong yields `ConfigKeyError: Key 'wandb_enabled' not in 'AtomRLConfig'`.
> A missing key fails *after* `RLTrainer.setup ... complete`, so the first few minutes
> look entirely normal.

---

## 8.5 Verifying the result

`examples/run.sh` performs the judgement described in this section after every run: it
extracts the step-1 metrics and compares them against the reference values in
`examples/DAPO/examples.yaml`, counts occurrences of `Traceback` / `OutOfMemory` /
`CUDA error` / `HSA_STATUS`, counts weight-sync buckets that skipped a tensor, checks that
the final step was logged, and reports PASS or FAIL. For reading the numbers yourself, see
below.

```bash
bash examples/run.sh 1                                              # run and judge
bash examples/run.sh check 1 --log $DATA_ROOT/logs/atomrl-dapo-qwen3-8b-vllm-bf16-xxx.log
```

### 8.5.1 Reference values

**Measurement conditions**: 8x MI355X (gfx950), image
`dapo-gfx950-rocm7.2.3-260921b` (digest `sha256:eede1d8fcdf5…`) with AtomRL at
`22502fa` — both matter, see §8.1.1 — the command being
`bash release/run_example.sh <N>`, the launcher of that time, which `examples/run.sh`
has since replaced with the same table and checks (equivalent to a full row of §8.2.2),
**`seed=10086`** (fixed inside `run_dapo.sh`), and metrics read at **step 1**
(`step=1`).

| # | config (`examples/DAPO/configs/`) | steps | resp | trainer-log span | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl` (signed) | runs |
|---|---|---|---|---|---|---|---|---|
| 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 158 s | **0.00110** ±30% | **0.628** ±25% | 0.00102 | 3 |
| 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 127 s | **0.00481** ±30% | **0.791** ±25% | 0.00467 | 3 |
| 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` (`TRAIN_FP8=1`) | 3 | 512 | 133 s | **0.00410** ±30% | **0.790** ±25% | 0.00408 | 3 |
| 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 467 s | **0.00399** ±50% | **0.599** ±50% | 0.00374 | 5 |
| 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 364 s | **0.000954** ±50% | **0.667** ±60% | 0.000899 | 3 |
| 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 524 s | **0.00144** ±50% | **0.620** ±60% | 0.00140 | 3 |
| 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 501 s | **0.00161** ±50% | **0.631** ±60% | 0.00161 | 3 |
| 9 | `dapo_qwen3moe_a3b_ray_atom_bf16_4k_smoke.yaml` | 3 | 4096 | 542 s | **0.00153** ±50% | **0.683** ±60% | 0.00149 | 3 |

The two bold columns with tolerances are what `examples/run.sh` turns into PASS / FAIL; each
reference is the mean over the number of runs in the `runs` column.

⚠️ **The whole table was re-measured on this image on 2026-09-21**, three runs per
example and five for example 4, where most rows previously had one. Every `k3_kl` moved
by between **−8.9% and +10.5%** against the reference it replaces — the new ATOM, the
rebased aiter and this round's changes did not measurably alter train/rollout alignment.
The point of re-measuring was the spread, not the values.

**The spread is what the extra runs bought.** The worst deviation of any run from its
own example's mean ranges from 2.0% (example 2) to 32.1% (example 4). Tolerances are
unchanged, and all 26 runs pass against this table.

⚠️ **Example 4 is the widest of them; do not tighten its tolerance to a single
observation.** Its five runs are 0.00271 / 0.00339 / 0.00426 / 0.00460 / 0.00499, a
factor of 1.8. Its reference before this round was 0.00287, and three runs mid-round sat
high enough to look like a systematic shift from `ATOM_FORCE_ATTN_TRITON` (§8.1.1); the
fifth run came back at 0.00271, which makes it spread rather than shift.

**The span column carries no tolerance and is not part of the verdict** — it is
dominated by how warm the caches are and has been measured up to ±15% apart on the same
machine. The launcher's end-to-end wall clock is 20–35 s longer.

**Result on this image: 26 runs over the eight examples, exit code 0 throughout**, all
four error counts **zero**, every weight-sync bucket at `skipped=0`, the check **26/26
PASS**, with every run's `k3_kl` within **±32.1%** of its example's reference — example
4 being the widest, as above. The per-run record is in
[`VALIDATION.md`](../../release/VALIDATION.md).

**Example 9 versus example 6 — what the rollout engine costs.** Same model, same
training config, ATOM instead of vLLM: `k3_kl` is 0.00138 against 0.00158, well inside
tolerance, so **switching the rollout engine does not move train/rollout alignment
measurably.** What it moves is time: 56.7 s per step against 106.7 s, about 1.9x faster,
paid for with 409 s of setup instead of 203 s. So a 3-step smoke reports ATOM as slower
end to end (579 s against 523 s) while the per-step figure is what matters for a real
run. Example 7 pairs the same vLLM rollout with a Megatron actor and lands at 103.0 s.

### 8.5.2 Criteria

- **`rollout_corr/k3_kl` is the primary criterion** — ±30% for the 512 group
  (examples 1/2/3), ±50% for the 4096 group (examples 4–7 and 9). It is non-negative and
  far steadier than `entropy`, so it is the metric to judge a reproduction on.
- **`entropy` is the secondary criterion** — ±25% for the 512 group, ±50% for the 4096
  group, ±60% for the three MoE examples (6, 7 and 9). It is a mean over the batch that
  survives `filter_groups`, so its variance is high, especially on MoE. **Judge MoE on
  `k3_kl`.**
- **`rollout_corr/kl` is only an order-of-magnitude criterion.** It is a signed mean, so
  only the absolute value is checked, against a band from one tenth to ten times the
  reference. **One order of magnitude above is what counts as wrong**, and the usual
  cause is model-sensitive RMSNorm not being enabled on one side.
- **`rollout_corr/ppl_ratio` is informational** and not part of the verdict.
- **Every weight-sync bucket must report `skipped=0`.** The four error counts cannot
  stand in for this one: a rollout that silently fails to update some of its weights
  exits 0, logs every step and raises nothing, so it reads as a slow accuracy regression
  rather than a fault.

  ```bash
  grep -oE 'bucket done - updated=[0-9]+, skipped=[0-9]+' <log> | sort | uniq -c
  ```

  A healthy run is `skipped=0` on every line. Examples 1, 2, 3, 6 and 7 use the vLLM
  path and print no such line at all, which also counts as clean.

> When the numbers do not match, **first confirm you ran the same config**:
> `grep -m1 'CONFIG=' $DATA_ROOT/logs/atomrl-<id>-*.launcher.log`
> prints the config, `MODE`, `TRAIN_FP8` and `STEPS` actually used. A config mismatch
> and a numerical regression look identical in the metrics, but the former is far more
> common.

---

## 8.6 Problem handling

**1. VRAM is not released when a run finishes.** After a smoke ends normally each card
may still hold about 90.9 GB: the Ray workers have exited but the memory has not been
returned to the driver. Restart the container between runs, otherwise the next run gets a
smaller KV cache budget. The launcher does this before every run.

```bash
docker restart atomrl-ex-release
```

**2. Switching ATOM precision *or model* requires clearing the compile caches.** The
torch inductor cache is not isolated per run, so going from example 4 straight to
example 5 (or back) fails in AOTAutograd. The launcher records the previous ATOM
precision and model (`compile_cache_key` in `examples.yaml`) and clears only when either
changed (`--keep-cache` disables this).

```bash
docker exec atomrl-ex-release bash -lc \
  'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
```

⚠️ **The model collides too, not just the precision.** ATOM keys the cache by
`mode/actor_id/replica/rank` and nothing else, so examples 5 and 9 — both `atombf16`,
different models — share one cache directory. Running 9 after 5 replays the 8B graph
against the 30B MoE and dies during engine init with
`assert_size_stride ... stride 2048==4096` (the two hidden sizes). It reads like a
corrupt engine rather than a stale cache. **Driving the examples yourself, clear the
caches between every pair that differs in precision or model.**

**3. Judge a long run's liveness from the log, not with `pgrep`.** Processes started via
`docker exec` do not share a process tree with your shell, so `pgrep` returns 0 across
sessions. **Watch whether the log file is still growing:**

```bash
watch -n 30 'ls -l $DATA_ROOT/logs/atomrl-dapo-qwen3-8b-atom-fp8-xxx.log'
```

**4. `docker restart` terminates a run in progress.** The launcher restarts the container
before every run, so do not start a second `examples/run.sh` against the same host while
one is still running; give the second one its own site file with another `container`
prefix if it must share the host.

**5. `waiting for baton release` in the log is not a hang.** The 8 training actors are
queued behind one of them finishing a JIT compile. It should not appear with this image,
where every kernel is precompiled, but can if you mount your own aiter source.

**6. `filter_groups round N` does not appear for every example.** Only configs with
dynamic sampling enabled emit it, and **examples 2 and 3 have it switched off** — at
`max_response_length: 512` a base model rarely finishes a problem, so dynamic sampling
would filter out every group. Both still complete all 3 steps and pass the check.

**7. When overriding the `aiter` source, change `AITER_JIT_DIR` too.** Compiled kernels
are bound to the aiter revision that produced them, and reusing the old directory fails
at import time with a message that mentions neither aiter nor the branch:

```
AttributeError: module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'
```

Add `-e AITER_JIT_DIR=/tmp/aiter-jit-<your-branch>`.

**8. `flydsl` must be upgraded together with `aiter`.** The base image ships 0.1.4.2
while the pinned AITER source requires `>= 0.2.4` (this image pins 0.3.2). A mismatch shows up
as the import-time error below, raised from ATOM's `model_ops/moe.py` with no mention of
aiter at all:

```
ImportError: Unsupported `flydsl` version: expected >=`0.2.4`, got `0.1.8`.
```

**9. FP8 training divergence** (very low entropy, `grad_norm` and `rollout_corr/kl` both
around 1e4): see [6. Troubleshooting](06-troubleshooting.md).

**10. A negative KV pool on the wake after a weight sync** means the ATOM rollout is
releasing its memory on sleep instead of keeping it resident. It only bites the MoE
example, and only after step 0 has already succeeded:

```
ATOM/atom/rollout/memory_manager.py:137  resume_memory
AssertionError: Not enough memory for KV cache with block size(16). At least 1 block
  (1.50MB) is required, but available_for_kv=-30805.98MB (budget=86.40GB,
   peak_torch=57.68GB, non_torch=52.88GB, safety=5.76GB, free=177.39GB)
```

`non_torch` is the rest of the node, mostly the colocated trainer, and `free=177.39GB` is
the tell: the memory exists, it is just not credited to the rollout engine. Raising
`gpu_memory_utilization` does not fix it and neither does freeing the actors' allocator
cache — item 1 above is why.

⚠️ **This stopped reproducing on 2026-09-16** and the pin it motivated is gone; sleep
releases by default. It is kept here because the margin was thin — the pool came out at
−1435 MB against an 86 GB budget — so a bigger model could bring it back. If it does,
`atom_cfg.engine_kwargs.sleep_keeps_memory_resident=true` still keeps the pool resident.

---

## 8.7 Further reading

| Need | Where |
|---|---|
| Changing the source, swapping models, not using the image | [1. Environment setup](01-env-setup.md) → [2. Dependencies](02-dependencies.md) → [4. Launching](04-launching.md) |
| Rebuilding the data (§8.4.3 is the condensed version) | [3. Models and data](03-data.md) |
| A failure not covered by §8.6 | [6. Troubleshooting](06-troubleshooting.md) |
| Two-node disaggregated serving (example 8, `bash examples/run.sh 8` with two hosts) | [5. Multi-node RDMA](05-multinode-rdma.md), [7. Disaggregated two-node RDMA](07-disaggregated-rdma.md) |

How the image is built and how the versions are pinned lives in
[`release/`](../../release/) at the repository root.
