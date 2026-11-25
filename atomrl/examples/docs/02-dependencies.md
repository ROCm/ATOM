> [Examples README](../README.md) > Dependencies

# 2. Dependencies

## 2.1 Base dependencies (all examples)

```bash
sudo docker exec "$CONTAINER" bash -lc '
set -e
export ATOMRL_DIR="${ATOMRL_DIR:-$RL_ROOT/ATOM/atomrl}"
# Allow git introspection when container root touches host-mounted repos
# (editable install / setuptools_scm).
git config --global --add safe.directory "$ATOMRL_DIR" || true
git config --global --add safe.directory "$LUMEN_DIR" || true
git config --global --add safe.directory "$AITER_DIR" || true
git config --global --add safe.directory "$RL_ROOT/ATOM" || true

cd "$AITER_DIR" && AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop || pip install -e .
cd "$RL_ROOT/ATOM" && pip install --no-build-isolation -e . --no-deps
LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation -e "$RL_ROOT/ATOM/atomrl"
pip install "accelerate>=0.28" datasets "math_verify[antlr4_13_2]" \
  safetensors wandb
'
```

> Without `safe.directory`, `pip install -e` fails with
> `fatal: detected dubious ownership`. Install the root `atom` package and the
> separate `atomrl` package; the example still puts the checkout and its AITER
> shim on `PYTHONPATH` so mounted source changes take precedence.

### 2.2 Additionally for examples 6 and 7: flydsl

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install "flydsl==0.1.8" && python3 -c "
import flydsl, transformers, vllm
print(\"flydsl\", flydsl.__version__, \"transformers\", transformers.__version__, \"vllm\", vllm.__version__)"'
```
> Expect `flydsl 0.1.8 transformers 5.12.0 vllm 0.23.0`.

**Why this is mandatory**: the image ships flydsl 0.1.4.2, which makes
`from aiter import flash_attn_varlen_func` raise a version-incompatibility error and
**kills the training forward pass outright**.

> Do not downgrade back to the wheel's pin. `run_dapo.sh` puts the local
> `$AITER_DIR` first on `PYTHONPATH`, so what actually runs is the aiter source from the
> repo, which requires flydsl >= `0.1.5.dev515`. The image's bundled `amd-aiter` wheel pins
> `flydsl<0.1.5` and reports `cannot import name 'fly_values'` after the upgrade. The two
> are mutually exclusive, and going through `run_dapo.sh`'s PYTHONPATH is the correct side.

> **transformers must be 5.x.** It fuses the Qwen3-MoE experts into 3D tensors, and the
> repo's weight sync (`atomrl/engine/inference/vllm_moe_weight_sync.py`) is written
> against that layout.

### 2.3 Additionally for example 7: megatron-core / Apex / TransformerEngine

The three verified components and revisions:

- **megatron-core `0.18.2`**: `pip install --no-deps "megatron-core==0.18.2"`.
- **ROCm Apex `daed85255d51476425080e7e6203f0bee6d7e4cc`**: from source,
  `setup.py install --cpp_ext --cuda_ext`, with `PYTORCH_ROCM_ARCH=gfx950`.
- **ROCm TransformerEngine `6e541a10419a6e31bdc98b1516db04eb81a463b6`**
  (-> `2.15.0.dev0+6e541a1`): from source, `pip install -v . --no-build-isolation`,
  about 9 minutes.

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install --no-deps "megatron-core==0.18.2"'
```

> **megatron-core alone is not enough, and the failure does not mention TE.** Without
> TransformerEngine, `megatron.core`'s TE spec builder returns `None` submodules and
> example 7 dies during `init_model` with
>
> ```text
> File ".../megatron/core/models/gpt/gpt_layer_specs.py", line 355,
>   in get_gpt_layer_with_transformer_engine_spec
> TypeError: 'NoneType' object is not callable
> ```
>
> Apex being absent is only a warning (`Apex is not installed. Falling back to Torch
> Norm`) and does not stop the run; TE does.

Fetch both at the pinned revisions, then build:

```bash
sudo docker exec "$CONTAINER" bash -lc '
set -eux
APEX_REV=daed85255d51476425080e7e6203f0bee6d7e4cc
TE_REV=6e541a10419a6e31bdc98b1516db04eb81a463b6
git config --global --add safe.directory "*"
[ -d "$DATA_ROOT/apex_src/.git" ] || git clone https://github.com/ROCm/apex.git "$DATA_ROOT/apex_src"
git -C "$DATA_ROOT/apex_src" checkout "$APEX_REV"
git -C "$DATA_ROOT/apex_src" submodule update --init --recursive --jobs 16
[ -d "$DATA_ROOT/te_src/.git" ] || git clone https://github.com/ROCm/TransformerEngine.git "$DATA_ROOT/te_src"
git -C "$DATA_ROOT/te_src" checkout "$TE_REV"
git -C "$DATA_ROOT/te_src" submodule update --init --recursive --jobs 16'

# Build TE. Set NVTE_ROCM_ARCH / PYTORCH_ROCM_ARCH to this machine's arch:
# gfx950 for MI355X, gfx942 for MI325X/MI308X (`rocm-smi --showproductname`).
sudo docker exec "$CONTAINER" bash -lc '
set -eux
python3 -m pip uninstall -y transformer-engine transformer_engine \
  transformer-engine-torch transformer_engine_torch || true
cd "$DATA_ROOT/te_src" && rm -rf build dist transformer_engine.egg-info
mkdir -p "$DATA_ROOT/tmp"
export TMPDIR="$DATA_ROOT/tmp"
export NVTE_FRAMEWORK=pytorch NVTE_USE_ROCM=1
export NVTE_ROCM_ARCH=gfx950 PYTORCH_ROCM_ARCH=gfx950
export NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=1
export MAX_JOBS=48
# ROCm 7.2.3 hipcc -v exits 1 when given no input files, which the CK-JIT
# compiler-ABI probe reads as "compiler unusable". Skipping the probe does not
# skip actual kernel compilation.
export TORCH_DONT_CHECK_COMPILER_ABI=1
python3 -m pip install -v . --no-build-isolation'
```

> Budget **20-40 minutes and about 20 GB** of disk for this, most of it in AOTriton.
> The 9 minutes quoted for the `pip install` step alone does not include fetching
> the ~5.1 GiB of submodules or the AOTriton configure/build. Check `df -h` first;
> a full `$DATA_ROOT` fails deep in the build.

> **Never `pip install transformer_engine` from PyPI.** That installs the NVIDIA
> build, which fails on import with an undefined symbol.

> Do not install `megatron-bridge`. The HF <-> Megatron weight conversion is handled by
> `atomrl/engine/training/bridges/`.

---

## 2.4 Verify — basic imports (all examples)

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import torch, vllm, ray, atomrl
from atomrl.engine.inference.vllm_ray_server import VLLMRayServer
print("torch", torch.__version__, "vllm", vllm.__version__,
      "ray", ray.__version__, "GPUs", torch.cuda.device_count())
print("import OK")
PY
'
```
> Expect `GPUs 8`, `vllm 0.23.0`, `import OK`.

## 2.5 Verify — full import chain (all examples)

Run this to confirm every editable install resolves to the correct source directory
rather than a stale pip package.

> **`PYTHONPATH` is not optional here.** Without it `import aiter` resolves to the
> image's bundled `amd-aiter` wheel, which pins `flydsl<0.1.5` and dies on
> `ImportError: cannot import name 'fly_values' from 'flydsl.compiler.protocol'`
> after §2.2 upgraded flydsl to 0.1.8. That is the wrong side of the mutual
> exclusion described there, not a broken install. `run_dapo.sh` exports the same
> prefix at startup, so this check has to reproduce it to test what actually runs.

```bash
sudo docker exec -e HIP_VISIBLE_DEVICES=0 "$CONTAINER" bash -lc '
export ATOMRL_DIR="${ATOMRL_DIR:-$RL_ROOT/ATOM/atomrl}"
export PYTHONPATH="$ATOMRL_DIR:$AITER_DIR:${PYTHONPATH:-}"
python3 - <<PY
import sys

checks = []
import aiter;    checks.append(("aiter",    aiter.__file__))
import atomrl;  checks.append(("atomrl",  atomrl.__file__))
import atomrl.lumen;    checks.append(("lumen",    atomrl.lumen.__file__))

try:
    import vllm; checks.append(("vllm", vllm.__file__))
except ImportError:
    checks.append(("vllm", "NOT INSTALLED"))

for name, path in checks:
    print(f"  {name:12s} {path}")

import os
ATOMRL = os.environ["ATOMRL_DIR"]
for name, expected_prefix in [
    ("lumen",   f"{os.environ['LUMEN_DIR']}/"),
    ("atomrl", f"{ATOMRL}/"),
    ("aiter",   f"{os.environ['AITER_DIR']}/"),
]:
    mod = sys.modules.get(name)
    if mod and hasattr(mod, "__file__") and mod.__file__:
        assert mod.__file__.startswith(expected_prefix), \
            f"{name} imported from {mod.__file__}, expected {expected_prefix}"
print("all source installs verified")
PY
'
```

### Additionally for example 7: verify Megatron imports

```bash
sudo docker exec -e HIP_VISIBLE_DEVICES=0 "$CONTAINER" bash -lc '
python3 - <<PY
import megatron.core; print("megatron.core ok:", megatron.core.__file__)

try:
    from megatron.training import get_args
    print("megatron.training ok")
except ImportError:
    print("MISSING — need ROCm Megatron-LM, not PyPI megatron-core")

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay
    print("RouterReplay ok")
except ImportError:
    print("MISSING — need ROCm fork with router_replay")
PY
'
```

**Also confirm the TE layer spec actually builds.** Importing `megatron.core`
succeeds without TransformerEngine, so the import check above passes on a machine
where example 7 cannot run. This is the check that distinguishes them — it must
print a `ModuleSpec`, not raise:

```bash
sudo docker exec -e HIP_VISIBLE_DEVICES=0 "$CONTAINER" bash -lc '
python3 - <<PY
import transformer_engine
print("TE", transformer_engine.__version__)          # expect 2.15.0.dev0+6e541a10
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as spec,
)
print("megatron TE layer spec OK:", type(spec()).__name__)
PY
'
```

> `TypeError: NoneType object is not callable` here means TE is missing or was
> built for the wrong arch — go back to §2.3.

## 2.6 Verify — flash-attn ROCm ABI (example 7 / source-built flash-attn)

When flash-attn is built from source (rather than using the container's bundled version),
the Python wrapper may have an extra `num_splits` argument that the native extension does
not support. `run_grpo.sh` idempotently removes the unsupported parameter at startup, but
this test confirms the forward+backward path works end to end:

```bash
sudo docker exec -e HIP_VISIBLE_DEVICES=0 "$CONTAINER" bash -lc '
python3 - <<PY
import torch
from flash_attn import flash_attn_varlen_func

q = torch.randn(12, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
cu = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, k, v, cu, cu, 7, 7, causal=True)
out.float().sum().backward()
torch.cuda.synchronize()
print("flash_varlen_forward_backward_ok", tuple(out.shape))
PY
'
```

> Expected output: `flash_varlen_forward_backward_ok (12, 2, 128)`.

---

## 2.7 vLLM AITER RMSNorm patch (required for examples 2, 3, 4)

FP8 rollout runs with `VLLM_ROCM_USE_AITER=1`, so vLLM uses the AITER RMSNorm. It has to
be passed `use_model_sensitive_rmsnorm=1` to match the model-sensitive RMSNorm on the
Lumen training side; otherwise `rollout_corr/kl` comes out too high.

> This patch modifies **the vllm wheel inside the container**, so it is lost when the
> container is recreated with `docker rm` and **must be applied once per new container**.
> Examples 1, 5, 6 and 7 use BF16 rollout (`VLLM_ROCM_USE_AITER=0`) and can skip this
> section.

The script is idempotent and touches only the two *plain* RMSNorm paths in
`kernels/aiter_ops.py` / `_aiter_ops.py`, never a quant-fusion path:

```bash
cat > "$RL_ROOT/patch_vllm_aiter_rmsnorm.py" <<'PYEOF'
#!/usr/bin/env python3
"""vLLM AITER RMSNorm Patch (model-sensitive / T5-like) for the rollout side.

Patches the *plain* (non-quant) AITER RMSNorm paths inside the container's
installed vLLM wheel so they pass ``use_model_sensitive_rmsnorm=1``. Quant-fusion
paths (rmsnorm2d_fwd_with_dynamicquant / *_fp8_group_quant / ...) are NOT touched.
Idempotent: safe to run repeatedly.
"""

import importlib.util
import os
import sys

RMS_OLD = """    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon)"""

RMS_NEW = """    if not getattr(_rms_norm_impl, "_lumen_logged", False):
        print("[vllm-aiter] rms_norm use_model_sensitive_rmsnorm=1", flush=True)
        _rms_norm_impl._lumen_logged = True

    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon, use_model_sensitive_rmsnorm=1)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon, use_model_sensitive_rmsnorm=1)"""

ADD_OLD = """    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
    )
    return out, residual_out"""

ADD_NEW = """    if not getattr(_rocm_aiter_rmsnorm2d_fwd_with_add_impl, "_lumen_logged", False):
        print(
            "[vllm-aiter] rmsnorm2d_fwd_with_add use_model_sensitive_rmsnorm=1",
            flush=True,
        )
        _rocm_aiter_rmsnorm2d_fwd_with_add_impl._lumen_logged = True
    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
        use_model_sensitive_rmsnorm=1,
    )
    return out, residual_out"""

REPLACEMENTS = ((RMS_OLD, RMS_NEW), (ADD_OLD, ADD_NEW))


def _vllm_dir() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("ERROR: vllm is not importable in this interpreter")
    return spec.submodule_search_locations[0]


def _patch_file(path: str) -> int:
    if not os.path.isfile(path):
        print(f"[skip] {path} (not found)")
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    changed = 0
    for old, new in REPLACEMENTS:
        if new in src:
            continue  # already patched
        if old in src:
            src = src.replace(old, new, 1)
            changed += 1
    if changed:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        print(f"[patched] {path} ({changed} site(s))")
    else:
        already = sum(new in src for _, new in REPLACEMENTS)
        print(f"[ok] {path} (already patched)" if already else f"[skip] {path} (no plain call sites)")
    return changed


def main() -> None:
    vdir = _vllm_dir()
    targets = [os.path.join(vdir, "kernels", "aiter_ops.py"), os.path.join(vdir, "_aiter_ops.py")]
    total = sum(_patch_file(p) for p in targets)
    print(f"[done] vLLM AITER RMSNorm patch: {total} replacement(s) applied")


if __name__ == "__main__":
    main()
PYEOF
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT" && python3 patch_vllm_aiter_rmsnorm.py'
```
> Expect `[patched] .../kernels/aiter_ops.py (2 site(s))`, or `[ok] ... (already patched)`
> if it was applied before.

Verify the patch took effect:

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import inspect
from vllm.kernels import aiter_ops as k
ms = all("use_model_sensitive_rmsnorm=1" in inspect.getsource(getattr(k, a))
         for a in ["_rms_norm_impl", "_rocm_aiter_rmsnorm2d_fwd_with_add_impl"])
print("RMSNorm model-sensitive patch:", ms, "(must be True for FP8)")
PY
'
```

---

## 2.8 ATOM JIT precompilation (required for examples 4, 5)

Examples 4 and 5 use the local `ATOM` and local `aiter` sources rather than the kernels
built into the vLLM wheel. When ATOM starts a rollout it JIT-compiles aiter kernels on
demand, and `module_rmsnorm` is the slow one. **This is an environment setup cost, not
training performance** — compile it separately up front:

```bash
# Confirm the submodule is present (§3), otherwise generate.py will not be found.
test -f "$RL_ROOT/aiter/3rdparty/composable_kernel/example/ck_tile/10_rmsnorm2d/generate.py"

# The first run can take fifteen to twenty minutes; it is done only once
# PRECOMPILE_DONE appears.
sudo docker exec "$CONTAINER" bash -lc '
export PYTHONPATH="$AITER_DIR:${PYTHONPATH:-}"
python3 - <<PY
import torch
from aiter import rmsnorm2d_fwd
print("PRECOMPILE_START", flush=True)
x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
w = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
y = rmsnorm2d_fwd(x, w, 1e-6, use_model_sensitive_rmsnorm=1)
torch.cuda.synchronize()
print("PRECOMPILE_DONE", y.shape, y.dtype, flush=True)
PY
'
```

A `Ctrl-C`, container restart or `pkill` can leave a stale lock behind. The symptom is a
process that keeps printing `waiting for baton release at .../lock_module_rmsnorm` while
no `ninja` / `hipcc` / `clang-22` compile process exists. Clean up and rerun the
precompile:

```bash
sudo docker exec "$CONTAINER" bash -lc '
rm -rf "$AITER_DIR/aiter/jit/build/lock_module_rmsnorm" \
       "$AITER_DIR/aiter/jit/build/module_rmsnorm"
'
```

> Examples 2 and 3 do not need this step; they use the vLLM/AITER path already installed
> in the image.
