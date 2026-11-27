> [Examples README](../README_cn.md) > 装依赖

# 2. 装依赖

## 2.1 基础依赖（所有例子）

```bash
sudo docker exec "$CONTAINER" bash -lc '
set -e
export ATOMRL_DIR="${ATOMRL_DIR:-$RL_ROOT/ATOM/atomrl}"
# 容器 root 访问宿主挂载仓库时允许 git introspection（editable install / setuptools_scm）
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

> 不加 `safe.directory` 会在 `pip install -e` 时报 `fatal: detected dubious ownership`。
> 需要分别安装根目录的 `atom` 包和 `atomrl` 包。`run_dapo.sh` 仍会把
> `$RL_ROOT/ATOM` 及 AITER shim 加进 `PYTHONPATH`，确保挂载的源码优先。

### 2.2 例子 6、7 追加：flydsl

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install "flydsl==0.1.8" && python3 -c "
import flydsl, transformers, vllm
print(\"flydsl\", flydsl.__version__, \"transformers\", transformers.__version__, \"vllm\", vllm.__version__)"'
```
> 期望 `flydsl 0.1.8 transformers 5.12.0 vllm 0.23.0`。

**为什么必做**：镜像自带的 flydsl 0.1.4.2 会让 `from aiter import flash_attn_varlen_func`
报版本不兼容，**训练前向直接挂**。

> 不要按 wheel 的 pin 回退。`run_dapo.sh` 把本地 `$AITER_DIR` 放在 `PYTHONPATH` 最前，
> 运行时用的是仓库里的 aiter 源码（要求 flydsl >= `0.1.5.dev515`），而镜像自带的
> `amd-aiter` wheel 反过来 pin `flydsl<0.1.5`，升级后它会报 `cannot import name 'fly_values'`。
> 两者互斥，走 `run_dapo.sh` 的 PYTHONPATH 才是对的。

> **transformers 必须是 5.x**，它把 Qwen3-MoE 的专家融合成 3D 张量，仓库的权重同步
> （`atomrl/engine/inference/vllm_moe_weight_sync.py`）按这个布局写。

### 2.3 例子 7 追加：megatron-core / Apex / TransformerEngine

已验证的三个组件和 revision：

- **megatron-core `0.18.2`**：`pip install --no-deps "megatron-core==0.18.2"`。
- **ROCm Apex `daed85255d51476425080e7e6203f0bee6d7e4cc`**：源码
  `setup.py install --cpp_ext --cuda_ext`，带 `PYTORCH_ROCM_ARCH=gfx950`。
- **ROCm TransformerEngine `6e541a10419a6e31bdc98b1516db04eb81a463b6`**
  （-> `2.15.0.dev0+6e541a1`）：源码 `pip install -v . --no-build-isolation`，约 9 分钟。

```bash
sudo docker exec "$CONTAINER" bash -lc 'pip install --no-deps "megatron-core==0.18.2"'
```

> **只装 megatron-core 不够，而且报错里不会提到 TE。** 缺 TransformerEngine 时
> `megatron.core` 的 TE spec 构造器返回 `None`，例子 7 在 `init_model` 阶段死于：
>
> ```text
> File ".../megatron/core/models/gpt/gpt_layer_specs.py", line 355,
>   in get_gpt_layer_with_transformer_engine_spec
> TypeError: 'NoneType' object is not callable
> ```
>
> 缺 Apex 只是一条告警（`Apex is not installed. Falling back to Torch Norm`），不影响
> 运行；缺 TE 则跑不起来。

按固定 revision 拉源码，然后编译：

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

# 编译 TE。NVTE_ROCM_ARCH / PYTORCH_ROCM_ARCH 要填本机架构：
# MI355X 是 gfx950，MI325X/MI308X 是 gfx942（用 rocm-smi --showproductname 确认）。
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
# ROCm 7.2.3 的 hipcc -v 在没有输入文件时返回 1，CK-JIT 的编译器 ABI 探测会把它
# 误判成"编译器不可用"。跳过这个探测不影响真正的 kernel 编译。
export TORCH_DONT_CHECK_COMPILER_ABI=1
python3 -m pip install -v . --no-build-isolation'
```

> 这一步要留出 **20-40 分钟和约 20 GB 磁盘**，大头在 AOTriton。上面写的 9 分钟只是
> `pip install` 那一步，不含拉取约 5.1 GiB submodule 和 AOTriton 的 configure/build。
> 开工前先看 `df -h`：`$DATA_ROOT` 满了会在编译中途才失败。

> **绝对不要从 PyPI `pip install transformer_engine`**，那会装成 NVIDIA 版，导入即
> undefined symbol。

> 不要装 `megatron-bridge`。Qwen3 的 HF <-> Megatron 转换由
> `atomrl/engine/training/bridges/` 负责。

---

## 2.4 验证 — 基础导入（所有例子）

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
> 期望 `GPUs 8`、`vllm 0.23.0`、`import OK`。

## 2.5 验证 — 完整导入链（所有例子）

确认每个 editable install 都指向正确的源码目录，而非残留的 pip 包。

> **这里的 `PYTHONPATH` 不是可选项。** 不设它，`import aiter` 会解析到镜像自带的
> `amd-aiter` wheel；该 wheel pin 了 `flydsl<0.1.5`，而 §2.2 已把 flydsl 升到 0.1.8，
> 于是报 `ImportError: cannot import name 'fly_values' from 'flydsl.compiler.protocol'`。
> 那是§2.2 所说互斥关系里**错误的那一侧**，不是安装坏了。`run_dapo.sh` 启动时会导出
> 同样的前缀，所以这个检查必须复现它，才是在验证真正跑起来的那份代码。

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

### 例子 7 追加：验证 Megatron 导入

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

**还要确认 TE 的 layer spec 真的能构造出来。** 没有 TransformerEngine 时
`import megatron.core` 依然成功，所以上面这段导入检查在跑不了例子 7 的机器上也会通过。
下面这条才能区分两者——它必须打印出 `ModuleSpec` 而不是抛异常：

```bash
sudo docker exec -e HIP_VISIBLE_DEVICES=0 "$CONTAINER" bash -lc '
python3 - <<PY
import transformer_engine
print("TE", transformer_engine.__version__)          # 期望 2.15.0.dev0+6e541a10
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as spec,
)
print("megatron TE layer spec OK:", type(spec()).__name__)
PY
'
```

> 这里报 `TypeError: NoneType object is not callable` 说明 TE 没装或架构编错了，回 §2.3。

## 2.6 验证 — flash-attn ROCm ABI（例子 7 / 源码编译的 flash-attn）

从源码编译的 flash-attn 的 Python wrapper 可能比 native extension 多传末尾 `num_splits`。
`run_grpo.sh` 启动时会幂等删除这个不支持的参数，但这个测试可以确认 forward+backward
端到端工作：

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

> 期望输出：`flash_varlen_forward_backward_ok (12, 2, 128)`。

---

## 2.7 vLLM AITER RMSNorm patch（例子 2、3、4 必需）

FP8 rollout 走 `VLLM_ROCM_USE_AITER=1`，vLLM 用 AITER RMSNorm，必须传
`use_model_sensitive_rmsnorm=1` 才能与训练侧 Lumen 的 model-sensitive RMSNorm 对齐，
否则 `rollout_corr/kl` 偏大。

> 该 patch 改的是**容器内的 vllm wheel**，`docker rm` 重建容器后会丢，**新容器必打一次**。
> 例子 1、5、6、7 是 BF16 rollout（`VLLM_ROCM_USE_AITER=0`），可跳过本节。

脚本幂等，只改 `kernels/aiter_ops.py` / `_aiter_ops.py` 两条 plain RMSNorm 路径，
不碰任何 quant-fusion 路径：

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
> 期望 `[patched] .../kernels/aiter_ops.py (2 site(s))`，已打过时是 `[ok] ... (already patched)`。

验证 patch 生效：

```bash
sudo docker exec "$CONTAINER" bash -lc '
python3 - <<PY
import inspect
from vllm.kernels import aiter_ops as k
ms = all("use_model_sensitive_rmsnorm=1" in inspect.getsource(getattr(k, a))
         for a in ["_rms_norm_impl", "_rocm_aiter_rmsnorm2d_fwd_with_add_impl"])
print("RMSNorm model-sensitive patch:", ms, "(FP8 需为 True)")
PY
'
```

---

## 2.8 ATOM JIT 预编译（例子 4、5 必需）

例子 4、5 用本地 `ATOM` + 本地 `aiter` 源码，不同于 vLLM wheel 内置的 kernel。
ATOM 启动 FP8 rollout 时会按需 JIT 编译 aiter kernel，其中 `module_rmsnorm` 最慢。
**这是环境安装成本，不要算进训练性能**，先单独编好：

```bash
# 先确认 submodule 已拉取（§3），否则找不到 generate.py
test -f "$RL_ROOT/aiter/3rdparty/composable_kernel/example/ck_tile/10_rmsnorm2d/generate.py"

# 首次可能耗时十几到二十分钟；看到 PRECOMPILE_DONE 才算完成
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

被 `Ctrl-C` / 容器重启 / `pkill` 打断后可能留下 stale lock，表现为后续进程一直打印
`waiting for baton release at .../lock_module_rmsnorm`，但没有 `ninja` / `hipcc` / `clang-22`
编译进程。清理后重跑：

```bash
sudo docker exec "$CONTAINER" bash -lc '
rm -rf "$AITER_DIR/aiter/jit/build/lock_module_rmsnorm" \
       "$AITER_DIR/aiter/jit/build/module_rmsnorm"
'
```

> 例子 2、3 不需要这一步，它们走镜像内已安装的 vLLM/AITER 路径。
