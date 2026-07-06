# 把 ATOM 的 EP-MoE 换成 FlyDSL fused(MegaMoE)—— **mega_moe_v1 分支专版**

> 只讲 FlyDSL `mega_moe_v1` 分支这一版:怎么接线、怎么跑、会踩哪些坑。
>
> 目标:在 **8×gfx950(MI3xx)单节点**上,让 `deepseek-ai/DeepSeek-V4-Pro`(a8w4:act fp8 / weight fp4)的 MoE 走 MegaMoE 融合算子(dispatch⊕gemm1 megakernel + gemm2 内联 combine),并通过 gsm8k 精度门槛(≥0.90)、再跑 perf。
>
> 术语:**fused = MegaMoE**(端到端 EP-MoE 单算子);tp+dp = aiter TP;ep+dp mori = 原生 mori EP。

---

## 0. 一句话

fused 就是:在 atom 的 MoE `apply()` 里**早返回** `MegaMoE.forward_bf16(x, topk_weights, topk_ids)`,用它一把做完 dispatch+gemm1+quant+gemm2+combine,**完全绕过** aiter experts 和 mori 的 dispatch/combine。你要做的只是:①把权重转成 MegaMoE 布局;②把 apply 接到 MegaMoE;③别让启动期的退化数据把它跑崩。

---

## 1. 仓库/代码关系

| 仓库 | 位置 | 作用 |
|---|---|---|
| atom | 容器 `/app/ATOM` | 推理引擎;你要改的 3 个文件在这 |
| aiter | 容器 `/app/aiter-test` | 算子库(tp/mori experts 用);fused **不碰**它的 experts |
| mori | site-packages(容器自带) | 对称堆 shmem;MegaMoE 的 dispatch/combine 缓冲建在 mori.shmem 上 |
| FlyDSL | 工作区 `/home/yashao/FlyDSL`,分支 `mega_moe_v1` | ① Python 包 `python/flydsl`(编译器) ② kernel 定义 `kernels/`(MegaMoE 在此) |

**fused 用到的类和文件**(接线时按这个 import,别写错):
- `MegaMoE` 在 `kernels/mega_moe.py`(`from kernels.mega_moe import MegaMoE`)
- gemm1 megakernel:`kernels/mega_moe_gemm1.py`;gemm2+combine op:`kernels/mixed_moe_gemm_2stage.py`;dispatch/combine:`kernels/dispatch_combine_intranode_op.py`
- tune 表:`kernels/tuning_configs/flydsl_gfx950_mi355x_{MegaStage1,MegaGemm2}_ep8.json`

**关键:fused 用的是 `kernels/mega_moe.py`,通过 `ATOM_FLYDSL_KERNELS_PATH=/home/yashao/FlyDSL` 让 atom `sys.path` 能 `from kernels.mega_moe import MegaMoE`;而 `import flydsl`(编译器)用容器自带的那份即可。** 这是 mega_moe_v1 能免去大量版本冲突的核心(见 §2)。

> 宿主 `/shared/amdgpu/home/yanbo_shao_qle/deepseek` 挂到容器 `/home/yashao`。`/app/*` 是容器 overlay,**换容器会丢**,需重做本文的部署。

---

## 2. 环境:FlyDSL 用哪份?

**结论:用容器自带的 flydsl(site-packages)+ 工作区的 `kernels/`,不要 `pip install -e` 工作区 FlyDSL。**

原因(实测验证):
- 容器 `atom-dev:latest`(atom 0.1.4.dev197 / aiter #3966 / **flydsl 0.2.2**)自带的 flydsl,能**直接 import 并数值正确运行** mega_moe_v1 的 kernels(独立 bench:v4_pro a8w4 bs8192 `mega-vs-baseline=4.3e-5`,e2e 1.16×)。
- `import flydsl` 解析到 site-packages(编译器 + 为本容器编好的 `_mlir`),`from kernels.mega_moe import ...` 解析到工作区(`ATOM_FLYDSL_KERNELS_PATH`)。两者**天然配套**。
- **为什么别 `pip install -e` 工作区 FlyDSL**:
  1. editable 会**全局覆盖** site-packages 的 flydsl → tp+dp / mori 的 aiter experts 用的 flydsl kernel 与之不配套,在 **isl≥1024 长 prefill** 下崩(`'Int32' shrui` / `arith.muli` / `extsi i64`)。
  2. 工作区 `python/flydsl/_mlir` 是**别的容器编的**,换容器有 ABI 风险,加载可能直接挂。

**验证 flydsl 可用**(接线前先跑一次,几秒):
```bash
podman exec <容器> bash -lc '
export ATOM_FLYDSL_KERNELS_PATH=/home/yashao/FlyDSL
python3 -c "import flydsl,sys; print(\"flydsl:\",flydsl.__file__); \
sys.path.insert(0,\"/home/yashao/FlyDSL\"); import kernels.mega_moe as m; print(\"MegaMoE OK:\",m.MegaMoE)"'
# 期望:flydsl 指向 /opt/venv/.../site-packages/flydsl,MegaMoE OK
```

### 起容器
```bash
podman run -d --name <容器> \
  --device=/dev/kfd --device=/dev/dri --ipc=host \
  --security-opt seccomp=unconfined --cap-add=CAP_SYS_PTRACE \
  --group-add video --group-add render --network host \
  -v /shared/amdgpu/home/yanbo_shao_qle/deepseek:/home/yashao:rw -v /mnt:/mnt:rw \
  -w /home/yashao --entrypoint sleep docker.io/rocm/atom-dev:latest infinity
```
- **`--ipc=host` 就别加 `--shm-size`**(podman 会报冲突)。
- 工作区切分支:`cd /home/yashao/FlyDSL && git checkout -B mega_moe_v1 origin/mega_moe_v1`。

---

## 3. atom 侧要改的 4 处(fused-only 最小集)

> **重要认知:fused 时 apply() 早返回,mori/flydsl 的 dispatch/combine(modular kernel)构建但从不被调用。** 所以 fused **不需要** `init_flydsl_op` / `flydsl_op` 那套(那是给"零融合 flydsl 组"用的)。这让接线大幅简化。

### 3.1 新增文件 `atom/model_ops/fused_moe/flydsl_mega_experts.py`
三个函数:
- `build_mega_weights(layer)`:从 raw fp4 权重建 MegaMoE 布局(见 §4),存到 `layer._mega_w1/_mega_w1_scale/_mega_w2/_mega_w2_scale`。
- `get_or_build_mega_moe(...)`:**进程级单例缓存**(按 shape/quant/mtpr/tile),`from kernels.mega_moe import MegaMoE` 构造。61 层共享 1 个 MegaMoE 实例(权重是运行时指针,不 bake),否则 shmem/JIT 撑爆。
- `run_mega_moe(layer, x, topk_weights, topk_ids, ...)`:换 `mega.stage1.w1/w1_scale` + `mega.w2/w2_scale` 为本层权重指针,`return mega.forward_bf16(x, wts, ids)`。

关键参数:
- `gemm2_tile` 默认交给 `-1,-1,-1` → 让 MegaMoE 自动加载 MegaGemm2 tune 表(prefill tile_m=64 / decode 16-32)。
- `mtpr`(max_tok_per_rank)= `ATOM_MEGA_MTPR`(默认 8192),**必须 2 的幂**;`run_tokens>mtpr` 时向上取 2 的幂。
- `experts` 从本 rank local 权重反推:`local_E = _mega_w1.shape[0] // (2*inter_dim)`,`experts = local_E * world`。
- **w1/w1_scale/w2/w2_scale 都用本 rank 的 local(epr)专家**:mega_moe_v1 的 `MegaMoeStage1` 原生按 "ATOM local convention" 索引 w1(注释写死),给全局 384 会越界崩。

### 3.2 `atom/model_ops/fused_moe/mori_prepare_finalize.py`:加两个开关函数
```python
def _use_flydsl():        # ATOM_USE_FLYDSL=1
    return os.environ.get("ATOM_USE_FLYDSL","0").strip().lower() in {"1","true","on","yes"}
def _use_flydsl_fused():   # 需 ATOM_USE_FLYDSL=1 且 ATOM_USE_FLYDSL_FUSED=1
    return _use_flydsl() and os.environ.get("ATOM_USE_FLYDSL_FUSED","0").strip().lower() in {"1","true","on","yes"}
```
(文件顶部记得 `import os`。)

### 3.3 `atom/model_ops/moe.py` 的 mxfp4 MoE 方法 —— **两处 hook**
**hook A(`process_weights_after_loading`,在 atom 自己 shuffle w13/w2 之前)**:
```python
from atom.model_ops.fused_moe.mori_prepare_finalize import _use_flydsl_fused
if _use_flydsl_fused():
    from atom.model_ops.fused_moe.flydsl_mega_experts import build_mega_weights
    build_mega_weights(layer)      # 必须在 shuffle 覆盖 w13_weight 之前,用 raw fp4
# ...原来的 shuffle_weight / moe_shuffle_scale 照旧...
```
**hook B(`apply`,在 `return self.fused_experts(...)` 之前早返回)**:
```python
from atom.model_ops.fused_moe.mori_prepare_finalize import _use_flydsl_fused
if _use_flydsl_fused() and getattr(layer, "_mega_w1", None) is not None:
    from atom.model_ops.fused_moe.flydsl_mega_experts import run_mega_moe
    return run_mega_moe(layer, x, topk_weights, topk_ids,
                        model_dim=self.hidden_size, inter_dim=self.intermediate_size,
                        experts=global_num_experts, topk=top_k, quant="a8w4")
return self.fused_experts(...)      # 原样
```
> 注意 dev197 的 moe.py 里同名 `apply`/`process_weights_after_loading` 有多份(不同量化类),StrReplace 时用足够上下文锚定到 **mxfp4 那个类**(用 `w13_swizzle_layout`/`moe_shuffle_scale`/`**moe_extra_args` 附近的唯一片段)。

### 3.4 `atom/model_engine/model_runner.py`:warmup/dummy token 打散(⚠️ 见 §6 坑1,这是**最新发现的坑**)

---

## 4. 权重布局对齐(精度命门,最容易错)

MegaMoE 要的布局 = **裸 `shuffle_weight`(FlyDSL `tests/utils`,layout=(16,16))+ `fp4_utils.e8m0_shuffle`**,**不是** aiter 的 shuffle、**不带 GU 交错**:

```python
from tests.utils import shuffle_weight            # FlyDSL 仓库的,不是 aiter!
from tests.kernels.utils import fp4_utils

# w1 = 本 rank local epr 专家(mega_moe_v1 原生 local,别给全局 384)
w13 = layer.w13_weight.data          # [E_local, 2*inter, hidden//2] fp4-packed uint8
w13f = w13.reshape(E*2*inter, hidden//2).view(torch.float4_e2m1fn_x2)
layer._mega_w1       = shuffle_weight(w13f).view(torch.uint8).contiguous()
layer._mega_w1_scale = fp4_utils.e8m0_shuffle(w13_scale.reshape(E*2*inter, hidden//32)).view(torch.uint8).contiguous()
# w2 同理,最后 .view(-1) 摊平
layer._mega_w2       = shuffle_weight(w2f).view(torch.uint8).contiguous().view(-1)
layer._mega_w2_scale = fp4_utils.e8m0_shuffle(w2_scale_2d).view(torch.uint8).contiguous().view(-1)
```
- `model_dim/inter_dim` 用 atom **padded** 维(`self.hidden_size` / `self.intermediate_size`)。
- **w1 必须 local(本 rank 48 专家)**;w2 也 local。mega_moe_v1 kernel 按 local 索引,给全局会越界崩。
- 校验小抄:V4-Pro 每卡 `_mega_w1=(294912, 3584)`(=48×2×3072 行 × 7168/2 列),`w13_scale` uint8。

---

## 5. 运行 + 验证(严格按这三步,能把问题锁在最小范围)

### 步骤1:容器自带 mori 的 ep+dp 精度(证明容器/atom/mori 健康)
```bash
# run_v4_acc_one.sh <cfg> <mode> <par>
podman exec <容器> bash -lc 'cd /home/yashao/_tmp_atom && setsid bash run_v4_acc_one.sh dpep_mori baseline dpep >/home/yashao/_tmp_atom/_acc_mori.out 2>&1 </dev/null & disown'
# 期望 gsm8k ≈0.95/0.95(≥0.90 门槛)。实测 0.95 ✓
```

### 步骤2:独立 kernel bench(证明 flydsl + mega_moe_v1 kernel 数值正确,与 atom 接线无关)
```bash
podman exec <容器> bash -lc 'cd /home/yashao/FlyDSL && MORI_SHMEM_HEAP_SIZE=42949672960 \
  torchrun --standalone --nproc_per_node=8 \
  tests/kernels/bench_moe_intranode_stage1_groupgemm.py --network v4_pro --quant a8w4 \
  --bs-list 2048,8192 --iters 5 --warmup 2'
# 期望:[FULL-E2E] v4_pro a8w4 bs=8192 -> PASS,mega-vs-baseline~4e-5
```
> **诊断价值**:步骤2 PASS 但步骤3 崩 ⇒ 问题 100% 在 atom 接线(权重布局 / warmup 路由 / mtpr),**不是 kernel/flydsl 版本**。别再去动 flydsl。

### 步骤3:atom fused e2e 精度
```bash
podman exec <容器> bash -lc 'cd /home/yashao/_tmp_atom && setsid bash run_v4_acc_one.sh dpep_fused fused dpep >/home/yashao/_tmp_atom/_acc_fused.out 2>&1 </dev/null & disown'
# 看 bench_results/v4_accuracy_newpr/dpep_fused_gsm8k.log,过 0.90 即算精度通过
```

### 日志位置
| 内容 | 路径(容器 /home/yashao = 宿主 workspace) |
|---|---|
| server 日志 | `_tmp_atom/serverv4_dpep_{mode}_cudagraph.log` |
| 精度驱动输出 | `_tmp_atom/_acc_*.out` |
| gsm8k 结果 | `bench_results/v4_accuracy_newpr/<cfg>_gsm8k.log` |
| 独立 bench | 自己 `>` 重定向的文件 |

---

## 6. 坑汇总(mega_moe_v1 仍需注意的,按严重度)

### 坑1 ⚠️【当前未完全解决】warmup / dummy 退化路由 → HIP illegal memory access
- **现象**:fused server 在 model load / warmup 阶段崩,`hipModuleLoadData(&module,data) failed 'hipErrorIllegalAddress'` → `flydsl_gpu_module_load_to_device failed error code -1`,栈在 `run_mega_moe → forward_bf16 → forward_prequant → _g2.run → combine_no_stage1`。
- **根因**:atom 的 `warmup_model` / `dummy_execution` / `dummy_prefill_execution` 用 `Sequence([0]*N)`(**所有 token id=0**)→ gate 把所有 token 路由到**同一批专家** → 单专家 recv ≫ `max_recv=world*mtpr` → 溢出缓冲。cudagraph capture 也走这些 dummy。
- **佐证**:独立 bench 用均衡随机路由就 PASS ⇒ 是 atom 退化 dummy 路由触发,**不是 kernel bug**。
- **缓解方向(试过 [i%8192] 打散仍崩,需继续)**:
  - ① 让 dummy **直接构造均衡 topk_ids**(而非靠 token id 间接影响 gate);
  - ② warmup/capture 时对 fused 走安全回退;
  - ③ forward 前打印 per-expert max-count 确认路由是否真被打散;
  - ⚠️ 若在 run_mega_moe 里按 `is_dummy_run` 替换 topk_ids,**注意 cudagraph capture 会把替换后的静态张量固化**,导致真实推理路由错——不要在会被 capture 的路径里 new 张量。
- `forward_context.is_dummy_run` 可判定 dummy(`atom/utils/forward_context.py`)。

### 坑2【perf 命门】重复权重挤垮 KV cache
- build 出 `_mega_*` 后,原 `w13_weight/w2_weight` 是**死 fallback**(fused 前向不走),若不释放 → 显存存两份 → KV 块数被砍 35~55×(如 85436→1558)→ c4096 深队列被准入限流、TTFT 爆炸、吞吐反被 tp+dp 反超。
- **修法**:hook A build 完后把 `layer.w13_weight/w2_weight` 置 `torch.empty(0)` 并**早 return 跳过多余 shuffle**。
- ⚠️ 精度不受影响(死路径),但 **perf 必须做**,否则 fused c4096 数据不可比。

### 坑3 mtpr 与 profile skew
- mtpr 必须 2 的幂;profile_run 历史上用 8192 skew dummy 是 OOM 触发点(与坑1 同源)。

### 坑4 maxseqs caveat(perf 口径)
- DP=8 下 decode 单卡批 = `min(conc, maxseqs)/8`。maxseqs=256 会把 c512/c4096 都钉在 32/卡 → 高并发吞吐饱和趋同(非真实并发)。
- 放开建议:c64 用 ms≤256;c512 用 ms512;c4096:1k/1k 用 ms2048、**8k/1k 用 ms1024**(8k 每序列 KV 大,ms2048 反而 KV 饥饿)。

### 坑5 通用
- server ready 判据:`Uvicorn running` / `Application startup complete`;错误关键字 `Traceback|RuntimeError|HIP error|OutOfMemory|Address already|deficit`。**waitready 别用过宽的 `|Error`**(会误匹配 HF 良性警告 "can yield errors")。
- 多人机器开跑前 `rocm-smi --showuse` 确认 8 卡空闲。
- V4 `--max-num-seqs 4096` 启动 OOM → 用 256/1024 + util 0.85。

---

## 7. env 速查表

| env | 值 | 用途 |
|---|---|---|
| `ATOM_USE_FLYDSL` | 1 | 启用 FlyDSL 路径(fused 前置) |
| `ATOM_USE_FLYDSL_FUSED` | 1 | 启用 MegaMoE 融合(fused 开关) |
| `ATOM_FLYDSL_KERNELS_PATH` | `/home/yashao/FlyDSL` | 让 atom 能 `from kernels.mega_moe import` |
| `ATOM_MOE_GU_ITLV` | 1 | MoE gate-up 交错(V4 配方) |
| `MORI_SHMEM_HEAP_SIZE` | `17179869184`(16G) | mori 对称堆;MegaMoE 缓冲建其上 |
| `ATOM_MEGA_MTPR` | 8192 | max_tok_per_rank,2 的幂 |
| `ATOM_MEGA_GEMM2_TILE` | 不设(=`-1,-1,-1`) | 让 MegaMoE 自动用 tune 表;设了则强制固定 tile |
| `ATOM_DISABLE_MMAP` / `AITER_BF16_FP8_MOE_BOUND` | true / 0 | V4-Pro 配方 |
| tp+dp/mori experts tuned | `AITER_CONFIG_FMOE=<dsv4 csv>` + `AITER_EP_KEEP_TOPK=1` | perf 时让 aiter experts 命中 tuned(否则 topk 6→5 错位) |

---

## 8. 快速排错决策树

```
fused server 起不来 / 崩
├─ 报 import kernels.mega_moe 失败      → 检查 ATOM_FLYDSL_KERNELS_PATH + 工作区在 mega_moe_v1 分支
├─ 报 flydsl compile / ir_value / arith  → 你是不是 pip install -e 了工作区 flydsl?卸掉,用容器自带(§2)
├─ HIP illegal @ combine_no_stage1(warmup) → 坑1:退化 dummy 路由,先跑步骤2 bench 确认 kernel 没问题
├─ 精度 0.12 / 0.22                      → 权重布局(§4)/ w1 给成全局了 / sort_block_m↔tile_m 错配
├─ 精度过但 c4096 吞吐被 tp+dp 反超      → 坑2:重复权重没释放,KV 被挤垮
└─ 步骤2 独立 bench 就 FAIL             → 才是 flydsl/kernel 问题(mega_moe_v1 上少见),再查 flydsl 版本
```

---

## 9. 一页跑通 checklist

- [ ] 容器 atom-dev:latest 起好,8 卡可见,`import flydsl` + `kernels.mega_moe` OK(§2)
- [ ] 工作区 FlyDSL 在 `mega_moe_v1` 分支
- [ ] 部署 4 处:`flydsl_mega_experts.py`(import 用 `kernels.mega_moe`)、moe.py 两 hook、mori_prepare_finalize 两函数、model_runner warmup 打散(§3)
- [ ] perf 前:补权重释放(坑2)
- [ ] 步骤1 mori 基线 0.95 ✓ → 步骤2 bench PASS ✓ → 步骤3 fused gsm8k ≥0.90
- [ ] 精度过再 perf:tp+dp(原生 flydsl)+ fused,8k/1k & 1k/1k,注意 maxseqs(坑5)与权重释放(坑2)

---

> 状态标注(截至本文):步骤1 ✅ / 步骤2 ✅ / 步骤3 fused e2e 卡在**坑1(warmup 退化路由崩溃)**,尚未拿到 fused gsm8k 分数与 perf。修好坑1 后即可跑通。
