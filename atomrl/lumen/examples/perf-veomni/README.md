# VeOmni Qwen-Image / Wan2.1 训练性能：配置调优 + Lumen 优化

> English version: [README_EN.md](README_EN.md)

本文给出 VeOmni 上两个扩散模型训练的三级性能阶梯：**示例原样 → 只改配置能调出的最优 →
在配置最优之上叠加 Lumen 优化后的最优**。配置级只改 VeOmni 的命令行参数；可复用优化通过
`LumenConfig` 启用，`veomni_patches/` 只保留 VeOmni 特定 workaround。VeOmni
与已安装 AITER 的源码都不修改。当前 VAE 卷积路径位于
`atomrl.lumen.ops.conv`。

> 下方性能表由旧的实验性 FlyDSL 路径测得，只作为历史记录，不能作为当前
> 包内卷积实现的验证数据；当前路径需要重新测试。

---

## 1. 结论

| 场景 | 原始 | 配置最优 | Lumen 最优 | 配置带来 | Lumen 在配置之上带来 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen-Image 1024² 在线训练 | 1.855 s | 0.971 s | **0.680 s** | 1.91× | 1.43× | **2.73×** |
| Wan2.1 81×368×544 在线训练 | 3.569 s | 2.654 s | **1.160 s** | 1.34× | 2.29× | **3.08×** |
| Wan2.1 视频嵌入（`offline_embedding`） | 2.027 s | 1.493 s | **0.340 s** | 1.36× | 4.39× | **5.96×** |

单位为秒/步（8 卡、global batch 8、10 步；去掉第 1 步后取中位数；3 次重复按轮次交替）。

- **配置**的收益来自两处：一是把示例 yaml 里偏离 VeOmni 默认值的设置改回来，即
  `empty_cache_steps` 从 1 改回 500，Wan 的 `num_workers` 从 0 改回 2、`pin_memory` 改回 true；
  二是用显存换时间，即关梯度检查点，Qwen 再关 reshard。
- **Lumen** 的收益主要来自两处：attention 反向改用 aiter 的融合 kernel（Qwen −12.5%，Wan −18.8%），
  以及 VAE 转 BF16 并改走 FlyDSL 卷积（Wan 训练合计 −44%，视频嵌入 −77%）。
- 改为"先离线嵌入、再从嵌入训练"，训练步时还能再降：Qwen-Image **0.613 s**（比在线最优 −9.9%），
  Wan2.1 **0.824 s**（−29.0%），约 1 个 epoch 回本（§2.6）。

适用范围：8× AMD Instinct MI350X（gfx950），ROCm 7.14 / torch 2.12，VeOmni `573848a`，FSDP2 全参数
SFT，Qwen-Image 20B DiT @1024²、Wan2.1-T2V-1.3B @81 帧 368×544。

---

## 2. 数据

### 2.1 阶梯

每一级在上一级的基础上**只加一项**改动，所以每行的"相对上一级"就是该项在前面所有改动之上的增量。
A、B 级只改 VeOmni 配置；L 级保持配置最优不变，逐项加 `LUMEN_PATCH`。p 为相对上一级的
Mann-Whitney 秩和检验（每步样本跨重复池化）；散布 = 3 次重复各自中位数的极差 / 中位数。
级号在两个模型间对齐，所以某一级不适用时会出现空号。

**Qwen-Image 1024² 在线训练**

| 级 | 改动 | 步时 | 散布 | 相对上一级 | p | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|
| A0 | 示例原样（`empty_cache_steps=1`、梯度检查点开） | 1.855 s | 0.4% | — | — | 38.8 GB |
| B1 | `--train.empty_cache_steps 500` | 1.205 s | 1.0% | −35.0% | 3e-9 | 38.8 GB |
| B2 | `--train.gradient_checkpointing.enable false` | 1.045 s | 0.8% | −13.3% | 3e-9 | 72.2 GB |
| B3 | `--train.accelerator.fsdp_config.reshard_after_forward false` | **0.971 s** | 0.6% | −7.1% | 8e-7 | 108.9 GB |
| L1 | `vae_bf16`：VAE 以 BF16 运行 | 0.889 s | 0.6% | −8.4% | 5e-7 | 108.7 GB |
| L2 | `vae_conv`：VAE 卷积走 aiter FlyDSL | 0.881 s | 1.4% | −0.9%，小于散布，不计 | 7e-4 | 108.8 GB |
| L4 | `sdpa_efficient`：attention 反向走 aiter 融合 `fmha_bwd` | 0.771 s | 3.0% | −12.5% | 3e-8 | 108.8 GB |
| L5 | `attn_triton_fwd`：DiT attention 前向走 aiter Triton | 0.742 s | 0.8% | −3.8% | 4e-7 | 108.8 GB |
| L6 | `rmsnorm_fuse`：RMSNorm 前向融合为 1 个 kernel | 0.702 s | 2.2% | −5.3% | 4e-7 | 100.4 GB |
| L7 | `local_adamw`：AdamW 在本地分片上执行 | **0.680 s** | 2.7% | −3.1% | 3e-6 | 100.4 GB |

**Wan2.1 在线训练**

| 级 | 改动 | 步时 | 散布 | 相对上一级 | p | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|
| A0 | 示例原样（另有 `num_workers 0`、`pin_memory false`） | 3.569 s | 0.5% | — | — | 21.2 GB |
| B1 | `--train.empty_cache_steps 500` | 3.101 s | 0.3% | −13.1% | 4e-8 | 21.2 GB |
| B2 | `--train.gradient_checkpointing.enable false` | 2.827 s | 0.5% | −8.8% | 5e-8 | 70.9 GB |
| B4 | `--data.dataloader.num_workers 2 --data.dataloader.pin_memory true` | **2.654 s** | 0.4% | −6.1% | 5e-8 | 70.9 GB |
| L1 | `vae_bf16` | 1.633 s | 0.3% | −38.5% | 3e-9 | 70.7 GB |
| L2 | `vae_conv_video`：含 T>1 与 feature cache 的卷积 | 1.535 s | 1.0% | −6.0% | 3e-9 | 70.9 GB |
| L3 | `conv_pad_in_kernel`：因果卷积的 padding 交给 kernel | 1.495 s | 0.4% | −2.6% | 3e-9 | 70.9 GB |
| L4 | `sdpa_efficient` | 1.215 s | 0.9% | −18.8% | 8e-7 | 70.9 GB |
| L5 | `attn_triton_fwd` | 1.170 s | 1.0% | −3.7% | 6e-6 | 70.9 GB |
| L7 | `local_adamw` | **1.160 s** | 1.5% | −0.8%，小于散布，不计 | 4e-3 | 70.9 GB |

**Wan2.1 视频嵌入**（`offline_embedding` 只跑 VAE 与文本编码器，没有 DiT、反向和优化器）

| 级 | 改动 | 步时 | 散布 | 相对上一级 | p | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|
| A0 | 示例原样 | 2.027 s | 0.3% | — | — | 14.7 GB |
| B1 | `--train.empty_cache_steps 500` | 1.677 s | 0.3% | −17.2% | 7e-7 | 14.7 GB |
| B4 | 数据加载 `num_workers 2`、`pin_memory` | **1.493 s** | 0.2% | −11.0% | 3e-9 | 14.7 GB |
| L1 | `vae_bf16` | 0.475 s | 0.5% | −68.2% | 3e-9 | 12.8 GB |
| L2 | `vae_conv_video` | 0.376 s | 4.3% | −20.7% | 5e-8 | 12.9 GB |
| L3 | `conv_pad_in_kernel` | **0.340 s** | 5.3% | −9.7% | 3e-9 | 12.8 GB |

说明：

- 有意跳过的级：Qwen 的数据加载不是瓶颈（B4）；Wan 关 reshard 此前实测无效（B3）；RMSNorm 融合不适用
  于 Wan，它的 DiT 用的 `torch.nn.RMSNorm` 本来就是融合 kernel（L6）；Qwen 的卷积全部走 T=1 转 conv2d，
  没有要交给 kernel 的时间维 padding（L3）；嵌入任务没有反向与优化器（B2、L4–L7）。
- **两处增量不计为收益**：Qwen 的 L2（−0.9%）与 Wan 的 L7（−0.8%）都小于组内散布。Qwen 的 VAE 卷积在
  L1 之后每步只剩约 13 ms GPU 时间（§2.2），FlyDSL 的主场是 Wan 的视频 VAE。
- 第 1 步（含 FlyDSL JIT、`torch.compile` 和各类 autotune）单独看：Qwen 配置级 32–34 s，Lumen 级
  36–39 s；Wan 各级 29–32 s；嵌入任务 17–20 s。均在 FlyDSL JIT 磁盘缓存已热的条件下测得；冷缓存时
  Wan 首步另需数分钟编译。
- 显存：关梯度检查点与关 reshard 是用显存换时间（Qwen 38.8 → 108.9 GB，Wan 21.2 → 70.9 GB）。
  RMSNorm 融合后峰值反而低 8.4 GB，因为编译后的前向留给反向的中间张量更少。
- 测量期间 GPU 0 被他人设为 `perf_determinism`；运行中采样，其忙时平均频率为 2180 MHz，其余 7 张卡
  为 2155–2192 MHz，没有降频。

### 2.2 每项优化的 kernel 级证据

取自各级单独的 profiler 运行（第 4–5 步，无 Python 栈），单位为 rank 0 每步的 GPU 毫秒数，由
`kernel_diff.py` 比较相邻两级得出。profiler 会让步时略微变长，所以这里只用于看"哪个 kernel 换成了
哪个"，步时以 §2.1 为准。

| 项 | 模型 | 改前 → 改后 |
|---|---|---|
| B1 `empty_cache_steps 500` | Qwen | A0 每步约 660 ms 的 GPU 空转坑（其中 `hipFree` 220–260 ms），到 B3 时最大的坑只剩 19 ms |
| | Wan | A0 每步 450–465 ms 的空转坑，到 B4（含数据加载改动）时最大的坑只剩 8 ms |
| B2 关梯度检查点 | Qwen | GEMM 调用 600 → 300、`attn_fwd` 149 → 89：前向重算消失 |
| B3 关 reshard | Qwen | RCCL 312 → 203 ms（A0→B3）：前向后不再重新 all-gather |
| L1 `vae_bf16` | Qwen | CK FP32 `grouped_conv` 20 次 79.3 ms → CK BF16 卷积约 12.8 ms |
| | Wan | CK FP32 `grouped_conv` 460 次 1078.3 ms → CK BF16 卷积 188.9 ms；FP32 转置 73.7 ms 消失 |
| L2 FlyDSL 卷积 | Qwen | CK BF16 卷积约 12.8 ms → `conv3d_implicit` 22 次 3.95 ms + 转置 0.69 ms |
| | Wan | CK BF16 卷积 460 次 187.9 ms + 转置约 48 ms → `conv3d_implicit` 525 次 146.5 ms + 转置 19.2 ms |
| L3 padding 交给 kernel | Wan | `aten::pad` 拷贝与填充 1131 次 40.7 ms → 295 次 6.6 ms；`conv3d_implicit` 146.5 → 140.7 ms（L3 专用调优表，§2.5） |
| L4 `sdpa_efficient` | Qwen | AOTriton `bwd_kernel_dk_dv` + `bwd_kernel_dq` 202.8 ms → aiter `fmha_bwd_hd128_bf16_a32` + postprocess 77.2 ms |
| | Wan | 同上 607.7 ms → 313.9 ms |
| L5 `attn_triton_fwd` | Qwen | `attn_fwd` 61 次 72.2 ms → 1 次 7.8 ms（剩下的是 VAE）；aiter Triton `_attn_fwd` 60 次 23.5 ms |
| | Wan | `attn_fwd` 81 次 170.3 ms → 21 次 20.4 ms；Triton `_attn_fwd` 60 次 87.1 ms |
| L6 `rmsnorm_fuse` | Qwen | 逐元素与归约 kernel 每步少 120–240 次调用，新增 1 个 Triton 融合 kernel（120 次 2.3 ms）；GPU 合计 −48.5 ms |
| L7 `local_adamw` | Qwen | kernel 无变化（主机侧优化）；优化器阶段主机开销 44.5 → 6.4 ms |
| | Wan | 优化器阶段主机开销 20.1 → 2.8 ms |

每次运行结束时，示例入口还会打印补丁的实际生效情况，作为"补丁已应用"之外的第二道证据。以 Lumen 最优
配置为例，每步的统计如下：

- Qwen：aiter 卷积查调优表 22 次，全部命中；SDPA 有 60 次改走 Triton 前向；RMSNorm 有 120 次走融合路径。
- Wan：525 次卷积中 315 次精确命中调优表，其余 210 次落在 6 个未过 3% 门槛的形状上，按规则继续用启发式。

### 2.3 数值

按性质分三档：

| 档位 | 优化项 | 证据 |
|---|---|---|
| **逐位一致** | B1、B2、B3 | Qwen 的 loss 与 grad_norm 10/10 步与 A0 逐位相同 |
| | B4 数据加载 | Wan 在 8 个 rank 上 80/80 个样本的输入字节相同、顺序相同（`LUMEN_BATCH_HASH=1`） |
| | L3 `conv_pad_in_kernel` | Wan VAE latent 与预 pad 路径逐位相同（81×368×544）；同一轮训练中 L2 与 L3 的 loss 10/10 步相同 |
| | L7 `local_adamw` | 在确定性的 B3 上，loss 与 grad_norm 10/10 步与 stock AdamW 逐位相同 |
| **确定地改变数值** | L1 `vae_bf16` + L2 FlyDSL 卷积 | 对 FP32 encode 的 SNR：Qwen 50.4 dB（原生 BF16 为 50.3 dB），Wan 50.7 dB（原生 BF16 为 50.4 dB）；各自的 3 次重复逐位一致 |
| | L5 `attn_triton_fwd` | 输出与 dQ/dK/dV 对 FP32 为 52.9/52.5/52.6/52.6 dB，与 efficient 后端完全相同；LSE 误差 1.9e-6 |
| | L6 `rmsnorm_fuse` | 对 FP32 为 55.6 dB，原实现为 52.6 dB（中间结果保持 FP32，更准） |
| **引入非确定性** | L4 `sdpa_efficient` | 融合反向用 FP32 原子累加 dQ（kernel 名中的 `a32`）；Qwen 同配置的重复间 loss 最大相差 0.66%，而 A0–L2 各级的重复间为 0 |

Wan 的 loss 本身不可复现（示例原样的配置在重复间就最多相差 1.4%），所以 Wan 的数值判定只用 latent
逐位比较、SNR 和输入数据哈希，不看 loss。

### 2.4 GPU 时间占比：收益从哪来

取自 profiler 运行（rank 0，ms/步）。`trace_report.py` 会把 CK 卷积归到 GEMM、把 aiter `fmha_bwd` 归到
other，下表已按实际类别更正。

| 类别 | Qwen A0 | Qwen B3 | Qwen L7 | Wan A0 | Wan B4 | Wan L7 | 嵌入 B4 | 嵌入 L3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 步时（profiler 下） | 1921 | 1001 | 746 | 3384 | 2650 | 1157 | 1500 | 377 |
| GPU 忙碌 | 57% | 87% | 83% | 85% | 98% | 96% | 98% | 88% |
| attention | 322 | 276 | 110 | 932 | 782 | 423 | 34 | 21 |
| VAE 卷积 | 79 | 79 | 5 | 1082 | 1082 | 160 | 1082 | 159 |
| GEMM | 330 | 251 | 252 | 240 | 193 | 154 | 51 | 14 |
| 通信（RCCL） | 312 | 203 | 197 | 44 | 31 | 44 | 0 | 9 |
| 逐元素与其他 | 301 | 215 | 166 | 509 | 448 | 277 | 279 | 106 |
| 优化器 | 30 | 26 | 28 | 3 | 3 | 3 | — | — |

- **两个模型的瓶颈几乎互补。** Qwen 的 DiT 大（20B），VAE 只处理一帧图像，所以时间花在 attention、GEMM
  和通信上；Wan 的 DiT 小（1.3B），但 VAE 要编码 81 帧视频，配置调好之后 VAE 卷积是最大的一项（42%）。
  所以同一套 VAE 优化（L1–L3）在 Wan 上值 −44%，在 Qwen 上只值 −9%。
- B1 的收益在 trace 里表现为空转坑：Qwen A0 每步有一个约 660 ms 的 GPU 空转坑，占全部空转时间的 89%，
  其中 220–260 ms 花在 `hipFree` 上，即 `empty_cache_steps=1` 让每一步都把缓存的显存逐块还给驱动。
  从 A0 到 B3，GPU 忙碌率由 57% 升到 87%。
- 在 Lumen 最优配置下，剩余的大头是：Qwen 的通信（GPU 时间的 25%，与计算争用 CU）；Wan 的 attention
  （38%，其中 `fmha_bwd` 311 ms）。

### 2.5 卷积：Lumen 包内实现

`LumenConfig(aiter_conv=True)` 通过 `atomrl.lumen.ops.conv` 接管支持的
`nn.Conv2d` 和 `nn.Conv3d`。Lumen 负责 autograd 安全路由：前向调用 AITER
公共卷积 API，反向使用 PyTorch gradient 算子。示例不再向 AITER 安装目录
复制文件、修改 `AITER_CONFIGS` 或 monkey-patch AITER 模块。

### 2.6 离线嵌入工作流（可选）

| | Qwen-Image | Wan2.1 |
|---|---:|---:|
| 在线训练（Lumen 最优） | 0.680 s | 1.160 s |
| 一次性嵌入，每步 8 个样本（VAE 补丁同 Lumen 最优） | 0.059 s | 0.340 s |
| 从嵌入训练（`offline_training`） | **0.613 s**（−9.9%，p 3e-9） | **0.824 s**（−29.0%，p 4e-8） |
| 回本点 | 0.88 epoch | 1.01 epoch |

回本点 = 嵌入耗时 / (在线步时 − 离线步时)：训练超过约 1 个 epoch，"先嵌入再训练"就更快。前提是训练时
不需要在线的数据增强。`offline_training` 会把样本重新分配到不同的 rank 和步：实测 80 个样本的嵌入与在线
计算逐字节相同，只是顺序被重排，所以逐步 loss 无法与在线训练逐一对照，只在统计上等价。Qwen 的
`offline_training` 必须启用 `qwen_offline_fix`，否则第一步就崩溃（VeOmni `573848a` 的缺陷，见 §3.7）。

---

## 3. Runbook

所有命令都在容器内执行，路径沿用 `env.sh` 的默认布局：一个宿主目录挂载为 `/work`。

### 3.1 准备容器

需要单节点 8× MI350X / MI355X（gfx950）、ROCm 驱动，以及约 200 GB 空间（两个模型加数据）。

```bash
export ROOT=/path/to/fast-storage/perf-veomni
mkdir -p "$ROOT"
docker pull amdagi/veomni:rocm7.14_torch2.12_py3.12
docker run -d --name perf-veomni \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --ipc host --shm-size 32g --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable --ulimit memlock=-1:-1 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -v "$ROOT":/work -w /work \
  amdagi/veomni:rocm7.14_torch2.12_py3.12 sleep infinity

dx() { docker exec -w /work perf-veomni bash -lc "$*"; }
export EX=/work/Lumen/examples/perf-veomni
```

`HIP_VISIBLE_DEVICES` 与 `CUDA_VISIBLE_DEVICES` 必须同时设置，单卡脚本也一样，否则两者冲突会直接崩溃。

### 3.2 代码与环境

```bash
dx 'git clone https://github.com/ROCm/ATOM.git /work/ATOM &&
    git -C /work/ATOM checkout main &&
    ln -s /work/ATOM/atomrl/lumen /work/Lumen'
dx 'git clone https://github.com/ByteDance-Seed/VeOmni.git /work/VeOmni &&
    git -C /work/VeOmni checkout 573848a00fcd7329c2411346c6f4a983e9f67e3f'
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/setup_env.sh"
```

`setup_env.sh` 以 `--no-deps` 安装 VeOmni，并验证 Lumen 卷积入口。期望末尾输出：

```text
Lumen convolution: available (must be available)
SETUP OK
```

不要执行 `uv sync`、`pip install -e '.[gpu]'`，也不要安装任何 torch、triton、`cu12`/`cu13`/`nvidia-*` 包。

### 3.3 模型与数据

```bash
# Qwen-Image，以及 80 条样本（10 步 × 8 个 rank）
dx ". $EX/env.sh && hf download Qwen/Qwen-Image --local-dir \$QWEN_IMAGE_DIR"
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/make_data.py 80 \$DATA_DIR/train_80.jsonl"

# Wan2.1-T2V-1.3B，以及 Tom-and-Jerry 公开数据集的前 400 条
dx ". $EX/env.sh && hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir \$WAN_DIR"
dx '. /work/Lumen/examples/perf-veomni/env.sh && RAW=$WORK/data/tom-and-jerry && mkdir -p "$RAW" &&
    hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --repo-type dataset \
      --include captions.txt --local-dir "$RAW" &&
    hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --repo-type dataset \
      --include videos.txt --local-dir "$RAW" &&
    head -n 400 "$RAW/videos.txt" | xargs -n 25 hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset \
      --repo-type dataset --local-dir "$RAW" --quiet'
```

VeOmni 的转换脚本要求 `captions.txt` 与 `videos.txt` 一一对应，所以先按实际下载成功的视频生成子集，再转成
parquet：

```bash
docker exec -i -w /work perf-veomni python3 - <<'PY'
from pathlib import Path
src, dst = Path('/work/data/tom-and-jerry'), Path('/work/data/tom-and-jerry-subset')
dst.mkdir(parents=True, exist_ok=True)
if not (dst / 'videos').exists():
    (dst / 'videos').symlink_to(src / 'videos', target_is_directory=True)
pairs = zip(src.joinpath('videos.txt').read_text().splitlines(), src.joinpath('captions.txt').read_text().splitlines())
keep = [(v, c) for v, c in list(pairs)[:400] if (src / v).exists()]
dst.joinpath('videos.txt').write_text(''.join(f'{v}\n' for v, _ in keep))
dst.joinpath('captions.txt').write_text(''.join(f'{c}\n' for _, c in keep))
assert len(keep) >= 80, len(keep)
PY
dx ". $EX/env.sh && python3 \$VEOMNI_DIR/scripts/multimodal/convert_data/tom-and-jerry.py \
  --dataset_path \$WORK/data/tom-and-jerry-subset --output_dir \$WAN_DATA_DIR"
```

### 3.4 单卡验收（先于任何 8 卡运行）

```bash
G="HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=\$LUMEN_PYTHONPATH"
dx ". $EX/env.sh && $G python3 \$EXAMPLE_DIR/verify_aiter_conv.py --model qwen"
dx ". $EX/env.sh && $G python3 \$EXAMPLE_DIR/verify_aiter_conv.py --model wan --padk"
dx ". $EX/env.sh && $G python3 \$EXAMPLE_DIR/verify_sdpa_triton_fwd.py"
dx ". $EX/env.sh && $G python3 \$EXAMPLE_DIR/verify_rmsnorm_fuse.py"
dx ". $EX/env.sh && cd \$EXAMPLE_DIR && $G python3 -m pytest -q test_patches.py"
```

期望：打补丁后 latent 对 FP32 的 SNR 与原生 BF16 同量级（Qwen 50.4 vs 50.3 dB，Wan 50.7 vs 50.4 dB），
pad-in-kernel 与 pre-pad 逐位一致（`bit-identical`）；Wan 在 L3 下每次 encode 有 315/525 次命中
调优表；attention 两条路径的 SNR 都在 52.5–52.9 dB；RMSNorm 融合后前向从 8 个 kernel 变为 1 个，SNR
55.6 dB（原实现 52.6 dB）；测试全部通过。

### 3.5 三级阶梯

```bash
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/ladder.sh qwen"      # 10 级 × 3 次，约 70 分钟
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/ladder.sh wan"       # 10 级 × 3 次，约 50 分钟
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/ladder.sh wanemb"    # 6 级 × 3 次，约 25 分钟
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/ladder.sh offline"   # 可选工作流，需先跑完 wanemb
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/summarize.py --control pvq-A0 --chain -- \
      pvq-A0 pvq-B1 pvq-B2 pvq-B3 pvq-L1 pvq-L2 pvq-L4 pvq-L5 pvq-L6 pvq-L7"
```

`ladder.sh` 按轮次跑重复（每一级跑一次，再全部跑第二次……），机器的缓慢漂移会均匀落在各级上。运行结束时
它会打印对应的 `summarize.py` 命令。`--chain` 表示逐级比较：每行对比上一行，并给出相对 A0 的累计倍数。
`DRY_RUN=1` 只打印命令不执行；`ladder.sh qwen A0 B3 L7` 只跑列出的几级。脚本运行期间不要修改它，因为
bash 是边读边执行的。

### 3.6 每一级的完整命令

阶梯里的每一级都等价于下面这条命令，只是 `LUMEN_PATCH` 和末尾的参数不同。Wan 的命令改用
`run_wan_10steps.sh`，并加上 `WAN_TASK=online_training` 或 `WAN_TASK=offline_embedding`：

```bash
dx ". $EX/env.sh && RUN_NAME=<名字> RUN_SUFFIX=r1 LUMEN_PATCH=<补丁> \
      bash \$EXAMPLE_DIR/run_10steps.sh custom <VeOmni 参数...>"
```

| 场景 | 级 | `LUMEN_PATCH` | VeOmni 参数 |
|---|---|---|---|
| 共同 | A0 | （空） | （无） |
| 共同 | B1 | （空） | `--train.empty_cache_steps 500` |
| Qwen / Wan | B2 | （空） | B1 + `--train.gradient_checkpointing.enable false` |
| Qwen | B3 = 配置最优 | （空） | B2 + `--train.accelerator.fsdp_config.reshard_after_forward false` |
| Wan | B4 = 配置最优 | （空） | B2 + `--data.dataloader.num_workers 2 --data.dataloader.pin_memory true` |
| 嵌入 | B4 = 配置最优 | （空） | B1 + 数据加载两项 |
| Qwen | L7 = Lumen 最优 | `vae_bf16,vae_conv,sdpa_efficient,attn_triton_fwd,rmsnorm_fuse,local_adamw` | 同 B3 |
| Wan | L7 = Lumen 最优 | `vae_bf16,vae_conv_video,conv_pad_in_kernel,sdpa_efficient,attn_triton_fwd,local_adamw` | 同 B4 |
| 嵌入 | L3 = Lumen 最优 | `vae_bf16,vae_conv_video,conv_pad_in_kernel` | 同 B4 |

中间各级按 §2.1 的顺序每次加一个 `LUMEN_PATCH` 项。离线工作流：先用 `--train.training_task offline_embedding
--data.offline_embedding_save_dir <目录>` 生成嵌入，再用 `--train.training_task offline_training --data.train_path
<目录>` 训练；训练时不加 VAE 补丁（不加载 VAE），Qwen 另加 `qwen_offline_fix`。

### 3.7 `LUMEN_PATCH` 取值

| 取值 | 作用 | 代码位置 |
|---|---|---|
| `vae_bf16` | 冻结的 VAE 以 BF16 运行，`encode()` 出口把 latent 转回 FP32 | `train_dit_lumen.py` |
| `vae_conv` / `vae_conv_video` | VAE 卷积走 Lumen 的 AITER-backed Conv2d/Conv3d 路径；前者用于图像（T=1 精确转 conv2d），后者还覆盖 T>1 与 feature cache | `lumen_vae_conv.py`、`atomrl.lumen.ops.conv` |
| `conv_pad_in_kernel` | 因果卷积不再先 `F.pad` 拷贝整块，改由 kernel 用掩码处理对称 padding | `lumen_vae_conv.py` |
| `sdpa_efficient` | 关闭 SDPA 的 flash 后端；在 ROCm 上反向改走 aiter 融合 `fmha_bwd`。**引入非确定性** | `atomrl.lumen.ops.attention.hf_patch` |
| `attn_triton_fwd` | 仅对 DiT 的 attention（无 mask、非因果、BF16、head_dim 128）改用 aiter Triton 前向，反向仍用 efficient；需同时加 `sdpa_efficient` | `atomrl.lumen.ops.attention.hf_patch` |
| `rmsnorm_fuse` | diffusers `RMSNorm` 的前向经 `torch.compile` 融合；只处理带权重的 RMSNorm，从不碰 LayerNorm | `atomrl.lumen.ops.normalization.rmsnorm` |
| `local_adamw` | fused AdamW 直接在 FSDP2 本地分片上执行，绕过 DTensor 分发；结果逐位一致 | `atomrl.lumen.quantize.local_adamw` |
| `qwen_offline_fix` | 修复 VeOmni `573848a` 的缺陷：Qwen-Image `offline_training` 在未加载 VAE 时读取 `self.vae.config` 导致首步崩溃；应由上游修复 | `veomni_patches/qwen_offline.py` |

后五项在 trainer 构建模型之前、VeOmni 被导入之前装上，默认全部关闭，只有写进 `LUMEN_PATCH` 才会启用。
`LUMEN_BATCH_HASH=1` 打印 condition model 每个输入
的哈希，仅用于校验，不要拿来计时。

### 3.8 证据采集

```bash
# 任一级的 trace（第 4–5 步），写到 $OUT_DIR/trace/<RUN_NAME>
dx ". $EX/env.sh && RUN_NAME=prof-w-L3 WAN_TASK=online_training \
      LUMEN_PATCH=vae_bf16,vae_conv_video,conv_pad_in_kernel bash \$EXAMPLE_DIR/profile.sh wan \
      --train.empty_cache_steps 500 --train.gradient_checkpointing.enable false \
      --data.dataloader.num_workers 2 --data.dataloader.pin_memory true"

python3 kernel_diff.py <A.json.gz> <B.json.gz>           # 哪些 kernel 出现、消失或变慢
python3 kernel_blame.py <trace> --category elementwise   # 按发起的 aten 算子归因（F.pad 拷贝）
python3 trace_report.py <trace>                          # GPU 忙碌率、类别占比、空转坑
python3 opt_phase.py <trace 目录> ...                    # 优化器阶段的主机开销
python3 loss_equal.py <参考 run> <run> ...               # loss / grad_norm 是否逐位一致
python3 determinism.py <级> ...                          # 同一级的重复之间是否一致
python3 compare_batch_hash.py <日志 A> <日志 B>          # 两次运行的输入是否逐字节相同
```

### 3.9 验收标准

- 每个 run 的 `.meta.txt` 中 `EXIT_CODE : 0`，VeOmni 与 Lumen 的 dirty 计数均为 0。
- 带 VAE 补丁的 run，日志里要有 `Lumen convolution: AITER`，且
  `aiter conv calls` 中 `torch` 为 0。只看到 `patched` 不足以证明 kernel
  真的在运行。
- 结论中每项 L 优化都要有 kernel 级证据（§2.2），不能只凭步时推断。
- 一项增量只有同时大于两级的组内散布、且 p < 0.05 才算收益。
- 同一时间只能跑一个 8 卡任务（runner 用 `$LOG_DIR/.run.lock` 保证），开跑前用 `rocm-smi --showpids`
  确认没有他人的负载。

### 3.10 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| AITER 卷积导入失败 | 已安装 AITER 缺少公共 Conv2d/Conv3d API | 安装当前兼容的 AITER 版本 |
| BF16 VAE 后 timestep 查找 `IndexError` | `encode()` 输出没有转回 FP32 | 使用本示例的入口与 `vae_bf16` |
| 实际步数少于目标，但 exit 0 | 数据行数少于 `步数 × rank 数` | 准备足够的样本 |
| `Conflicting visibility of agent-N` | 两个 GPU 可见性变量不一致 | 同时设置 HIP 与 CUDA 的可见性变量 |
| Qwen `offline_training` 首步 `'NoneType' ... 'config'` | VeOmni 缺陷 | 在 `LUMEN_PATCH` 中加 `qwen_offline_fix` |
| Wan 首次运行长时间停在第 1 步 | 22 个卷积形状的 FlyDSL JIT | 等待编译完成；磁盘缓存会被后续运行复用 |

---

## 4. 文件索引

| 文件 | 作用 |
|---|---|
| `ladder.sh` | 三级阶梯的全部级定义与按轮次交替的重复 |
| `run_10steps.sh` / `run_wan_10steps.sh` | 单次 8 卡 10 步运行；`custom` 模式读取 `LUMEN_PATCH`，其余参数透传给 VeOmni |
| `profile.sh` | 用 VeOmni 自带的 profiler 采任一级的 trace |
| `train_dit_lumen.py` | VeOmni 训练入口的包装：按 `LUMEN_PATCH` 装配补丁，并报告后端路由计数 |
| `lumen_vae_conv.py` | VAE 卷积改写：T=1 精确转 conv2d、T>1 替换底层 kernel、padding 交给 kernel |
| `veomni_patches/` | 仅保留 VeOmni 特定的 `qwen_offline.py`；卷积、attention、RMSNorm 和 optimizer 优化直接调用 Lumen |
| `verify_aiter_conv.py` / `verify_sdpa_triton_fwd.py` / `verify_rmsnorm_fuse.py` | 单卡数值与性能校验 |
| `test_patches.py` | `veomni_patches/` 的单元测试 |
| `summarize.py` | 步时中位数、散布、p、峰值显存、loss 偏差；`--chain` 逐级比较 |
| `kernel_diff.py` / `kernel_blame.py` / `trace_report.py` / `opt_phase.py` | trace 分析 |
| `loss_equal.py` / `determinism.py` / `compare_batch_hash.py` | 数值与数据一致性 |

全部代码都在本目录里，Lumen 库本身没有改动。
