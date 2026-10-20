# MiniMax-M3 P+D Gluon C32 实验

在已完成的 P-side Gluon C32（[run 39398997799](https://github.com/ROCm/ATOM/actions/runs/39398997799)，ATOM `741717be1f48900f25c1e3b8be454553da4b7777`）上，只追加 dense decode 到官方 Gluon unified attention 的接入。

实验分支：`mengqing/minimax-m3-gluon-pd-c32`。AITER 继续固定为 `4fe2800a5c252adff8ee28d199bea66f8b18efac`：官方 v0.1.26 Gluon kernel + split 策略，仅带已验证的 ATOM 原生 FP8 KV 格式与 scale 接口适配。

| 路径 | 上次 P-side Gluon C32 | 本次 P+D Gluon C32 |
| --- | --- | --- |
| P / cached dense prefill，含 draft | Gluon unified attention | 相同 |
| D / target dense speculative verification，q=4 | 原有 paged decode（该配置选 FlyDSL） | Gluon unified attention |
| D / EAGLE3 dense decode，初次 q=4、后续 q=1 | 原有 paged decode | Gluon unified attention |
| 57 层 sparse attention | MiniMax sparse 专用 kernel | 相同 |
| 无缓存的 dense prefill、profiling | 原有路径 | 相同 |

这里的 P+D 指 P/D 两侧对应的 dense paged attention。Gluon 由官方启发式决定是否 split，不强制所有 batch 都拆分。既不替换 sparse attention，也不切换到旧的 `pa_decode_gluon`。

## 固定的实验条件

- 8 × MI355X；P TP4 + D TP4，PP1。
- `MiniMax-M3-MXFP4` + `Inferact/MiniMax-M3-EAGLE3`，3 speculative tokens；不更换成 EAGLE-GQA。
- 原生 SHUFFLE FP8 KV，block size 128。Target 继续使用 per-token scales；draft 继续使用原来的 scalar scale。
- LMCache、Mooncake KV transfer、GPU/KV 预算、gather workspace 配置、图捕获形状、监控设置均沿用上次运行。
- AgentX C32，3600 秒，seed 42，393 条 dataset entries，1M context；相同镜像与 dispatch inputs。

矩阵检查仍对照 gather C32 [run 39034151209](https://github.com/ROCm/ATOM/actions/runs/39034151209)，只允许 AITER kernel pin 差异。相对上次 P-side Gluon C32，本次矩阵与 AITER pin 完全相同，运行时代码变化集中在 D 的 dense attention 接入。

## 代码与 graph padding

`atom/model_ops/attention_mha.py` 的 dense `dispatch_backend()` 在 decode 时选择 `paged_attention_gluon()`。Sparse 类自己的 dispatch 不变。实际调用位于 `atom/model_ops/gluon_prefill.py::run_gluon_decode()`，复用与 prefill 相同的原生页表、FP8 pool、scale view 和官方 kernel。

CUDA graph 的 padded query 不能直接使用 target 的平尾 `cu_seqlens_q`：官方 split reduce 按每个 query 归并，空 context 会导致 segment 数计算出现除零。适配层因此：

1. 根据 query tensor 的固定形状生成 q=1 或 q=4 的矩形 offsets，不捕获 Python `scheduled_bs`。
2. 将临时 context lengths 的最小值设为 qlen。ATOM `BlockTableState` 已将 padded 页表行清零，因此 dummy query 只访问合法的物理 page 0。
3. 依据 GPU `slot_mapping == -1` 将 padded 输出清零。Draft 推进后 padded context 可能为正，不能仅依靠 context 是否为零判断有效请求。使用填零而非乘零，避免保留未初始化 dummy KV 的 NaN。

只有 O(batch) 的临时元数据与输出 mask；不 gather KV、不转换或复制整个 KV pool、不更改 scheduler 元数据。这些适配操作的耗时计入实验结果。原来的 KV 预算保留，以便直接比较；启动后的实际 GPU blocks 仍需从 CI 日志核对。

## 验证

- CPU tests 覆盖原有 P 路径、native pool/scale alias、target/draft q=1/4、padding 的零长度与正长度、动态 live slots，以及 scheduler 元数据不被修改。
- GPU 启动前的 `check_m3_gluon_adapter.py` 保留原来的 9 个 writer/attention 检查，新增 target/draft × q=1/4 的 4 组 graph 检查。
- 每组 graph 在 2→1→3→0 个有效请求之间 replay，同时改变 query、context lengths 和物理页表；与独立 Torch attention 参考结果对照，并检查 eager 与 graph 一致、padded 输出为零。
- GPU gate 要求确实调用官方 gfx950 Gluon，并覆盖 split；禁止 attention 回退到 Triton。官方 reduce 仍按 v0.1.26 的实现调度。

CPU 与离线编译无法代替 GPU 数值检查。CI 必须先通过这些 GPU gate，才开始加载模型和跑 C32；性能与正确性结果以本次 CI 为准。
