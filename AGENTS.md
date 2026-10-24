# 本机及本目录工作规则

在本 checkout 工作前，读取上级 [AGENTS.md](../AGENTS.md) 和 [GPU 技能](../.agents/skills/amd-gpu-budget/SKILL.md)。本文件确保从 ATOM 这个 Git 根目录启动的新会话也读取上级约束。

- 测试、脚本、说明和运行产物遵守 [仓库外辅助文件技能](../.agents/skills/repo-external-artifacts/SKILL.md)：放在上级 `investigation/` 下，本仓库只保留实现修改。
- 精度验收遵守 [V4.1 数值验收技能](../.agents/skills/dsv41-numerical-validation/SKILL.md)：浮点及量化数值容差内通过，不追逐位一致；通过代表性检查后推进性能工作，整网使用质量指标。

- 动态选择空闲 GPU；同一时间上级工作目录的所有会话、后台任务和容器合计最多占用两张物理 GPU，不固定编号。
- 用户已明确授权的本轮 TP4 mono/native 四卡测试例外按上级 `AGENTS.md` 执行；不得将该例外扩展成其他任务默认使用四卡。
- GPU 初始化前按实时所选 UUID 设置 `ROCR_VISIBLE_DEVICES`，保留子进程限制并登记分配；不得使用默认 TP8 示例。
- 启动 gfx1250 模型、测试或 kernel 前，读取 [本机兼容性与故障记录](../investigation/gfx1250-constraints.md)。gfx1250 母版路径通过 `ATOM_DSV41_GFX1250_REFERENCE_MONO=1` 显式启用，当前验收范围为 TP2 c1、FP8 index、完整 40 层；TP4 保持 trial。原 gfx950 路径不能直接用于 gfx1250，禁用 mono 也不能代替其他 gfx1250 启用修改。
- 本机配置记录 `SWDEV-588490`：禁止amdgpu卸载重载；当前 `gpu_recovery=0`。fault/hang后保存现场、清理本任务并主动核验GPU可用性；可用就继续有界隔离和重试，不因一次fault自动停工，也不要求先证明根因修复。不擅自reset、重载驱动或重启机器。具体流程及busy非零的健康检查例外遵守上级GPU技能。
- 仓库部分启动/停止脚本会按名字批量 kill 进程、清除共享缓存，并默认用 8 卡。使用前检查并限定清理到本任务的 PID/进程树，不能影响其他人的任务。
