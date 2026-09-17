# Metrics review 实验归档

归档日期：2026-09-17。关联 [Metrics review](../../metrics_review_3ff61024_c2c68165.md) 和 [ATOM Wiki](../../atom_wiki.md)。

这里保存精简前的 metrics 微基准证据。原始脚本与结果从会话临时目录逐字节复制，后续阅读不再依赖 `/tmp`。这些数据测量观测包装的增量成本，不是实际 detokenizer 性能，也不是模型推理、吞吐或 GPU 性能 A/B。根据最新讨论，本轮不再安排实际性能 A/B。

## 归档内容

| 文件 | 用途 |
| --- | --- |
| [原始脚本](original/atom_metrics_review_bench.py.txt) | 保留当时的完整脚本，包括原工作区绝对路径和旧接口，不对原件做适配 |
| [MutexValue 原始结果](results/atom_metrics_review_bench_mutex.json) | 普通 Prometheus 存储的原始 JSON |
| [MmapedValue 原始结果](results/atom_metrics_review_bench_mmap.json) | multiprocess 存储的原始 JSON |
| [CPU 测试原始备份](original/cpu_metrics_tests_before_simplification.py.txt) | 此前工作区未跟踪的 CPU 测试，在精简实现修改它之前保存的原件；不是新增回归测试 |
| [复现入口](reproduce_benchmark.py) | 原脚本的便携版本：显式指定源码目录、校验被测源码，测量逻辑保持原样 |
| [manifest.json](manifest.json) | 提交范围、归档来源、测量口径、源码和原件 SHA-256 |
| [SHA256SUMS](SHA256SUMS) | 本目录资料的完整性校验清单，不含清单自身 |

`source/` 保存从 `c2c68165d02c0db0be46dcdb780570237359415c` 读取的四个被测模块快照：

- [ttft_breakdown.py](source/atom/entrypoints/openai/ttft_breakdown.py.txt)
- [streaming_dispatch.py](source/atom/entrypoints/openai/streaming_dispatch.py.txt)
- [cpu_metrics.py](source/atom/entrypoints/openai/cpu_metrics.py.txt)
- [run_labels.py](source/atom/model_engine/run_labels.py.txt)

原始脚本、旧测试与源码快照以 `.py.txt` 文本附件保存，内容逐字节保留，避免作为当前 Python 代码进入 CI 格式化和静态检查。它们是分析资料，不是可直接启动的完整 ATOM 副本。它们依赖的其余源码应从上述提交获取。归档时工作区 HEAD 为 `a673da60a6c0eb6892edfb88a39822eef34df4b8`，该 HEAD 的这四个模块与 review 终点一致；工作区正在进行的精简修改另计。

## 方法与结果边界

原始结果记录 Python 3.10.12；每项运行七次，取中位数，计时器为 `time.thread_time()`，单位为微秒/调用。通常每轮 50,000 次；CPU collector 为 100 次、进程树刷新为 5 次、API histogram 导出为 100 次。

脚本设置 `ATOM_TTFT_TRACE=0`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。multiprocess 组在 import Prometheus 前创建独立临时目录。detokenizer 固定返回 `x`，目的是隔离 metrics 包装成本。进程采样项只反映当时进程树，不能直接推广至实际推理部署。

依赖包版本和完整机器配置没有在原实验中记录，因此归档不补写推测值。新环境重跑所得数字不保证与历史结果相同，不能据此宣称端到端性能提升。原始结果也不包含本次随后完成的 trace 关闭路径优化。

## 可选的历史复现

使用安装好项目依赖的 Python 环境，从仓库根目录执行。下面的命令把历史源码解包到独立临时目录，不切换或覆盖当前工作区；结果写入新的临时目录，不覆盖归档。此处提供复现方法，本轮归档没有重新运行该微基准或任何性能 A/B。

```bash
metrics_archive="$PWD/cmq_docs/experiments/metrics_review_3ff61024_c2c68165"
metrics_replay_dir="$(mktemp -d /tmp/atom-metrics-replay.XXXXXX)"
mkdir -p "$metrics_replay_dir/source" "$metrics_replay_dir/results"
git archive c2c68165d02c0db0be46dcdb780570237359415c | tar -x -C "$metrics_replay_dir/source"

env -u PROMETHEUS_MULTIPROC_DIR python "$metrics_archive/reproduce_benchmark.py" \
  --repo "$metrics_replay_dir/source" > "$metrics_replay_dir/results/mutex.json"
env -u PROMETHEUS_MULTIPROC_DIR python "$metrics_archive/reproduce_benchmark.py" \
  --repo "$metrics_replay_dir/source" --multiprocess > "$metrics_replay_dir/results/mmap.json"
```

复现入口会校验四个被测模块的 SHA-256；如果传入当前已精简的工作区，会明确报出版本不匹配，避免把旧实验与新实现混用。

完整性校验从仓库根目录执行：

```bash
(cd cmq_docs/experiments/metrics_review_3ff61024_c2c68165 && sha256sum -c SHA256SUMS)
```

本目录随 `mengqng/metrics_simplification` 分支的 metrics 精简提交纳入 Git 版本管理。原有本地 Git exclude 规则保持不变，本次显式加入这些归档文件；以后新增的 Wiki 资料仍需显式加入 Git。远程同步需要另行 push。
