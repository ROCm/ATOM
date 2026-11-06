# DeepSeek-V4.1 fusedmoe：R13 完整服务配置

唯一交付分支为 `yifehuan/dsv41_fusedmoe`。模型实现位于 `atom/models/deepseek_v41/`；固定运行profile位于 `tools/dsv41/r13_profile.json`，运行入口为 `tools/dsv41/run.py`。不依赖个人目录中的历史experiment入口。

该配置为 owner/packed 注意力＋retained prefill＋grouped attention＋预量化 FFN＋EP4＋local router/packet＋固定专家分配的完整组合，面向 random 8192/1024、C256 工作负载。固定专家分配只针对该工作负载，不承诺其他路由分布获得同样收益。

## 环境

- 4张MI355X/gfx950；本次使用8卡独占节点中的0–3卡。不要与其他GPU工作负载重叠。
- 历史镜像：`sha256:32c81e56e4680065b6ea3e0ca8363a00cd3ccb932d8990e83d799b8bbb58a4b3`。
- AITER：`729feefb7210ebc3a87ebd00a44e0c43f30bf3f2`，使用该依赖原生算术。grouped attention的私有native包装代码在本分支，保留原生device模板与编译参数。
- checkpoint在容器内挂载为`/models/DeepSeek-V4.1-Flash`。Python环境需具备对应torch、Triton、FlyDSL、AITER和benchmark依赖。
- attention TP4、routed专家EP4、shared TP4，BF16 KV、FP4 index、DSpark5、level3/FULL、max-num-seqs256、memory utilization0.85。

## 运行

下列命令在已配置GPU/Python环境的仓库根执行。`/path/to/aiter`必须是上述依赖；输出路径须是本次独立目录，不覆盖旧结果。

```bash
python tools/dsv41/run.py preflight \
  --aiter-source /path/to/aiter --output /path/to/run/entry
python tools/dsv41/run.py client-preflight \
  --aiter-source /path/to/aiter --output /path/to/run/client
python tools/dsv41/run.py benchmark \
  --aiter-source /path/to/aiter --output /path/to/run/B1 \
  --cache-dir /path/to/private-cache
```

`benchmark`会启动并正常退出一个服务器，执行593个预热请求、2560个正式请求。输入random8192、输出1024、range0.8、正式seed2026102518；ignore-eos，实际总token数以原始请求结果为准。候选预热顺序为4/512/64/12/1，覆盖owner小容量及完整图键；主线为512/64/16/1。

当前对照主线使用独立检出的`ebd5f84f3`：增加`--mode native --model-source /path/to/main-ebd5f84f3`。按A1/B1/B2/A2串行执行，合并吞吐按总输出token/总耗时计算。两侧使用同一环境、缓存准备和预热协议；正式新增JIT或图捕获使该次测量无效，不能扣除时间后报分。

仅启动服务可使用`serve`动作，地址为127.0.0.1:19125；该动作本身不代表完成了593次预热或性能验收。需要外部访问时单独建立已授权的代理，节点043本次无需反向代理。

新入口仅在`DSV41_CONSOLIDATED_ENTRY=1`时启用，子进程启动与worker spawn继承同一个bootstrap。主线和候选分别设置完整profile；benchmark client清除owner/EP服务标志，防止客户端触发模型部署检查。模型flags保持默认关闭，使用本recipe完整启用。

## 代码说明

`tools/dsv41/acceptance/`只保留正式服务、预热与验证需要的代码，按一个显式安装顺序加载。`owner_head_attention.py`是 grouped attention 不支持形状时的回退；`FFN_COST_FIXED_POLICY.json`是 owner 策略输入；`owner_attention_observation.py`是具名诊断接口，均在调用链上。
