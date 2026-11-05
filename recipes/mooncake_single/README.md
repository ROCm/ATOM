# V4.1 单机 C256：#2755 + LMCache MP / Mooncake Store

这是实际实验代码分支，基于#2755提交`ab5368a71c9ea22e2454d090f8752288d8245999`，保留本会话的V4.1/MP集成改动及诊断指标。不是纯main，不携带二进制、SDK、模型或数据。

两组都用相同的session affinity和迁移策略。`local`只有256GiB MP L1；`store`在同一个L1后增加512GiB Mooncake Store。单机共享缓存服务允许rank之间恢复KV，但各GPU的显存KV仍独立。

## 运行环境

在已具备ATOM/LMCache运行环境的Linux容器内执行。验证环境为MI355X/gfx950、4GPU、ROCm7.2.4、torch2.10、Python3.12，模型**DeepSeek-V4.1-Flash-1017**。CPU内存需容纳缓存和模型运行时；Store是DRAM，不是磁盘。

本次使用的版本：

- 服务镜像：`rocm/atom-dev@sha256:b0029ebfa2f7fa67f7b65791c999e610b624676ee723e528ef7c07e40ca5b374`
- LMCache：`1d6ee40c812001c670a62e8e9a62d6422665ea70`，最新上游MP接口，无yhl覆盖。
- Mooncake：`4704c67f13ffbada9fbe7785d854c0a53e1e73b4`。
- AgentX客户端镜像：`rocm/atom-dev@sha256:50b431e6a9eb364961c5b07808a700ce2f17f6ea46ae3dbdd72ff0e28908155b`。

服务镜像自带的LMCache不等同上述版本，需使用环境里已安装好的对应LMCache。此分支不会替你安装/编译依赖；Store组还要求可执行的`mooncake_master`及其动态库。其他GPU架构需使用适配目标环境的依赖，不复用gfx950二进制。

## 启动

在仓库根目录执行，将MODEL替换为实际路径。确保所选四卡空闲；脚本不申请机器，不清理其他进程。

```bash
export MODEL=/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# 若LMCache是源码形式使用，再将已安装好扩展的LMCache目录加到PYTHONPATH。
python -c 'import lmcache.integration.atom.multi_process_adapter; import lmcache.cuda_ops'

# A：本地CPU缓存
python recipes/mooncake_single/run_single.py \
  --arm local --model "$MODEL" --gpus 0,1,2,3 \
  --host-ip 127.0.0.1 --output /tmp/atom-single-local-01
```

前台运行；另开终端压测。完成后Ctrl-C停止本脚本的子进程，等GPU资源释放，再启动B：

```bash
# B：同配置增加Store；不在PATH时传--master-bin /path/to/mooncake_master
python recipes/mooncake_single/run_single.py \
  --arm store --model "$MODEL" --gpus 0,1,2,3 \
  --host-ip 127.0.0.1 --output /tmp/atom-single-store-01
```

Store组自动启动独立master、MP及ATOM。默认API22300、DP22400/22420起、MP22600、MP HTTP22601、master22561/metadata22562/metrics22563。不同实验可通过`--port`整体平移端口；每次使用新的`--output`避免混入旧日志。跨容器/跨主机访问时设置可达的`--host-ip`，并按环境配置`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`。

参数固定为TP4+DPA、EP/MoRI关闭、DSpark5、BF16 KV/FP8 index、GPU util0.65、64seq/rank、16k batched tokens/16k prefill chunk。可用`--gpu-util`、`--l1-gib`、`--store-gib`调整容量，但A/B应保持其他参数一致。

## C256 AgentX

先确认实际生成可用，不只看health：

```bash
curl http://127.0.0.1:22300/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"Reply READY:\",\"max_tokens\":8,\"temperature\":0}"
```

在具有上述AgentX版本的客户端环境执行；客户端需能访问服务、模型tokenizer和公开数据源。MODEL须与服务模型名一致，URL按环境修改。

```bash
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://127.0.0.1:22300 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model "$MODEL" --tokenizer "$MODEL" --tokenizer-trust-remote-code \
  --concurrency 256 --benchmark-duration 900 --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 --use-server-token-count \
  --no-gpu-telemetry --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir /tmp/agentx-local-01 --no-server-metrics
```

B组使用相同命令，只更换结果目录为`/tmp/agentx-store-01`。900s指正式发压，不包含warmup/尾部排空。不要同时跑A和B；原始失败记录也要保留。

## 策略与结果解读

两组均设置：session affinity=1、prefix spread=1、policy=idle_ratio、idle_tokens=32768、owner_backlog_tokens=65536、ratio=0、similar_tokens=16384、early_load=1。这是当前32k实验配置，不是早先64k/2倍的策略。Store使用TCP、4workers、1GiBbuffer、512GiBsegment；MP chunk256、LRU、TTL900s、watermark0.98。

本脚本通过`ATOM_KV_OFFLOAD=lmcache_mp`组合offload，**不要再额外指定一个offload `--kv-transfer-config`**。

比较`profile_export_aiperf.json`的TTFT avg/P50/P90/P99、输出吞吐、缓存命中率、错误及取消数。每卡吞吐除以4。总token吞吐包括缓存命中的输入，不等于实际prefill计算量。

服务日志在`--output`内；MP指标`http://127.0.0.1:22601/metrics`、ATOM指标`http://127.0.0.1:22300/metrics`。建议每10秒采样，区分L1/L2命中、真实running/waiting与路由estimated token debt。

此对照给Store组增加了512GiB容量，衡量的是整体缓存方案收益，不能单独证明算法更优。实验曾在一台节点两次遇到warmup HTTP connection reset，之后换节点；尚无已确认的单机Store性能提升。脚本没有PITT专用watchdog/自动抓栈，内部worker异常时health也可能仍200；若出现无进展，保留日志定位，不要仅凭容器存活判断正常。
