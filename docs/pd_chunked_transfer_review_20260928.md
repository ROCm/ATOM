# Chunked KV Transfer 代码 Review 与修复记录

- 日期：2026-09-28
- 功能基线：`5da808aa1407d4b0e5e575c6091fc2ad4e5af8e1`（`support kv transfer by chunk`）
- Review 范围：在前一轮 `TODO(cmq)` 修复之后，重新检查 chunked KV transfer 的并发、取消、完成通知、内存生命周期和空闲推进路径。
- 本轮结果：确认并修复 4 个问题，新增 5 个参数展开后的回归用例。修复已写入工作区，本文记录时尚未提交。

本文记录本轮追加 review 的 4 个问题。前一轮的变量重命名、参数说明和 SWA-only 注册修复不重复计入本轮问题数量。

## 问题概览

| 编号 | 问题 | 影响 | 修复方法 |
| --- | --- | --- | --- |
| R1 | 取消后重复请求提前报告失败 | 原 RDMA 仍在写入时，D 可能开始回收或复用目标内存 | 在检查取消状态之前去重，由原 reader 负责终态通知 |
| R2 | P 空闲时未入场请求缺少超时推进 | D 可能一直等不到成功或失败通知 | listener 使用定时 poll，并在循环中检查 pending 请求 |
| R3 | chunk source 回收时遗留 ready event | 每个请求遗留 GPU event 的持有引用，长时间运行会累积 | 在 source-safe 回收路径删除该请求的 event |
| R4 | D 重新分配时重复登记已完成的 handoff | 等待映射持有 sequence，却不会再收到清理它的通知 | 仅在 D 实际发起 remote prefill 时登记 |

## R1：取消后重复请求提前报告失败

### 触发条件与影响

同一个 consumer 的原始 `write_request` 已经进入 RDMA 写入。此时 P 取消该请求，随后又收到具有相同 consumer identity 的重复请求。

修复前的执行顺序如下：

1. 原 reader 正在向 D 的目标 block 写入。
2. `state.cancel()` 将状态标记为取消，但不会立即终止正在执行的 RDMA。
3. 重复请求进入 `ChunkedPrefill.acquire()`，先命中取消检查并抛出异常。
4. 重复请求的异常路径立即向 D 发送 `success=False`。
5. D 若已满足其他完成条件，就可以将本次接收判为失败并进入内存回收或本地重算；原 reader 此时可能仍在写入。

PP/TP 汇总只能等待各参与方的终态，不能判断某个终态是否由仍在写入的原任务提前报告。因此必须在 producer 端保证通知的所有权。

### 根因

`acquire()` 原来先检查 `self.cancelled`，再检查 `consumer_identity in self.claims`。取消状态覆盖了重复请求检查，使重复请求获得了独立报告失败的机会。

### 修复方法

在持有 `self.cv` 的情况下，先检查 identity 是否已经被认领，再检查取消状态：

```python
if consumer_identity in self.claims:
    return False
if self.cancelled:
    raise RuntimeError("prefill was cancelled")
```

已经认领的重复请求直接返回，不启动第二次传输，也不发送自己的终态通知。原 reader 继续负责在 RDMA 返回、退出发送路径后报告完成或失败。取消后的新 identity 仍然被拒绝。

代码：[ChunkedPrefill.acquire](../atom/kv_transfer/disaggregation/chunked_prefill.py#L114)。

### 回归验证

测试：[test_duplicate_cancelled_request_waits_for_original_rdma](../tests/test_pd_chunked_transfer.py#L763)。

测试使用可阻塞的 CPU 写入代替 NIC，先将原 reader 停在写入阶段，再执行取消和重复请求。验证阻塞期间 D 没有收到失败通知、目标 reservation 仍被保留；放行原 reader 后，D 才收到一次失败通知。

修复前该测试失败：原 RDMA 尚未放行时，`d.failed_recving` 已包含请求 ID。

## R2：P 空闲时未入场请求缺少超时推进

### 触发条件与影响

D 提前发送 chunked `write_request`，但对应请求一直没有在 P 入场。P 没有其他模型工作，也没有后续网络消息。

请求虽然进入 `_pending_chunked_requests`，却可能一直等不到超时通知，导致 D 的接收等待无法结束。

### 根因

`_dispatch_ready_chunked_requests()` 包含 `PREFILL_LOOKUP_TIMEOUT` 检查，但它需要被实际调用。原 listener 使用阻塞的 `recv_multipart()`；P 没有已分配请求和 deferred source 时，也不能依赖 scheduler 持续轮询 worker。

只有超时判断而没有独立的推进触发点，不能保证超时生效。

### 修复方法

listener 每轮先处理 pending 请求，再通过 `sock.poll(timeout=1000)` 等待消息：

```python
while True:
    self._dispatch_ready_chunked_requests()
    if not sock.poll(timeout=1000):
        continue
    parts = sock.recv_multipart()
```

无消息时，poll 最多等待约 1 秒后重新检查；有消息时正常处理。这样未入场请求的超时处理不再依赖下一次 forward 或下一条消息。`PREFILL_LOOKUP_TIMEOUT` 本身没有改变。

代码：[MooncakeConnector._write_listener](../atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py#L1553)。

### 回归验证

测试：[test_idle_write_listener_expires_unadmitted_chunk_request](../tests/test_pd_chunked_transfer.py#L845)。

测试模拟只收到一条请求、随后 socket 一直空闲的情况，并推进模拟时钟。验证 listener 无需调用 scheduler 的 `get_finished()`、无需第二条消息，就能报告一次失败并清空 pending 请求。

修复前该测试失败：失败通知的调用次数为 0。

## R3：chunk source 回收时遗留 ready event

### 触发条件与影响

启用相关 index staging 路径时，最终 prefill forward 仍会通过 `record_kv_cache_ready()`，以本地 request ID 为 key，将 GPU event 存入 `_kv_cache_ready_events`。

chunked 发送使用自己的 publication event。请求完成或取消后，如果旧路径记录的 event 没有清理，字典会继续持有其引用，长时间处理请求时会累积资源占用。

### 根因

旧发送路径有对应的 event 清理。新增 chunked source 回收路径删除了 `_chunked_prefills` 和 `_chunked_local_ids`，但遗漏 `_kv_cache_ready_events`。

### 修复方法

在 `get_finished()` 确认 `state.source_safe()` 后，和其他请求状态一起删除 event：

```python
self._chunked_local_ids.pop(state.req_id, None)
self._kv_cache_ready_events.pop(state.req_id, None)
```

这里使用 `state.req_id`，与记录 event 时使用的本地请求 ID 一致。清理发生在 source-safe 阶段，正常完成和取消后的安全回收都能覆盖。

代码：[MooncakeConnector.get_finished](../atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py#L1500)。

### 回归验证

测试：[test_chunk_source_retirement_releases_legacy_ready_event](../tests/test_pd_chunked_transfer.py#L801)。

该测试按正常完成、取消两种情况展开，验证本请求的 event 条目被删除，而其他 legacy 请求的条目仍然保留。

修复前两个用例都失败：source 已回收，但字典中仍存在请求 `7` 的 event 条目。CPU 测试验证的是引用清理逻辑，没有测量真实 GPU event 资源占用。

## R4：D 重新分配时重复登记已完成的 handoff

### 触发条件与影响

D 已成功接收 KV 和 handoff，随后在 decode 期间被抢占，再次分配 block 并进行本地 prefill。

sequence 仍保留 `chunked_transfer=True`，但 `do_remote_prefill` 已经被消费并置为 `False`。修复前，重新分配会将它再次加入 `_chunked_receiving`。由于此次是本地重算，没有新的 remote handoff，映射中的 sequence 引用没有对应完成通知来清理。

### 根因

`update_state_after_alloc()` 原来仅依据 `chunked_transfer` 判断是否执行 chunked 初始化，把“请求曾使用过 chunked transfer”和“本次分配需要等待 remote handoff”当成了同一件事。

### 修复方法

把 chunked 初始化条件收窄为：

```python
if params.get("chunked_transfer") and (
    self.is_producer or params.get("do_remote_prefill")
):
    ...
```

P 仍在分配时建立 source claim；D 只有在本次确实需要 remote prefill 时才计算该次 digest 并登记待接收 handoff。接收完成后的本地重算不再创建无终态来源的等待记录。

代码：[MooncakeConnectorScheduler.update_state_after_alloc](../atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py#L431)。

### 回归验证

测试：[test_completed_chunk_receive_is_not_registered_again_on_reallocation](../tests/test_pd_chunked_transfer.py#L814)。

测试先完成一次 remote handoff，再使用保留 transfer 参数的 sequence 调用重新分配后的状态更新。验证 `_chunked_receiving` 保持为空，而且没有产生新的 connector 工作。

修复前该测试失败：重新分配后，sequence 又出现在 `_chunked_receiving` 中。

## 验证结果与复现方式

本轮新增 4 个测试函数，其中 event 清理测试展开为 2 个用例，共 5 个用例。定向验证中，修复前 5 个用例全部失败；修复后相关组合回归结果为：

```text
357 passed in 2.56s
```

实际使用的验证环境为本机已有的 `rocm/atom-dev:nightly_202609191458` 镜像。仓库以只读方式挂载，禁用容器网络，没有映射 GPU/RDMA 设备，没有拉取新的镜像或依赖。

在具备所需 Python 依赖的环境下，可从仓库根目录执行：

```bash
python3 -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_pd_chunked_transfer.py \
  tests/test_pd_pp.py \
  tests/test_pd_recv_failure.py \
  tests/test_pd_source_claim.py \
  tests/test_dcp_sharded_transfer.py \
  tests/test_deepseek_v4_transfer_regions.py \
  tests/test_kv_aggregator.py \
  tests/test_kv_drain_liveness.py \
  tests/test_pp_kv_status.py \
  tests/test_multi_connector.py
```

其他检查结果：

- Python 格式检查通过。
- `git diff --check` 通过。
- Ruff 没有新增问题；`PrefillHandoff.from_wire()` 原有的 `TRY004` 提示仍存在。本轮没有改变该处使用 `ValueError` 表示 wire 数据校验失败的行为。

## 验证边界

这些测试使用真实 connector、scheduler 或地址规划代码，并以 CPU 内存复制、可控事件、模拟 socket 和模拟时钟替代相关硬件及 I/O。它们验证本轮问题涉及的控制流、通知顺序和资源引用清理。

本轮没有执行真实跨节点 GPU/RDMA 端到端验证，也没有开展模型数值准确性或性能测试。因此，357 项 CPU 测试通过不能作为这些方面已经得到验证的结论。
