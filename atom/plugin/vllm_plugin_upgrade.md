---
Name: 升级vllm版本并完成vllm plugin atom适配
Description:  对vllm-atom框架进行vllm版本升级，并解决因升级导致的vllm plugin模型功能精度等兼容问题
---

# 版本升级

先按照指定版本完成vllm的版本升级安装，然后对部分plugin模型做兼容性适配，最后对整个模型列表进行完整测试，要求升级后的所有模型还能保持原有精度。


请你按照以下步骤进行

```text
实现进度：
- [x] 1) 升级vllm版本
- [x] 2) CI模型验证
- [x] 3) nightly模型精度测试
- [x] 4) benchmark模型性能测试
```

### 1) 升级vllm版本

请按照以下步骤对vllm版本进行升级：

0. 参考/workspace/ATOM下的atom_release.dockerfile构建atom image，然后from image参考docker/vllm_release.dockerfile构建vllm image vllm-upgrade;
1. 参考以下命令创建并进入docker(如果当前已在docker环境中，则无需创建docker，跳过这一步)，当前docker基于vllm 0.22.1适配了atom plugin;
```
podman run -it --cap-add=SYS_PTRACE --network=host --security-opt seccomp=unconfined --name xxx --device=/dev/kfd --device=/dev/dri -v /shared/amdgpu/home/perzhang_qle:/workspace -v /shared/data:/data --group-add keep-groups --ipc=host docker.io/rocm/atom-dev:vllm-upgrade /bin/bash
```
2. 在/app/upgrade目录下git clone vllm仓库，然后切换至v0.25.1；
3. 删除虚拟环境vllm；
4. 本地编译安装vllm环境；
5. 请你在新的文档里记录关键命令，以及升级前后哪些依赖发生版本变更。

注意事项：
- 你要在atom docker中先升级rocm版本的torch，torch请使用https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torch-2.11.0%2Brocm7.2.2.lw.git4e323059-cp312-cp312-linux_x86_64.whl版本，对比当前环境的依赖版本和安装vllm需要的依赖版本，对版本变更的依赖包需要在总结中记录。

### 2) CI模型验证

vllm升级完成后，如果更新了torch版本，请你先删除/app/aiter-test编译的cache文件，运行模型验证时jit编译，原模型在老的vllm版本可以正确运行，新vllm版本下可能会有plugin接口可能存在兼容或者精度问题，请你修复。
```
rm -rf aiter/jit/*so aiter/jit/build
```
模型列表(模型权重在挂载目录/data/amd_int/models)：
* GLM-5.1-FP8 TP4
* Kimi-K2.5-MXFP4 TP4
* Qwen3.5-397B-A17B-MXFP4 TP4
* DeepSeek-V4-Flash TP4
* gpt-oss-120b TP1
* DeepSeek-V3.2-FP8 TP4
* GLM-4.7-FP8 MTP TP4
* MiniMax-M2.5 TP4

1. 在ATOM/.github下找到模型的nighly accuracy精度测试脚本；
2. 分别运行以上模型列表，每次运行前请先清除显存再启动server，模型运行失败时请直接修复；
3. 模型精度不足时，请先尝试解决精度问题，如未能解决，可以重建docker，不升级在vllm在0.22.1版本中运行精度测试做对比再定位解决；
3. 统计每个模型的运行结果。

### 3) nightly模型精度测试

CI模型测试完成后，请你接下来完成nightly全部MI355模型测试，你要先完成全部vllm backend的模型测试，然后对有错误的模型尝试进行修复，按照以下步骤执行

1. 在ATOM/.github下找到所有模型的nighly accuracy精度测试脚本；
2. 分别运行以上模型列表，每次运行前请先清除显存再启动server，你可以利用所有的空闲卡同时跑多个任务；
3. 模型精度不足时，请先尝试解决精度问题，如未能解决，可以重建docker，不升级在vllm在0.22.1版本中运行精度测试做对比再定位解决；
3. 统计每个模型的运行结果。

### 3) benchmark模型性能测试

完成精度验证后，请你接下来按步骤完成MI355的benchmark测试，并将测试结果保存下来。

1. 在ATOM/.github下找到vllm nightly模型的benchmark测试脚本；
2. 分别运行以上模型列表，每次运行前请先清除显存再启动server；
3. 统计每个模型的运行结果。


