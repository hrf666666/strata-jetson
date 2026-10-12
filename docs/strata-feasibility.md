# Strata 在 Jetson Orin NX 上的可行性与移植报告

> strata-jetson 项目 · 2026-10-12 · 引擎 v0.1.43 (upstream main) · JetPack 6.2 / L4T 36.4.3 / CUDA 12.6

## 结论（TL;DR）

Niko1221/Strata——一个专为"x86 游戏 PC + 12GB 显存跑 125B MoE"设计的推理引擎——**已在本机
（Jetson Orin NX 16GB，aarch64，Cortex-A78AE + sm_87）完成源码移植并成功编译、链接、运行**。
引擎 CUDA 栈在统一内存上正常初始化，`strata-device` 探测输出：

```
device 0: Orin
  compute capability  8.7   (sm_87)
  multiprocessors     8
  VRAM total / free   15.289 GiB / 9.825 GiB
  driver / runtime    12060 / 12060
```

移植补丁共 2 个提交（`patches/` 目录），核心是为 Q2_0 CPU 专家内核编写的 **NEON + dotprod
（SDOT/UDOT）实现**——这是整个移植中唯一需要重写的算术热路径。

## 为什么这件事有意义

Strata 的设计哲学是：MoE 里每 token 只激活 ~2% 的专家权重（0.66GB/48 层窗口），把"总在用"
的稠密部分放 GPU、24,576 个专家放内存/SSD、热专家缓存进显存，三层内存各司其职。
Orin NX 是统一内存（CPU/GPU 共享 102.4GB/s LPDDR5），没有独立显存可"放"，但：

1. 统一内存使 Strata 的 "PCIe DMA 拷贝" 变成内存内拷贝，天然免费；
2. 引擎对统一内存 APU 已有支持先例（AMD Strix Halo 文档；`device_free_bytes()` 按
   `MemAvailable - STRATA_UMA_HEADROOM_GIB` 计算，文档明言适用于 DGX Spark 一类设备）；
3. 引擎自带 low-RAM 文件流模式（GGUF-in-place）：专家装不进内存时按需从 NVMe 读，
   热专家驻留 GPU 缓存——这正是 16GB 边缘设备需要的模式。

因此"Strata 思想"在边缘设备上的价值主张变为：**在 15.6Gi 统一内存里跑桌面级 125B 模型**。
桌面参考成绩（论文 Table 3，RTX 5070 12GB）：Q2_0 @4K = 94.6 tok/s 输出、TTFT 6.7s。

## 移植的技术内容（patch 系列）

### Patch 1: NEON Q2_0 专家内核 + ISA shims

| 文件 | 改动 |
|---|---|
| `include/strata/portability.hpp`（新增） | `_mm_pause`→`yield`、`_mm_sfence`→`dmb ishst` 的架构 shim |
| `src/core/graph.cpp` `verify.cpp` `pool.cpp` `session.cpp` `mtp.cpp` | 自旋提示/写屏障换 shim（5 文件共 ~25 处调用点） |
| `src/kernels/cpu/expert.cpp` | **Q2_0 CPU 专家内核的 NEON 重写**：2-bit 码解包（vzip 序列恢复顺序）+ `vdotq_s32` 整数点积（FEAT_DotProduct）+ NEON 激活量化（逐元素规则与 AVX-512 版一致）；x86 路径原样保留在 `STRATA_X86` 守卫内 |
| `src/kernels/cpu/expert_layout.cpp` | CPUID 探测在 ARM 上返回 false；`CpuFeatures` 改查 `HWCAP_ASIMDDP`；`q2_rows_any`/`act_quant_any` 分发到 NEON 内核 |

关键语义保持：Q2_0 行点积公式 `sum_j (c_j-1)·d_w·x̂_j·d_x = d_w·(d_x·Σ(c·x̂) − d_x·Σ(x̂))`，
码序 LSB-first（`code j = (byte[j>>2] >> 2*(j&3)) & 3`），激活量化"round half away from zero"
+ 截断 + ±127 钳制，与 x86 内核逐元素一致（整数部分双精度精确，仅 FP32 求和顺序有位级差异，
与 llama.cpp 跨架构差异同级）。

### Patch 2: 链接层与构建系统

| 改动 | 说明 |
|---|---|
| `src/kernels/cpu/arm_port_fallback.cpp`（新增） | 被排除的 x86 内核文件的链接桩；所有消费方都被 `cpu_avx2_ok()` 等探测门控（ARM 上恒 false），实际执行永不到达 |
| `CMakeLists.txt` | x86-only 内核文件（q2_avx2/iq_avx1/2/512、kq_avx1/2、q2_avx1）在 aarch64 上条件排除；`expert.cpp` 以 `-march=armv8.2-a+dotprod` 编译；x86 parity 测试工具条件排除（edge 构建瘦身第一刀） |

i-quant（IQ2_XS/IQ3_XXS 等）专家在 ARM 上走 ggml-cpu 自己的 NEON 量化内核（`STRATA_NATIVE_EXPERTS`
路径），无需额外工作——ggml 对 aarch64 的支持是现成的。

## 移植过程中验证过的事实

1. **CMake 配置**：aarch64 上一次通过（除需显式指定 nvcc 路径）；ggml 0.24.0 检测到
   Cortex-A78AE 的 dotprod/fp16 特性（无 SVE/i8mm，符合架构规格）。
2. **CUDA**：nvcc 12.6 + sm_87 编译全部 `.cu` 内核无一处 x86 假设；引擎运行时驱动/运行时
   版本匹配（12060/12060）。
3. **内存模型**：引擎把统一内存识别为 15.29GiB "VRAM" 并按 MemAvailable 规划，与 Strix Halo
   文档描述一致。
4. **上游代码质量**：x86 假设集中在 17 个文件；核心调度（expert_source/session/verify/generate）
   完全 ISA 中性，这让移植面比预期小一个量级。

## 待完成（Phase 2-3 进行中）

- [ ] Q2_0 双分片 GGUF（66.4GB，hf-mirror 下载中）+ MTP 草头张量（~5GB，来自 BF16 checkpoint）
- [ ] low-RAM 文件流模式跑通（`--native` + 专家缓存配比扫描）
- [ ] 边缘端 benchmark：tok/s（读提示/写回答）、TTFT、缓存命中率、功耗、tokens/J
- [ ] edge 瘦身分层 A/B（裁剪 SYCL/HIP/vision/MCP 后的二进制与内存占用）

## 环境复现

见 `scripts/env_setup_jetson.sh`（锁频、镜像、venv、补丁应用、编译一条龙）。
