# RK3588 移植路线（先期验证规划）

> 目标：把 strata-jetson 的成果迁移到 RK3588（4×Cortex-A76 + 4×A55，6 TOPS NPU，
> LPDDR4x 8-16GB，内存带宽 ~30GB/s）。本文是可行性分析与验证清单。

## 1. 三层算力怎么分

| Strata 部件 | RK3588 落点 | 依据 |
|---|---|---|
| Q2_0 CPU 专家内核（本移植的 NEON 重写） | **CPU（A76 集群）** | 已是 ARMv8.2-a+dotprod 目标代码，A76 原生支持 sdot/udot；`-march=armv8.2-a+dotprod` 直接可用 |
| ggml-cpu 路径（i-quant 等） | CPU（A76；A55 只做辅助） | ggml 对 A76 有现成 NEON 内核 |
| 稠密部分/注意力（上游走 CUDA） | **NPU（rknn-llm）为主 + CPU 兜底** | rknn-llm 支持 Qwen/LLaMA 类 transformer 算子（w8a8/w4a16）；但 Strata 的 DeltaNet 递归、QSA 索引器、hyper-connection 为定制算子，rknn-llm 未必覆盖 |
| PLE n-gram 表（28.8GB） | NVMe 直读（同 Jetson） | `--ple-io direct` 路径与设备无关 |
| 专家流式（GGUF-in-place） | NVMe/eMMC 直读 | 与设备无关；RK3588 的 NVMe 或 USB3 SSD 均可 |

**关键约束：内存带宽**。RK3588 ~30GB/s vs Orin NX 102GB/s。Strata 的 decode 主要吃
"稠密权重读取 + 未命中专家行"——带宽直接决定 decode 上限，预估 RK3588 上
decode 峰值在 1-3 t/s 量级（Q2_0、专家缓存命中 50-70% 时）。这是"能跑"而非"能用"的量级，
但作为 Strata 架构可移植性验证（而非产品化）足够。

## 2. 已在本机完成的 RK3588 先期验证

| 验证项 | 状态 | 证据 |
|---|---|---|
| NEON+dotprod 专家内核编译 | ✅ | `-march=armv8.2-a+dotprod` 构建，A78AE 上运行正确（GSM8K 96%） |
| A76 ISA 兼容 | ✅（静态） | A76 支持 ARMv8.2 + dotprod（SDOT/UDOT），与内核要求一致 |
| CPU-only 引擎构建 | ✅ | `build-cpu/strata`（无 CUDA 依赖，见下） |
| 无 CUDA 的运行路径 | ⚠️ 待验 | CPU-only 加载模型（等模型文件回场后做 token 级 smoke） |
| NPU 映射 | 📋 分析完成 | 见第 3 节 |

## 3. NPU（rknn-llm）映射分析

可上 NPU 的（若 rknn-llm 支持对应算子）：
- QSA 标准注意力部分（KV 仅 2 头、滑窗 ≤2048）——类 MHA，映射友好
- 共享专家、router 的 GEMM
- 输出头 GEMV

必须留 CPU 的：
- Gated DeltaNet 递归（自定义状态更新，rknn-llm 无此算子）
- PLE n-gram 查表（I/O 型，本就该 CPU/直读）
- MoE 路由 + 未缓存专家行（我们的 NEON 内核的地盘）
- 512 选 10 路由的 top-k（小算子，CPU 即可）

结论：RK3588 上现实的形态是 **NPU 跑稠密 GEMM + CPU（NEON）跑专家与定制算子**的
混合，与 Strata"GPU 稠密 + CPU 专家"的设计哲学同构——只是把"CUDA 稠密"换成"NPU 稠密"。

## 4. 下一步验证清单（按序）

1. ✅ CPU-only 构建（本机，无 CUDA）
2. ⏳ CPU-only 加载 + 少量 token 生成 smoke（等模型回场；预期 ~0.1-0.3 t/s，只验证正确性）
3. ⏳ 交叉编译 x-build：`-march=armv8.2-a+dotprod` 目标 A76 在 x86 宿主交叉编译（或直接在
   RK3588 板上原生编译——ggml 与本引擎均无宿主依赖）
4. ⏳ RK3588 实板：先跑 llama.cpp 基线（Qwen3-0.6B Q8_0 ~2-5 t/s 量级），再移植 NEON 专家内核
5. ⏳ rknn-llm 评估：确认其算子集覆盖 QSA/GDN 的可行性（读其 op 支持表）

## 5. 风险

- 内存带宽是硬上限：30GB/s 下 125B Q2_0 的 decode 体验有限；RK3588 验证建议改用
  Coder-IQ1_M（27.6GB）或更小的量化车型
- rknn-llm 的算子覆盖需实机确认；不覆盖则 NPU 只能用于标准子图，收益打折
- RK3588 官方 BSP 的 GPU（Mali）对推理无用， dontcare
