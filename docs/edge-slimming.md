# 边缘瘦身（Edge Slimming）：量化分析与部署档案

> strata-jetson · 实测于 Jetson Orin NX 16GB · 引擎 0.1.43 (aarch64 移植版)

## 结论先行

对"在边缘设备上部署 Strata"这个目标，**瘦身的收益在部署卫生（磁盘/攻击面/维护面），
不在推理性能**。原因有二，均有实测数据：

1. **引擎二进制 93% 是 CUDA 内核**：`engine-cuda12/strata` 61.2MB 中，`.nv_fatbin` 占 56.1MB。
   我们只编译了 sm_87 一个架构（桌面 x86 版要背 75/89/120 多架构 fatbin）——单架构已是
   数学上的最小形态，`strip` 仅省 1.1MB。
2. **运行时内存与速度由模型和缓存策略决定**：加载后 RAM 14.8GB（mmap 专家 + 5.19GiB 热专家
   缓存 + KV），引擎代码段 <100MB。真正移动数字的是内存配比与拓扑调优
   （实测 `--pool-workers 3` 在 Orin 2×4 核簇上 +59% 解码速度），不是删代码。

## 实测数据

### 二进制构成（engine-cuda12/strata）

| 段 | 大小 | 说明 |
|---|---|---|
| .nv_fatbin | 56.1 MB | sm_87 GPU 内核（不可再减：单架构） |
| .text | 5.2 MB | CPU 代码（含 NEON 专家内核） |
| .rodata | 1.0 MB | 常量（i-quant 码本等） |
| 其余 | ~1.5 MB | eh_frame/bss 等 |
| strip 收益 | -1.1 MB | 符号表 |

### 部署清单（setup 实际安装的）

| 项 | 大小 | 备注 |
|---|---|---|
| engine-cuda12/strata | 61.2 MB | 唯一必需二进制（setup 未安装任何 companion 工具 ✔） |
| pack（专家重排） | 34 GB | Q2_0 布局，随模型走 |
| GGUF 双分片 | 61.8 GB | shard1 模型 + shard2 PLE 表（`--ple-io direct` 永不入内存） |
| MTP 草头 | ~0.8 GB | 投机解码（pack 格式强制 --spec≥2，不可关） |
| tokenizer + 配置 | ~10 MB | |

### Python 服务层（serve/，10713 行）

| 模块 | 行数 | 边缘判定 |
|---|---|---|
| server.py | 6459 | 核心（OpenAI 兼容 API） |
| frontend.py | 1326 | Web 聊天界面——边缘可去，但被 server.py 顶层硬 import（9 个符号） |
| responses.py | 690 | OpenAI Responses API（Codex）——**31 处调用点深度耦合**，硬拆风险大于收益 |
| mcp.py | 676 | MCP 服务器——11 处调用点 |
| prometheus.py | 160 | 已经是惰性 import（函数内）✔ 上游本来就做成可选 |
| replicas/telemetry 等 | 859 | 多副本/遥测——边缘可去 |

**裁剪收益实测口径**：全删也只省 ~100KB 源码 + 个位数 MB 运行时 RAM；
`responses.py` 的 31 处耦合决定了"裁剪版 server"需要一次正经重构而非打补丁。
**边缘推荐：不裁 server，用 `--host 127.0.0.1` + 无 api-key 之外的默认配置即可**；
若强需求，按 frontend/mcp/prometheus（已可选）→ responses 的顺序做重构。

### 源码树（开发仓库 → 部署物）

| | 体积 |
|---|---|
| 完整仓库（不含 .git） | 55 MB |
| 部署必需（src+include+cmake+CMakeLists+third_party+最小 serve） | ~3 MB |
| 永不进部署物 | docs/media 11MB、bench 23MB、sycl 7.8MB、tools 22MB、tests（构建期已 `STRATA_BUILD_TESTS=OFF` 排除） |

## 一键边缘部署

`scripts/edge_deploy.sh` 产出最小部署目录：引擎二进制 + 必需 serve 模块 + 配置/tokenizer，
跳过文档/媒体/工具/测试/SYCL/HIP。

## 已做的瘦身（构建期，全部已生效）

1. `STRATA_BUILD_TESTS=OFF`：150 个测试/parity 目标不编译（其中 4 个 x86-only 工具在 ARM 上
   由本移植的条件编译排除）。
2. 单架构 `CMAKE_CUDA_ARCHITECTURES=87`。
3. 7 个 x86 内核翻译单元（iq_avx1/2/512、kq_avx1/2、q2_avx1、q2_avx2）在 ARM 构建中条件排除。
4. SYCL/HIP 后端从未启用（CUDA-only 配置）。
