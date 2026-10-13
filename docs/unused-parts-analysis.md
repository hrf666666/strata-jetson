# 推理时"未启用部件"清单（哪些带上了没用上）

> 基于 Jetson Orin NX 实测会话日志（strata-q2_0.log）与源码核对 · 引擎 0.1.43

## A. 引擎侧：编译进来了、但本次部署没启用

| 部件 | 状态 | 原因 | 边缘/RK3588 启示 |
|---|---|---|---|
| CPU prefill share | ❌ 未武装 | 需 0.3Gi+3Gi 余量，16GB 上 RAM 太紧 | RK3588 若 16GB 且无 GUI，可开启：prefill 摊到 CPU，GPU 专心解码 |
| aux cpus（辅助线程池） | ❌ off | "每个 CPU 都是 host 或 worker，无空闲簇" | 8 核全编入了 worker 拓扑，此项天然无用 |
| KV int8 / KV streaming | ❌ 未启用 | 8K 上下文内默认 fp16（>8K 才自动 int8） | RK3588 内存更紧：`--kv int8` 是必开项 |
| 多 GPU / PCIe 分层 | ❌ 单卡 | Orin 单 SoC | RK3588 单 NPU，同构 |
| 自适应专家换出（adaptive tier） | ⚠️ 部分 | profile 预填 5060 槽、"no eviction"（无对话期换出） | 长会话下换出策略值得开 |
| 视觉编码器（mmproj） | ❌ 未部署 | --vision no，mmproj 文件未下载 | 纯文本边缘场景可永久剔除（编译期隔离见 edge-slimming.md） |
| conversation parking / prompt cache | ⚠️ 单用户场景基本闲置 | prompt cache 64/69 命中来自重复评测 prompt | 多轮场景才有效 |

## B. 服务层：装了没用（对 API 调用方而言）

| 模块 | 行数 | 本次用途 |
|---|---|---|
| frontend.py（Web 聊天界面） | 1326 | 未使用（只走 OpenAI chat API） |
| responses.py（Codex Responses API） | 690 | 未使用 |
| mcp.py（MCP 服务器） | 676 | 未使用 |
| prometheus/telemetry/replicas | 1438 | 未使用 |
| winjob/logit_bias/cache_drop | ~370 | 未使用 |

合计 ~4.5K 行 Python 在运行但空转（内存占用小，CPU 为 0——惰性注册）。

## C. 源码树里从未参与本部署的部分

sycl/（Intel 后端 7.8MB）、HIP/ROCm 路径、tools/hip 查表、bench/ 23MB、tests/、
docs/media 11MB、7 个 x86 内核翻译单元（ARM 上已条件排除）。

## D. 实际干活的（对照）

CUDA sm_87 内核（稠密+注意力+缓存专家，93% 二进制体积）、NEON+dotprod CPU 专家内核
（缓存未命中路径）、ggml-cpu NEON（i-quant 路径）、MTP 草头+PLE 直读、expert cache、
OpenAI chat API 端点。

**精简到"只保留核心功能"的形态 = 上述 D 清单**：引擎二进制（去视觉）+ ggml-cpu + tokenizer
+ chat 端点。这正是 CPU-only 构建验证的目标形态，也是 RK3588 移植的功能子集。
