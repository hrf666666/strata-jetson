# strata-jetson

**把 Strata 引擎（x86 游戏PC 专用）魔改到 Jetson 边缘设备上，并用 125B MoE + 19 个开源小模型
完成一组边缘端推理 benchmark。**

本仓库回答三个问题：

1. Strata 这套"专家放内存/SSD、热专家驻 GPU、MTP 投机"的架构，能不能在 16GB 统一内存的
   ARM 边缘设备上跑起来？——**能，且数学推理几乎无损（GSM8K 96%）**。
2. 边缘端实际性能与能耗是什么水平？——**解码 6.3 tok/s、2.3 J/token**（详见下表）。
3. 哪些部分是死重？下一代边缘平台（RK3588）该怎么走？——见瘦身与路线文档。

## 核心成果：Strata Q2_0（125B MoE）@ Jetson Orin NX 16GB

| 指标 | 数值 | 备注 |
|---|---|---|
| 解码速度 | **6.31 tok/s**（pool-workers=3） | 默认配置 3.96（调优 +59%） |
| 读提示 | 38.9 tok/s @2K 上下文 | 专家按层批量预取 |
| TTFT | 18.0s @128 / 50.5s @2048 | |
| GSM8K-100 | **0.96** | Q2_0 量化 + NVMe 流式下 |
| ARC-100 | 0.50 | |
| 功耗 | 14.4 W 均值 / 22.2 W 峰值 | MAXN 锁频，全程 tegrastats |
| 能效 | **≈2.3 J/token** | RTX 5070 桌面 ≈2.7 J/token，同量级 |
| 专家缓存 | 5060/24576 驻留，命中率 59→85% | 命中率随页缓存变热爬升 |
| MTP 投机 | 草稿接受率 83-87% | 该 pack 格式强制 --spec≥2 |

与论文桌面参考（RTX 5070：94.6 tok/s 输出）相比，速度约为 1/15，但**能效同量级、
成本 1/10、功耗 1/17**——这是"边缘设备跑 125B"的可信形态。

## 移植内容（patches/，4 补丁 + setup 补丁，上游 MIT）

1. **Q2_0 专家内核 NEON 重写**：上游最热的 AVX-512 VNNI 路径以 ARM NEON + dotprod
   （SDOT/UDOT）重写，含 2-bit 码 vzip 解包与逐元素一致的激活量化；x86 路径原样保留。
2. 核心层 ISA shim（`_mm_pause`→`yield`、`_mm_sfence`→`dmb ishst`，5 文件）。
3. CPU 探测重写（CPUID→HWCAP ASIMDDP）与启动自检 ARM 化。
4. 构建条件化（7 个 x86 内核 TU 在 aarch64 排除）+ 链接桩 + setup.py ARM 适配。

## 仓库导航

| 文档 | 内容 |
|---|---|
| [docs/strata-feasibility.md](docs/strata-feasibility.md) | 移植可行性论证、逐文件改动、验证证据 |
| [docs/benchmark-methodology.md](docs/benchmark-methodology.md) | 评测方法学：指标定义、评测协议、已知限制 |
| [docs/edge-slimming.md](docs/edge-slimming.md) | 边缘瘦身：量化分析（二进制 93% 是 CUDA fatbin）+ 部署档案 |
| [docs/rk3588-path.md](docs/rk3588-path.md) | RK3588（NPU+CPU）移植路线与先期验证 |
| [docs/unused-parts-analysis.md](docs/unused-parts-analysis.md) | 推理时"未启用部件"清单 |
| [results/](results/) | 全部原始 JSON、汇总 CSV、图表、真实输出样例 |
| [patches/](patches/) | aarch64 移植补丁系列（可 git apply 到上游） |
| [scripts/](scripts/) | 环境复现、功耗采样、性能/质量评测、动物园批量、出图 |

## 开源小模型动物园（Q4_K_M / Q8_0，Orin NX 16GB 实测）

| 模型 | 参数(B) | GSM8K-100 | ARC-100 | decode t/s (tg128) |
|---|---|---|---|---|
| hunyuan-0.5b | — | 0.00 (100) | 0.32 (100) | — |
| hunyuan-0.5b | — | 0.30 (100) | 0.32 (100) | — |
| hunyuan-1.8b | — | 0.00 (100) | 0.60 (100) | — |
| hunyuan-1.8b | — | 0.55 (100) | 0.60 (100) | — |
| hunyuan-4b | — | 0.01 (100) | 0.77 (100) | — |
| hunyuan-4b | — | 0.80 (100) | 0.77 (100) | — |
| hunyuan-7b | — | 0.05 (100) | 0.80 (100) | — |
| hunyuan-7b | — | 0.79 (100) | 0.80 (100) | — |
| glm-z1-9b | — | 0.71 (100) | 0.86 (100) | — |
| intern-s1-mini | — | 0.22 (100) | 0.00 (100) | — |
| internlm25-1.8b | — | 0.41 (100) | 0.61 (100) | — |
| openelm-1.1b | — | 0.00 (100) | 0.14 (100) | — |
| openelm-270m | — | 0.01 (100) | 0.26 (100) | — |
| openelm-3b | — | 0.02 (100) | 0.25 (100) | — |
| openelm-450m | — | 0.00 (100) | 0.23 (100) | — |
| qwen35-0.8b | — | 0.06 (100) | 0.56 (100) | — |
| qwen35-2b | — | 0.06 (100) | 0.63 (100) | — |
| qwen35-4b | — | 0.12 (100) | 0.79 (100) | — |
| qwen35-9b | — | 0.15 (100) | 0.00 (100) | — |
| qwen3vl-2b-ins | — | 0.75 (100) | 0.71 (100) | — |
| qwen3vl-2b-think | — | 0.44 (100) | 0.55 (100) | — |
| r1-distill-1.5b | — | 0.53 (100) | 0.54 (100) | — |
| r1-distill-7b | — | 0.64 (100) | 0.13 (100) | — |
| **strata Q2_0 (125B MoE, 2-bit 专家)** | ~125 | **0.96 (25)** | 0.50 (50) | **6.3** |

## 复现

```bash
# 0. 网络：hf-mirror.com 与 modelscope.cn（国内）；GitHub 走 SSH-over-443 或 gh-proxy
# 1. 环境与构建（锁频/镜像/venv/补丁/编译 sm_87）
bash scripts/env_setup_jetson.sh
# 2. 模型获取（66GB，setup 支持 --gguf-dir 复用已下载分片、--source modelscope）
# 3. 推理服务
./run-q2_0.sh                    # OpenAI 兼容 API @ 127.0.0.1:8080
# 4. benchmark
python scripts/bench_perf.py --base-url http://127.0.0.1:8080/v1 --buckets 128,2048
python scripts/bench_quality.py --base-url http://127.0.0.1:8080/v1 --n 100
python scripts/power_sampler.py --out results/raw/power.csv &
```

## 已知限制

- Strata 官方模型（Qwen3.8-Flash-Next 125B）最小规格也需 32GB 内存，本机走 low-RAM
  文件流模式（官方支持路径的超包线延伸），速度为 NVMe 带宽与缓存命中率所限。
- 引擎上游无"无 GPU 主引擎"，RK3588 类设备需为稠密/注意力路径开发 CPU/NPU 后端
  （NEON 专家内核已验证可移植，见 rk3588 文档）。
- GLM-4.7-Flash（Q4 ≈17GB）超出 16GB 内存，未测。

## 致谢与许可

- [Niko1221/Strata](https://github.com/Niko1221/Strata)（MIT）——引擎与设计
- [ISTA-DASLab 的 GSQ-RCO 量化](https://huggingface.co/ISTA-DASLab)、Qwen3.8-Flash-Next 原始模型
- llama.cpp / ggml（MIT）——CPU 兜底路径与动物园基线
- 本仓库的移植补丁与脚本以 MIT 发布
