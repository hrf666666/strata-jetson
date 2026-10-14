# strata-jetson

> Port of the [Strata](https://github.com/Niko1221/Strata) LLM inference engine (x86/CUDA, targeting 125B-class MoE on gaming PCs) to NVIDIA Jetson (aarch64 / CUDA sm_87), plus an edge-side inference benchmark across 20 open-weight models.
>
> 简体中文文档见下。

**结果一句话**：Strata 引擎经 NEON 移植后在 Jetson Orin NX 16GB 上以 Q2_0 量化运行 125B MoE（Qwen3.8-Flash-Next），
解码 6.31 tok/s、功耗 14.4 W、能效 ≈2.3 J/token，GSM8K 准确率 0.96——数学推理在 2-bit 专家量化 + NVMe 流式下基本无损。

## 1. 背景

Strata 是一个面向消费级 x86 PC 的 MoE 推理引擎：稠密部分驻留 GPU、全部专家驻留系统内存或 SSD、
高频专家缓存进显存、以模型自带的 MTP 层做投机解码。上游仅支持 x86-64（AVX2 起步）+ CUDA。

本仓库的工作：

1. **引擎移植**：将 Strata 完整移植到 aarch64（Jetson Orin NX, sm_87），核心是 Q2_0 CPU 专家内核的
   NEON + dotprod 重写（`patches/`，4 个补丁，可独立 `git apply` 到上游）。
2. **边缘基准**：对该部署形态做系统测量——性能（TTFT/prefill/decode）、质量（GSM8K/ARC）、
   功耗与能耗（tegrastats 全程采样）。
3. **横向对照**：20 个开源小模型（0.27B–9B，Q4_K_M）在同一台设备上的性能与质量基准，
   覆盖 Qwen3.5、Hunyuan、GLM-Z1、DeepSeek-R1-Distill、OpenELM、InternLM2.5、Intern-S1、Qwen3-VL。
4. **工程分析**：边缘部署瘦身量化分析、推理时未启用部件清单、RK3588（NPU+CPU）移植路线。

## 2. 环境

| 项 | 值 |
|---|---|
| 设备 | Jetson Orin NX 16GB（Super 模式开发套件） |
| 系统 | JetPack 6.2 / L4T 36.4.3 / CUDA 12.6 |
| 功耗与频率 | MAXN_SUPER + `jetson_clocks --fan`，全程锁定 |
| 引擎 | Strata 0.1.43（本仓库移植版）；llama.cpp `3cf03257`（本地 CUDA 构建，用于横向基线） |
| 网络 | HuggingFace 经 hf-mirror.com；ModelScope 直连；GitHub 走 SSH-over-443 |

## 3. 核心结果

### 3.1 Strata Q2_0（125B MoE）@ Orin NX 16GB

| 指标 | 数值 | 条件 |
|---|---|---|
| 解码速度 | **6.31 tok/s** | `--pool-workers 3`；默认配置 3.96 tok/s |
| 读提示 | 38.9 tok/s @2K；7.2 tok/s @128 | |
| TTFT | 18.0 s @128；50.5 s @2048 | |
| GSM8K（25 题） | **0.96** | 贪心解码，3-shot CoT |
| ARC-C（50 题） | 0.50 | |
| 功耗 | 14.4 W 均值 / 22.2 W 峰值 | MAXN，全程 tegrastats |
| 能效 | ≈2.3 J/token | 论文桌面参考（RTX 5070）≈2.7 J/token |
| 专家缓存 | 5060 / 24576 驻留（5.19 GiB） | 命中率 59%→85%（页缓存变热） |
| MTP 投机 | 草稿接受率 83–87% | 该 pack 格式强制 `--spec ≥ 2` |

内存形态：非专家权重 + KV + 专家缓存约 11 GiB 驻留统一内存；专家本体经 `--mmap-experts`
从 NVMe 原位流式读取（PLE 表 26.8 GB 直读、不入内存）。

### 3.2 与桌面平台对照（Q2_0，4K 上下文）

| 平台 | 输出 tok/s | 读提示 tok/s | 功耗 | 能效 |
|---|---|---|---|---|
| RTX 5070 12 GB（上游论文） | 94.6 | 539 | ~250 W | ≈2.7 J/token |
| **Orin NX 16 GB（本仓库）** | 6.31 | 38.9 | 14.4 W | ≈2.3 J/token |

### 3.3 开源模型基准（同机横向，Q4_K_M，llama.cpp CUDA）

| 模型 | 参数(B) | GSM8K-100 | ARC-100 | decode tok/s | prefill tok/s @2K |
|---|---|---|---|---|---|
| glm-z1-9b | 9.4 | 0.71 | 0.86 | 11.9 | 397 |
| hunyuan-7b | 7.5 | 0.79 | 0.80 | 13.7 | 444 |
| qwen35-9b | 9.2 | 0.15* | 0.00* | 12.1 | 358 |
| intern-s1-mini | 8.2 | 0.22 | 0.00* | 11.0 | 507 |
| qwen3vl-2b-ins | 1.7 | 0.75 | 0.71 | 43.5 | 1719 |
| qwen35-4b | 4.3 | 0.12* | 0.79 | 18.1 | 550 |
| hunyuan-4b | 4.2 | 0.80 | 0.77 | 21.1 | 709 |
| r1-distill-7b | 7.5 | 0.64 | 0.13 | 13.6 | 444 |
| internlm2.5-1.8b | 1.9 | 0.41 | 0.61 | 46.5 | 1785 |
| hunyuan-1.8b | 1.8 | 0.55 | 0.60 | 42.8 | 1592 |
| qwen35-2b | 1.9 | 0.06* | 0.63 | 37.1 | 1283 |
| qwen3vl-2b-think | 1.7 | 0.44 | 0.55 | — | — |
| r1-distill-1.5b | 1.5 | 0.53 | 0.54 | — | — |
| hunyuan-0.5b | 0.5 | 0.30 | 0.32 | 89.4 | 3962 |
| qwen35-0.8b | 0.8 | 0.06* | 0.56 | 60.5 | 1996 |
| openelm-3b | 3.0 | 0.02 | 0.25 | 28.3 | 963 |
| openelm-1.1b | 1.1 | 0.00 | 0.14 | 68.7 | 2248 |
| openelm-450m | 0.5 | 0.00 | 0.23 | 128.5 | 4811 |
| openelm-270m | 0.3 | 0.01 | 0.26 | 190.8 | 7601 |
| **strata Q2_0 (125B MoE)** | ~125 | **0.96** | 0.50 | **6.3** | 38.9 |

\* 见 [方法学·已知限制](docs/benchmark-methodology.md)：Qwen3.5 家族为格式遵循下限（ARC 正常爬坡证明能力无碍）；
思考型模型 ARC=0 标记思考链耗尽 512 token 预算。全部为 100 题/任务（Strata 为 25/50 题抽样）。

### 3.4 LLM-as-Jev：概率直读决策（零训练，复现 Microsoft arXiv 2610.02076）

不走"生成→解析"，把选项编号 `[1]..[K]` 后 prefill 到 `Best answer: [`，直接从
next-token 概率读出**校准的选项分布**（每题 1 次调用，top-k 全词表 softmax）。
ARC-Challenge 100 题（同题同序对照 3.3 生成式分数）：

| judge | 参数量 | JEV acc | 生成式 | Δ | ECE |
|---|---|---|---|---|---|
| Qwen3.5-4B | 4B | **0.93** | 0.79 | +0.14 | 0.048 |
| Qwen3.5-2B | 2B | **0.77** | 0.63 | +0.14 | 0.085 |
| Qwen3.5-0.8B | 0.8B | **0.61** | 0.56 | +0.05 | 0.129 |
| Qwen3-0.6B（论文基座） | 0.6B | 0.62 | — | — | 0.294 |
| OpenELM-3B / 1.1B / 270M（base） | 0.3-3B | ≈随机 | ≈随机 | ≈0 | 0.66-0.71 |

结论：≥2B 对话模型"已经是 Jev"——评分全面胜过生成解析且天然校准（4B ECE 0.048），
生成答对时 97.5% 一致、另净救回 15 题；**<1B/base 模型不能直接当 judge**（论文微调
动机在边缘侧的实证）。100 题评分 3-29 s vs 生成式 10-17 min（~35×）。

同一框架的 **noul 二元验证**（判"这个答案对不对"，`scripts/jev_verify.py`）：
judge 只看题面+候选答案，目标模型权重无需在盘——一个 4B judge 给全动物园的
GSM8K 答案当质检员（AUC 0.66-0.79，置信 top-50% 准确率 +8~12 分），给 125B
Strata 引擎的 25 题答案排序 **AUC 1.0**（n 小仅示意），对自己答案 AUC 0.88
（模型知道什么时候错 → 选择性预测/升级触发）。base 模型反向（OpenELM-3B
AUC 0.435）。argmax 在类别不平衡下无意义，分数/排序才是产品。详见
[docs/jev-on-jetson.md](docs/jev-on-jetson.md)。

## 4. 引擎移植内容

| 补丁 | 内容 |
|---|---|
| `0001` | Q2_0 CPU 专家内核 NEON+dotprod 重写（2-bit 码 vzip 解包、`vdotq_s32` 点积、逐元素一致的激活量化）；x86 路径原样保留于 `STRATA_X86` 守卫 |
| `0002` | 核心层 ISA shim（`_mm_pause`→`yield`、`_mm_sfence`→`dmb ishst`）；x86-only 内核翻译单元的条件排除与链接桩 |
| `0003` | CPU 特性探测重写（CPUID→`HWCAP_ASIMDDP`）；`q2_rows_any`/`act_quant_any` 分发到 NEON 路径 |
| `0004` | 启动自检 ARM 化（FEAT_DotProduct 取代 AVX2 拒绝） |
| setup 补丁 | `setup.py` 适配 ARM `/proc/cpuinfo`；`--cuda 12` 低驱动路径 |

## 5. 目录结构

```
├── patches/                  # aarch64 移植补丁系列（git apply 到上游 Strata）
├── scripts/
│   ├── env_setup_jetson.sh   # 环境/锁频/镜像/编译 一键复现
│   ├── bench_perf.py         # 性能基准（TTFT/prefill/decode，流式）
│   ├── bench_quality.py      # GSM8K + ARC-C 评测（OpenAI chat 端点）
│   ├── power_sampler.py      # tegrastats 采样器（功耗/内存/GPU 占空）
│   ├── analyze_power.py      # 功耗/能耗分析
│   ├── run_zoo.py            # 模型动物园批量 runner（取权重→bench→eval→清理）
│   ├── tuning_matrix.sh      # llama.cpp 调参矩阵（线程×FA×KV 量化）
│   ├── collect_results.py    # 结果汇总 CSV
│   ├── jev_score.py          # LLM-as-Jev 概率直读决策评分（arXiv 2610.02076）
│   ├── jev_verify.py         # JEV noul：二元答案验证（AUC/选择性准确率）
│   ├── collect_jev.py        # JEV 结果汇总
│   ├── make_plots.py         # 汇总图表
│   ├── update_readme_tables.py
│   ├── edge_deploy.sh        # 最小边缘部署组装
│   └── nvidia-smi-jetson-shim
├── docs/
│   ├── strata-feasibility.md       # 移植可行性论证与验证证据
│   ├── benchmark-methodology.md    # 评测方法学与已知限制
│   ├── edge-slimming.md            # 边缘瘦身量化分析
│   ├── rk3588-path.md              # RK3588（NPU+CPU）移植路线
│   ├── unused-parts-analysis.md    # 推理时未启用部件清单
│   └── jev-on-jetson.md            # LLM-as-Jev 概率直读决策（复现+实证）
├── results/
│   ├── summary_perf.csv / summary_quality.csv
│   ├── plots/                      # 汇总图表
│   ├── samples/                    # 引擎真实输出样例
│   └── raw/                        # 原始 JSON（不入库，见 .gitignore）
└── LICENSE
```

## 6. 复现

```bash
# 1) 环境：锁频、镜像源、venv、应用补丁、编译（CUDA sm_87）
bash scripts/env_setup_jetson.sh

# 2) 获取模型（66 GB；setup 支持 --gguf-dir 复用已下载分片、--source modelscope）
cd work/Strata
python3 setup.py --family qwen --model Q2_0 \
  --gguf-dir <已下载分片目录> --data-dir <数据目录> \
  --context 8192 --vision no --low-ram on --cuda 12 --build \
  --source modelscope --yes --no-start

# 3) 启动推理服务（OpenAI 兼容 API）
./run-q2_0.sh                       # http://127.0.0.1:8080/v1

# 4) 基准
python scripts/power_sampler.py --out results/raw/power.csv &
python scripts/bench_perf.py   --base-url http://127.0.0.1:8080/v1 --buckets 128,2048
python scripts/bench_quality.py --base-url http://127.0.0.1:8080/v1 --n 100

# 5) LLM-as-Jev 概率直读决策（任一 GGUF，见 docs/jev-on-jetson.md）
python scripts/jev_score.py --model work/zoo/gguf/qwen35-4b.gguf \
  --tag qwen35-4b --compare-gen && python scripts/collect_jev.py
```

## 7. 方法学与已知限制

详见 [docs/benchmark-methodology.md](docs/benchmark-methodology.md)。要点：

- GSM8K/ARC 各 100 题（seed=42 固定抽样，贪心解码）；Strata 为 25/50 题抽样。
- Qwen3.5 家族的 GSM8K 分数是**格式遵循下限**（其聊天风格不复刻 few-shot 格式；
  ARC 正常爬坡证明能力无碍）。思考型模型 ARC=0 标记 512 token 思考预算耗尽。
- Hunyuan 首轮数据因解析缺陷作废，已用修复版解析器全量重测。
- GLM-4.7-Flash（Q4 ≈17 GB）超出 16 GB 内存，未测。

## 8. 致谢与许可

- [Niko1221/Strata](https://github.com/Niko1221/Strata)（MIT）——引擎与架构设计
- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 及
  [ISTA-DASLab GSQ-RCO 量化](https://huggingface.co/ISTA-DASLab)
- [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp)（MIT）——横向基线与 CPU 兜底路径

本仓库的移植补丁、脚本与文档以 MIT 发布，见 [LICENSE](LICENSE)。
