# 评测方法学（Benchmark Methodology）

> 适用于 results/ 下全部数据。撰写于 2026-10-13，全部数字为 Jetson Orin NX 16GB 实测。

## 环境与状态锁定

| 项 | 值 |
|---|---|
| 设备 | Jetson Orin NX 16GB Engineering Ref Dev Kit（Super 模式） |
| 系统 | JetPack 6.2 / L4T 36.4.3 / CUDA 12.6 |
| 功耗与频率 | `nvpmodel -m 0`（MAXN_SUPER）+ `jetson_clocks --fan`，全程锁定 |
| 桌面 | 基准期间 `systemctl stop gdm3`（GUI 关闭，可用 RAM 1.2→更多） |
| 采样 | `tegrastats --interval 200` 全程记录（RAM/GR3D/功耗/温度）→ CSV |
| 引擎 | Strata 0.1.43（strata-jetson aarch64 移植版，patches/0001-0004）与 llama.cpp `3cf03257`（CUDA sm_87 本地构建） |

## 性能指标定义

| 指标 | 定义 | 工具 |
|---|---|---|
| TTFT | 首 token 延迟（流式，从请求发出到首个可见 delta） | bench_perf.py |
| prefill t/s | prompt tokens / TTFT（按桶 128 / 2048，每次 repeat 换唯一 prompt 防服务端缓存重用） | 同上 |
| decode t/s | 生成 tokens /（总时长 − TTFT），temperature=0 | 同上 |
| 读提示 t/s（llama-bench） | pp128 / pp2048，n=128，r=3 | llama-bench -o json |
| 写回答 t/s（llama-bench） | tg128，n=128，r=3 | 同上 |
| 功耗 | VDD_IN 轨均值（200ms 采样） | power_sampler.py |
| 能耗比 | 功耗均值 / decode t/s = J/token | analyze_power.py |
| 专家缓存命中率 | 引擎自报（hits/lookups, decode 阶段） | strata 引擎日志 |
| TTFT/对齐说明 | Strata 报告口径为"读提示/写回答"，与 llama-bench 的 pp/tg 同义对照 | — |

## 质量评测协议

| 项 | GSM8K | ARC-Challenge |
|---|---|---|
| 题量 | 100（seed=42 固定抽样，test split） | 100（同） |
| 提示 | 3-shot CoT（训练集固定 3 例，`#### N` 格式） | 0-shot，"Answer with just the letter" |
| 解码 | 贪心（temperature=0），max_tokens 512 | 贪心，max_tokens 512（思考模型预算） |
| 判分 | 最终数字精确匹配（优先 `#### N`；否则自下而上首个含数字行；剥 `<answer>/<think>` 标签） | 最后一个独立 A-E 字母；回退：回答唯一复述某选项原文 |
| 端点 | 统一 OpenAI `chat/completions`（Strata 无 completions 端点，故全表用 chat） | 同 |

### 已知限制（如实声明）

1. **3-shot 而非 8-shot**：全模型一致故横向可比；与公开榜单绝对值不可直接比。
2. **生成式 ARC**：非 loglikelihood 协议，弱指令跟随模型的字母分可能低估（已加选项文本回退缓解）。
3. **思考模型 budget**：max_tokens 512 含思考链；R1/Z1/Qwen3.8 的有效答题预算因此缩水，分数视为其"短思考"下限。
4. **hunyuan 首轮数据**：解析器缺陷（`<answer>` 标签）导致 GSM8K 全 null 的那批数据已被修复后重测替代。
5. **量化不可比项**：Strata Q2_0（2bit 专家）vs 动物园 Q4_K_M——同表展示为"边缘部署形态"对比，非同量化对比。
6. **Qwen3.5 家族的 GSM8K 分数是格式遵循下限**：该家族的新聊天风格会用英文写元推理、不复刻
   few-shot 的 `#### N` 格式，512 token 预算常被冗长推理耗尽，导致 GSM8K 0.06-0.15（ARC 正常爬坡
   0.56→0.79 证明能力无碍）。分数解读为"严格少样本协议下的格式遵循分"，非该家族数学能力上限。
7. **思考模型的 ARC=0 是预算耗尽的标记**（如 intern-s1-mini、qwen35-9b）：思考链吃满 512 token
   后 content 为空，字母解析无输入。协议对所有模型一致，故横向排序仍有意义。

## 公平性约定

- 所有模型同机、同锁频、同采样间隔、同评测协议串行执行。
- 模型权重评测后即删（磁盘管道），结果 JSON（含每题记录与原始回答摘录）全部入库。
- Strata 引擎的两组配置（默认 / pool-workers=3）分别成表，不混用。

## 附：Jetson 推理调参矩阵（"榨干潜力"实验，qwen35-4b Q4_K_M）

llama-bench 全组合扫描：线程 {4,6,8} × flash-attn {0,1} × KV 量化 {f16,q8_0}²（24 组，
`results/raw/tuning_matrix_*.json`）。已完成 fa=0 切片 17 组：

| 发现 | 数据 |
|---|---|
| decode 与 CPU 线程数无关 | t=4/6/8 全部 17.8 tok/s（4B 模型 decode 由 GPU/带宽决定） |
| flash-attn 对 prefill +6% | 549 vs 519 tok/s @pp2048（t=6） |
| KV 量化（q8_0）在本尺寸无速度收益 | f16 与 q8_0 差异在噪声内（省显存的场景才有意义） |
| fa=1 + qwen35-4b GGUF 存在加载失败组合 | "failed to create context"，已记录为该转换文件的兼容性问题 |

结论：Orin NX 上小模型 decode 已由 GPU/带宽封顶，调参空间在 **prefill（开 FA）** 与
**内存配比（Strata 场景）**；对大 MoE（Strata）则如正文：`--pool-workers` 拓扑是最大杠杆（+59%）。
