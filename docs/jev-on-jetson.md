# LLM-as-Jev：在 Jetson 上零训练实现 Jev 式校准决策

对应论文：**"LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When and How to
Fine-Tune Them"**（Microsoft Research，Yinheng Li 等，arXiv 2610.02076，2026-10）。

## 论文核心思想

"Jev 式决策模型"（Jev = Jevons 悖论的梗）指：给定预定义选项集，直接返回一个**校准的
类别概率分布**，而不是自由生成文本。工业界通常为此单独训练 judge/verifier 模型；
论文证明**任意现成因果 LLM 不改一行权重就是这样的决策模型**：

1. 把选项编号为 `[1]..[K]` 写进用户消息；
2. 把助手回合 prefill 到 `Best answer: [`；
3. 对每个候选后缀 `jj]` 计算联合 token 对数概率（KV 缓存共享、候选分支并行
   teacher-forcing），softmax 后即得候选分布 —— **架构零改动、零训练**。

论文关键结论（与我们复测一致）：

- 免训练冻结模型的评分决策**优于解析自由生成输出**（知识已经在概率里，生成反而丢）；
- 概率分布**天然校准**（Qwen3.5-4B ECE≈0.057，无需 post-hoc 温度缩放）；
- 小模型（如 Qwen3-0.6B）生成能力差但概率分布仍有用，微调后进一步提升；
- 评分与生成对同一题的 argmax 高度一致——分歧集中在"生成表达失败"的题上。

## 本仓库的实现（`scripts/jev_score.py`）

在 llama.cpp server（`work/llama-build`，rev 3cf03257）上实现了同款评分管线，
利用了两个 server 端点：

| 步骤 | 端点 | 说明 |
|---|---|---|
| 模板渲染 | `POST /apply-template` | 用 GGUF 内置 chat template 渲染用户消息，`chat_template_kwargs={"enable_thinking": false}` 关闭思考模式；无模板的基座模型（OpenELM）回退为原始文本 |
| 概率读取 | `POST /completion` | prompt 追加 `Best answer: [`，`max_tokens=1, n_probs=100, temperature=0` |

**关键工程简化**：K≤9 时各候选数字首 token 互不相同（尾 token `]` 为公共后缀，
softmax 归一化下相消），因此**每题只需 1 次调用**读 top-k 概率，而不是论文的
K 次并行分支。server 的 `n_probs` 路径对**全词表 logits 做 softmax** 后取 top-k
（`server-common.cpp get_token_probabilities`），返回的是精确概率而非 top-k 重归一。
候选覆盖度（coverage）作为健全性指标随结果记录。

每题成本：1 次 prefill（ARC 题干约 100-300 token）。Orin NX 上 qwen35-4b
100 题共 **29 秒**；同等生成式评测（512 token/题）需约 17 分钟。**~35× 加速**。

## 评测设置

- 任务：ARC-Challenge 100 题（seed=42，与 `bench_quality.py` 完全同题同序）；
- 指标：accuracy（argmax）、ECE（10 等宽置信桶）、多类 Brier、平均 NLL/熵、
  候选覆盖度；
- 对照：与 `results/raw/quality_*.json` 里各模型的生成式 ARC 分数按题对齐，
  报告 agreement（评分 argmax == 生成解析答案）。

## 结果（Orin NX 16GB Super，Q4_K_M GGUF 全 GPU，ARC-Challenge 100 题 seed=42）

| judge 模型 | 参数量(B) | JEV acc | 生成式 acc | Δ | ECE | Brier | agreement | 耗时(s) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B (Q8) | 0.6 | **0.62** | —¹ | — | 0.294 | 0.647 | — | 5.2 |
| Qwen3.5-0.8B | 0.8 | **0.61** | 0.56 | +0.05 | 0.129 | 0.523 | 0.52 | 9.4 |
| Qwen3.5-2B | 2.0 | **0.77** | 0.63 | +0.14 | 0.085 | 0.356 | 0.59 | 12.9 |
| Qwen3.5-4B | 4.0 | **0.93** | 0.79 | +0.14 | 0.048 | 0.109 | 0.79 | 29.0 |
| OpenELM-270M | 0.27 | 0.20 | 0.26 | −0.06 | 0.672 | 1.370 | 0.51 | 2.8 |
| OpenELM-1.1B | 1.1 | 0.24 | 0.14 | +0.10 | 0.661 | 1.373 | 0.53 | 6.4 |
| OpenELM-3B | 3.0 | 0.26 | 0.25 | +0.01 | 0.713 | 1.433 | 0.48 | 13.5 |

¹ Qwen3-0.6B 仅做过 6 题冒烟生成式测试，无 100 题基线；它正是论文微调实验的基座，
  其免训练 JEV 0.62 与论文"小模型分布可用但欠校准"的描述一致。

### 发现

1. **"LLM 已经是 Jev"对 ≥2B 对话模型成立，且更强**：Qwen3.5 全系 JEV 准确率
   全面超过自家生成式解析（+5/+14/+14 分），4B 达 0.93；ECE 随规模单调变好，
   4B 的 **0.048** 与论文报告的 ≈0.057 同量级——概率分布天然校准，无需温度缩放。
2. **分歧的方向性**：qwen35-4b 在生成式答对的 79 题里 97.5%（77 题）与评分
   argmax 一致，仅 2 题评分更差；反过来 **15 题生成失败但评分 argmax 正确**——
   生成表达失败不等于"不知道"，概率里有答案。
3. **对基座小模型不成立**：OpenELM 三兄弟（无指令微调）JEV ≈ 随机（0.20-0.26，
   4 选 1 随机 0.25），ECE 0.66-0.71 严重失准。论文"什么时候需要微调"的答案
   在边缘侧很具体：**<1B 或 base 模型不能直接当 Jev judge**。
4. **决策类负载的边缘路径**：100 题评分 3-29 秒 vs 生成式评测 10-17 分钟
   （~35×）；分类/路由/验证/审核类任务在 Jetson 上应走评分路径，还免掉了
   输出解析的整套脆弱逻辑（本仓库 zoo 评测为此打过 4 轮解析补丁）。

### 局限

- 只复现了论文的 **choice**（多选一）任务与免训练部分；noul（二元验证）、
  score（序数打分）与 LoRA 微调（listwise 损失）未实现——前者可在 GSM8K
  候选解验证上直接套用，后者受限于 Jetson 训练吞吐，留作后续。
- ARC 选项 3/4/5 混合，K≤9 的"首 token 判别"简化对 K≥10（两位数编号）不适用，
  那时需回到论文的后缀 trie 逐分支打分（每候选一次 prefill）。

## 边缘侧的意义

1. **一个模型顶两个用**：同一份权重既当生成器又当 verifier/judge/router，
   16GB 统一内存的 Jetson 上省下整个 judge 模型的显存与加载时间；
2. **校准免费**：概率分布自带置信度，低置信样本可直接转人工/更大模型，
   这是生成+解析管线给不出的信号；
3. **35× 延迟优势**：决策类任务（分类/审核/路由/验证）在边缘上应走评分路径，
   而非先生成再解析；
4. **小模型翻身**：生成烂的模型（见 zoo 里 OpenELM 的生成式分数）其概率分布
   可能仍可用——边缘选型时应同时测两条路径。

## 复现

```bash
PY310=~/miniconda3/envs/py310/bin
$PY310/python3 scripts/jev_score.py --model work/zoo/gguf/qwen35-4b.gguf \
    --tag qwen35-4b --compare-gen
# 汇总所有 judge
$PY310/python3 scripts/collect_jev.py
```

原始分布逐题存于 `results/jev/jev_<tag>.json`。
