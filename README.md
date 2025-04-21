# Self-supervised Length Control Prompts Optimization (Self-LCPO)

## 📌 项目名称：
基于大模型的推理长度自适应控制与语义一致性优化框架

## 📖 项目简介：
本项目旨在探索如何通过自监督机制，动态控制大语言模型（LLM）的推理路径长度，以提升其在问答、逻辑推理等任务中的准确率和 token 使用效率。我们提出的 Self-LCPO 框架结合了大模型自评估（LLM-as-a-Judge）与偏好驱动的贝亚斯优化（PBO），可在无监督场景中识别最优推理长度配置。

该方法无需重训，模块化程度高，适用于 Deepseek-R1、GPT-4o 等不同类型的本地或 API 模型，并在 AGIVal-MATH、BBH-Navigate 等 benchmark 上实现显著性胜胜提升。

## 🔍 背景与动机：
近年来的研究发现，大语言模型在推理任务中的表现与其生成长度呈现出非线性关系。过短的推理路径导致信息不全，过长的推理则引入冗余乃自误导性内容。因此，为每个任务动态寻找最优的 token 长度，成为提升模型效率与质量的关键。

Self-LCPO 采用 listwise 排序机制构造偏好信号，通过高斯过程建模不同长度配置的优势，并高效搜索性能最优点。

## ⚙️ 系统组成：

1. **Runner 模块**：负责数据集读取、构造带 token 限制的提示（EXECUTE_PROMPT），并并发请求 LLM 获取输出。
2. **Optimizer 模块**：利用 BoTorch 实现 Pairwise GP，对 token 长度偏好建模，通过 LLM 排序反馈优化模型。
3. **Evaluator 模块**：构造排序提示（EVALUATE_PROMPT），引导 LLM 从多个维度（准确性、完整性、清晰性）评估不同长度下的输出质量。

## 🔁 方法流程：

1. 初始化：设定多个预设 token 长度，生成初始回答集合；
2. 排序：使用 LLM 进行 listwise 排序，获取偏好结构；
3. 优化：利用 GP 模型建模偏好函数，采样最优长度候选；
4. 迭代：重复上述过程，直到收敛或达到迭代上限；
5. 输出：记录最优 token 设置与对应回答，生成评估报告与可视化图。

## 🔧 安装方法：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🛆 自动更新依赖：
```bash
pip install pipreqs -i https://pypi.tuna.tsinghua.edu.cn/simple
pipreqs ./ --encoding=utf8 --force
```
## 📊 评估策略：
评估完全依赖 LLM-as-a-Judge，无需人工标注。排序提示格式如下：

```python
EVALUATE_PROMPT = """You are comparing the performance of the same LLM with different numbers of tokens.
For each "count" generate a response by answering multiple QA pairs. Each "A" is the answer to its corresponding "Q".
There are {cnt_answers} counts. You must rank **all of them**. Return the index of count from best to worst.
Conclude your response with the final answer formatted in <ranking>the_index_ranking</ranking>.
For example:<ranking>[2, 0, 1, 3]</ranking>

### Reference QA pairs:
{reference_block}

### QA responses from each count:
{answer_block}

### Token count list:
{token_list}

"""

SELF_EVALUATE_PROMPT = """You are comparing the performance of the same LLM with different numbers of tokens.
For each "count" generate a response by answering multiple questions.
There are {cnt_answers} counts. You must rank **all of them**. Return the index of count from best to worst.
Conclude your response with the final answer formatted in <ranking>the_index_ranking</ranking>.
For example:<ranking>[2, 0, 1, 3]</ranking>

### Questions:
{question_block}

### QA responses from each count:
{answer_block}

### Token count list:
{token_list}

"""
```

可支持 listwise、pairwise 和 hybrid 三种偏好建模模式。

## 🧰 技术栈：
- Python 3.10+
- asyncio 并发控制
- OpenAI / 自定义 LLM 接口（支持 HuggingFace、API 模型）
- BoTorch + GPyTorch for Gaussian Process Optimization
- Matplotlib 可视化输出

## 🧠 适用场景：
- 推理任务中的 prompt 长度自动搜索与控制
- 无监督语义一致性优化
- token 成本敏感场景（如在线部署、边缘计算）

## 🚀 扩展方向：
- 集成多模态任务中的 token 长度建模（如图文混合）
- 加入 BLEU/F1/ROUGE 等外部评价指标融合评分
- 面向代码生成、复杂问答任务的通用长度优化引擎