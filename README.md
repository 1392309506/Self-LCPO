# Self-supervised length control prompts optimization

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

## 📊 评估策略：

评估完全依赖 LLM-as-a-Judge，无需人工标注。排序提示格式如下：

```python
You will be shown multiple answers generated with different token limits.
Please rank them from best to worst based on correctness, reasoning, completeness, and clarity.
Return a Python list of indices (e.g., [2, 0, 1]). Do not explain your ranking.
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

## 🔧 安装方法：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🛆 自动更新依赖：

```bash
pip install pipreqs -i https://pypi.tuna.tsinghua.edu.cn/simple
pipreqs ./ --encoding=utf8 --force
```

---

> 如需论文介绍、实验结果图或更多 benchmark 数据支持，请参考项目文件夹中的论文文档。

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

## 📊 评估策略：

评估完全依赖 LLM-as-a-Judge，无需人工标注。排序提示格式如下：

```python
You will be shown multiple answers generated with different token limits.
Please rank them from best to worst based on correctness, reasoning, completeness, and clarity.
Return a Python list of indices (e.g., [2, 0, 1]). Do not explain your ranking.
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

## 🔧 安装方法：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🛆 自动更新依赖：

```bash
pip install pipreqs -i https://pypi.tuna.tsinghua.edu.cn/simple
pipreqs ./ --encoding=utf8 --force
```

---

> 如需论文介绍、实验结果图或更多 benchmark 数据支持，请参考项目文件夹中的论文文档。

自监督长度控制提示优化（Self-LCPO）

安装依赖：

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

自动更新`requirements.txt`

```
pip install pipreqs -i https://pypi.tuna.tsinghua.edu.cn/simple
pipreqs ./ --encoding=utf8 --force
```

**项目名称：** 基于大模型的 Token 控制与语义一致性优化实验

---

**项目简介：**

本项目旨在探索如何通过控制提示词（prompt）的 token 长度，有效提升大语言模型（LLM）在问答任务中的推理质量与语义一致性。我们设计了一种基于偏好贝叶斯优化（Preference-based Bayesian Optimization, PBO）的方法，利用大模型自身作为评估者（LLM-as-a-judge），通过 listwise 排序比较不同 token 限制下的回答质量，动态寻找最优的提示 token 数量配置。

---

**背景与动机：**

大语言模型的生成能力受 prompt 长度显著影响。token 数越多通常允许更丰富的推理与细节，但也可能引入冗余、偏题或幻觉问题。因此，寻找一个既充分又高效的 token 长度，对于实际应用中的推理与问答场景尤为关键。

与以往单一 token 设置不同，本项目使用 listwise ranking 的方式对多个生成版本进行语义评估，以更精细地识别“最佳思考长度”。

---

**系统组成：**

1. **Runner 模块**：用于执行实际的生成任务，包括读取数据集中的问答对（QA pairs）、构造带 token 限制的执行提示（EXECUTE_PROMPT），并并发调用 LLM 获取回答结果。
2. **Optimizer 模块**：核心优化器，基于 BoTorch 的 PairwiseGP 建模 token 长度偏好，支持 listwise 比较，通过 LLM 输出的排序结果迭代更新高斯过程，并利用采集函数推荐下一轮候选 token 长度。
3. **评估提示设计（EVALUATE_PROMPT）**：以动态拼接的 QA 回答为输入，指导 LLM 对不同 token 限制下的生成进行排序。评估维度涵盖 Correctness、Reasoning Quality、Completeness 与 Clarity。

---

**方法流程：**

1. 对每个问题，使用多个预设 token 长度生成回答（warm-up 阶段）；
2. 使用 LLM 执行 listwise 排序，评估不同 token 下的输出质量；
3. 将排序结果转化为偏好对，训练 Pairwise GP 模型；
4. 使用采集函数（AnalyticExpectedUtilityOfBestOption）选择下一轮 token 候选长度；
5. 重复步骤 1-4，直到收敛或达到最大迭代步数；
6. 输出最优 token 配置与对应的回答集合，进行可视化与保存。

---

**评估策略：**

评估完全依赖大模型自身进行判断（LLM-as-judge），排序提示格式如下：

```
You will be shown multiple answers generated with different token limits.
Please rank them from best to worst based on correctness, reasoning, completeness, and clarity.
Return a Python list of indices (e.g., [2, 0, 1]). Do not explain your ranking.
```

---

**技术栈：**

- Python 3.10+
- asyncio 并发控制
- OpenAI / 自定义 LLM 接口
- BoTorch + GPyTorch for GP Optimization
- Matplotlib 可视化

---

**适用场景：**

- 多轮推理场景的提示结构优化
- 低资源下的 prompt token trade-off 研究
- LLM 生成任务的内生式语义一致性评估

---

**扩展方向：**

- 支持 Pairwise/Hybrid 模式下的排序提示设计
- 融合 BLEU/F1/Rouge 等外部指标混合评估
- 多模态提示下的 token 控制研究（图文、代码）

---

如需协助构建完整实验报告、补充评估结果或部署 DEMO，请继续与我对接。
