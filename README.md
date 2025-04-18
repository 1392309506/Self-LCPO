# Self-supervised length control prompts optimization

自监督长度控制提示优化（Self-LCPO）

安装依赖：

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

自动更新`requirements.txt`文件

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


