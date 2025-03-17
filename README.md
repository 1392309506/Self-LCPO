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

**目前的问题**

我之前删除了metagpt库，llm_client出现了一些差错，我需要给它加回来。。。

# ConfigLoader

用于读取配置文件config_llm.yaml

```
# 实验参数
experiment:
  n_i_values: [50, 100, 200, 500, 1000]
  max_questions: 50

# 数据集配置
datasets:
  Navigate: navigate.yaml
#  FEVER: fever_subset.json
#  LogiQA: logiqa_subset.json

# 模型配置
models:
  - name: "gpt-3.5-turbo"
    api-type: "openai"
    base_url: "https://api.chatanywhere.tech"
    api_keys: ""
    params:
      model: "gpt-3.5-turbo"
      temperature: 0.3
      max_tokens: 1024
  - name: "deepseek-r1"
    api-type: "openai"
    base_url: "https://api.deepseek.com/v1"
    api_keys: ""
    params:
      model: "deepseek-r1"
      temperature: 0.3
      max_tokens: 1024
```

# ExperimentRunner 接口文档

## 简介

`ExperimentRunner` 是一个用于评估大模型（如 GPT-3.5-turbo）性能的 Python 工具。它加载数据集、调用模型进行推理、计算 F1 分数，并保存和可视化实验结果。

这个类用于验证长度和模型性能的关系，实验可选用不同的大模型作为参数

## 运行方式

```bash
python experiment_runner.py --config config/config_llm.yaml
```

#### `_load_dataset(self, dataset_path: str) -> list`

**参数：**

- `dataset_path (str)`: 数据集文件路径。

**返回：**

- `list`：解析后的数据集。

**功能：** 从 YAML 文件加载数据集。

------

#### `call_model(self, model_cfg: dict, prompt: str) -> tuple`

**参数：**

- `model_cfg (dict)`: 模型配置，包括 API Key 和参数。
- `prompt (str)`: 输入给模型的提示词。

**返回：**

- `tuple(str, int)`: 模型响应内容及消耗的 token 数。

**功能：** 调用 OpenAI API 获取模型输出，并自动进行失败重试。

------

#### `extract_answer(self, response: str) -> str`

**参数：**

- `response (str)`: 模型的原始输出。

**返回：**

- `str`: 提取出的答案。

**功能：** 从模型响应中提取答案。

------

#### `calculate_f1(self, predicted: str, expected: str) -> float`

**参数：**

- `predicted (str)`: 预测结果。
- `expected (str)`: 真实答案。

**返回：**

- `float`: 计算得到的 F1 分数。

**功能：** 计算 F1 分数，以衡量预测结果的准确性。

------

#### `evaluate_model(self, model_cfg: dict, data: list, n_i: int, prompt: str) -> float`

**参数：**

- `model_cfg (dict)`: 模型配置。
- `data (list)`: 数据集列表。
- `n_i (int)`: 生成的 token 数限制。
- `prompt (str)`: 提示词。

**返回：**

- `float`: 平均 F1 分数。

**功能：** 使用 `call_model` 调用模型并计算 F1 分数。

------

#### `_save_results(self) -> None`

**功能：** 将实验结果保存到 `results/results.json`。

------

#### `visualize(self) -> None`

**功能：** 使用 `matplotlib` 可视化实验结果，并保存为 `results/performance.png`。

------

#### `run(self, dataset_name: str) -> None`

**参数：**

- `dataset_name (str)`: 需要评估的数据集名称。

**功能：** 执行完整的实验流程，包括数据加载、模型调用、结果计算与可视化。

### 配置文件格式（YAML）

示例 `config_llm.yaml`：

```yaml
experiment:
  n_i_values: [10, 50, 100]
models:
  - name: gpt-3.5-turbo
    api_key: "your-openai-api-key"
    base_url: "https://api.openai.com/v1"
    params:
      model: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 512
datasets:
  dataset1: "data/dataset1.yaml"
```

### 错误处理

- 数据集文件不存在时，抛出 `FileNotFoundError`。
- API 调用失败时，`tenacity` 进行自动重试。
- JSON 解析失败时，抛出 `ValueError`。

### 日志

日志信息默认输出至控制台，可通过 `logging` 进行调试。

# F1_Evaluator 接口文档

## 简介

`F1_Evaluator` 主要用于评估 LLM（大语言模型）的 F1 分数。它支持从不同数据集加载数据，并调用线上 LLM 进行查询，最终计算 F1 分数。

这个类用于计算经过SPO优化后的提示，从workspace中读取并选用大模型进行执行，然后进行F1分数的计算。

## 初始化

```python
F1_Evaluator(
    optimized_path: str,
    model_url: str,
    dataset_name: str,
    dataset_path: str,
    api_key: str = "",
)
```

### 参数

- `optimized_path` (str): 训练优化结果存储路径。
- `model_url` (str): LLM API 的 URL。
- `dataset_name` (str): 需要评估的数据集名称。
- `dataset_path` (str): 本地数据集的路径（适用于 GPQA 数据集）。
- `api_key` (str, 可选): 用于 LLM API 认证的密钥。

## 方法

### `get_final_prompt()`

```python
def get_final_prompt(self) -> str
```

#### 功能

获取优化过程中最优的 `prompt`。

#### 返回值

- `str`: 返回最优 prompt。

------

### `load_data()`

```python
def load_data(self) -> List[Dict]
```

#### 功能

加载指定数据集。

#### 返回值

- `List[Dict]`: 加载的数据集，每个样本是一个字典。

#### 支持的数据集

- `bbh-navigate`, `bigbench`: 通过 Hugging Face `bigbench` 数据集加载 `navigate` 子集。
- `liar`, `wsc`, `avg.perf.`: 通过 `load_dataset` 加载。
- `gpqa`: 从本地 JSON 文件加载。

------

### `query_llm(prompt: str, question: str)`

```python
def query_llm(self, prompt: str, question: str) -> str
```

#### 功能

向 LLM 发送查询，并获取回答。

#### 参数

- `prompt` (str): 需要提供给 LLM 的 prompt。
- `question` (str): 需要 LLM 生成回答的问题。

#### 返回值

- `str`: LLM 返回的文本回答。

------

### `compute_f1(prediction: str, ground_truth: str)`

```python
def compute_f1(self, prediction: str, ground_truth: str) -> float
```

#### 功能

计算预测答案与真实答案的 F1 分数。

#### 参数

- `prediction` (str): LLM 预测的答案。
- `ground_truth` (str): 标准答案。

#### 返回值

- `float`: 计算出的 F1 分数。

------

### `execute(qa: List[Dict])`

```python
def execute(self, qa: List[Dict]) -> List[Dict]
```

#### 功能

使用 LLM 处理输入问题，并返回回答结果。

#### 参数

- `qa` (List[Dict]): 包含 `question` 键的字典列表。

#### 返回值

- `List[Dict]`: 每个字典包含 `question` 和 LLM 生成的 `answer`。

------

### `evaluate()`

```python
def evaluate(self) -> float
```

#### 功能

计算数据集上的平均 F1 分数。

#### 返回值

- `float`: 平均 F1 分数。

## 命令行使用方法

```bash
python f1_evaluator.py \
    --uid 3991ad42-c46b-4f2f-9dde-de015aaf5bde \
    --name Navigate \
    --model-url https://api.chatanywhere.com.cn/v1 \
    --api-key YOUR_API_KEY \
    --dataset-name bigbench \
    --dataset-path dataset
```

## 依赖

- `argparse`
- `requests`
- `datasets`
- `json`
- `pathlib`
- `typing`

## 日志

`F1_Evaluator` 使用 `LoggerUtil` 进行日志记录，日志输出包括数据加载情况、API 请求状态、F1 计算详情等。

## 示例

```python
evaluator = F1_Evaluator(
    optimized_path="workspace/3991ad42-c46b-4f2f-9dde-de015aaf5bde/Navigate",
    model_url="https://api.chatanywhere.com.cn/v1",
    dataset_name="bigbench",
    dataset_path="dataset",
    api_key="YOUR_API_KEY"
)
avg_f1 = evaluator.evaluate()
print(f"Average F1 Score: {avg_f1:.4f}")
```

# 三分迭代优化算法

记O为性能指标，O越大的长度对应的提示性能越好。

模型假设：

对于一个大模型在仅使用一种数据集实验的情况下，限定的输出长度提示的性能有且只有一个最高值，且该最高值为唯一极值点。

```
left=left_0,right=right_0	#限定最优长度所在区间
O_left=score(Prompt(left))		#计算设定长度为left的提示的性能
O_right=score(Prompt(right))	#计算设定长度为right的提示的性能
for right-left<det:
	L1=(right-right)/3+l		#取三分位L1,L2
	L2=2*L1-left
	O_l=score(Prompt(L1))		#计算长度为L1,L2的prompt的性能
	O_2=score(Prompt(L2))
	if O_1>O_2 && O_2>O_right:	#取左边
		right=L1
		O_right=O_1
	else if O_1>O_left && O_2>O_right:	#取中间
		left=L1,right=L2
		O_left=O_1,O_right=O_2
	else if O_2>O_1 && O_1>O_left:	#取右边
		left=L2
		O_left=O_2
length = (left+right)/2
O_max = score(prompt(length))
```

