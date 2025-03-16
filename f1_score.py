import argparse
import requests
from pathlib import Path
import json
from typing import List, Dict
from datasets import load_dataset

from utils.data_utils import DataUtils

from utils.logger_util import LoggerUtil
logger=LoggerUtil.get_logger("F1_Evaluator")

class F1_Evaluator:
    # 可在其他类中调用 F1_Evaluator.evaluate计算输出后的F1分数
    def __init__(self, optimized_path: str,
                 model_url: str,
                 dataset_name: str,
                 dataset_path: str,
                 api_key: str = "",
                 ):
        logger.info(f"初始化 F1_Evaluator，优化路径: {optimized_path}")
        self.root_path = Path(optimized_path)
        self.data_utils = DataUtils(self.root_path)
        self.model_url = model_url  # LLM 的 API 地址
        self.dataset_name = dataset_name.lower()
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.api_type = "openai"

    def get_final_prompt(self):
        """获取最后一轮迭代优化出的 prompt"""
        best_round = self.data_utils.get_best_round()
        if not best_round:
            logger.error("无法找到最佳 prompt，默认返回空字符串")
            return ""
        logger.info(f"加载到最佳 prompt: {best_round['prompt']}")
        return best_round["prompt"]

    def load_data(self) -> List[Dict]:
        """
        加载数据集，根据 dataset_name 判断加载方式：
          - 如果是 "bbh-navigate" 或 "bigbench"，则使用 Hugging Face datasets 加载 BigBench 的 navigate 子集；
          - 如果是 "liar"、"wsc" 或 "avg.perf."（及其变体），则直接使用 load_dataset 导入；
          - 如果是 "gpqa"，则从本地 JSON 文件加载；
          - 其他情况，输出不支持的提示。
        返回加载的样本数据
        """
        logger.info(f"加载数据集: {self.dataset_name}")
        if self.dataset_name in ["bbh-navigate", "bigbench"]:
            dataset = load_dataset("bigbench", "navigate")
            logger.info(f"成功加载 BigBench 'navigate' 子集，共 {len(dataset['test'])} 条数据")
            return dataset["test"]
        elif self.dataset_name == "liar":
            dataset = load_dataset("liar")
            logger.info(f"成功加载 LIAR 数据集，共 {len(dataset['test'])} 条数据")
            return dataset["test"]
        elif self.dataset_name == "wsc":
            dataset = load_dataset("wsc")
            logger.info(f"成功加载 WSC 数据集，共 {len(dataset['test'])} 条数据")
            return dataset["test"]
        elif self.dataset_name in ["avg.perf.", "avg_perf", "avgperf"]:
            dataset = load_dataset("avg_perf")
            logger.info(f"成功加载 Avg.Perf. 数据集，共 {len(dataset['test'])} 条数据")
            return dataset["test"]
        elif self.dataset_name == "gpqa":
            dataset_file = Path(self.dataset_path)
            if not dataset_file.exists():
                logger.error(f"数据集文件不存在: {self.dataset_path}")
                return []
            try:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"成功加载 GPQA 数据集，共 {len(data)} 条数据")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"解析 JSON 失败: {e}")
                return []
        else:
            logger.error(f"不支持的数据集类型: {self.dataset_name}")
            return []
    def query_llm(self, prompt: str, question: str):
        """同步调用线上 LLM 查询接口"""
        logger.info(f"查询 LLM: {self.model_url}，问题: {question}")
        payload = {"prompt": f"{prompt}\n\n{question}", "max_tokens": 256}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Api-Type": self.api_type,
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(self.model_url, json=payload, headers=headers)
            response.raise_for_status()
            answer = response.json().get("text", "").strip()
            logger.info(f"LLM 返回答案: {answer}")
            return answer
        except requests.RequestException as e:
            logger.error(f"请求线上 API 失败: {e}")
            return ""

    def compute_f1(self, prediction: str, ground_truth: str):
        """计算 F1 分数"""
        logger.info(f"计算 F1，预测: {prediction}，标准答案: {ground_truth}")
        pred_tokens = prediction.split()
        truth_tokens = ground_truth.split()
        common = set(pred_tokens) & set(truth_tokens)

        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        logger.info(f"F1 计算结果: {f1:.4f}")
        return f1

    def execute(self, qa: List[Dict]) -> List[Dict]:
        """执行查询"""
        logger.info(f"开始处理 {len(qa)} 条数据")
        results = []
        for item in qa:
            question = item.get("question")
            prompt = self.get_final_prompt()
            answer = self.query_llm(prompt, question)
            results.append({"question": question, "answer": answer})
        return results

    def evaluate(self):
        """计算F1分数"""
        logger.info("开始评估")
        prompt = self.get_final_prompt()
        data = self.load_data()
        if not data:
            logger.error("未加载到数据，评估终止。")
            return 0.0

        answers = self.execute(data)
        f1_scores = []
        for item, result in zip(data, answers):
            question = item.get("question")
            ground_truth = item.get("answer")
            prediction = result.get("answer")
            if not question or not ground_truth:
                logger.warning("样本数据缺少 question 或 answer 字段，跳过该样本。")
                continue
            f1 = self.compute_f1(prediction, ground_truth)
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        logger.info(f"📊 数据集平均 F1 分数: {avg_f1:.4f}")
        return avg_f1

def parse_args():
    parser = argparse.ArgumentParser(description="SPO PromptOptimizer CLI")
    parser.add_argument("--uid", type=str, default="3991ad42-c46b-4f2f-9dde-de015aaf5bde", help="优化输出路径的 UID")
    parser.add_argument("--name", type=str, default="Navigate", help="项目名称")
    parser.add_argument("--model-url", type=str, default="https://api.chatanywhere.com.cn/v1", help="LLM 模型接口地址")
    parser.add_argument("--api-key", type=str, default="sk-iX0M9keAJemCgNFqvQMVLyWkcembRT27ix50aymLnvZ18QuT", help="线上 API 的密钥")
    parser.add_argument("--dataset-name", type=str, default="bigbench", help="数据集名称")
    parser.add_argument("--dataset-path", type=str, default="dataset", help="本地数据集路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = F1_Evaluator(
        optimized_path=str(Path("workspace") / args.uid / args.name),
        model_url=args.model_url,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        api_key=args.api_key,
    )
    evaluator.evaluate()
