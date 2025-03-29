import argparse
import asyncio
import requests
from pathlib import Path
import json
from typing import List, Dict

from utils import load_utils
from utils.prompt_utils import PromptUtils
from utils.llm_client import QWQ_LLM, RequestType
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("F1_Evaluator")


class F1_Evaluator:
    def __init__(self, optimized_path: str = "",
                 model_url: str = "",
                 dataset_name: str = "",
                 dataset_path: str = "",
                 api_key: str = ""):
        logger.info(f"初始化 F1_Evaluator")
        self.root_path = Path(optimized_path)
        self.prompt_utils = PromptUtils(self.root_path)
        # self.model_url = model_url  # LLM 的 API 地址
        # self.dataset_name = dataset_name.lower()
        # self.dataset_path = dataset_path
        # self.api_key = api_key
        # self.api_type = "openai"
        # self.data_utils = DataUtils(self.root_path)
        # QWQ_LLM.initialize(
        #     optimize_kwargs={"use_ollama": False},
        #     evaluate_kwargs={"use_ollama": False},
        #     execute_kwargs={"use_ollama": False},
        #     analyze_kwargs={"use_ollama": False},
        #     generate_kwargs={"use_ollama": False}
        # )
        # self.llm_client = QWQ_LLM.get_instance()

    async def query_llm_async(self, prompt: str, question: str) -> str:
        """异步调用 QWQ_LLM 客户端接口获取回答"""
        full_prompt = f"{prompt}\n\n{question}"
        logger.info(f"查询 LLM，问题: {question}")
        try:
            # 这里我们选择 EXECUTE 类型调用，可根据实际场景调整 RequestType
            answer = await self.llm_client.responser(request_type=RequestType.EXECUTE,
                                                     messages=[{"role": "user", "content": full_prompt}])
            logger.info(f"LLM 返回答案: {answer}")
            return answer
        except Exception as e:
            logger.error(f"调用 LLM 出现异常: {e}")
            return ""

    def query_llm(self, prompt: str, question: str) -> str:
        """对外提供同步接口，内部调用异步 query_llm_async"""
        return asyncio.run(self.query_llm_async(prompt, question))

    # def load_data(self) -> List[Dict]:
    #     """
    #     加载数据集，根据 dataset_name 判断加载方式：
    #       - 如果是 "bbh-navigate" 或 "bigbench"，则使用 Hugging Face datasets 加载 BigBench 的 navigate 子集；
    #       - 如果是 "liar"、"wsc" 或 "avg.perf."（及其变体），则直接使用 load_dataset 导入；
    #       - 如果是 "gpqa"，则从本地 JSON 文件加载；
    #       - 其他情况，输出不支持的提示。
    #     返回加载的样本数据
    #     """
    #     logger.info(f"加载数据集: {self.dataset_name}")
    #     if self.dataset_name in ["bbh-navigate", "bigbench"]:
    #         dataset = load_dataset("bigbench", "navigate")
    #         logger.info(f"成功加载 BigBench 'navigate' 子集，共 {len(dataset['test'])} 条数据")
    #         return dataset["test"]
    #     elif self.dataset_name == "liar":
    #         dataset = load_dataset("liar")
    #         logger.info(f"成功加载 LIAR 数据集，共 {len(dataset['test'])} 条数据")
    #         return dataset["test"]
    #     elif self.dataset_name == "wsc":
    #         dataset = load_dataset("wsc")
    #         logger.info(f"成功加载 WSC 数据集，共 {len(dataset['test'])} 条数据")
    #         return dataset["test"]
    #     elif self.dataset_name in ["avg.perf.", "avg_perf", "avgperf"]:
    #         dataset = load_dataset("avg_perf")
    #         logger.info(f"成功加载 Avg.Perf. 数据集，共 {len(dataset['test'])} 条数据")
    #         return dataset["test"]
    #     elif self.dataset_name == "gpqa":
    #         dataset_file = Path(self.dataset_path)
    #         if not dataset_file.exists():
    #             logger.error(f"数据集文件不存在: {self.dataset_path}")
    #             return []
    #         try:
    #             with open(dataset_file, 'r', encoding='utf-8') as f:
    #                 data = json.load(f)
    #             logger.info(f"成功加载 GPQA 数据集，共 {len(data)} 条数据")
    #             return data
    #         except json.JSONDecodeError as e:
    #             logger.error(f"解析 JSON 失败: {e}")
    #             return []
    #     else:
    #         logger.error(f"不支持的数据集类型: {self.dataset_name}")
    #         return []

    def calculate_f1(self, prediction: str, ground_truth: str):
        """计算单个 F1 分数"""
        pred_tokens = prediction.split()
        truth_tokens = ground_truth.split()
        common = set(pred_tokens) & set(truth_tokens)

        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return f1

    def calculate_f1_list(self, data: List[Dict[str, str]], predictions: List[str]) -> float:
        """计算数据集的平均 F1 分数"""
        required_keys = {"question", "answer"}
        f1_scores = []

        if len(data) != len(predictions):
            raise ValueError("data 和 predictions 长度不一致")
        for item, prediction in zip(data, predictions):
            if not required_keys.issubset(item.keys()):
                logger.error(f"数据字段缺失：{item}")

            question = item.get("question")
            ground_truth = item.get("answer")

            if not question or not ground_truth:
                logger.warning("样本数据缺少 question 或 answer 字段，跳过该样本。")
                continue

            # logger.info(f"🔍 问题: {question}, 答案: {ground_truth}, 预测: {prediction}")

            f1 = self.calculate_f1(prediction, ground_truth)
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        logger.info(f"📊 数据集平均 F1 分数: {avg_f1:.4f}")
        return avg_f1

    def execute(self, prompt: str, qa: List[Dict]) -> List[Dict]:
        """执行llm查询"""
        logger.info(f"开始处理 {len(qa)} 条数据")
        results = []
        for item in qa:
            question = item.get("question")
            answer = self.query_llm(prompt, question)
            results.append({"question": question, "answer": answer})
        return results

    def evaluate_spo(self):
        """计算F1分数"""
        logger.info("开始评估")
        prompt = self.prompt_utils.get_final_prompt()
        data = load_utils.load_meta_data()
        if not data:
            logger.error("未加载到数据，评估终止。")
            return 0.0

        answers = self.execute(prompt, data)
        avg_f1 = self.calculate_f1_list(data, answers)
        logger.info(f"📊 数据集平均 F1 分数: {avg_f1:.4f}")
        return avg_f1


def parse_args():
    parser = argparse.ArgumentParser(description="SPO PromptOptimizer CLI")
    parser.add_argument("--uid", type=str, default="3991ad42-c46b-4f2f-9dde-de015aaf5bde", help="优化输出路径的 UID")
    parser.add_argument("--name", type=str, default="Navigate", help="项目名称")
    parser.add_argument("--model-url", type=str, default="https://api.chatanywhere.com.cn/v1", help="LLM 模型接口地址")
    parser.add_argument("--api-key", type=str, default="sk-iX0M9keAJemCgNFqvQMVLyWkcembRT27ix50aymLnvZ18QuT",
                        help="线上 API 的密钥")
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
    # evaluator.evaluate_spo()
