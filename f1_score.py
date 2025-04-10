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
    def __init__(self, optimized_path: str = "",):
        logger.info(f"初始化 F1_Evaluator")
        self.root_path = Path(optimized_path)
        self.prompt_utils = PromptUtils(self.root_path)

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

    def calculate_f1(self, prediction: str, ground_truth: str) -> float:
        """计算单个 F1 分数（支持空值与无效预测容错）"""
        if not prediction or not isinstance(prediction, str) or prediction.strip().upper() == "ERROR":
            logger.warning(f"预测值无效：{prediction}，将视为 F1=0.0")
            return 0.0

        if not ground_truth or not isinstance(ground_truth, str):
            logger.warning(f"标签值无效：{ground_truth}，将视为 F1=0.0")
            return 0.0

        pred_tokens = prediction.strip().split()
        truth_tokens = ground_truth.strip().split()

        if not pred_tokens or not truth_tokens:
            logger.warning(f"预测或标签为空 token，prediction: {prediction}, ground_truth: {ground_truth}")
            return 0.0

        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def calculate_f1_list(self, data: List[Dict[str, str]], predictions: List[str]) -> float:
        """计算数据集的平均 F1 分数（自动过滤非法 prediction）"""
        required_keys = {"question", "answer"}
        f1_scores = []
        skipped = 0

        if len(data) != len(predictions):
            raise ValueError("data 和 predictions 长度不一致")

        for item, prediction in zip(data, predictions):
            if not required_keys.issubset(item.keys()):
                logger.error(f"数据字段缺失：{item}")
                continue

            question = item.get("question")
            ground_truth = item.get("answer")

            if not question or not ground_truth:
                logger.warning("样本数据缺少 question 或 answer 字段，跳过该样本。")
                continue

            if prediction is None or not isinstance(prediction, str) or prediction.strip().upper() == "ERROR":
                logger.warning(f"跳过非法预测值：{prediction}")
                skipped += 1
                continue

            f1 = self.calculate_f1(prediction, ground_truth)
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        logger.info(f"📊 有效样本数: {len(f1_scores)}，跳过: {skipped}，平均 F1 分数: {avg_f1:.4f}")
        return avg_f1

    def calculate_ACC(self, data: List[Dict[str, str]], predictions: List[str]) -> float:
        correct_count = 0
        total_count = len(data)
        if total_count != len(predictions):
            raise -1
        for i in range(total_count):
            # 从字典中获取正确答案（使用 'answer' 作为键）
            correct_answer = data[i].get('answer')
            predicted_answer = predictions[i]
            # 进行精确匹配比较
            if correct_answer == predicted_answer:
                correct_count += 1
        # 处理空列表情况
        return correct_count / total_count if total_count > 0 else 0.0

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
    )
    # evaluator.evaluate_spo()
