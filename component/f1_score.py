import argparse
from typing import List, Dict

from utils.logger_utils import LoggerUtil
logger = LoggerUtil.get_logger("F1_Evaluator")

class F1_Evaluator:
    def __init__(self):
        logger.info(f"初始化 F1_Evaluator")
        self.len=0

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
        skip = 0
        for i in range(total_count):
            # 从字典中获取正确答案（使用 'answer' 作为键）
            correct_answer = data[i].get('answer')
            prediction = predictions[i]
            if prediction is None or not isinstance(prediction, str) or prediction.strip().upper() == "ERROR":
                skip+=1
            # 进行精确匹配比较
            if correct_answer == prediction:
                correct_count += 1
        self.len=total_count-skip
        # 处理空列表情况
        return correct_count / self.len if total_count > 0 else 0.0
    def get_len(self)-> int:
        return self.len

def parse_args():
    parser = argparse.ArgumentParser(description="F1 Evaluator")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = F1_Evaluator()
    # evaluator.evaluate_spo()
