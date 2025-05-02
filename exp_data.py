from __future__ import annotations

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
from random import randint
from typing import List

from component.config_loader import ConfigLoader
from prompt.extract_prompt import EXTRACT_SCORE_PROMPT

from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from component.f1_score import F1_Evaluator
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("exp_data")

EVALUATE_PROMPT = """
This evaluation question belongs to a dataset:
- Current Dataset Accuracy: {acc}
- Average Token Consumption: {token}
- Difficulty Analysis:
    - If this problem requires multi-step reasoning or specialized knowledge, evaluate it as having higher difficulty.
    - If this problem can be solved with simple reasoning or common knowledge, evaluate it as lower difficulty.

Based on the above information, please score the difficulty of this problem on a scale from 1 to 10, with 1 being the easiest and 10 being the most difficult. Please provide a brief explanation for the score.

Question:
"""


class Dataset_Evaluator:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, acc: float=0, token: float = 0,
                 is_extract: str = "true", max_concurrent_requests: int = 5):
        """
        初始化Dataset_Evaluator类。

        :param config: 配置加载器，包含所有配置信息
        :param model_name: 使用的模型名称
        :param dataset: 数据集名称
        :param token: 用于测试的token数量
        :param is_extract: 是否需要从答案中提取标准答案（默认为“true”）
        :param max_concurrent_requests: 最大并发请求数（默认为5）

        """
        self.config = config
        self.model_name = model_name
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.qa = self.loadUtil.load_json(0)

        self.F1_Evaluator = F1_Evaluator()
        self.qa_answers_by_ni = {}  # 存储每个 token 限制下的 QA 对
        self.llm = ChatLLM(
            api_type=self.model.get("api_type"),
            api_key=self.model.get("api_key"),
            base_url=self.model.get("base_url"),
            params=self.model.get("params"),
            name="exp_prompt"
        )

        self.max_concurrent_requests = max_concurrent_requests  # 设置并发请求数
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        self.acc = acc  # 设置初始准确率
        self.token = token  # 设置token数量
        self.f1_score = 0
        self.cnt = 1
        self.is_extract = is_extract  # 是否进行提取

        # 设置extract模型
        extract_model = config.models["gpt"]
        self.extract_llm = ChatLLM(
            api_type=extract_model.get("api_type"),
            api_key=extract_model.get("api_key"),
            base_url=extract_model.get("base_url"),
            params=extract_model.get("params"),
            name="extract_llm"
        )

    def _save_results(self):
        """保存评估结果，包括 F1 分数、准确率以及模型的回答"""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳+随机码文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        random_code = hex(randint(0, 65535))[2:].upper()
        folder = Path("results") / f"{timestamp}_{random_code}"
        folder.mkdir(parents=True, exist_ok=True)

        # 保存分数
        results_path = folder / "results.json"
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({
                    "token": self.token,
                    "f1_score": self.f1_score,
                    "accuracy": self.acc,
                    "total_token": self.llm.get_total_token(),
                    "dataset": self.dataset
                }, f, indent=4, ensure_ascii=False)
            logger.info(f"Results saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        # 保存QA对
        qa_path = folder / "qa_answers.json"
        try:
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(self.qa_answers_by_ni, f, indent=2, ensure_ascii=False)
            logger.info(f"QA pairs saved to {qa_path}")
        except Exception as e:
            logger.error(f"Failed to save QA pairs: {e}")

    async def _execute_prompt(self) -> List[str]:
        """并发执行提示"""
        tasks = [self._fetch_answer(item) for item in self.qa]
        results = await asyncio.gather(*tasks)
        return results

    async def _fetch_answer(self, item: dict) -> str | None:
        """发送异步请求获取答案（失败返回 None，不污染 F1 数据）"""
        question = item.get("question")
        content = EVALUATE_PROMPT.format(acc=self.acc,token=self.token) + "\n" + question

        async with self._semaphore:
            try:
                messages = [{"role": "user", "content": content}]
                response = await self.llm.chat(messages)

                if response is None or not hasattr(response, "content") or response.content is None:
                    logger.warning(f"❌ 模型响应为空，跳过该问题: {question[:40]}...")
                    return None

                answer = LoadUtils.extract_content(response.content, "answer")
                standard_answer = item.get("answer")

                if self.is_extract == "true":
                    # LLM 提取答案（修正）
                    if answer is None:
                        answer = await self._extract(standard=standard_answer, personal=response.content, question=question)

                logger.info(f"{self.cnt}: {answer} | {standard_answer}")
                self.cnt += 1
                return answer
            except Exception as e:
                logger.error(f"⚠️ 模型调用失败，跳过问题：{question[:40]}... 错误：{str(e)}")
                return None

    async def _fetch_difficulty_score(self, item: dict) -> int | None:
        """获取问题的难度分数"""
        question = item.get("question")
        content = EVALUATE_PROMPT.format(acc=self.acc, token=self.token) + "\n" + question

        async with self._semaphore:
            try:
                messages = [{"role": "user", "content": content}]
                response = await self.llm.chat(messages)

                if response is None or not hasattr(response, "content") or response.content is None:
                    logger.warning(f"❌ 模型响应为空，跳过该问题: {question[:40]}...")
                    return None

                # 提取结构化的难度分数（1-10）
                score = LoadUtils.extract_content(response.content, "score")  # 假设你已经提取了结构化分数
                if self.is_extract == "true":
                    # LLM 提取答案（修正）
                    if score is None:
                        answer = await self._extract()

                logger.info(f"{self.cnt}: {answer} | {standard_answer}")

                return int(score)
            except Exception as e:
                logger.error(f"⚠️ 模型调用失败，跳过问题：{question[:40]}... 错误：{str(e)}")
                return None

    async def _extract(self, standard: str, personal: str, question: str) -> str:
        """
        使用 EXTRACT_SCORE_PROMPT 比较标准答案与模型生成的答案，返回结构化的评分结果。

        :param standard: 标准答案
        :param personal: 模型生成的答案
        :param question: 问题内容，用于上下文的引用
        :return: 根据比较结果返回“correct”、“incorrect”或“undetermined”
        """
        prompt = EXTRACT_SCORE_PROMPT.format(standard=standard, personal=personal)
        content = prompt + "\n" + question

        # 发送请求到 extract_llm
        messages = [{"role": "user", "content": content}]
        response = await self.extract_llm.chat(messages)

        # 提取结构化输出（例如 "correct", "incorrect", "undetermined"）
        result = LoadUtils.extract_content(response.content, "judge")

        if result == "correct":
            personal = standard  # 如果答案是正确的，返回标准答案
        elif result == "incorrect":
            personal = "None"  # 如果答案不正确，返回 "None"
        else:
            personal = "undetermined"  # 如果无法确定，返回 "undetermined"

        return personal

    async def run(self):
        """
        执行评估过程，包括获取模型答案、计算评估结果、保存结果
        """
        # 获取模型的答案
        answers = await self._execute_prompt()

        # 存储 QA 对
        qa_pairs = [
            {"question": item.get("question"), "answer": answer}
            for item, answer in zip(self.qa, answers)
        ]
        try:
            self.qa_answers_by_ni[self.token] = qa_pairs

            # 计算评估分数
            self.f1_score = self.F1_Evaluator.calculate_f1_list(self.qa, answers)
            self.acc = self.F1_Evaluator.calculate_ACC(self.qa, answers)
            avg_token = self.llm.get_total_token() / self.F1_Evaluator.get_len()

            logger.info(f"Total tokens used: {self.llm.get_total_token()}")
            logger.info(
                f"Accuracy: {self.acc:.4f} | F1 Score: {self.f1_score:.4f} | Average Tokens per Answer: {avg_token:.2f}")

            # 保存结果
            self._save_results()
            logger.info("Evaluation completed successfully.")

        except Exception as e:
            # 计算并保存分数（如果出现异常）
            self.f1_score = self.F1_Evaluator.calculate_f1_list(self.qa, answers)
            self.acc = self.F1_Evaluator.calculate_ACC(self.qa, answers)
            avg_token = self.llm.get_total_token() / self.F1_Evaluator.get_len()

            logger.info(f"Total tokens used: {self.llm.get_total_token()}")
            logger.info(
                f"Accuracy: {self.acc:.4f} | F1 Score: {self.f1_score:.4f} | Average Tokens per Answer: {avg_token:.2f}")

            self._save_results()
            logger.error(f"Evaluation failed: {str(e)}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Dataset Evaluation with LLM')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="o3", help="使用的模型名称")
    parser.add_argument("--dataset", type=str, default="gpqa", help="评估使用的数据集名称")
    parser.add_argument("--acc", type=float, default=0, help="初始准确率")
    parser.add_argument("--token", type=float, default=0, help="用于测试的token数量")
    parser.add_argument("--is_extract", type=str, default="true", help="是否需要提取答案（默认：true）")
    parser.add_argument("--max_concurrent_requests", type=int, default=5, help="最大并发请求数（默认：5）")

    return parser.parse_args()


def main():
    """
    评估主函数
    """
    # 解析命令行参数
    args = parse_args()
    logger.info(f"Running with config: {args.config}, model: {args.model_name}, dataset: {args.dataset}, "
                f"initial accuracy: {args.acc}, token: {args.token}, is_extract: {args.is_extract}, "
                f"max_concurrent_requests: {args.max_concurrent_requests}")

    try:
        # 加载配置
        config = ConfigLoader(args.config)

        # 初始化 Dataset_Evaluator 类
        evaluator = Dataset_Evaluator(
            config=config,
            model_name=args.model_name,
            dataset=args.dataset,
            acc=args.acc,
            token=args.token,
            is_extract=args.is_extract,
            max_concurrent_requests=args.max_concurrent_requests
        )

        # 执行评估
        evaluator.run()

    except Exception as e:
        logger.error(f"Error occurred during the evaluation process: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()