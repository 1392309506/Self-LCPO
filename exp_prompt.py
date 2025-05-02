from __future__ import annotations

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from component.config_loader import ConfigLoader
from prompt.execute_prompt import SPO_PROMPT, COT_PROMPT
from prompt.extract_prompt import EXTRACT_ANSWER_PROMPT

from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from component.f1_score import F1_Evaluator
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("exp_prompt")

class Prompt_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, token: int = 0, prompt: str = "",
                 template: str = "", is_extract:str="true"):
        self.config = config
        self.model_name = model_name
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.qa = self.loadUtil.load_json(0)

        self.F1_Evaluator = F1_Evaluator()
        self.llm = ChatLLM(
            api_type=self.model.get("api_type"),
            api_key=self.model.get("api_key"),
            base_url=self.model.get("base_url"),
            params=self.model.get("params"),
            name="exp_prompt"
        )
        self.max_concurrent_requests = 5
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.token = token
        self.f1_score = 0
        self.acc = 0
        self.cnt = 1
        self.prompt = prompt
        self.template = template
        self.is_extract = is_extract

        extract_model = config.models["gpt"]
        self.extract_llm = ChatLLM(
            api_type=extract_model.get("api_type"),
            api_key=extract_model.get("api_key"),
            base_url=extract_model.get("base_url"),
            params=extract_model.get("params"),
            name="extract_llm"
        )

        self.results = []

    def _save_results(self):
        """保存评估结果，包括 F1 分数、准确率以及模型的回答"""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 使用“exp_prompt”作为二级文件夹，日期作为三级文件夹
        timestamp = datetime.now().strftime("%Y%m%d")
        folder_name = f"{timestamp}_{self.model_name}_{self.dataset}"
        folder = results_dir / "exp_prompt" / timestamp / folder_name
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

        # 保存 results
        results_file_path = folder / "results_data.json"
        try:
            # 过滤掉为 None 的元素
            filtered_results = [result for result in self.results if result is not None]

            with open(results_file_path, "w", encoding="utf-8") as f:
                json.dump(filtered_results, f, indent=4, ensure_ascii=False)
            logger.info(f"Results data saved to {results_file_path}")
        except Exception as e:
            logger.error(f"Failed to save results data: {e}")

    async def _execute_prompt(self, n: int) -> List[str]:
        """并发执行提示"""
        prompt = ""
        if self.prompt == "spo":
            prompt = SPO_PROMPT
        elif self.prompt == "cot":
            prompt = COT_PROMPT
        else:
            prompt = self.template.format(count=n)

        tasks = [self._fetch_answer(item, prompt) for item in self.qa]
        results = await asyncio.gather(*tasks)
        return results

    async def _fetch_answer(self, item: dict, prompt: str = "") -> Optional[str]:
        """发送异步请求获取答案（失败返回 None，不污染 F1 数据）"""
        question = item.get("question")
        content = prompt + "\n" + question

        async with self._semaphore:
            try:
                messages = [{"role": "user", "content": content}]
                response = await self.llm.chat(messages)

                if response is None or not hasattr(response, "content") or response.content is None:
                    logger.warning(f"❌ 模型响应为空，跳过该问题: {question[:40]}...")
                    return None

                answer = LoadUtils.extract_content(response.content, "answer")
                standard_answer = item.get("answer")
                if answer == None:
                    if self.is_extract == "true":
                    # LLM 提取答案（修正）
                        answer = await self._extract(standard=standard_answer, personal=response.content, question=question)
                    else :
                        answer = "None"
                token_count = self.llm.get_current_token()  # 获取当前API请求消耗的token数量
                # 记录答案和token消耗
                result = {
                    "question": question,
                    "answer": answer,
                    "token_consumed": token_count
                }
                self.results.append(result)  # 将每个问题的结果保存到self.qa_results中
                logger.info(str(self.cnt) + "( " + str(token_count) + " ): " + answer + " | " + standard_answer)
                self.cnt += 1
                print("ok")
                return answer
            except Exception as e:
                logger.error(f"⚠️ 模型调用失败，跳过问题：{question[:40]}... 错误：{str(e)}")
                return None

    async def _extract(self, standard: str, personal: str, question: str) -> str:
        prompt = EXTRACT_ANSWER_PROMPT.format(standard=standard, personal=personal)
        messages = [{"role": "user", "content": prompt}]
        response = await self.extract_llm.chat(messages)

        judge = LoadUtils.extract_content(response.content, "judge")
        if judge == "1" or judge == "[1]" or judge == "(1)":
            personal = standard
        else:
            personal = "None"
        return personal

    async def run(self):
        """执行实验流程，获取每道题的答案和消耗的token数量，并保存结果"""
        logger.info(f"开始训练的token数量为：{self.token}")
        await self._execute_prompt(self.token)
        answers = [item["answer"] for item in self.results if item["answer"] is not None]
        try:
            # 计算 F1 和 ACC
            self.f1_score = self.F1_Evaluator.calculate_f1_list(self.qa, answers)
            self.acc = self.F1_Evaluator.calculate_ACC(self.qa, answers)
            avg_token = self.llm.get_total_token() / len(self.qa)
            logger.info(f"total_token={self.llm.get_total_token()}")
            logger.info(
                f"self.token={self.token} | F1 Score={self.f1_score:.4f} | ACC Score={self.acc:.4f} | avg_token={avg_token:.2f}")

            # 保存结果
            self._save_results()
            logger.info("实验完成")

        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")
            # 计算 分数
            self.f1_score = self.F1_Evaluator.calculate_f1_list(self.qa, answers)
            self.acc = self.F1_Evaluator.calculate_ACC(self.qa, answers)
            avg_token = self.llm.get_total_token() / len(self.qa)
            logger.info(f"total_token={self.llm.get_total_token()}")
            logger.info(
                f"self.token={self.token} | F1 Score={self.f1_score:.4f} | ACC Score={self.acc:.4f} | avg_token={avg_token}")

            self._save_results()


def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="o3", help="使用的模型名称")
    parser.add_argument("--dataset", type=str, default="gpqa", help="评估使用的数据集名称")
    parser.add_argument("--token", type=int, default=0, help="用于测试的 token 数量")
    parser.add_argument("--prompt", type=str, default="", help="用于测试的特殊提示")
    parser.add_argument("--template", type=str, default="GPQA_PROMPT", help="使用的prompt模板")
    parser.add_argument("--is_extract", type=str, default="true", help="是否需要提取")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)
    try:
        config = ConfigLoader(args.config)
        runner = Prompt_Runner(config, args.model_name, args.dataset, token=args.token, prompt=args.prompt,
                               template=args.template,is_extract=args.is_extract)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.close()
    except Exception as e:
        logger.error(f"实验启动失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
