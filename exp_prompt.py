from __future__ import annotations

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
from random import randint
from typing import List

from component.config_loader import ConfigLoader
from prompt.execute_prompt import BLANK_PROMPT, SPO_PROMPT, COT_PROMPT
from prompt.extract_prompt import EXTRACT_ANSWER_PROMPT

from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from component.f1_score import F1_Evaluator
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("exp_llm")


class Prompt_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, token: int = 100, prompt: str = "",
                 template: str = "GPQA_PROMPT"):
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
        self.max_concurrent_requests = 5
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.token = token
        self.f1_score = 0
        self.acc = 0
        self.cnt = 1
        self.prompt = prompt
        self.template = template

        extract_model = config.models["gpt"]
        self.extract_llm = ChatLLM(
            api_type=extract_model.get("api_type"),
            api_key=extract_model.get("api_key"),
            base_url=extract_model.get("base_url"),
            params=extract_model.get("params"),
            name="extract_llm"
        )

    def _save_results(self):
        """保存 F1 分数 和 QA 对"""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳+随机码文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        random_code = hex(randint(0, 65535))[2:].upper()
        folder = Path("results") / f"{timestamp}_{random_code}"
        folder.mkdir(parents=True, exist_ok=True)

        # 保存 分数
        results_path = folder / "results.json"
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"set_token": self.token,
                           "f1_score": self.f1_score,
                           "acc": self.acc,
                           "total_token": self.llm.get_total_token(),
                           "dataset": self.dataset}, f, indent=4, ensure_ascii=False)
            logger.info(f"F1 分数已保存至 {results_path}")
        except Exception as e:
            logger.error(f"保存 F1 分数失败: {e}")

        # 保存 QA 对
        qa_path = folder / "qa_answers.json"
        try:
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(self.qa_answers_by_ni, f, indent=2, ensure_ascii=False)
            logger.info(f"QA 对已保存至 {qa_path}")
        except Exception as e:
            logger.error(f"保存 QA 对失败: {e}")

    async def _execute_prompt(self, n: int) -> List[str]:
        """并发执行提示"""
        prompt = ""
        if self.prompt == "blank":
            prompt = BLANK_PROMPT
        elif self.prompt == "spo":
            prompt = SPO_PROMPT
        elif self.prompt == "cot":
            prompt = COT_PROMPT
        else:
            prompt = self.template.format(count=n)
            # if self.model_name == "math":
            #     prompt = MATH_PROMPT.format(count=n)
            # elif self.model_name == "gpqa":
            #     prompt = GPQA_PROMPT.format(count=n)
            # elif self.model_name == "bbh":
            #     prompt = BBH_PROMPT.format(count=n)
            # elif self.model_name == "liar":
            #     prompt = LIAR_PROMPT.format(count=n)

        tasks = [self._fetch_answer(item, prompt) for item in self.qa]
        results = await asyncio.gather(*tasks)
        return results

    async def _fetch_answer(self, item: dict, prompt: str = "") -> str | None:
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
                # LLM 提取答案（修正）
                if answer != standard_answer:
                    answer = await self._extract(standard=standard_answer, personal=response.content)

                logger.info(str(self.cnt) + ": " + answer + " | " + standard_answer)
                self.cnt += 1
                return answer
            except Exception as e:
                logger.error(f"⚠️ 模型调用失败，跳过问题：{question[:40]}... 错误：{str(e)}")
                return None

    async def _extract(self, standard: str, personal: str) -> str:
        prompt = EXTRACT_ANSWER_PROMPT.format(standard=standard, personal=personal)
        # prompt = EXTRACT_PROMPT.format(content=personal)
        messages = [{"role": "user", "content": prompt}]
        response = await self.extract_llm.chat(messages)
        # answer = LoadUtils.extract_content(response.content, "answer")

        judge = LoadUtils.extract_content(response.content, "judge")
        if judge == "1":
            personal = standard
        else:
            personal = "None"
        return personal

    async def run(self):
        """执行实验流程"""
        try:
            logger.info(f"开始训练的token数量为：{self.token}")
            answers = await self._execute_prompt(self.token)

            # 存储 QA 对
            qa_pairs = [
                {"question": item.get("question"), "answer": answer}
                for item, answer in zip(self.qa, answers)
            ]
            self.qa_answers_by_ni[self.token] = qa_pairs

            # 计算 分数
            self.f1_score = self.F1_Evaluator.calculate_f1_list(self.qa, answers)
            self.acc = self.F1_Evaluator.calculate_ACC(self.qa, answers)
            avg_token = self.llm.get_total_token() / self.F1_Evaluator.get_len()
            logger.info(f"total_token={self.llm.get_total_token()}")
            logger.info(
                f"self.token={self.token} | F1 Score={self.f1_score:.4f} | ACC Score={self.acc:.4f} | avg_token={avg_token}")

            self._save_results()
            logger.info("实验完成")

        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="ds", help="使用的模型名称")
    parser.add_argument("--dataset", type=str, default="liar", help="评估使用的数据集名称")
    parser.add_argument("--token", type=int, default=0, help="用于测试的 token 数量")
    parser.add_argument("--prompt", type=str, default="", help="用于测试的特殊提示")
    parser.add_argument("--template", type=str, default="GPQA_PROMPT", help="使用的prompt模板")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)
    try:
        config = ConfigLoader(args.config)
        runner = Prompt_Runner(config, args.model_name, args.dataset, token=args.token, prompt=args.prompt,
                               template=args.template)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.close()
    except Exception as e:
        logger.error(f"实验启动失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
