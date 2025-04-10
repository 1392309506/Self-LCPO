from __future__ import annotations

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
from random import randint
from typing import List

from config_loader import ConfigLoader
from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from f1_score import F1_Evaluator
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("exp_llm")


class IO_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str):
        self.config = config
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.qa = self.loadUtil.load_json(0)
        self.F1_Evaluator = F1_Evaluator()
        self.qa_pairs = {}  # 存储每个 token 限制下的 QA 对
        self.llm = ChatLLM(
            api_type=self.model.get("api_type"),
            api_key=self.model.get("api_key"),
            base_url=self.model.get("base_url"),
            params=self.model.get("params"),
        )
        self.max_concurrent_requests = 5
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.result = 0
        self.cnt = 0

    def _save_results(self):
        """保存 F1 分数 和 QA 对"""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳+随机码文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        random_code = hex(randint(0, 65535))[2:].upper()
        folder = Path("results") / f"{timestamp}_{random_code}"
        folder.mkdir(parents=True, exist_ok=True)

        # 保存 F1 分数
        results_path = folder / "results.json"
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"Python Project": "IO",
                            "f1_score": self.result,
                            "dataset": self.dataset}, f, indent=4, ensure_ascii=False)
            logger.info(f"F1 分数已保存至 {results_path}")
        except Exception as e:
            logger.error(f"保存 F1 分数失败: {e}")

        # 保存 QA 对
        qa_path = folder / "qa_answers.json"
        try:
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
            logger.info(f"QA 对已保存至 {qa_path}")
        except Exception as e:
            logger.error(f"保存 QA 对失败: {e}")

    async def _execute_prompt(self)-> List[str]:
        """并发执行提示"""
        tasks = [self._fetch_answer(item.get("question")) for item in self.qa]
        results = await asyncio.gather(*tasks)
        return results

    async def _fetch_answer(self, question: str) -> str | None:
        """发送异步请求获取答案（失败返回 None，不污染 F1 数据）"""
        async with self._semaphore:
            try:
                messages = [{"role": "user", "content": question}]
                response = await self.llm.chat(messages)

                if response is None or not hasattr(response, "content"):
                    logger.warning(f"❌ 模型响应为空，响应为: {response}\n跳过该问题: {question[:40]}...")
                    return None

                answer = response.content
                self.cnt = self.cnt + 1
                print(str(self.cnt) + ": " + answer)
                return answer

            except Exception as e:
                logger.error(f"⚠️ 模型调用失败，跳过问题：{question[:40]}... 错误：{str(e)}")
                return None

    async def run(self):
        """执行实验流程"""
        try:
            answers = await self._execute_prompt()

            # 存储 QA 对
            self.qa_pairs = [
                {"question": item.get("question"), "answer": answer}
                for item, answer in zip(self.qa, answers)
            ]

            # 计算 F1 分数
            f1_score = self.F1_Evaluator.calculate_f1_list(self.qa, answers)
            logger.info(f"F1 Score={f1_score:.4f}")
            self.result = f1_score

            self._save_results()
            logger.info("实验完成")

        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="gpt", help="使用的模型名称")
    parser.add_argument("--dataset", type=str, default="math", help="评估使用的数据集名称")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    try:
        config = ConfigLoader(args.config)
        runner = IO_Runner(config, args.model_name, args.dataset)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    except Exception as e:
        logger.error(f"实验启动失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
