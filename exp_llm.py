from __future__ import annotations

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from prompt.extract_prompt import EXTRACT_ANSWER_PROMPT
from prompt.dataset_prompt import MATH_PROMPT,GPQA_PROMPT,WSC_PROMPT,BBH_PROMPT,STR_PROMPT,BOOLQ_PROMPT

from component.config_loader import ConfigLoader
from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from component.evaluator import Evaluator

from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("exp_llm")


class IO_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str):
        self.config = config
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.qa = self.loadUtil.load_json(100)
        self.F1_Evaluator = Evaluator()
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
        self.prompts = {
            "math": MATH_PROMPT,
            "gpqa": GPQA_PROMPT,
            "wsc": WSC_PROMPT,
            "bbh": BBH_PROMPT,
            "str": STR_PROMPT,
            "boolq": BOOLQ_PROMPT
        }
        self.template = self.prompts[dataset]

        extract_model = config.models["gpt"]
        self.extract_llm = ChatLLM(
            api_type=extract_model.get("api_type"),
            api_key=extract_model.get("api_key"),
            base_url=extract_model.get("base_url"),
            params=extract_model.get("params"),
            name="extract_llm"
        )
        self.acc_list={}
        self.result = []

    import matplotlib.pyplot as plt

    def _save_results(self):
        """保存 ACC 分数，并绘制结果图"""
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳+随机码文件夹名
        timestamp = datetime.now().strftime("%Y%m%d")
        folder = results_dir / f"{timestamp}_LLM_{self.dataset}"
        folder.mkdir(parents=True, exist_ok=True)

        # 提取 acc_list 中的 ACC 值（acc_list[token] = qa_pairs）
        acc_result = {}
        for token, qa_list in self.acc_list.items():
            acc = self.F1_Evaluator.calculate_ACC(self.qa, [qa["answer"] for qa in qa_list])
            acc_result[token] = acc

        # 保存 JSON
        results_path = folder / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(acc_result, f, indent=2, ensure_ascii=False)

        # 绘制 ACC 曲线图
        tokens = sorted(acc_result.keys())
        acc_values = [acc_result[t] for t in tokens]

        plt.figure()
        plt.plot(tokens, acc_values, marker='o')
        plt.title(f"ACC vs Token Count ({self.dataset})")
        plt.xlabel("Token Count")
        plt.ylabel("Accuracy (ACC)")
        plt.grid(True)
        plt.tight_layout()

        fig_path = folder / "acc_curve.png"
        plt.savefig(fig_path)
        plt.close()

        logger.info(f"结果和图表已保存至：{folder}")

    async def _execute_prompt(self, n: int) -> List[str]:
        """并发执行提示"""
        prompt = self.template.format(count=n)
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
                # print(response.content)
                standard_answer = item.get("answer")
                # LLM 提取答案（修正）
                if answer != standard_answer:
                    answer = await self._extract(standard=standard_answer, personal=response.content, question=question)
                if answer != standard_answer:
                    print(content)
                logger.info(str(self.cnt) + ": " + answer + " | " + standard_answer)
                self.cnt += 1
                return answer
            except Exception as e:
                logger.error(f"⚠️ 模型调用失败，跳过问题：{question[:40]}... 错误：{str(e)}")
                return None

    async def _extract(self, standard: str, personal: str, question: str) -> str:
        prompt = EXTRACT_ANSWER_PROMPT.format(standard=standard, personal=personal)
        messages = [{"role": "user", "content": prompt}]
        response = await self.extract_llm.chat(messages)
        personal = LoadUtils.extract_content(response.content, "answer")

        judge = LoadUtils.extract_content(response.content, "judge")
        if judge == "1":
            personal = standard
        else:
            personal = "None"
        return personal

    async def run(self):
        """执行实验流程"""
        try:
            init_tokenlist = list(range(1300,2701,400))
            for token in init_tokenlist:
                self.llm.token2zero()
                logger.info(f"开始训练的token数量为：{token}")
                answers = await self._execute_prompt(token)

                # 存储 QA 对
                qa_pairs = [
                    {"question": item.get("question"), "answer": answer}
                    for item, answer in zip(self.qa, answers)
                ]
                self.acc_list[token] = qa_pairs  # ✅ 关键：存入 acc_list

                f1_score = self.F1_Evaluator.calculate_f1_list(self.qa, answers)
                acc = self.F1_Evaluator.calculate_ACC(self.qa, answers)
                total_token = self.llm.get_total_token()
                prompt_token = self.llm.get_prompt_token()
                completion_token = self.llm.get_completion_token()
                avg_token = self.llm.get_total_token() / self.F1_Evaluator.get_len()
                result = {
                    "suggest_token": token,
                    "acc": acc,
                    "total_token": total_token,
                    "prompt_token": prompt_token,
                    "completion_token": completion_token,
                }
                self.result.append(result)
                print(result)

                logger.info(f"total_token={self.llm.get_total_token()}")
                logger.info(
                    f"self.token={token} | F1 Score={f1_score:.4f} | ACC Score={acc:.4f} | avg_token={avg_token}")

            self._save_results()
            logger.info("实验完成")

        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="o3", help="使用的模型名称")
    parser.add_argument("--dataset", type=str, default="gpqa", help="评估使用的数据集名称")
    parser.add_argument("--prompt", type=str, default="", help="用于测试的特殊提示")
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
