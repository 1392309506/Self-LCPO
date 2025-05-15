from __future__ import annotations

import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path
import random
from typing import List

from component.length_optimizer import TokenLengthOptimizer
from component.config_loader import ConfigLoader
from prompt.dataset_prompt import MATH_PROMPT,GPQA_PROMPT,WSC_PROMPT,BBH_PROMPT,STR_PROMPT,BOOLQ_PROMPT
from prompt.extract_prompt import EXTRACT_ANSWER_PROMPT
from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from component.evaluator import Evaluator
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("exp_lcpo")


class LCPO_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, is_truth : str,
                 sample_k: int = 0, n_steps: int = 5,
                 protect_token: int = 0, initial_tokens=None,
                 is_extract:str="true", protect_prompt:str=""):
        if initial_tokens is None:
            initial_tokens = list(range(100, 8000, 1000))
        self.config = config
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.qa = self.loadUtil.load_json(sample_k)

        self.Evaluator = Evaluator()
        self.qa_answers_by_ni = {}  # 存储每个 token 限制下的 QA 对
        self.llm = ChatLLM(
            api_type=self.model.get("api_type"),
            api_key=self.model.get("api_key"),
            base_url=self.model.get("base_url"),
            params=self.model.get("params"),
            name="exp_lcpo",
        )
        self.opt = TokenLengthOptimizer(token_bounds=(min(initial_tokens), max(initial_tokens)), config=config,
                                        model_name=model_name, llm=self.llm,
                                        qa=self.qa, is_truth=is_truth)
        self.max_concurrent_requests = 5  # 建议 5~15，根据模型和账户配额灵活设置
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.n_steps = n_steps
        self.sorted_tokens = []
        self.token_list = []
        self.all_tested_tokens = set()
        self.protect_token = protect_token
        self.protect_prompt = protect_prompt
        self.model_name = model_name
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
            name="extract_lcpo"
        )
        self.initial_tokens = initial_tokens
        self.is_extract = is_extract
        self.is_truth = is_truth

        self.cnt = 0
        self.results = []


    async def _save_results(self):
        """保存最优 prompt、预测结果、token 使用信息等到统一目录"""
        if not hasattr(self, 'sorted_tokens') or not self.sorted_tokens:
            logger.warning("没有排序结果，跳过保存。")
            return

        best_token = self.sorted_tokens[0]
        logger.info(f"🏆 最佳 token: {best_token}")

        logger.info(f"📥 使用最佳 token 执行预测")

        # 构造保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        random_code = hex(random.randint(0, 65535))[2:].upper()
        folder = Path("results") / f"{timestamp}_{random_code}"
        folder.mkdir(parents=True, exist_ok=True)

        # 构造 summary.json 内容
        total_token = self.llm.get_total_token()
        best_prompt = self.template.format(count=best_token, question=self.qa[0]["question"])
        summary_data = {
            "best_token": best_token,
            "total_token_usage": total_token,
            "best_prompt": best_prompt
        }

        summary_path = folder / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info(f"📝 已保存 summary.json 到 {summary_path}")

        logger.info(f"✅ 所有实验结果已保存到 {folder}")
    async def _execute_protect_prompt(self) -> List[str]:
        """并发执行提示"""
        prompt = self.protect_prompt

        tasks = [self._fetch_answer(item, prompt) for item in self.qa]
        results = await asyncio.gather(*tasks)
        return results

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

                if response is None or not hasattr(response, "content"):
                    logger.warning(f"❌ 模型响应为空，跳过该问题: {question[:40]}...")
                    return "None"
                if response.content is None:
                    logger.warning(f"❌ 模型响应为空，跳过该问题: {question[:40]}...")
                    return "None"

                answer = response.content
                # 需要人工标注：答案作为监督信号。否则过程作为监督信号
                if self.is_truth == "true" :
                    answer = LoadUtils.extract_content(answer, "answer")

                standard_answer = item.get("answer")
                # LLM 提取答案（修正）
                if self.is_extract == "true":
                    if answer == None:
                        judge = await self._extract(standard=standard_answer, personal=response.content)
                        if judge == 1 or judge == "1":
                            answer = standard_answer
                        elif answer == None:
                            answer = "None"
                logger.info(f"question: {question[:40]}已执行完毕")
                return answer
            except Exception as e:
                logger.error(f"⚠️ 模型调用失败，跳过问题：{question[:40]}... 错误：{str(e)}")
                return None

    async def _extract(self, standard: str, personal: str) -> str:
        prompt = EXTRACT_ANSWER_PROMPT.format(standard=standard, personal=personal)
        messages = [{"role": "user", "content": prompt}]
        response = await self.extract_llm.chat(messages)
        ranking = LoadUtils.extract_content(response.content, "judge")
        return ranking

    async def _warmup(self):
        """执行 warm-up 初始化阶段"""

        for n_i in self.initial_tokens:
            logger.info(f"开始训练的token数量为：{n_i}")
            answers = await self._execute_prompt(n_i)
            self.qa_answers_by_ni[n_i] = answers

        # 单独处理protect
        if self.protect_token != 0:
            logger.info(f"特殊处理：训练的token数量为：{self.protect_token}")
            protect_answers = await self._execute_protect_prompt()
            self.qa_answers_by_ni[self.protect_token] = protect_answers
            self.initial_tokens.append(self.protect_token)

        self.token_list = list(self.initial_tokens)
        self.all_tested_tokens = set(self.initial_tokens)
        logger.info("✅ warm-up 结束")

        init_qa_dict = {n_i: self.qa_answers_by_ni[n_i] for n_i in self.initial_tokens}
        ranked_indices = await self.opt.listwise(qa_dict=init_qa_dict)
        print(ranked_indices)
        self.sorted_tokens = [self.token_list[i] for i in ranked_indices]


    async def run(self):
        """主实验入口：包含 warm-up + 多轮优化"""
        try:
            await self._warmup()

            logger.info("🏁 模型训练结束")
            await self._save_results()
            logger.info("✅ 实验完成")
            logger.info("token = " + str(self.llm.get_total_token()))

        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    # config
    parser.add_argument('--config', type=str, default='../../config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="o3", help="Project name")
    # train
    parser.add_argument("--dataset", type=str, default="wsc", help="Project name")
    parser.add_argument("--sample_k", type=int, default=2, help="抽样的QA数量（0表示全部）")
    parser.add_argument("--n_steps", type=int, default=10, help="贝叶斯优化迭代轮次")
    parser.add_argument("--protect_token", type=int, default=0, help="特殊token花销")
    parser.add_argument("--protect_prompt", type=str, default="COT_PROMPT", help="特殊token模板")
    parser.add_argument("--is_truth", type=str, default="false", help="是否有人工标注")
    parser.add_argument("--is_extract", type=str, default="false", help="是否需要提取")
    # init_token_list
    parser.add_argument("--init_left", type=int, default=100, help="初试token_list边界左值")
    parser.add_argument("--init_right", type=int, default=8000, help="初试token_list边界右值")
    parser.add_argument("--init_step", type=int, default=1000, help="初试token_list边界步长")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)
    try:

        initial_tokens = list(range(args.init_left, args.init_right, args.init_step))
        config = ConfigLoader(args.config)
        runner = LCPO_Runner(config=config, model_name=args.model_name, dataset=args.dataset,
                             sample_k=args.sample_k, n_steps=args.n_steps, protect_token=args.protect_token,
                              initial_tokens=initial_tokens, is_truth=args.is_truth,
                             is_extract=args.is_extract)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.run_until_complete(loop.shutdown_asyncgens())  # 确保关闭 async generator
        loop.close()
    except Exception as e:
        logger.error(f"实验启动失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
