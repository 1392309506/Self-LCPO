from __future__ import annotations

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
import random
from typing import List

from component.length_optimizer import TokenLengthOptimizer
from component.config_loader import ConfigLoader
from prompt.extract_prompt import EXTRACT_ANSWER_PROMPT
from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from component.evaluator import Evaluator
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("exp_lcpo_benchmark")


class LCPO_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, is_truth : str,
                 n_steps: int = 5, protect_token: int = 0, template: str = "GPQA_PROMPT",
                 initial_tokens=list(range(1000, 5001, 500)), is_extract:str="true", protect_prompt:str=""):
        self.config = config
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.qa_all = self.loadUtil.load_json(0)
        split_point = max(1, len(self.qa_all) // 10)

        self.qa = self.qa_all[:split_point]  # 前 10% 作为训练集
        self.qa_test = self.qa_all[split_point:]  # 后 90% 作为测试集

        self.Evaluator = Evaluator()
        self.qa_answers_by_ni = {}  # 存储每个 token 限制下的 QA 对
        self.llm = ChatLLM(
            api_type=self.model.get("api_type"),
            api_key=self.model.get("api_key"),
            base_url=self.model.get("base_url"),
            params=self.model.get("params"),
            name="exp_lcpo",
        )
        self.opt = TokenLengthOptimizer(token_bounds=(min(initial_tokens), max(initial_tokens)), config=config, model_name=model_name, llm=self.llm,
                                        qa=self.qa,is_truth=is_truth)
        self.max_concurrent_requests = 5  # 建议 5~15，根据模型和账户配额灵活设置
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.n_steps = n_steps
        self.sorted_tokens = []
        self.token_list = []
        self.all_tested_tokens = set()
        self.protect_token = protect_token
        self.protect_prompt = protect_prompt
        self.model_name = model_name
        self.template = template

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

    def benchmark_listwise(self, qa_dict: dict[int, list[dict]]) -> list[int]:
        """
        使用 ACC 作为评价指标，对 token 数进行 listwise 排序（替代 LLM 推理方式）

        参数:
            qa_dict: dict[token] = list[{'question':..., 'answer':..., 'pred':...}]
        返回:
            list[int]: 排序后的 token 数值（从高 ACC 到低）
        """
        token_list = list(qa_dict.keys())
        acc_list = []

        for token in token_list:
            qa_pairs = qa_dict[token]

            correct = 0
            total = 0
            for item in qa_pairs:
                pred = item.get("pred", "").strip()
                ans = item.get("answer", "").strip()
                if pred != "None" and ans != "None":
                    total += 1
                    if pred == ans:
                        correct += 1

            acc = correct / total if total > 0 else 0.0
            logger.info("Token = "+str(token)+" | ACC = "+str(acc))
            acc_list.append(acc)
        # 返回 token 数值排序（从高到低）
        return [token_list[i] for i in sorted(range(len(acc_list)), key=lambda i: -acc_list[i])]
    async def _warmup(self):
        """
        Warm-up 阶段：采集初始 token 数的回答 + 保存预测 + 基于 benchmark 的 listwise 排序
        """
        logger.info("🚀 开始 warm-up 初始化阶段")

        for n_i in self.initial_tokens:
            logger.info(f"📏 执行初始 token = {n_i} 的 prompt 请求")
            answers = await self._execute_prompt(n_i)  # 每个 answer 应包含 pred 和 answer 字段
            self.qa_answers_by_ni[n_i] = answers

        if self.protect_token != 0:
            logger.info(f"🔐 特殊处理：protect token = {self.protect_token}")
            protect_answers = await self._execute_protect_prompt()
            self.qa_answers_by_ni[self.protect_token] = protect_answers
            self.initial_tokens.append(self.protect_token)

        self.token_list = list(self.initial_tokens)
        self.all_tested_tokens = set(self.initial_tokens)
        logger.info("✅ Warm-up 完成，开始基于 benchmark 进行排序")

        # 构建 QA dict：{token: list[qa dict]}
        init_qa_dict = {n_i: self.qa_answers_by_ni[n_i] for n_i in self.initial_tokens}
        ranked_tokens = self.benchmark_listwise(init_qa_dict)
        ranked_indices = [self.initial_tokens.index(t) for t in ranked_tokens]
        self.opt.update_listwise(self.initial_tokens, ranked_indices)

        logger.info("✅ 初始化排序完成")

    async def _iterative_optimization(self):
        """执行多轮优化迭代（滑动窗口控制 token 集）"""
        n_steps = self.n_steps

        for step in range(n_steps):
            logger.info(f"\n[Step {step + 1}] 当前候选池: {self.token_list}")
            best_token = 0

            try:
                best_token = self.opt.get_best_token()
                logger.info(f"🎯 当前 GP 模型预测最优 token 可能为: {best_token}")
            except Exception:
                logger.warning("⚠️ 尚无可用的 GP 最优 token，使用默认值 0")

            # 排序当前池并删除最差的若干个（保留 best_token）
            current_qa_dict = {n: self.qa_answers_by_ni[n] for n in self.token_list}
            ranked_tokens = self.benchmark_listwise(current_qa_dict)
            ranked_indices = [self.token_list.index(t) for t in ranked_tokens]
            self.sorted_tokens = [self.token_list[i] for i in ranked_indices]

            n_change = max(1, len(self.token_list) // 3)
            tokens_to_remove = self.sorted_tokens[-n_change:]

            for t in tokens_to_remove:
                if t != best_token and t != self.protect_token:
                    self.token_list.remove(t)
                    logger.info(f"🗑️  移除 token: {t}")
                else:
                    logger.info(f"🔒 保留 best_token: {t}（禁止移除）")
                    n_change -= 1

            # 生成新 token，exclude 中不要包含 best_token
            exclude_set = (set(self.qa_answers_by_ni.keys()) | set(self.token_list)) - {best_token}
            new_tokens = self.opt.suggest_next(
                n_suggestions=max(n_change, 1),
                anchor_token=best_token,
                exclude=exclude_set,
            )

            logger.info(f"新增 token: {new_tokens}")

            # 过滤：排除空值（避免 NoneType 错误）和重复项
            filtered_new_tokens = [
                token for token in new_tokens
                if token is not None and token not in self.all_tested_tokens
            ]
            self.token_list.extend(filtered_new_tokens)
            logger.info(f"➕ 过滤后新增 : {filtered_new_tokens}")

            # 执行新 token 的生成（仅处理过滤后的 token）
            for n_i in filtered_new_tokens:
                logger.info(f"生成 token={n_i} 的回答中")
                answers = await self._execute_prompt(n_i)
                self.qa_answers_by_ni[n_i] = answers
                self.all_tested_tokens.add(n_i)  # 标记为已处理

            # 只在所有 token 生成与处理完毕后一次性更新模型
            current_qa_dict = {n: self.qa_answers_by_ni[n] for n in self.token_list}
            ranked_tokens = self.benchmark_listwise(current_qa_dict)
            ranked_indices = [self.token_list.index(t) for t in ranked_tokens]
            self.sorted_tokens = [self.token_list[i] for i in ranked_indices]
            logger.info(f"📊 排序 index: {ranked_indices}")
            logger.info(f"➡️  token 排名: {self.sorted_tokens}")

            self.opt.update_listwise(self.token_list, ranked_indices)
            logger.info("✅ 模型已更新")

            if len(self.token_list) < 4:
                logger.info(f"✅ token_list 长度过短，提前终止迭代 (当前为 {len(self.token_list)})")
                break

    async def run(self):
        """主实验入口：包含 warm-up + 多轮优化"""
        try:
            await self._warmup()
            print("🔎 token_history:", self.opt.token_history)
            print("🔎 comparisons:", self.opt.comparisons)
            logger.info("🚀 进入贝叶斯优化迭代阶段")
            await self._iterative_optimization()

            logger.info("🏁 模型训练结束")
            await self._save_results()
            logger.info("✅ 实验完成")
            logger.info("token = "+str(self.llm.get_total_token()))

        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    # config
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="o3", help="Project name")
    # train
    parser.add_argument("--dataset", type=str, default="gpqa", help="Project name")
    parser.add_argument("--n_steps", type=int, default=6, help="贝叶斯优化迭代轮次")
    parser.add_argument("--protect_token", type=int, default=0, help="特殊token花销")
    parser.add_argument("--protect_prompt", type=str, default="COT_PROMPT", help="特殊token模板")
    parser.add_argument("--template", type=str, default="GPQA_PROMPT", help="使用的prompt模板")
    parser.add_argument("--is_truth", type=str, default="true", help="是否有人工标注")
    parser.add_argument("--is_extract", type=str, default="true", help="是否需要提取")
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
                             n_steps=args.n_steps, protect_token=args.protect_token,
                             template=args.template, initial_tokens=initial_tokens, is_truth=args.is_truth,
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
