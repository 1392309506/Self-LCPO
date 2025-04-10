from __future__ import annotations

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
import random
from typing import List

from component.length_optimizer import TokenLengthOptimizer
from config_loader import ConfigLoader
from prompt.execute_prompt import EXECUTE_PROMPT
from prompt.dataset_prompt import MATH_PROMPT, GPQA_PROMPT
from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from f1_score import F1_Evaluator
from utils.logger_utils import LoggerUtil
logger = LoggerUtil.get_logger("exp_lcpo")

class LCPO_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, sample_k: int=0, n_steps: int=5):
        self.config = config
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.qa = self.loadUtil.load_json(sample_k)
        self.F1_Evaluator = F1_Evaluator()
        self.qa_answers_by_ni = {}  # 存储每个 token 限制下的 QA 对
        self.llm = ChatLLM(
            api_type=self.model.get("api_type"),
            api_key=self.model.get("api_key"),
            base_url=self.model.get("base_url"),
            params=self.model.get("params"),
        )
        self.opt = TokenLengthOptimizer(token_bounds=(100, 4000), config=config, model_name=model_name,llm=self.llm)
        self.max_concurrent_requests = 5  # 建议 5~15，根据模型和账户配额灵活设置
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.n_steps = n_steps
        self.sorted_tokens = []
        self.token_list = []

    async def _save_results(self):
        """保存最优 Prompt 与对应的 F1 分数到新文件夹"""
        if not hasattr(self, 'sorted_tokens') or not self.sorted_tokens:
            logger.warning("没有实验结果需要保存。")
            return
        best_token = self.sorted_tokens[0]
        answers = await self._execute_prompt(best_token)
        # 存储每个 n_i 下的 QA 对
        qa_pairs = [
            {"question": item.get("question"), "answer": answer}
            for item, answer in zip(self.qa, answers)
        ]
        self.qa_answers_by_ni[best_token] = qa_pairs
        qa = self.loadUtil.load_json(sample_k=0)
        f1_score = self.F1_Evaluator.calculate_f1_list(qa, answers)
        # 获取最优 token 与其 F1 分数
        logger.info(f"🏆 最佳 token: {best_token}, F1 分数: {f1_score:.4f}")

        # 生成时间戳+随机码文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        random_code = hex(random.randint(0, 65535))[2:].upper()
        folder = Path("results") / f"{timestamp}_{random_code}"
        folder.mkdir(parents=True, exist_ok=True)

        # 生成 best_prompt
        best_prompt = MATH_PROMPT.format(
            question=self.qa[0]["question"],
            count=best_token
        )

        # 保存 Prompt
        prompt_path = folder / "best_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(best_prompt)

        # 保存 F1 分数
        f1_path = folder / "f1_score.json"
        with open(f1_path, "w", encoding="utf-8") as f:
            json.dump({"best_token": best_token, "f1": f1_score}, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 最优结果已保存至: {folder}")

    async def _execute_prompt(self,n:int)-> List[str]:
        """并发执行提示"""
        tasks = [self._fetch_answer(item.get("question"),n) for item in self.qa]
        results = await asyncio.gather(*tasks)
        return results

    async def _fetch_answer(self, question: str, n: int) -> str | None:
        """发送异步请求获取答案（失败返回 None，不污染 F1 数据）"""
        async with self._semaphore:
            try:
                # prompt = EXECUTE_PROMPT.format(question=question, count=n)
                prompt = GPQA_PROMPT.format(question=question, count=n)
                messages = [{"role": "user", "content": prompt}]
                response = await self.llm.chat(messages)

                if response is None or not hasattr(response, "content"):
                    logger.warning(f"❌ 模型响应为空，跳过该问题: {question[:40]}...")
                    return None

                answer = LoadUtils.extract_content(response.content, "answer")
                return answer

            except Exception as e:
                logger.error(f"⚠️ 模型调用失败，跳过问题：{question[:40]}... 错误：{str(e)}")
                return None

    async def _warmup(self):
        """执行 warm-up 初始化阶段"""
        # initial_tokens = self.config.experiment["n_i_values"]
        initial_tokens = list(range(100, 4001, 400))

        self.token_list = list(initial_tokens)
        self.all_tested_tokens = set(initial_tokens)

        for n_i in initial_tokens:
            logger.info(f"开始训练的token数量为：{n_i}")
            answers = await self._execute_prompt(n_i)
            qa_pairs = [
                {"question": item.get("question"), "answer": answer}
                for item, answer in zip(self.qa, answers)
            ]
            self.qa_answers_by_ni[n_i] = qa_pairs

        logger.info("✅ warm-up 结束")

        init_qa_dict = {n_i: self.qa_answers_by_ni[n_i] for n_i in initial_tokens}
        ranked_indices = await self.opt.listwise(init_qa_dict)
        self.opt.update_listwise(initial_tokens, ranked_indices)

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
            ranked_indices = await self.opt.listwise(current_qa_dict)
            self.sorted_tokens = [self.token_list[i] for i in ranked_indices]

            n_delete = max(1, len(self.token_list) // 3)
            tokens_to_remove = self.sorted_tokens[-n_delete:]

            for t in tokens_to_remove:
                if t != best_token:
                    self.token_list.remove(t)
                    logger.info(f"🗑️  移除 token: {t}")
                else:
                    logger.info(f"🔒 保留 best_token: {t}（禁止移除）")

            # 生成新 token，exclude 中不要包含 best_token
            n_add = max(1, len(self.token_list) // 2)
            exclude_set = (set(self.qa_answers_by_ni.keys()) | set(self.token_list)) - {best_token}
            new_tokens = self.opt.suggest_next(
                n_suggestions=n_add,
                anchor_token=best_token,
                exclude=exclude_set,
            )

            logger.info(f"➕ 新增 token: {new_tokens}")

            # 加一轮判断，如果 token 已存在或已处理过，则不添加
            seen = set(self.all_tested_tokens)  # 已处理过的 token
            filtered_new_tokens = []
            for token in new_tokens:
                # 排除空值（避免 NoneType 错误）和重复项
                if token is not None and token not in seen:
                    seen.add(token)  # 标记当前批次的 token 已添加
                    filtered_new_tokens.append(token)
            self.token_list.extend(filtered_new_tokens)

            # 执行新 token 的生成（仅处理过滤后的 token）
            for n_i in filtered_new_tokens:
                logger.info(f"生成 token={n_i} 的回答中")
                answers = await self._execute_prompt(n_i)
                qa_pairs = [
                    {"question": item.get("question"), "answer": answer}
                    for item, answer in zip(self.qa, answers)
                ]
                self.qa_answers_by_ni[n_i] = qa_pairs
                self.all_tested_tokens.add(n_i)  # 标记为已处理

            # 只在所有 token 生成与处理完毕后一次性更新模型
            current_qa_dict = {n: self.qa_answers_by_ni[n] for n in self.token_list}
            ranked_indices = await self.opt.listwise(current_qa_dict)
            self.sorted_tokens = [self.token_list[i] for i in ranked_indices]
            logger.info(f"📊 排序 index: {ranked_indices}")
            logger.info(f"➡️  token 排名: {self.sorted_tokens}")

            self.opt.update_listwise(self.token_list, ranked_indices)
            logger.info("✅ 模型已更新")


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

        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="gpt", help="Project name")
    parser.add_argument("--dataset", type=str, default="math", help="Project name")
    parser.add_argument("--sample_k", type=int, default=3, help="抽样的QA数量（0表示全部）")
    parser.add_argument("--n_steps", type=int, default=5, help="贝叶斯优化迭代轮次")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    try:
        config = ConfigLoader(args.config)
        runner = LCPO_Runner(config, args.model_name, args.dataset, args.sample_k, args.n_steps)
        # asyncio.run(runner.run())

        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.run_until_complete(loop.shutdown_asyncgens())  # 确保关闭 async generator
        loop.close()
    except Exception as e:
        logger.error(f"实验启动失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
