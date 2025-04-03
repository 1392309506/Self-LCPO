import asyncio
import json
import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from config_loader import ConfigLoader
from prompt.execute_prompt import EXECUTE_PROMPT
from utils.load_utils import LoadUtils
from chat.chat_llm_openai import ChatLLM

from f1_score import F1_Evaluator
from utils.logger_utils import LoggerUtil
logger = LoggerUtil.get_logger("exp_llm")

class LLM_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, sample_k: int=0):
        self.config = config
        self.model_name = model_name
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.prompt, self.requirements, self.qa, self.count_str = self.loadUtil.load_meta_data(sample_k)
        self.F1_Evaluator = F1_Evaluator()
        self.results={}
        self.qa_answers_by_ni = {}  # 存储每个 token 限制下的 QA 对
        self.llm = ChatLLM(api_key=self.model.get("api_key"),base_url=self.model.get("base_url"))
        self.max_concurrent_requests = 200  # 建议 5~15，根据模型和账户配额灵活设置
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    def _visualize(self):
        """可视化实验结果"""
        # 检查是否有实验结果
        if not hasattr(self, 'results') or not self.results:
            logger.warning("没有实验结果可以可视化")
            return

        # 提取 n_i 值和对应的 F1 分数，并排序
        n_i_values = sorted(self.results.keys())
        f1_scores = [self.results[n_i] for n_i in n_i_values]

        # 绘制折线图
        plt.figure(figsize=(8, 6))
        plt.plot(n_i_values, f1_scores, marker='o', linestyle='-', color='b', label='F1 Score')
        plt.xlabel("n_i (Token 数量)")
        plt.ylabel("F1 Score")
        plt.title("模型性能对比")
        plt.legend()
        plt.grid(True)

        # 保存图表到文件
        output_path = Path("results/performance.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.show()

    def _save_results(self):
        """保存 F1 分数 和 QA 对"""
        if not hasattr(self, 'results') or not self.results:
            logger.warning("没有实验结果需要保存。")
            return

        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 保存 F1 分数
        results_path = results_dir / "results.json"
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)
            logger.info(f"F1 分数已保存至 {results_path}")
        except Exception as e:
            logger.error(f"保存 F1 分数失败: {e}")

        # 保存 QA 对
        qa_path = results_dir / "qa_answers.json"
        try:
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(self.qa_answers_by_ni, f, indent=2, ensure_ascii=False)
            logger.info(f"QA 对已保存至 {qa_path}")
        except Exception as e:
            logger.error(f"保存 QA 对失败: {e}")

    async def _execute_prompt(self,n:int)-> List[str]:
        """执行提示"""
        tasks = [self._fetch_answer(item.get("question"),n) for item in self.qa]
        results = await asyncio.gather(*tasks)
        return results

    async def _fetch_answer(self, question: str, n: int) -> str:
        """发送异步请求获取答案"""
        try:
            async with self._semaphore:
                prompt = EXECUTE_PROMPT.format(question=question, count=n)
                messages = [{"role": "user", "content": prompt}]
                response = await self.llm.chat(messages)
                answer = LoadUtils.extract_content(response.content, "answer")
                analysis = LoadUtils.extract_content(response.content, "analysis")
                # print(response.content)
                return answer
        except Exception as e:
            logger.error(f"模型调用失败，问题：{question}，错误：{str(e)}")
            return "ERROR"

    async def run(self):
        """执行实验流程"""
        exp_params = self.config.experiment  # 使用 experiment 属性

        try:
            for n_i in range(400, 4001, 150):
                logger.info(f"开始训练的token数量为：{n_i}")
                answers = await self._execute_prompt(n_i)
                # 存储每个 n_i 下的 QA 对
                qa_pairs = [
                    {"question": item.get("question"), "answer": answer}
                    for item, answer in zip(self.qa, answers)
                ]
                self.qa_answers_by_ni[n_i] = qa_pairs

                # 计算 F1 分数
                f1_score = self.F1_Evaluator.calculate_f1_list(self.qa, answers)
                logger.info(f"n_i={n_i} | F1 Score={f1_score:.4f}")
                self.results[n_i] = f1_score

            logger.info("模型训练结束")
            self._save_results()
            self._visualize()
            logger.info("实验完成")
        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Project name")
    parser.add_argument("--dataset", type=str, default="Navigate", help="Project name")
    parser.add_argument("--sample_k", type=int, default=0, help="抽样的QA数量（0表示全部）")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    try:
        config = ConfigLoader(args.config)
        runner = LLM_Runner(config, args.model_name, args.dataset, args.sample_k)
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
