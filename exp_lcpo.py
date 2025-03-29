import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config_loader import ConfigLoader
from utils.load_utils import LoadUtils
from utils.general_llm_client import General_LLM
from f1_score import F1_Evaluator
from utils.logger_utils import LoggerUtil

logger = LoggerUtil.get_logger("exp_lcpo")


class LCPO_Runner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, sample_k: int = 0):
        self.config = config
        self.model_name = model_name
        self.dataset = dataset
        self.sample_k = sample_k
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.GeneralLLM = General_LLM(model=self.model)
        self.F1_Evaluator = F1_Evaluator(
            # model_url=config.models[model_name].get("base_url"),
            # dataset_name=args.dataset_name,
            # dataset_path=args.dataset_path,
            # api_key=args.api_key,
        )
        self.results = {}

    def _save_results(self):
        """保存实验结果"""
        if not hasattr(self, 'results') or not self.results:
            logger.warning("没有实验结果需要保存。")
            return

        results_path = Path("results/results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)
            logger.info(f"实验结果已保存至 {results_path}")
        except Exception as e:
            logger.error(f"保存实验结果失败: {e}")

    def visualize(self):
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

    def run(self):
        """执行实验流程"""
        exp_params = self.config.experiment  # 使用 experiment 属性

        try:
            prompt, requirements, qa, count_str = self.loadUtil.load_meta_data(self.sample_k)

            left = 0
            right = 1000

            while right - left > 20:
                ln = (left + right) / 3
                rn = ln * 2 - left
                prompt_ln = f"{prompt} Think for exactly {ln} tokens."
                prompt_rn = f"{prompt} Think for exactly {rn} tokens."

                answers = []
                for question, expected_answer in qa:
                    try:
                        # 拼接提示
                        full_prompt_ln = f"Prompt: {prompt_ln}\nQuestion: {question}\nAnswer:"
                        full_prompt_rn = f"Prompt: {prompt_rn}\nQuestion: {question}\nAnswer:"

                        # 调用模型生成回答
                        answer_ln = self.GeneralLLM.generate_response(full_prompt_ln)
                        answer_rn = self.GeneralLLM.generate_response(full_prompt_rn)

                        answers.append(answer)
                    except Exception as e:
                        logger.error(f"模型调用失败，问题：{question}，错误：{str(e)}")
                        answers.append({"question": question, "answer": "ERROR", "expected": expected_answer})

                f1_score = self.F1_Evaluator.calculate_f1_list(qa, answers)
                logger.info(f"n_i={n_i} | F1 Score={f1_score:.4f}")
                self.results[n_i] = f1_score

            logger.info("模型训练结束")
            self._save_results()
            self.visualize()
            logger.info("实验完成")
        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")


def parse_args():
    # # LLM parameter
    # parser.add_argument("--opt-model", type=str, default="claude-3-5-sonnet-20240620", help="Model for optimization")
    # parser.add_argument("--opt-temp", type=float, default=0.7, help="Temperature for optimization")
    # parser.add_argument("--eval-model", type=str, default="gpt-4o-mini", help="Model for evaluation")
    # parser.add_argument("--eval-temp", type=float, default=0.3, help="Temperature for evaluation")
    # parser.add_argument("--exec-model", type=str, default="gpt-4o-mini", help="Model for execution")
    # parser.add_argument("--exec-temp", type=float, default=0, help="Temperature for execution")

    # # PromptOptimizer parameter
    # parser.add_argument("--workspace", type=str, default="workspace", help="Path for optimized output")
    # parser.add_argument("--initial-round", type=int, default=1, help="Initial round number")
    # parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of rounds")
    # parser.add_argument("--template", type=str, default="Poem.yaml", help="Template file name")
    # parser.add_argument("--name", type=str, default="Poem", help="Project name")

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
        runner = LCPO_Runner(config, args.model_name, args.dataset, args.sample_k)
        runner.run()
    except Exception as e:
        logger.error(f"实验启动失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
