import json
import argparse
from pathlib import Path
import matplotlib as plt
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config_loader import ConfigLoader
from utils.load_utils import LoadUtils
from utils.general_llm_client import General_LLM
from f1_score import F1_Evaluator
from utils.logger_utils import LoggerUtil
logger = LoggerUtil.get_logger("exp_llm")

class ExperimentRunner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str, sample_k: int=0):
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
        self.results=[]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_model(self, model_cfg: dict, prompt: str):
        """执行模型调用"""
        client = OpenAI(
            api_key=model_cfg["api_key"],  # 直接使用 model_cfg 中的 api_key
            base_url=model_cfg.get("base_url")  # 使用 model_cfg 中的 base_url
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **model_cfg["params"]
        )
        print(f"使用了 {response.usage.completion_tokens} tokens")
        return response.choices[0].message.content, response.usage.completion_tokens

    def calculate_f1(self, predicted: str, expected: str) -> float:
        """计算F1分数"""
        logger.info("开始计算F1分数：")
        # 示例实现
        predicted_tokens = set(predicted.split())
        expected_tokens = set(expected.split())
        tp = len(predicted_tokens & expected_tokens)
        fp = len(predicted_tokens - expected_tokens)
        fn = len(expected_tokens - predicted_tokens)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    def evaluate_model(self, model_cfg: dict, data: list, n_i: int, prompt: str) -> float:
        """评估模型性能"""
        total_f1 = 0
        for item in data:
            response, _ = self.call_model(model_cfg, prompt)
            f1 = self.calculate_f1(response, item["expected_answer"])
            total_f1 += f1
        return total_f1 / len(data[:n_i] if n_i is not None else len(data))

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
            for n_i in exp_params["n_i_values"]:
                modified_prompt = f"{prompt} Think for exactly {n_i} tokens."
                answers = []
                logger.info(f"开始训练的token数量为："+str(n_i))

                for question, expected_answer in qa:
                    try:
                        # 拼接提示
                        full_prompt = f"Prompt: {modified_prompt}\nQuestion: {question}\nAnswer:"

                        # 调用模型生成回答
                        answer = self.GeneralLLM.generate_response(full_prompt)
                        answers.append(answer)
                    except Exception as e:
                        logger.error(f"模型调用失败，问题：{question}，错误：{str(e)}")
                        answers.append({"question": question, "answer": "ERROR", "expected": expected_answer})

                f1_score=self.F1_Evaluator.calculate_f1_list(qa, answers)
                logger.info(f"n_i={n_i} | F1 Score={f1_score:.4f}")
                self.results[n_i] = f1_score

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
        runner = ExperimentRunner(config, args.model_name, args.dataset, args.sample_k)
        runner.run()
    except Exception as e:
        logger.error(f"实验启动失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
