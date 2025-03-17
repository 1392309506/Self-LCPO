import json
import logging
import argparse
from pathlib import Path

import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config_loader import ConfigLoader

class ExperimentRunner:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.results = {m["name"]: {} for m in config.models}

    def _load_dataset(self, dataset_path: str):
        """安全加载数据集"""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"数据集文件 {path} 不存在")

        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

        # try:
        #     with open(path, 'r') as f:
        #         return json.load(f)
        # except json.JSONDecodeError:
        #     raise ValueError(f"数据集文件 {path} 格式错误")

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

    def extract_answer(self, response: str) -> str:
        """从模型响应中提取答案"""
        # 示例实现
        return response.strip()

    def calculate_f1(self, predicted: str, expected: str) -> float:
        """计算F1分数"""
        logging.info("开始计算F1分数：")
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
        results_path = Path("results/results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        logging.info(f"实验结果已保存至 {results_path}")

    def visualize(self):
        """可视化实验结果"""
        import matplotlib.pyplot as plt

        for model_name, results in self.results.items():
            n_i_values = list(results.keys())
            f1_scores = list(results.values())
            plt.plot(n_i_values, f1_scores, label=model_name)

        plt.xlabel("n_i")
        plt.ylabel("F1 Score")
        plt.title("模型性能对比")
        plt.legend()
        plt.savefig("results/performance.png")
        plt.show()

    def run(self, dataset_name: str):
        """执行实验流程"""
        exp_params = self.config.experiment  # 使用 experiment 属性

        for model_cfg in self.config.models:
            logging.info(f"\n开始测试模型：{model_cfg['name']}==================")
            logging.info(f"模型参数：{model_cfg}")

        print(self.config.datasets.items())
        '''
        多一个参数path。
        数据集处理可能有问题？模型接口调用可能有问题？
        
        
        '''
        # try:
        #     data = self._load_dataset(path).get("qa", [
        #     ])
        #     logging.info(f"加载数据集 {dataset_name}，共 {len(data)} 条数据")
        #
        #     for n_i in exp_params["n_i_values"]:
        #         prompt = ("Please think step by step."
        #                   "Ensure the response concludes with the answer in the XML format:) "
        #                   "<answer>[Yes or No]</answer>."
        #                   f"Think for {n_i} tokens. ")
        #         f1 = self.evaluate_model(model_cfg, data, n_i, prompt)
        #         print(f"f1= {f1} ")
        #         self.results[model_cfg["name"]][n_i] = f1
        #         logging.info(f"n_i={n_i or '无限制'} | F1={f1:.4f}")
        #
        # except Exception as e:
        #     logging.error(f"数据集 {dataset_name} 处理失败: {str(e)}")

        self._save_results()
        self.visualize()
        logging.info("实验完成")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    args = parser.parse_args()

    try:
        config = ConfigLoader(args.config)
        runner = ExperimentRunner(config)
        runner.run(dataset_name="gpt-3.5-turbo")
    except Exception as e:
        logging.error(f"实验启动失败: {str(e)}")
        exit(1)
