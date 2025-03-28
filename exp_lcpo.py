import json
import logging
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
logger = LoggerUtil.get_logger("exp_lcpo")

class ExperimentRunner:
    def __init__(self, config: ConfigLoader, model_name: str, dataset: str):
        self.config = config
        self.model_name = model_name
        self.dataset = dataset
        self.model = config.models[model_name]
        self.loadUtil = LoadUtils(file_name=config.datasets[dataset])
        self.GeneralLLM = General_LLM(model=self.model)
        self.F1_Evaluator = F1_Evaluator(
            # model_url=config.models[model_name].get("base_url"),
            # dataset_name=args.dataset_name,
            # dataset_path=args.dataset_path,
            # api_key=args.api_key,
        )

        def run(self):
            """执行实验流程"""
            exp_params = self.config.experiment  # 使用 experiment 属性
            results = {}

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

    parser = argparse.ArgumentParser(description='大模型长度控制提示优化实验')
    parser.add_argument('--config', type=str, default='config/config_llm.yaml',
                        help='配置文件路径（默认：config/config_llm.yaml）')
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Project name")
    parser.add_argument("--dataset", type=str, default="Navigate", help="Project name")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    try:
        config = ConfigLoader(args.config)
        runner = ExperimentRunner(config,args.model_name,args.dataset)
        runner.run()
    except Exception as e:
        logger.error(f"实验启动失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
