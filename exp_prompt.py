import argparse
import asyncio
import json
from pathlib import Path

from utils.load_utils import LoadUtils
from utils.logger_utils import LoggerUtil
from utils.prompt_utils import PromptUtils
from f1_score import F1_Evaluator

logger = LoggerUtil.get_logger("EvaluateSinglePrompt")

async def main(result_dir: str, dataset_yaml: str, sample_k: int = 0):
    # 初始化路径
    result_path = Path(result_dir)
    prompt_file = result_path / "best_prompt.txt"
    f1_output_file = result_path / "eval_result.json"
    prediction_output_file = result_path / "predictions.json"

    # 检查 prompt 是否存在
    if not prompt_file.exists():
        logger.error(f"找不到 prompt 文件: {prompt_file}")
        return

    # 读取 prompt
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    logger.info(f"✅ 成功加载 prompt: {prompt_file}")

    # 加载数据
    loader = LoadUtils(dataset_yaml)
    _, _, qa_list, _ = loader.load_meta_data(sample_k=sample_k)
    logger.info(f"✅ 成功加载 QA 数据集，共 {len(qa_list)} 条")

    # 初始化 F1 评估器
    evaluator = F1_Evaluator()
    predictions = []
    for item in qa_list:
        q = item["question"]
        a = await evaluator.query_llm_async(prompt, q)
        predictions.append({
            "question": q,
            "predicted": a,
            "ground_truth": item["answer"]
        })

    # 计算 F1 分数
    pred_texts = [p["predicted"] for p in predictions]
    avg_f1 = evaluator.calculate_f1_list(qa_list, pred_texts)
    correct_count = sum([1 for p in predictions if p["predicted"].strip().lower() == p["ground_truth"].strip().lower()])
    acc = correct_count / len(predictions)

    # 保存结果
    with f1_output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "f1": round(avg_f1, 4),
            "accuracy": round(acc, 4),
            "total": len(predictions),
        }, f, indent=2, ensure_ascii=False)

    with prediction_output_file.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 已保存评估结果至: {f1_output_file}")
    logger.info(f"📊 F1 Score: {avg_f1:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估单个 Prompt 在数据集上的表现")
    parser.add_argument("--result-dir", type=str, default="navigate.yaml", required=True, help="结果目录，如: results/20240406_0215_AB1D/")
    parser.add_argument("--dataset", type=str, default="navigate.yaml", help="YAML 数据集文件名")
    parser.add_argument("--sample-k", type=int, default=0, help="抽样的 QA 数量，0 表示使用全部")

    args = parser.parse_args()
    asyncio.run(main(args.result_dir, args.dataset, args.sample_k))
