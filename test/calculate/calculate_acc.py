import json
from pathlib import Path


def load_data(json_file):
    """从JSON文件加载数据"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {json_file} 不存在")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file} 不是有效的JSON格式")
        return None


def calculate_acc(data):
    """
    计算准确率(ACC)
    参数:
        data: 包含问题、标准答案和模型回答的字典列表
    返回:
        acc: 准确率(0-1)
        correct_count: 正确回答数
        total_count: 总问题数
        details: 详细结果列表
    """
    if not data or not isinstance(data, list):
        return 0, 0, 0, []

    details = []
    correct_count = 0

    for item in data:
        is_correct = item.get('answer', '').strip().upper() == item.get('standard_answer', '').strip().upper()
        details.append({
            'question': item.get('question', '')[:50] + '...' if 'question' in item else 'N/A',
            'model_answer': item.get('answer', 'N/A'),
            'standard_answer': item.get('standard_answer', 'N/A'),
            'is_correct': is_correct,
            'tokens': item.get('token_consumed', 0)
        })
        if is_correct:
            correct_count += 1

    total_count = len(data)
    acc = correct_count / total_count if total_count > 0 else 0
    return acc, correct_count, total_count, details


def generate_report(acc, correct, total, details, output_file=None):
    """生成分析报告"""
    report = [
        f"分析结果：",
        f"总问题数: {total}",
        f"正确回答: {correct}",
        f"错误回答: {total - correct}",
        f"准确率(ACC): {acc:.4f} ({acc * 100:.2f}%)",
        "\n详细结果:"
    ]

    # 添加正确/错误示例
    correct_samples = [d for d in details if d['is_correct']]
    wrong_samples = [d for d in details if not d['is_correct']]

    report.append("\n正确回答示例 (前3个):")
    for sample in correct_samples[:3]:
        report.append(f"- 问题: {sample['question']}")
        report.append(f"  模型: {sample['model_answer']} | 标准: {sample['standard_answer']}")

    report.append("\n错误回答示例 (前3个):")
    for sample in wrong_samples[:3]:
        report.append(f"- 问题: {sample['question']}")
        report.append(f"  模型: {sample['model_answer']} | 标准: {sample['standard_answer']}")

    # 计算平均token消耗
    avg_tokens = sum(d['tokens'] for d in details) / total if total > 0 else 0
    report.append(f"\n平均token消耗: {avg_tokens:.1f}")

    full_report = '\n'.join(report)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"报告已保存到 {output_file}")

    return full_report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='计算模型回答准确率(ACC)')
    parser.add_argument('-input_file', default='../../results/exp_prompt/20250514/cot_o3_gpqa/results_data.json', help='输入的JSON文件路径')
    parser.add_argument('-output', default='result.json', help='输出报告文件路径(可选)')
    args = parser.parse_args()

    # 加载数据
    data = load_data(args.input_file)
    if not data:
        return

    # 计算ACC
    acc, correct, total, details = calculate_acc(data)

    # 生成报告
    report = generate_report(acc, correct, total, details, args.output)
    print(report)


if __name__ == "__main__":
    main()