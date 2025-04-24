import pandas as pd
import json
import random


# 从 CSV 文件读取数据
def read_data_from_csv(filename):
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(filename)
    # 提取相关列
    data = []
    for index, row in df.iterrows():
        question = row['Pre-Revision Question']
        correct_answer = row['Pre-Revision Correct Answer']
        incorrect_answers = [
            row['Pre-Revision Incorrect Answer 1'],
            row['Pre-Revision Incorrect Answer 2'],
            row['Pre-Revision Incorrect Answer 3']
        ]

        # 将问题和答案处理为字典
        question_data = {
            'question': question,
            'correct_answer': correct_answer,
            'incorrect_answers': incorrect_answers
        }
        data.append(question_data)
    return data


# 生成选择题数据
def generate_multiple_choice(data):
    multiple_choice_data = []

    for item in data:
        question = item['question']
        correct_answer = item['correct_answer']
        incorrect_answers = item['incorrect_answers']

        # 合并正确答案和错误答案
        options = [correct_answer] + incorrect_answers
        random.shuffle(options)  # 打乱选项顺序

        # 生成包含选项的选择题
        question_with_options = {
            'question': f"{question} (A: {options[0]}, B: {options[1]}, C: {options[2]}, D: {options[3]})",
            'answer': get_correct_option(options, correct_answer)  # 获取正确答案的选项
        }

        multiple_choice_data.append(question_with_options)

    return multiple_choice_data


# 获取正确答案的选项（A/B/C/D）
def get_correct_option(options, correct_answer):
    return ['A', 'B', 'C', 'D'][options.index(correct_answer)]


# 将数据写入 JSON 文件
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 主流程
def main():
    # 读取 CSV 文件
    csv_file = 'gpqa_diamond.csv'  # 替换为你的 CSV 文件路径
    data = read_data_from_csv(csv_file)

    # 生成选择题格式数据
    multiple_choice_data = generate_multiple_choice(data)

    # 保存为 JSON 文件
    save_to_json(multiple_choice_data, 'diamond.json')

    print("数据已成功转换为 JSON 格式！")


# 执行主流程
if __name__ == "__main__":
    main()
