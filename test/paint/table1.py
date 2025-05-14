import re
import json


def extract_data_from_txt(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割数据集
    datasets = content.split('\n')[:2]  # 只处理前两行（StrategyQA和gpqa）

    results = {}

    for dataset in datasets:
        if not dataset.strip():
            continue

        # 提取数据集名称
        dataset_name = dataset.split('\t')[0].strip()
        results[dataset_name] = []

        # 提取每个token配置的数据
        for cell in dataset.split('\t')[1:]:
            if not cell.strip():
                continue

            try:
                # 将字符串转换为字典
                data = eval(cell)  # 注意：使用eval有安全风险，确保数据来源可信

                # 提取所需字段
                extracted = {
                    'suggest_token': data['suggest_token'],
                    'acc': data['acc'],
                    'total_token': data['total_token']
                }
                results[dataset_name].append(extracted)

            except (SyntaxError, KeyError):
                print(f"跳过无法解析的数据: {cell}")
                continue

    return results


def save_to_json(data, output_file):
    """将提取的数据保存为JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    input_file = "table1.txt"  # 替换为您的输入文件
    output_file = "table1.json"  # 输出文件

    extracted_data = extract_data_from_txt(input_file)
    save_to_json(extracted_data, output_file)

    print("数据提取完成！结果已保存到", output_file)
    print("提取的数据结构:")
    print(json.dumps(extracted_data, indent=2))