import json


def jsonl_to_qa_pairs(input_file, output_file):
    """
    读取 JSONL 文件，提取 QA 对，并保存为 JSON 文件

    Args:
        input_file (str): 输入的 JSONL 文件路径
        output_file (str): 输出的 JSON 文件路径
    """
    qa_pairs = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                qa_pair = {
                    "question": data.get("problem", ""),
                    "answer": data.get("answer", ""),
                }
                qa_pairs.append(qa_pair)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line.strip()}. Error: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    print(f"Successfully converted {len(qa_pairs)} QA pairs to {output_file}")


# 使用示例
if __name__ == "__main__":
    input_jsonl = "OlymMATH-EN-HARD.jsonl"  # 替换为你的输入文件路径
    output_json = "OlyMath.json"  # 替换为你的输出文件路径
    jsonl_to_qa_pairs(input_jsonl, output_json)