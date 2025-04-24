import json
import json

def extract_qa_pairs_to_file(input_path, output_path):
    qa_list = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                question = item.get("question", "").strip()
                answer = item.get("answer", "").strip()
                qa_list.append({"question": question, "answer": answer})
            except json.JSONDecodeError as e:
                print(f"跳过格式错误行：{e}")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(qa_list, outfile, ensure_ascii=False, indent=2)

    print(f"已保存 {len(qa_list)} 条 QA 对到：{output_path}")

# 示例使用
extract_qa_pairs_to_file("/root/lty/Self-LCPO/dataset/GSM8K/GSM8K.jsonl", "GSM8K.json")
