import json
import re

def clean_ab_answers(input_path="wsc.json", output_path="wsc.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        answer = item["answer"].strip()
        # 移除 A. 或 B. 中的句号和空格
        answer = re.sub(r"^(A|B)[.。]$", r"\1", answer)
        cleaned_data.append({
            "question": item["question"].strip(),
            "answer": answer
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned {len(cleaned_data)} QA pairs and saved to {output_path}")

# 运行函数
clean_ab_answers()
