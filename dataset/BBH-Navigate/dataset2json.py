# 读取原始 QA 对列表格式（带括号答案），提取并简化答案为选项字母
import json

input_path = "temporal_sequences.json"
output_path = "temporal_sequences_cleaned.json"

# 加载数据
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 处理为标准格式：去掉答案括号，仅保留选项字母
cleaned_data = []
for item in raw_data:
    cleaned_data.append({
        "question": item["question"].strip(),
        "answer": item["answer"].strip().strip("() ")
    })

# 保存为新 JSON 文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

output_path
