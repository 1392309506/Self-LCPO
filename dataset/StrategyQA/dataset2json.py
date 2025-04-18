# 尝试读取用户上传的 strategyqa 文件（此前未上传，这里使用假设文件名）
import json

strategyqa_path = "dev.json"  # 请确认上传后的路径是否一致
output_path = "dev.json"

# 加载原始数据
with open(strategyqa_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 解析 QA 对列表
qa_list = []
for item in raw_data:
    qa_list.append({
        "question": item.get("question", "").strip(),
        "answer": "Yes" if item["answer"] is True else "No"
    })

# 保存为简化后的 QA JSON 文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(qa_list, f, indent=2, ensure_ascii=False)

output_path
