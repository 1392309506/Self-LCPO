import json

def convert_liar_to_qa_format(input_path="valid.json", output_path="valid.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_data = []

    for item in data:
        claim = item["question"]
        label = item["answer"]
        qa = {
            "question": f"Given the claim: '{claim}', classify the veracity of this statement into one of the following categories: [true, mostly-true, half-true, barely-true, false, pants-fire].",
            "answer": label
        }
        qa_data.append(qa)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(qa_data)} items to QA format.")

# 执行转换
convert_liar_to_qa_format()
