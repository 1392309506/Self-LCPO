import csv
import json

def preprocess_liar_tsv_to_json(tsv_file_path, json_file_path):
    data = []
    with open(tsv_file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                question = row[2]  # 第三列：声明文本
                answer = row[1]    # 第二列：真实性标签
                data.append({"question": question, "answer": answer})
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

preprocess_liar_tsv_to_json('train.tsv', 'train.json')
preprocess_liar_tsv_to_json('valid.tsv', 'valid.json')
preprocess_liar_tsv_to_json('test.tsv', 'test.json')
